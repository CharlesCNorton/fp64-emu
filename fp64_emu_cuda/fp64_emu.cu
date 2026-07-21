// fp64-emu: emulated FP64 GEMM on INT8 tensor cores (Ozaki-II / CRT).
// Operands scale to exact integer significands per row / per column; the
// exact product is recovered from residues modulo pairwise-coprime moduli
// (one INT8 GEMM each) by balanced-Garner reconstruction and rescaled by
// powers of two. M, N, K carry no alignment constraints.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int MAXP = 20;
// Word count of the packed Garner factor rows: four int8 factors per word.
constexpr int MPGROUPS = (MAXP + 3) / 4;

// Grid for the elementwise passes: one block-row per matrix row (y), a capped
// block span across the row (x), so no element needs a 64-bit div/mod.
dim3 grid_rows(int64_t rows, int64_t cols) {
  const int64_t gx = std::min<int64_t>((cols + kThreads - 1) / kThreads, 8);
  const int64_t gy = std::min<int64_t>(rows, 65535);
  return dim3((unsigned)std::max<int64_t>(gx, 1), (unsigned)std::max<int64_t>(gy, 1));
}

// Balanced Barrett reduction: mu32 = floor(2^31 / p), bias a multiple of p
// with bias >= |y|, so xu is non-negative and congruent to y; unsigned
// arithmetic throughout since xu can exceed 2^31.
__device__ __forceinline__ int barrett_bal32(int y, int p, uint32_t mu32, uint32_t bias) {
  const uint32_t xu = (uint32_t)((long long)y + (long long)bias);
  const uint32_t q = (uint32_t)(((uint64_t)xu * mu32) >> 31);
  uint32_t r = xu - q * (uint32_t)p;
  while (r >= (uint32_t)p) r -= (uint32_t)p;
  int rb = (int)r;
  // Balance to [-p/2, (p-1)/2] under integer division; p = 256 gives [-128, 127].
  if (rb > (p - 1) / 2) rb -= p;
  return rb;
}

// Signed 128-bit accumulator as two 64-bit halves; lowers to the same
// multiply-high instructions __int128 would, and builds where it is absent.
struct s128 {
  unsigned long long lo;
  long long hi;
};
__device__ __forceinline__ s128 s128_from(long long v) {
  s128 r; r.lo = (unsigned long long)v; r.hi = (v < 0) ? -1LL : 0LL; return r;
}
__device__ __forceinline__ s128 s128_add(s128 a, s128 b) {
  s128 r; r.lo = a.lo + b.lo;
  r.hi = a.hi + b.hi + ((r.lo < a.lo) ? 1LL : 0LL);
  return r;
}
// a << s, for 0 < s < 64.
__device__ __forceinline__ s128 s128_shl(s128 a, int s) {
  s128 r;
  r.hi = (long long)(((unsigned long long)a.hi << s) | (a.lo >> (64 - s)));
  r.lo = a.lo << s;
  return r;
}
// One Horner step a * p + d; p a small positive prime, d a balanced digit.
__device__ __forceinline__ s128 s128_muladd(s128 a, int p, long long d) {
  const unsigned long long up = (unsigned long long)p;
  const unsigned long long lo = a.lo * up;
  const long long hi = (long long)__umul64hi(a.lo, up) + a.hi * (long long)p;
  s128 r; r.lo = lo + (unsigned long long)d;
  r.hi = hi + ((r.lo < lo) ? 1LL : 0LL) + ((d < 0) ? -1LL : 0LL);
  return r;
}
// a * m + d for m a group product (< 2^57); the range rule bounds a, so the
// product stays inside s128.
__device__ __forceinline__ s128 s128_mul64_add(s128 a, unsigned long long m, long long d) {
  const unsigned long long lo = a.lo * m;
  const long long hi = (long long)__umul64hi(a.lo, m) + a.hi * (long long)m;
  s128 r; r.lo = lo + (unsigned long long)d;
  r.hi = hi + ((r.lo < lo) ? 1LL : 0LL) + ((d < 0) ? -1LL : 0LL);
  return r;
}
// Correctly rounded conversion to double: the dropped tail folds into the low
// bit of the top 64, far below the round position, so RNE decides identically.
__device__ __forceinline__ double s128_to_double(s128 a) {
  const bool neg = a.hi < 0;
  unsigned long long hi = (unsigned long long)a.hi, lo = a.lo;
  if (neg) { lo = ~lo + 1ULL; hi = ~hi + ((lo == 0ULL) ? 1ULL : 0ULL); }
  if (hi == 0ULL) { double m = __ull2double_rn(lo); return neg ? -m : m; }
  const int sh = 64 - __clzll((long long)hi);   // 1 <= sh <= 64
  unsigned long long top, tail;
  if (sh == 64) { top = hi; tail = lo; }
  else { top = (hi << (64 - sh)) | (lo >> sh); tail = lo & ((1ULL << sh) - 1ULL); }
  const double m = ldexp(__ull2double_rn(top | ((tail != 0ULL) ? 1ULL : 0ULL)), sh);
  return neg ? -m : m;
}

// Per-row (A) / per-column (B) shift scaling the max |element| into
// [2^(bits-1), 2^bits), plus lg[r] = log2 of the scaled row 2-norm for the
// Cauchy-Schwarz product bound; non-finite data yields +inf, which falls back
// to the worst-case range rule. One launch scans both operands.
__global__ void row_shift_kernel(const double* __restrict__ A, const double* __restrict__ B,
                                 int64_t M, int64_t N, int64_t cols, int bits,
                                 int16_t* __restrict__ sftA, float* __restrict__ lgA,
                                 int16_t* __restrict__ sftB, float* __restrict__ lgB) {
  const int64_t br = blockIdx.x;
  const bool isA = br < M;
  const double* __restrict__ x = isA ? A : B;
  const int64_t r = isA ? br : br - M;
  double mx = 0.0, ss = 0.0;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const double v = x[r * cols + c];
    mx = fmax(mx, fabs(v));
    ss += v * v;
  }
  __shared__ double sm[kThreads], sq[kThreads];
  sm[threadIdx.x] = mx;
  sq[threadIdx.x] = ss;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + s]);
      sq[threadIdx.x] += sq[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double m = sm[0];
    const int e = (m > 0.0 && isfinite(m)) ? ilogb(m) : 0;
    const int sh = (bits - 1) - e;          // max element -> [2^(bits-1), 2^bits)
    const double s2 = sq[0];
    const float lg = (s2 > 0.0 && isfinite(s2)) ? (float)(0.5 * log2(s2) + sh)
                                                : (m > 0.0 ? INFINITY : -INFINITY);
    if (isA) { sftA[r] = (int16_t)sh; lgA[r] = lg; }
    else     { sftB[r] = (int16_t)sh; lgB[r] = lg; }
  }
}

// Column max / sum-of-squares partials for an operand consumed in its [K, N]
// layout: thread j owns global column j (batch member j / cols_per), blockIdx.y
// a row slab, so reads stay coalesced across columns. Deterministic: fixed
// slab order, reduced by col_shift_finalize_kernel.
__global__ void col_shift_partial_kernel(const double* __restrict__ X, int64_t totalcols,
                                         int64_t rows_per, int64_t cols_per, int64_t kslab,
                                         double* __restrict__ pmax, double* __restrict__ pss) {
  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= totalcols) return;
  const int64_t b = j / cols_per, n = j - b * cols_per;
  const int64_t k0 = (int64_t)blockIdx.y * kslab;
  const int64_t k1 = min(k0 + kslab, rows_per);
  const double* __restrict__ base = X + b * rows_per * cols_per + n;
  double mx = 0.0, ss = 0.0;
  for (int64_t k = k0; k < k1; k++) {
    const double v = base[k * cols_per];
    mx = fmax(mx, fabs(v));
    ss += v * v;
  }
  pmax[(int64_t)blockIdx.y * totalcols + j] = mx;
  pss[(int64_t)blockIdx.y * totalcols + j] = ss;
}

// Slab reduction to the same shift and log2-norm outputs row_shift_kernel
// emits per row.
__global__ void col_shift_finalize_kernel(const double* __restrict__ pmax,
                                          const double* __restrict__ pss, int64_t totalcols,
                                          int nslab, int bits, int16_t* __restrict__ sft,
                                          float* __restrict__ lg) {
  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= totalcols) return;
  double mx = 0.0, ss = 0.0;
  for (int s = 0; s < nslab; s++) {
    mx = fmax(mx, pmax[(int64_t)s * totalcols + j]);
    ss += pss[(int64_t)s * totalcols + j];
  }
  const int e = (mx > 0.0 && isfinite(mx)) ? ilogb(mx) : 0;
  const int sh = (bits - 1) - e;
  lg[j] = (ss > 0.0 && isfinite(ss)) ? (float)(0.5 * log2(ss) + sh)
                                     : (mx > 0.0 ? INFINITY : -INFINITY);
  sft[j] = (int16_t)sh;
}

// Largest extra shift g keeping the Cauchy-Schwarz product bound inside the
// signed CRT range: 2g + lgA + lgB + margin <= log2(M) - 1, g <= 53 - bits.
__global__ void pick_bits_kernel(const float* __restrict__ lgA, int64_t m,
                                 const float* __restrict__ lgB, int64_t n,
                                 float log2M, int bits, int16_t* __restrict__ sftA,
                                 int16_t* __restrict__ sftB, int* __restrict__ bits_eff) {
  __shared__ float sa[kThreads], sb[kThreads];
  __shared__ int s_g;
  float a = -INFINITY, b = -INFINITY;
  for (int64_t i = threadIdx.x; i < m; i += blockDim.x) a = fmaxf(a, lgA[i]);
  for (int64_t j = threadIdx.x; j < n; j += blockDim.x) b = fmaxf(b, lgB[j]);
  sa[threadIdx.x] = a;
  sb[threadIdx.x] = b;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sa[threadIdx.x] = fmaxf(sa[threadIdx.x], sa[threadIdx.x + s]);
      sb[threadIdx.x] = fmaxf(sb[threadIdx.x], sb[threadIdx.x + s]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    // Margin: rounding slack of round(A*2^s) plus the float log2 error.
    const float need = sa[0] + sb[0] + 0.5f;
    int g = 0;
    if (isfinite(need)) g = (int)floorf((log2M - 1.0f - need) * 0.5f);
    // Negative g narrows the operands, keeping any modulus count range-safe.
    g = min(g, 53 - bits);
    int be = bits + g;
    if (be < 16) be = 16;                  // cshift = bits-15 must stay positive
    s_g = be - bits;
    *bits_eff = be;
  }
  __syncthreads();
  // Fold the gain into the shifts so the rest of the pipeline sees one width.
  const int g = s_g;
  if (g) {
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) sftA[i] = (int16_t)(sftA[i] + g);
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) sftB[j] = (int16_t)(sftB[j] + g);
  }
}

// Tile geometry of the in-file modular GEMM; the extract kernel writes the
// tile-blocked operand layout for it, so the constants live above both.
constexpr int GBM = 128, GBN = 256, GBK = 64, GTHREADS = 256;
constexpr int GRING = 4, GAHEAD = 3;   // slots; producers run GAHEAD chunks ahead

// Residue extraction. Atld = round(x * 2^sft), exact in fp64; balanced int8
// residues per modulus as [P, rows, cols], plus correction planes hi7 =
// round(Atld * 2^(7-bits)) and q7 = round((x*2^sft - Atld) * 2^8), whose
// cross GEMMs carry the first-order quantization error at scale 2^(bits-15).
// tiled != 0 writes each plane tile-blocked [rows/T][K/GBK][T][GBK]
// (T = GBM for A, GBN for B), zero-padded to the tile grid. Batched calls
// flatten rows to [batch * rows_per]; the tiled grid is per batch member, so
// no tile straddles a batch boundary.
// kstride / koff separate the operand row pitch from the extracted column
// window, so a K segment reads in place: element (r, c) at r * kstride +
// koff + c, planes sized to the window.
__global__ void extract_kernel(const double* __restrict__ Ain, const double* __restrict__ Bin,
                               int64_t M, int64_t N, int64_t rows_perA, int64_t rows_perB,
                               int64_t colsA, int64_t colsB, int64_t kstrideA,
                               int64_t kstrideB, int64_t koff, int nt, int tiled, int do_corr,
                               const int16_t* __restrict__ sftA, const int16_t* __restrict__ sftB,
                               int num_primes, const int* __restrict__ primes,
                               const int* __restrict__ bits_p,
                               int8_t* __restrict__ outA, int8_t* __restrict__ hiA,
                               int8_t* __restrict__ qA, int8_t* __restrict__ outB,
                               int8_t* __restrict__ hiB, int8_t* __restrict__ qB) {
  const int bits = *bits_p;   // chosen on device by pick_bits_kernel
  // Per-prime tables for the three-limb reduction below. 2^36 and 2^18 are the
  // limb weights; 2^bits undoes the bias that makes the scaled integer positive.
  __shared__ int s_p[MAXP];
  __shared__ uint32_t s_mu32[MAXP], s_c36[MAXP], s_c18[MAXP], s_cb[MAXP];
  for (int t0 = threadIdx.x; t0 < num_primes; t0 += blockDim.x) {
    const unsigned p = (unsigned)primes[t0];
    s_p[t0] = (int)p;
    s_mu32[t0] = (uint32_t)((1u << 31) / p);
    s_c36[t0] = (uint32_t)(((uint64_t)1 << 36) % p);
    s_c18[t0] = (uint32_t)(((uint64_t)1 << 18) % p);
    // Adding (p - 2^bits mod p) keeps the limb sum positive and congruent,
    // undoing the +2^bits bias with no post-reduction sign fixup.
    s_cb[t0] = p - (uint32_t)(((uint64_t)1 << bits) % p);
  }
  __syncthreads();
  // Balanced residue from the three 18-bit limbs of (t + 2^bits); each limb
  // product is under 2^26 and the sum under 2^27, so q underestimates v/p by
  // at most one and a single conditional subtract completes the 32-bit Barrett.
  auto residue = [&](uint32_t a, uint32_t b, uint32_t c, int pi) -> signed char {
    const uint32_t p = (uint32_t)s_p[pi];
    const uint32_t v = a * s_c36[pi] + b * s_c18[pi] + c + s_cb[pi];
    const uint32_t q = (uint32_t)(((uint64_t)v * s_mu32[pi]) >> 31);
    uint32_t r = v - q * p;
    if (r >= p) r -= p;
    int rb = (int)r;
    if (rb > ((int)p - 1) / 2) rb -= (int)p;   // even moduli: 128 -> -128
    return (signed char)rb;
  };
  // The split runs entirely in integer arithmetic on the raw fp64 encoding
  // and returns the scaled integer biased by 2^bits for the residue helper;
  // hi and q span the full signed int8 range.
  // Round-to-nearest-even of mag * 2^(-s), exact for any shift.
  auto rne_shr = [](uint64_t mag, int s) -> uint64_t {
    if (s <= 0) return (-s < 64) ? (mag << (-s)) : 0ULL;
    if (s > 63) return 0ULL;
    const uint64_t keep = mag >> s;
    const uint64_t rem = mag & ((1ULL << s) - 1ULL);
    const uint64_t half = 1ULL << (s - 1);
    return keep + ((rem > half || (rem == half && (keep & 1ULL))) ? 1ULL : 0ULL);
  };
  auto split = [&](double x, int srow, signed char& h, signed char& q) -> uint64_t {
    const long long xb = __double_as_longlong(x);
    const bool neg = xb < 0;
    const int e = (int)((xb >> 52) & 0x7FF);
    const uint64_t m = (uint64_t)xb & 0xFFFFFFFFFFFFFULL;
    // x = mag * 2^(t - 52) with t = (unbiased exponent) + srow; subnormals
    // read as exponent 1 with no implicit bit.
    const uint64_t mag = e ? ((1ULL << 52) | m) : m;
    const int t = (e ? e : 1) - 1075 + srow + 52;
    const uint64_t Tu = rne_shr(mag, 52 - t);          // round(|x| * 2^srow)
    const long long td = neg ? -(long long)Tu : (long long)Tu;
    if (do_corr) {
      // resid * 2^8 = mag * 2^(t+8-52) - T * 2^8, exact.
      const long long r8 = (long long)rne_shr(mag, 44 - t) - (long long)(Tu << 8);
      const long long h8 = (long long)rne_shr(Tu, bits - 7);
      const long long hs = neg ? -h8 : h8;
      const long long qs = neg ? -r8 : r8;
      h = (signed char)(hs > 127 ? 127 : (hs < -127 ? -127 : hs));
      q = (signed char)(qs > 127 ? 127 : (qs < -127 ? -127 : qs));
    }
    return (uint64_t)(td + ((long long)1 << bits));
  };

  // Row index from the grid's y dimension. Four columns per thread when the
  // pitch allows char4 stores: one full 128-byte warp transaction per plane.
  // nt marks a B operand consumed in its [K, N] layout: rows are k, shifts
  // are per column, and its planes keep that layout for the NN cuBLAS form.
  const int64_t rows_all = M + N;
  for (int64_t br = blockIdx.y; br < rows_all; br += gridDim.y) {
    const bool isA = br < M;
    const double* __restrict__ x = isA ? Ain : Bin;
    const int64_t r = isA ? br : br - M;
    const int64_t cols = isA ? colsA : colsB;
    const bool vec4 = ((cols & 3) == 0);
    const int64_t nktT = (cols + GBK - 1) / GBK;
    // Plane stride and row base under the selected layout; a tiled column
    // lands at trow + (c / GBK) * tblk + (c % GBK). Correction planes share it.
    const int64_t tside = isA ? GBM : GBN;
    const int64_t tblk = tside * GBK;
    const int64_t rows_per = isA ? rows_perA : rows_perB;
    const int64_t tgrid = ((rows_per + tside - 1) / tside) * nktT * tblk;
    const int64_t total = tiled ? ((isA ? M : N) / rows_per) * tgrid
                                : (isA ? M : N) * cols;
    const int64_t bb = tiled ? (r / rows_per) : 0;
    const int64_t rl = r - bb * rows_per;
    const int64_t trow = tiled ? (bb * tgrid + (rl / tside) * nktT * tblk + (rl % tside) * GBK)
                               : r * cols;
    int8_t* __restrict__ out = isA ? outA : outB;
    int8_t* __restrict__ hi7 = isA ? hiA : hiB;
    int8_t* __restrict__ q7 = isA ? qA : qB;
    const int16_t* __restrict__ sfts = isA ? sftA : sftB;
    const bool percol = (!isA) && nt;
    const int16_t srow = percol ? (int16_t)0 : sfts[r];
    const int16_t* __restrict__ scol = percol ? sfts + (r / rows_perB) * cols : sfts;
    const int64_t xrow = r * (isA ? kstrideA : kstrideB) + koff;
    if (vec4) {
      const int64_t nv = cols >> 2;
      for (int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; v < nv;
           v += (int64_t)gridDim.x * blockDim.x) {
        const int64_t c0 = v << 2;
        const int64_t o = tiled ? trow + (c0 / GBK) * tblk + (c0 % GBK) : trow + c0;
        uint32_t la[4], lb[4], lc[4];
        signed char hb[4], qb[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
          const uint64_t uu = split(x[xrow + c0 + u], percol ? scol[c0 + u] : srow,
                                    hb[u], qb[u]);
          la[u] = (uint32_t)(uu >> 36);
          lb[u] = (uint32_t)((uu >> 18) & 0x3FFFFu);
          lc[u] = (uint32_t)(uu & 0x3FFFFu);
        }
        if (do_corr) {
          *reinterpret_cast<char4*>(&hi7[o]) = make_char4(hb[0], hb[1], hb[2], hb[3]);
          *reinterpret_cast<char4*>(&q7[o]) = make_char4(qb[0], qb[1], qb[2], qb[3]);
        }
        for (int pi = 0; pi < num_primes; pi++)
          *reinterpret_cast<char4*>(&out[(int64_t)pi * total + o]) =
              make_char4(residue(la[0], lb[0], lc[0], pi), residue(la[1], lb[1], lc[1], pi),
                         residue(la[2], lb[2], lc[2], pi), residue(la[3], lb[3], lc[3], pi));
      }
    } else {
      for (int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; c < cols;
           c += (int64_t)gridDim.x * blockDim.x) {
        const int64_t o = tiled ? trow + (c / GBK) * tblk + (c % GBK) : trow + c;
        signed char h, q;
        const uint64_t uu = split(x[xrow + c], percol ? scol[c] : srow, h, q);
        if (do_corr) {
          hi7[o] = h;
          q7[o] = q;
        }
        const uint32_t a = (uint32_t)(uu >> 36), b = (uint32_t)((uu >> 18) & 0x3FFFFu),
                       cc = (uint32_t)(uu & 0x3FFFFu);
        for (int pi = 0; pi < num_primes; pi++)
          out[(int64_t)pi * total + o] = residue(a, b, cc, pi);
      }
    }
  }
}

// ---- batched int8 tensor-core GEMM with a residue epilogue ------------------
// C[b] = A[b] @ B[b]^T on mma.sync.m16n8k32, exact int32 accumulate. 128x256
// CTA tile, 8 warps of 64x64 (32 mma behind every 8 ldmatrix); cp.async
// through a 4-slot mbarrier ring, no mainloop __syncthreads. Stages are
// unpadded 64-byte rows under the 64B XOR swizzle (chunk c at shared address
// S lands at c ^ ((S >> 7) & 3)), which ldmatrix mirrors. M, N, K arbitrary.

__device__ __forceinline__ void g_cp16(void* smem, const void* gmem, int src_bytes) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void g_cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
__device__ __forceinline__ void g_mbar_init(uint64_t* bar, unsigned count) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void g_mbar_arrive_cp(uint64_t* bar) {
#if __CUDA_ARCH__ >= 900
  // sm_90+ emulates cp.async and arrival on async completion is unsound there
  // (exact-equality failures on sm_120): drain own copies, then arrive plainly.
  g_cp_wait_all();
  const unsigned a = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("{\n .reg .b64 s;\n mbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(a));
#else
  const unsigned a = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(a));
#endif
}
__device__ __forceinline__ void g_mbar_arrive(uint64_t* bar) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("{\n .reg .b64 s;\n mbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(a));
}
__device__ __forceinline__ void g_mbar_wait(uint64_t* bar, unsigned phase) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(bar);
  unsigned done = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(a), "r"(phase));
  } while (!done);
}
__device__ __forceinline__ void g_ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void g_ldm_x2(unsigned& r0, unsigned& r1, const void* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void g_mma_s8(int* d, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Stage one GBK-wide chunk of a row-major [ROWS, K] panel into 64B-swizzled
// shared rows (the pattern TMA's SWIZZLE_64B writes); 16B-aligned full chunks
// use cp.async, K-tails and misaligned rows byte loads.
template <int ROWS>
__device__ __forceinline__ void g_stage(int8_t* dst, const int8_t* __restrict__ base,
                                        long gbase, int r0, int rows_max, int k0, int K, int tid) {
  for (int c = tid; c < ROWS * (GBK / 16); c += GTHREADS) {
    const int r = c >> 2, ci = c & 3, kc = ci * 16, gk = k0 + kc, gr = r0 + r;
    const bool inr = (gr < rows_max) && (gk < K);
    const int bytes = inr ? min(16, K - gk) : 0;
    const int8_t* src = &base[inr ? gbase + (long)gr * K + gk : gbase];
    int8_t* drow = &dst[r * GBK];
    const unsigned key = ((unsigned)__cvta_generic_to_shared(drow) >> 7) & 3;
    int8_t* d = drow + ((ci ^ (int)key) * 16);
    if (bytes == 16 && ((reinterpret_cast<uintptr_t>(src) & 15) == 0)) {
      g_cp16(d, src, 16);
    } else {
#pragma unroll
      for (int j = 0; j < 16; j++) d[j] = (j < bytes) ? src[j] : (int8_t)0;
    }
  }
}

// Stage one contiguous [ROWS, GBK] block of the tile-blocked layout:
// streaming 16B reads, no bounds checks; the extract pass zero-padded the grid.
template <int ROWS>
__device__ __forceinline__ void g_stage_tiled(int8_t* dst, const int8_t* __restrict__ block,
                                              int tid) {
  for (int c = tid; c < ROWS * (GBK / 16); c += GTHREADS) {
    const int r = c >> 2, ci = c & 3;
    int8_t* drow = &dst[r * GBK];
    const unsigned key = ((unsigned)__cvta_generic_to_shared(drow) >> 7) & 3;
    g_cp16(drow + ((ci ^ (int)key) * 16), block + (long)c * 16, 16);
  }
}

// One launch covers all (P+2)*nbatch plane-batch slots, plane-major: slots
// below P*nbatch reduce to balanced int8 in c8, the rest store raw int32
// correction products into c32.
__global__ void __launch_bounds__(GTHREADS)
residue_gemm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                    int8_t* __restrict__ c8, int32_t* __restrict__ c32,
                    const int* __restrict__ primes, int P, int nbatch,
                    int M, int N, int K, int gswz, int tiled) {
  extern __shared__ __align__(128) int8_t smem[];
  int8_t* As = smem;
  int8_t* Bs = smem + GRING * GBM * GBK;
  const int ASTAGE = GBM * GBK, BSTAGE = GBN * GBK;
  __shared__ uint64_t full_bar[GRING], empty_bar[GRING];

  const int nmt = (M + GBM - 1) / GBM, nnt = (N + GBN - 1) / GBN;
  const int tile = blockIdx.x, gsz = gswz * nnt;
  const int gid = tile / gsz, gof = tile % gsz;
  const int m_first = gid * gswz, mh = min(gswz, nmt - m_first);
  const int mt = m_first + (gof % mh), nt = gof / mh;
  const int m0 = mt * GBM, n0 = nt * GBN;

  const int bz = blockIdx.z;
  const int nkt = (K + GBK - 1) / GBK;
  // Plane strides: row-major planes are rows*K; tiled planes are the padded
  // tile grid. The per-stage tiled block address folds mt/nt in once.
  const long aplane = tiled ? (long)nmt * nkt * (GBM * GBK) : (long)M * K;
  const long bplane = tiled ? (long)nnt * nkt * (GBN * GBK) : (long)N * K;
  const long abase = (long)bz * aplane, bbase = (long)bz * bplane;
  const int8_t* Atile = A + abase + (long)mt * nkt * (GBM * GBK);
  const int8_t* Btile = B + bbase + (long)nt * nkt * (GBN * GBK);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warpM = warp & 1, warpN = warp >> 1;       // 2 x 4 warps
  const int rbase = warpM * 64, cbase_w = warpN * 64;  // 64x64 warp tile

  if (tid < GRING) { g_mbar_init(&full_bar[tid], GTHREADS); g_mbar_init(&empty_bar[tid], GTHREADS); }
  __syncthreads();

  auto stage = [&](int j) {
    const int st = j % GRING;
    if (tiled) {
      g_stage_tiled<GBM>(As + st * ASTAGE, Atile + (long)j * (GBM * GBK), tid);
      g_stage_tiled<GBN>(Bs + st * BSTAGE, Btile + (long)j * (GBN * GBK), tid);
    } else {
      g_stage<GBM>(As + st * ASTAGE, A, abase, m0, M, j * GBK, K, tid);
      g_stage<GBN>(Bs + st * BSTAGE, B, bbase, n0, N, j * GBK, K, tid);
    }
    g_mbar_arrive_cp(&full_bar[st]);
  };

  int Cacc[4][8][4];
#pragma unroll
  for (int mi = 0; mi < 4; mi++)
#pragma unroll
    for (int ni = 0; ni < 8; ni++)
#pragma unroll
      for (int t = 0; t < 4; t++) Cacc[mi][ni][t] = 0;

  // Fragment double-buffering: one buffer's 32 mma issue while the other's
  // eight ldmatrix land; B pairs two n-tiles per x4, and the 16B chunk index
  // XORs with (shared address >> 7) & 3, mirroring the staging swizzle.
  unsigned afr[2][4][4], bfr[2][8][2];
  auto ld_frags = [&](const int8_t* Ac, const int8_t* Bc, int kk, int buf) {
#pragma unroll
    for (int mi = 0; mi < 4; mi++) {
      const int8_t* arow = &Ac[(rbase + mi * 16 + (lane & 15)) * GBK];
      const unsigned ac = (unsigned)(kk * 2 + (lane >> 4)) ^
                          (((unsigned)__cvta_generic_to_shared(arow) >> 7) & 3);
      g_ldm_x4(afr[buf][mi][0], afr[buf][mi][1], afr[buf][mi][2], afr[buf][mi][3],
               reinterpret_cast<const uint16_t*>(arow + ac * 16));
    }
    const int g = lane >> 3;
#pragma unroll
    for (int nb = 0; nb < 4; nb++) {
      const int8_t* brow = &Bc[(cbase_w + (nb * 2 + (g >> 1)) * 8 + (lane & 7)) * GBK];
      const unsigned bc = (unsigned)(kk * 2 + (g & 1)) ^
                          (((unsigned)__cvta_generic_to_shared(brow) >> 7) & 3);
      g_ldm_x4(bfr[buf][nb * 2][0], bfr[buf][nb * 2][1], bfr[buf][nb * 2 + 1][0], bfr[buf][nb * 2 + 1][1],
               reinterpret_cast<const uint16_t*>(brow + bc * 16));
    }
  };
  auto mma_burst = [&](int buf) {
#pragma unroll
    for (int mi = 0; mi < 4; mi++)
#pragma unroll
      for (int ni = 0; ni < 8; ni++) g_mma_s8(Cacc[mi][ni], afr[buf][mi], bfr[buf][ni]);
  };

  constexpr int nkk = GBK / 32;
  for (int s = 0; s < GAHEAD && s < nkt; s++) stage(s);
  g_mbar_wait(&full_bar[0], 0);
  ld_frags(As, Bs, 0, 0);
  for (int kt = 0; kt < nkt; kt++) {
    const int j = kt + GAHEAD;
    if (j < nkt) {
      if (j >= GRING) g_mbar_wait(&empty_bar[j % GRING], ((j / GRING) - 1) & 1);
      stage(j);
    }
    int8_t* Ac = As + (kt % GRING) * ASTAGE;
    int8_t* Bc = Bs + (kt % GRING) * BSTAGE;
#pragma unroll
    for (int kk = 0; kk < nkk; kk++) {
      const int cur = (kt * nkk + kk) & 1;
      if (kk + 1 < nkk) {
        ld_frags(Ac, Bc, kk + 1, cur ^ 1);
      } else if (kt + 1 < nkt) {
        g_mbar_wait(&full_bar[(kt + 1) % GRING], ((kt + 1) / GRING) & 1);
        ld_frags(As + ((kt + 1) % GRING) * ASTAGE, Bs + ((kt + 1) % GRING) * BSTAGE, 0, cur ^ 1);
      }
      mma_burst(cur);
      if (kk == nkk - 1) g_mbar_arrive(&empty_bar[kt % GRING]);
    }
  }

  // Epilogue: fragments store straight to global as adjacent-column pairs
  // (char2 / int2), no shared staging or barriers. 31-bit Barrett: xu = acc +
  // off (off the multiple of p nearest 2^31), q = (xu * mu31) >> 31.
  g_cp_wait_all();
  const bool residue = (bz < P * nbatch);
  const int p = residue ? primes[bz / nbatch] : 0;
  const unsigned mu31 = residue ? ((1u << 31) / (unsigned)p) : 0;
  const unsigned off = residue ? ((0x80000000u / (unsigned)p) * (unsigned)p) : 0;
  const long c8base = (long)bz * M * N, c32base = (long)(bz - P * nbatch) * M * N;
  auto bal8 = [&](int v) -> int8_t {
    const unsigned xu = (unsigned)v + off;                       // |v| < 2^31 -> xu fits u32
    unsigned r = xu - (unsigned)(((unsigned long long)xu * mu31) >> 31) * (unsigned)p;
    while (r >= (unsigned)p) r -= (unsigned)p;                   // q underestimates by <= 2
    int rb = (int)r;
    if (rb > p / 2) rb -= p;
    return (int8_t)rb;
  };
  // Pair stores need 2-byte (int8) / 8-byte (int32) alignment, which holds
  // whenever N is even; edge tiles and odd N take the guarded scalar path.
  const int r00 = m0 + rbase + (lane >> 2);
  const int c00 = n0 + cbase_w + (lane & 3) * 2;
  const bool interior = (m0 + GBM <= M) && (n0 + GBN <= N) && ((N & 1) == 0);
#pragma unroll
  for (int mi = 0; mi < 4; mi++)
#pragma unroll
    for (int ni = 0; ni < 8; ni++) {
      const int r = r00 + mi * 16, c = c00 + ni * 8;
      if (residue) {
        const char2 v0 = make_char2(bal8(Cacc[mi][ni][0]), bal8(Cacc[mi][ni][1]));
        const char2 v1 = make_char2(bal8(Cacc[mi][ni][2]), bal8(Cacc[mi][ni][3]));
        if (interior) {
          *reinterpret_cast<char2*>(&c8[c8base + (long)r * N + c]) = v0;
          *reinterpret_cast<char2*>(&c8[c8base + (long)(r + 8) * N + c]) = v1;
        } else {
          if (r < M && c < N) c8[c8base + (long)r * N + c] = v0.x;
          if (r < M && c + 1 < N) c8[c8base + (long)r * N + c + 1] = v0.y;
          if (r + 8 < M && c < N) c8[c8base + (long)(r + 8) * N + c] = v1.x;
          if (r + 8 < M && c + 1 < N) c8[c8base + (long)(r + 8) * N + c + 1] = v1.y;
        }
      } else {
        if (interior) {
          *reinterpret_cast<int2*>(&c32[c32base + (long)r * N + c]) =
              make_int2(Cacc[mi][ni][0], Cacc[mi][ni][1]);
          *reinterpret_cast<int2*>(&c32[c32base + (long)(r + 8) * N + c]) =
              make_int2(Cacc[mi][ni][2], Cacc[mi][ni][3]);
        } else {
          if (r < M && c < N) c32[c32base + (long)r * N + c] = Cacc[mi][ni][0];
          if (r < M && c + 1 < N) c32[c32base + (long)r * N + c + 1] = Cacc[mi][ni][1];
          if (r + 8 < M && c < N) c32[c32base + (long)(r + 8) * N + c] = Cacc[mi][ni][2];
          if (r + 8 < M && c + 1 < N) c32[c32base + (long)(r + 8) * N + c + 1] = Cacc[mi][ni][3];
        }
      }
    }
}

// ---- TMA variant of the residue GEMM (sm_90+, K % 16 == 0) ------------------
// Same pipeline and epilogue; panels arrive by cp.async.bulk.tensor (thread 0
// posts mbarrier.arrive.expect_tx), TMA zero-fills overhangs, and hardware
// SWIZZLE_64B matches the ldmatrix addressing above.
constexpr int TRING = 3, TAHEAD = 2;
constexpr int TBN = 128, TTHREADS = 256;   // 128x128 tile, 8 warps, 2 CTAs/SM
constexpr int TALD = GBK;   // 64: unpadded TMA box row pitch

__device__ __forceinline__ void g_mbar_arrive_tx(uint64_t* bar, unsigned tx) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("{\n .reg .b64 s;\n mbarrier.arrive.expect_tx.shared.b64 s, [%0], %1;\n}\n" ::"r"(a), "r"(tx));
}
__device__ __forceinline__ void g_tma_g2s_3d(void* smem, const CUtensorMap* map,
                                             int c0, int c1, int c2, uint64_t* bar) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3, %4}], [%5];\n"
               ::"r"(s), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(b));
}

__global__ void __launch_bounds__(TTHREADS, 2)
residue_gemm_tma_kernel(const __grid_constant__ CUtensorMap tmapA, const __grid_constant__ CUtensorMap tmapB,
                        int8_t* __restrict__ c8, int32_t* __restrict__ c32,
                        const int* __restrict__ primes, int P, int nbatch,
                        int M, int N, int K, int gswz) {
#if __CUDA_ARCH__ >= 900
  extern __shared__ __align__(128) int8_t smem[];
  int8_t* As = smem;
  int8_t* Bs = smem + TRING * GBM * TALD;
  const int ASTAGE = GBM * TALD, BSTAGE = TBN * TALD;
  __shared__ uint64_t full_bar[TRING], empty_bar[TRING];

  const int nmt = (M + GBM - 1) / GBM, nnt = (N + TBN - 1) / TBN;
  const int tile = blockIdx.x, gsz = gswz * nnt;
  const int gid = tile / gsz, gof = tile % gsz;
  const int m_first = gid * gswz, mh = min(gswz, nmt - m_first);
  const int mt = m_first + (gof % mh), nt = gof / mh;
  const int m0 = mt * GBM, n0 = nt * TBN;

  const int bz = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warpM = warp & 3, warpN = warp >> 2;
  const int rbase = warpM * 32, cbase_w = warpN * 64;
  const int nkt = (K + GBK - 1) / GBK;
  constexpr unsigned TXB = (unsigned)(GBM + TBN) * GBK;

  if (tid < TRING) { g_mbar_init(&full_bar[tid], 1); g_mbar_init(&empty_bar[tid], TTHREADS); }
  __syncthreads();

  auto stage = [&](int j) {
    const int st = j % TRING;
    if (tid == 0) {
      g_mbar_arrive_tx(&full_bar[st], TXB);
      g_tma_g2s_3d(As + st * ASTAGE, &tmapA, j * GBK, m0, bz, &full_bar[st]);
      g_tma_g2s_3d(Bs + st * BSTAGE, &tmapB, j * GBK, n0, bz, &full_bar[st]);
    }
  };

  int Cacc[2][8][4];
#pragma unroll
  for (int mi = 0; mi < 2; mi++)
#pragma unroll
    for (int ni = 0; ni < 8; ni++)
#pragma unroll
      for (int t = 0; t < 4; t++) Cacc[mi][ni][t] = 0;

  for (int s = 0; s < TAHEAD && s < nkt; s++) stage(s);
  for (int kt = 0; kt < nkt; kt++) {
    const int j = kt + TAHEAD;
    if (tid == 0 && j < nkt) {
      if (j >= TRING) g_mbar_wait(&empty_bar[j % TRING], ((j / TRING) - 1) & 1);
      stage(j);
    }
    g_mbar_wait(&full_bar[kt % TRING], (kt / TRING) & 1);
    int8_t* Ac = As + (kt % TRING) * ASTAGE;
    int8_t* Bc = Bs + (kt % TRING) * BSTAGE;
#pragma unroll
    for (int kk = 0; kk < GBK / 32; kk++) {
      unsigned a[2][4], b[8][2];
#pragma unroll
      for (int mi = 0; mi < 2; mi++) {
        const int8_t* arow = &Ac[(rbase + mi * 16 + (lane & 15)) * TALD];
        const unsigned aswz = ((unsigned)__cvta_generic_to_shared(arow) >> 7) & 3;
        const unsigned ac = (unsigned)(kk * 2 + (lane >> 4)) ^ aswz;
        g_ldm_x4(a[mi][0], a[mi][1], a[mi][2], a[mi][3],
                 reinterpret_cast<const uint16_t*>(arow + ac * 16));
      }
#pragma unroll
      for (int nb = 0; nb < 4; nb++) {
        const int g = lane >> 3;
        const int8_t* brow = &Bc[(cbase_w + (nb * 2 + (g >> 1)) * 8 + (lane & 7)) * TALD];
        const unsigned bswz = ((unsigned)__cvta_generic_to_shared(brow) >> 7) & 3;
        const unsigned bc = (unsigned)(kk * 2 + (g & 1)) ^ bswz;
        g_ldm_x4(b[nb * 2][0], b[nb * 2][1], b[nb * 2 + 1][0], b[nb * 2 + 1][1],
                 reinterpret_cast<const uint16_t*>(brow + bc * 16));
      }
#pragma unroll
      for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 8; ni++) g_mma_s8(Cacc[mi][ni], a[mi], b[ni]);
    }
    g_mbar_arrive(&empty_bar[kt % TRING]);
  }

  // Epilogue as in the cp.async kernel: direct paired stores, 31-bit Barrett.
  const bool residue = (bz < P * nbatch);
  const int p = residue ? primes[bz / nbatch] : 0;
  const unsigned mu31 = residue ? ((1u << 31) / (unsigned)p) : 0;
  const unsigned off = residue ? ((0x80000000u / (unsigned)p) * (unsigned)p) : 0;
  const long c8base = (long)bz * M * N, c32base = (long)(bz - P * nbatch) * M * N;
  auto bal8 = [&](int v) -> int8_t {
    const unsigned xu = (unsigned)v + off;
    unsigned r = xu - (unsigned)(((unsigned long long)xu * mu31) >> 31) * (unsigned)p;
    while (r >= (unsigned)p) r -= (unsigned)p;
    int rb = (int)r;
    if (rb > p / 2) rb -= p;
    return (int8_t)rb;
  };
  const int r00 = m0 + rbase + (lane >> 2);
  const int c00 = n0 + cbase_w + (lane & 3) * 2;
  const bool interior = (m0 + GBM <= M) && (n0 + TBN <= N) && ((N & 1) == 0);
#pragma unroll
  for (int mi = 0; mi < 2; mi++)
#pragma unroll
    for (int ni = 0; ni < 8; ni++) {
      const int r = r00 + mi * 16, c = c00 + ni * 8;
      if (residue) {
        const char2 v0 = make_char2(bal8(Cacc[mi][ni][0]), bal8(Cacc[mi][ni][1]));
        const char2 v1 = make_char2(bal8(Cacc[mi][ni][2]), bal8(Cacc[mi][ni][3]));
        if (interior) {
          *reinterpret_cast<char2*>(&c8[c8base + (long)r * N + c]) = v0;
          *reinterpret_cast<char2*>(&c8[c8base + (long)(r + 8) * N + c]) = v1;
        } else {
          if (r < M && c < N) c8[c8base + (long)r * N + c] = v0.x;
          if (r < M && c + 1 < N) c8[c8base + (long)r * N + c + 1] = v0.y;
          if (r + 8 < M && c < N) c8[c8base + (long)(r + 8) * N + c] = v1.x;
          if (r + 8 < M && c + 1 < N) c8[c8base + (long)(r + 8) * N + c + 1] = v1.y;
        }
      } else {
        if (interior) {
          *reinterpret_cast<int2*>(&c32[c32base + (long)r * N + c]) =
              make_int2(Cacc[mi][ni][0], Cacc[mi][ni][1]);
          *reinterpret_cast<int2*>(&c32[c32base + (long)(r + 8) * N + c]) =
              make_int2(Cacc[mi][ni][2], Cacc[mi][ni][3]);
        } else {
          if (r < M && c < N) c32[c32base + (long)r * N + c] = Cacc[mi][ni][0];
          if (r < M && c + 1 < N) c32[c32base + (long)r * N + c + 1] = Cacc[mi][ni][1];
          if (r + 8 < M && c < N) c32[c32base + (long)(r + 8) * N + c] = Cacc[mi][ni][2];
          if (r + 8 < M && c + 1 < N) c32[c32base + (long)(r + 8) * N + c + 1] = Cacc[mi][ni][3];
        }
      }
    }
#endif
}

// Elementwise accumulators for the K-segmented pipeline: balanced int8
// residue planes sum into int32 (bounded by segments * 128), raw int32
// correction planes into int64.
__global__ void accum_res_kernel(const int8_t* __restrict__ src, int32_t* __restrict__ acc,
                                 int64_t total) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x)
    acc[i] += (int32_t)src[i];
}

__global__ void accum_corr_kernel(const int32_t* __restrict__ src, int64_t* __restrict__ acc,
                                  int64_t total) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (int64_t)gridDim.x * blockDim.x)
    acc[i] += (int64_t)src[i];
}

// Reduce a chunk of raw int32 product planes to balanced int8 residues; src
// is [nplanes, total] as cuBLAS wrote it, dst the matching residue slice.
__global__ void reduce_planes_kernel(const int32_t* __restrict__ src, int8_t* __restrict__ dst,
                                     int64_t total, int nplanes, int nbatch,
                                     const int* __restrict__ primes, int p0, int64_t kdepth) {
  for (int pl = blockIdx.y; pl < nplanes; pl += gridDim.y) {
    const unsigned p = (unsigned)primes[(p0 + pl) / nbatch];
    const uint32_t mu32 = (uint32_t)((1u << 31) / p);
    const uint64_t half = p / 2, maxabs = (uint64_t)kdepth * half * half;
    const uint32_t bias = (uint32_t)(p * ((maxabs + p - 1) / p));
    const int64_t off = (int64_t)pl * total;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x)
      dst[off + i] = (int8_t)barrett_bal32(src[off + i], (int)p, mu32, bias);
  }
}

// Balanced-Garner reconstruction + rescale. res: [num_primes, rows*cols],
// int8 balanced residues or raw int32 reduced on load. Mixed-radix digits
//   d[t] = ( r_t - sum_{u<t} d[u]*(M_u mod p_t) ) * (M_t mod p_t)^{-1} (mod p_t)
// with factors balanced to int8 and packed four per word, so the O(P^2)
// inner product runs as dp4a. C[i,j] = P[i,j] * 2^(-sftA[i]-sftB[j]).
template <typename RT, typename CT>
__global__ void reconstruct_kernel(const RT* __restrict__ res8, int64_t rows,
                                   int64_t cols, int64_t rows_per,
                                   int num_primes, const int* __restrict__ primes,
                                   const int* __restrict__ mppack,
                                   const int* __restrict__ Minv,
                                   const long long* __restrict__ m64,
                                   const int16_t* __restrict__ sftA, const int16_t* __restrict__ sftB,
                                   const CT* __restrict__ corr1, const CT* __restrict__ corr2,
                                   const int* __restrict__ bits_p, int64_t kdepth,
                                   bool use_corr, bool exact,
                                   double* __restrict__ C, int64_t ldc) {
  const int cshift = *bits_p - 15;   // correction plane scale; see extract_kernel
  const int NP = num_primes;
  // Tables reused by every output element; stage them in shared once per block.
  __shared__ int s_p[MAXP];
  __shared__ int s_minv[MAXP];
  __shared__ uint32_t s_mu32[MAXP], s_bias[MAXP];
  __shared__ int s_mp[MAXP * MPGROUPS];
  __shared__ long long s_m64[2];
  for (int t0 = threadIdx.x; t0 < 2; t0 += blockDim.x) s_m64[t0] = m64[t0];
  for (int t0 = threadIdx.x; t0 < num_primes; t0 += blockDim.x) {
    const unsigned p = (unsigned)primes[t0];
    s_p[t0] = (int)p; s_minv[t0] = Minv[t0];
    s_mu32[t0] = (uint32_t)((1u << 31) / p);
    // Bias covering the raw int32 plane bound K*(p/2)^2; it grows with p^2,
    // and a fixed constant would overflow uint32 near p = 255.
    const uint64_t half = p / 2, maxabs = (uint64_t)kdepth * half * half;
    s_bias[t0] = (uint32_t)(p * ((maxabs + p - 1) / p));
  }
  for (int t0 = threadIdx.x; t0 < num_primes * MPGROUPS; t0 += blockDim.x) s_mp[t0] = mppack[t0];
  __syncthreads();

  const int64_t total = rows * cols;
  // Row index from the grid's y dimension; no 64-bit div/mod per element.
  for (int64_t i = blockIdx.y; i < rows; i += gridDim.y) {
  const int16_t sa = sftA[i];
  // Batched calls flatten rows; each batch member has its own column shifts.
  const int16_t* __restrict__ sftBb = sftB + (i / rows_per) * cols;
  const int64_t irow = i * cols;
  for (int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < cols;
       j += (int64_t)gridDim.x * blockDim.x) {
    const int64_t idx = irow + j;
    // Residue planes are contiguous, so res8[t*total + idx] is coalesced per
    // t; digits live packed four per int32 word in registers.
    int32_t dpack[MPGROUPS];
#pragma unroll
    for (int g = 0; g < MPGROUPS; g++) dpack[g] = 0;
    auto dig = [&](int t) -> int {
      return (int)(signed char)(uint8_t)((uint32_t)dpack[t >> 2] >> ((t & 3) * 8));
    };
    // Both digit loops unroll to MAXP and cut at the runtime count so every
    // dpack index is compile-time; runtime indexing demotes to local memory.
#pragma unroll
    for (int t = 0; t < MAXP; t++) {
      if (t >= NP) break;
      const int rt = (sizeof(RT) == 1)
                         ? (int)res8[(int64_t)t * total + idx]
                         : barrett_bal32((int)res8[(int64_t)t * total + idx], s_p[t], s_mu32[t],
                                         s_bias[t]);
      int acc = 0;   // sum_{u<t} d[u] * (M_u mod p_t); |acc| < 2^19
#pragma unroll
      for (int g = 0; g < MPGROUPS; g++) acc = __dp4a(dpack[g], s_mp[t * MPGROUPS + g], acc);
      const int y = (rt - acc) * s_minv[t];    // |y| < 2^27
      const int dt = barrett_bal32(y, s_p[t], s_mu32[t], (uint32_t)s_p[t] << 21);
      dpack[t >> 2] |= ((int32_t)(uint8_t)(signed char)dt) << ((t & 3) * 8);
    }
    // Recombine: one int64 Horner per group of <= 7 moduli (in-group partials
    // stay under 2^55), groups combined high-to-low with one wide multiply
    // each, value = a0 + M0*(a1 + M1*a2); then the correction (corr1+corr2)*
    // 2^cshift and a single rounding. exact combines in int128, else in fp64.
    const int ng = (NP + 6) / 7;
    long long a0 = 0, a1 = 0, a2 = 0;
#pragma unroll
    for (int t = MAXP - 1; t >= 0; t--) {
      if (t >= NP) continue;
      const long long d = (long long)dig(t);
      if (t >= 14) a2 = a2 * s_p[t] + d;
      else if (t >= 7) a1 = a1 * s_p[t] + d;
      else a0 = a0 * s_p[t] + d;
    }
    const long long corr = use_corr ? (long long)corr1[idx] + (long long)corr2[idx] : 0LL;
    const int sh = -(int)sa - (int)sftBb[j];
    if (exact) {
      s128 acc = s128_from(ng == 3 ? a2 : (ng == 2 ? a1 : a0));
      if (ng == 3) acc = s128_mul64_add(acc, (unsigned long long)s_m64[1], a1);
      if (ng >= 2) acc = s128_mul64_add(acc, (unsigned long long)s_m64[0], a0);
      acc = s128_add(acc, s128_shl(s128_from(corr), cshift));
      C[i * ldc + j] = ldexp(s128_to_double(acc), sh);
    } else {
      double acc = (double)(ng == 3 ? a2 : (ng == 2 ? a1 : a0));
      if (ng == 3) acc = acc * (double)s_m64[1] + (double)a1;
      if (ng >= 2) acc = acc * (double)s_m64[0] + (double)a0;
      acc += ldexp((double)corr, cshift);
      C[i * ldc + j] = ldexp(acc, sh);
    }
  }
  }
}

}  // namespace

int64_t fp64_emu_maxp() { return MAXP; }

// Modular inverse of a mod m (m prime, small), brute force over [1, m).
static int modinv_host(int a, int m) {
  a %= m; if (a < 0) a += m;
  for (int x = 1; x < m; x++) if ((long long)a * x % m == 1) return x;
  return 0;   // unreachable for prime m, gcd(a,m)=1
}

// Host reconstruction tables: mppack[t*GROUPS+g] holds the balanced int8
// factors M_u mod p_t for u = 4g..4g+3, zeroed at u >= t; minv[t] =
// (M_t mod p_t)^{-1}; m64[g] = prod of moduli 7g..7g+6 (< 2^57) for the
// grouped recombine, identity for empty groups.
static void build_recon_tables(const int* primes, int P, std::vector<int>& mppack,
                               std::vector<int>& minv, std::vector<long long>& m64) {
  mppack.assign((size_t)P * MPGROUPS, 0);
  minv.assign(P, 0);
  m64.assign(2, 1);
  for (int g = 0; g < 2; g++) {
    long long prod = 1;
    for (int t = g * 7; t < std::min(g * 7 + 7, P); t++) prod *= primes[t];
    m64[g] = prod;
  }
  for (int t = 0; t < P; t++) {
    const long long pt = primes[t];
    long long run = 1 % pt;                     // M_0 mod p_t
    for (int u = 0; u < P; u++) {
      if (u == t) minv[t] = modinv_host((int)run, (int)pt);   // (M_t mod p_t)^{-1}
      if (u < t) {
        int bal = (int)run;                     // balance to [-p/2, (p-1)/2]
        if (bal > ((int)pt - 1) / 2) bal -= (int)pt;
        mppack[(size_t)t * MPGROUPS + (u >> 2)] |=
            ((int)(unsigned char)(signed char)bal) << ((u & 3) * 8);
      }
      run = run * (primes[u] % pt) % pt;        // -> M_{u+1} mod p_t
    }
  }
}

// cuTensorMapEncodeTiled fetched through the runtime (no libcuda link dep).
static void* fp64emu_tmap_fn() {
  static void* fn = [] {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult qr;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &f, cudaEnableDefault, &qr) != cudaSuccess) f = nullptr;
    return f;
  }();
  return fn;
}

// 3D int8 tensor map for [batch, rows, K] with a (1, box_rows, GBK) box,
// hardware SWIZZLE_64B. Fails (false) when unavailable or the layout is
// rejected (K % 16 != 0 etc), in which case the cp.async kernel runs instead.
static bool fp64emu_encode_tmap(CUtensorMap* map, const void* base, int64_t batch, int64_t rows,
                                int64_t K, uint32_t box_rows) {
  using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  EncodeFn fn = reinterpret_cast<EncodeFn>(fp64emu_tmap_fn());
  if (!fn) return false;
  const cuuint64_t gdim[3] = {(cuuint64_t)K, (cuuint64_t)rows, (cuuint64_t)batch};
  const cuuint64_t gstr[2] = {(cuuint64_t)K, (cuuint64_t)(rows * K)};
  const cuuint32_t box[3] = {(cuuint32_t)GBK, box_rows, 1};
  const cuuint32_t els[3] = {1, 1, 1};
  return fn(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(base), gdim, gstr, box, els,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// Device-side reconstruction tables, cached per modulus set and device.
struct ReconTables {
  torch::Tensor primes, mppack, minv, m64;
  float log2M = 0.0f;   // total CRT range, for the Cauchy-Schwarz sizing
};

static const ReconTables& recon_tables(torch::Tensor const& primes_cpu, torch::Device dev) {
  auto pc = primes_cpu.to(torch::kCPU, torch::kInt32).contiguous();
  const int P = (int)pc.numel();
  std::vector<int> key(pc.const_data_ptr<int>(), pc.const_data_ptr<int>() + P);
  const auto full_key = std::make_pair(key, (int)dev.index());

  static std::mutex mtx;
  static std::map<std::pair<std::vector<int>, int>, ReconTables> cache;
  std::lock_guard<std::mutex> g(mtx);
  auto it = cache.find(full_key);
  if (it != cache.end()) return it->second;

  std::vector<int> h_mppack, h_minv;
  std::vector<long long> h_m64;
  build_recon_tables(key.data(), P, h_mppack, h_minv, h_m64);
  ReconTables t;
  t.primes = pc.to(dev);
  t.mppack = torch::from_blob(h_mppack.data(), {(int64_t)P * MPGROUPS}, torch::kInt32).to(dev);
  t.minv = torch::from_blob(h_minv.data(), {P}, torch::kInt32).to(dev);
  t.m64 = torch::from_blob(h_m64.data(), {2}, torch::kInt64).to(dev);
  for (int i = 0; i < P; i++) t.log2M += (float)std::log2((double)key[i]);
  return cache.emplace(full_key, std::move(t)).first->second;
}

// The faster cuBLAS INT8 entry point (strided-batched vs a per-plane loop) is
// architecture dependent; it is timed once per shape and cached.
struct GemmShape {
  int P;
  int64_t M, N, K;
  bool operator<(const GemmShape& o) const {
    if (P != o.P) return P < o.P;
    if (M != o.M) return M < o.M;
    if (N != o.N) return N < o.N;
    return K < o.K;
  }
};

template <typename FS, typename FL>
static bool prefer_gemm_loop(const GemmShape& key, FS&& strided, FL&& loop, cudaStream_t stream) {
  static std::mutex mtx;
  static std::map<GemmShape, bool> cache;
  {
    std::lock_guard<std::mutex> g(mtx);
    auto it = cache.find(key);
    if (it != cache.end()) {
      (it->second ? loop() : strided());
      return it->second;
    }
  }
  cudaEvent_t e0, e1, e2;
  cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  cudaEventRecord(e0, stream);
  strided();
  cudaEventRecord(e1, stream);
  loop();                       // same result; the second write wins
  cudaEventRecord(e2, stream);
  cudaEventSynchronize(e2);
  float ts = 0.f, tl = 0.f;
  cudaEventElapsedTime(&ts, e0, e1);
  cudaEventElapsedTime(&tl, e1, e2);
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
  const bool use_loop = tl < ts;
  std::lock_guard<std::mutex> g(mtx);
  cache[key] = use_loop;
  return use_loop;
}

// Launch the reconstruction over either residue type.
template <typename RT, typename CT = int32_t>
static void launch_reconstruct(const RT* src, dim3 grid, cudaStream_t stream, int P, int64_t rows,
                               int64_t N, int64_t rows_per, const int* primes, const int* mppack,
                               const int* minv, const long long* m64, const int16_t* sftA,
                               const int16_t* sftB, const CT* c1, const CT* c2,
                               const int* bits_p, int64_t kdepth, bool use_corr, bool exact,
                               double* out) {
  reconstruct_kernel<RT, CT><<<grid, kThreads, 0, stream>>>(
      src, rows, N, rows_per, P, primes, mppack, minv, m64, sftA, sftB, c1, c2, bits_p, kdepth,
      use_corr, exact, out, N);
}

// FP64EMU_PATH=fused|cublas|chunked pins the modular-GEMM path; unset
// dispatches by the measured rule. Read per call so one process can A/B.
static int path_override() {
  const char* s = std::getenv("FP64EMU_PATH");
  if (!s) return 0;
  if (std::strcmp(s, "fused") == 0) return 1;
  if (std::strcmp(s, "cublas") == 0) return 2;
  if (std::strcmp(s, "chunked") == 0) return 3;
  return 0;
}

// Shared pipeline: scale, extract, the (P+2)*batch modular GEMMs, fused
// reconstruct. Batched operands flatten to [batch*M, K] / [batch*N, K] rows;
// M, N, K are per batch member.
// e[0..4], when non-null, bracket [scale+extract, gemms, marker, reconstruct].
static void run_crt(torch::Tensor& out, torch::Tensor const& A, torch::Tensor const& B,
                    int64_t batch, int64_t M, int64_t K, int64_t N,
                    torch::Tensor const& primes_cpu, torch::Tensor const& inv_cpu,
                    torch::Tensor const& mu_cpu, int bits, bool use_corr, bool exact,
                    bool b_nt, cudaStream_t stream, cudaEvent_t* e) {
  const int64_t MF = batch * M, NF = batch * N;
  const int P = (int)primes_cpu.numel();
  auto i8 = A.options().dtype(torch::kInt8);
  auto i32 = A.options().dtype(torch::kInt32);
  auto i16 = A.options().dtype(torch::kInt16);
  // mu_cpu and inv_cpu predate the 32-bit reductions and derived tables;
  // accepted for ABI compatibility, never uploaded.
  (void)mu_cpu;
  (void)inv_cpu;
  const ReconTables& tab = recon_tables(primes_cpu, A.device());
  const torch::Tensor& primes = tab.primes;
  const torch::Tensor& mppack = tab.mppack;
  const torch::Tensor& minv = tab.minv;
  const torch::Tensor& mm64 = tab.m64;

  if (e) cudaEventRecord(e[0], stream);
  auto sftA = torch::empty({MF}, i16), sftB = torch::empty({NF}, i16);
  auto f32 = A.options().dtype(torch::kFloat32);
  auto lgA = torch::empty({MF}, f32), lgB = torch::empty({NF}, f32);
  auto bits_dev = torch::empty({1}, i32);
  if (b_nt) {
    // B arrives as [batch, K, N]: its shifts are per column, gathered by the
    // coalesced slab kernels; A keeps the per-row pass.
    row_shift_kernel<<<MF, kThreads, 0, stream>>>(
        A.const_data_ptr<double>(), A.const_data_ptr<double>(), MF, 0, K, bits,
        sftA.data_ptr<int16_t>(), lgA.data_ptr<float>(),
        sftB.data_ptr<int16_t>(), lgB.data_ptr<float>());
    const int64_t nsl0 =
        std::min<int64_t>(std::max<int64_t>(131072 / std::max<int64_t>(NF, 1), 1), 128);
    const int64_t kslab = (K + nsl0 - 1) / nsl0;
    const int nslab = (int)((K + kslab - 1) / kslab);
    auto f64o = A.options().dtype(torch::kFloat64);
    auto pmax = torch::empty({(int64_t)nslab, NF}, f64o);
    auto pss = torch::empty({(int64_t)nslab, NF}, f64o);
    const int gcx = (int)((NF + kThreads - 1) / kThreads);
    col_shift_partial_kernel<<<dim3(gcx, nslab), kThreads, 0, stream>>>(
        B.const_data_ptr<double>(), NF, K, N, kslab, pmax.data_ptr<double>(),
        pss.data_ptr<double>());
    col_shift_finalize_kernel<<<gcx, kThreads, 0, stream>>>(
        pmax.const_data_ptr<double>(), pss.const_data_ptr<double>(), NF, nslab, bits,
        sftB.data_ptr<int16_t>(), lgB.data_ptr<float>());
  } else {
    row_shift_kernel<<<MF + NF, kThreads, 0, stream>>>(
        A.const_data_ptr<double>(), B.const_data_ptr<double>(), MF, NF, K, bits,
        sftA.data_ptr<int16_t>(), lgA.data_ptr<float>(),
        sftB.data_ptr<int16_t>(), lgB.data_ptr<float>());
  }
  // Unused CRT range becomes extra mantissa bits; the bound is Cauchy-Schwarz,
  // taken over the whole batch, so one operand width serves every member. The
  // range is capped at 2^126 so the int128 recombine holds at any depth.
  pick_bits_kernel<<<1, kThreads, 0, stream>>>(
      lgA.const_data_ptr<float>(), MF, lgB.const_data_ptr<float>(), NF,
      std::min(tab.log2M, 126.0f), bits,
      sftA.data_ptr<int16_t>(), sftB.data_ptr<int16_t>(), bits_dev.data_ptr<int>());
  const int* bits_p = bits_dev.const_data_ptr<int>();
  // Engine chosen before the extract, whose operand layout depends on it:
  // cuBLAS consumes row-major, the cp.async fused kernel the tiled layout.
  const bool aligned4 = (M % 4 == 0) && (N % 4 == 0) && (K % 4 == 0);
  // The cuBLAS branch materializes NP raw int32 product planes, ~4x the fused
  // branch's int8 residues; it is skipped when they would not comfortably fit.
  const int NP = use_corr ? P + 2 : P;
  const double planes_bytes = (double)NP * (double)batch * (double)M * (double)N * 4.0;
  // The fit test charges the not-yet-allocated operand planes too; the driver
  // is queried only when the answer can change the choice.
  const double operand_bytes = (double)(P + 2) * (double)(MF + NF) * (double)K;
  size_t free_b = 0, total_b = 0;
  bool planes_fit = true;
  if (planes_bytes + operand_bytes >= 128.0 * 1024.0 * 1024.0) {
    cudaMemGetInfo(&free_b, &total_b);
    planes_fit = planes_bytes + operand_bytes < 0.80 * (double)free_b;
  }
  const bool has_tma = at::cuda::getCurrentDeviceProperties()->major >= 9;
  const int ov = path_override();
  // Paths: 0 = fused kernel (int8-residue epilogue), 1 = cuBLAS into full
  // int32 planes, 2 = cuBLAS chunked with on-the-fly reduction. With no
  // override the fused-vs-cuBLAS choice is timed once per shape and cached.
  int path_sel;
  if (ov == 1) path_sel = 0;
  else if (ov == 2) path_sel = aligned4 ? (planes_fit ? 1 : 2) : 0;
  else if (ov == 3) path_sel = aligned4 ? 2 : 0;
  else if (!aligned4) path_sel = 0;
  else path_sel = -1;   // measured below

  // FP64EMU_NO_TMA pins the fused path to cp.async staging; FP64EMU_NO_TILED
  // keeps the cp.async kernel on the row-major operand layout.
  const bool no_tma = std::getenv("FP64EMU_NO_TMA") != nullptr;
  const bool no_tiled = std::getenv("FP64EMU_NO_TILED") != nullptr;

  auto run_path = [&](int psel) {
  const bool cublas_sel = psel >= 1;
  const bool chunk_sel = psel == 2;
  // Tile-blocked operands whenever the cp.async fused kernel will run; cuBLAS
  // and the TMA tensor maps require row-major.
  const bool tiled = !cublas_sel && !(has_tma && !no_tma && (K & 15) == 0) && !no_tiled;
  // nt inputs feed cuBLAS in their [K, N] layout through the NN form; the
  // fused engine consumes [N, K], so its candidate pays a transpose copy that
  // the probe times against the direct form.
  const bool ntdirect = b_nt && cublas_sel;
  torch::Tensor Buse;
  const double* Bptr = B.const_data_ptr<double>();
  int64_t browsF = NF, brows_per = N, bcols = K;
  if (b_nt && !ntdirect) {
    Buse = (B.dim() == 3) ? B.transpose(1, 2).contiguous() : B.t().contiguous();
    Bptr = Buse.const_data_ptr<double>();
  } else if (ntdirect) {
    browsF = batch * K;
    brows_per = K;
    bcols = N;
  }
  // Row-major [K, N] planes are column-major [N, K]: the NN form replaces the
  // TN form with the same per-slot stride.
  const cublasOperation_t opB = ntdirect ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int ldb = ntdirect ? (int)N : (int)K;

  // Operand batches, plane-major over plane-batch slots: A = [Alo; Ahi7;
  // Aq7], B = [Blo; Bq7; Bhi7], so correction slot group 0 is Ahi7 @ Bq7^T
  // and group 1 is Aq7 @ Bhi7^T. Ragged tiled edges start from zeroed planes,
  // so the pad multiplies as zero.
  const int64_t nktT = (K + GBK - 1) / GBK;
  const int64_t planeA = tiled ? ((M + GBM - 1) / GBM) * nktT * (int64_t)(GBM * GBK) : M * K;
  const int64_t planeB = tiled ? ((N + GBN - 1) / GBN) * nktT * (int64_t)(GBN * GBK) : N * K;
  const int64_t fplaneA = batch * planeA, fplaneB = batch * planeB;
  const bool ragged = tiled && ((M % GBM) || (N % GBN) || (K % GBK));
  auto Aall = ragged ? torch::zeros({NP, fplaneA}, i8) : torch::empty({NP, fplaneA}, i8);
  auto Ball = ragged ? torch::zeros({NP, fplaneB}, i8) : torch::empty({NP, fplaneB}, i8);
  int8_t* Ap = Aall.data_ptr<int8_t>();
  int8_t* Bp = Ball.data_ptr<int8_t>();
  extract_kernel<<<grid_rows(MF + browsF, K), kThreads, 0, stream>>>(
      A.const_data_ptr<double>(), Bptr, MF, browsF, M, brows_per, K, bcols, K, bcols, 0,
      ntdirect ? 1 : 0, tiled ? 1 : 0,
      use_corr ? 1 : 0,
      sftA.const_data_ptr<int16_t>(), sftB.const_data_ptr<int16_t>(), P,
      primes.const_data_ptr<int>(), bits_p,
      Ap, use_corr ? Ap + (int64_t)P * fplaneA : Ap,
      use_corr ? Ap + (int64_t)(P + 1) * fplaneA : Ap,
      Bp, use_corr ? Bp + (int64_t)(P + 1) * fplaneB : Bp,
      use_corr ? Bp + (int64_t)P * fplaneB : Bp);

  if (e) cudaEventRecord(e[1], stream);
  const int32_t* corr1_ptr;
  const int32_t* corr2_ptr;
  torch::Tensor res8, corr, craw;

  const int64_t slots = (int64_t)P * batch;
  if (cublas_sel && chunk_sel) {
    // Peak memory is set by the chunk: each slot group reduces to int8
    // before the next group multiplies.
    const int64_t total = M * N;
    res8 = torch::empty({slots, M, N}, i8);
    if (use_corr) corr = torch::empty({2 * batch, M, N}, i32);
    // Chunk size from the memory free now, operands already allocated.
    size_t fb2 = 0, tb2 = 0;
    cudaMemGetInfo(&fb2, &tb2);
    int64_t ch = (int64_t)((0.25 * (double)fb2) / ((double)total * 4.0));
    ch = std::min<int64_t>(std::max<int64_t>(ch, 1), std::min<int64_t>(8 * batch, slots));
    auto buf = torch::empty({ch, M, N}, i32);
    const int gx = (int)std::min<int64_t>((total + kThreads - 1) / kThreads, 2048);
    const int32_t alpha = 1, beta = 0;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const int8_t* Ac = Aall.const_data_ptr<int8_t>();
    const int8_t* Bc = Ball.const_data_ptr<int8_t>();
    auto gemm_group = [&](int64_t base, int c, void* dst) {
      cublasStatus_t cst = cublasGemmStridedBatchedEx(
          handle, opB, CUBLAS_OP_N, (int)N, (int)M, (int)K,
          &alpha, Bc + base * N * K, CUDA_R_8I, ldb, (int64_t)N * K,
          Ac + base * M * K, CUDA_R_8I, (int)K, (int64_t)M * K,
          &beta, dst, CUDA_R_32I, (int)N, total,
          c, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      TORCH_CHECK(cst == CUBLAS_STATUS_SUCCESS, "chunked cublasGemmStridedBatchedEx: ", (int)cst);
    };
    for (int64_t base = 0; base < slots; base += ch) {
      const int c = (int)std::min<int64_t>(ch, slots - base);
      gemm_group(base, c, buf.data_ptr<int32_t>());
      reduce_planes_kernel<<<dim3(gx, c), kThreads, 0, stream>>>(
          buf.const_data_ptr<int32_t>(), res8.data_ptr<int8_t>() + base * total,
          total, c, (int)batch, primes.const_data_ptr<int>(), (int)base, K);
    }
    if (use_corr) gemm_group(slots, (int)(2 * batch), corr.data_ptr<int32_t>());
    corr1_ptr = use_corr ? corr.const_data_ptr<int32_t>() : nullptr;
    corr2_ptr = use_corr ? corr1_ptr + batch * total : nullptr;
  } else if (cublas_sel) {
    craw = torch::empty({(int64_t)NP * batch, M, N}, i32);
    int32_t* cbase = craw.data_ptr<int32_t>();
    const int32_t alpha = 1, beta = 0;
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // Row-major C[m,n] = sum_k A[b][m,k] B[b][n,k] as column-major [N,M]:
    // op(A)=Ball^T [N,K], op(B)=Aall [K,M]. TN INT8, INT32 accumulate.
    const int8_t* Ac = Aall.const_data_ptr<int8_t>();
    const int8_t* Bc = Ball.const_data_ptr<int8_t>();
    const int64_t sA = M * K, sB = N * K, sC = M * N;
    const int nslots = (int)((int64_t)NP * batch);
    cublasStatus_t cst = CUBLAS_STATUS_SUCCESS;
    auto strided = [&] {
      cst = cublasGemmStridedBatchedEx(
          handle, opB, CUBLAS_OP_N, (int)N, (int)M, (int)K,
          &alpha, Bc, CUDA_R_8I, ldb, sB, Ac, CUDA_R_8I, (int)K, sA,
          &beta, cbase, CUDA_R_32I, (int)N, sC,
          nslots, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    };
    auto looped = [&] {
      for (int pl = 0; pl < nslots && cst == CUBLAS_STATUS_SUCCESS; pl++)
        cst = cublasGemmEx(
            handle, opB, CUBLAS_OP_N, (int)N, (int)M, (int)K,
            &alpha, Bc + pl * sB, CUDA_R_8I, ldb, Ac + pl * sA, CUDA_R_8I, (int)K,
            &beta, cbase + pl * sC, CUDA_R_32I, (int)N,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    };
    prefer_gemm_loop(GemmShape{nslots, M, N, K}, strided, looped, stream);
    TORCH_CHECK(cst == CUBLAS_STATUS_SUCCESS, "cublas INT8 GEMM failed: ", (int)cst);
    corr1_ptr = use_corr ? cbase + (int64_t)P * batch * M * N : nullptr;
    corr2_ptr = use_corr ? cbase + (int64_t)(P + 1) * batch * M * N : nullptr;
  } else {
    res8 = torch::empty({slots, M, N}, i8);  // balanced residues, one byte per element
    corr = torch::empty({use_corr ? 2 * batch : 0, M, N}, i32);   // correction cross terms
    const int nmt = (int)((M + GBM - 1) / GBM), nnt = (int)((N + GBN - 1) / GBN);
    // L2 swizzle width: wide once the tile grid is large, 8 otherwise.
    const int gswz = ((long)nmt * nnt >= 128L * 128L) ? 32 : 8;
    bool launched = false;
    // The TMA tensor maps describe row-major planes, so TMA is only coherent
    // when the extract wrote that layout.
    const int zslots = (int)((int64_t)NP * batch);
    if (!tiled && !no_tma && has_tma && (K & 15) == 0) {
      CUtensorMap tA, tB;
      if (fp64emu_encode_tmap(&tA, Ap, (int64_t)NP * batch, M, K, (uint32_t)GBM) &&
          fp64emu_encode_tmap(&tB, Bp, (int64_t)NP * batch, N, K, (uint32_t)TBN)) {
        const int tnnt = (int)((N + TBN - 1) / TBN);
        const int tsh = TRING * (GBM + TBN) * TALD;   // 49152
        static bool tset = false;
        if (!tset) {
          cudaFuncSetAttribute(residue_gemm_tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, tsh);
          tset = true;
        }
        residue_gemm_tma_kernel<<<dim3(nmt * tnnt, 1, zslots), TTHREADS, tsh, stream>>>(
            tA, tB, res8.data_ptr<int8_t>(), use_corr ? corr.data_ptr<int32_t>() : nullptr,
            primes.const_data_ptr<int>(), P, (int)batch, (int)M, (int)N, (int)K, gswz);
        launched = true;
      }
    }
    if (!launched) {
      const int gsh = GRING * (GBM + GBN) * GBK;   // 98304; > the epilogue's staging tile
      static bool gset = false;
      if (!gset) {
        cudaFuncSetAttribute(residue_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, gsh);
        gset = true;
      }
      residue_gemm_kernel<<<dim3(nmt * nnt, 1, zslots), GTHREADS, gsh, stream>>>(
          Ap, Bp, res8.data_ptr<int8_t>(), use_corr ? corr.data_ptr<int32_t>() : nullptr,
          primes.const_data_ptr<int>(), P, (int)batch, (int)M, (int)N, (int)K, gswz,
          tiled ? 1 : 0);
    }
    corr1_ptr = use_corr ? corr.const_data_ptr<int32_t>() : nullptr;
    corr2_ptr = use_corr ? corr1_ptr + batch * M * N : nullptr;
  }

  if (e) cudaEventRecord(e[2], stream);  // (no copy/reduce phase; marker only)
  if (e) cudaEventRecord(e[3], stream);
  const dim3 rgrid = grid_rows(MF, N);
  if (cublas_sel && !chunk_sel) {
    launch_reconstruct<int32_t>(
        craw.const_data_ptr<int32_t>(), rgrid, stream, P, MF, N, M,
        primes.const_data_ptr<int>(),
        mppack.const_data_ptr<int>(), minv.const_data_ptr<int>(),
        reinterpret_cast<const long long*>(mm64.const_data_ptr<int64_t>()),
        sftA.const_data_ptr<int16_t>(), sftB.const_data_ptr<int16_t>(),
        corr1_ptr, corr2_ptr, bits_p, K, use_corr, exact, out.data_ptr<double>());
  } else {
    launch_reconstruct<int8_t>(
        res8.const_data_ptr<int8_t>(), rgrid, stream, P, MF, N, M,
        primes.const_data_ptr<int>(),
        mppack.const_data_ptr<int>(), minv.const_data_ptr<int>(),
        reinterpret_cast<const long long*>(mm64.const_data_ptr<int64_t>()),
        sftA.const_data_ptr<int16_t>(), sftB.const_data_ptr<int16_t>(),
        corr1_ptr, corr2_ptr, bits_p, K, use_corr, exact, out.data_ptr<double>());
  }
  if (e) cudaEventRecord(e[4], stream);
  };   // run_path

  if (path_sel >= 0) {
    run_path(path_sel);
    return;
  }
  // Measured choice, cached per (plane count, shape, device, alternative);
  // both candidates compute identical bits, so the overwrite is harmless.
  const int alt = planes_fit ? 1 : 2;
  struct PathKey {
    int NP;
    long long B, M, N, K;
    int dev, alt, nt;
    bool operator<(const PathKey& o) const {
      return std::tie(NP, B, M, N, K, dev, alt, nt) <
             std::tie(o.NP, o.B, o.M, o.N, o.K, o.dev, o.alt, o.nt);
    }
  };
  static std::mutex pmtx;
  static std::map<PathKey, int> pcache;
  const PathKey key{NP, (long long)batch, (long long)M, (long long)N, (long long)K,
                    (int)A.device().index(), alt, b_nt ? 1 : 0};
  int cached = -1;
  {
    std::lock_guard<std::mutex> g(pmtx);
    auto it = pcache.find(key);
    if (it != pcache.end()) cached = it->second;
  }
  if (cached >= 0) {
    run_path(cached);
    return;
  }
  // Each candidate warms before its timed run: one-time setup and cold clocks
  // are comparable to the real gap, and a cold sample picks the wrong path.
  const bool warm_probe = true;
  cudaEvent_t p0, p1, p2, p3;
  cudaEventCreate(&p0);
  cudaEventCreate(&p1);
  cudaEventCreate(&p2);
  cudaEventCreate(&p3);
  if (warm_probe) run_path(0);
  cudaEventRecord(p0, stream);
  run_path(0);
  cudaEventRecord(p1, stream);
  bool alt_ok = true;
  try {
    if (warm_probe) run_path(alt);
    cudaEventRecord(p2, stream);
    run_path(alt);
  } catch (const c10::Error&) {
    alt_ok = false;   // typically an allocation failure on the int32 planes
    cudaEventRecord(p2, stream);
  }
  cudaEventRecord(p3, stream);
  cudaEventSynchronize(p3);
  float tf = 0.f, ta = 0.f;
  cudaEventElapsedTime(&tf, p0, p1);
  cudaEventElapsedTime(&ta, p2, p3);
  cudaEventDestroy(p0);
  cudaEventDestroy(p1);
  cudaEventDestroy(p2);
  cudaEventDestroy(p3);
  const int winner = (!alt_ok || tf <= ta) ? 0 : alt;
  {
    std::lock_guard<std::mutex> g(pmtx);
    pcache[key] = winner;
  }
  // Drop the probe's cached blocks so the winner's layout rebuilds cleanly.
  c10::cuda::CUDACachingAllocator::emptyCache();
  if (!alt_ok) run_path(0);   // the failed run may have left out incomplete
}

// K-segmented pipeline for depths past the single-launch int32 accumulator
// bound: each 65536-column segment runs the fused engine in place (kstride /
// koff), balanced residues accumulate in int32 (bounded by segments * 128)
// and correction products in int64; one fold to int8 and one reconstruction
// close the sum. The range cap in pick_bits keeps the int128 recombine sound
// at any depth.
static void run_crt_bigk(torch::Tensor& out, torch::Tensor const& A, torch::Tensor const& B,
                         int64_t batch, int64_t M, int64_t K, int64_t N,
                         torch::Tensor const& primes_cpu, int bits, bool use_corr, bool exact,
                         cudaStream_t stream) {
  const int64_t MF = batch * M, NF = batch * N;
  const int P = (int)primes_cpu.numel();
  const int NP = use_corr ? P + 2 : P;
  auto i8 = A.options().dtype(torch::kInt8);
  auto i16 = A.options().dtype(torch::kInt16);
  auto i32 = A.options().dtype(torch::kInt32);
  auto i64o = A.options().dtype(torch::kInt64);
  const ReconTables& tab = recon_tables(primes_cpu, A.device());
  const torch::Tensor& primes = tab.primes;

  auto sftA = torch::empty({MF}, i16), sftB = torch::empty({NF}, i16);
  auto f32 = A.options().dtype(torch::kFloat32);
  auto lgA = torch::empty({MF}, f32), lgB = torch::empty({NF}, f32);
  auto bits_dev = torch::empty({1}, i32);
  row_shift_kernel<<<MF + NF, kThreads, 0, stream>>>(
      A.const_data_ptr<double>(), B.const_data_ptr<double>(), MF, NF, K, bits,
      sftA.data_ptr<int16_t>(), lgA.data_ptr<float>(),
      sftB.data_ptr<int16_t>(), lgB.data_ptr<float>());
  pick_bits_kernel<<<1, kThreads, 0, stream>>>(
      lgA.const_data_ptr<float>(), MF, lgB.const_data_ptr<float>(), NF,
      std::min(tab.log2M, 126.0f), bits,
      sftA.data_ptr<int16_t>(), sftB.data_ptr<int16_t>(), bits_dev.data_ptr<int>());
  const int* bits_p = bits_dev.const_data_ptr<int>();

  const int64_t Kc = 65536;
  const int64_t slots = (int64_t)P * batch;
  auto acc = torch::zeros({slots, M, N}, i32);
  torch::Tensor corr64;
  if (use_corr) corr64 = torch::zeros({2 * batch, M, N}, i64o);
  auto res8 = torch::empty({slots, M, N}, i8);
  auto corr = torch::empty({use_corr ? 2 * batch : 0, M, N}, i32);

  const int nmt = (int)((M + GBM - 1) / GBM), nnt = (int)((N + GBN - 1) / GBN);
  const int gswz = ((long)nmt * nnt >= 128L * 128L) ? 32 : 8;
  const int zslots = (int)((int64_t)NP * batch);
  const int gsh = GRING * (GBM + GBN) * GBK;
  static bool bkset = false;
  if (!bkset) {
    cudaFuncSetAttribute(residue_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, gsh);
    bkset = true;
  }
  const int64_t etotal = slots * M * N;
  const int egx = (int)std::min<int64_t>((etotal + kThreads - 1) / kThreads, 4096);

  for (int64_t k0 = 0; k0 < K; k0 += Kc) {
    const int64_t kc = std::min(Kc, K - k0);
    const int64_t nktT = (kc + GBK - 1) / GBK;
    const int64_t planeA = (int64_t)((M + GBM - 1) / GBM) * nktT * (GBM * GBK);
    const int64_t planeB = (int64_t)((N + GBN - 1) / GBN) * nktT * (GBN * GBK);
    const int64_t fplaneA = batch * planeA, fplaneB = batch * planeB;
    const bool ragged = (M % GBM) || (N % GBN) || (kc % GBK);
    auto Aall = ragged ? torch::zeros({NP, fplaneA}, i8) : torch::empty({NP, fplaneA}, i8);
    auto Ball = ragged ? torch::zeros({NP, fplaneB}, i8) : torch::empty({NP, fplaneB}, i8);
    int8_t* Ap = Aall.data_ptr<int8_t>();
    int8_t* Bp = Ball.data_ptr<int8_t>();
    extract_kernel<<<grid_rows(MF + NF, kc), kThreads, 0, stream>>>(
        A.const_data_ptr<double>(), B.const_data_ptr<double>(), MF, NF, M, N, kc, kc, K, K,
        k0, 0, 1, use_corr ? 1 : 0,
        sftA.const_data_ptr<int16_t>(), sftB.const_data_ptr<int16_t>(), P,
        primes.const_data_ptr<int>(), bits_p,
        Ap, use_corr ? Ap + (int64_t)P * fplaneA : Ap,
        use_corr ? Ap + (int64_t)(P + 1) * fplaneA : Ap,
        Bp, use_corr ? Bp + (int64_t)(P + 1) * fplaneB : Bp,
        use_corr ? Bp + (int64_t)P * fplaneB : Bp);
    residue_gemm_kernel<<<dim3(nmt * nnt, 1, zslots), GTHREADS, gsh, stream>>>(
        Ap, Bp, res8.data_ptr<int8_t>(), use_corr ? corr.data_ptr<int32_t>() : nullptr,
        primes.const_data_ptr<int>(), P, (int)batch, (int)M, (int)N, (int)kc, gswz, 1);
    accum_res_kernel<<<egx, kThreads, 0, stream>>>(res8.const_data_ptr<int8_t>(),
                                                   acc.data_ptr<int32_t>(), etotal);
    if (use_corr)
      accum_corr_kernel<<<egx, kThreads, 0, stream>>>(
          corr.const_data_ptr<int32_t>(), corr64.data_ptr<int64_t>(), 2 * batch * M * N);
  }

  // Fold the accumulated residues to balanced int8; kdepth = 1 bounds the
  // bias at (p/2)^2, covering 128 segments of 128.
  const int64_t total = M * N;
  const int rgx = (int)std::min<int64_t>((total + kThreads - 1) / kThreads, 2048);
  reduce_planes_kernel<<<dim3(rgx, (unsigned)slots), kThreads, 0, stream>>>(
      acc.const_data_ptr<int32_t>(), res8.data_ptr<int8_t>(), total, (int)slots, (int)batch,
      primes.const_data_ptr<int>(), 0, 1);
  const int64_t* c1 = use_corr ? corr64.const_data_ptr<int64_t>() : nullptr;
  launch_reconstruct<int8_t, int64_t>(
      res8.const_data_ptr<int8_t>(), grid_rows(MF, N), stream, P, MF, N, M,
      primes.const_data_ptr<int>(), tab.mppack.const_data_ptr<int>(),
      tab.minv.const_data_ptr<int>(),
      reinterpret_cast<const long long*>(tab.m64.const_data_ptr<int64_t>()),
      sftA.const_data_ptr<int16_t>(), sftB.const_data_ptr<int16_t>(),
      c1, use_corr ? c1 + batch * total : nullptr,
      bits_p, K, use_corr, exact, out.data_ptr<double>());
}

// out[m,n] = A[m,k] @ B[n,k]^T. exact selects the int128 recombine; bits is
// the mantissa width kept per operand.
void fp64_emu_mm(torch::Tensor& out, torch::Tensor const& A, torch::Tensor const& B,
                 torch::Tensor const& primes_cpu, torch::Tensor const& inv_cpu,
                 torch::Tensor const& mu_cpu, bool exact, int64_t bits, bool use_corr) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kFloat64 &&
                  B.dtype() == torch::kFloat64,
              "A,B must be f64 CUDA");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.is_contiguous() && B.is_contiguous(), "2D contig");
  const int64_t M = A.size(0), K = A.size(1), N = B.size(0);
  TORCH_CHECK(B.size(1) == K, "expect B as [N, K]");
  TORCH_CHECK(out.dtype() == torch::kFloat64 && out.sizes() == torch::IntArrayRef({M, N}) &&
                  out.is_contiguous(),
              "out f64 [M,N]");
  TORCH_CHECK((int)primes_cpu.numel() <= MAXP, "too many primes");
  // Single-launch int32 accumulation holds to K = 131071 (a balanced residue
  // mod 256 reaches -128, so a plane accumulates K*128^2, 2^31 at K = 131072);
  // deeper products run K-segmented, to 128 segments of 65536.
  TORCH_CHECK(K <= (int64_t)128 * 65536, "K bound: 128 segments of 65536");
  TORCH_CHECK(bits >= 16 && bits <= 53, "bits must be in [16, 53]");
  const at::cuda::CUDAGuard guard(A.device());
  if (K <= 131071)
    run_crt(out, A, B, 1, M, K, N, primes_cpu, inv_cpu, mu_cpu, (int)bits, use_corr, exact,
            false, at::cuda::getCurrentCUDAStream(), nullptr);
  else
    run_crt_bigk(out, A, B, 1, M, K, N, primes_cpu, (int)bits, use_corr, exact,
                 at::cuda::getCurrentCUDAStream());
}

// out[m,n] = A[m,k] @ Bt[k,n] with Bt consumed in its [K, N] layout: no
// transpose copy on the direct path; the dispatch probe still times the
// transposed fused candidate against it.
void fp64_emu_mm_nt(torch::Tensor& out, torch::Tensor const& A, torch::Tensor const& Bt,
                    torch::Tensor const& primes_cpu, torch::Tensor const& inv_cpu,
                    torch::Tensor const& mu_cpu, bool exact, int64_t bits, bool use_corr) {
  TORCH_CHECK(A.is_cuda() && Bt.is_cuda() && A.dtype() == torch::kFloat64 &&
                  Bt.dtype() == torch::kFloat64,
              "A,Bt must be f64 CUDA");
  TORCH_CHECK(A.dim() == 2 && Bt.dim() == 2 && A.is_contiguous() && Bt.is_contiguous(),
              "2D contig");
  const int64_t M = A.size(0), K = A.size(1), N = Bt.size(1);
  TORCH_CHECK(Bt.size(0) == K, "expect Bt as [K, N]");
  TORCH_CHECK(out.dtype() == torch::kFloat64 && out.sizes() == torch::IntArrayRef({M, N}) &&
                  out.is_contiguous(),
              "out f64 [M,N]");
  TORCH_CHECK((int)primes_cpu.numel() <= MAXP, "too many primes");
  TORCH_CHECK(K <= 131071, "K bound for int32 modular GEMM");
  TORCH_CHECK(bits >= 16 && bits <= 53, "bits must be in [16, 53]");
  const at::cuda::CUDAGuard guard(A.device());
  run_crt(out, A, Bt, 1, M, K, N, primes_cpu, inv_cpu, mu_cpu, (int)bits, use_corr, exact,
          true, at::cuda::getCurrentCUDAStream(), nullptr);
}

// out[b,m,n] = A[b,m,k] @ B[b,n,k]^T, uniform shapes across the batch. One
// modular-GEMM launch covers every plane-batch slot; the z-extent caps the
// batch at (nprimes + 2) * batch <= 65535.
void fp64_emu_bmm(torch::Tensor& out, torch::Tensor const& A, torch::Tensor const& B,
                  torch::Tensor const& primes_cpu, torch::Tensor const& inv_cpu,
                  torch::Tensor const& mu_cpu, bool exact, int64_t bits, bool use_corr) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kFloat64 &&
                  B.dtype() == torch::kFloat64,
              "A,B must be f64 CUDA");
  TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && A.is_contiguous() && B.is_contiguous(),
              "3D contig");
  const int64_t batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(1);
  TORCH_CHECK(B.size(0) == batch && B.size(2) == K, "expect B as [batch, N, K]");
  TORCH_CHECK(out.dtype() == torch::kFloat64 &&
                  out.sizes() == torch::IntArrayRef({batch, M, N}) && out.is_contiguous(),
              "out f64 [batch,M,N]");
  const int64_t NP = primes_cpu.numel() + (use_corr ? 2 : 0);
  TORCH_CHECK((int)primes_cpu.numel() <= MAXP, "too many primes");
  TORCH_CHECK(NP * batch <= 65535, "(nprimes + 2) * batch exceeds the launch z-extent");
  TORCH_CHECK(K <= (int64_t)128 * 65536, "K bound: 128 segments of 65536");
  TORCH_CHECK(bits >= 16 && bits <= 53, "bits must be in [16, 53]");
  const at::cuda::CUDAGuard guard(A.device());
  if (K <= 131071)
    run_crt(out, A, B, batch, M, K, N, primes_cpu, inv_cpu, mu_cpu, (int)bits, use_corr, exact,
            false, at::cuda::getCurrentCUDAStream(), nullptr);
  else
    run_crt_bigk(out, A, B, batch, M, K, N, primes_cpu, (int)bits, use_corr, exact,
                 at::cuda::getCurrentCUDAStream());
}

// Batched form of fp64_emu_mm_nt: Bt is [batch, K, N], consumed in place.
void fp64_emu_bmm_nt(torch::Tensor& out, torch::Tensor const& A, torch::Tensor const& Bt,
                     torch::Tensor const& primes_cpu, torch::Tensor const& inv_cpu,
                     torch::Tensor const& mu_cpu, bool exact, int64_t bits, bool use_corr) {
  TORCH_CHECK(A.is_cuda() && Bt.is_cuda() && A.dtype() == torch::kFloat64 &&
                  Bt.dtype() == torch::kFloat64,
              "A,Bt must be f64 CUDA");
  TORCH_CHECK(A.dim() == 3 && Bt.dim() == 3 && A.is_contiguous() && Bt.is_contiguous(),
              "3D contig");
  const int64_t batch = A.size(0), M = A.size(1), K = A.size(2), N = Bt.size(2);
  TORCH_CHECK(Bt.size(0) == batch && Bt.size(1) == K, "expect Bt as [batch, K, N]");
  TORCH_CHECK(out.dtype() == torch::kFloat64 &&
                  out.sizes() == torch::IntArrayRef({batch, M, N}) && out.is_contiguous(),
              "out f64 [batch,M,N]");
  const int64_t NP = primes_cpu.numel() + (use_corr ? 2 : 0);
  TORCH_CHECK((int)primes_cpu.numel() <= MAXP, "too many primes");
  TORCH_CHECK(NP * batch <= 65535, "(nprimes + 2) * batch exceeds the launch z-extent");
  TORCH_CHECK(K <= 131071, "K bound for int32 modular GEMM");
  TORCH_CHECK(bits >= 16 && bits <= 53, "bits must be in [16, 53]");
  const at::cuda::CUDAGuard guard(A.device());
  run_crt(out, A, Bt, batch, M, K, N, primes_cpu, inv_cpu, mu_cpu, (int)bits, use_corr, exact,
          true, at::cuda::getCurrentCUDAStream(), nullptr);
}

void fp64_emu_mm_timed(torch::Tensor& out, torch::Tensor& times, torch::Tensor const& A,
                       torch::Tensor const& B, torch::Tensor const& primes_cpu,
                       torch::Tensor const& inv_cpu, torch::Tensor const& mu_cpu, bool exact,
                       int64_t bits, bool use_corr) {
  TORCH_CHECK(A.size(1) <= 131071, "K bound for int32 modular GEMM");
  TORCH_CHECK(bits >= 16 && bits <= 53, "bits must be in [16, 53]");
  const at::cuda::CUDAGuard guard(A.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaEvent_t e[5];
  for (int i = 0; i < 5; i++) cudaEventCreate(&e[i]);
  run_crt(out, A, B, 1, A.size(0), A.size(1), B.size(0), primes_cpu, inv_cpu, mu_cpu,
          (int)bits, use_corr, exact, false, stream, e);
  cudaEventSynchronize(e[4]);
  float* t = times.data_ptr<float>();
  for (int i = 0; i < 4; i++) cudaEventElapsedTime(&t[i], e[i], e[i + 1]);
  for (int i = 0; i < 5; i++) cudaEventDestroy(e[i]);
}
