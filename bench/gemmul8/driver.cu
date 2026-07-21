// GEMMul8 timing driver: direct calls into gemmul8::gemm, no hook. Prints
// median ms and TFLOP/s for the requested modulus count and mode.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gemmul8.hpp"
#ifdef _MSC_VER
// The public workSize explicit instantiations are dropped by the MSVC build
// (their template arguments name anonymous-namespace constexpr variables), so
// the implementation header is compiled into this TU instead.
#include "worksize/worksize_impl.hpp"
#endif

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "cuda %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cublas %s:%d %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)

__global__ void fill_kernel(double* p, size_t n, unsigned long long seed) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long x = seed + i * 2654435761ULL;
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
  p[i] = ((double)(x >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: driver <n> <moduli> <fast 0|1>\n"); return 2; }
  const int n = atoi(argv[1]);
  const unsigned moduli = (unsigned)atoi(argv[2]);
  const bool fast = atoi(argv[3]) != 0;
  const size_t N = (size_t)n * n;

  double *A, *B, *C;
  CK(cudaMalloc(&A, N * 8)); CK(cudaMalloc(&B, N * 8)); CK(cudaMalloc(&C, N * 8));
  fill_kernel<<<(unsigned)((N + 255) / 256), 256>>>(A, N, 1ULL);
  fill_kernel<<<(unsigned)((N + 255) / 256), 256>>>(B, N, 2ULL);
  CK(cudaMemset(C, 0, N * 8));
  CK(cudaDeviceSynchronize());

  cublasHandle_t h; CB(cublasCreate(&h));
  const double alpha = 1.0, beta = 0.0;

  const size_t lwork = gemmul8::workSize<false>(n, n, n, (int)moduli);
  void* work = nullptr;
  CK(cudaMalloc(&work, lwork));

  auto once = [&] {
    gemmul8::gemm<double>(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n,
                          &beta, C, n, moduli, fast, work);
  };

  for (int i = 0; i < 3; i++) once();
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> ms;
  for (int i = 0; i < 10; i++) {
    CK(cudaEventRecord(e0)); once(); CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float t = 0; CK(cudaEventElapsedTime(&t, e0, e1));
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  const double med = ms[ms.size() / 2];
  printf("n=%d moduli=%u fast=%d %.3f ms %.2f TF\n", n, moduli, (int)fast, med,
         2.0 * (double)n * n * n / (med * 1e-3) / 1e12);
  return 0;
}
