---
library_name: kernels
license: apache-2.0
---

# fp64-emu

fp64-emu implements RARE, range-adaptive residue emulation: an FP64 matrix
multiply on integer tensor cores that extends the Ozaki-II residue (CRT)
scheme with runtime operand sizing, round-to-nearest extraction with
first-order correction, and register-resident balanced-Garner
reconstruction. It is loadable through `kernels` and requires compute
capability 8.0 or higher and is measured here on five architectures: 8.0,
8.6, 8.9, 9.0 and 12.0. On parts where FP64 runs at 1/64 of the FP32 rate it reaches
26.1x the native FP64 rate at `n = 8192` on RTX PRO 6000 Blackwell, at accuracy
exceeding native FP64 DGEMM in every distribution tested; on exactly
representable inputs the product is reproduced bit for bit.

Most consumer and workstation GPUs run double precision at 1/64 of their
single-precision rate, while their INT8 tensor cores are the fastest units on
the die. This kernel computes a double-precision matrix product on those INT8
units instead: each operand is scaled to integers, multiplied modulo a set of
small moduli (one INT8 GEMM per modulus), and the exact product is
reconstructed from the residues by the Chinese Remainder Theorem. The result is
a drop-in float64 `mm` that is an order of magnitude faster than native FP64
wherever FP64 is throttled, and at least as accurate everywhere.

![Residue planes stacking into a tower while the error field burns down and a correct-bits meter passes the native FP64 mark](https://huggingface.co/kernels/phanerozoic/fp64-emu/resolve/main/media/hero.gif)

*One INT8 GEMM per modulus: each residue plane lands, the per-element error
against a ~106-bit double-double reference burns down, and the correct-bits
meter climbs. The 14 residue planes reach 48.4 bits; the two orange
quantization-correction planes carry the result to 53.7, past the 50.8 bits
native FP64 DGEMM measures on the same instance (`M = N = 128`, `K = 4096`,
randn, on the Ada part; native DGEMM accuracy is itself part-dependent, and
the same instance measures 47.6 bits on Blackwell).*

## Usage

```python
import torch
from kernels import get_kernel

emu = get_kernel("phanerozoic/fp64-emu", version=1, trust_remote_code=True)

A  = torch.randn(4096, 4096, dtype=torch.float64, device="cuda")
Bt = torch.randn(4096, 4096, dtype=torch.float64, device="cuda")

C = emu.mm_nt(A, Bt)                           # full FP64 accuracy, self-sizing
C = emu.mm_nt(A, Bt, nprimes=13, corr=False)   # 13 GEMMs, ~45 bits
```

The default sizes itself from `K` and requires no configuration for full
accuracy at any supported depth. `version` selects the release branch;
`trust_remote_code` is required by `kernels` for publishers without the
trusted-publisher mark. `mm(A, B)` expects `B` as `[N, K]`; `mm_nt(A, Bt)`
takes the standard `[K, N]` layout.

## API

| Symbol | Purpose |
|---|---|
| `mm_nt(A, Bt, nprimes, exact, bits, corr)` | emulated FP64 product, `Bt` consumed in its standard `[K, N]` layout, no transpose copy |
| `mm(A, B, nprimes, exact, bits, corr)` | emulated FP64 product, `B` given as `[N, K]` |
| `bmm_nt(A, Bt, ...)` / `bmm(A, B, ...)` | batched product over uniform `[batch, ...]` operands, one launch chain for the whole batch |
| `plan_config(K)` | `(nprimes, corr)` reaching FP64-equivalent accuracy at depth `K` |
| `mm_timed(...)` | product plus per-phase timings |

## Method

`C = A @ B` is recovered from products modulo pairwise-coprime moduli. Each row
of A and column of B is scaled by a power of two so its largest element fills
an exactly representable integer, leaving a quantization residual of at most
half a unit. For each modulus the balanced residues fit INT8 and one
tensor-core GEMM gives `C mod m`. Balanced-Garner reconstruction combines the
planes, adds the quantization correction, and rescales. The modulus count is
linear in the target precision (Ozaki-II / CRT), against the quadratic slice
count of mantissa-slicing schemes. Four properties set the cost:

- Balanced residues in `[-(m-1)/2, m/2]` admit moduli to 256, so the
  composites 255, 253, 247 and 217 join the primes at nearly 8 bits each,
  worth about a fifth fewer GEMMs than primes alone.
- Operands are sized by the Cauchy-Schwarz bound
  `max_i‖A_i‖₂ · max_j‖B_j‖₂` rather than the worst case. Unused CRT range
  becomes mantissa bits, and adversarial input narrows the result rather than
  corrupting it.
- Two additional INT8 GEMMs recover the first-order quantization cross terms
  `Σ(Ã·rB + rA·B̃)`, lifting accuracy above the ~51.5-bit ceiling of operand
  quantization alone; they vanish identically on integer inputs.
- Every modular reduction is a 32-bit Barrett; operand extraction splits each
  scaled integer into three 18-bit limbs, two multiplies and one reduction per
  modulus.

The modular GEMMs run on two engines. The tensor-core kernel in this
repository computes 64x64 products per warp on `mma.sync.m16n8k32` over an
mbarrier-pipelined tile, TMA-staged on `sm_90+` and `cp.async`-staged below;
in the `cp.async` form the extraction writes each operand plane tile-blocked
in the order the kernel consumes it, so staging is contiguous streaming reads,
with ragged edges zero-padded. Its epilogue Barrett-reduces every product to
its balanced INT8 residue and stores fragment pairs directly, one byte per
element into reconstruction with no staging barriers. The other engine is
cuBLAS strided-batched INT8 GEMMs into raw int32 planes. Which engine is
faster depends on architecture and size, so the dispatch times both once per
shape and caches the choice, as it does for the strided-batched call against a
per-plane loop inside the cuBLAS engine. When the plane set exceeds free
memory the planes stream in chunks; configurations that skip the correction
GEMMs skip the correction operands' extraction as well. `M`, `N` and `K` carry
no alignment constraints.

## Operating points

`nprimes` is the modulus count and costs one INT8 GEMM each; `corr` adds the
two correction GEMMs. The default selects the smallest pair reaching
FP64-equivalent accuracy at the given `K`: 16 GEMMs through `K = 8192`, 17
beyond.

| configuration | GEMMs | correct bits |
|---|---|---|
| default (`plan_config`) | 16 | 52.6 – 54.1 |
| `nprimes=13, corr=False` | 13 | 42.9 – 48.5 |
| `exact=False` | as above | ~51–52, ~2% faster |

## Throughput

Square `n`, random normal operands, median of timed executions, planning
excluded. Native FP64 is `torch.matmul` on float64. The reduced configuration
runs one modulus fewer than the default with the correction GEMMs off.

RTX PRO 6000 Blackwell (Server Edition), SM 12.0:

| n | native | default | vs native | reduced | vs native |
|---|---|---|---|---|---|
| 256 | 0.41 TF | 0.60 TF | 1.5x | 0.62 TF | 1.5x |
| 1024 | 1.37 TF | 11.40 TF | 8.3x | 12.44 TF | 9.1x |
| 2048 | 1.51 TF | 21.76 TF | 14.4x | 25.69 TF | 17.0x |
| 4096 | 1.52 TF | 32.40 TF | 21.3x | 38.55 TF | 25.4x |
| 8192 | 1.53 TF | 39.89 TF | 26.1x | 47.91 TF | 31.3x |
| 16384 | 1.53 TF | 39.95 TF | 26.0x | 48.01 TF | 31.3x |

Across architectures at `n = 8192`:

| part | SM | native | default | vs native | reduced | vs native |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell (Server) | 12.0 | 1.53 TF | 39.89 TF | 26.1x | 47.91 TF | 31.3x |
| RTX 3070 Ti Laptop | 8.6 | 0.28 TF | 4.45 TF | 15.8x | 5.25 TF | 18.7x |
| RTX 6000 Ada | 8.9 | 1.31 TF | 14.92 TF | 11.4x | 18.66 TF | 14.2x |
| A100-SXM4-80GB | 8.0 | 17.58 TF | 19.91 TF | 1.13x | 24.13 TF | 1.37x |
| H200 | 9.0 | 58.18 TF | 59.50 TF | 1.02x | 69.53 TF | 1.20x |

The ratio to native scales with a part's INT8-to-FP64 throughput ratio. On
A100 and H200, whose FP64 units are unthrottled, the emulation is slower
below `n = 8192` and modestly faster at `n = 8192` and above; below
`n ≈ 128` per-call cost exceeds native FP64 on every part measured.

## Batched

`bmm` carries a uniform batch through one launch chain: scaling, extraction,
a single modular-GEMM launch over every plane-batch slot, and one
reconstruction, so per-call cost is paid once rather than per member. One
operand width serves the whole batch, sized by its widest member; per-member
accuracy guarantees are otherwise those of `mm`. Measured against native
`torch.bmm` FP64 and against looping `mm` per member.

RTX PRO 6000 Blackwell (Server Edition):

| batch x n | native bmm | emu bmm | vs native | vs looped mm |
|---|---|---|---|---|
| 64 x 256 | 1.55 TF | 2.91 TF | 1.9x | 4.5x |
| 32 x 512 | 1.57 TF | 7.00 TF | 4.4x | 1.6x |
| 16 x 1024 | 1.58 TF | 13.92 TF | 8.8x | 1.1x |
| 8 x 2048 | 1.58 TF | 23.12 TF | 14.6x | 1.1x |
| 4 x 4096 | 1.59 TF | 32.71 TF | 20.5x | 1.1x |

RTX 6000 Ada:

| batch x n | native bmm | emu bmm | vs native | vs looped mm |
|---|---|---|---|---|
| 64 x 256 | 1.17 TF | 1.93 TF | 1.6x | 3.1x |
| 32 x 512 | 1.18 TF | 4.28 TF | 3.6x | 1.4x |
| 16 x 1024 | 1.30 TF | 8.04 TF | 6.2x | 1.2x |
| 8 x 2048 | 1.28 TF | 9.56 TF | 7.5x | 1.0x |
| 4 x 4096 | 1.31 TF | 12.61 TF | 9.7x | 1.0x |

Consumer Ampere measures 2.2x to 9.9x native over the same grid. On A100
and H200 the loop amortization holds (3.1x and 4.0x over looped `mm` at
64 x 256) while native `torch.bmm` keeps the lead, as in the single-GEMM
tables. The loop advantage concentrates below `n = 1024`, where per-call
cost dominates a single small product.

## Accuracy

Correct bits against a double-double reference accumulated at ~106 bits,
cross-checked against exact rational arithmetic; `M = N = 128`, max-norm
relative error, measured on the Blackwell part. The other architectures agree
within measurement spread; the reconstruction is exact integer arithmetic and
carries no architecture dependence. Distributions: random normal; per-element
exponent jitter of `2^U(-8,8)` and `2^U(-16,16)`; shared column scaling
spanning `2^-20` and `2^-40`; a `2^30` spike element per row.

| dist | K | native FP64 | default | reduced |
|---|---|---|---|---|
| randn | 512 | 49.4 | 53.4 | 45.9 |
| randn | 4096 | 47.6 | 53.6 | 44.3 |
| wide8 | 512 | 49.4 | 53.7 | 46.2 |
| wide8 | 4096 | 47.6 | 53.7 | 45.1 |
| wide16 | 512 | 49.4 | 53.3 | 46.8 |
| wide16 | 4096 | 48.1 | 53.7 | 45.2 |
| illcond | 512 | 49.1 | 53.7 | 47.1 |
| illcond | 4096 | 47.7 | 53.3 | 45.4 |
| illcond40 | 512 | 49.6 | 53.6 | 48.6 |
| illcond40 | 4096 | 48.0 | 53.5 | 46.9 |
| spike | 512 | 48.7 | 54.0 | 45.7 |
| spike | 4096 | 47.8 | 53.2 | 43.3 |

The default sits at the output-rounding floor and above native in all twelve
cells: each inner product is formed exactly in the residue domain and rounded
once, where native DGEMM accumulates `K` roundings. On exactly representable
inputs the default reconstructs the product bit for bit, verified by
`torch.equal` against integer oracles: random integers, an all-ones case at
`K = 2048`, and a dyadic dynamic-range case spanning `2^-12..2^0` column
scales.

## Comparison with GEMMul8

GEMMul8 (Ozaki-II, RIKEN-RCCS, INT8) is built from source and measured on
the same hardware, scored against the same double-double reference. The
comparison holds the GEMM count equal at every size, with each system at
its smallest FP64-equivalent configuration: fp64-emu's default (14 moduli
plus 2 correction GEMMs, 15 plus 2 at `n = 16384`) runs against GEMMul8 at
16 moduli (17 at `n = 16384`). fp64-emu measures 53.1 - 53.9 correct bits
across the grid, and GEMMul8 measures 51.8 - 53.3.

GEMMul8 correct bits by modulus count, `K = 4096` (randn / spike):

| moduli | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|
| bits | 48.1 / 46.3 | 52.3 / 50.2 | 53.0 / 52.7 | 53.0 / 53.3 | 53.0 / 53.2 |

At 16 moduli its accuracy holds 52.2 - 53.3 bits through `K = 8192` and
51.8 on spike at `K = 16384`, where 17 restores 53.2.

GeForce RTX 3070 Ti (consumer Ampere, SM 8.6):

| n | fp64-emu | GEMMul8 | ratio |
|---|---|---|---|
| 512 | 0.94 TF | 0.64 TF | 1.47x |
| 1024 | 2.10 TF | 1.43 TF | 1.47x |
| 2048 | 3.58 TF | 2.27 TF | 1.58x |
| 4096 | 3.65 TF | 3.14 TF | 1.16x |
| 8192 | 4.45 TF | 3.80 TF | 1.17x |

RTX 6000 Ada (SM 8.9):

| n | fp64-emu | GEMMul8 | ratio |
|---|---|---|---|
| 512 | 2.85 TF | 0.75 TF | 3.80x |
| 1024 | 8.28 TF | 4.40 TF | 1.88x |
| 2048 | 12.71 TF | 9.46 TF | 1.34x |
| 4096 | 13.56 TF | 12.20 TF | 1.11x |
| 8192 | 14.92 TF | 13.78 TF | 1.08x |
| 16384 | 14.63 TF | 13.72 TF | 1.07x |

RTX PRO 6000 Blackwell (Server Edition, SM 12.0):

| n | fp64-emu | GEMMul8 | ratio |
|---|---|---|---|
| 2048 | 21.76 TF | 13.14 TF | 1.66x |
| 4096 | 32.40 TF | 20.65 TF | 1.57x |
| 8192 | 39.89 TF | 30.78 TF | 1.30x |
| 16384 | 39.95 TF | 34.88 TF | 1.15x |

H200 (Hopper, SM 9.0), where native FP64 is unthrottled and both systems
trail it below `n = 8192`:

| n | fp64-emu | GEMMul8 | ratio |
|---|---|---|---|
| 2048 | 22.63 TF | 19.95 TF | 1.13x |
| 4096 | 43.68 TF | 44.86 TF | 0.97x |
| 8192 | 59.50 TF | 58.61 TF | 1.02x |
| 16384 | 57.67 TF | 66.53 TF | 0.87x |

fp64-emu leads at every size on the parts with throttled FP64; on Hopper
GEMMul8 leads at the largest sizes.

At reduced precision the comparison sets fp64-emu's 13-GEMM configuration
(44.3 / 43.3 correct bits) against GEMMul8's fast mode at 13 moduli
(43.9 / 44.6). The ratio of fp64-emu to GEMMul8 throughput:

| n | RTX 3070 Ti | RTX 6000 Ada | Blackwell (Server) | H200 |
|---|---|---|---|---|
| 512 | 1.17x | 2.92x | - | - |
| 1024 | 1.17x | 2.27x | - | - |
| 2048 | 1.18x | 1.20x | 1.38x | 0.97x |
| 4096 | 1.03x | 1.23x | 1.34x | 0.83x |
| 8192 | 1.05x | 1.18x | 1.21x | 0.88x |
| 16384 | - | 0.94x | 1.01x | 0.74x |

At `n = 16384` the range rule moves fp64-emu's reduced configuration to 14
GEMMs against GEMMul8's 13, and GEMMul8 leads that row in proportion to the
count; pinning `nprimes=13` at that depth measures 17.28 against 17.60 TF on
the Ada, parity at equal counts.

## NVIDIA's cuBLAS FP64 emulation

cuBLAS gained FP64 emulation on Blackwell-class parts in CUDA 13.0
Update 2. With cuBLAS 13.6 on consumer Ampere and on Ada, enabling the
emulation strategy leaves both output and timing bit-identical to native
FP64: the feature does not engage below Blackwell. On the RTX PRO 6000
Blackwell, NVIDIA reports up to a 13x speedup over native FP64; fp64-emu
measures 26.1x on the same part at accuracy above native.

## End to end

A tiled right-looking Cholesky factorization (`bench/cholesky.py`) routes
its trailing updates through `bmm` while the panel factorization and
triangular solves stay native in both configurations. On the RTX 6000 Ada
at `n = 16384`, the update phase drops from 1.23 s to 0.69 s and the total
from 1.43 s to 0.94 s, a 1.5x end-to-end speedup; consumer Ampere at
`n = 8192` moves from 0.83 s to 0.56 s. The factorization residuals are
equal between the two engines at 3 - 4 x 10^-16.

## Requirements and limits

- NVIDIA GPU with compute capability 8.0 or higher, verified on 8.0, 8.6,
  8.9, 9.0 and 12.0. The modular GEMM uses `mma.sync.aligned.m16n8k32`,
  `cp.async` and `mbarrier`, which are unavailable below sm_80.
- float64 inputs. A single launch covers `K < 131072`, set by the int32
  accumulator: a balanced residue modulo 256 reaches -128, so the worst-case
  plane accumulates `K·128²`. Deeper products run K-segmented with residue
  accumulation, to `K = 128 · 65536` (8.4M); depth trades against the
  126-bit recombine range through the runtime operand sizing.
- The ops register fake kernels for `torch.compile`, and the steady-state
  call is CUDA-graph capturable; graph replay removes the host overhead that
  dominates below `n ≈ 512`.
- `M`, `N` and `K` carry no alignment constraints. Batched calls require
  uniform member shapes and `(nprimes + 2) * batch <= 65535`.
- Published variants are Linux x86_64. On Windows, or for a torch build
  without a matching variant, `load_local.py` in the repository JIT-builds
  the same source and exposes the identical API.

## Repository and reproduction

Source home: [github.com/CharlesCNorton/fp64-emu](https://github.com/CharlesCNorton/fp64-emu),
whose `bench/` directory carries the measurement harness behind every table
here: the double-double accuracy reference, the card and phase benchmarks,
the batched and small-n drills, and the GEMMul8 side-by-side drivers. The
manuscript describing RARE is in `paper/`. Compiled variants are
distributed from this kernel repository.

## Citation

A manuscript describing RARE is under submission; until it appears, cite
this repository:

```bibtex
@misc{fp64emu2026,
  author = {Norton, Charles C.},
  title  = {fp64-emu: range-adaptive residue emulation of FP64 GEMM
            on integer tensor cores},
  year   = {2026},
  url    = {https://huggingface.co/kernels/phanerozoic/fp64-emu}
}
```

## References

Ozaki, Ogita, Oishi, Rump 2012 (Numerical Algorithms 59); Ozaki, Uchino,
Imamura 2025 (Ozaki Scheme II, arXiv:2504.08009; GEMMul8, RIKEN-RCCS); Ootomo,
Ozaki, Yokota 2024 (ozIMMU, enp1s0/ozIMMU); NVIDIA, floating-point emulation
in cuBLAS (Developer Blog, 2025).

## License

Apache-2.0.
