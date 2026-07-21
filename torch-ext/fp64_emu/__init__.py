"""fp64-emu: emulated FP64 GEMM on integer tensor cores (Ozaki-II / CRT).

C = A @ B is recovered from products modulo pairwise-coprime moduli (one INT8
tensor-core GEMM each), combined by balanced-Garner reconstruction with a
first-order quantization correction. Accuracy meets or exceeds native FP64
DGEMM; exactly representable inputs reproduce bit for bit, and a runtime
Cauchy-Schwarz pass keeps any modulus count range-safe.
"""
import math
from functools import lru_cache
from typing import List, Optional

import torch

from ._ops import ops
from . import _meta  # noqa: F401  (fake kernels for torch.compile)

# Pairwise-coprime moduli to the balanced-residue int8 ceiling of 256, largest
# first; composites are admissible. The set matches GEMMul8 (RIKEN-RCCS, MIT).
_PRIMES = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199,
           197, 193, 191, 181, 179, 173]
_MAXP = 20   # mirrors MAXP in fp64_emu.cu


def _modinv(a, m):
    a %= m
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    raise ValueError


@lru_cache(maxsize=None)
def _tables(nprimes):
    primes = _PRIMES[:nprimes]
    P = len(primes)
    inv = [0] * (P * P)
    for a in range(P):
        for b in range(P):
            if a < b:
                inv[a * P + b] = _modinv(primes[a] % primes[b], primes[b])
    mu = [(1 << 62) // p for p in primes]        # Barrett reciprocals, BSH=62
    return (torch.tensor(primes, dtype=torch.int32),
            torch.tensor(inv, dtype=torch.int32),
            torch.tensor(mu, dtype=torch.int64))


def mm(A: torch.Tensor, B: torch.Tensor, nprimes: int = None,
       exact: bool = True, bits: int = 53, corr: bool = None) -> torch.Tensor:
    """Emulated FP64 C = A @ B, with B given as [N, K].

    nprimes: modulus count, one INT8 GEMM each; default sized by plan_config.
    exact: int128 recombine (correctly rounded) vs fp64 recombine (~52 bits).
    bits: mantissa width kept per operand; 53 keeps the full significand.
    corr: the two cross GEMMs cancelling first-order quantization error.
    """
    assert A.dim() == 2 and B.dim() == 2 and A.size(1) == B.size(1)
    M, K = A.shape
    N = B.size(0)
    if nprimes is None or corr is None:
        dp, dc = plan_config(K)
        nprimes = dp if nprimes is None else nprimes
        corr = dc if corr is None else corr
    primes, inv, mu = _tables(nprimes)
    out = torch.empty(M, N, dtype=torch.float64, device=A.device)
    # Skip the contiguous() dispatch when it is a no-op.
    Ac = A if A.is_contiguous() else A.contiguous()
    Bc = B if B.is_contiguous() else B.contiguous()
    ops.fp64_emu_mm(out, Ac, Bc, primes, inv, mu, exact, bits, corr)
    return out


def mm_nt(A: torch.Tensor, Bt: torch.Tensor, nprimes: int = None,
          exact: bool = True, bits: int = 53, corr: bool = None) -> torch.Tensor:
    """C = A @ Bt with Bt given as [K, N] (standard matmul layout), consumed
    in place: no transpose copy on the direct path."""
    assert A.dim() == 2 and Bt.dim() == 2 and A.size(1) == Bt.size(0)
    K = A.size(1)
    if K > 131071:   # the in-place layout is single-launch; deep K goes via mm
        return mm(A, Bt.t().contiguous(), nprimes, exact, bits, corr)
    if nprimes is None or corr is None:
        dp, dc = plan_config(K)
        nprimes = dp if nprimes is None else nprimes
        corr = dc if corr is None else corr
    primes, inv, mu = _tables(nprimes)
    out = torch.empty(A.size(0), Bt.size(1), dtype=torch.float64, device=A.device)
    Ac = A if A.is_contiguous() else A.contiguous()
    Bc = Bt if Bt.is_contiguous() else Bt.contiguous()
    ops.fp64_emu_mm_nt(out, Ac, Bc, primes, inv, mu, exact, bits, corr)
    return out


def bmm(A: torch.Tensor, B: torch.Tensor, nprimes: int = None,
        exact: bool = True, bits: int = 53, corr: bool = None) -> torch.Tensor:
    """Batched emulated FP64 C[b] = A[b] @ B[b], with B given as [batch, N, K].

    One operand width serves the whole batch, sized by its widest member; the
    per-member guarantees are otherwise those of `mm`. All GEMMs of the batch
    run in one launch, so per-call overhead is paid once, not per member.
    """
    assert A.dim() == 3 and B.dim() == 3 and A.size(0) == B.size(0) and A.size(2) == B.size(2)
    K = A.size(2)
    if nprimes is None or corr is None:
        dp, dc = plan_config(K)
        nprimes = dp if nprimes is None else nprimes
        corr = dc if corr is None else corr
    primes, inv, mu = _tables(nprimes)
    out = torch.empty(A.size(0), A.size(1), B.size(1), dtype=torch.float64, device=A.device)
    Ac = A if A.is_contiguous() else A.contiguous()
    Bc = B if B.is_contiguous() else B.contiguous()
    ops.fp64_emu_bmm(out, Ac, Bc, primes, inv, mu, exact, bits, corr)
    return out


def bmm_nt(A: torch.Tensor, Bt: torch.Tensor, nprimes: int = None,
           exact: bool = True, bits: int = 53, corr: bool = None) -> torch.Tensor:
    """C[b] = A[b] @ Bt[b] with Bt given as [batch, K, N] (torch.bmm layout),
    consumed in place: no transpose copy on the direct path."""
    assert A.dim() == 3 and Bt.dim() == 3 and A.size(0) == Bt.size(0) and A.size(2) == Bt.size(1)
    K = A.size(2)
    if K > 131071:
        return bmm(A, Bt.transpose(1, 2).contiguous(), nprimes, exact, bits, corr)
    if nprimes is None or corr is None:
        dp, dc = plan_config(K)
        nprimes = dp if nprimes is None else nprimes
        corr = dc if corr is None else corr
    primes, inv, mu = _tables(nprimes)
    out = torch.empty(A.size(0), A.size(1), Bt.size(2), dtype=torch.float64, device=A.device)
    Ac = A if A.is_contiguous() else A.contiguous()
    Bc = Bt if Bt.is_contiguous() else Bt.contiguous()
    ops.fp64_emu_bmm_nt(out, Ac, Bc, primes, inv, mu, exact, bits, corr)
    return out


def mm_timed(A: torch.Tensor, B: torch.Tensor, nprimes: int = None, exact: bool = True,
             bits: int = 53, corr: bool = None):
    """Returns (C, phase_ms[4]) = [scale+extract, gemms, marker, reconstruct]."""
    M, K = A.shape
    N = B.size(0)
    if nprimes is None or corr is None:
        dp, dc = plan_config(K)
        nprimes = dp if nprimes is None else nprimes
        corr = dc if corr is None else corr
    primes, inv, mu = _tables(nprimes)
    out = torch.empty(M, N, dtype=torch.float64, device=A.device)
    times = torch.zeros(4, dtype=torch.float32)
    ops.fp64_emu_mm_timed(out, times, A.contiguous(), B.contiguous(), primes, inv, mu,
                          exact, bits, corr)
    return out, times.tolist()


def range_bits(nprimes: int) -> float:
    """Signed CRT range of the first `nprimes` primes, in bits."""
    r = 0.0
    for p in _PRIMES[:nprimes]:
        r += math.log2(p)
    return r - 1.0


def plan_config(K: int) -> tuple:
    """(nprimes, corr) reaching FP64-equivalent accuracy at depth K; the
    kernel's runtime Cauchy-Schwarz pass keeps the choice range-safe."""
    need = 2 * 49 + math.ceil(math.log2(max(K, 1))) + 1 - 3.4
    for n in range(1, _MAXP + 1):
        if range_bits(n) >= need:
            return n, True
    return _MAXP, True


def plan(K: int, bits: int = 53) -> int:
    """Smallest worst-case range-safe nprimes at depth K and width bits: the
    exact product needs 2*bits + ceil(log2 K) + 1 bits."""
    need = 2 * bits + math.ceil(math.log2(max(K, 1))) + 1
    for n in range(1, _MAXP + 1):
        if range_bits(n) >= need:
            return n
    raise ValueError(f"K={K} at bits={bits} needs more range than {_MAXP} primes give")


__all__ = ["mm", "mm_nt", "bmm", "bmm_nt", "mm_timed", "plan", "range_bits"]
