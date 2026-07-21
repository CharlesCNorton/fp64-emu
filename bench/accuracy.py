"""Correct-bits measurement for fp64-emu against a double-double reference.

The reference accumulates each inner product in ~106-bit double-double
arithmetic on the GPU, so it sits ~50 bits below anything being measured. It is
cross-checked once against an exact Python-integer product, which is independent
of every floating-point path under test.
"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402

_SPLIT = float(2 ** 27 + 1)


def _two_prod(a, b):
    """Dekker: exact a*b as an unevaluated sum p + e (no FMA needed)."""
    p = a * b
    ca, cb = _SPLIT * a, _SPLIT * b
    ah = ca - (ca - a)
    al = a - ah
    bh = cb - (cb - b)
    bl = b - bh
    e = (((ah * bh - p) + ah * bl) + al * bh) + al * bl
    return p, e


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def dd_matmul_nt(A, Bt, chunk=1):
    """A [M,K] @ Bt [K,N] accumulated in double-double; returns (hi, lo)."""
    M, K = A.shape
    N = Bt.shape[1]
    hi = torch.zeros(M, N, dtype=torch.float64, device=A.device)
    lo = torch.zeros_like(hi)
    for k in range(K):
        p, e = _two_prod(A[:, k:k + 1], Bt[k:k + 1, :])
        hi, s = _two_sum(hi, p)
        lo = lo + (s + e)
        hi, lo = _two_sum(hi, lo)
    return hi, lo


def dd_error_vs_exact(A, Bt, hi, lo):
    """Max-norm relative error of the dd value hi+lo against the exact product.

    Floats are exact rationals, so Fraction arithmetic gives the true product
    and the true dd residual, with no floating-point path in the comparison.
    """
    from fractions import Fraction
    a, bt = A.cpu().tolist(), Bt.cpu().tolist()
    h, l = hi.cpu().tolist(), lo.cpu().tolist()
    K = len(a[0])
    worst = Fraction(0)
    scale = Fraction(0)
    for i in range(len(a)):
        arow = [Fraction(x) for x in a[i]]
        for j in range(len(bt[0])):
            ex = sum(arow[k] * Fraction(bt[k][j]) for k in range(K))
            d = abs(ex - (Fraction(h[i][j]) + Fraction(l[i][j])))
            worst = max(worst, d)
            scale = max(scale, abs(ex))
    return float(worst / scale) if scale else 0.0


def correct_bits(C, hi, lo):
    """Correct bits of C against the dd value hi+lo, max-norm relative."""
    err = ((C - hi) - lo).abs().max().item()
    scale = hi.abs().max().item()
    if err == 0.0:
        return 53.0, 0.0
    rel = err / scale
    return -math.log2(rel), rel


def make(dist, M, K, N, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float64, device="cuda", generator=g)
    Bt = torch.randn(K, N, dtype=torch.float64, device="cuda", generator=g)
    if dist == "randn":
        pass
    elif dist in ("wide8", "wide16"):
        w = 8 if dist == "wide8" else 16
        A *= torch.exp2(torch.randint(-w, w + 1, A.shape, device="cuda", generator=g).double())
        Bt *= torch.exp2(torch.randint(-w, w + 1, Bt.shape, device="cuda", generator=g).double())
    elif dist in ("illcond", "illcond40"):
        span = 20 if dist == "illcond" else 40
        s = torch.exp2(-span * torch.rand(K, device="cuda", generator=g, dtype=torch.float64))
        A *= s[None, :]
        Bt *= s[:, None]
    elif dist == "spike":
        idx = torch.randint(0, K, (M,), device="cuda", generator=g)
        A[torch.arange(M, device="cuda"), idx] *= 2.0 ** 30
    else:
        raise ValueError(dist)
    return A, Bt


DISTS = ["randn", "wide8", "wide16", "illcond", "illcond40", "spike"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dists", nargs="+", default=DISTS)
    ap.add_argument("--K", type=int, nargs="+", default=[512, 4096])
    ap.add_argument("--bits", type=int, nargs="+",
                    default=[53, 50, 47, 44, 42, 40, 38, 36, 32])
    ap.add_argument("--mn", type=int, default=128)
    ap.add_argument("--validate", action="store_true",
                    help="cross-check the dd reference against exact integers")
    args = ap.parse_args()

    emu = jit.build()

    if args.validate:
        for dist in ("wide8", "illcond40"):
            A, Bt = make(dist, 8, 128, 8, seed=7)
            hi, lo = dd_matmul_nt(A, Bt)
            d = dd_error_vs_exact(A, Bt, hi, lo)
            nb, _ = correct_bits(A @ Bt, hi, lo)
            print(f"dd reference vs exact rationals [{dist}]: rel {d:.3e} "
                  f"= {-math.log2(d):.1f} bits   (native fp64 here: {nb:.1f} bits)")
        print()

    M = N = args.mn
    hdr = f"{'dist':>10} {'K':>6} {'bits':>5} {'nprimes':>8} {'gemms':>6} {'correct bits':>13} {'native':>7}"
    print(hdr)
    print("-" * len(hdr))
    for dist in args.dists:
        for K in args.K:
            A, Bt = make(dist, M, K, N, seed=hash((dist, K)) & 0xFFFF)
            hi, lo = dd_matmul_nt(A, Bt)
            nb, _ = correct_bits(A @ Bt, hi, lo)
            B = Bt.t().contiguous()
            for b in args.bits:
                try:
                    p = emu.plan(K, b)
                except ValueError:
                    continue
                C = emu.mm(A, B, nprimes=p, exact=True, bits=b)
                cb, _ = correct_bits(C, hi, lo)
                print(f"{dist:>10} {K:>6} {b:>5} {p:>8} {p+2:>6} {cb:>13.1f} {nb:>7.1f}")
            print()


if __name__ == "__main__":
    main()
