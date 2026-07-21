"""Head-to-head: fp64-emu vs whatever backs torch's FP64 matmul on this card.

Run once bare for the native-FP64 baseline, and once under
LD_PRELOAD=libgemmul8.so so that cublasDgemm is intercepted by GEMMul8. Both
modes are timed the same way and scored against the same double-double
reference, so the accuracy numbers are directly comparable.

GEMMul8's hook does not target strided-batched routines, so fp64-emu's own
cublasGemmStridedBatchedEx path runs unemulated either way.
"""
import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402
from accuracy import correct_bits, dd_matmul_nt, make  # noqa: E402


def timed(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["emu", "dgemm"], required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--bits", nargs="+", default=["53", "50:off", "47:off"],
                    help="configs as BITS or BITS:off to disable the correction GEMMs")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--acc-mn", type=int, default=128, help="size for the accuracy probe")
    ap.add_argument("--acc-K", type=int, default=4096)
    ap.add_argument("--dists", nargs="+", default=["randn", "spike"])
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    # "47:off" -> bits 47, no correction; "p14:on" -> 14 moduli, correction on.
    # The modulus form pins the GEMM count and lets the kernel choose the width.
    cfgs = []
    for spec in [str(s) for s in args.bits]:
        s = str(spec)
        head, _, c = s.partition(":")
        if head.startswith("p"):
            cfgs.append(("p", int(head[1:]), c != "off"))
        else:
            cfgs.append(("b", int(head), c != "off"))

    emu = jit.build() if args.mode == "emu" else None
    print(f"# device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"# mode: {args.mode} {args.label}")

    # ---- accuracy against the shared double-double reference ----
    print(f"\n{'dist':>10} {'K':>6} {'config':>14} {'correct bits':>13}")
    print("-" * 48)
    for dist in args.dists:
        A, Bt = make(dist, args.acc_mn, args.acc_K, args.acc_mn, seed=1234)
        hi, lo = dd_matmul_nt(A, Bt)
        if args.mode == "dgemm":
            cb, _ = correct_bits(A @ Bt, hi, lo)
            print(f"{dist:>10} {args.acc_K:>6} {'dgemm':>14} {cb:>13.1f}")
        else:
            B = Bt.t().contiguous()
            for kind, v, c in cfgs:
                p, b = (v, 53) if kind == "p" else (emu.plan(args.acc_K, v), v)
                cb, _ = correct_bits(emu.mm(A, B, nprimes=p, bits=b, corr=c), hi, lo)
                tag = f"b{b} p{p} c{'1' if c else '0'} g{p + (2 if c else 0)}"
                print(f"{dist:>10} {args.acc_K:>6} {tag:>14} {cb:>13.1f}")

    # ---- throughput ----
    print(f"\n{'n':>6} {'config':>14} {'ms':>9} {'TFLOP/s':>9}")
    print("-" * 42)
    for n in args.sizes:
        flop = 2.0 * n ** 3
        A = torch.randn(n, n, dtype=torch.float64, device="cuda")
        Bt = torch.randn(n, n, dtype=torch.float64, device="cuda")
        if args.mode == "dgemm":
            t = timed(lambda: A @ Bt, args.iters)
            print(f"{n:>6} {'dgemm':>14} {t*1e3:>9.2f} {flop/t/1e12:>9.2f}")
        else:
            B = Bt.t().contiguous()
            for kind, v, c in cfgs:
                p, b = (v, 53) if kind == "p" else (emu.plan(n, v), v)
                t = timed(lambda: emu.mm(A, B, nprimes=p, bits=b, corr=c), args.iters)
                tag = f"b{b} p{p} c{'1' if c else '0'} g{p + (2 if c else 0)}"
                print(f"{n:>6} {tag:>14} {t*1e3:>9.2f} {flop/t/1e12:>9.2f}")
            del B
        del A, Bt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
