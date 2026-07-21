"""Default-config phase split and per-path end-to-end timing across sizes.

Attributes runtime to extract / GEMMs / reconstruct and reports the GEMM
phase's effective INT8 rate against the device ceiling.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402

emu = jit.build()


def timed(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


print(torch.cuda.get_device_name(0))
sizes = [int(x) for x in sys.argv[1:]] or [512, 1024, 2048, 4096, 8192, 16384]
for n in sizes:
    A = torch.randn(n, n, dtype=torch.float64, device="cuda")
    B = torch.randn(n, n, dtype=torch.float64, device="cuda")
    P, corr = emu.plan_config(n)
    g = P + (2 if corr else 0)
    flop = 2.0 * n ** 3
    parts = [f"n={n:<6} P={P}+{2 if corr else 0}"]
    for path in ("auto", "fused", "cublas", "chunked"):
        if path == "auto":
            os.environ.pop("FP64EMU_PATH", None)
        else:
            os.environ["FP64EMU_PATH"] = path
        try:
            t = timed(lambda: emu.mm(A, B, nprimes=P, corr=corr))
            parts.append(f"{path} {t:8.2f} ms {flop/t/1e9:6.2f} TF")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as ex:
            parts.append(f"{path} FAIL({type(ex).__name__})")
            torch.cuda.empty_cache()
    os.environ.pop("FP64EMU_PATH", None)
    print(" | ".join(parts))

    best = None
    for _ in range(5):
        _, ph = emu.mm_timed(A, B, nprimes=P, corr=corr)
        if best is None or sum(ph) < sum(best):
            best = ph
    ext, gm, _, rec = best
    tot = max(ext + gm + rec, 1e-9)
    print(f"    phases: extract {ext:7.2f} ({100*ext/tot:4.1f}%) | gemms {gm:7.2f} "
          f"({100*gm/tot:4.1f}%) | recon {rec:7.2f} ({100*rec/tot:4.1f}%) | "
          f"gemm-phase INT8 {g*flop/gm/1e9:5.0f} TOPS")
    del A, B
    torch.cuda.empty_cache()
