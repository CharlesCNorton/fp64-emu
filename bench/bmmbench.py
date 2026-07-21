"""Batched throughput: emu.bmm vs a looped emu.mm vs native torch.bmm."""
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402

emu = jit.build()


def timed(fn, iters=8, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
print(f"{'batch x n':>12} {'native bmm':>11} {'emu bmm':>9} {'x nat':>6} "
      f"{'looped mm':>10} {'bmm/loop':>9}")
cfgs = [(64, 256), (32, 512), (16, 1024), (8, 2048), (4, 4096)]
for B, n in [tuple(int(x) for x in a.split("x")) for a in sys.argv[1:]] or cfgs:
    A = torch.randn(B, n, n, dtype=torch.float64, device="cuda")
    Bt = torch.randn(B, n, n, dtype=torch.float64, device="cuda")
    Bm = Bt.transpose(1, 2).contiguous()
    flop = 2.0 * B * n**3
    tn = timed(lambda: torch.bmm(A, Bt))
    tb = timed(lambda: emu.bmm(A, Bm))
    tl = timed(lambda: torch.stack([emu.mm(A[b], Bm[b]) for b in range(B)]))
    print(f"{B:>5} x {n:<5} {flop/tn/1e12:>9.2f} TF {flop/tb/1e12:>6.2f} TF "
          f"{tn/tb:>6.1f} {flop/tl/1e12:>7.2f} TF {tl/tb:>8.2f}x", flush=True)
    del A, Bt, Bm
    torch.cuda.empty_cache()
