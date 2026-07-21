"""Card-protocol throughput: native, default (plan_config), reduced (one fewer
modulus, correction off), median of timed executions, planning excluded."""
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402


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


emu = jit.build()
print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
print(f"{'n':>6} {'config':>12} {'ms':>10} {'TFLOP/s':>9} {'x native':>9}")
for n in [int(x) for x in sys.argv[1:]] or (256, 1024, 2048, 4096, 8192, 16384):
    flop = 2.0 * n ** 3
    it = 3 if n >= 16384 else (30 if n <= 1024 else 8)
    A = torch.randn(n, n, dtype=torch.float64, device="cuda")
    Bt = torch.randn(n, n, dtype=torch.float64, device="cuda")
    B = Bt.t().contiguous()
    tn = timed(lambda: A @ Bt, max(3, it // 2))
    print(f"{n:>6} {'native':>12} {tn*1e3:>10.3f} {flop/tn/1e12:>9.2f} {1.0:>9.2f}")
    p, c = emu.plan_config(n)
    for cfg, pp, cc in (("default", p, c), ("reduced", p - 1, False)):
        t = timed(lambda: emu.mm(A, B, nprimes=pp, corr=cc), it)
        print(f"{n:>6} {cfg:>12} {t*1e3:>10.3f} {flop/t/1e12:>9.2f} {tn/t:>9.2f}",
              flush=True)
    del A, Bt, B
    torch.cuda.empty_cache()
