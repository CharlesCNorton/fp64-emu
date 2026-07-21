"""Small-n overhead: eager mm vs CUDA-graph replay of the same call."""
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402

emu = jit.build()


def timed(fn, iters=200, warm=20):
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


print(torch.cuda.get_device_name(0))
print(f"{'n':>6} {'eager us':>9} {'graph us':>9} {'ratio':>6} {'native us':>10} {'graph ok':>9}")
for n in (64, 128, 256, 512):
    torch.manual_seed(n)
    A = torch.randn(n, n, dtype=torch.float64, device="cuda")
    B = torch.randn(n, n, dtype=torch.float64, device="cuda")
    te = timed(lambda: emu.mm(A, B))
    tn = timed(lambda: A @ B.t())
    for _ in range(3):
        emu.mm(A, B)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            C = emu.mm(A, B)
        g.replay()
        torch.cuda.synchronize()
        ok = torch.equal(C, emu.mm(A, B))
        tg = timed(g.replay)
        print(f"{n:>6} {te*1e6:>9.1f} {tg*1e6:>9.1f} {te/tg:>6.2f} {tn*1e6:>10.1f} {ok!s:>9}")
    except Exception as ex:  # noqa: BLE001
        print(f"{n:>6} {te*1e6:>9.1f} {'FAIL':>9} {type(ex).__name__}: {str(ex)[:90]}")
    del A, B
    torch.cuda.empty_cache()
