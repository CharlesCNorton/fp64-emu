"""Energy per GEMM: mean board power sampled by nvidia-smi during a
sustained loop, reported as joules per product and GFLOP per joule."""
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
if (Path(__file__).resolve().parent / "jit.py").exists():
    from jit import build
    emu = build()
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import load_local
    emu = load_local.load()


def _sample(stop, out):
    while not stop.is_set():
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        try:
            out.append(float(r.stdout.strip().splitlines()[0]))
        except (ValueError, IndexError):
            pass
        time.sleep(0.1)


def measure(fn, seconds=8.0):
    fn()
    torch.cuda.synchronize()
    stop, out = threading.Event(), []
    th = threading.Thread(target=_sample, args=(stop, out))
    th.start()
    t0 = time.perf_counter()
    calls = 0
    while time.perf_counter() - t0 < seconds:
        fn()
        torch.cuda.synchronize()
        calls += 1
    el = time.perf_counter() - t0
    stop.set()
    th.join()
    p = statistics.mean(out[2:]) if len(out) > 4 else float("nan")
    return el / calls, p


n = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
A = torch.randn(n, n, dtype=torch.float64, device="cuda")
B = torch.randn(n, n, dtype=torch.float64, device="cuda")
flop = 2.0 * n**3
print(torch.cuda.get_device_name(0), f"| n={n}")
for name, fn in (("native", lambda: A @ B.t()), ("default", lambda: emu.mm(A, B))):
    t, p = measure(fn)
    print(f"{name:>8}: {t*1e3:8.2f} ms/GEMM  {p:6.1f} W  {p*t:7.2f} J/GEMM  "
          f"{flop/(p*t)/1e9:7.2f} GFLOP/J", flush=True)
