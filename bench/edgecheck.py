"""Edge-tile and odd-shape correctness for the direct-store epilogue."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402

emu = jit.build()
fails = 0
for M, K, N in [(129, 4095, 127), (17, 23, 29), (256, 511, 254), (255, 1000, 999),
                (1000, 1000, 1000), (128, 64, 256), (384, 8192, 512)]:
    a = torch.randn(M, K, dtype=torch.float64, device="cuda")
    b = torch.randn(N, K, dtype=torch.float64, device="cuda")
    r = a @ b.t()
    e = (emu.mm(a, b) - r).abs().max().item() / r.abs().max().item()
    ok = e < 1e-11
    fails += 0 if ok else 1
    print(f"{M}x{K}x{N}: rel {e:.2e} {'OK' if ok else 'FAIL'}")
print("fails:", fails)
sys.exit(1 if fails else 0)
