"""bmm path cross-check: every dispatch path must emit identical bits."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402

emu = jit.build()
fails = 0
for B, M, K, N in [(4, 128, 512, 96), (3, 100, 333, 77), (8, 256, 1024, 256),
                   (2, 129, 4095, 127)]:
    torch.manual_seed(B)
    A = torch.randn(B, M, K, dtype=torch.float64, device="cuda")
    Bm = torch.randn(B, N, K, dtype=torch.float64, device="cuda")
    outs = {}
    for path in ("fused", "cublas", "chunked"):
        os.environ["FP64EMU_PATH"] = path
        outs[path] = emu.bmm(A, Bm)
    os.environ.pop("FP64EMU_PATH", None)
    ref = torch.stack([A[b] @ Bm[b].t() for b in range(B)])
    rel = (outs["fused"] - ref).abs().max().item() / ref.abs().max().item()
    eq = torch.equal(outs["fused"], outs["cublas"]) and torch.equal(outs["fused"], outs["chunked"])
    looped = torch.stack([emu.mm(A[b], Bm[b]) for b in range(B)])
    ok = eq and rel < 1e-11
    fails += 0 if ok else 1
    print(f"{B}x{M}x{K}x{N}: paths equal {eq} | rel {rel:.2e} | "
          f"vs looped mm max|d| {(outs['fused'] - looped).abs().max().item():.2e} "
          f"{'OK' if ok else 'FAIL'}")
print("fails:", fails)
sys.exit(1 if fails else 0)
