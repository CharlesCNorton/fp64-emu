"""Tiled right-looking Cholesky with batched trailing updates: the update
GEMMs run through emu.bmm or torch.bmm, everything else stays native, and
both engines share identical gather/scatter, so the comparison isolates the
batched product."""
import statistics
import sys
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


def factor(A, t, update_bmm):
    """In-place lower-Cholesky of a copy of A with tile size t; returns
    (L, seconds spent in the batched trailing updates)."""
    W = A.clone()
    n = W.size(0)
    nt = n // t
    upd = 0.0
    for k in range(nt):
        k0 = k * t
        Lkk = torch.linalg.cholesky(W[k0:k0 + t, k0:k0 + t])
        W[k0:k0 + t, k0:k0 + t] = Lkk
        if k + 1 == nt:
            break
        panel = torch.linalg.solve_triangular(
            Lkk, W[k0 + t:, k0:k0 + t].t(), upper=False).t().contiguous()
        W[k0 + t:, k0:k0 + t] = panel
        rem = nt - k - 1
        tiles = panel.view(rem, t, t)
        I, J = torch.tril_indices(rem, rem)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        S = update_bmm(tiles[I].contiguous(), tiles[J].contiguous())
        for p in range(I.numel()):
            i0, j0 = k0 + t + int(I[p]) * t, k0 + t + int(J[p]) * t
            W[i0:i0 + t, j0:j0 + t] -= S[p]
        torch.cuda.synchronize()
        upd += time.perf_counter() - t0
    return torch.tril(W), upd


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    torch.manual_seed(0)
    G = torch.randn(n, n, dtype=torch.float64, device="cuda")
    A = G @ G.t() / n + 2.0 * torch.eye(n, dtype=torch.float64, device="cuda")
    del G
    nrm = A.norm()
    engines = {
        "native": lambda X, Y: torch.bmm(X, Y.transpose(1, 2)),
        "emu": lambda X, Y: emu.bmm(X, Y),
    }
    print(f"{torch.cuda.get_device_name(0)} | n={n} t={t}")
    for name, eng in engines.items():
        ts, us = [], []
        L = None
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            L, upd = factor(A, t, eng)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
            us.append(upd)
        resid = ((A - L @ L.t()).norm() / nrm).item()
        print(f"{name:>7}: total {statistics.median(ts):6.2f} s | updates "
              f"{statistics.median(us):6.2f} s | residual {resid:.2e}", flush=True)


if __name__ == "__main__":
    main()
