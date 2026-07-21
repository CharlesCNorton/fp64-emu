"""Application drills: each algorithm runs with native fp64 GEMMs and with the
emulated default, reporting end-to-end time and an application-level accuracy
metric. Loads via _local/jit.py or the shipped load_local.py."""
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "jit.py").exists():
    from jit import build
    emu = build()
else:
    import load_local
    emu = load_local.load()

dev = "cuda"
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory
big = vram > 20e9
print(f"== {name} ({vram / 2**30:.0f} GiB) | torch {torch.__version__} ==", flush=True)


def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def g_native(A, B):
    return A @ B


def g_emu(A, B):
    return emu.mm_nt(A, B)


# ---- 1. raw throughput, reprising the card's operating points ---------------
def drill_throughput():
    n = 8192 if big else 4096
    A = torch.randn(n, n, dtype=torch.float64, device=dev)
    B = torch.randn(n, n, dtype=torch.float64, device=dev)
    tn = timeit(lambda: A @ B)
    te = timeit(lambda: emu.mm_nt(A, B))
    tr = timeit(lambda: emu.mm_nt(A, B, nprimes=13, corr=False))
    ref = A @ B
    d = (emu.mm_nt(A, B) - ref).abs().max().item() / ref.abs().max().item()
    fl = 2.0 * n**3
    print(f"[throughput] n={n}: native {fl/tn/1e12:5.2f} TF | "
          f"default {fl/te/1e12:5.2f} TF ({tn/te:.1f}x) | "
          f"reduced {fl/tr/1e12:5.2f} TF ({tn/tr:.1f}x) | emu-native rel {d:.1e}",
          flush=True)


# ---- 2. exact combinatorics: 4-step path counts on a random digraph ---------
# Adjacency is 0/1, so every count is an exact integer well under 2^53: native
# fp64 GEMM is exact here and the emulated product must match it bit for bit.
def drill_paths():
    n = 4096
    torch.manual_seed(7)
    A = (torch.rand(n, n, device=dev) < 0.05).double()

    def paths4(g):
        A2 = g(A, A)
        return g(A2, A2)

    P4n = paths4(g_native)
    P4e = paths4(g_emu)
    eq = torch.equal(P4n, P4e)
    tn = timeit(lambda: paths4(g_native), reps=3)
    te = timeit(lambda: paths4(g_emu), reps=3)
    print(f"[paths] {n}-node digraph, A^4 (max count {P4n.max().item():.2e}): "
          f"bitwise equal {eq} | native {tn*1e3:6.0f} ms, emu {te*1e3:6.0f} ms ({tn/te:.1f}x)",
          flush=True)


# ---- 3. CholeskyQR on an ill-conditioned tall-skinny matrix -----------------
# Q from chol(A^T A); the Gram matrix squares the condition number, so GEMM
# accuracy shows directly in the orthogonality residual.
def drill_cholqr():
    m, k = (65536, 512) if big else (32768, 512)
    torch.manual_seed(1)
    # Genuine cond = 1e6: scaled singular values between two rotations, so the
    # conditioning does not factor out of the Cholesky as a diagonal would.
    Q1, _ = torch.linalg.qr(torch.randn(m, k, dtype=torch.float64, device=dev))
    Q2, _ = torch.linalg.qr(torch.randn(k, k, dtype=torch.float64, device=dev))
    s = torch.logspace(0, -6, k, dtype=torch.float64, device=dev)
    A = (Q1 * s) @ Q2.t()
    At = A.t().contiguous()
    I = torch.eye(k, dtype=torch.float64, device=dev)

    def qr_resid(G):
        L = torch.linalg.cholesky(G)
        Q = torch.linalg.solve_triangular(L, At, upper=False).t()
        return (Q.t() @ Q - I).norm().item()

    tn = timeit(lambda: At @ A)
    te = timeit(lambda: emu.mm_nt(At, A))
    rn = qr_resid(At @ A)
    re = qr_resid(emu.mm_nt(At, A))
    print(f"[cholqr] {m}x{k}, cond ~1e6: ||Q^T Q - I||_F native {rn:.2e} vs emu {re:.2e} "
          f"({rn/re:.1f}x tighter) | Gram {tn*1e3:.1f} ms vs {te*1e3:.1f} ms",
          flush=True)


# ---- 4. Newton-Schulz polar factor ------------------------------------------
# X <- 1.5 X - 0.5 X (X^T X), pure GEMMs; converges to the orthogonal polar
# factor. Metric: the orthogonality floor it converges to, and wall time.
def drill_polar():
    n = 3072 if big else 2048
    torch.manual_seed(2)
    A = torch.randn(n, n, dtype=torch.float64, device=dev)
    A = A * torch.logspace(0, -2, n, dtype=torch.float64, device=dev)
    I = torch.eye(n, dtype=torch.float64, device=dev)

    def polar(g, iters=50):
        X = A / A.norm()
        for _ in range(iters):
            Y = g(X.t().contiguous(), X)
            X = 1.5 * X - 0.5 * g(X, Y)
        return X

    for tag, g in (("native", g_native), ("emu   ", g_emu)):
        t0 = time.perf_counter()
        X = polar(g)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        r = (X.t() @ X - I).norm().item()
        print(f"[polar] {tag} n={n}, 50 iters: ||X^T X - I||_F = {r:.2e} | {t:5.2f} s",
              flush=True)


# ---- 5. matrix exponential: scaling-and-squaring Taylor ---------------------
# All GEMMs through the engine under test; reference is torch.matrix_exp
# (Pade in native fp64). The repeated squarings amplify any GEMM error.
def drill_expm():
    n = 2048
    torch.manual_seed(3)
    A = torch.randn(n, n, dtype=torch.float64, device=dev) / math.sqrt(n)
    I = torch.eye(n, dtype=torch.float64, device=dev)
    t0 = time.perf_counter()
    ref = torch.matrix_exp(A)
    torch.cuda.synchronize()
    tref = time.perf_counter() - t0

    def expm(g, s=6, terms=18):
        As = A * (2.0 ** -s)
        E = I + As
        T = As
        for k in range(2, terms):
            T = g(T, As) / k
            E = E + T
        for _ in range(s):
            E = g(E, E)
        return E

    for tag, g in (("native", g_native), ("emu   ", g_emu)):
        t = timeit(lambda: expm(g), reps=3, warm=1)
        err = ((expm(g) - ref).norm() / ref.norm()).item()
        print(f"[expm] {tag} n={n}, Taylor+squaring: rel vs matrix_exp {err:.2e} | "
              f"{t*1e3:6.0f} ms (torch.matrix_exp: {tref*1e3:.0f} ms)", flush=True)


drill_throughput()
drill_paths()
drill_cholqr()
drill_polar()
drill_expm()
print("done.", flush=True)
