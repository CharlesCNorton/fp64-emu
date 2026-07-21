"""Pack the kernel sources and benchmark into one self-contained job script.

The metered node gets a single file with no repo checkout and no network
dependency beyond its base image, so the billed time is compile plus measure.
"""
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "job_blackwell.py"

PAYLOAD = [
    "fp64_emu_cuda/fp64_emu.cu",
    "torch-ext/torch_binding.cpp",
    "torch-ext/torch_binding.h",
    "torch-ext/fp64_emu/__init__.py",
    "torch-ext/fp64_emu/_meta.py",
    "bench/registration.h",
    "bench/accuracy.py",
]

HEAD = '''"""fp64-emu benchmark, self-contained. Writes sources, builds, measures."""
import base64, math, os, statistics, sys, time, types
from pathlib import Path

FILES = {}

WORK = Path(os.environ.get("FP64EMU_WORK", "/tmp/fp64emu"))
for rel, b64 in FILES.items():
    p = WORK / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64))
print("staged", len(FILES), "files in", WORK, flush=True)

import torch
from torch.utils.cpp_extension import load

cc = torch.cuda.get_device_capability(0)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cc[0]}.{cc[1]}")
print(f"device: {torch.cuda.get_device_name(0)}  CC {cc[0]}.{cc[1]}  torch {torch.__version__}", flush=True)
t0 = time.time()
load(name="fp64_emu_jit",
     sources=[str(WORK / "torch-ext/torch_binding.cpp"), str(WORK / "fp64_emu_cuda/fp64_emu.cu")],
     extra_include_paths=[str(WORK / "torch-ext"), str(WORK / "bench")],
     extra_cflags=["-DCUDA_KERNEL", "/O2" if os.name == "nt" else "-O3"],
     extra_cuda_cflags=["-DCUDA_KERNEL", "-O3"],
     extra_ldflags=["cublas.lib"] if os.name == "nt" else ["-lcublas"],
     is_python_module=False, verbose=False)
print(f"build: {time.time()-t0:.1f}s", flush=True)

ops_mod = types.ModuleType("fp64_emu._ops"); ops_mod.ops = torch.ops.fp64_emu_jit
sys.modules["fp64_emu._ops"] = ops_mod
pkg = types.ModuleType("fp64_emu"); pkg.__path__ = [str(WORK / "torch-ext/fp64_emu")]
sys.modules["fp64_emu"] = pkg
exec(compile((WORK / "torch-ext/fp64_emu/__init__.py").read_text(), "__init__.py", "exec"), pkg.__dict__)
emu = pkg
sys.path.insert(0, str(WORK / "bench"))
sys.modules["jit"] = types.ModuleType("jit")   # accuracy.py imports it; unused here
from accuracy import correct_bits, dd_matmul_nt, make


def timed(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


DISTS = ["randn", "wide8", "wide16", "illcond", "illcond40", "spike"]

# ---- correctness first: on sm_90+ the dispatch takes the TMA fused kernel,
# which no other machine available here can execute at all. ----
print("\\n## correctness\\n", flush=True)
fails = 0


def check(name, ok):
    global fails
    if not ok:
        fails += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)


def set_path(spec):
    """spec: auto | fused (TMA where available) | fusedcp (cp.async+tiled) | cublas."""
    os.environ.pop("FP64EMU_PATH", None)
    os.environ.pop("FP64EMU_NO_TMA", None)
    if spec == "fusedcp":
        os.environ["FP64EMU_PATH"] = "fused"
        os.environ["FP64EMU_NO_TMA"] = "1"
    elif spec != "auto":
        os.environ["FP64EMU_PATH"] = spec


for tag in ("auto", "fused", "fusedcp", "cublas"):
    set_path(tag)
    torch.manual_seed(1)
    Ai = torch.randint(-4096, 4097, (128, 256), device="cuda").double()
    Bi = torch.randint(-4096, 4097, (96, 256), device="cuda").double()
    ref = (Ai.long().cpu() @ Bi.long().cpu().T).double().cuda()
    check(f"[{tag}] integer oracle bit-exact", torch.equal(emu.mm(Ai, Bi), ref))
    Ko = 2048
    Ao = torch.ones(64, Ko, dtype=torch.float64, device="cuda")
    Bo = torch.ones(48, Ko, dtype=torch.float64, device="cuda")
    refo = torch.full((64, 48), float(Ko), dtype=torch.float64, device="cuda")
    check(f"[{tag}] all-ones range rule", torch.equal(emu.mm(Ao, Bo), refo))
    for (M_, K_, N_) in ((129, 4095, 127), (17, 23, 29), (512, 2047, 33)):
        a = torch.randn(M_, K_, dtype=torch.float64, device="cuda")
        b = torch.randn(N_, K_, dtype=torch.float64, device="cuda")
        r = a @ b.T
        rel = (emu.mm(a, b) - r).abs().max().item() / r.abs().max().item()
        check(f"[{tag}] shape {M_}x{K_}x{N_} (K%4={K_ % 4})",
              rel < 1e-11)
for tag in ("auto", "fused", "fusedcp", "cublas"):
    set_path(tag)
    torch.manual_seed(11)
    Ab = torch.randint(-2048, 2049, (4, 96, 256), device="cuda").double()
    Bb = torch.randint(-2048, 2049, (4, 64, 256), device="cuda").double()
    refb = torch.stack([(Ab[b].long().cpu() @ Bb[b].long().cpu().T).double()
                        for b in range(4)]).cuda()
    check(f"[{tag}] bmm integer oracle bit-exact", torch.equal(emu.bmm(Ab, Bb), refb))
    torch.manual_seed(14)
    Ant = torch.randint(-4096, 4097, (128, 256), device="cuda").double()
    Bnt = torch.randint(-4096, 4097, (256, 96), device="cuda").double()
    refnt = (Ant.long().cpu() @ Bnt.long().cpu()).double().cuda()
    check(f"[{tag}] mm_nt in-place layout bit-exact",
          torch.equal(emu.mm_nt(Ant, Bnt, nprimes=20), refnt))
set_path("auto")
torch.manual_seed(16)
Kd = 200_000
Ad = torch.randint(-1, 2, (48, Kd), device="cuda").double()
Bd = torch.randint(-1, 2, (32, Kd), device="cuda").double()
refd = (Ad.long().cpu() @ Bd.long().cpu().T).double().cuda()
check("deep-K segmented bit-exact", torch.equal(emu.mm(Ad, Bd), refd))
del Ad, Bd, refd
torch.cuda.empty_cache()
print(f"\\ncorrectness failures: {fails}", flush=True)

print("\\n## path comparison (TMA vs cp.async+tiled vs cuBLAS)\\n", flush=True)
print(f"{'n':>6} {'tma ms':>10} {'cpasync ms':>11} {'cublas ms':>10} {'winner':>8}")
for n in (1024, 2048, 4096, 8192, 16384):
    A = torch.randn(n, n, dtype=torch.float64, device="cuda")
    B = torch.randn(n, n, dtype=torch.float64, device="cuda")
    ts = {}
    for path in ("fused", "fusedcp", "cublas"):
        set_path(path)
        try:
            ts[path] = timed(lambda: emu.mm(A, B), 4)
        except torch.OutOfMemoryError:
            ts[path] = float("inf")
            torch.cuda.empty_cache()
    set_path("auto")
    w = min(ts, key=ts.get)
    print(f"{n:>6} {ts['fused']*1e3:>10.2f} {ts['fusedcp']*1e3:>11.2f} "
          f"{ts['cublas']*1e3:>10.2f} {w:>8}", flush=True)
    del A, B
    torch.cuda.empty_cache()

def cfgs_for(K):
    """Out-of-box default, then a reduced-cost point one modulus down."""
    p, c = emu.plan_config(K)
    return [(p, c, "default"), (p - 1, False, "fast")]


print("\\n## accuracy (correct bits vs double-double reference)\\n", flush=True)
print(f"{'dist':>10} {'K':>6} {'config':>18} {'gemms':>6} {'bits':>7} {'native':>7}")
for dist in DISTS:
    for K in (512, 4096):
        A, Bt = make(dist, 128, K, 128, seed=abs(hash((dist, K))) % 9999)
        hi, lo = dd_matmul_nt(A, Bt)
        nb, _ = correct_bits(A @ Bt, hi, lo)
        B = Bt.t().contiguous()
        for p, c, name in cfgs_for(K):
            cb, _ = correct_bits(emu.mm(A, B, nprimes=p, bits=53, corr=c), hi, lo)
            tag = f"{name} p{p} c{int(c)}"
            print(f"{dist:>10} {K:>6} {tag:>18} {p + (2 if c else 0):>6} {cb:>7.1f} {nb:>7.1f}", flush=True)
        del A, Bt, B, hi, lo
        torch.cuda.empty_cache()

print("\\n## throughput (TFLOP/s on 2 n^3)\\n", flush=True)
print(f"{'n':>6} {'config':>18} {'gemms':>6} {'ms':>10} {'TFLOP/s':>9} {'x native':>9}")
for n in (64, 256, 1024, 2048, 4096, 8192, 16384):
    A = torch.randn(n, n, dtype=torch.float64, device="cuda")
    Bt = torch.randn(n, n, dtype=torch.float64, device="cuda")
    B = Bt.t().contiguous()
    flop = 2.0 * n ** 3
    nit = 3 if n >= 16384 else (100 if n <= 256 else 6)
    tn = timed(lambda: A @ Bt, max(2, nit // 2), warmup=1)
    print(f"{n:>6} {'native fp64':>18} {'-':>6} {tn*1e3:>10.3f} {flop/tn/1e12:>9.2f} {1.0:>9.2f}", flush=True)
    for p, c, name in cfgs_for(n):
        t = timed(lambda: emu.mm(A, B, nprimes=p, bits=53, corr=c), nit)
        tag = f"{name} p{p} c{int(c)}"
        print(f"{n:>6} {tag:>18} {p + (2 if c else 0):>6} {t*1e3:>10.3f} "
              f"{flop/t/1e12:>9.2f} {tn/t:>9.2f}", flush=True)
    del A, Bt, B
    torch.cuda.empty_cache()

print("\\n## batched: emu.bmm vs native torch.bmm vs looped emu.mm\\n", flush=True)
print(f"{'batch x n':>12} {'native':>8} {'bmm':>8} {'x nat':>7} {'loop':>8} {'bmm/loop':>9}")
for Bn, n in ((64, 256), (32, 512), (16, 1024), (8, 2048), (4, 4096)):
    A = torch.randn(Bn, n, n, dtype=torch.float64, device="cuda")
    Bt = torch.randn(Bn, n, n, dtype=torch.float64, device="cuda")
    Bm = Bt.transpose(1, 2).contiguous()
    flop = 2.0 * Bn * n ** 3
    tn = timed(lambda: torch.bmm(A, Bt), 6)
    tb = timed(lambda: emu.bmm(A, Bm), 6)
    tl = timed(lambda: torch.stack([emu.mm(A[b], Bm[b]) for b in range(Bn)]), 4)
    print(f"{Bn:>5} x {n:<5} {flop/tn/1e12:>6.2f} {flop/tb/1e12:>6.2f} {tn/tb:>7.1f} "
          f"{flop/tl/1e12:>6.2f} {tl/tb:>8.2f}x", flush=True)
    del A, Bt, Bm
    torch.cuda.empty_cache()

print("\\n## GEMMul8 side by side (same card, torch dgemm hooked by LD_PRELOAD)\\n", flush=True)
import subprocess

G8 = "/tmp/g8"
if os.environ.get("JOB_SKIP_G8"):
    print("skipped by JOB_SKIP_G8", flush=True)
    print("\\nDONE", flush=True)
    sys.exit(0)
try:
    import tarfile
    import urllib.request

    tgz = "/tmp/g8.tar.gz"
    urllib.request.urlretrieve(
        "https://github.com/RIKEN-RCCS/GEMMul8/archive/refs/heads/main.tar.gz", tgz)
    with tarfile.open(tgz) as tf:
        top = tf.getnames()[0].split("/")[0]
        tf.extractall("/tmp")
    os.rename(f"/tmp/{top}", G8)
    r = subprocess.run(["make", "-C", G8, "-j"], capture_output=True, text=True, timeout=1800)
    so = Path(G8) / "lib" / "libgemmul8.so"
    if r.returncode != 0 or not so.exists():
        print("GEMMul8 build failed; last output:\\n" + r.stdout[-2000:] + r.stderr[-2000:],
              flush=True)
        so = None
except Exception as ex:  # noqa: BLE001 - report and continue, our numbers stand alone
    print(f"GEMMul8 phase skipped: {ex}", flush=True)
    so = None

BENCH = (
    "import torch, time, statistics\\n"
    "torch.cuda.init()\\n"
    "for n in (2048, 4096, 8192, 16384):\\n"
    "    A = torch.randn(n, n, dtype=torch.float64, device='cuda')\\n"
    "    B = torch.randn(n, n, dtype=torch.float64, device='cuda')\\n"
    "    for _ in range(3): A @ B\\n"
    "    torch.cuda.synchronize(); ts = []\\n"
    "    for _ in range(8):\\n"
    "        t0 = time.perf_counter(); A @ B; torch.cuda.synchronize()\\n"
    "        ts.append(time.perf_counter() - t0)\\n"
    "    t = statistics.median(ts)\\n"
    "    print(f'{n} {t*1e3:.2f} ms {2.0*n**3/t/1e12:.2f} TF', flush=True)\\n"
    "    del A, B; torch.cuda.empty_cache()\\n"
)
if so is not None:
    for mod, fast, name in ((16, 0, "mod16 accurate"), (17, 0, "mod17 accurate"),
                            (18, 0, "mod18 accurate"), (13, 1, "mod13 fast")):
        env = dict(os.environ)
        env.update({"LD_PRELOAD": str(so),
                    "GEMMUL8_NUM_MOD_D_GEMM": str(mod),
                    "GEMMUL8_FASTMODE_D_GEMM": str(fast)})
        env.pop("FP64EMU_PATH", None)
        print(f"-- GEMMul8 {name}", flush=True)
        r = subprocess.run([sys.executable, "-c", BENCH], env=env,
                           capture_output=True, text=True, timeout=1200)
        print(r.stdout, flush=True)
        if r.returncode != 0:
            print("stderr tail:\\n" + r.stderr[-1500:], flush=True)

print("\\nDONE", flush=True)
'''


def main():
    files = {}
    for rel in PAYLOAD:
        data = (ROOT / rel).read_bytes()
        files[rel] = base64.b64encode(data).decode()
    body = HEAD.replace("FILES = {}", "FILES = " + repr(files), 1)
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(files)} files)")


if __name__ == "__main__":
    main()
