"""Accuracy of the out-of-box call, mm(A, B) with no tuning, across depths."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jit  # noqa: E402
from accuracy import correct_bits, dd_matmul_nt, make  # noqa: E402

emu = jit.build()
print(f"{'K':>7} {'cfg':>12} {'gemms':>6} {'randn':>7} {'wide16':>7} {'illc40':>7} {'spike':>7}")
print("-" * 58)
for K in (64, 256, 1024, 4096):
    p, c = emu.plan_config(K)
    row = []
    for dist in ("randn", "wide16", "illcond40", "spike"):
        A, Bt = make(dist, 128, K, 128, seed=7)
        hi, lo = dd_matmul_nt(A, Bt)
        C = emu.mm(A, Bt.t().contiguous())          # defaults only
        row.append(correct_bits(C, hi, lo)[0])
    print(f"{K:>7} {f'p{p} c{int(c)}':>12} {p+2:>6} " + " ".join(f"{v:>7.1f}" for v in row))
