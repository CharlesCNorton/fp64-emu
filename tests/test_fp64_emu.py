import math
import pytest
import torch

import kernels

emu = kernels.get_kernel("phanerozoic/fp64-emu", version=1, trust_remote_code=True)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
@pytest.mark.kernels_ci
def test_integer_products_accurate():
    """Exact-integer inputs, fast recombine: fp64-competitive (>= 45 bits)."""
    torch.manual_seed(0)
    A = torch.randint(-1000, 1001, (128, 256), device="cuda").double()
    Bt = torch.randint(-1000, 1001, (256, 96), device="cuda").double()
    ref = (A.long().cpu() @ Bt.long().cpu()).double().cuda()
    C = emu.mm_nt(A, Bt, nprimes=18)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    bits = -math.log2(rel) if rel > 0 else 53.0
    assert bits >= 45.0, bits


@requires_cuda
@pytest.mark.kernels_ci
def test_exact_mode_bitwise_integer_oracle():
    """exact=True with enough primes reconstructs the integer product exactly
    (int128 recombine, correctly rounded): bitwise equal to the true product."""
    torch.manual_seed(1)
    A = torch.randint(-4096, 4097, (128, 256), device="cuda").double()
    Bt = torch.randint(-4096, 4097, (256, 96), device="cuda").double()
    ref = (A.long().cpu() @ Bt.long().cpu()).double().cuda()   # exact, < 2^53
    C = emu.mm_nt(A, Bt, nprimes=20, exact=True)
    assert torch.equal(C, ref)


@requires_cuda
@pytest.mark.kernels_ci
@pytest.mark.parametrize("nprimes", [18, 20])
def test_random_fp64_accuracy(nprimes):
    torch.manual_seed(2)
    A = torch.randn(512, 1024, dtype=torch.float64, device="cuda")
    Bt = torch.randn(1024, 512, dtype=torch.float64, device="cuda")
    ref = A @ Bt
    C = emu.mm_nt(A, Bt, nprimes=nprimes)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    bits = -math.log2(rel) if rel > 0 else 53.0
    assert bits >= 45.0, bits


@requires_cuda
@pytest.mark.kernels_ci
def test_range_rule_adversarial_aligned():
    """All-ones operands scale to 2^52, so each dot product is K * 2^104; the
    default configuration must reconstruct it exactly."""
    K = 2048
    A = torch.ones(64, K, dtype=torch.float64, device="cuda")
    Bt = torch.ones(K, 48, dtype=torch.float64, device="cuda")
    C = emu.mm_nt(A, Bt, exact=True)   # default nprimes = 20
    ref = torch.full((64, 48), float(K), dtype=torch.float64, device="cuda")
    assert torch.equal(C, ref)


@requires_cuda
@pytest.mark.kernels_ci
def test_dyadic_dynamic_range_bitexact():
    """Integers under per-column power-of-two scales spanning 2^-12..2^0; the
    exact int64 product must be reproduced bit for bit at defaults."""
    torch.manual_seed(5)
    M, K, N = 64, 512, 48
    ek = torch.randint(-12, 1, (K,), device="cuda")
    Ai = torch.randint(-256, 257, (M, K), device="cuda")
    Bi = torch.randint(-256, 257, (N, K), device="cuda")
    # Exact dyadic construction: integer shifts then one exact power-of-two
    # scalar (CUDA torch.pow(2.0, t) is not exact to the last ulp).
    A = torch.bitwise_left_shift(Ai, ek + 12).double() * (2.0 ** -12)
    B = torch.bitwise_left_shift(Bi, ek + 12).double() * (2.0 ** -12)
    w = (4 ** (ek + 12)).long().cpu()                # per-k weight, exact in int64
    ref_int = (Ai.long().cpu() * w) @ Bi.long().cpu().T   # |sum| < 2^51: exact (CPU int64 matmul)
    ref = (ref_int.double() * (2.0 ** -24)).cuda()   # exact rescale
    assert torch.equal(emu.mm(A, B), ref)


@requires_cuda
@pytest.mark.kernels_ci
def test_bmm_single_batch_matches_mm():
    """A batch of one takes the identical sizing path: bitwise equal to mm."""
    torch.manual_seed(4)
    A = torch.randn(192, 384, dtype=torch.float64, device="cuda")
    B = torch.randn(160, 384, dtype=torch.float64, device="cuda")
    assert torch.equal(emu.bmm(A[None], B[None])[0], emu.mm(A, B))


@requires_cuda
@pytest.mark.kernels_ci
def test_bmm_integer_oracle_bitexact():
    torch.manual_seed(6)
    A = torch.randint(-4096, 4097, (5, 96, 256), device="cuda").double()
    B = torch.randint(-4096, 4097, (5, 64, 256), device="cuda").double()
    ref = torch.stack([(A[b].long().cpu() @ B[b].long().cpu().T).double()
                       for b in range(5)]).cuda()
    assert torch.equal(emu.bmm(A, B), ref)


@requires_cuda
@pytest.mark.kernels_ci
def test_bmm_odd_shapes_accuracy():
    """Non-tile-multiple per-member shapes exercise the per-batch tiled grid."""
    torch.manual_seed(8)
    A = torch.randn(3, 100, 333, dtype=torch.float64, device="cuda")
    Bt = torch.randn(3, 333, 77, dtype=torch.float64, device="cuda")
    ref = torch.bmm(A, Bt)
    C = emu.bmm_nt(A, Bt)
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    bits = -math.log2(rel) if rel > 0 else 53.0
    assert bits >= 45.0, bits


@requires_cuda
@pytest.mark.kernels_ci
def test_deep_k_integer_bitexact():
    """K past the single-launch bound runs segmented; exact integers must
    reproduce bit for bit through the residue accumulators."""
    torch.manual_seed(16)
    K = 200_000
    A = torch.randint(-1, 2, (48, K), device="cuda").double()
    B = torch.randint(-1, 2, (32, K), device="cuda").double()
    ref = (A.long().cpu() @ B.long().cpu().T).double().cuda()
    assert torch.equal(emu.mm(A, B), ref)


@requires_cuda
@pytest.mark.kernels_ci
def test_deep_k_randn_accuracy():
    torch.manual_seed(17)
    K = 150_000
    A = torch.randn(64, K, dtype=torch.float64, device="cuda")
    B = torch.randn(48, K, dtype=torch.float64, device="cuda")
    ref = A @ B.t()
    rel = (emu.mm(A, B) - ref).abs().max().item() / ref.abs().max().item()
    bits = -math.log2(rel) if rel > 0 else 53.0
    assert bits >= 40.0, bits


@requires_cuda
@pytest.mark.kernels_ci
def test_mm_nt_paths_agree():
    """The [K, N]-direct cuBLAS form and the transposed fused form emit
    identical bits from the shared column shifts."""
    import os
    torch.manual_seed(15)
    A = torch.randn(256, 1024, dtype=torch.float64, device="cuda")
    Bt = torch.randn(1024, 200, dtype=torch.float64, device="cuda")
    outs = []
    for p in ("fused", "cublas", "chunked"):
        os.environ["FP64EMU_PATH"] = p
        outs.append(emu.mm_nt(A, Bt))
    os.environ.pop("FP64EMU_PATH", None)
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


@requires_cuda
@pytest.mark.kernels_ci
def test_cuda_graph_capture():
    """The steady-state call is capture-safe; replay recomputes from the
    current contents of the captured input buffers."""
    torch.manual_seed(13)
    A = torch.randn(128, 256, dtype=torch.float64, device="cuda")
    B = torch.randn(96, 256, dtype=torch.float64, device="cuda")
    for _ in range(3):
        emu.mm(A, B)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        C = emu.mm(A, B)
    A.copy_(torch.randn(128, 256, dtype=torch.float64, device="cuda"))
    B.copy_(torch.randn(96, 256, dtype=torch.float64, device="cuda"))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(C, emu.mm(A, B))


@requires_cuda
@pytest.mark.kernels_ci
def test_torch_compile_traces():
    torch.manual_seed(12)
    A = torch.randn(64, 128, dtype=torch.float64, device="cuda")
    B = torch.randn(48, 128, dtype=torch.float64, device="cuda")
    eager = emu.mm(A, B)
    compiled = torch.compile(emu.mm, dynamic=False)
    assert torch.equal(compiled(A, B), eager)


@requires_cuda
@pytest.mark.kernels_ci
def test_bmm_deterministic():
    torch.manual_seed(9)
    A = torch.randn(4, 128, 512, dtype=torch.float64, device="cuda")
    B = torch.randn(4, 96, 512, dtype=torch.float64, device="cuda")
    assert torch.equal(emu.bmm(A, B), emu.bmm(A, B))


@requires_cuda
@pytest.mark.kernels_ci
def test_deterministic():
    torch.manual_seed(3)
    A = torch.randn(256, 512, dtype=torch.float64, device="cuda")
    Bt = torch.randn(512, 256, dtype=torch.float64, device="cuda")
    assert torch.equal(emu.mm_nt(A, Bt, nprimes=18), emu.mm_nt(A, Bt, nprimes=18))


@requires_cuda
@pytest.mark.kernels_ci
def test_beats_native_fp64():
    import time
    n = 4096
    A = torch.randn(n, n, dtype=torch.float64, device="cuda")
    Bt = torch.randn(n, n, dtype=torch.float64, device="cuda")
    def t(fn):
        for _ in range(3): fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(5): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / 5
    assert t(lambda: A @ Bt) / t(lambda: emu.mm_nt(A, Bt, nprimes=18)) > 3.0
