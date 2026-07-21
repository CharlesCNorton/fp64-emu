"""Build fp64-emu in place and expose it as the `fp64_emu` package.

Published kernel builds are Linux-only, so on SAURON the .cu is compiled with
torch.utils.cpp_extension against the repo's real torch_binding.cpp and bound
under a private op namespace, which `fp64_emu._ops` is then stubbed to.
"""
import os
import sys
import types
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent.parent
LOCAL = Path(__file__).resolve().parent
NAME = "fp64_emu_jit"


def build(verbose: bool = False, extra_cuda: tuple = ()) -> types.ModuleType:
    """Compile the kernel, register the ops, and return the imported package."""
    win = os.name == "nt"
    load(
        name=NAME,
        sources=[str(ROOT / "torch-ext" / "torch_binding.cpp"),
                 str(ROOT / "fp64_emu_cuda" / "fp64_emu.cu")],
        extra_include_paths=[str(ROOT / "torch-ext"), str(LOCAL)],
        extra_cflags=["-DCUDA_KERNEL", "/O2" if win else "-O3"],
        extra_cuda_cflags=["-DCUDA_KERNEL", "-O3", "-lineinfo", *extra_cuda],
        extra_ldflags=["cublas.lib"] if win else ["-lcublas"],
        is_python_module=False,
        verbose=verbose,
    )

    ops_mod = types.ModuleType("fp64_emu._ops")
    ops_mod.ops = getattr(torch.ops, NAME)
    sys.modules["fp64_emu._ops"] = ops_mod

    pkg = types.ModuleType("fp64_emu")
    pkg.__path__ = [str(ROOT / "torch-ext" / "fp64_emu")]
    sys.modules["fp64_emu"] = pkg
    src = (ROOT / "torch-ext" / "fp64_emu" / "__init__.py").read_text(encoding="utf-8")
    exec(compile(src, str(ROOT / "torch-ext" / "fp64_emu" / "__init__.py"), "exec"),
         pkg.__dict__)
    return pkg


if __name__ == "__main__":
    emu = build(verbose="-v" in sys.argv)
    print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
    cc = torch.cuda.get_device_capability(0)
    print("compute capability %d.%d | maxp %d" % (cc[0], cc[1], torch.ops.fp64_emu_jit.fp64_emu_maxp()))
    A = torch.randn(256, 512, dtype=torch.float64, device="cuda")
    Bt = torch.randn(512, 256, dtype=torch.float64, device="cuda")
    C, ref = emu.mm_nt(A, Bt), A @ Bt
    rel = (C - ref).abs().max().item() / ref.abs().max().item()
    import math
    print("smoke: rel %.3e -> %.1f bits" % (rel, -math.log2(rel) if rel > 0 else 53.0))
