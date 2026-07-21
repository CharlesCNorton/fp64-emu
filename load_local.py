"""JIT-build fp64-emu from this checkout and expose the identical API.

For Windows or torch builds without a published variant; `load()` returns the
`fp64_emu` package. Requires nvcc and, on Windows, MSVC (vcvars64).
"""
import os
import sys
import types
from pathlib import Path

import torch
from torch.utils.cpp_extension import load as _load

ROOT = Path(__file__).resolve().parent
NAME = "fp64_emu_jit"


def load(verbose: bool = False) -> types.ModuleType:
    """Compile the kernel, register its ops, and return the `fp64_emu` package."""
    win = os.name == "nt"
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}"
    _load(
        name=NAME,
        sources=[str(ROOT / "torch-ext" / "torch_binding.cpp"),
                 str(ROOT / "fp64_emu_cuda" / "fp64_emu.cu")],
        extra_include_paths=[str(ROOT / "torch-ext"), str(ROOT / "dev" / "include")],
        extra_cflags=["-DCUDA_KERNEL", "/O2" if win else "-O3"],
        extra_cuda_cflags=["-DCUDA_KERNEL", "-O3"],
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
