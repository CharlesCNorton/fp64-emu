// Shim for kernel-builder's registration.h, for local JIT builds.
// cpp_extension.load(is_python_module=False) loads via torch.ops.load_library,
// so TORCH_LIBRARY runs at dlopen and no PyInit module object is needed.
#pragma once

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#define REGISTER_EXTENSION(NAME)
