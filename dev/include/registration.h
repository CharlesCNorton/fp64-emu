// Shim for kernel-builder's registration.h in local JIT builds: the library
// loads via torch.ops.load_library, so no PyInit module object is needed.
#pragma once

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#define REGISTER_EXTENSION(NAME)
