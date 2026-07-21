#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("fp64_emu_maxp() -> int");
  ops.def("fp64_emu_mm(Tensor! out, Tensor A, Tensor B, Tensor primes_cpu, Tensor inv_cpu, Tensor mu_cpu, bool exact, int bits=53, bool use_corr=True) -> ()");
  ops.def("fp64_emu_bmm(Tensor! out, Tensor A, Tensor B, Tensor primes_cpu, Tensor inv_cpu, Tensor mu_cpu, bool exact, int bits=53, bool use_corr=True) -> ()");
  ops.def("fp64_emu_mm_nt(Tensor! out, Tensor A, Tensor Bt, Tensor primes_cpu, Tensor inv_cpu, Tensor mu_cpu, bool exact, int bits=53, bool use_corr=True) -> ()");
  ops.def("fp64_emu_bmm_nt(Tensor! out, Tensor A, Tensor Bt, Tensor primes_cpu, Tensor inv_cpu, Tensor mu_cpu, bool exact, int bits=53, bool use_corr=True) -> ()");
  ops.def("fp64_emu_mm_timed(Tensor! out, Tensor! times, Tensor A, Tensor B, Tensor primes_cpu, Tensor inv_cpu, Tensor mu_cpu, bool exact, int bits=53, bool use_corr=True) -> ()");

  ops.impl("fp64_emu_maxp", &fp64_emu_maxp);
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("fp64_emu_mm", torch::kCUDA, &fp64_emu_mm);
  ops.impl("fp64_emu_bmm", torch::kCUDA, &fp64_emu_bmm);
  ops.impl("fp64_emu_mm_nt", torch::kCUDA, &fp64_emu_mm_nt);
  ops.impl("fp64_emu_bmm_nt", torch::kCUDA, &fp64_emu_bmm_nt);
  ops.impl("fp64_emu_mm_timed", torch::kCUDA, &fp64_emu_mm_timed);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
