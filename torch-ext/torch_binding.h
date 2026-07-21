#pragma once

#include <torch/torch.h>

int64_t fp64_emu_maxp();
void fp64_emu_mm(torch::Tensor &out, torch::Tensor const &A, torch::Tensor const &B,
                 torch::Tensor const &primes_cpu, torch::Tensor const &inv_cpu,
                 torch::Tensor const &mu_cpu, bool exact, int64_t bits, bool use_corr);
void fp64_emu_bmm(torch::Tensor &out, torch::Tensor const &A, torch::Tensor const &B,
                  torch::Tensor const &primes_cpu, torch::Tensor const &inv_cpu,
                  torch::Tensor const &mu_cpu, bool exact, int64_t bits, bool use_corr);
void fp64_emu_mm_nt(torch::Tensor &out, torch::Tensor const &A, torch::Tensor const &Bt,
                    torch::Tensor const &primes_cpu, torch::Tensor const &inv_cpu,
                    torch::Tensor const &mu_cpu, bool exact, int64_t bits, bool use_corr);
void fp64_emu_bmm_nt(torch::Tensor &out, torch::Tensor const &A, torch::Tensor const &Bt,
                     torch::Tensor const &primes_cpu, torch::Tensor const &inv_cpu,
                     torch::Tensor const &mu_cpu, bool exact, int64_t bits, bool use_corr);
void fp64_emu_mm_timed(torch::Tensor &out, torch::Tensor &times, torch::Tensor const &A,
                       torch::Tensor const &B, torch::Tensor const &primes_cpu,
                       torch::Tensor const &inv_cpu, torch::Tensor const &mu_cpu, bool exact,
                       int64_t bits, bool use_corr);
