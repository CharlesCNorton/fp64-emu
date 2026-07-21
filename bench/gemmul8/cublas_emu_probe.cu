// Native vs emulated cublasDgemm (CUDA >= 13.0u2): median ms, TFLOP/s, and
// an optional C dump for cross-accuracy. Emulation engages via the
// CUBLAS_EMULATION_STRATEGY environment variable, so one binary serves both
// modes. Usage: probe <n> [dumpfile]
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "cuda %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cublas %d: %d\n", __LINE__, (int)s); exit(1); } } while (0)

__global__ void fill_kernel(double* p, size_t n, unsigned long long seed) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long x = seed + i * 2654435761ULL;
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
  p[i] = ((double)(x >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: probe <n> [dumpfile]\n"); return 2; }
  const int n = atoi(argv[1]);
  const size_t N = (size_t)n * n;
  double *A, *B, *C;
  CK(cudaMalloc(&A, N * 8)); CK(cudaMalloc(&B, N * 8)); CK(cudaMalloc(&C, N * 8));
  fill_kernel<<<(unsigned)((N + 255) / 256), 256>>>(A, N, 1ULL);
  fill_kernel<<<(unsigned)((N + 255) / 256), 256>>>(B, N, 2ULL);
  CK(cudaDeviceSynchronize());
  cublasHandle_t h; CB(cublasCreate(&h));
  const double alpha = 1.0, beta = 0.0;
  auto once = [&] {
    CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n));
  };
  for (int i = 0; i < 3; i++) once();
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  std::vector<float> ms;
  for (int i = 0; i < 10; i++) {
    CK(cudaEventRecord(e0)); once(); CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float t = 0; CK(cudaEventElapsedTime(&t, e0, e1));
    ms.push_back(t);
  }
  std::sort(ms.begin(), ms.end());
  const double med = ms[ms.size() / 2];
  const char* strat = getenv("CUBLAS_EMULATION_STRATEGY");
  printf("n=%d strategy=%s %.3f ms %.2f TF\n", n, strat ? strat : "default", med,
         2.0 * (double)n * n * n / (med * 1e-3) / 1e12);
  if (argc > 2) {
    std::vector<double> host(N);
    CK(cudaMemcpy(host.data(), C, N * 8, cudaMemcpyDeviceToHost));
    FILE* f = fopen(argv[2], "wb");
    fwrite(host.data(), 8, N, f);
    fclose(f);
  }
  return 0;
}
