# GEMMul8 side-by-side drivers

`driver.cu` times `gemmul8::gemm<double>` directly against a GEMMul8 source
build; usage `driver <n> <moduli> <fast 0|1>`, median of 10 timed runs.
`worksize_impl.hpp` is compiled into the driver TU because MSVC drops the
public `workSize` instantiations.

On Linux the hook form measures GEMMul8 through torch instead:

    LD_PRELOAD=libgemmul8.so GEMMUL8_NUM_MOD_D_GEMM=18 GEMMUL8_FASTMODE_D_GEMM=0 \
      python3 ../compare.py --mode dgemm --sizes 512 1024 2048 4096 8192
