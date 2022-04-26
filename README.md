# Fast Convoluion Implementation via CUDA

## 1. Introduction
- Implementation list
    0. Naive convolution (CPU)
        - include/conv_cpu.cuh
        - parallelized via OpenMP
    1. Naive convolution (GPU)
        - include/conv_gpu_naive.cuh
    2. GEMM (im2col)
        - include/conv_gpu_matmul.cuh
    3. (TODO) FFT
    4. (TODO) Strassen's method
    5. (TODO) Winograd's method


## 2. How to Run
- build
    - make DEBUG=OFF
        - Skip a routine for checking computataion results
    - make DEBUG=ON
        - Do a routine for checking computataion results
- execute
    - make run