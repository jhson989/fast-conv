#pragma once

#include <vector>
#include <cstdio>
#include <helper.cuh>

template <typename T>
__global__ void __kernel_conv_naive(
    T* input, T* filter, T* output,
    const int BATCH_NUM, 
    const int INPUT_C, const int INPUT_H,const int INPUT_W, 
    const int FILTER_H, const int FILTER_W, 
    const int PAD_H, const int PAD_W, 
    const int STRIDE_H, const int STRIDE_W, 
    const int OUTPUT_C, const int OUTPUT_H, const int OUTPUT_W
) {

    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int out_c = blockIdx.y * blockDim.y + threadIdx.y;
    int out_hw = blockIdx.x * blockDim.x + threadIdx.x;
    int out_h = out_hw/OUTPUT_W;
    int out_w = out_hw%OUTPUT_W;

    if (out_c<OUTPUT_C && out_hw<OUTPUT_H*OUTPUT_W) {

        T value = 0;
        int y = STRIDE_H*out_h-PAD_H;
        int x = STRIDE_W*out_w-PAD_W;
    
        for (int c=0; c<INPUT_C; c++) {
            for (int h=0;h<FILTER_H; h++) {
                for (int w=0;w<FILTER_W; w++) {
    
                    if ( (0<=(y+h)&&(y+h)<INPUT_H) && (0<=(x+w)&&(x+w)<INPUT_W)  ) {
                        value += filter[out_c*(INPUT_C*FILTER_H*FILTER_W) + c*(FILTER_H*FILTER_W) + h*(FILTER_W) + w] * input[batch*(INPUT_C*INPUT_H*INPUT_W) + c*(INPUT_H*INPUT_W) + (y+h)*(INPUT_W) + (x+w)];
                    }
    
                }
            }
        }
        
        output[batch*(OUTPUT_C*OUTPUT_H*OUTPUT_W) + out_c*(OUTPUT_H*OUTPUT_W) + out_h*(OUTPUT_W) + out_w] = value;
    }
}

template <typename T, int WARP_SIZE>
void conv_gpu_naive(
    const std::vector<T>& input,
    const std::vector<T>& filter,
    std::vector<T>& output, 
    const int BATCH_NUM, 
    const int INPUT_C, const int INPUT_H,const int INPUT_W, 
    const int FILTER_H, const int FILTER_W, 
    const int PAD_H, const int PAD_W, 
    const int STRIDE_H, const int STRIDE_W, 
    const int OUTPUT_C, const int OUTPUT_H, const int OUTPUT_W
) {
    printf("GPU naive implementation launched...\n");

    /***************************************************************************
     * GPU memory initialization
     ***************************************************************************/

    // Alloc GPU memory
    T *d_input, *d_filter, *d_output;
    cudaErrChk( cudaMalloc((void**)&d_input, sizeof(T)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMalloc((void**)&d_filter, sizeof(T)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W) );
    cudaErrChk( cudaMalloc((void**)&d_output, sizeof(T)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W) );
    
    // Memcpy from host to device
    cudaErrChk( cudaMemcpy(d_input, input.data(), sizeof(T)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaMemcpy(d_filter, filter.data(),sizeof(T)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );



    /***************************************************************************
     * Launch kernel
     ***************************************************************************/

    const dim3 dim_threads(WARP_SIZE, WARP_SIZE, 1);
    const dim3 dim_blocks((OUTPUT_H*OUTPUT_W+WARP_SIZE-1)/WARP_SIZE, (OUTPUT_C+WARP_SIZE-1)/WARP_SIZE, BATCH_NUM);

    #ifdef DEBUG_ON
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );
    #endif

    // GPU kernel
    __kernel_conv_naive<<<dim_blocks, dim_threads>>> (d_input, d_filter, d_output, BATCH_NUM, INPUT_C,INPUT_H,INPUT_W, FILTER_H,FILTER_W, PAD_H,PAD_W, STRIDE_H,STRIDE_W, OUTPUT_C,OUTPUT_H,OUTPUT_W);
    cudaErrChk( cudaMemcpy(output.data(), d_output, sizeof(T)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W, cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    #ifdef DEBUG_ON
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" - Elapsed time: %.3f s\n", msec_total*1e-3);
    #endif



    /***************************************************************************
     * Finalize
     ***************************************************************************/

    cudaErrChk( cudaFree(d_input) );
    cudaErrChk( cudaFree(d_filter) );
    cudaErrChk( cudaFree(d_output) );
}