#pragma once

#include <vector>
#include <cstdio>
#include <helper.cuh>


/****************************************************************************************
 * Device code for im2col conversion
 ****************************************************************************************/
template <typename T>
__global__ void __kernel_im2col(
     const T* input, T* col,
     const int BATCH_NUM, 
     const int INPUT_C, const int INPUT_H,const int INPUT_W, 
     const int FILTER_H, const int FILTER_W, 
     const int PAD_H, const int PAD_W, 
     const int STRIDE_H, const int STRIDE_W, 
     const int OUTPUT_C, const int OUTPUT_H, const int OUTPUT_W,
     const int BATCH_OFFSET
) {
    
    int batch = blockIdx.y * blockDim.y + threadIdx.y;
    int out = blockIdx.x * blockDim.x + threadIdx.x;

    if (out<OUTPUT_H*OUTPUT_W) {
        
        int y = STRIDE_H*(out/OUTPUT_W)-PAD_H;
        int x = STRIDE_W*(out%OUTPUT_W)-PAD_W;
    
        int idx =0;
        for (int c=0; c<INPUT_C; c++) {
            for (int h=0;h<FILTER_H; h++) {
                for (int w=0;w<FILTER_W; w++) {
    
                    if ( (0<=(y+h)&&(y+h)<INPUT_H) && (0<=(x+w)&&(x+w)<INPUT_W)  ) {
                        col[batch*BATCH_OFFSET + (idx*OUTPUT_H*OUTPUT_W) + out] = input[batch*(INPUT_C*INPUT_H*INPUT_W) + c*(INPUT_H*INPUT_W) + (y+h)*(INPUT_W) + (x+w)];
                    } else {
                        col[batch*BATCH_OFFSET + (idx*OUTPUT_H*OUTPUT_W) + out] = 0;
                    }
                    idx++;
                }
            }
        }


    }
}
 

/****************************************************************************************
 * Device code for matmul
 ****************************************************************************************/
template <typename T>
__global__ void __kernel_conv_matmul(
    const T* A, const T* B, T* C,
    const int BATCH_NUM, const int M, const int K, const int N
) {

    int batch = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y<M && x<N) {
        T sum = 0;
        for (int k=0; k<K; k++) {
            sum += A[y*K + k] * B[batch*K*N + k*N + x];
        }
        C[batch*M*N + y*N+x] = sum;
    } 

}



/****************************************************************************************
 * Host code for convolution matmul
 ****************************************************************************************/
template <typename T, int WARP_SIZE>
void conv_gpu_matmul(
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

    printf("GPU matmul implementation launched...\n");

    /***************************************************************************
     * GPU memory initialization
     ***************************************************************************/

    const int KERNEL = INPUT_C*FILTER_H*FILTER_W;
    const int OUTPUT = OUTPUT_H*OUTPUT_W;
    // Alloc GPU memory
    T *d_input, *d_col, *d_filter, *d_output;
    cudaErrChk( cudaMalloc((void**)&d_input, sizeof(T)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) ); 
    cudaErrChk( cudaMalloc((void**)&d_col, sizeof(T)*BATCH_NUM*KERNEL*OUTPUT) ); // ( BATCH_NUM * KERNEL * OUTPUT )
    cudaErrChk( cudaMalloc((void**)&d_filter, sizeof(T)*OUTPUT_C*KERNEL) ); // (  OUTPUT_C * KERNEL )
    cudaErrChk( cudaMalloc((void**)&d_output, sizeof(T)*BATCH_NUM*OUTPUT_C*OUTPUT) ); // ( OUTPUT_C * KERNEL ) * ( BATCH_NUM * KERNEL * OUTPUT ) ==> ( BATCH_NUM * OUTPUT_C * OUTPUT )
    
    // Memcpy from host to device
    cudaErrChk( cudaMemcpy(d_input, input.data(), sizeof(T)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaMemcpy(d_filter, filter.data(),sizeof(T)*OUTPUT_C*INPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );


    /***************************************************************************
     * Kernel Configuration
     ***************************************************************************/
    #ifdef DEBUG_ON
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    #endif

    /***************************************************************************
     * im2col
     ***************************************************************************/
   
    const dim3 dim_threads_im2col(WARP_SIZE*WARP_SIZE, 1);
    const dim3 dim_blocks_im2col((OUTPUT+dim_threads_im2col.x-1)/dim_threads_im2col.x, BATCH_NUM);

    #ifdef DEBUG_ON
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );
    #endif

    // GPU kernel
    __kernel_im2col<<<dim_blocks_im2col, dim_threads_im2col>>> (d_input, d_col, BATCH_NUM, INPUT_C,INPUT_H,INPUT_W, FILTER_H,FILTER_W, PAD_H,PAD_W, STRIDE_H,STRIDE_W, OUTPUT_C,OUTPUT_H,OUTPUT_W, KERNEL*OUTPUT);
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    #ifdef DEBUG_ON
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" - Im2Col elapsed time: %.3f s\n", msec_total*1e-3);
    #endif


    /***************************************************************************
     * Launch matmul kernel
     ***************************************************************************/

    const dim3 dim_threads_matmul(WARP_SIZE, WARP_SIZE, 1);
    const dim3 dim_blocks_matmul((OUTPUT+dim_threads_matmul.x-1)/dim_threads_matmul.x, (OUTPUT_C+dim_threads_matmul.y-1)/dim_threads_matmul.y, BATCH_NUM);
 
    #ifdef DEBUG_ON
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );
    #endif

    // GPU kernel
    __kernel_conv_matmul<<<dim_blocks_matmul, dim_threads_matmul>>> (d_filter, d_col, d_output, /*BATCH_NUM*/BATCH_NUM, /*M*/OUTPUT_C, /*K*/KERNEL, /*N*/OUTPUT);
    cudaErrChk( cudaMemcpy(output.data(), d_output, sizeof(T)*BATCH_NUM*OUTPUT_C*OUTPUT, cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );

    #ifdef DEBUG_ON
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" - Matmul elapsed time: %.3f s\n", msec_total*1e-3);
    #endif



    /***************************************************************************
     * Finalize
     ***************************************************************************/

    cudaErrChk( cudaFree(d_input) );
    cudaErrChk( cudaFree(d_col) );
    cudaErrChk( cudaFree(d_filter) );
    cudaErrChk( cudaFree(d_output) );
}