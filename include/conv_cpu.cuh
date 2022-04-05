#pragma once
#include <vector>
#include <cstdio>
#include <helper.cuh>

template <typename T>
void conv_cpu(
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
    printf("CPU naive implementation launched...\n");

    #ifdef DEBUG_ON
    float msec_total = 0.0f;
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );
    #endif


    for (int batch=0; batch<BATCH_NUM; batch++) {
        for (int out_c=0; out_c<OUTPUT_C; out_c++) {
            for (int out_h=0; out_h<OUTPUT_H; out_h++) {
                for (int out_w=0; out_w<OUTPUT_W; out_w++) {
    
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
        }
    }

    #ifdef DEBUG_ON
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );
    printf(" - Elapsed time: %.3f s\n", msec_total*1e-3);
    #endif

}
