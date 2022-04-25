/**************************************************************************************
 * CUDA Convolution Operation Example
 * - naive implementation of Conv
 * - Im2Col implementation of Conv
 ***************************************************************************************/

#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cstdio>

#include <helper.cuh>
#include <conv_cpu.cuh>
#include <conv_gpu_naive.cuh>
#include <conv_gpu_im2col.cuh>

/***************************************************************************
 * Problem configuration
 ***************************************************************************/

#define DTYPE float

const int BATCH_NUM = 2;
const int INPUT_H = 25;
const int INPUT_W = 25;
const int INPUT_C = 8;

const int FILTER_H = 3;
const int FILTER_W = 3;

const int PAD_H = 1;
const int PAD_W = 1;

const int STRIDE_H = 1;
const int STRIDE_W = 1;

const int OUTPUT_H = (INPUT_H-FILTER_H+2*PAD_H)/STRIDE_H + 1;
const int OUTPUT_W = (INPUT_W-FILTER_W+2*PAD_W)/STRIDE_W + 1;
const int OUTPUT_C = 3;


int main(void) {

    srand(1);
    printf("===================================================================\n");
    printf("CUDA Convolution Operation Example\n");
    printf(" - Input size: [%d,%d,%d,%d], filter size: [%d,%d,%d,%d], pad: [%d,%d], stride: [%d,%d] -> output size: [%d,%d,%d,%d]\n",
            BATCH_NUM,INPUT_C,INPUT_H,INPUT_W, INPUT_C,OUTPUT_C,FILTER_H,FILTER_W, PAD_H,PAD_W, STRIDE_H,STRIDE_W, BATCH_NUM,OUTPUT_C,OUTPUT_H,OUTPUT_W);
    printf(" - Size of input[%.3fGB], output[%.3fGB]: \n",
            1.0f*sizeof(DTYPE)*BATCH_NUM*INPUT_C*INPUT_W*INPUT_H*1e-9, 1.0f*sizeof(DTYPE)*BATCH_NUM*OUTPUT_C*OUTPUT_W*OUTPUT_H*1e-9);        
    printf(" - Target algorithm: \n");
    printf("    - Naive implementation of Conv\n");
    printf("    - Im2Col implementation of Conv\n");
    printf("===================================================================\n");
    printf("\n");


    /***************************************************************************
     * Data initialization
     ***************************************************************************/

    // Define data
    std::vector<DTYPE> input(BATCH_NUM*INPUT_C*INPUT_W*INPUT_H); // B*C*H*W
    std::vector<DTYPE> filter(INPUT_C*OUTPUT_C*FILTER_W*FILTER_H); // OUT_C*IN_C*H*W
    std::vector<DTYPE> output(BATCH_NUM*OUTPUT_C*OUTPUT_W*OUTPUT_H, 0); // B*C*H*W
    std::vector<DTYPE> gt(BATCH_NUM*OUTPUT_C*OUTPUT_W*OUTPUT_H, 0); // C*H*W

    // Initial with random value
    std::generate(input.begin(), input.end(), [](){return (std::rand()%101-50)/10;});
    std::generate(filter.begin(), filter.end(), [](){return (std::rand()%101-50)/10;});
    
    
    /***************************************************************************
     * Get ground truth via CPU
     ***************************************************************************/
    conv_cpu<DTYPE>(input, filter, gt, BATCH_NUM, INPUT_C,INPUT_H,INPUT_W, FILTER_H,FILTER_W, PAD_H,PAD_W, STRIDE_H,STRIDE_W, OUTPUT_C,OUTPUT_H,OUTPUT_W);


    /***************************************************************************
     * Launch GPU naive implementation
     ***************************************************************************/
    conv_gpu_naive<DTYPE, 16>(input, filter, output, BATCH_NUM, INPUT_C,INPUT_H,INPUT_W, FILTER_H,FILTER_W, PAD_H,PAD_W, STRIDE_H,STRIDE_W, OUTPUT_C,OUTPUT_H,OUTPUT_W);
    #ifdef DEBUG_ON
    check_result(output, gt);
    #endif

    conv_gpu_matmul<DTYPE, 16>(input, filter, output, BATCH_NUM, INPUT_C,INPUT_H,INPUT_W, FILTER_H,FILTER_W, PAD_H,PAD_W, STRIDE_H,STRIDE_W, OUTPUT_C,OUTPUT_H,OUTPUT_W);
    #ifdef DEBUG_ON
    check_result(output, gt);
    #endif

    
    return 0;
}

