#pragma once
#include <cstdio>

#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
       fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}


template <typename T> void check_result(const std::vector<T>& output, const std::vector<T>& gt) {

    for (size_t i=0; i<gt.size(); i++) {
        if (output[i] != gt[i]) {
            std::cout << " - [[[ERR]]] Checking result failed !!! output["<<i<<"] = "<<output[i]<<" != gt("<<gt[i]<<")" << std::endl;
            return;
        }
    }
    std::cout << " - Checking result succeeded !!!" << std::endl;

}

