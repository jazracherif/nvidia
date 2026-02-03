#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * CUDA_CHECK macro for error checking CUDA API calls
 * 
 * Usage:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 *   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
 * 
 * Prints error message to stderr if the CUDA call fails, including:
 * - File name and line number
 * - Error code
 * - Error description
 */
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

#endif // CUDA_UTILS_CUH
