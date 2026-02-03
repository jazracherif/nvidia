#include <cuda_runtime.h>
#include <cuda/cmath>
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.h"


__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}


void initArray(float* A, int vectorLength){
  for (int i=0; i< vectorLength; i++ ){
    A[i] = 1;
  }
}

void serialVecAdd(float* A, float* B, float* comparisonResult, int vectorLength){
  for (int i=0; i< vectorLength; i++ ){
    comparisonResult[i] = A[i] + B[i];
  }
}

bool vectorApproximatelyEqual(float* C, float* comparisonResult, int vectorLength){
  for (int i=0; i< vectorLength; i++ ){
    if(fabs(comparisonResult[i] - C[i]) > 1e-5)
      return false;
  }
  return true;
}

void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    CUDA_CHECK(cudaMallocManaged(&A, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&B, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&C, vectorLength*sizeof(float)));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    CUDA_CHECK(cudaGetLastError());
    // Wait for the kernel to complete execution
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    cudaFree(C);
    free(comparisonResult);

}

void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));
    
    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    CUDA_CHECK(cudaMallocHost(&A, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&B, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&C, vectorLength*sizeof(float)));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));

    // Copy data to the GPU
    CUDA_CHECK(cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(devC, 0, vectorLength*sizeof(float)));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    CUDA_CHECK(cudaGetLastError());

    // wait for kernel execution to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault));

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devC));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(B));
    CUDA_CHECK(cudaFreeHost(C));
    free(comparisonResult);
}

int main(int argc, char* argv[]){
  unifiedMemExample(1024);

  explicitMemExample(1024);
}