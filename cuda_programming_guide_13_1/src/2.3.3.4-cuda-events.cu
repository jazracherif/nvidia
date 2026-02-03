#include <cuda_runtime.h>
#include <cuda/cmath>
#include <unistd.h>
#include "cuda_utils.cuh"

__global__ void vecInit(float* A, int value, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

     if(workIndex < vectorLength)
     {
         // Perform computation
         A[workIndex] = value;
     }
}

__global__ void vecMul(float* A, int value, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

     if(workIndex < vectorLength)
     {
         // Perform computation
         A[workIndex] *= value;
     }
}

__global__ void vecInitRandom(float* A, int maxValue, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

     if(workIndex < vectorLength)
     {
         // Simple pseudo-random number generator
         unsigned int seed = workIndex * 1103515245 + 12345;
         seed = (seed / 65536) % 32768;
         // Generate random value between 1 and maxValue
         A[workIndex] = 1.0f + (seed % maxValue);
     }
}

__global__ void computeIntensiveKernel(float* data, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (idx < length) {
        float value = data[idx];
        
        // Perform significant computational work
        // Mathematical operations that take time
        for (int i = 0; i < 100; i++) {
            value = sinf(value) * cosf(value);
            value = sqrtf(fabsf(value) + 0.1f);
            value = expf(value * 0.01f);
            value = logf(fabsf(value) + 1.0f);
            value = powf(value, 0.95f);
        }
        
        // Write result back
        data[idx] = value;
    }
}
bool allCPUWorkDone(int work_left){
  printf("work_left %d\n", work_left);
  if (work_left == 0)
    return true;
  return false;

}

void doNextChunkOfCPUWork(int &work_left){
  if (work_left == 0)
    return;

  sleep(0.1);
  work_left--;
}

int main(int argc, char* argv[])
{
  cudaEvent_t event;
  cudaStream_t stream1;
  cudaStream_t stream2;

  int work_left = 5; // 5 iteration of CPU work
  size_t size = 1024 * 1024;
  float *d_data, *d_data2;

  // Create some data
  CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_data2, size * sizeof(float)));

  // float *h_data = (float *)malloc(size);
  float *h_data = NULL;
  CUDA_CHECK(cudaMallocHost(&h_data, size * sizeof(float)));

  // create the streams
  CUDA_CHECK(cudaStreamCreate(&stream1));   // Processing stream
  CUDA_CHECK(cudaStreamCreate(&stream2));   // Copying stream
  bool copyStarted = false;

  //  create the event
  CUDA_CHECK(cudaEventCreate(&event));

  int threadsPerBlock = 256;
  int numBlocks = cuda::ceil_div(size, threadsPerBlock);

  // launch kernel1 into the stream
  vecInit<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_data, 0, size);
  CUDA_CHECK(cudaGetLastError());
  // enqueue an event following kernel1
  CUDA_CHECK(cudaEventRecord(event, stream1));

  // launch kernel2 into the stream
  vecInitRandom<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_data2, 100, size);
  CUDA_CHECK(cudaGetLastError());

  computeIntensiveKernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_data2, size);
  CUDA_CHECK(cudaGetLastError());

  // while the kernels are running do some work on the CPU
  // but check if kernel1 has completed because then we will start
  // a device to host copy in stream2
  while ( !allCPUWorkDone(work_left) || !copyStarted ) {
      doNextChunkOfCPUWork(work_left);

      // peek to see if kernel 1 has completed
      // if so enqueue a non-blocking copy into stream2
      if ( !copyStarted ) {
          if( cudaEventQuery(event) == cudaSuccess ) {
              printf("start async copy\n");
              CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost, stream2));
              copyStarted = true;
          }
      }
  }

  // wait for both streams to be done
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));

  // destroy the event
  CUDA_CHECK(cudaEventDestroy(event));

  // destroy the streams and free the data
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFree(d_data2));
  CUDA_CHECK(cudaFreeHost(h_data));
}