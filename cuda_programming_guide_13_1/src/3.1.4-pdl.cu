/*
 * Programmatic Dependent Launch (PDL) Example
 * 
 * This program demonstrates CUDA's Programmatic Dependent Launch feature, which allows
 * fine-grained control over when a secondary kernel can begin executing relative to
 * the primary kernel's progress.
 * 
 * Key Concepts:
 * 
 * 1. Primary Kernel:
 *    - Performs heavy computation and writes critical results to output array
 *    - Calls cudaTriggerProgrammaticLaunchCompletion() to signal that critical data is ready
 *    - Continues with additional non-critical work that can overlap with secondary kernel
 * 
 * 2. Secondary Kernel:
 *    - Performs independent work that doesn't require primary kernel's output
 *    - Calls cudaGridDependencySynchronize() to wait for primary kernel's completion signal
 *    - After synchronization, safely reads and transforms primary kernel's output
 * 
 * 3. Launch Configuration:
 *    - Secondary kernel launched with cudaLaunchAttributeProgrammaticStreamSerialization
 *    - This attribute enables the dependency tracking between kernels
 * 
 * Benefits:
 * - Reduces launch latency by allowing secondary kernel to be queued earlier
 * - Provides fine-grained dependency management at the kernel level
 * - Ensures data dependencies are respected while minimizing idle time
 * 
 * Note on Concurrent Execution:
 * - Despite testing various configurations (small/large grids, separate streams, 
 *   limited primary blocks), no concurrent execution was observed in profiling
 * - PDL appears to focus on launch ordering and latency reduction rather than 
 *   true kernel overlap on tested hardware (GB10, compute capability 12.1)
 * - The feature is still valuable for expressing dependencies and potentially 
 *   reducing scheduling overhead
 * 
 * Memory Arrays:
 * - input: Initial data for primary kernel
 * - output: Primary kernel writes results here; secondary kernel reads and updates it
 * - dummy: Captures primary kernel's post-trigger work to prevent compiler optimization
 */

#include "cuda_utils.cuh"
#include <cuda/cmath>
#include <cuda_profiler_api.h>

// Device function for simulating heavy computational work
__device__ float busy_work(float input_value, int iterations) {
    float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += sinf(input_value * i) * cosf(input_value + i);
        result = sqrtf(fabsf(result) + 1.0f);
    }
    return result;
}

__global__ void primary_kernel(float* input, float* output, float* dummy, int size) {
    // Initial work that should finish before starting secondary kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Heavy computation: compute-intensive work that produces results        
        // Write output that secondary kernel will use
        output[idx] = busy_work(input[idx], 10000);  // Increased from 1000 to 10000
    }

    // Trigger the secondary kernel - this signals that the critical output is ready
    cudaTriggerProgrammaticLaunchCompletion();

    // Work that can coincide with the secondary kernel
    // This work doesn't touch output array to avoid races with secondary kernel
    if (idx < size) {
        // Additional non-critical work that can overlap with secondary kernel
        // Significantly increased to make the overlap more visible
        float extra = busy_work(input[idx] * 2.0f, 200000);  // Increased from 5000 to 20000
        // Write to dummy array to prevent compiler from optimizing away this work
        dummy[idx] = extra;
    }
}

__global__ void secondary_kernel(float* data, int size)
{
    // Initialization, Independent work, etc.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Independent work that doesn't need primary kernel's output
    float local_computation = 0.0f;
    if (idx < size) {
        local_computation = busy_work((float)idx, 3000);  // Increased from 50 to 3000
    }

    // Will block until all primary kernels the secondary kernel is dependent on have
    // completed and flushed results to global memory
    cudaGridDependencySynchronize();

    // Dependent work - now we can safely use output from primary kernel
    if (idx < size) {
        // Read the output from primary kernel and update it
        float primary_result = data[idx];
        
        // Perform transformation on primary's output
        float updated_result = primary_result * 2.0f + local_computation * 0.001f;
        updated_result = sqrtf(fabsf(updated_result) + 1.0f);
        
        // Write updated result back to data array
        data[idx] = updated_result;
    }
}

// Launch the secondary kernel with the special attribute
int main(int argc, char* argv[]){
  cudaStream_t stream1, stream2;

  // Resource availability study:
  // Use larger problem size but launch primary with VERY LIMITED blocks
  // to ensure secondary kernel can start while primary is still running
  int size = 256 * 1024; // 256K elements
  int threadsPerBlock = 256;
  int numBlocks = cuda::ceil_div(size, threadsPerBlock);
  
  // Launch primary with only 2 blocks per SM (96 blocks total for 48 SMs)
  // Each SM can handle 6 blocks, so this uses only 2/6 = 33% capacity per SM
  // This leaves 4 blocks/SM (67% capacity) available for secondary kernel
  int primaryBlocks = 48 * 2; // 2 blocks per SM = 96 blocks total
  int secondaryBlocks = numBlocks;

  CUDA_CHECK(cudaStreamCreate(&stream1));   // Primary stream
  CUDA_CHECK(cudaStreamCreate(&stream2));   // Secondary stream

  // Allocate and initialize input/output arrays
  float *d_input, *d_output, *d_dummy;
  size_t bytes = size * sizeof(float);
  
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));
  CUDA_CHECK(cudaMalloc(&d_dummy, bytes));  // Dummy array for overlapping work
  
  // Initialize input with some data
  float* h_input = (float*)malloc(bytes);
  for (int i = 0; i < size; i++) {
    h_input[i] = static_cast<float>(i % 100) / 100.0f;
  }
  CUDA_CHECK(cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream1));

  // Set Up the attribute
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  // Set the attribute in a kernel launch configuration
  cudaLaunchConfig_t config = {0};

  // Base launch configuration for secondary kernel
  config.gridDim = secondaryBlocks; // Full grid for secondary
  config.blockDim = threadsPerBlock;
  config.dynamicSmemBytes= 0;
  config.stream = stream2;  // Use separate stream for potential concurrent execution

  // Add special attribute for PDL
  config.attrs = attribute;
  config.numAttrs = 1;

  printf("Launching primary kernel with %d blocks (out of %d total), %d threads per block\n", 
         primaryBlocks, numBlocks, threadsPerBlock);
  printf("Primary uses 2 blocks/SM (33%% capacity), leaving 4 blocks/SM (67%% capacity) for secondary\n");
  printf("Secondary kernel: %d blocks total (requires ~6 waves across 48 SMs)\n", secondaryBlocks);
  printf("Expected: ~192 secondary blocks (4/SM × 48 SMs) can overlap with primary in first wave\n");
  
  // Start profiling here
  CUDA_CHECK(cudaProfilerStart());
  
  // Launch primary kernel with LIMITED blocks to allow concurrent execution
  primary_kernel<<<primaryBlocks, threadsPerBlock, 0, stream1>>>(d_input, d_output, d_dummy, size);

  printf("Launching secondary kernel (dependent on primary)\n");
  
  // Launch secondary (dependent) kernel using the configuration with
  // the attribute - only needs output array from primary
  cudaLaunchKernelEx(&config, secondary_kernel, d_output, size);
  
  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));
  
  // Stop profiling here
  CUDA_CHECK(cudaProfilerStop());
  
  // Copy results back and verify
  float* h_output = (float*)malloc(bytes);
  CUDA_CHECK(cudaMemcpyAsync(h_output, d_output, bytes, cudaMemcpyDeviceToHost, stream2));
  CUDA_CHECK(cudaStreamSynchronize(stream2));
  
  printf("First 10 results:\n");
  for (int i = 0; i < 10; i++) {
    printf("  output[%d] = %.6f\n", i, h_output[i]);
  }
  
  // Cleanup
  free(h_input);
  free(h_output);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_dummy));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  
  printf("Programmatic Dependent Launch completed successfully\n");
  return 0;
}
