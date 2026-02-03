
#include "cuda_utils.h"

__global__ void primary_kernel() {
    // Initial work that should finish before starting secondary kernel

    // Trigger the secondary kernel
    cudaTriggerProgrammaticLaunchCompletion();

    // Work that can coincide with the secondary kernel
}

__global__ void secondary_kernel()
{
    // Initialization, Independent work, etc.

    // Will block until all primary kernels the secondary kernel is dependent on have
    // completed and flushed results to global memory
    cudaGridDependencySynchronize();

    // Dependent work
}

// Launch the secondary kernel with the special attribute
main(){
  cudaStream_t stream;

  int threadsPerBlock = 256;
  int numBlocks = cuda::ceil_div(size, threadsPerBlock);

  CUDA_CHECK(cudaStreamCreate(&stream));   // Processing stream

  // Set Up the attribute
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  // Set the attribute in a kernel launch configuration
  cudaLaunchConfig_t config = {0};

  // Base launch configuration
  config.gridDim = numBlocks;
  config.blockDim = threadsPerBlock;
  config.dynamicSmemBytes= 0;
  config.stream = stream;

  // Add special attribute for PDL
  config.attrs = attribute;
  config.numAttrs = 1;

  // Launch primary kernel
  primary_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>();

  // Launch secondary (dependent) kernel using the configuration with
  // the attribute
  cudaLaunchKernelEx(&config, secondary_kernel);
  cudaDeviceSynchronize();
}
