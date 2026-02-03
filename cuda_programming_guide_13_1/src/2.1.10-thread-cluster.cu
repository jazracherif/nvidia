
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>
#include "cuda_utils.cuh"


// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel_increment(float *input, int N)
{
  namespace cg = cooperative_groups;

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;
  // printf("clusterBlockRank %d\n",clusterBlockRank);

  if (idx < N){
    input[idx] += 1;
  }

}


/**
 * Fails compilation on GB10 with error
 * 
 * ```
 *  cluster.cu(21): error: __cluster_dims__ is not supported for this GPU architecture
 *  __attribute__((global)) void __attribute__((cluster_dims(2, 1, 1))) cluster_kernel_increment(float *input, int N)
 * ```
 * 
 */
int main(int argc, char* argv[])
{
  int N = 512;
  float* A = NULL;
  CUDA_CHECK(cudaMallocManaged(&A, N*sizeof(float)));

  for (int i=0; i< N; i++)
    A[i] = 1;

  // Kernel invocation with compile time cluster size
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

  // The grid dimension is not affected by cluster launch, and is still enumerated
  // using number of blocks.
  // The grid dimension must be a multiple of cluster size.
  cluster_kernel_increment<<<numBlocks, threadsPerBlock>>>(A, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(A));
}