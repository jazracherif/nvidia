/**
 * @id        98
 * @name      Valid Maximal
 * @benefit   A comprehensive example with all optional tags populated.
 * @strategy  Uses tiling and vector loads for maximum throughput.
 * @algorithm Row-major traversal with coalesced accesses.
 * @before    Naive element-wise kernel without any optimization.
 * @after     Tiled copy into shared memory followed by computation.
 * @kernel_before maximal_before_kernel
 * @kernel_after  maximal_after_kernel
 *
 * @metric dram__bytes_read.sum | DRAMRead | down | Reduced reads via tiling
 * @metric ?lts__throughput.avg.pct_of_peak_sustained_active | L2Pct | up | Better L2 usage
 * @metric fma_peak_active_warps.any | Occupancy | any | Should stay above threshold
 */

__global__ void maximal_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] *= 2.0f;
}

__global__ void maximal_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] *= 2.0f;
}
