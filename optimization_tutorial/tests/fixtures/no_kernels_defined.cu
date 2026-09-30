/**
 * @id        106
 * @name      No Kernels Defined
 * @benefit   Header references kernels that do not exist in the source.
 * @strategy  Testing detection of missing __global__ kernel definitions.
 * @algorithm None
 * @before    N/A
 * @after     N/A
 * @kernel_before ghost_kernel_before
 * @kernel_after  nonexistent_kernel_after
 *
 * @metric dram__bytes_read.sum | DRAMRead | down | Reduced reads
 */

__global__ void some_other_before(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void some_other_after(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
