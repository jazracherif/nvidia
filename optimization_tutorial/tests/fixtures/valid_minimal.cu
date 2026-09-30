/**
 * @id        99
 * @name      Valid Minimal
 * @benefit   Simple valid header for testing.
 * @strategy  Basic kernel pattern.
 * @algorithm Minimal work per thread.
 * @before    No optimization applied.
 * @after     With minimal optimization.
 * @kernel_before valid_minimal_before_kernel
 * @kernel_after  valid_minimal_after_kernel
 *
 * @metric simt_warps_created.sum | WarpsCreated | up | More warps indicates occupancy gain
 */

__global__ void valid_minimal_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void valid_minimal_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
