/**
 * @id        97
 * @name      Valid Multiline Tags
 * @benefit   Tests that multiline continuation lines work correctly.
 * @strategy  This is a long description that continues over multiple lines
 *            to test the continuation parsing logic in _split_tags.
 * @algorithm Simple reduction pattern
 * @before    No optimization, naive loop
 * @after     Optimized with shared memory tiling
 * @kernel_before multiline_before_kernel
 * @kernel_after  multiline_after_kernel
 *
 * @metric seal__throughput.avg.pct_of_peak_sustained_active | SEALPct | up | Seal throughput improved
 */

__global__ void multiline_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void multiline_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
