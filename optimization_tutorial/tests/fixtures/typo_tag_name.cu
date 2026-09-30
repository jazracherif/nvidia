/**
 * @id        101
 * @nme         Typo Tag Name
 * @benefit   Tests behavior when a tag name has a typo (nme instead of name).
 * @strategy  Testing malformed tag detection.
 * @algorithm None
 * @before    N/A
 * @after     N/A
 * @kernel_before typo_tag_before_kernel
 * @kernel_after  typo_tag_after_kernel
 *
 * @metric dram__bytes_read.sum | DRAMRead | down | Reduced reads
 */

__global__ void typo_tag_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void typo_tag_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
