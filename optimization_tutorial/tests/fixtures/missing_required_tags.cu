/**
 * @id        102
 * @name      Missing Required Tags
 * @strategy  Testing what happens when required tags are absent.
 * @algorithm None
 * @before    N/A
 * @after     N/A
 * @metric dram__bytes_read.sum | DRAMRead | down | Reduced reads
 */

__global__ void missing_required_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void missing_required_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
