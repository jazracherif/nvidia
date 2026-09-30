/**
 * @id        abc
 * @name      Non-Integer ID
 * @benefit   Tests parsing when @id is not a pure integer.
 * @strategy  The parser requires @id to be all digits.
 * @algorithm None
 * @before    N/A
 * @after     N/A
 * @kernel_before non_int_id_before_kernel
 * @kernel_after  non_int_id_after_kernel
 *
 * @metric dram__bytes_read.sum | DRAMRead | down | Reduced reads
 */

__global__ void non_int_id_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void non_int_id_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
