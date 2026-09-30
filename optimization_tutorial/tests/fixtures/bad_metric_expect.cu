/**
 * @id        104
 * @name      Invalid Metric Expect
 * @benefit   Tests parsing with an invalid value in the expect field.
 * @strategy  Testing that only up/down/any are accepted.
 * @algorithm None
 * @before    N/A
 * @after     N/A
 * @kernel_before bad_expect_before_kernel
 * @kernel_after  bad_expect_after_kernel
 *
 * @metric dram__bytes_read.sum | DRAMRead | maybe | Should have been "up"
 */

__global__ void bad_expect_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void bad_expect_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
