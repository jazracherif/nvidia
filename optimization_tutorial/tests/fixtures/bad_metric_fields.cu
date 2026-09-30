/**
 * @id        103
 * @name      Bad Metric Fields
 * @benefit   Tests parsing with wrong number of pipe-separated fields in @metric.
 * @strategy  Testing malformed metric detection.
 * @algorithm None
 * @before    N/A
 * @after     N/A
 * @kernel_before bad_metric_before_kernel
 * @kernel_after  bad_metric_after_kernel
 *
 * @metric dram__bytes_read.sum | DRAMRead | up
 */

__global__ void bad_metric_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void bad_metric_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
