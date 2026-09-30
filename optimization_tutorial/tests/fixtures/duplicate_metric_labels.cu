/**
 * @id        107
 * @name      Duplicate Metric Labels
 * @benefit   Two metrics share the same column label.
 * @strategy  Testing duplicate label detection.
 * @algorithm None
 * @before    N/A
 * @after     N/A
 * @kernel_before dup_label_before_kernel
 * @kernel_after  dup_label_after_kernel
 *
 * @metric dram__bytes_read.sum | DRAMRead | down | Reads from global memory
 * @metric dram__bytes_written.sum | DRAMRead | up | Writes to global memory
 */

__global__ void dup_label_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void dup_label_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
