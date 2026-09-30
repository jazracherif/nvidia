 // @id        100
// @name      No Header Delimiters
// @benefit   This file deliberately has no block comment delimiters.
// @strategy  Testing what happens when the header regex finds nothing.
// @algorithm None
// @before    N/A
// @after     N/A
// @kernel_before no_header_before_kernel
// @kernel_after  no_header_after_kernel
//
// @metric dram__bytes_read.sum | DRAMRead | down | Reduced reads

__global__ void no_header_before_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void no_header_after_kernel(float *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) data[idx] += 2.0f;
}
