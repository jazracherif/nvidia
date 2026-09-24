/**
 * @id        14
 * @name      Reduction Pattern
 * @benefit   Improves warp efficiency by aligning thread work patterns with memory access patterns.
 * @strategy  Compare two reduction approaches: one where threads process elements at stride,
 *            and another where threads process consecutive elements in a hierarchical reduction.
 * @algorithm Hierarchical reduction of 1M floats using different thread assignment patterns.
 * @before    Threads are assigned to every other element, with stride doubling each iteration.
 *            Only threads at the current stride value participate, leading to poor warp efficiency.
 * @after     Threads are aligned to consecutive elements, reducing upper half in each step
 *            with block size 2x the problem size, improving warp efficiency and instruction throughput.
 * @kernel_before reduction_pattern_before
 * @kernel_after  reduction_pattern_after
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric smsp__average_thread_inst_executed_per_inst_executed.ratio | WarpEfficiency Ratio | up | Average threads running per warp instruction issued  — should be better in the after version
 * @metric smsp__average_thread_inst_executed_per_inst_executed.pct | WarpEfficiency Pct | up | Average threads running per warp instruction issued as a ratio of the maximum possible number of active threads  — should be better in the after version
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM pipeline utilization — better pattern should improve overall throughput
 */

#include "common.cuh"
#include <cuda/atomic>

using namespace opt;

constexpr int BLOCK_SIZE = 256;

// BEFORE: Stride-based approach - threads process every other element, stride doubles each iteration
__global__ void reduction_pattern_before(float* input, float* result, int N) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    __shared__ float input_s[2*BLOCK_SIZE]; // thread 0-255
    // copy to shared memory, we have consecutive values of block element
    input_s[threadIdx.x] = (i < N) ? input[i] : 0.f;
    input_s[threadIdx.x + BLOCK_SIZE] = (i + BLOCK_SIZE < N) ? input[i + BLOCK_SIZE] : 0.f;

    unsigned int tid = 2 * threadIdx.x;
    // Each thread processes elements at stride positions
    for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2){
        __syncthreads();
        if (threadIdx.x % stride == 0) {            
            input_s[tid] += input_s[tid + stride]; 
        }
    }
    
    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*result);
        output_ref.fetch_add(input_s[0], cuda::memory_order_relaxed);
    }
}


// AFTER: Consecutive element approach - threads work on consecutive elements in each step
__global__ void reduction_pattern_after(float* input, float* result, int N) {
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    __shared__ float input_s[2*BLOCK_SIZE]; // thread 0-255
    // copy to shared memory, we have consecutive values of block element
    input_s[threadIdx.x] = (i < N) ? input[i] : 0.f;
    input_s[threadIdx.x + BLOCK_SIZE] = (i + BLOCK_SIZE < N) ? input[i + BLOCK_SIZE] : 0.f;

    // Each thread processes elements at stride positions
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];;
        }
    }
    
    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*result);
        output_ref.fetch_add(input_s[0], cuda::memory_order_relaxed);
    }
}


void launch_reduction_pattern_before(float* d_data, int N, float* d_result) {
    dim3 block(BLOCK_SIZE);
    // Ensure total threads = N/2: grid.x = ceil((N/2) / 256)
    int blocks = (N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
    reduction_pattern_before<<<blocks, block>>>(d_data, d_result, N);
}

void launch_reduction_pattern_after(float* d_data, int N, float* d_result) {
    dim3 block(BLOCK_SIZE);
    // Ensure total threads = N/2: grid.x = ceil((N/2) / 256)
    int blocks = (N + 2 * BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
    reduction_pattern_after<<<blocks, block>>>(d_data, d_result, N);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);
    
    int N_ = 1000 * N;
    DeviceArray<float> d_data(N_), d_result(1);
    d_data.upload(patternFloats(N_));

    bool verified = true;
    if (wantsVerify(mode)) {
        // Verify correctness by running both and comparing results
        d_result.zero();
        launch_reduction_pattern_before(d_data, N_, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());
        const float ref_before = d_result.download()[0];

        d_result.zero();
        launch_reduction_pattern_after(d_data, N_, d_result);
        CUDA_CHECK(cudaDeviceSynchronize());
        const float ref_after = d_result.download()[0];

        // printf("ref_before: %f, ref_after: %f\n", ref_before, ref_after);
        // Allow small floating point differences due to different reduction order
        verified = std::abs(ref_before - ref_after) < 1e-2f * (std::abs(ref_before) + std::abs(ref_after) + 1e-6f);
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_reduction_pattern_before(d_data, N_, d_result); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_reduction_pattern_after(d_data, N_, d_result); });

    return report(mode, "14. Reduction Pattern", tb, ta, verified);
}