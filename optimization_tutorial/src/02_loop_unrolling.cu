/**
 * @id        2
 * @name      Loop Unrolling
 * @benefit   Removes loop-counter arithmetic and branches, and lets the compiler
 *            promote a dynamically indexed local array into registers.
 * @strategy  Give the loop a compile-time trip count and mark it #pragma unroll,
 *            so every index into localArr becomes a constant.
 * @algorithm Element-wise scale out[i] = in[i] * 2.0f over 1M floats, with each
 *            thread staging its 4 elements through a local array.
 * @before    #pragma unroll 1 forbids unrolling, so localArr keeps a runtime index
 *            and must live in local memory, which is backed by DRAM.
 * @after     #pragma unroll makes all four indices compile-time constants, so
 *            localArr is held entirely in registers and local traffic disappears.
 * @kernel_before loop_unrolling_before_kernel
 * @kernel_after  loop_unrolling_after_kernel
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric smsp__inst_executed.sum | Instructions | down | Total instructions — unrolling removes the counter increment and branch test per iteration
 * @metric l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum | LocalLd(B) | down | Local-memory read bytes — non-zero means localArr spilled to the DRAM-backed stack; zero means it lives in registers
 * @metric l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum | LocalSt(B) | down | Local-memory write bytes — should reach zero once the array is promoted to registers
 * @metric smsp__sass_average_branch_targets_threads_uniform.pct | BranchUniform% | up | Branch uniformity — the unrolled body has no loop branch left to diverge on
 */

#include "common.cuh"

using namespace opt;

// BEFORE: unrolling suppressed, so localArr is indexed at runtime.
__global__ void loop_unrolling_before_kernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 4 + 3 < n) {
        float localArr[4];
        #pragma unroll 1
        for (int i = 0; i < 4; ++i) {
            localArr[i] = in[tid * 4 + i];
            out[tid * 4 + i] = localArr[i] * 2.0f;
        }
    }
}

void launch_loop_unrolling_before(const float* d_in, float* d_out, int n) {
    int threads = 256;
    int blocks  = ((n / 4) + threads - 1) / threads;
    loop_unrolling_before_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

// AFTER: fully unrolled, so every localArr index is a constant.
__global__ void loop_unrolling_after_kernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid * 4 + 3 < n) {
        float localArr[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            localArr[i] = in[tid * 4 + i];
            out[tid * 4 + i] = localArr[i] * 2.0f;
        }
    }
}

void launch_loop_unrolling_after(const float* d_in, float* d_out, int n) {
    int threads = 256;
    int blocks  = ((n / 4) + threads - 1) / threads;
    loop_unrolling_after_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<float> d_in(N), d_out(N);
    d_in.upload(patternFloats(N));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_out.zero();
        launch_loop_unrolling_before(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_out.download();

        d_out.zero();
        launch_loop_unrolling_after(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_out.download());
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_loop_unrolling_before(d_in, d_out, N); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_loop_unrolling_after(d_in, d_out, N); });

    return report(mode, "2. Loop Unrolling", tb, ta, verified);
}
