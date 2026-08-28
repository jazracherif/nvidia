/**
 * @id        3
 * @name      Control Divergence
 * @benefit   Prevents warp lane serialization, keeping all 32 lanes of a warp
 *            executing the same instruction stream.
 * @strategy  Delete the branch: evaluate both alternatives and select between
 *            them with 0/1 arithmetic weights, which no warp can diverge on.
 * @algorithm Per-element dispatch over 1M floats — even threads apply
 *            dummyTaskA(x) = 2x + 1, odd threads apply dummyTaskB(x) = 0.5x - 1.
 * @before    A branch on threadIdx.x parity splits every warp, so the hardware
 *            runs the taken and not-taken sides one after the other.
 * @after     Both tasks are computed unconditionally and multiplied by a 0/1
 *            weight, so there is no branch and no serialization.
 * @kernel_before control_divergence_before_kernel
 * @kernel_after  control_divergence_after_kernel
 *
 * The branchless form trades more arithmetic for less serialization, so total
 * instruction count is not expected to fall.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric smsp__sass_average_branch_targets_threads_uniform.pct | BranchUniform% | up | Share of branches where all 32 lanes agree — should reach 100% once the branch is gone
 * @metric smsp__inst_executed.sum | Instructions | any | Total instructions — the branchless version deliberately computes both tasks, so this may rise even as time falls
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM utilization — without lane serialization every issued instruction does 32 lanes of useful work
 */

#include "common.cuh"

using namespace opt;

// BEFORE: parity branch splits every warp into two serialized halves.
__global__ void control_divergence_before_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (threadIdx.x % 2 == 0) {
            out[idx] = dummyTaskA(in[idx]);
        } else {
            out[idx] = dummyTaskB(in[idx]);
        }
    }
}

void launch_control_divergence_before(const float* d_in, float* d_out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    control_divergence_before_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

// AFTER: branchless select; multiplying by exact 0.0f/1.0f keeps results bit-identical.
__global__ void control_divergence_after_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float task = (threadIdx.x % 2 == 0) ? 1.0f : 0.0f;
        out[idx] = dummyTaskA(in[idx]) * task + dummyTaskB(in[idx]) * (1.0f - task);
    }
}

void launch_control_divergence_after(const float* d_in, float* d_out, int n) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    control_divergence_after_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<float> d_in(N), d_out(N);
    d_in.upload(patternFloats(N));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_out.zero();
        launch_control_divergence_before(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_out.download();

        d_out.zero();
        launch_control_divergence_after(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_out.download());
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_control_divergence_before(d_in, d_out, N); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_control_divergence_after(d_in, d_out, N); });

    return report(mode, "3. Control Divergence", tb, ta, verified);
}
