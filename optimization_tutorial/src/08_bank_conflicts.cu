/**
 * @id        8
 * @name      Bank Conflicts
 * @benefit   Removes serialized replays when many lanes of a warp address the same
 *            shared-memory bank.
 * @strategy  Pad the row stride of the 2D shared array by one element, which skews
 *            each row into a different bank and turns a column walk into a
 *            conflict-free stride.
 * @algorithm 32x32 matrix transpose through shared memory in a single block:
 *            sData is written row-wise and read column-wise.
 * @before    sData[32][32] — a column access has stride 32, so all 32 lanes land in
 *            the same bank and the access replays 32 times.
 * @after     sData[32][33] — the +1 pad makes bank = (row * 33) % 32 distinct for
 *            every row, so the same column access is conflict free.
 * @kernel_before bank_conflicts_before_kernel
 * @kernel_after  bank_conflicts_after_kernel
 *
 * This kernel is tiny, so it runs many more iterations than the others to time
 * reliably. Shared-memory bytes moved are identical; only the replay count differs.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum | ShmemLdConflicts | down | Shared load bank conflicts — counts the extra cycles a 32-way conflict costs; should reach 0 after padding
 * @metric l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum | ShmemLdWaves | down | Shared load wavefronts — every extra wavefront is one serialized replay
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM utilization — replays stall the shared-memory pipeline and throttle the SM
 */

#include "common.cuh"

using namespace opt;

// Single tiny block; needs far more iterations than NITER to time reliably.
constexpr int BANK_NITER = 10000;
constexpr int DIM        = 32;

// BEFORE: unpadded stride, so a column access is a 32-way bank conflict.
__global__ void bank_conflicts_before_kernel(float* out) {
    __shared__ float sData[DIM][DIM];
    sData[threadIdx.x][threadIdx.y] = threadIdx.x + threadIdx.y;
    __syncthreads();

    out[threadIdx.x * DIM + threadIdx.y] = sData[threadIdx.y][threadIdx.x];
}

void launch_bank_conflicts_before(float* d_out) {
    dim3 block(DIM, DIM);
    bank_conflicts_before_kernel<<<1, block>>>(d_out);
}

// AFTER: +1 padding skews rows across banks, removing the conflict.
__global__ void bank_conflicts_after_kernel(float* out) {
    __shared__ float sData[DIM][DIM + 1];
    sData[threadIdx.x][threadIdx.y] = threadIdx.x + threadIdx.y;
    __syncthreads();

    out[threadIdx.x * DIM + threadIdx.y] = sData[threadIdx.y][threadIdx.x];
}

void launch_bank_conflicts_after(float* d_out) {
    dim3 block(DIM, DIM);
    bank_conflicts_after_kernel<<<1, block>>>(d_out);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<float> d_out(DIM * DIM);

    bool verified = true;
    if (wantsVerify(mode)) {
        d_out.zero();
        launch_bank_conflicts_before(d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_out.download();

        d_out.zero();
        launch_bank_conflicts_after(d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        verified = allEqual(ref, d_out.download());
    }

    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(BANK_NITER, [&] { launch_bank_conflicts_before(d_out); });
    if (wantsAfter(mode))
        ta = timeKernelMs(BANK_NITER, [&] { launch_bank_conflicts_after(d_out); });

    return report(mode, "8. Bank Conflicts", tb, ta, verified);
}
