/**
 * @id        10
 * @name      Warp Primitives
 * @benefit   Removes block-wide barriers and shared-memory staging from a reduction
 *            by exchanging values directly between registers within a warp.
 * @strategy  Use __shfl_down_sync to reduce inside each warp, then combine the small
 *            number of per-warp partials with a single barrier.
 * @algorithm Sum reduction of 1M floats into one scalar, 256 threads per block.
 * @before    Classic shared-memory tree reduction: 8 halving steps, each separated
 *            by a __syncthreads() that stalls the whole block.
 * @after     Each warp reduces its 32 lanes in registers with 5 shuffles, writes one
 *            partial to shared memory, and one warp combines the 8 partials.
 * @kernel_before warp_primitives_before_kernel
 * @kernel_after  warp_primitives_after_kernel
 *
 * The two variants sum in different orders, so the scalar results agree only to
 * floating-point tolerance.
 *
 * @metric gpu__time_duration.sum | Duration | down | Kernel wall time — the primary speedup signal
 * @metric l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum | ShmemLdWaves | down | Shared load wavefronts — the shuffle version reads shared memory only for the 8 warp partials
 * @metric l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum | ShmemStWaves | down | Shared store wavefronts — 256 stores per block become 8
 * @metric smsp__inst_executed.sum | Instructions | down | Total instructions — 8 barriers and their surrounding loads and stores are replaced by 5 shuffles
 * @metric sm__throughput.avg.pct_of_peak_sustained_elapsed | SM% | up | SM utilization — without barrier stalls warps issue more continuously
 * @metric smsp__warps_active.avg.pct_of_peak_sustained_active | Occupancy% | any | Achieved occupancy — the shuffle version needs less shared memory, which can raise occupancy
 */

#include "common.cuh"

using namespace opt;

constexpr int THREADS = 256;

// BEFORE: shared-memory tree reduction with a barrier per halving step.
__global__ void warp_primitives_before_kernel(const float* in, float* out) {
    __shared__ float sData[THREADS];
    int tid = threadIdx.x;
    sData[tid] = in[blockIdx.x * blockDim.x + tid];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sData[tid] += sData[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out, sData[0]);
    }
}

void launch_warp_primitives_before(const float* d_in, float* d_out, int n) {
    warp_primitives_before_kernel<<<n / THREADS, THREADS>>>(d_in, d_out);
}

// AFTER: register-level warp shuffle reduction, one barrier for the warp partials.
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_primitives_after_kernel(const float* in, float* out) {
    __shared__ float warpSums[THREADS / 32];
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warpId = tid >> 5;

    float val = in[blockIdx.x * blockDim.x + tid];
    val = warpReduceSum(val);

    if (lane == 0) warpSums[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        float sum = (tid < (blockDim.x / 32)) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (tid == 0) atomicAdd(out, sum);
    }
}

void launch_warp_primitives_after(const float* d_in, float* d_out, int n) {
    warp_primitives_after_kernel<<<n / THREADS, THREADS>>>(d_in, d_out);
}

int main(int argc, char** argv) {
    const Mode mode = parseMode(argc, argv);

    DeviceArray<float> d_in(N), d_sum(1);
    d_in.upload(patternFloats(N));

    bool verified = true;
    if (wantsVerify(mode)) {
        d_sum.zero();
        launch_warp_primitives_before(d_in, d_sum, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::vector<float> ref = d_sum.download();

        d_sum.zero();
        launch_warp_primitives_after(d_in, d_sum, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Different summation order over 1M terms; compare relatively.
        verified = allClose(d_sum.download(), ref, 1e-4f, 1.0f);
    }

    // The accumulator is not reset between timed iterations; it only grows and is
    // never read, and resetting would time the memset instead of the kernel.
    float tb = 0.f, ta = 0.f;
    if (wantsBefore(mode))
        tb = timeKernelMs(NITER, [&] { launch_warp_primitives_before(d_in, d_sum, N); });
    if (wantsAfter(mode))
        ta = timeKernelMs(NITER, [&] { launch_warp_primitives_after(d_in, d_sum, N); });

    return report(mode, "10. Warp Primitives", tb, ta, verified);
}
