/**
 * Shared scaffolding for the per-optimization benchmarks in src/.
 *
 * Each src/NN_name.cu is a standalone program built into build/NN_name.
 * It takes one argument selecting what to run:
 *
 *   both    (default)  time both variants, verify they agree, print a row
 *   row                same as both, used by `make run` for the summary table
 *   before             time only the unoptimized variant   (used by ncu)
 *   after              time only the optimized variant     (used by ncu)
 *   header             print the summary table header and exit
 *
 * Running a single variant is what lets Nsight Compute profile with a fixed
 * --launch-skip 1: timeKernelMs always issues exactly one warmup launch before
 * the timed loop, so the first timed launch is always launch index 1.
 */
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << static_cast<int>(err_) << " \""          \
                      << cudaGetErrorString(err_) << "\"\n";                  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

namespace opt {

// Timed iterations per variant.  Kernels that are too short to time reliably
// override this locally.
constexpr int NITER = 100;

// Default problem size: 1M floats, also 1024x1024 for the matrix kernels.
constexpr int N = 1 << 20;

__device__ inline float dummyTaskA(float v)   { return v * 2.0f + 1.0f; }
__device__ inline float dummyTaskB(float v)   { return v * 0.5f - 1.0f; }
__device__ inline float dummyProcess(float v) { return v * v + 3.0f; }

// ---------------------------------------------------------------------------
// Mode selection
// ---------------------------------------------------------------------------

enum class Mode { Both, Row, Before, After };

inline void printHeader() {
    std::cout << std::left << std::setw(28) << "Optimization"
              << std::right << std::setw(13) << "Before(ms)"
              << std::setw(13) << "After(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(9)  << "Verify" << "\n"
              << std::string(73, '-') << "\n";
}

// Exits directly for the "header" pseudo-mode so `make run` can print the
// table header without a dedicated binary.
inline Mode parseMode(int argc, char** argv) {
    const std::string a = (argc > 1) ? argv[1] : "both";
    if (a == "both")   return Mode::Both;
    if (a == "row")    return Mode::Row;
    if (a == "before") return Mode::Before;
    if (a == "after")  return Mode::After;
    if (a == "header") { printHeader(); std::exit(EXIT_SUCCESS); }
    std::cerr << "usage: " << argv[0]
              << " [both|row|before|after|header]\n";
    std::exit(EXIT_FAILURE);
}

inline bool wantsBefore(Mode m) { return m != Mode::After;  }
inline bool wantsAfter (Mode m) { return m != Mode::Before; }
// Verification launches both kernels, so it must be skipped when profiling a
// single variant or it would perturb ncu's launch indexing.
inline bool wantsVerify(Mode m) { return m == Mode::Both || m == Mode::Row; }

// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    ~DeviceArray() { cudaFree(ptr_); }

    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* get() const     { return ptr_; }
    operator T*() const { return ptr_; }
    size_t size() const { return count_; }

    void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T))); }

    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr_, host.data(), host.size() * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    std::vector<T> download() const {
        std::vector<T> host(count_);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr_, count_ * sizeof(T),
                              cudaMemcpyDeviceToHost));
        return host;
    }

private:
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// Deterministic inputs, so verification is reproducible run to run
// ---------------------------------------------------------------------------

inline std::vector<float> patternFloats(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u;
        v[i] = 0.5f + static_cast<float>(h % 1000u) * 0.001f;
    }
    return v;
}

// Skewed towards bin 0 so the histogram kernels see realistic atomic contention.
inline std::vector<int> patternBins(size_t n, int bins) {
    std::vector<int> v(n);
    for (size_t i = 0; i < n; ++i) {
        uint32_t h = static_cast<uint32_t>(i) * 2654435761u;
        v[i] = (i % 4 == 0) ? 0 : static_cast<int>(h % static_cast<uint32_t>(bins));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

// One warmup launch, then nIter timed launches; returns average milliseconds.
inline float timeKernelMs(int nIter, const std::function<void()>& fn) {
    fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < nIter; ++i) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / static_cast<float>(nIter);
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

template <typename T>
inline bool allEqual(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        std::cerr << "  size mismatch: " << a.size() << " vs " << b.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "  mismatch at " << i << ": " << a[i] << " vs " << b[i] << "\n";
            return false;
        }
    }
    return true;
}

// Tiled and register-blocked kernels sum in a different order than the naive
// version, so float results agree only to a tolerance.
inline bool allClose(const std::vector<float>& a, const std::vector<float>& b,
                     float rtol = 1e-4f, float atol = 1e-4f) {
    if (a.size() != b.size()) {
        std::cerr << "  size mismatch: " << a.size() << " vs " << b.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > atol + rtol * std::fabs(b[i])) {
            std::cerr << "  mismatch at " << i << ": " << a[i] << " vs " << b[i]
                      << " (diff " << diff << ")\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

inline int report(Mode mode, const char* name,
                  float beforeMs, float afterMs, bool verified) {
    if (mode == Mode::Before) {
        std::cout << name << " before: " << beforeMs << " ms\n";
        return EXIT_SUCCESS;
    }
    if (mode == Mode::After) {
        std::cout << name << " after:  " << afterMs << " ms\n";
        return EXIT_SUCCESS;
    }

    std::cout << std::left << std::setw(28) << name
              << std::right << std::fixed << std::setprecision(4)
              << std::setw(13) << beforeMs
              << std::setw(13) << afterMs
              << std::setprecision(2) << std::setw(9) << (beforeMs / afterMs) << "x"
              << std::setw(9) << (verified ? "ok" : "FAIL") << "\n";
    return verified ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace opt
