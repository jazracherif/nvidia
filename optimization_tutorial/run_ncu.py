#!/usr/bin/env python3
"""
Nsight Compute (ncu) driver for CUDA Optimizations benchmark.

Runs ncu for each optimization's before/after kernel pair and prints a
side-by-side hardware-metric comparison table.

Usage:
    python run_ncu.py <path/to/cuda_optimizations_binary>

Requirements:
    - ncu >= 2021.1 on PATH  (typically /usr/local/cuda/bin/ncu or
      /opt/nvidia/nsight-compute/<ver>/ncu)
    - The binary must already be compiled:
        nvcc -O3 -arch=native src/cuda_optimizations_examples.cu -o cuda_optimizations

Notes:
    - ncu replays each kernel several times to collect hardware counters,
      so this script takes a few minutes to complete.
    - Optimization 1 (Occupancy Tuning) uses the same kernel name for both
      before and after; the script uses --launch-skip to distinguish them.
      NITER below must match the NITER constant in cuda_optimizations_examples.cu.
"""

import csv
import io
import os
import shutil
import subprocess
import sys
from typing import Optional

# Must match NITER in cuda_optimizations_examples.cu
NITER = 100

# Per-optimization metric groups.
# Each entry: (ncu_perfworks_metric_name, column_label, description)
# description: what the metric measures and why it reveals this optimization's benefit.
OPTIMIZATION_METRICS: dict[str, list[tuple[str, str, str]]] = {
    "1. Occupancy Tuning": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time — primary speedup signal"),
        ("smsp__warps_active.avg.pct_of_peak_sustained_active",
         "Occupancy%",
         "Fraction of max warps active — the direct target: more warps hide memory latency"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM pipeline utilization — rises as higher occupancy keeps execution units busy"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed",
         "DRAM%",
         "DRAM bandwidth utilization — memory-bound kernels benefit most from better warp coverage"),
    ],

    "2. Loop Unrolling": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("smsp__inst_executed.sum",
         "Instructions",
         "Total GPU instructions — unrolling removes loop-counter increments and branch checks"),
        ("l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
         "LocalLd(B)",
         "Local-memory (DRAM stack) read bytes — non-zero = localArr spilled; 0 = promoted to registers"),
        ("l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
         "LocalSt(B)",
         "Local-memory write bytes — should drop to 0 after unroll promotes array to register space"),
        ("smsp__sass_average_branch_targets_threads_uniform.pct",
         "BranchUniform%",
         "Branch uniformity — unrolled loop removes intra-loop branches entirely"),
    ],

    "3. Control Divergence": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("smsp__sass_average_branch_targets_threads_uniform.pct",
         "BranchUniform%",
         "% of branches where all 32 warp threads agree — 100% after branchless arithmetic rewrite"),
        ("smsp__inst_executed.sum",
         "Instructions",
         "Total instructions — branchless form computes both tasks; divergent serializes warp lanes"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — no lane serialization means the full 32-thread warp executes every instruction"),
    ],

    "4. Memory Coalescing": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
         "GlobalLdSectors",
         "128-byte DRAM sectors loaded — uncoalesced warp issues 1 sector/thread; coalesced merges 32 threads into 1"),
        ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
         "GlobalLdReqs",
         "Global load requests — sectors/requests quantifies the coalescing inefficiency factor"),
        ("dram__bytes_read.sum",
         "DRAMRead(B)",
         "Total DRAM bytes fetched — coalesced access eliminates redundant cache-line transfers"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed",
         "DRAM%",
         "DRAM bandwidth utilization — higher means the bus is better used"),
        ("l1tex__t_sector_hit_rate.pct",
         "L1hit%",
         "L1 cache hit rate — coalesced reads improve spatial locality and cache reuse"),
    ],

    "5. Shared Memory Tiling": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("dram__bytes_read.sum",
         "DRAMRead(B)",
         "DRAM bytes read — tiling reuses each TILE_WIDTH×TILE_WIDTH block from SRAM, reducing DRAM reads ~TILE_WIDTH-fold"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed",
         "DRAM%",
         "DRAM bandwidth — lower relative to SM throughput means more work is served from shared memory"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
         "ShmemLdWaves",
         "Shared memory load wavefronts — confirms SRAM is being used to serve multiply-reused tiles"),
        ("l1tex__t_sector_hit_rate.pct",
         "L1hit%",
         "L1 hit rate — tiling improves temporal locality within each phase"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — compute-bound tiled kernel runs at higher pipeline efficiency"),
    ],

    "6. Register Tiling": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
         "ShmemLdWaves",
         "Shared memory load wavefronts — 2×2 register tiling halves shared reads per output element"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
         "ShmemStWaves",
         "Shared memory store wavefronts — cooperative 2×2 stores reduce store pressure proportionally"),
        ("l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum",
         "ShmemLdSectors",
         "Shared load sectors — sectors/wavefront > 1 indicates bank conflicts on wider accesses"),
        ("smsp__inst_executed.sum",
         "Instructions",
         "Total instructions — register accumulation replaces repeated shared memory lookups per k step"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — register operands have lower latency than shared memory reads"),
    ],

    "7. Vector Loads": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("smsp__inst_executed_pipe_lsu.sum",
         "LSU Instr",
         "Load/store unit instructions — float4 issues one LDG.E.128 instead of four LDG.E.32 (4× reduction)"),
        ("smsp__inst_executed.sum",
         "Total Instr",
         "Total instructions — reduced LSU pressure frees issue slots for other execution pipes"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed",
         "DRAM%",
         "DRAM bandwidth — wider 128-bit transactions better saturate the memory bus"),
        ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
         "GlobalLdSectors",
         "Global load sectors — same data volume, 4× fewer transactions"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — fewer LSU instructions lets other pipes run more freely"),
    ],

    "8. Bank Conflicts": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum",
         "ShmemLdSectors",
         "Shared memory load sectors — each bank conflict replays the access; 32-way conflict = 32× sectors"),
        ("l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum",
         "ShmemLdReqs",
         "Shared load requests — sectors/requests is the bank conflict multiplier (32→ before, 1 after)"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
         "ShmemLdWaves",
         "Shared load wavefronts — each extra wavefront is a serialized replay caused by a bank conflict"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — bank conflicts stall the shared memory pipeline and reduce effective throughput"),
    ],

    "9. Privatization": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("l1tex__t_bytes_pipe_lsu_mem_global_op_atom.sum",
         "GlobalAtom(B)",
         "Global atomic bytes — privatization moves atomics to shared memory, eliminating DRAM-level contention"),
        ("dram__bytes_read.sum",
         "DRAMRead(B)",
         "DRAM reads — global atomic read-modify-write causes high traffic; privatization eliminates most of it"),
        ("dram__bytes_write.sum",
         "DRAMWrite(B)",
         "DRAM writes — one flush per block after privatization vs one write per atomic in the naive version"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
         "ShmemStWaves",
         "Shared memory store wavefronts — confirms the optimized version uses fast shared-memory atomics"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — less stalling on global atomic contention allows higher pipeline fill"),
    ],

    "10. Warp Primitives": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
         "ShmemLdWaves",
         "Shared memory load wavefronts — warp shuffle needs no shared memory at all; should collapse to ~0"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
         "ShmemStWaves",
         "Shared memory store wavefronts — also eliminated: the register-level reduction needs no SRAM staging"),
        ("smsp__inst_executed.sum",
         "Instructions",
         "Total instructions — fewer __syncthreads barriers and no shared-memory load/store sequences"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — no barrier stalls means warps issue instructions more continuously"),
        ("smsp__warps_active.avg.pct_of_peak_sustained_active",
         "Occupancy%",
         "Achieved occupancy — warp-shuffle version has lower shared-memory footprint, potentially improving occupancy"),
    ],

    "11. Double Buffering": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("smsp__inst_executed.sum",
         "Instructions",
         "Total instructions — one fewer __syncthreads() per tile iteration removes a global barrier instruction"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — fewer barrier stalls means warps stall less between load and compute phases"),
        ("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
         "ShmemLdWaves",
         "Shared memory load wavefronts — same data volume confirms semantic equivalence between both versions"),
        ("smsp__warps_active.avg.pct_of_peak_sustained_active",
         "Occupancy%",
         "Achieved occupancy — single-block kernel; shows whether the SM pipeline is kept busy between tiles"),
    ],

    "12. Thread Coarsening": [
        ("gpu__time_duration.sum",
         "Duration(ns)",
         "Kernel wall time"),
        ("smsp__inst_executed.sum",
         "Instructions",
         "Total instructions — 4× fewer threads means 4× less index arithmetic and bounds-check overhead"),
        ("smsp__warps_active.avg.pct_of_peak_sustained_active",
         "Occupancy%",
         "Achieved occupancy — intentionally reduced; speedup comes from lower per-thread overhead, not higher occupancy"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed",
         "DRAM%",
         "DRAM bandwidth — same data volume; coarser threads may achieve better per-warp memory locality"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed",
         "SM%",
         "SM utilization — amortized scheduling overhead means more compute cycles per SM clock"),
    ],
}

# (display_label, before_kernel, after_kernel, before_launch_skip, after_launch_skip)
#
# launch_skip=1 skips the single warmup launch so ncu profiles the first *timed*
# launch.  Opt 1 uses the same kernel for both variants; after_skip accounts for
# 1 warmup + NITER timed before-launches + 1 warmup after-launch = NITER + 2.
OPTIMIZATIONS = [
    ("1. Occupancy Tuning",
     "occupancy_vectorAdd_kernel", "occupancy_vectorAdd_kernel",
     1, NITER + 2),
    ("2. Loop Unrolling",
     "loop_unrolling_before_kernel",    "loop_unrolling_after_kernel",    1, 1),
    ("3. Control Divergence",
     "control_divergence_before_kernel", "control_divergence_after_kernel", 1, 1),
    ("4. Memory Coalescing",
     "memory_coalescing_before_kernel", "memory_coalescing_after_kernel", 1, 1),
    ("5. Shared Memory Tiling",
     "shared_tiling_matmul_before",     "shared_tiling_matmul_after",     1, 1),
    ("6. Register Tiling",
     "register_tiling_before_kernel",   "register_tiling_after_kernel",   1, 1),
    ("7. Vector Loads",
     "vector_loads_before_kernel",      "vector_loads_after_kernel",      1, 1),
    ("8. Bank Conflicts",
     "bank_conflicts_before_kernel",    "bank_conflicts_after_kernel",    1, 1),
    ("9. Privatization",
     "privatization_before_kernel",     "privatization_after_kernel",     1, 1),
    ("10. Warp Primitives",
     "warp_primitives_before_kernel",   "warp_primitives_after_kernel",   1, 1),
    ("11. Double Buffering",
     "double_buffering_before_kernel",  "double_buffering_after_kernel",  1, 1),
    ("12. Thread Coarsening",
     "thread_coarsening_before_kernel", "thread_coarsening_after_kernel", 1, 1),
]


def run_ncu(binary: str, kernel: str, launch_skip: int,
            metrics: list[str]) -> Optional[dict]:
    """Profile one kernel launch with ncu and return {metric_name: float}."""
    cmd = [
        "ncu", "--csv", "--quiet",
        "--kernel-name",  kernel,
        "--launch-skip",  str(launch_skip),
        "--launch-count", "1",
        "--metrics",      ",".join(metrics),
        binary,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"\n  [TIMEOUT] ncu exceeded 5 min for kernel '{kernel}'",
              file=sys.stderr)
        return None
    except FileNotFoundError:
        print("ERROR: 'ncu' not found on PATH.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            print(f"\n  [ncu stderr] {stderr}", file=sys.stderr)

    # ncu always quotes CSV fields; unquoted lines are binary stdout noise
    csv_lines = [l for l in result.stdout.splitlines() if l.startswith('"')]
    if not csv_lines:
        return None

    reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
    agg: dict = {}
    cnt: dict = {}
    for row in reader:
        name    = row.get("Metric Name", "").strip()
        val_str = row.get("Metric Value", "").strip().replace(",", "")
        if not name:
            continue
        try:
            val = float(val_str)
            agg[name] = agg.get(name, 0.0) + val
            cnt[name] = cnt.get(name, 0) + 1
        except ValueError:
            pass
    return {k: v / cnt[k] for k, v in agg.items()}


def fmt_val(d: Optional[dict], metric: str) -> str:
    """Return a formatted metric value from the dict, or 'n/a'."""
    if d is None:
        return "n/a"
    for key, val in d.items():
        if key == metric or key.endswith(metric):
            if val >= 1e9:
                return f"{val:.3e}"
            elif val >= 1e4:
                return f"{val:.0f}"
            else:
                return f"{val:.2f}"
    return "n/a"


def fmt_change(before_str: str, after_str: str) -> str:
    """Return signed percentage change string, or 'n/a'."""
    try:
        b, a = float(before_str), float(after_str)
        if b == 0:
            return "n/a"
        pct = (a - b) / abs(b) * 100
        return f"{"+" if pct >= 0 else ""}{pct:.1f}%"
    except (ValueError, TypeError):
        return "n/a"


def print_opt_section(label: str, bk: str, ak: str,
                      metrics_info: list[tuple],
                      mb: Optional[dict], ma: Optional[dict]) -> None:
    """Print a detailed before/after metrics table for one optimization."""
    W = 84
    CL, CB, CA, CC = 20, 14, 14, 10
    print("=" * W)
    print(f" {label}")
    print(f"   before: {bk}")
    print(f"   after:  {ak}")
    print("-" * W)
    print(f"{'Metric':<{CL}}{'Before':>{CB}}{'After':>{CA}}{'Change':>{CC}}")
    print("-" * W)
    for metric, col_label, description in metrics_info:
        bv = fmt_val(mb, metric)
        av = fmt_val(ma, metric)
        ch = fmt_change(bv, av)
        print(f"{col_label:<{CL}}{bv:>{CB}}{av:>{CA}}{ch:>{CC}}")
        print(f"  \u2192 {description}")
    print()


def main() -> None:
    binary = sys.argv[1] if len(sys.argv) > 1 else "./cuda_optimizations"

    if not shutil.which("ncu"):
        print("ERROR: 'ncu' not found on PATH.\n"
              "  Try: export PATH=/usr/local/cuda/bin:$PATH", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(binary):
        print(f"ERROR: binary not found: {binary}", file=sys.stderr)
        sys.exit(1)

    print(f"Binary : {binary}")
    print(f"NITER  : {NITER}  (must match .cu source)\n")

    for label, bk, ak, bskip, askip in OPTIMIZATIONS:
        metrics_info = OPTIMIZATION_METRICS.get(label, [])
        if not metrics_info:
            print(f"WARNING: no metrics defined for '{label}'", file=sys.stderr)
            continue

        metric_names = [m[0] for m in metrics_info]

        print(f"  Profiling {label}...", end=" ", flush=True)
        mb = run_ncu(binary, bk, bskip, metric_names)
        ma = run_ncu(binary, ak, askip, metric_names)
        print("done")

        print_opt_section(label, bk, ak, metrics_info, mb, ma)


if __name__ == "__main__":
    main()
