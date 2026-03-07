# libcudf-tutorial Tutorial – groupby sum example

Demonstrates creating an Arrow-backed table in C++ with libcudf and running a `groupby().sum()` reduction.

## Table of Contents

- [1. Prerequisites](#1-prerequisites)
  - [1.1 Create the conda environment (first time only)](#11-create-the-conda-environment-first-time-only)
- [2. Build](#2-build)
- [3. Generate sample data](#3-generate-sample-data)
  - [3.1 Basic usage](#31-basic-usage-10-rows--dataordersparquet)
  - [3.2 Options](#32-options)
- [4. Run](#4-run)
  - [4.1 Inline Arrow mode (no arguments)](#41-inline-arrow-mode-no-arguments)
  - [4.2 Parquet mode (optional argument)](#42-parquet-mode-optional-argument)
- [5. Profile with NVIDIA Nsight Compute](#5-profile-with-nvidia-nsight-compute)
  - [5.1 Basic command-line profile](#51-basic-command-line-profile)
  - [5.2 Target specific kernels](#52-target-specific-kernels)
  - [5.3 Useful metric sets](#53-useful-metric-sets)
  - [5.4 Notes](#54-notes)
- [6. Profile with NVIDIA Nsight Systems](#6-profile-with-nvidia-nsight-systems)
  - [6.1 Basic profile (save a report)](#61-basic-profile-save-a-report)
  - [6.2 Trace CUDA API + kernels + memory copies](#62-trace-cuda-api--kernels--memory-copies)
  - [6.3 Print a quick summary to stdout (no GUI needed)](#63-print-a-quick-summary-to-stdout-no-gui-needed)
  - [6.4 Source file and line info in backtraces](#64-source-file-and-line-info-in-backtraces)
  - [6.5 Notes](#65-notes)
- [7. Clean](#7-clean)
- [8. Files](#8-files)

## 1. Prerequisites

- NVIDIA GPU with CUDA drivers installed
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- A `libcudf-tutorial` conda environment with the Rapids stack

### 1.1 Create the conda environment (first time only)

```bash
conda create -n libcudf-tutorial -c rapidsai -c conda-forge -c nvidia \
    cudf=26.02 libcudf=26.02 rmm=26.02 arrow-cpp cmake make cxx-compiler \
    cuda-version=13.0
```

> Adjust `cudf` and `cuda-version` to match your installed CUDA toolkit.

## 2. Build

```bash
conda activate libcudf-tutorial
make
```

This runs CMake with `CMAKE_PREFIX_PATH=$CONDA_PREFIX`, which lets CMake find
cudf, Arrow and RMM automatically, then compiles the binary into `build/`.

## 3. Generate sample data

`make_tpch_orders.py` creates a typed Parquet (or Arrow IPC) file under `data/`
that can be used as input to `libcudf_tpch_orders_groupby`.  It requires `pyarrow`:

```bash
pip install pyarrow
```

### 3.1 Basic usage (10 rows → `data/orders.parquet`)

```bash
python make_tpch_orders.py
```

### 3.2 Options

| Flag | Default | Description |
|------|---------|-------------|
| `--rows N` | `10` | Number of rows to generate |
| `--output FILENAME` | `orders.parquet` | Output filename (written inside `./data/`) |
| `--format parquet\|ipc` | `parquet` | Output format: Parquet or Arrow IPC (Feather v2) |

Example — generate 1000 rows into a custom file:

```bash
python make_tpch_orders.py --rows 1000 --output my_orders.parquet
```

The generated table matches the [TPC-H Orders](https://www.tpc.org/tpch/) schema.
The script prints the schema and a preview of the first five rows:

```
Schema:
o_orderkey: int64
o_custkey: int64
o_orderstatus: string
o_totalprice: double
o_orderdate: date32[day]
o_orderpriority: string
o_clerk: string
o_shippriority: int32
o_comment: string

First 5 rows:
 o_orderkey  o_custkey o_orderstatus  o_totalprice  o_orderdate o_orderpriority             o_clerk  o_shippriority           o_comment
          1          1             F          1.99   2024-01-01        1-URGENT  Clerk#000000001               0   order 1 placed
          2          2             O          2.99   2024-01-01          2-HIGH  Clerk#000000002               0   order 2 placed
 ...

Wrote 10 rows × 9 cols → data/orders.parquet
```

Columns produced (TPC-H Orders schema):

| # | Name | Type | Description | Role |
|---|------|------|-------------|------|
| 0 | `o_orderkey` | `int64` | Unique order identifier | |
| 1 | `o_custkey` | `int64` | FK → Customer table | |
| 2 | `o_orderstatus` | `utf8` | `'F'` fulfilled \| `'O'` open \| `'P'` pending | groupby key |
| 3 | `o_totalprice` | `float64` | Total order value in USD | sum value |
| 4 | `o_orderdate` | `date32` | Date the order was placed | |
| 5 | `o_orderpriority` | `utf8` | `'1-URGENT'` through `'5-LOW'` | |
| 6 | `o_clerk` | `utf8` | Clerk ID, e.g. `Clerk#000000001` | |
| 7 | `o_shippriority` | `int32` | Shipping priority (always 0 in TPC-H) | |
| 8 | `o_comment` | `utf8` | Free-form comment (≤79 chars) | |

## 4. Run

### 4.1 Inline Arrow mode (no arguments)

Builds a small in-process Arrow `RecordBatch` with the two TPC-H Orders
columns used by the groupby — `o_orderstatus` (utf8) as key and
`o_totalprice` (float64) as value — then runs
`groupby(o_orderstatus).sum(o_totalprice)`:

```bash
./build/libcudf_tpch_orders_groupby
```

Expected output:

```
=== Input Arrow RecordBatch ===
o_orderstatus: ["F", "O", "F", "P", "O"]
o_totalprice:  [173665.47, 46929.18, 193846.25, 32151.78, 121200.00]

=== groupby(o_orderstatus).sum(o_totalprice) ===
o_orderstatus:
F
O
P
sum(o_totalprice):
367511.72
168129.18
32151.78
```

### 4.2 Parquet mode (optional argument)

Pass a Parquet file path to load data from disk instead.  The file is
expected to follow the TPC-H Orders schema produced by `make_tpch_orders.py`.
By default the groupby runs on `o_orderstatus` (key) and sums `o_totalprice` (value).

```bash
./build/libcudf_tpch_orders_groupby data/orders.parquet
```

Override the default column names via the `from_parquet()` arguments in
`libcudf_tpch_orders_groupby.cu` if you want to group by a different column.

Expected output:

```
=== Loaded Parquet: data/orders.parquet (9 cols, N rows) ===
    key_col="o_orderstatus" [2]
    value_col="o_totalprice" [3]
=== groupby(o_orderstatus).sum(o_totalprice) ===
o_orderstatus:
F
O
P
sum(o_totalprice):
<aggregated totals per status>
```

## 5. Profile with NVIDIA Nsight Compute

[Nsight Compute](https://developer.nvidia.com/nsight-compute) (`ncu`) profiles individual CUDA kernels with hardware performance counters.

### 5.1 Basic command-line profile

Collect all default metrics and save a report file:

```bash
ncu --set full \
    -o reports/libcudf_groupby \
    ./build/libcudf_tpch_orders_groupby ./data/orders.parquet
```

> `--set full` collects the complete default metric set.  Substitute `--set basic` for a faster, lighter collection.

### 5.2 Target specific kernels

libcudf launches many internal kernels.  Narrow the profile to kernels whose names match a pattern:

```bash
# Profile only groupby-related kernels
ncu --kernel-name-base function \
    --kernel-name "groupby\|reduce\|hash" \
    --set full \
    -o reports/libcudf_groupby \
    ./build/libcudf_tpch_orders_groupby ./data/orders.parquet
```

### 5.3 Useful metric sets

| Flag | What it collects |
|------|-----------------|
| `--set basic` | Achieved occupancy, memory throughput, SM efficiency |
| `--set full` | All default counters across compute, memory, scheduler |
| `--section MemoryWorkloadAnalysis` | L1/L2/HBM bandwidth and cache hit rates |
| `--section ComputeWorkloadAnalysis` | Pipe utilization, warp stalls |
| `--section SpeedOfLight` | Achieved vs peak compute and memory throughput |

Example — collect only memory and roofline sections:

```bash
ncu --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    -o reports/libcudf_mem \
    ./build/libcudf_tpch_orders_groupby ./data/orders.parquet
```

### 5.4 Notes

- Profiling requires root or `CAP_SYS_ADMIN` (or `/proc/sys/kernel/perf_event_paranoid ≤ 2`).  If you see a permissions error:
  ```bash
  sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'
  ```
- Create the output directory first: `mkdir -p reports`
- The binary must be built in `Release` mode (the default) for meaningful results; debug builds add overhead.

## 6. Profile with NVIDIA Nsight Systems

[Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`) gives a system-wide timeline — CPU threads, CUDA API calls, kernel launches, memory copies, and NVTX annotations — making it the right first step before diving into per-kernel metrics with Nsight Compute.

### 6.1 Basic profile (save a report)

```bash
mkdir -p reports
nsys profile \
    --output reports/libcudf_groupby \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby ./data/orders.parquet
```

This produces `reports/libcudf_groupby.nsys-rep`.  Open it in the Nsight Systems GUI:

```bash
nsys-ui reports/libcudf_groupby.nsys-rep
```

Or on a remote machine, copy the `.nsys-rep` file locally and open it with the Nsight Systems desktop app.

### 6.2 Trace CUDA API + kernels + memory copies

```bash
nsys profile \
    --trace cuda,osrt,nvtx \
    --output reports/libcudf_groupby \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby ./data/orders.parquet
```

| `--trace` value | What is recorded |
|-----------------|-----------------|
| `cuda` | CUDA API calls, kernel launches, memcpy/memset |
| `osrt` | OS runtime: pthreads, semaphores, signals |
| `nvtx` | User-defined NVTX ranges (if any) |
| `cudnn` | cuDNN API calls |

Print a summary of all NVTX ranges by name and total duration (no GUI needed):

```bash
nsys stats --report nvtx_pushpop_sum reports/libcudf_groupby.nsys-rep
```

libcudf uses the push/pop NVTX API, so `nvtx_pushpop_*` to show api calls in nsigth, see some choices here:

| Report | Output style | What it shows |
|--------|-------------|---------------|
| `nvtx_pushpop_sum` | Aggregated | One row per unique range name: total time, call count, min/avg/max duration — best for identifying slow phases |
| `nvtx_pushpop_trace` | Per-instance | One row per individual range invocation with start time, end time, duration and thread ID — useful for spotting outliers or variance across calls |
| `nvtx_startend_sum` | Aggregated | Same as above but for ranges created with the start/end API (used by PyTorch autograd, user code spanning threads) |
| `nvtx_sum` | Aggregated | Combined view merging both push/pop and start/end ranges into one table |

### 6.3 Print a quick summary to stdout (no GUI needed)

```bash
nsys profile \
    --stats true \
    --output reports/libcudf_groupby \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby ./data/orders.parquet
```

`--stats true` prints a text table of top kernels, API calls, and memory operations after the run.

### 6.4 Source file and line info in backtraces

To resolve kernel launches and CPU calls back to file names and line numbers,
first rebuild with debug symbols (already enabled in `CMakeLists.txt` via
`--generate-line-info` for CUDA and `-g` for C++):

```bash
make clean && make
```

Then profile with CPU sampling and DWARF unwinding:

```bash
nsys profile \
    --trace cuda,nvtx,osrt \
    --sample cpu \
    --cpuctxsw system-wide \
    --backtrace dwarf \
    --cuda-memory-usage true \
    --output reports/libcudf_groupby \
    --force-overwrite true \
    ./build/libcudf_tpch_orders_groupby ./data/orders.parquet
```

| Flag | Effect |
|------|--------|
| `--sample cpu` | Periodic CPU call-stack samples — shows which host-side code drives kernel launches |
| `--backtrace dwarf` | Resolves stack frames to file name and line number using DWARF debug info |
| `--cpuctxsw system-wide` | Captures thread context switches system-wide for scheduling visibility |

In `nsys-ui`:
- Click any kernel in the timeline → **Source** tab in the bottom panel shows the CUDA call stack with `file:line` resolved
- The **CPU Sampling** row shows a flame graph that maps to your source lines in `libcudf_tpch_orders_groupby.cu`

> Line info for libcudf's own internal kernels resolves only if the Rapids conda package was built with `-lineinfo` (the `rapidsai` channel release builds typically are).

### 6.5 Notes

- `nsys` does **not** require elevated permissions for basic CUDA tracing.
- Use Nsight Systems first to find which kernels take the most time, then drill into them with `ncu`.
- The `.nsys-rep` file can be opened on any machine with the Nsight Systems GUI installed (no GPU required for viewing).

## 7. Clean

```bash
make clean
```

## 8. Files

| File | Description |
|------|-------------|
| `libcudf_tpch_orders_groupby.cu` | Main C++/CUDA source |
| `CMakeLists.txt`      | CMake build definition |
| `Makefile`            | Thin wrapper around CMake |
| `make_tpch_orders.py` | Python script to generate TPC-H Orders Parquet/IPC data |
