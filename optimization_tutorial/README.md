# CUDA Optimization Tutorial

This repository contains a collection of CUDA optimization examples demonstrating various techniques to improve GPU performance. Each example focuses on a specific optimization strategy with before/after comparisons using NVIDIA Nsight Compute profiling.

## Project Structure

```
.
├── src/                 # Source CUDA files for each optimization technique
├── build/               # Compiled executables
├── results/             # Profiling results and reports
├── docs/                # Documentation and metrics lists
├── common.cuh           # Common header file with utility functions
├── cuda_optimizations_examples.cu  # Main example file
├── Makefile             # Build system
├── optimizations.py     # Python helper scripts
├── run_ncu.py           # NCU profiling script
└── README.md            # This file
```

## Optimization Techniques

The tutorials cover the following CUDA optimization techniques:

1. **Occupancy Tuning** - Maximizing GPU utilization by optimizing thread block sizes
2. **Loop Unrolling** - Reducing loop overhead and improving instruction-level parallelism
3. **Control Divergence** - Minimizing warp divergence in conditional statements
4. **Memory Coalescing** - Optimizing memory access patterns for better bandwidth utilization
5. **Shared Memory Tiling** - Using shared memory effectively to reduce global memory accesses
6. **Register Tiling** - Managing register usage for optimal occupancy
7. **Vector Loads** - Utilizing vectorized memory operations for better throughput
8. **Bank Conflicts** - Avoiding shared memory bank conflicts
9. **Privatization** - Eliminating race conditions and improving performance
10. **Warp Primitives** - Using optimized warp-level operations
11. **Double Buffering** - Overlapping computation with memory transfers
12. **Thread Coarsening** - Reducing thread divergence and improving efficiency

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- NVIDIA Nsight Compute (NCU) profiler
- Python 3.x

### Building the Examples

```bash
# Build all examples
make build

# Clean build artifacts
make clean

# Run all examples and generate summary
make run

# Perform full profiling with NCU
make profile
```

## Profiling Workflow

The project uses NVIDIA Nsight Compute for performance analysis:

1. `make profile` - Runs profiling on all examples
2. Results are saved in the `results/` directory
3. Each optimization example includes before/after comparisons
4. Detailed performance metrics and analysis are generated automatically

## Key Features

- **Before/After Comparisons**: Each example shows performance improvements
- **Automated Profiling**: Scripts handle NCU execution and result collection
- **Comprehensive Documentation**: Metrics and analysis included in `docs/`
- **Modular Design**: Easy to extend with new optimization examples

## Usage

### Running Individual Examples

```bash
# Build a specific example
make build
./build/05_shared_memory_tiling

# Run all examples with summary
make run
```

### Viewing Profiling Results

Profiling results are stored in the `results/` directory. Each optimization example generates:
- NCU report files (.ncu-rep)
- Performance analysis text files
- Comparison data between before/after implementations

## Contributing

Feel free to add new optimization examples or improvements to existing ones. The structure is designed to be easily extensible for additional techniques.

## License

This project is licensed under the MIT License.