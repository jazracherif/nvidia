#include <stdio.h>
#include <cuda_runtime.h>

// Dummy kernel signatures matching your actual kernels
__global__ void primary_kernel(float* input, float* output, float* dummy, int size) {}
__global__ void secondary_kernel(float* data, int size) {}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("\n");
    
    int blockSize = 256;
    int minGridSize, gridSize;
    
    // Check primary kernel occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, primary_kernel, 0, 0);
    printf("Primary kernel optimal block size: %d\n", blockSize);
    
    int numBlocksPrimary;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPrimary, primary_kernel, 256, 0);
    printf("Primary kernel (256 threads): %d blocks per SM\n", numBlocksPrimary);
    float occupancyPrimary = (numBlocksPrimary * 256 / (float)prop.maxThreadsPerMultiProcessor) * 100;
    printf("Primary kernel occupancy: %.1f%%\n", occupancyPrimary);
    printf("\n");
    
    // Check secondary kernel occupancy  
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, secondary_kernel, 0, 0);
    printf("Secondary kernel optimal block size: %d\n", blockSize);
    
    int numBlocksSecondary;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksSecondary, secondary_kernel, 256, 0);
    printf("Secondary kernel (256 threads): %d blocks per SM\n", numBlocksSecondary);
    float occupancySecondary = (numBlocksSecondary * 256 / (float)prop.maxThreadsPerMultiProcessor) * 100;
    printf("Secondary kernel occupancy: %.1f%%\n", occupancySecondary);
    printf("\n");
    
    // Calculate if both can run concurrently
    printf("=== Concurrent Execution Analysis ===\n");
    printf("If primary uses %d blocks/SM and secondary uses %d blocks/SM:\n", 
           numBlocksPrimary, numBlocksSecondary);
    printf("Total blocks needed: %d\n", numBlocksPrimary + numBlocksSecondary);
    printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    
    if (numBlocksPrimary + numBlocksSecondary <= prop.maxBlocksPerMultiProcessor) {
        printf("✓ Both kernels CAN run concurrently\n");
    } else {
        printf("✗ Kernels CANNOT overlap - not enough resources\n");
        printf("Suggestion: Reduce grid size or block size\n");
    }
    
    return 0;
}
