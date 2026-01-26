#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define PRINT_SMPM(prop) printf("  Shared memory per multiprocessor: %zu bytes (%.2f KB)\n", \
                                 (prop).sharedMemPerMultiprocessor, \
                                 (prop).sharedMemPerMultiprocessor / 1024.0)

#define PRINT_MTPM(prop) printf("  Max threads per multiprocessor: %d\n", \
                                 (prop).maxThreadsPerMultiProcessor)

#define PRINT_MPC(prop) printf("  Multiprocessor count: %d\n", \
                                (prop).multiProcessorCount)

#define PRINT_MBW(prop) printf("  Memory bus width: %d-bit\n", \
                                (prop).memoryBusWidth)

#define PRINT_MBPM(deviceId) do { \
    int value = 0; \
    cudaDeviceGetAttribute(&value, cudaDevAttrMaxBlocksPerMultiprocessor, deviceId); \
    printf("  Max blocks per multiprocessor: %d\n", value); \
} while(0)

#define PRINT_SMPB(prop) printf("  Shared memory per block: %zu bytes (%.2f KB)\n", \
                                 (prop).sharedMemPerBlock, \
                                 (prop).sharedMemPerBlock / 1024.0)

#define PRINT_RGPM(prop) printf("  Registers per multiprocessor: %d\n", \
                                 (prop).regsPerMultiprocessor)

#define PRINT_RGPB(prop) printf("  Registers per block: %d\n", \
                                 (prop).regsPerBlock)

#define PRINT_TCM(prop) printf("  Total constant memory: %zu bytes (%.2f KB)\n", \
                                (prop).totalConstMem, \
                                (prop).totalConstMem / 1024.0)

#define PRINT_L2CS(prop) printf("  L2 cache size: %d bytes (%.2f MB)\n", \
                                 (prop).l2CacheSize, \
                                 (prop).l2CacheSize / (1024.0 * 1024.0))

#define PRINT_CC(prop) printf("  Compute capability: %d.%d\n", \
                               (prop).major, (prop).minor)

#define PRINT_SM(prop) printf("  SM version: %d\n", (prop).major * 10 + (prop).minor)

#define PRINT_CMA(deviceId) do { \
    int value = 0; \
    cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentManagedAccess, deviceId); \
    printf("  Concurrent Managed Access: %s\n", value ? "Yes" : "No"); \
} while(0)

#define PRINT_PMA(deviceId) do { \
    int value = 0; \
    cudaDeviceGetAttribute(&value, cudaDevAttrPageableMemoryAccess, deviceId); \
    printf("  Pageable Memory Access: %s\n", value ? "Yes" : "No"); \
} while(0)

#define PRINT_PMAHPT(deviceId) do { \
    int value = 0; \
    cudaDeviceGetAttribute(&value, cudaDevAttrPageableMemoryAccessUsesHostPageTables, deviceId); \
    printf("  Pageable Memory Access Uses Host Page Tables: %s\n", value ? "Yes" : "No"); \
} while(0)

#define PRINT_UMS(deviceId) do { \
    int concurrentManagedAccess = -1; \
    cudaDeviceGetAttribute(&concurrentManagedAccess, cudaDevAttrConcurrentManagedAccess, deviceId); \
    int pageableMemoryAccess = -1; \
    cudaDeviceGetAttribute(&pageableMemoryAccess, cudaDevAttrPageableMemoryAccess, deviceId); \
    int pageableMemoryAccessUsesHostPageTables = -1; \
    cudaDeviceGetAttribute(&pageableMemoryAccessUsesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, deviceId); \
    printf("  Unified Memory Support (CMA:%d, PMA:%d, PMAHPT:%d): ", \
           concurrentManagedAccess, pageableMemoryAccess, pageableMemoryAccessUsesHostPageTables); \
    if(concurrentManagedAccess) { \
        if(pageableMemoryAccess) { \
            printf("full unified memory support"); \
            if(pageableMemoryAccessUsesHostPageTables) \
                { printf(" with hardware coherency\n"); } \
            else \
                { printf(" with software coherency\n"); } \
        } \
        else \
            { printf("full unified memory support for CUDA-made managed allocations\n"); } \
    } \
    else \
        { printf("limited unified memory support: Windows, WSL, or Tegra\n"); } \
} while(0)

void printToolkitVersion() {
    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 1000) / 10;
    printf("  CUDA Toolkit version: %d.%d\n", major, minor);
}

void printDriverVersion() {
    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    int major = driverVersion / 1000;
    int minor = (driverVersion % 1000) / 10;
    printf("  CUDA Driver API version: %d.%d\n", major, minor);
}

typedef enum {
    PROP_UNKNOWN = 0,       // Unknown property
    PROP_SMPM,              // sharedMemPerMultiprocessor
    PROP_MTPM,              // maxThreadsPerMultiProcessor
    PROP_MPC,               // multiProcessorCount
    PROP_MBW,               // memoryBusWidth
    PROP_MBPM,              // maxBlocksPerMultiprocessor
    PROP_SMPB,              // sharedMemPerBlock
    PROP_RGPM,              // regsPerMultiprocessor
    PROP_RGPB,              // regsPerBlock
    PROP_TCM,               // totalConstMem
    PROP_L2CS,              // l2CacheSize
    PROP_CC,                // Compute Capability (major.minor)
    PROP_SM,                // SM Version
    PROP_TOOLKIT,           // CUDA Toolkit Version
    PROP_DRIVER,            // CUDA Driver API Version
    PROP_CMA,               // Concurrent Managed Access
    PROP_PMA,               // Pageable Memory Access
    PROP_PMAHPT,            // Pageable Memory Access Uses Host Page Tables
    PROP_UMS,               // Unified Memory Support Summary
    PROP_ALL                // All supported properties
} PropertyType;

PropertyType getPropertyType(const char* prop) {
    if (strcmp(prop, "smpm") == 0) return PROP_SMPM;
    if (strcmp(prop, "mtpm") == 0) return PROP_MTPM;
    if (strcmp(prop, "mpc") == 0) return PROP_MPC;
    if (strcmp(prop, "mbw") == 0) return PROP_MBW;
    if (strcmp(prop, "mbpm") == 0) return PROP_MBPM;
    if (strcmp(prop, "smpb") == 0) return PROP_SMPB;
    if (strcmp(prop, "rgpm") == 0) return PROP_RGPM;
    if (strcmp(prop, "rgpb") == 0) return PROP_RGPB;
    if (strcmp(prop, "tcm") == 0) return PROP_TCM;
    if (strcmp(prop, "l2cs") == 0) return PROP_L2CS;
    if (strcmp(prop, "cc") == 0) return PROP_CC;
    if (strcmp(prop, "sm") == 0) return PROP_SM;
    if (strcmp(prop, "toolkit") == 0) return PROP_TOOLKIT;
    if (strcmp(prop, "driver") == 0) return PROP_DRIVER;
    if (strcmp(prop, "cma") == 0) return PROP_CMA;
    if (strcmp(prop, "pma") == 0) return PROP_PMA;
    if (strcmp(prop, "pmahpt") == 0) return PROP_PMAHPT;
    if (strcmp(prop, "ums") == 0) return PROP_UMS;
    if (strcmp(prop, "all") == 0) return PROP_ALL;
    return PROP_UNKNOWN;
}

void printUsage(const char* progName) {
    printf("Usage: %s <prop1>:<prop2>:...\n", progName);
    printf("Available properties (diminutives):\n");
    printf("  smpm    - sharedMemPerMultiprocessor\n");
    printf("  mtpm    - maxThreadsPerMultiProcessor\n");
    printf("  mpc     - multiProcessorCount\n");
    printf("  mbw     - memoryBusWidth\n");
    printf("  mbpm    - maxBlocksPerMultiprocessor\n");
    printf("  smpb    - sharedMemPerBlock\n");
    printf("  rgpm    - regsPerMultiprocessor\n");
    printf("  rgpb    - regsPerBlock\n");
    printf("  tcm     - totalConstMem\n");
    printf("  l2cs    - l2CacheSize\n");
    printf("  cc      - compute capability\n");
    printf("  sm      - SM version\n");
    printf("  toolkit - CUDA Toolkit version\n");
    printf("  driver  - CUDA Driver API version\n");
    printf("  cma     - concurrent managed access (1: full unified - 0: limited support)\n");
    printf("  pma     - pageable memory access (1: all system memory - 0: only explicit managed memory\n");
    printf("  pmahpt  - pageable memory access uses host page tables (1: Hw - 0: SW\n");
    printf("  ums  - Unified Memory Support Summary\n");
    printf("  all     - all supported properties\n");
    printf("Note: For NVIDIA driver version (e.g., 580.95.05), use nvidia-smi\n");
    printf("Example: %s smpm:smpb:cc\n", progName);
}

void queryDeviceProperties(int deviceId, const char* properties) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(error));
        return;
    }
    
    printf("Device %d: %s\n", deviceId, prop.name);
    
    // Parse colon-separated properties
    char* propsCopy = strdup(properties);
    char* token = strtok(propsCopy, ":");
    
    while (token != NULL) {
        PropertyType propType = getPropertyType(token);
        
        switch (propType) {
            case PROP_SMPM:
                PRINT_SMPM(prop);
                break;
            case PROP_MTPM:
                PRINT_MTPM(prop);
                break;
            case PROP_MPC:
                PRINT_MPC(prop);
                break;
            case PROP_MBW:
                PRINT_MBW(prop);
                break;
            case PROP_MBPM:
                PRINT_MBPM(deviceId);
                break;
            case PROP_SMPB:
                PRINT_SMPB(prop);
                break;
            case PROP_RGPM:
                PRINT_RGPM(prop);
                break;
            case PROP_RGPB:
                PRINT_RGPB(prop);
                break;
            case PROP_TCM:
                PRINT_TCM(prop);
                break;
            case PROP_L2CS:
                PRINT_L2CS(prop);
                break;
            case PROP_CC:
                PRINT_CC(prop);
                break;
            case PROP_SM:
                PRINT_SM(prop);
                break;
            case PROP_TOOLKIT:
                printToolkitVersion();
                break;
            case PROP_DRIVER:
                printDriverVersion();
                break;
            case PROP_CMA:
                PRINT_CMA(deviceId);
                break;
            case PROP_PMA:
                PRINT_PMA(deviceId);
                break;
            case PROP_PMAHPT:
                PRINT_PMAHPT(deviceId);
                break;
            case PROP_UMS:
                PRINT_UMS(deviceId);
                break;
            case PROP_ALL:
                PRINT_SMPM(prop);
                PRINT_MTPM(prop);
                PRINT_MPC(prop);
                PRINT_MBW(prop);
                PRINT_MBPM(deviceId);
                PRINT_SMPB(prop);
                PRINT_RGPM(prop);
                PRINT_RGPB(prop);
                PRINT_TCM(prop);
                PRINT_L2CS(prop);
                PRINT_CC(prop);
                PRINT_SM(prop);
                printToolkitVersion();
                printDriverVersion();
                PRINT_CMA(deviceId);
                PRINT_PMA(deviceId);
                PRINT_PMAHPT(deviceId);
                PRINT_UMS(deviceId);
                break;
            case PROP_UNKNOWN:
            default:
                printf("  Unknown property: %s\n", token);
                break;
        }
        token = strtok(NULL, ":");
    }
    
    free(propsCopy);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    // Query device properties
    for (int i = 0; i < deviceCount; i++) {
        queryDeviceProperties(i, argv[1]);
        printf("\n");
    }
    
    return 0;
}
