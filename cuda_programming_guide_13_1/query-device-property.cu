#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define PRINT_SMPM(prop) printf("  Shared memory per multiprocessor: %zu bytes (%.2f KB)\n", \
                                 (prop).sharedMemPerMultiprocessor, \
                                 (prop).sharedMemPerMultiprocessor / 1024.0)

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

#define PRINT_CC(prop) printf("  Compute capability: %d.%d\n", \
                               (prop).major, (prop).minor)

#define PRINT_SM(prop) printf("  SM version: %d\n", (prop).major * 10 + (prop).minor)

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
    PROP_SMPB,              // sharedMemPerBlock
    PROP_RGPM,              // regsPerMultiprocessor
    PROP_RGPB,              // regsPerBlock
    PROP_TCM,               // totalConstMem
    PROP_CC,                // Compute Capability (major.minor)
    PROP_SM,                // SM Version
    PROP_TOOLKIT,           // CUDA Toolkit Version
    PROP_DRIVER,            // CUDA Driver API Version
    PROP_ALL                // All supported properties
} PropertyType;

PropertyType getPropertyType(const char* prop) {
    if (strcmp(prop, "smpm") == 0) return PROP_SMPM;
    if (strcmp(prop, "smpb") == 0) return PROP_SMPB;
    if (strcmp(prop, "rgpm") == 0) return PROP_RGPM;
    if (strcmp(prop, "rgpb") == 0) return PROP_RGPB;
    if (strcmp(prop, "tcm") == 0) return PROP_TCM;
    if (strcmp(prop, "cc") == 0) return PROP_CC;
    if (strcmp(prop, "sm") == 0) return PROP_SM;
    if (strcmp(prop, "toolkit") == 0) return PROP_TOOLKIT;
    if (strcmp(prop, "driver") == 0) return PROP_DRIVER;
    if (strcmp(prop, "all") == 0) return PROP_ALL;
    return PROP_UNKNOWN;
}

void printUsage(const char* progName) {
    printf("Usage: %s <prop1>:<prop2>:...\n", progName);
    printf("Available properties (diminutives):\n");
    printf("  smpm    - sharedMemPerMultiprocessor\n");
    printf("  smpb    - sharedMemPerBlock\n");
    printf("  rgpm    - regsPerMultiprocessor\n");
    printf("  rgpb    - regsPerBlock\n");
    printf("  tcm     - totalConstMem\n");
    printf("  cc      - compute capability\n");
    printf("  sm      - SM version\n");
    printf("  toolkit - CUDA Toolkit version\n");
    printf("  driver  - CUDA Driver API version\n");
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
            case PROP_ALL:
                PRINT_SMPM(prop);
                PRINT_SMPB(prop);
                PRINT_RGPM(prop);
                PRINT_RGPB(prop);
                PRINT_TCM(prop);
                PRINT_CC(prop);
                PRINT_SM(prop);
                printToolkitVersion();
                printDriverVersion();
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
