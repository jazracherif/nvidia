#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define PRINT_SMPM(prop) printf("  Shared memory per multiprocessor: %zu bytes (%.2f KB)\n", \
                                 (prop).sharedMemPerMultiprocessor, \
                                 (prop).sharedMemPerMultiprocessor / 1024.0)

#define PRINT_SMPB(prop) printf("  Shared memory per block: %zu bytes (%.2f KB)\n", \
                                 (prop).sharedMemPerBlock, \
                                 (prop).sharedMemPerBlock / 1024.0)

typedef enum {
    PROP_UNKNOWN = 0,
    PROP_SMPM,
    PROP_SMPB,
    PROP_ALL
} PropertyType;

PropertyType getPropertyType(const char* prop) {
    if (strcmp(prop, "smpm") == 0) return PROP_SMPM;
    if (strcmp(prop, "smpb") == 0) return PROP_SMPB;
    if (strcmp(prop, "all") == 0) return PROP_ALL;
    return PROP_UNKNOWN;
}

void printUsage(const char* progName) {
    printf("Usage: %s <prop1>:<prop2>:...\n", progName);
    printf("Available properties (diminutives):\n");
    printf("  smpm  - sharedMemPerMultiprocessor\n");
    printf("  smpb  - sharedMemPerBlock\n");
    printf("  all   - all supported properties\n");
    printf("Example: %s smpm:smpb\n", progName);
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
            case PROP_ALL:
                PRINT_SMPM(prop);
                PRINT_SMPB(prop);
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
    
    // Check if properties contain smpm, smpb, or all
    if (strstr(argv[1], "smpm") != NULL || strstr(argv[1], "smpb") != NULL || strstr(argv[1], "all") != NULL) {
        for (int i = 0; i < deviceCount; i++) {
            queryDeviceProperties(i, argv[1]);
            printf("\n");
        }
    } else {
        printf("No shared memory properties (smpm, smpb, or all) specified.\n");
    }
    
    return 0;
}
