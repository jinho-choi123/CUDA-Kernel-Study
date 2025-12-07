/**
 * Example of Getting GPU Info
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int main(void) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Number of GPUs: %d\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("Device %d: %s\n", i, deviceProp.name);
    printf("  Total Global Memory: %zu GB\n",
           deviceProp.totalGlobalMem / 1024 / 1024 / 1024);
    printf("  Shared Memory per Block: %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max Threads Dim: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Max Grid Size: (%d, %d, %d)\n", deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("  Clock Rate: %d MHz\n", deviceProp.clockRate / 1024);
    printf("  Total Constant Memory: %zu bytes\n", deviceProp.totalConstMem);
    printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("\n\n\n");
  }
  return 0;
}
