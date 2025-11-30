/**
 * Example of CUDA Malloc/Free
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void checkDeviceMemory(void) {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  printf("Device memory (free/total) = %zu/%zu bytes\n", free, total);
}

int main(void) {
  int *devPtr;
  cudaError_t errCode;

  checkDeviceMemory();

  errCode = cudaMalloc(&devPtr, sizeof(int) * 1024 * 1024);
  printf("cudaMalloc - %s\n", cudaGetErrorName(errCode));
  checkDeviceMemory();

  errCode = cudaMemset(devPtr, 0, sizeof(int) * 1024 * 1024);
  printf("cudaMemset - %s\n", cudaGetErrorName(errCode));
  checkDeviceMemory();

  errCode = cudaFree(devPtr);
  printf("cudaFree - %s\n", cudaGetErrorName(errCode));
  checkDeviceMemory();

  return 0;
}
