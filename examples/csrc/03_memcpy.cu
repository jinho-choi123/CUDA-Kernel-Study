/**
 * Example of MemCpy
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printData(int *devDataPtr) {
  printf("%d ", devDataPtr[threadIdx.x]);
}

__global__ void setData(int *devDataPtr, int data) {
  devDataPtr[threadIdx.x] = data;
}

int main(void) {

  // Initialize host data
  int data[10] = {0};
  for (int i = 0; i < 10; i++) {
    data[i] = 1;
  }

  // Allocate device memory
  int *devDataPtr;
  cudaMalloc(&devDataPtr, sizeof(int) * 10);
  cudaMemset(devDataPtr, 0, sizeof(int) * 10);

  printf("Data in device: ");
  printData<<<1, 10>>>(devDataPtr);

  // Memcpy from host to device
  cudaMemcpy(devDataPtr, data, sizeof(int) * 10, cudaMemcpyHostToDevice);
  printf("\nHost -> Device: ");
  printData<<<1, 10>>>(devDataPtr);

  // set the data as 2
  setData<<<1, 10>>>(devDataPtr, 2);

  // Memcpy from device to host
  cudaMemcpy(data, devDataPtr, sizeof(int) * 10, cudaMemcpyDeviceToHost);
  printf("\nDevice -> Host: ");
  for (int i = 0; i < 10; i++) {
    printf("%d ", data[i]);
  }
  printf("\n");

  cudaFree(devDataPtr);

  return 0;
}
