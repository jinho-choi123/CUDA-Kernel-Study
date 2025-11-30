/**
 * Example of Vector Addition
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cu"
#include <stdio.h>

#define NUM_DATA 32

__global__ void vectorAddition(const float *A, const float *B, float *C,
                               int size) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId >= size) {
    // Thread out of bounds. do nothing.
    return;
  }

  // Perform element addition.
  // Single thread only handles 1 element addition.
  C[threadId] = A[threadId] + B[threadId];

  return;
}

int main(void) {
  float *a, *b, *c, *golden_c;

  int memSize = sizeof(float) * NUM_DATA;

  // initialize host data
  a = (float *)calloc(NUM_DATA, sizeof(float));
  b = (float *)calloc(NUM_DATA, sizeof(float));
  c = (float *)calloc(NUM_DATA, sizeof(float));
  golden_c = (float *)calloc(NUM_DATA, sizeof(float));
  for (int i = 0; i < NUM_DATA; i++) {
    a[i] = (float)(rand() % 100);
    b[i] = (float)(rand() % 100);
  }

  // Calculate the golden value of c
  for (int i = 0; i < NUM_DATA; i++) {
    golden_c[i] = a[i] + b[i];
  }

  // Allocate device memory
  float *devA, *devB, *devC;
  CUDA_CHECK(cudaMalloc(&devA, memSize));
  CUDA_CHECK(cudaMalloc(&devB, memSize));
  CUDA_CHECK(cudaMalloc(&devC, memSize));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(devA, a, memSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(devB, b, memSize, cudaMemcpyHostToDevice));

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (NUM_DATA + threadsPerBlock - 1) / threadsPerBlock;

  vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC,
                                                     NUM_DATA);
  // Wait for the kernel to end
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result from device to host
  CUDA_CHECK(cudaMemcpy(c, devC, memSize, cudaMemcpyDeviceToHost));

  // Print the result
  printf("Vector A: ");
  for (int i = 0; i < NUM_DATA; i++) {
    printf("%.1f ", a[i]);
  }
  printf("\n");

  printf("Vector B: ");
  for (int i = 0; i < NUM_DATA; i++) {
    printf("%.1f ", b[i]);
  }
  printf("\n");

  printf("Vector C: ");
  for (int i = 0; i < NUM_DATA; i++) {
    printf("%.1f ", c[i]);
  }
  printf("\n");

  // Compare the result
  for (int i = 0; i < NUM_DATA; i++) {
    if (c[i] != golden_c[i]) {
      printf("Error at index %d\n", i);
      return -1;
    }
  }

  // Free device memory
  CUDA_CHECK(cudaFree(devA));
  CUDA_CHECK(cudaFree(devB));
  CUDA_CHECK(cudaFree(devC));

  // Free host memory
  free(a);
  free(b);
  free(c);
  free(golden_c);

  return 0;
}
