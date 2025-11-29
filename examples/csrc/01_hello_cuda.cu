#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloCUDAKernel(void) { printf("Hello, CUDA from GPU!!\n"); }

int main(void) {
  printf("Hello GPU from CPU!\n");

  helloCUDAKernel<<<1, 10>>>();
  return 0;
}