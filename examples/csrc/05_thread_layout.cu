/**
 * Example of Thread Layout
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cu"
#include <stdio.h>

__global__ void checkThreadIndex(void) {
  printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, "
         "%d) gridDim: (%d, %d, %d)\n\n",
         threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,
         blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y,
         gridDim.z);
}

int main(void) {
  // 2x1x1 threads in a block
  dim3 dimBlock(2, 1, 1);
  // 1x2x3 blocks in a grid
  dim3 dimGrid(1, 2, 3);
  printf("dimBlock: (%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
  printf("dimGrid: (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
  checkThreadIndex<<<dimGrid, dimBlock>>>();

  // Wait for GPU to finish before exiting
  cudaDeviceSynchronize();
  return 0;
}
