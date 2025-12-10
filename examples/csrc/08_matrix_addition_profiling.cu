/**
 * Example of Large Matrix Addition
 * NOTE: NUM_ROW > 1024, NUM_COL > 1024
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cu"
#include <stdio.h>

#define NUM_ROW 1024
#define NUM_COL 1024

// number of datas to print
#define PRINT_NUM 32

// Matrix Addition Kernel using 1D Grid-layout and 1D Block-layout
__global__ void matrixAddition1(const float *A, const float *B, float *C,
                                int numElem) {
  // Thread Layout
  // Block: (32,) threads
  // Grid: ((NUM_ROW * NUM_COL + 31) / 32) blocks
  int gThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (gThreadIdx >= numElem) {
    // index out of bound
    return;
  }

  C[gThreadIdx] = A[gThreadIdx] + B[gThreadIdx];
}

// Matrix Addition Kernel using 2D Grid-layout and 2D Block-layout
__global__ void matrixAddition2(const float *A, const float *B, float *C,
                                int numElem) {
  // Thread Layout
  // Block: (32, 32) threads
  // Grid: (NUM_ROW / 32, NUM_COL / 32) blocks
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col >= NUM_COL || row >= NUM_ROW) {
    // index out of bound
    return;
  }

  C[row * NUM_COL + col] = A[row * NUM_COL + col] + B[row * NUM_COL + col];
}

// Matrix Addition Kernel using 2D Grid-layout and 1D Block-layout
__global__ void matrixAddition3(const float *A, const float *B, float *C,
                                int numElem) {
  // Thread Layout
  // Block: (1024, 1) threads
  // Grid: (NUM_ROW / 1024, NUM_COL) blocks
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockIdx.y;

  if (col >= NUM_COL || row >= NUM_ROW) {
    // index out of bound
    return;
  }

  C[row * NUM_COL + col] = A[row * NUM_COL + col] + B[row * NUM_COL + col];
}

int main(void) {
  int result = 0;

  float *a, *b, *c, *goldenC;

  int memSize = sizeof(float) * NUM_ROW * NUM_COL;

  // initialize host data
  a = (float *)calloc(NUM_ROW * NUM_COL, sizeof(float));
  b = (float *)calloc(NUM_ROW * NUM_COL, sizeof(float));
  c = (float *)calloc(NUM_ROW * NUM_COL, sizeof(float));
  goldenC = (float *)calloc(NUM_ROW * NUM_COL, sizeof(float));

  for (int i = 0; i < NUM_ROW * NUM_COL; i++) {
    a[i] = (float)(rand() % 100);
    b[i] = (float)(rand() % 100);
  }

  // Calculate the golden value for c
  for (int i = 0; i < NUM_ROW * NUM_COL; i++) {
    goldenC[i] = a[i] + b[i];
  }

  // Allocate device memory
  float *devA, *devB, *devC;
  CUDA_CHECK(cudaMalloc(&devA, memSize));
  CUDA_CHECK(cudaMalloc(&devB, memSize));
  CUDA_CHECK(cudaMalloc(&devC, memSize));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(devA, a, memSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(devB, b, memSize, cudaMemcpyHostToDevice));

  // Profiling the kernel
  float elapsedTime;
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // Launch Kernel

  // Launch Matrix Addition Kernel
  // You can choose any of the following kernels to run

  // 1D Grid-layout and 1D Block-layout (256,)
  // START: 1D Grid-layout and 1D Block-layout (256,)
  {
    CUDA_CHECK(cudaEventRecord(start));
    int threadsPerBlock = 256;
    int blocksPerGrid =
        ((NUM_ROW * NUM_COL + threadsPerBlock - 1) / threadsPerBlock);
    matrixAddition1<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC,
                                                        NUM_ROW * NUM_COL);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, end));
    printf("Time taken for 1D Grid-layout and 1D Block-layout(256,): %f ms\n",
           elapsedTime);
  }
  // END: 1D Grid-layout and 1D Block-layout (256,)

  // 2D Grid-layout and 2D Block-layout (32, 8)
  // START: 2D Grid-layout and 2D Block-layout (32, 8)
  {
    CUDA_CHECK(cudaEventRecord(start));
    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid((NUM_ROW + 31) / 32, (NUM_COL + 7) / 8);
    matrixAddition2<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC,
                                                        NUM_ROW * NUM_COL);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, end));
    printf("Time taken for 2D Grid-layout and 2D Block-layout(32, 8): %f ms\n",
           elapsedTime);
  }
  // END: 2D Grid-layout and 2D Block-layout (32, 8)

  // 2D Grid-layout and 2D Block-layout (16, 16)
  // START: 2D Grid-layout and 2D Block-layout (16, 16)
  {
    CUDA_CHECK(cudaEventRecord(start));
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NUM_ROW + 15) / 16, (NUM_COL + 15) / 16);
    matrixAddition2<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC,
                                                        NUM_ROW * NUM_COL);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, end));
    printf("Time taken for 2D Grid-layout and 2D Block-layout(16, 16): %f ms\n",
           elapsedTime);
  }
  // END: 2D Grid-layout and 2D Block-layout (16, 16)

  // Free device memory
  CUDA_CHECK(cudaFree(devA));
  CUDA_CHECK(cudaFree(devB));
  CUDA_CHECK(cudaFree(devC));

  // Free host memory
  free(a);
  free(b);
  free(c);
  free(goldenC);

  return result;
}
