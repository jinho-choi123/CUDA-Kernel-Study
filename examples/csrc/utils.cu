#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// MACROS FOR CUDA CHECKS
#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
