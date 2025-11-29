find_program(CMAKE_CUDA_COMPILER
  NAMES nvcc
  PATHS
    /usr/local/cuda/bin
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVCC DEFAULT_MSG CMAKE_CUDA_COMPILER)