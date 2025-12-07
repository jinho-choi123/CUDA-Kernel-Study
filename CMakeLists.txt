# CMakeLists.txt
cmake_minimum_required(VERSION 3.28)
project(cuda-kernel-study)

# Directory for Module Finders
# Current Module Finder: FindNVCC.cmake
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#### Setup C++ Compiler ####

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#### Setup CUDA Compiler(NVCC) ####

# Enable CUDA
enable_language(CUDA)

find_package(NVCC REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_COMPUTE_COMPATIBILITY 86)

#### Setup CUDA Toolkit ####

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)


#### add subdirectory ####
add_subdirectory(examples/)
