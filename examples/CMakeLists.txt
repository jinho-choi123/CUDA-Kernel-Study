
set(CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty")

# Build 01_hello_cuda
set(01_HELLO_CUDA_SOURCE "${CSRC_DIR}/01_hello_cuda.cu")
add_executable(hello_cuda ${01_HELLO_CUDA_SOURCE})
target_link_libraries(hello_cuda PRIVATE CUDA::cudart)

# Build 02_malloc_free
set(02_MALLOC_FREE "${CSRC_DIR}/02_malloc_free.cu")
add_executable(malloc_free ${02_MALLOC_FREE})
target_link_libraries(malloc_free PRIVATE CUDA::cudart)

# Build 03_memcpy
set(03_MEMCPY "${CSRC_DIR}/03_memcpy.cu")
add_executable(memcpy ${03_MEMCPY})
target_link_libraries(memcpy PRIVATE CUDA::cudart)

# Build 04_vector_addition
set(04_VECTOR_ADDITION "${CSRC_DIR}/04_vector_addition.cu")
add_executable(vector_addition ${04_VECTOR_ADDITION})
target_link_libraries(vector_addition PRIVATE CUDA::cudart)

# Build 05_thread_layout
set(05_THREAD_LAYOUT "${CSRC_DIR}/05_thread_layout.cu")
add_executable(thread_layout ${05_THREAD_LAYOUT})
target_link_libraries(thread_layout PRIVATE CUDA::cudart)

# Build 06_matrix_addition
set(06_MATRIX_ADDITION "${CSRC_DIR}/06_matrix_addition.cu")
add_executable(matrix_addition ${06_MATRIX_ADDITION})
target_link_libraries(matrix_addition PRIVATE CUDA::cudart)

# Build 07_get_gpu_info
set(07_GET_GPU_INFO "${CSRC_DIR}/07_get_gpu_info.cu")
add_executable(get_gpu_info ${07_GET_GPU_INFO})
target_link_libraries(get_gpu_info PRIVATE CUDA::cudart)

# Build 08_matrix_addition_profiling
set(08_MATRIX_ADDITION_PROFILING "${CSRC_DIR}/08_matrix_addition_profiling.cu")
add_executable(matrix_addition_profiling ${08_MATRIX_ADDITION_PROFILING})
target_link_libraries(matrix_addition_profiling PRIVATE CUDA::cudart)
