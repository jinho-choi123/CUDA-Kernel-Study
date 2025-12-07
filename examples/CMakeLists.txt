
set(CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty")

# Build 01_hello_cuda
set(01_HELLO_CUDA_SOURCE "${CSRC_DIR}/01_hello_cuda.cu")
add_executable(hello_cuda ${01_HELLO_CUDA_SOURCE})
target_link_libraries(hello_cuda PRIVATE CUDA::cudart)
target_include_directories(hello_cuda PRIVATE ${THIRD_PARTY_DIR}/spdlog/include)

# Build 02_malloc_free
set(02_MALLOC_FREE "${CSRC_DIR}/02_malloc_free.cu")
add_executable(malloc_free ${02_MALLOC_FREE})
target_link_libraries(malloc_free PRIVATE CUDA::cudart)
target_include_directories(malloc_free PRIVATE ${THIRD_PARTY_DIR}/spdlog/include)

# Build 03_memcpy
set(03_MEMCPY "${CSRC_DIR}/03_memcpy.cu")
add_executable(memcpy ${03_MEMCPY})
target_link_libraries(memcpy PRIVATE CUDA::cudart)
target_include_directories(memcpy PRIVATE ${THIRD_PARTY_DIR}/spdlog/include)

# Build 04_vector_addition
set(04_VECTOR_ADDITION "${CSRC_DIR}/04_vector_addition.cu")
add_executable(vector_addition ${04_VECTOR_ADDITION})
target_link_libraries(vector_addition PRIVATE CUDA::cudart)
target_include_directories(vector_addition PRIVATE ${THIRD_PARTY_DIR}/spdlog/include)

# Build 05_thread_layout
set(05_THREAD_LAYOUT "${CSRC_DIR}/05_thread_layout.cu")
add_executable(thread_layout ${05_THREAD_LAYOUT})
target_link_libraries(thread_layout PRIVATE CUDA::cudart)
target_include_directories(thread_layout PRIVATE ${THIRD_PARTY_DIR}/spdlog/include)
