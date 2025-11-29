
set(CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")

# Build 01_hello_cuda
set(01_HELLO_CUDA_SOURCE "${CSRC_DIR}/01_hello_cuda.cu")
add_executable(hello_cuda ${01_HELLO_CUDA_SOURCE})
target_link_libraries(hello_cuda PRIVATE CUDA::cudart)