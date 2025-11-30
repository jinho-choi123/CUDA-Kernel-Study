## CUDA Kernel Study

This repository is for studying CUDA kernel programming.

## Examples

- [01_hello_cuda](examples/01_hello_cuda)
- [02_malloc_free](examples/02_malloc_free)
- [03_memcpy](examples/03_memcpy)
- [04_vector_addition](examples/04_vector_addition)

## How to Run

```bash
uv sync
source .venv/bin/activate

cmake -B build -S .
cmake --build build

# Run the generated binaries
./build/examples/hello_cuda
./build/examples/malloc_free
# There are more binaries...

```
