#include <stdint.h>

extern "C"
__device__ int32_t pos_start_impl() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C"
__device__ int32_t pos_step_impl() {
  return blockDim.x * gridDim.x;
}
