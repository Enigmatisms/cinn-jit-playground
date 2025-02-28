#pragma once
#include <cuda_runtime.h>

__device__ inline float cinn_sum_fp32(const float left, const float right) { return left + right; }

#define CINN_DISCRETE_REDUCE_IMPL(REDUCE_TYPE, value)                          \
  int tid = threadIdx.y * blockDim.x + threadIdx.x;                            \
  __syncthreads();                                                             \
  shm[tid] = value;                                                            \
  __syncthreads();                                                             \
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {                \
    if (threadIdx.y < offset) {                                                \
      shm[tid] = cinn_##REDUCE_TYPE(shm[tid], shm[tid + offset * blockDim.x]); \
    }                                                                          \
    __syncthreads();                                                           \
  }                                                                            \
  return shm[threadIdx.x];

#define CINN_DISCRETE_REDUCE_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE)          \
  __device__ inline DTYPE cinn_discrete_reduce_##REDUCE_TYPE(const DTYPE value, DTYPE* shm) { \
    CINN_DISCRETE_REDUCE_IMPL(REDUCE_TYPE, value);                             \
  }

#define CINN_GRID_REDUCE_IMPL(REDUCE_TYPE, init_value, DTYPE)                       \
  DTYPE tmp_val = init_value;                                                       \
  for (int y = 0; y < gridDim.y; y++) {                                             \
      tmp_val = cinn_##REDUCE_TYPE(tmp_val, mem[y * spatial_size + spatial_index]); \
  }                                                                                 \
  return tmp_val;

#define CINN_GRID_REDUCE_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                   \
  __device__ inline DTYPE cinn_grid_reduce_##REDUCE_TYPE(const DTYPE* mem, int spatial_size, int spatial_index) { \
    CINN_GRID_REDUCE_IMPL(REDUCE_TYPE, (DTYPE)(INITIAL_VALUE), DTYPE);              \
  }

CINN_DISCRETE_REDUCE_MACRO(sum_fp32, 0.0f, float)
CINN_GRID_REDUCE_MACRO(sum_fp32, 0.0f, float)