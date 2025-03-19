#pragma once
#include "include/reduce.cuh"

#define CINN_BLOCK_SOA_REDUCE_IMPL(DTYPE, cinn_warp_shuffle_internal)  \
  DTYPE tmp_val = cinn_warp_shuffle_internal(value);               \
  if (return_warp || blockDim.x <= 32) {                           \
    return tmp_val;                                                \
  }                                                                \
  __syncthreads();                                                 \
  if (threadIdx.x % 32 == 0) {                                     \
    int tid = threadIdx.x / 32;                                    \
    value_buffer[tid] = tmp_val.value;                             \
    index_buffer[tid] = tmp_val.index;                             \
  }                                                                \
  __syncthreads();                                                 \
  if (threadIdx.x < (blockDim.x + 31) / 32) {                      \
    tmp_val = cinn_warp_shuffle_internal({value_buffer[threadIdx.x], index_buffer[threadIdx.x]});        \
    if (threadIdx.x == 0) {                                        \
      value_buffer[0] = tmp_val.value;                             \
      index_buffer[0] = tmp_val.index;                             \
    }                                                              \
  }                                                                \
  __syncthreads();                                                 \
  return {value_buffer[0], index_buffer[0]};

#define CINN_BLOCK_SOA_REDUCE_MACRO(REDUCE_TYPE, DTYPE, VTYPE, ITYPE) \
  __device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE##_soa(const DTYPE value, VTYPE* value_buffer, ITYPE* index_buffer, bool return_warp = false) { \
    CINN_BLOCK_SOA_REDUCE_IMPL(DTYPE, cinn_warp_shuffle_##REDUCE_TYPE##_internal); \
  }


CINN_BLOCK_SOA_REDUCE_MACRO(max_argidx_fp32_i32, argidx_fp32_i32, float, int)
CINN_BLOCK_SOA_REDUCE_MACRO(max_argidx_fp32_i64, argidx_fp32_i64, float, int64_t)

#undef CINN_BLOCK_SOA_REDUCE_MACRO
#undef CINN_BLOCK_SOA_REDUCE_IMPL