#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using float16 = half;


#define CINN_INT32_MAX 2147483647
#define CINN_INT32_MIN -2147483648
#define CINN_FP32_MAX 3.40282347e+38F
#define CINN_FP64_MAX 1.79769313486231571e+308

__host__ __device__ constexpr inline float16 raw_uint16_to_float16(uint16_t a) {
  return *reinterpret_cast<float16*>(&a);
}

#define CINN_FP16_MAX raw_uint16_to_float16(0x7bff)
#define CINN_FP16_MIN raw_uint16_to_float16(0xfbff)

#define EXPAND_REDUCE_INT32_MARCO(MARCO, ...)       \
  MARCO(sum_int32, 0, int, ##__VA_ARGS__)           \
  MARCO(prod_int32, 1, int, ##__VA_ARGS__)          \
  MARCO(max_int32, CINN_INT32_MIN, int, ##__VA_ARGS__) \
  MARCO(min_int32, CINN_INT32_MAX, int, ##__VA_ARGS__)

__device__ inline int cinn_sum_int32(const int left, const int right) { return left + right; }
__device__ inline int cinn_prod_int32(const int left, const int right) { return left * right; }
__device__ inline int cinn_max_int32(const int left, const int right) { return max(left, right); }
__device__ inline int cinn_min_int32(const int left, const int right) { return min(left, right); }

#define EXPAND_REDUCE_INT64_MARCO(MARCO, ...)                          \
  MARCO(sum_int64, 0, long long int, ##__VA_ARGS__)                    \
  MARCO(prod_int64, 1, long long int, ##__VA_ARGS__)                   \
  MARCO(max_int64, -9223372036854775808, long long int, ##__VA_ARGS__) \
  MARCO(min_int64, 9223372036854775807, long long int, ##__VA_ARGS__)

__device__ inline long long int cinn_sum_int64(const long long int left, const long long int right) {
  return left + right;
}
__device__ inline long long int cinn_prod_int64(const long long int left, const long long int right) {
  return left * right;
}
__device__ inline long long int cinn_max_int64(const long long int left, const long long int right) {
  return max(left, right);
}
__device__ inline long long int cinn_min_int64(const long long int left, const long long int right) {
  return min(left, right);
}

#define EXPAND_REDUCE_FP32_MACRO(MACRO, ...)           \
  MACRO(sum_fp32, 0.0f, float, ##__VA_ARGS__)          \
  MACRO(prod_fp32, 1.0f, float, ##__VA_ARGS__)         \
  MACRO(max_fp32, -CINN_FP32_MAX, float, ##__VA_ARGS__) \
  MACRO(min_fp32, CINN_FP32_MAX, float, ##__VA_ARGS__)

__device__ inline float cinn_sum_fp32(const float left, const float right) { return left + right; }
__device__ inline float cinn_prod_fp32(const float left, const float right) { return left * right; }
__device__ inline float cinn_max_fp32(const float left, const float right) { return max(left, right); }
__device__ inline float cinn_min_fp32(const float left, const float right) { return min(left, right); }

#ifdef CINN_CUDA_FP16

#define EXPAND_REDUCE_FP16_MACRO(MACRO, ...)                                           \
  MACRO(sum_fp16, float16(0.0), float16, ##__VA_ARGS__)                                \
  MACRO(prod_fp16, float16(1.0), float16, ##__VA_ARGS__)                               \
  MACRO(max_fp16, raw_uint16_to_float16(0xfbff), float16, ##__VA_ARGS__) \
  MACRO(min_fp16, raw_uint16_to_float16(0x7bff), float16, ##__VA_ARGS__)

__device__ inline float16 cinn_sum_fp16(const float16 left, const float16 right) { return left + right; }
__device__ inline float16 cinn_prod_fp16(const float16 left, const float16 right) { return left * right; }
__device__ inline float16 cinn_max_fp16(const float16 left, const float16 right) { return max(left, right); }
__device__ inline float16 cinn_min_fp16(const float16 left, const float16 right) { return min(left, right); }
#endif

#define EXPAND_REDUCE_FP64_MACRO(MACRO, ...)            \
  MACRO(sum_fp64, 0.0, double, ##__VA_ARGS__)           \
  MACRO(prod_fp64, 1.0, double, ##__VA_ARGS__)          \
  MACRO(max_fp64, -CINN_FP64_MAX, double, ##__VA_ARGS__) \
  MACRO(min_fp64, CINN_FP64_MAX, double, ##__VA_ARGS__)

__device__ inline double cinn_sum_fp64(const double left, const double right) { return left + right; }
__device__ inline double cinn_prod_fp64(const double left, const double right) { return left * right; }
__device__ inline double cinn_max_fp64(const double left, const double right) { return max(left, right); }
__device__ inline double cinn_min_fp64(const double left, const double right) { return min(left, right); }

#define EXPAND_REDUCE_BOOL_MACRO(MACRO, ...) \
  MACRO(all, true, bool, ##__VA_ARGS__)      \
  MACRO(any, false, bool, ##__VA_ARGS__)

__device__ inline bool cinn_all(const bool left, const bool right) { return left && right; }
__device__ inline bool cinn_any(const bool left, const bool right) { return left || right; }

#define CINN_WARP_SHUFFLE_INTERNAL_IMPL(REDUCE_TYPE, INITIAL_VALUE, DTYPE)                \
  __device__ inline DTYPE cinn_warp_shuffle_##REDUCE_TYPE##_internal(const DTYPE value) { \
    DTYPE tmp_val = value;                                                                \
    unsigned int mask = __activemask();                                                   \
    unsigned int lane = __popc(mask);                                                     \
    if (lane < 32) {                                                                      \
      for (int offset = 16; offset > 0; offset >>= 1) {                                   \
        DTYPE shfl_res = __shfl_down_sync(mask, tmp_val, offset);                         \
        if ((threadIdx.x & 0x1f) + offset >= lane) {                                      \
          shfl_res = (DTYPE)(INITIAL_VALUE);                                              \
        }                                                                                 \
        tmp_val = cinn_##REDUCE_TYPE(tmp_val, shfl_res);                                  \
      }                                                                                   \
    } else {                                                                              \
      for (int offset = 16; offset > 0; offset >>= 1) {                                   \
        tmp_val = cinn_##REDUCE_TYPE(tmp_val, __shfl_xor_sync(mask, tmp_val, offset));    \
      }                                                                                   \
    }                                                                                     \
    return tmp_val;                                                                       \
  }

EXPAND_REDUCE_INT32_MARCO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_INT64_MARCO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_FP32_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_FP64_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
EXPAND_REDUCE_BOOL_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_WARP_SHUFFLE_INTERNAL_IMPL)
#endif

#define CINN_BLOCK_REDUCE_IMPL(DTYPE, cinn_warp_shuffle_internal)  \
  DTYPE tmp_val = cinn_warp_shuffle_internal(value);               \
  if (return_warp || blockDim.x <= 32) {                           \
    return tmp_val;                                                \
  }                                                                \
  __syncthreads();                                                 \
  if (threadIdx.x % 32 == 0) {                                     \
    shm[threadIdx.x / 32] = tmp_val;                               \
  }                                                                \
  __syncthreads();                                                 \
  if (threadIdx.x < (blockDim.x + 31) / 32) {                      \
    tmp_val = cinn_warp_shuffle_internal(shm[threadIdx.x]);        \
    if (threadIdx.x == 0) {                                        \
      shm[0] = tmp_val;                                            \
    }                                                              \
  }                                                                \
  __syncthreads();                                                 \
  return shm[0];

#define CINN_BLOCK_REDUCE_MACRO(REDUCE_TYPE, INITIAL_VALUE, DTYPE) \
  __device__ inline DTYPE cinn_block_reduce_##REDUCE_TYPE(const DTYPE value, DTYPE* shm, bool return_warp = false) { \
    CINN_BLOCK_REDUCE_IMPL(DTYPE, cinn_warp_shuffle_##REDUCE_TYPE##_internal); \
  }

EXPAND_REDUCE_INT32_MARCO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_INT64_MARCO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_FP32_MACRO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_FP64_MACRO(CINN_BLOCK_REDUCE_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_BLOCK_REDUCE_MACRO)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_BLOCK_REDUCE_MACRO)
#endif



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

EXPAND_REDUCE_INT32_MARCO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_INT64_MARCO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_FP32_MACRO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_FP64_MACRO(CINN_DISCRETE_REDUCE_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_DISCRETE_REDUCE_MACRO)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_DISCRETE_REDUCE_MACRO)
#endif

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

EXPAND_REDUCE_INT32_MARCO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_INT64_MARCO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_FP32_MACRO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_FP64_MACRO(CINN_GRID_REDUCE_MACRO)
EXPAND_REDUCE_BOOL_MACRO(CINN_GRID_REDUCE_MACRO)

#ifdef CINN_CUDA_FP16
EXPAND_REDUCE_FP16_MACRO(CINN_GRID_REDUCE_MACRO)
#endif

#undef EXPAND_REDUCE_INT32_MARCO
#undef EXPAND_REDUCE_INT64_MARCO
#undef EXPAND_REDUCE_FP32_MACRO
#undef EXPAND_REDUCE_FP64_MACRO
#undef EXPAND_REDUCE_BOOL_MACRO

#ifdef CINN_CUDA_FP16
#undef EXPAND_REDUCE_FP16_MACRO
#endif

// ====================== start arg index ======================
#define ARGIDX_STRUCT_MACRO(TYPENAME, DTYPE, ITYPE) \
  struct TYPENAME { \
    DTYPE value; \
    ITYPE index; \
    __device__ TYPENAME() {} \
    __device__ TYPENAME(DTYPE value, ITYPE index) : value(value), index(index) {} \
    __device__ explicit operator ITYPE() const { return index; } \
  };

#define ARGIDX_SHFL_SYNC_MACRO(TYPENAME, DTYPE, ITYPE, SHFL_FUNC, ARG2_TYPE, ARG2) \
  __device__ inline TYPENAME SHFL_FUNC(unsigned mask, const TYPENAME& var, ARG2_TYPE ARG2, int width = 32) { \
    DTYPE value = SHFL_FUNC(mask, var.value, ARG2, width);                     \
    ITYPE index = SHFL_FUNC(mask, var.index, ARG2, width);                     \
    return {value, index};                                                     \
  }

#define ARGIDX_COMBINE_MACRO(TYPENAME) \
  __device__ inline TYPENAME cinn_min_##TYPENAME(TYPENAME a, TYPENAME b) { \
    return a.value < b.value ? a : b; \
  } \
  __device__ inline TYPENAME cinn_max_##TYPENAME(TYPENAME a, TYPENAME b) { \
    return a.value > b.value ? a : b; \
  } \
  __device__ inline TYPENAME min(TYPENAME a, TYPENAME b) { return cinn_min_##TYPENAME(a, b); } \
  __device__ inline TYPENAME max(TYPENAME a, TYPENAME b) { return cinn_max_##TYPENAME(a, b); }

#define ARGIDX_REDUCE_MACRO(TYPENAME, METHOD, DINIT) \
  CINN_WARP_SHUFFLE_INTERNAL_IMPL(METHOD##_##TYPENAME, TYPENAME(DINIT, 0), TYPENAME) \
  CINN_BLOCK_REDUCE_MACRO(METHOD##_##TYPENAME, TYPENAME(DINIT, 0), TYPENAME) \
  CINN_DISCRETE_REDUCE_MACRO(METHOD##_##TYPENAME, TYPENAME(DINIT, 0), TYPENAME) \
  CINN_GRID_REDUCE_MACRO(METHOD##_##TYPENAME, TYPENAME(DINIT, 0), TYPENAME)

#define EXPAND_ARGIDX_MACRO(DTYPE, DNAME, DMIN, DMAX, ITYPE, INAME) \
  ARGIDX_STRUCT_MACRO(argidx_##DNAME##_##INAME, DTYPE, ITYPE) \
  ARGIDX_COMBINE_MACRO(argidx_##DNAME##_##INAME) \
  ARGIDX_SHFL_SYNC_MACRO(argidx_##DNAME##_##INAME, DTYPE, ITYPE, __shfl_down_sync, unsigned, delta) \
  ARGIDX_SHFL_SYNC_MACRO(argidx_##DNAME##_##INAME, DTYPE, ITYPE, __shfl_xor_sync, int, laneMask) \
  ARGIDX_REDUCE_MACRO(argidx_##DNAME##_##INAME, min, DMAX) \
  ARGIDX_REDUCE_MACRO(argidx_##DNAME##_##INAME, max, DMIN)

#define EXPAND_ARGIDX_DTYPE_MACRO(DTYPE, DNAME, DMIN, DMAX) \
  EXPAND_ARGIDX_MACRO(DTYPE, DNAME, DMIN, DMAX, int, i32) \
  EXPAND_ARGIDX_MACRO(DTYPE, DNAME, DMIN, DMAX, int64_t, i64)

EXPAND_ARGIDX_DTYPE_MACRO(float16, fp16, -CINN_FP16_MAX, CINN_FP16_MAX)
EXPAND_ARGIDX_DTYPE_MACRO(float,   fp32, -CINN_FP32_MAX, CINN_FP32_MAX)    // fixme: temp
EXPAND_ARGIDX_DTYPE_MACRO(double,  fp64, -CINN_FP32_MAX, CINN_FP32_MAX)    // fixme: temp
EXPAND_ARGIDX_DTYPE_MACRO(int16_t, i16,  -32768,   32767)
EXPAND_ARGIDX_DTYPE_MACRO(int,     i32,  CINN_INT32_MIN,   CINN_INT32_MAX)  // fixme: temp
EXPAND_ARGIDX_DTYPE_MACRO(int64_t, i64,  CINN_INT32_MIN,   CINN_INT32_MAX)  // fixme: temp
EXPAND_ARGIDX_DTYPE_MACRO(uint8_t, u8,   0,   255)
// ====================== end arg index ======================

#undef CINN_BLOCK_REDUCE_IMPL
#undef CINN_BLOCK_REDUCE_MACRO
#undef CINN_GRID_REDUCE_IMPL
#undef CINN_GRID_REDUCE_MACRO
#undef CINN_WARP_SHUFFLE_INTERNAL_IMPL
#undef CINN_DISCRETE_REDUCE_IMPL
#undef CINN_DISCRETE_REDUCE_MACRO
