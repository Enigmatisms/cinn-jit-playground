#include "include/tensor.cuh"
#include "include/reduce.cuh"
#include "include/semaphore.cuh"
#include <thread>
#include <chrono>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

#define FLOAT4(v) (*(reinterpret_cast<float4*>(&v)))
#define CONST_FLOAT4(v) (*(reinterpret_cast<const float4*>(&v)))

__global__
void __launch_bounds__(512) kernel_reduce(
    const float* __restrict__ var, 
    float* __restrict__ var_1, 
    float* __restrict__ var_0, 
    float* __restrict__ var_3, 
    float* __restrict__ var_0_rf_0, 
    float* __restrict__ var_3_rf_0, 
    int32_t* __restrict__ semaphore
) {
  __builtin_assume(((int)blockIdx.x < 6));
  __builtin_assume(((int)blockIdx.y < 8));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 16));
  float _var_0_rf_temp_buffer [ 1 ];
  float _var_3_rf_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float *shm32__fp32_reduce = (float*)&dyn_shared_buffer[ 0 ];
  bool _is_last_block_done_temp_buffer [ 1 ];
  bool* is_last_block_done = _is_last_block_done_temp_buffer;
  float* var_0_rf = _var_0_rf_temp_buffer;
  float* var_0_rf__reduce_init = _var_0_rf_temp_buffer;
  float* var_3_rf = _var_3_rf_temp_buffer;
  float* var_3_rf__reduce_init = _var_3_rf_temp_buffer;
  float* var_0_rf_0__reduce_init = var_0_rf_0;
  float* var_3_rf_0__reduce_init = var_3_rf_0;
  var_0_rf__reduce_init[0] = 0.00000000f;
  var_3_rf__reduce_init[0] = 0.00000000f;
  for (int32_t reduce_k_0_0_reduce_k_1_0_reduce_k_2_0_fused = 0; reduce_k_0_0_reduce_k_1_0_reduce_k_2_0_fused < 1568; reduce_k_0_0_reduce_k_1_0_reduce_k_2_0_fused += 1) {
    float var_local = var[((((((reduce_k_0_0_reduce_k_1_0_reduce_k_2_0_fused * 128) + (int)threadIdx.y) + ((int)blockIdx.y * 16)) * 192) + ((int)blockIdx.x * 32)) + (int)threadIdx.x)];
    var_0_rf[0] = (var_0_rf[0] + var_local);
    var_3_rf[0] = (var_3_rf[0] + (var_local * var_local));
  };
  var_3_rf_0__reduce_init[((((int)blockIdx.x * 32) + (int)threadIdx.x) + ((int)blockIdx.y * 192))] = 0.00000000f;
  var_3_rf_0[((((int)blockIdx.x * 32) + (int)threadIdx.x) + ((int)blockIdx.y * 192))] = cinn_discrete_reduce_sum_fp32(var_3_rf[0], shm32__fp32_reduce);
  var_0_rf_0__reduce_init[((((int)blockIdx.x * 32) + (int)threadIdx.x) + ((int)blockIdx.y * 192))] = 0.00000000f;
  var_0_rf_0[((((int)blockIdx.x * 32) + (int)threadIdx.x) + ((int)blockIdx.y * 192))] = cinn_discrete_reduce_sum_fp32(var_0_rf[0], shm32__fp32_reduce);
  is_last_block_done[0] = cinn_grid_reduce_update_semaphore(semaphore);
  if (is_last_block_done[0]) {
    var_3[(((int)blockIdx.x * 32) + (int)threadIdx.x)] = cinn_grid_reduce_sum_fp32(var_3_rf_0, 192, (((int)blockIdx.x * 32) + (int)threadIdx.x));
    var_0[(((int)blockIdx.x * 32) + (int)threadIdx.x)] = cinn_grid_reduce_sum_fp32(var_0_rf_0, 192, (((int)blockIdx.x * 32) + (int)threadIdx.x));
  };
  if (((int)threadIdx.y == 0)) {
    if (is_last_block_done[0]) {
      var_1[(((int)blockIdx.x * 32) + (int)threadIdx.x)] = var_0[(((int)blockIdx.x * 32) + (int)threadIdx.x)];
    };
  };
}

template <int C>
__global__
void __launch_bounds__(192) kernel_64_56_192_div(
  const float* __restrict__ var_0, 
  const float* __restrict__ var_3, 
  const float* __restrict__ var_11, 
  float* __restrict__ var_16, 
  float* __restrict__ var_21, 
  const float* __restrict__ var_29, 
  const float* __restrict__ var_32, 
  float* __restrict__ var_35, 
  float* __restrict__ var_37
) {
  __builtin_assume(((int)blockIdx.x < 50176));
  __builtin_assume(((int)threadIdx.x < C));
  float* var_27 = var_16;
  float* var_28 = var_21;
  if (((int)blockIdx.x == 0)) {
    var_27[threadIdx.x] = ((0.899999976f * var_16[threadIdx.x]) + (0.100000024f * (var_0[threadIdx.x] / 200704.000f)));
    var_37[threadIdx.x] = rsqrtf((((var_3[threadIdx.x] / 200704.000f) - ((var_0[threadIdx.x] / 200704.000f) * (var_0[threadIdx.x] / 200704.000f))) + 9.99999975e-06f));
  };
  constexpr int b_proc = C * 4;
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    float var_11_local = var_11[(((i_j_k_a_fused_6 * C) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc))];
    float var_0_local = var_0[threadIdx.x];
    float var_3_local = var_3[threadIdx.x];
    float var_29_local = var_29[threadIdx.x];
    float var_32_local = var_32[threadIdx.x];
    var_35[(((i_j_k_a_fused_6 * C) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc))] = ((((var_11_local - (var_0_local / 200704.000f)) * rsqrtf((((var_3_local / 200704.000f) - ((var_0_local / 200704.000f) * (var_0_local / 200704.000f))) + 9.99999975e-06f))) * var_29_local) + var_32_local);
  };
  if (((int)blockIdx.x == 0)) {
    var_28[threadIdx.x] = ((0.899999976f * var_21[threadIdx.x]) + (0.100000024f * ((var_3[threadIdx.x] / 200704.000f) - ((var_0[threadIdx.x] / 200704.000f) * (var_0[threadIdx.x] / 200704.000f)))));
  };
}

template <int C>
__global__
void __launch_bounds__(192) kernel_64_56_192_div_shared(
  const float* __restrict__ var_0, 
  const float* __restrict__ var_3, 
  const float* __restrict__ var_11, 
  float* __restrict__ var_16, 
  float* __restrict__ var_21, 
  const float* __restrict__ var_29, 
  const float* __restrict__ var_32, 
  float* __restrict__ var_35, 
  float* __restrict__ var_37
) {
  __builtin_assume(((int)blockIdx.x < 50176));
  __builtin_assume(((int)threadIdx.x < C));
  float* var_27 = var_16;
  float* var_28 = var_21;
  if (((int)blockIdx.x == 0)) {
    var_27[threadIdx.x] = ((0.899999976f * var_16[threadIdx.x]) + (0.100000024f * (var_0[threadIdx.x] / 200704.000f)));
    var_37[threadIdx.x] = rsqrtf((((var_3[threadIdx.x] / 200704.000f) - ((var_0[threadIdx.x] / 200704.000f) * (var_0[threadIdx.x] / 200704.000f))) + 9.99999975e-06f));
  };
  constexpr int b_proc = C * 4;
  extern __shared__ float value_cache[];    // caching var_11
  FLOAT4(value_cache[threadIdx.x * 4]) = CONST_FLOAT4(var_11[(int)blockIdx.x * b_proc + threadIdx.x * 4]);
  __syncthreads();
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    float var_11_local = value_cache[(((i_j_k_a_fused_6 * C) + (int)threadIdx.x))];
    float var_0_local  = var_0[threadIdx.x];
    float var_3_local  = var_3[threadIdx.x];
    float var_29_local = var_29[threadIdx.x];
    float var_32_local = var_32[threadIdx.x];
    var_35[(((i_j_k_a_fused_6 * C) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc))] = ((((var_11_local - (var_0_local / 200704.000f)) * rsqrtf((((var_3_local / 200704.000f) - ((var_0_local / 200704.000f) * (var_0_local / 200704.000f))) + 9.99999975e-06f))) * var_29_local) + var_32_local);
  };
  if (((int)blockIdx.x == 0)) {
    var_28[threadIdx.x] = ((0.899999976f * var_21[threadIdx.x]) + (0.100000024f * ((var_3[threadIdx.x] / 200704.000f) - ((var_0[threadIdx.x] / 200704.000f) * (var_0[threadIdx.x] / 200704.000f)))));
  };
}

template <int C, int coarsening = 4>
__global__
void __launch_bounds__(256) kernel_64_56_192_shared_full(
    const float* __restrict__ var_0,        // 192
    const float* __restrict__ var_3,        // 192
    const float* __restrict__ var_11,       // 1024 * num_blocks (4 * 256 * num_blocks)
    float* __restrict__ var_16,             // 192
    float* __restrict__ var_21,             // 192
    const float* __restrict__ var_29,       // 192
    const float* __restrict__ var_32,       // 192
    float* __restrict__ var_35,             // 1024 * num_blocks (4 * 256 * num_blocks)
    float* __restrict__ var_37              // 192
) {
  __builtin_assume(((int)blockIdx.x < 37632));
  __builtin_assume(((int)threadIdx.x < 256));
  __shared__ float caching[C * 4];          
  extern __shared__ float value_cache[];    // caching var_11
  float* var_27 = var_16;
  float* var_28 = var_21;
  float* var_0_s  = &caching[0];
  float* var_3_s  = &caching[C * 1];
  float* var_29_s = &caching[C * 2];
  float* var_32_s = &caching[C * 3];
  
  // zero copy optimization? pipeline?
  constexpr int b_proc = 256 * coarsening;
  // the following code is coalesced (for 32-thread warp)
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    var_0_s[i]  =  var_0[(i + blockIdx.x * b_proc) % C];
    var_3_s[i]  =  var_3[(i + blockIdx.x * b_proc) % C];
    var_29_s[i] =  var_29[(i + blockIdx.x * b_proc) % C];
    var_32_s[i] =  var_32[(i + blockIdx.x * b_proc) % C];
  }
  FLOAT4(value_cache[threadIdx.x * 4]) = CONST_FLOAT4(var_11[(int)blockIdx.x * b_proc + threadIdx.x * 4]);

  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_0 = 0; append_var_0_append_var_1_append_var_2_i_fused_0 < coarsening; append_var_0_append_var_1_append_var_2_i_fused_0 += 1) {
    if ((((((((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) % 602112ll) / 10752ll) * 56ll) + (((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) % 10752ll) / 192ll)) + (((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) / 602112ll) * 3136ll)) == 0ll)) {
      var_27[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 192)] = ((0.899999976f * var_16[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 192)]) + (0.100000024f * (var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 192)] / 200704.000f)));
    };
  };
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_3 = 0; append_var_0_append_var_1_append_var_2_i_fused_3 < coarsening; append_var_0_append_var_1_append_var_2_i_fused_3 += 1) {
    if ((((((((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) % 602112ll) / 10752ll) * 56ll) + (((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) % 10752ll) / 192ll)) + (((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) / 602112ll) * 3136ll)) == 0ll)) {
      var_37[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 192)] = rsqrtf((((var_3[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 192)] / 200704.000f) - ((var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 192)] / 200704.000f) * (var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 192)] / 200704.000f))) + 9.99999975e-06f));
    };
  };

  __syncthreads();
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < coarsening; i_j_k_a_fused_6 += 1) {
    float var_11_local = value_cache[(((i_j_k_a_fused_6 * 256) + (int)threadIdx.x))];
    float var_0_local = var_0_s[(i_j_k_a_fused_6 * 256 + threadIdx.x) % C];
    float var_3_local = var_3_s[(i_j_k_a_fused_6 * 256 + threadIdx.x) % C];
    float var_29_local = var_29_s[(i_j_k_a_fused_6 * 256 + threadIdx.x) % C];
    float var_32_local = var_32_s[(i_j_k_a_fused_6 * 256 + threadIdx.x) % C];
    var_35[(((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc))] = ((((var_11_local - (var_0_local / 200704.000f)) * rsqrtf((((var_3_local / 200704.000f) - ((var_0_local / 200704.000f) * (var_0_local / 200704.000f))) + 9.99999975e-06f))) * var_29_local) + var_32_local);
  };
  // 一长串逻辑只为了限制某个情况下（blockidx与threadidx的组合）为0时输出，以免多次输出
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_6 = 0; append_var_0_append_var_1_append_var_2_i_fused_6 < coarsening; append_var_0_append_var_1_append_var_2_i_fused_6 += 1) {
    if (((((((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) % 602112ll) / 10752ll) * 56ll) + (((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) % 10752ll) / 192ll)) + (((((((int)blockIdx.x * coarsening) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) / 602112ll) * 3136ll) == 0ll)) {
      var_28[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)] = ((0.899999976f * var_21[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)]) + (0.100000024f * ((var_3[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)] / 200704.000f) - ((var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)] / 200704.000f) * (var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)] / 200704.000f)))));
    };
  };
}

__global__
void __launch_bounds__(256) kernel_64_56_192(
    const float* __restrict__ var_0, 
    const float* __restrict__ var_3, 
    const float* __restrict__ var_11, 
    float* __restrict__ var_16, 
    float* __restrict__ var_21, 
    const float* __restrict__ var_29, 
    const float* __restrict__ var_32, 
    float* __restrict__ var_35, 
    float* __restrict__ var_37)
{
  __builtin_assume(((int)blockIdx.x < 37632));
  __builtin_assume(((int)threadIdx.x < 256));
  float* var_27 = var_16;
  float* var_28 = var_21;
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_0 = 0; append_var_0_append_var_1_append_var_2_i_fused_0 < 4; append_var_0_append_var_1_append_var_2_i_fused_0 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) % 602112ll) / 10752ll) * 56ll) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) % 10752ll) / 192ll)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) / 602112ll) * 3136ll)) == 0ll)) {
      var_27[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 192)] = ((0.899999976f * var_16[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 192)]) + (0.100000024f * (var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 192)] / 200704.000f)));
    };
  };
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_3 = 0; append_var_0_append_var_1_append_var_2_i_fused_3 < 4; append_var_0_append_var_1_append_var_2_i_fused_3 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) % 602112ll) / 10752ll) * 56ll) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) % 10752ll) / 192ll)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) / 602112ll) * 3136ll)) == 0ll)) {
      var_37[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 192)] = rsqrtf((((var_3[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 192)] / 200704.000f) - ((var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 192)] / 200704.000f) * (var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 192)] / 200704.000f))) + 9.99999975e-06f));
    };
  };

  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    float var_11_local = var_11[(((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024))];
    float var_0_local = var_0[((((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024)) % 192)];
    float var_3_local = var_3[((((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024)) % 192)];
    float var_29_local = var_29[((((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024)) % 192)];
    float var_32_local = var_32[((((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024)) % 192)];
    
    var_35[(((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024))] = ((((var_11_local - (var_0_local / 200704.000f)) * rsqrtf((((var_3_local / 200704.000f) - ((var_0_local / 200704.000f) * (var_0_local / 200704.000f))) + 9.99999975e-06f))) * var_29_local) + var_32_local);
  };
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_6 = 0; append_var_0_append_var_1_append_var_2_i_fused_6 < 4; append_var_0_append_var_1_append_var_2_i_fused_6 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) % 602112ll) / 10752ll) * 56ll) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) % 10752ll) / 192ll)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) / 602112ll) * 3136ll)) == 0ll)) {
      var_28[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)] = ((0.899999976f * var_21[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)]) + (0.100000024f * ((var_3[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)] / 200704.000f) - ((var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)] / 200704.000f) * (var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 192)] / 200704.000f)))));
    };
  };
}

int main()
{
    std::cout << "Initializing data..." << std::endl;
    constexpr size_t N = 64, H = 56, W = 56, C = 192;
    Tensor<curandState> gen_input(N * H * W * C, RandInit{2024});
    Tensor<curandState> gen_weights(C, RandInit{2024});

    Tensor<float> X(N * H * W * C, Randn{gen_input});

    Tensor<float> var0(C, Zero<float>{});
    Tensor<float> var1(C, Zero<float>{});
    Tensor<float> var3(C, Zero<float>{});


    Tensor<float> var_0_rf(1536, Zero<float>{});
    Tensor<float> var_3_rf(1536, Zero<float>{});
    Tensor<int> semaphores(32, Zero<int>{});

    kernel_reduce<<<dim3(6, 8), dim3(32, 16), 2048>>>(X, var1, var0, var3, var_0_rf, var_3_rf, semaphores);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    Tensor<float> w1(C, Rand{gen_weights, 1, 2}); 
    Tensor<float> w2(C, Rand{gen_weights, 1, 2}); 

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    Tensor<float> out2(C, Zero<float>{});
    Tensor<float> out3(N * H * W * C, Zero<float>{});
    Tensor<float> out4(C, Zero<float>{});

    // used for comparison
    Tensor<float> var1_cp = var1;
    Tensor<float> out2_cp = out2;
    Tensor<float> out3_cp = out3;
    Tensor<float> out4_cp = out4;

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::cout << "Launching kernel..." << std::endl;

    constexpr int num_blocks = N * W * H / 4;
    constexpr int dyna_shared = C * 4 * sizeof(float);

    kernel_64_56_192_div_shared<C><<<num_blocks, C, dyna_shared>>>(var0, var3, X, var1, out2, w1, w2, out3, out4);
    kernel_64_56_192<<<37632, 256>>>(var0, var3, X, var1_cp, out2_cp, w1, w2, out3_cp, out4_cp);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::cout << "Comparing results..." << std::endl;
    out3_cp.to_host();
    out3.to_host();
    out3_cp.compare(out3);
    std::cout << "Comparison completed." << std::endl;

    return 0;
}