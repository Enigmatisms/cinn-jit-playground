#include "include/reduce.cuh"
#include "include/tensor.cuh"      
#include "include/soa_reduce.cuh"

__global__
void __launch_bounds__(256) block_reduce_argmax_small(
  const float* __restrict__ var /* _Buffer_<cinn_buffer_t*: S0,S1,32ll>(_var) */, 
  int* __restrict__ var_1 /* _Buffer_<cinn_buffer_t*: S0,S1>(_var_1) */, 
  int64_t S0, int64_t S1
) {
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) / 64ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8ll));
  argidx_fp32_i32 _var_0_loopalign_1_rf_temp_buffer [ 8ll ];
  argidx_fp32_i32 _var_0_loopalign_1_temp_buffer [ 8ll ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  argidx_fp32_i32* shm32__fp32_reduce = (argidx_fp32_i32*)&dyn_shared_buffer[ 0 ];
  argidx_fp32_i32* var_0_loopalign_1 = _var_0_loopalign_1_temp_buffer;
  argidx_fp32_i32* var_0_loopalign_1_rf = _var_0_loopalign_1_rf_temp_buffer;
  argidx_fp32_i32* var_0_loopalign_1_rf__reduce_init = _var_0_loopalign_1_rf_temp_buffer;
  for (int32_t i_j_fused_3 = 0ll; i_j_fused_3 < 8ll; i_j_fused_3 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_fused_3) * 8ll) + (int)threadIdx.y) < (S0 * S1))) {
      var_0_loopalign_1_rf__reduce_init[i_j_fused_3] = {-3.40282347e+38f, 0};
      for (int32_t reduce_k_0_2 = 0; reduce_k_0_2 < 1; reduce_k_0_2 += 1) {
        float var_local = var[(((((((int)blockIdx.x * 8ll) + i_j_fused_3) * 8ll) + (int)threadIdx.y) * 32ll) + (int)threadIdx.x)];
        var_0_loopalign_1_rf[i_j_fused_3] = max(var_0_loopalign_1_rf[i_j_fused_3], {var_local, (int)threadIdx.x});
      };
    };
  };
  for (int32_t i_j_fused_3 = 0ll; i_j_fused_3 < 8ll; i_j_fused_3 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_fused_3) * 8ll) + (int)threadIdx.y) < (S0 * S1))) {
      var_0_loopalign_1[i_j_fused_3] = cinn_block_reduce_max_argidx_fp32_i32(var_0_loopalign_1_rf[i_j_fused_3], shm32__fp32_reduce, true);
    };
  };
  for (int32_t i_j_fused_0 = 0ll; i_j_fused_0 < 8ll; i_j_fused_0 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_fused_0) * 8ll) + (int)threadIdx.y) < (S0 * S1))) {
      if (((int)threadIdx.x == 0)) {
        var_1[(((((int)blockIdx.x * 8ll) + i_j_fused_0) * 8ll) + (int)threadIdx.y)] = (int)var_0_loopalign_1[i_j_fused_0];
      };
    };
  };
}

__global__
void __launch_bounds__(256) block_reduce_argmax_small_improved(
  const float* __restrict__ var /* _Buffer_<cinn_buffer_t*: S0,S1,32ll>(_var) */, 
  int* __restrict__ var_1 /* _Buffer_<cinn_buffer_t*: S0,S1>(_var_1) */, 
  int64_t S0, int64_t S1
) {
  __builtin_assume(((int)blockIdx.x < (((S0 * S1) / 64ll) + 1ll)));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8ll));
  argidx_fp32_i32 _var_0_loopalign_1_rf_temp_buffer [ 1 ];
  argidx_fp32_i32 _var_0_loopalign_1_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  argidx_fp32_i32* shm32__fp32_reduce = (argidx_fp32_i32*)&dyn_shared_buffer[ 0 ];
  argidx_fp32_i32* var_0_loopalign_1 = _var_0_loopalign_1_temp_buffer;
  argidx_fp32_i32* var_0_loopalign_1_rf = _var_0_loopalign_1_rf_temp_buffer;
  argidx_fp32_i32* var_0_loopalign_1_rf__reduce_init = _var_0_loopalign_1_rf_temp_buffer;
  for (int32_t i_j_fused_3 = 0ll; i_j_fused_3 < 8ll; i_j_fused_3 += 1) {
    if (((((((int)blockIdx.x * 8ll) + i_j_fused_3) * 8ll) + (int)threadIdx.y) < (S0 * S1))) {
      var_0_loopalign_1_rf__reduce_init[0] = {-3.40282347e+38f, 0};
      float var_local = var[(((((((int)blockIdx.x * 8ll) + i_j_fused_3) * 8ll) + (int)threadIdx.y) * 32ll) + (int)threadIdx.x)];
      var_0_loopalign_1_rf[0] = max(var_0_loopalign_1_rf[0], {var_local, (int)threadIdx.x});
      var_0_loopalign_1[0] = cinn_block_reduce_max_argidx_fp32_i32(var_0_loopalign_1_rf[0], shm32__fp32_reduce, true);
      if (((int)threadIdx.x == 0)) {
        var_1[(((((int)blockIdx.x * 8ll) + i_j_fused_3) * 8ll) + (int)threadIdx.y)] = (int)var_0_loopalign_1[0];
      };
    };
  };
}


__global__
void __launch_bounds__(256) block_reduce_argmax_big(
    const float* __restrict__ var /* _Buffer_<cinn_buffer_t*: S0,S1,2048ll>(_var) */, 
    int* __restrict__ var_1 /* _Buffer_<cinn_buffer_t*: S0,S1>(_var_1) */, 
    int64_t S0, 
    int64_t S1
) {
  __builtin_assume(((int)blockIdx.x < (S0 * S1)));
  __builtin_assume(((int)threadIdx.x < 256));
  argidx_fp32_i32 _var_0_loopalign_2_rf_temp_buffer [ 1 ];
  argidx_fp32_i32 _var_0_loopalign_2_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  argidx_fp32_i32 *shm32__fp32_reduce = (argidx_fp32_i32*)&dyn_shared_buffer[ 0 ];
  argidx_fp32_i32* var_0_loopalign_2 = _var_0_loopalign_2_temp_buffer;
  argidx_fp32_i32* var_0_loopalign_2_rf = _var_0_loopalign_2_rf_temp_buffer;
  argidx_fp32_i32* var_0_loopalign_2_rf__reduce_init = _var_0_loopalign_2_rf_temp_buffer;
  var_0_loopalign_2_rf__reduce_init[0ll] = {-3.40282347e+38f, 0};
  for (int32_t reduce_k_0_4_1 = 0; reduce_k_0_4_1 < 8; reduce_k_0_4_1 += 1) {
    float var_local_0 = var[(((reduce_k_0_4_1 * 256ll) + (int)threadIdx.x) + ((int)blockIdx.x * 2048ll))];
    var_0_loopalign_2_rf[0ll] = max(var_0_loopalign_2_rf[0ll], {var_local_0, (int)threadIdx.x % 2048});
  };
  var_0_loopalign_2[0ll] = cinn_block_reduce_max_argidx_fp32_i32(var_0_loopalign_2_rf[0ll], shm32__fp32_reduce, false);
  if (((int)threadIdx.x == 0)) {
    var_1[(int)blockIdx.x] = int(var_0_loopalign_2[0ll]);
  };
}

__global__
void __launch_bounds__(256) block_reduce_argmax_big_improved(
    const float* __restrict__ var /* _Buffer_<cinn_buffer_t*: S0,S1,2048ll>(_var) */, 
    int* __restrict__ var_1 /* _Buffer_<cinn_buffer_t*: S0,S1>(_var_1) */, 
    int64_t S0, 
    int64_t S1
) {
  __builtin_assume(((int)blockIdx.x < (S0 * S1)));
  __builtin_assume(((int)threadIdx.x < 256));
  argidx_fp32_i32 _var_0_loopalign_2_rf_temp_buffer [ 1 ];
  argidx_fp32_i32 _var_0_loopalign_2_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  float* shm32_value_buffer = (float*)&dyn_shared_buffer[ 0 ];
  int* shm32_index_buffer = (int*)&dyn_shared_buffer[ 32 * sizeof(float) ];
  argidx_fp32_i32* var_0_loopalign_2 = _var_0_loopalign_2_temp_buffer;
  argidx_fp32_i32* var_0_loopalign_2_rf = _var_0_loopalign_2_rf_temp_buffer;
  argidx_fp32_i32* var_0_loopalign_2_rf__reduce_init = _var_0_loopalign_2_rf_temp_buffer;
  var_0_loopalign_2_rf__reduce_init[0ll] = {-3.40282347e+38f, 0};
  for (int32_t reduce_k_0_4_1 = 0; reduce_k_0_4_1 < 8; reduce_k_0_4_1 += 1) {
    float var_local_0 = var[(((reduce_k_0_4_1 * 256ll) + (int)threadIdx.x) + ((int)blockIdx.x * 2048ll))];
    var_0_loopalign_2_rf[0ll] = max(var_0_loopalign_2_rf[0ll], {var_local_0, (int)threadIdx.x % 2048});
  };
  var_0_loopalign_2[0ll] = cinn_block_reduce_max_argidx_fp32_i32_soa(var_0_loopalign_2_rf[0ll], shm32_value_buffer, shm32_index_buffer, false);
  if (((int)threadIdx.x == 0)) {
    var_1[(int)blockIdx.x] = int(var_0_loopalign_2[0ll]);
  };
}

__global__
void __launch_bounds__(256) non_coalesced_argmax(
  const float* __restrict__ var /* _Buffer_<cinn_buffer_t*: 32ll,256ll,S0,S1>(_var) */, 
  int64_t* __restrict__ var_1 /* _Buffer_<cinn_buffer_t*: 32ll,256ll,S1>(_var_1) */, 
  int64_t S0, int64_t S1
) {
  __builtin_assume(((int)blockIdx.x < (S1 * 1024ll)));
  __builtin_assume(((int)threadIdx.x < 32));
  __builtin_assume(((int)threadIdx.y < 8ll));
  argidx_fp32_i64 _var_0_loopalign_8_rf_1_temp_buffer [ 1 ];
  argidx_fp32_i64 _var_0_loopalign_8_temp_buffer [ 1 ];
  extern __shared__ uint8_t dyn_shared_buffer[];
  argidx_fp32_i64* shm32__fp32_reduce = (argidx_fp32_i64*)&dyn_shared_buffer[ 0 ];
  argidx_fp32_i64* var_0_loopalign_8 = _var_0_loopalign_8_temp_buffer;
  argidx_fp32_i64* var_0_loopalign_8_rf_1 = _var_0_loopalign_8_rf_1_temp_buffer;
  argidx_fp32_i64* var_0_loopalign_8_rf_1__reduce_init = _var_0_loopalign_8_rf_1_temp_buffer;
  var_0_loopalign_8_rf_1__reduce_init[0ll] = {-3.40282347e+38f, 0};
  for (int32_t reduce_k_0_7_9 = 0; reduce_k_0_7_9 < 8; reduce_k_0_7_9 += 1) {
    // 啥玩意啊？竟然有一半是没有用的，相当于是 coarsening factor = 4 而不是 8
    if ((((reduce_k_0_7_9 * 32ll) + (int)threadIdx.x) < S0)) {
      int local_index = (((((((((int)blockIdx.x * 8ll) + (int)threadIdx.y) / S1) * S0) + (reduce_k_0_7_9 * 32ll)) + (int)threadIdx.x) * S1) + ((((int)blockIdx.x * 8ll) + (int)threadIdx.y) % S1));
      float var_local_5 = var[local_index];
      var_0_loopalign_8_rf_1[0ll] = max(var_0_loopalign_8_rf_1[0ll], {var_local_5, (local_index / S1) % S0});
    };
  };
  var_0_loopalign_8[0ll] = cinn_block_reduce_max_argidx_fp32_i64(var_0_loopalign_8_rf_1[0ll], shm32__fp32_reduce, true);
  if (((int)threadIdx.x == 0)) {
    if ((0ll < S0)) {
      var_1[(((int)blockIdx.x * 8ll) + (int)threadIdx.y)] = (int64_t)var_0_loopalign_8[0ll];
    };
  };
}

void coalesced_test() {
    std::cout << "Initializing data..." << std::endl;
    constexpr size_t Axis1 = 2048, Axis2 = 64, C = 2048;

    Tensor<curandState> gen_input(2048 * 64 * 32, RandInit{2024});
    Tensor<float> to_reduce(Axis1 * Axis2 * C, Randn{gen_input}, true);
    Tensor<int> indices(Axis1 * Axis2, Zero<int>{});
    Tensor<int> indices_improve(Axis1 * Axis2, Zero<int>{});

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::cout << "Launching kernel..." << std::endl;

    constexpr size_t smem_size = 32 * sizeof(argidx_fp32_i32);
    printf("Size: %llu, %d, %d, %d\n", smem_size, Axis1, Axis2, C);
    block_reduce_argmax_big<<<Axis1 * Axis2, 256, smem_size>>>(to_reduce, indices, Axis1, Axis2);
    block_reduce_argmax_big_improved<<<Axis1 * Axis2, 256, smem_size>>>(to_reduce, indices_improve, Axis1, Axis2);

    std::cout << "Comparing results... (1)" << std::endl;
    indices.to_host();
    indices_improve.to_host();
    indices.compare(indices_improve);
    std::cout << "Comparison completed. (1)" << std::endl;

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // constexpr size_t C2 = 32;

    // Tensor<float> to_reduce2(Axis1 * Axis2 * C2, Rand{gen_input, 0, 1});
    // CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // block_reduce_argmax_small<<<2049, dim3(32, 8), smem_size>>>(to_reduce2, indices, Axis1, Axis2);
    // block_reduce_argmax_small_improved<<<2049, dim3(32, 8), smem_size>>>(to_reduce2, indices_improve, Axis1, Axis2);
    // CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    // std::cout << "Comparing results... (2)" << std::endl;
    // indices.to_host();
    // indices_improve.to_host();
    // indices.compare(indices_improve);
    // std::cout << "Comparison completed. (2)" << std::endl;

    // to_reduce2.to_host();
    // for (int i = 0; i < 4; i++) {
    //   int index = indices[i];
    //   printf("max index: %d, value = %f, line:\n", index, to_reduce2[i * C2 + index]);
    //   for (int j = 0; j < C2; j++) {
    //     printf("(%d, %.4f), ", j, to_reduce2[i * C2 + j]);
    //   }
    //   printf("\n");
    // }
}

void non_coalesced_test() {
    std::cout << "Initializing data..." << std::endl;
    constexpr size_t C1 = 32, C2 = 256, Axis1 = 128, Axis2 = 64;

    Tensor<curandState> gen_input(131 * 67, RandInit{2024});
    Tensor<float> to_reduce(Axis1 * Axis2 * C1 * C2, Randn{gen_input}, true);
    Tensor<int64_t> indices(Axis2 * C1 * C2, Zero<int64_t>{});

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::cout << "Launching kernel..." << std::endl;

    constexpr size_t smem_size = 32 * sizeof(argidx_fp32_i64);
    non_coalesced_argmax<<<65536, dim3(32, 8), smem_size>>>(to_reduce, indices, Axis1, Axis2);

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}


int main() {
    coalesced_test();
    return 0;
}





