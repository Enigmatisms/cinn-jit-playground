#include "include/tensor.cuh"
#include "include/reduce.cuh"
#include "include/semaphore.cuh"
#include <thread>
#include <chrono>
#include <cuda_fp16.h>

#define FLOAT4(v) (*(reinterpret_cast<float4*>(&v)))
#define FLOAT2(v) (*(reinterpret_cast<float2*>(&v)))
#define CONST_FLOAT4(v) (*(reinterpret_cast<const float4*>(&v)))
#define CONST_FLOAT2(v) (*(reinterpret_cast<const float2*>(&v)))

using float16 = half;
// resnet fn_cast_full_div_reshape_bc_sub_mul_full_div_full_div_bc_mul_bc_mul_bc_mul_bc_sub_sub_bc_mul_cast_yield_cast_full_div_reshape_bc_sub_mul_full_div_full_div_bc_mul_bc_mul_bc_mul_bc_sub_sub_bc_mul_cast_yield__1
__global__
void __launch_bounds__(256) kernel(
  const float16* __restrict__ var /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var) */, 
  const float* __restrict__ var_2 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_2) */, 
  const float* __restrict__ var_7 /* _Buffer_<cinn_buffer_t*: 256ll>(_var_7) */, 
  const float* __restrict__ var_8 /* _Buffer_<cinn_buffer_t*: 256ll>(_var_8) */, 
  const float* __restrict__ var_11 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_11) */, 
  const float* __restrict__ var_14 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_14) */, 
  const float* __restrict__ var_23 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_23) */, 
  const float16* __restrict__ var_30 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_30) */, 
  const float* __restrict__ var_33 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_33) */, 
  const float* __restrict__ var_38 /* _Buffer_<cinn_buffer_t*: 256ll>(_var_38) */, 
  const float* __restrict__ var_39 /* _Buffer_<cinn_buffer_t*: 256ll>(_var_39) */, 
  const float* __restrict__ var_42 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_42) */, 
  const float* __restrict__ var_45 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_45) */, 
  const float* __restrict__ var_54 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_54) */, 
  float16* __restrict__ var_29 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_29) */, 
  float16* __restrict__ var_60 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_60) */
) {
  __builtin_assume(((int)blockIdx.x < 100352));
  __builtin_assume(((int)threadIdx.x < 256));
  constexpr float inv_size = 1.f / 401408.000f;
  for (int32_t i_j_k_fused_a_163_fused_0 = 0; i_j_k_fused_a_163_fused_0 < 4; i_j_k_fused_a_163_fused_0 += 1) {
    float var_23_local_13 = var_23[(((((int)blockIdx.x * 4) + i_j_k_fused_a_163_fused_0) * 256) + (int)threadIdx.x)];
    float16 var_local_125 = var[(((((int)blockIdx.x * 4) + i_j_k_fused_a_163_fused_0) * 256) + (int)threadIdx.x)];
    float var_7_local_11 = var_7[(int)threadIdx.x];
    float var_8_local_11 = var_8[(int)threadIdx.x];
    float var_11_local_11 = var_11[(int)threadIdx.x];
    float var_2_local_65 = var_2[(int)threadIdx.x];
    float var_14_local_38 = var_14[(int)threadIdx.x];
    var_29[(((((int)blockIdx.x * 4) + i_j_k_fused_a_163_fused_0) * 256) + (int)threadIdx.x)] = ((float16)(((var_7_local_11 * var_8_local_11) * ((var_23_local_13 - (var_11_local_11 * inv_size)) - ((((float)(var_local_125)) - (var_2_local_65 * inv_size)) * (((var_14_local_38 * inv_size) * var_8_local_11) * var_8_local_11))))));
  };
  for (int32_t i_j_k_fused_a_165_fused_0 = 0; i_j_k_fused_a_165_fused_0 < 4; i_j_k_fused_a_165_fused_0 += 1) {
    float var_54_local_1 = var_54[(((((int)blockIdx.x * 4) + i_j_k_fused_a_165_fused_0) * 256) + (int)threadIdx.x)];
    float16 var_30_local_1 = var_30[(((((int)blockIdx.x * 4) + i_j_k_fused_a_165_fused_0) * 256) + (int)threadIdx.x)];
    float var_38_local_1 = var_38[(int)threadIdx.x];
    float var_39_local_1 = var_39[(int)threadIdx.x];
    float var_42_local_1 = var_42[(int)threadIdx.x];
    float var_33_local_4 = var_33[(int)threadIdx.x];
    float var_45_local_1 = var_45[(int)threadIdx.x];
    var_60[(((((int)blockIdx.x * 4) + i_j_k_fused_a_165_fused_0) * 256) + (int)threadIdx.x)] = ((float16)(((var_38_local_1 * var_39_local_1) * ((var_54_local_1 - (var_42_local_1 * inv_size)) - ((((float)(var_30_local_1)) - (var_33_local_4 * inv_size)) * (((var_45_local_1 * inv_size) * var_39_local_1) * var_39_local_1))))));
  };
}

__global__
void kernel_improved(
  const float16* __restrict__ var /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var) */,
  const float* __restrict__ var_2 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_2) */,
  const float* __restrict__ var_7 /* _Buffer_<cinn_buffer_t*: 256ll>(_var_7) */,
  const float* __restrict__ var_8 /* _Buffer_<cinn_buffer_t*: 256ll>(_var_8) */,
  const float* __restrict__ var_11 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_11) */,
  const float* __restrict__ var_14 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_14) */,
  const float* __restrict__ var_23 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_23) */,
  const float16* __restrict__ var_30 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_30) */,
  const float* __restrict__ var_33 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_33) */,
  const float* __restrict__ var_38 /* _Buffer_<cinn_buffer_t*: 256ll>(_var_38) */,
  const float* __restrict__ var_39 /* _Buffer_<cinn_buffer_t*: 256ll>(_var_39) */,
  const float* __restrict__ var_42 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_42) */,
  const float* __restrict__ var_45 /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_45) */,
  const float* __restrict__ var_54 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_54) */,
  float16* __restrict__ var_29 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_29) */,
  float16* __restrict__ var_60 /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_60) */
) {
  __builtin_assume(((int)blockIdx.x < 100352));
  __builtin_assume(((int)blockIdx.y < 2));
  __builtin_assume(((int)threadIdx.x < 128));
  extern __shared__ float _buffer[];
  float* f32_data = &_buffer[0];
  float16* f16_data = reinterpret_cast<float16*>(&_buffer[128 * 4]);

  FLOAT4(f32_data[threadIdx.x * 4]) = CONST_FLOAT4(var_23[((threadIdx.x / 32) * 2 + blockIdx.y) * 128 + blockIdx.x * 4 * 256 + (threadIdx.x % 32) * 4]);
  FLOAT2(f16_data[threadIdx.x * 4]) = CONST_FLOAT2(var[((threadIdx.x / 32) * 2 + blockIdx.y) * 128 + blockIdx.x * 4 * 256 + (threadIdx.x % 32) * 4]);
  constexpr float inv_size = 1.f / 401408.000f;
  __syncthreads();
  for (int32_t i_j_k_fused_a_163_fused_0 = 0; i_j_k_fused_a_163_fused_0 < 4; i_j_k_fused_a_163_fused_0 += 1) {
    // float var_23_local_13 = var_23[(((((int)(blockIdx.x * 2 + blockIdx.y) * 4) + i_j_k_fused_a_163_fused_0) * 128) + (int)threadIdx.x)];
    float var_23_local_13 = f32_data[i_j_k_fused_a_163_fused_0 * 128 + (int)threadIdx.x];
    float16 var_local_125 = f16_data[i_j_k_fused_a_163_fused_0 * 128 + (int)threadIdx.x];
    // float16 var_local_125 = var[(((((int)(blockIdx.x * 2 + blockIdx.y) * 4) + i_j_k_fused_a_163_fused_0) * 128) + (int)threadIdx.x)];
    float var_7_local_11 = var_7[(int)threadIdx.x + blockIdx.y * 128];
    float var_8_local_11 = var_8[(int)threadIdx.x + blockIdx.y * 128];
    float var_11_local_11 = var_11[(int)threadIdx.x + blockIdx.y * 128];
    float var_2_local_65 = var_2[(int)threadIdx.x + blockIdx.y * 128];
    float var_14_local_38 = var_14[(int)threadIdx.x + blockIdx.y * 128];
    var_29[(blockIdx.x * 4 + i_j_k_fused_a_163_fused_0) * 256 + blockIdx.y * 128 + threadIdx.x] = ((float16)(((var_7_local_11 * var_8_local_11) * ((var_23_local_13 - (var_11_local_11 * inv_size)) - ((((float)(var_local_125)) - (var_2_local_65 * inv_size)) * (((var_14_local_38 * inv_size) * var_8_local_11) * var_8_local_11))))));
  };
  __syncthreads();
  FLOAT4(f32_data[threadIdx.x * 4]) = CONST_FLOAT4(var_54[((threadIdx.x / 32) * 2 + blockIdx.y) * 128 + blockIdx.x * 4 * 256 + (threadIdx.x % 32) * 4]);
  FLOAT2(f16_data[threadIdx.x * 4]) = CONST_FLOAT2(var_30[((threadIdx.x / 32) * 2 + blockIdx.y) * 128 + blockIdx.x * 4 * 256 + (threadIdx.x % 32) * 4]);
  __syncthreads();

  for (int32_t i_j_k_fused_a_165_fused_0 = 0; i_j_k_fused_a_165_fused_0 < 4; i_j_k_fused_a_165_fused_0 += 1) {
    // float var_54_local_1 = var_54[(((((int)(blockIdx.x * 2 + blockIdx.y) * 4) + i_j_k_fused_a_165_fused_0) * 128) + (int)threadIdx.x)];
    float var_54_local_1 = f32_data[i_j_k_fused_a_165_fused_0 * 128 + (int)threadIdx.x];
    float16 var_30_local_1 = f16_data[i_j_k_fused_a_165_fused_0 * 128 + (int)threadIdx.x];
    // float16 var_30_local_1 = var_30[(((((int)(blockIdx.x * 2 + blockIdx.y) * 4) + i_j_k_fused_a_165_fused_0) * 128) + (int)threadIdx.x)];
    float var_38_local_1 = var_38[(int)threadIdx.x + blockIdx.y * 128];
    float var_39_local_1 = var_39[(int)threadIdx.x + blockIdx.y * 128];
    float var_42_local_1 = var_42[(int)threadIdx.x + blockIdx.y * 128];
    float var_33_local_4 = var_33[(int)threadIdx.x + blockIdx.y * 128];
    float var_45_local_1 = var_45[(int)threadIdx.x + blockIdx.y * 128];
    var_60[(blockIdx.x * 4 + i_j_k_fused_a_165_fused_0) * 256 + blockIdx.y * 128 + threadIdx.x] = ((float16)(((var_38_local_1 * var_39_local_1) * ((var_54_local_1 - (var_42_local_1 * inv_size)) - ((((float)(var_30_local_1)) - (var_33_local_4 * inv_size)) * (((var_45_local_1 * inv_size) * var_39_local_1) * var_39_local_1))))));
  };
}


int main()
{
    std::cout << "Initializing data..." << std::endl;
    constexpr size_t N = 128, H = 56, W = 56, C = 256;

    Tensor<curandState> gen_input(256, RandInit{2024});
    Tensor<curandState> gen_weights(C, RandInit{2024});


    Tensor<half> var(N * H * W * C, Rand{gen_input, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var) */
    Tensor<float> var_2(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_2) */
    Tensor<float> var_7(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 256ll>(_var_7) */
    Tensor<float> var_8(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 256ll>(_var_8) */
    Tensor<float> var_11(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_11) */
    Tensor<float> var_14(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_14) */
    Tensor<float> var_23(N * H * W * C, Rand{gen_input, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_23) */
    Tensor<half> var_30(N * H * W * C, Rand{gen_input, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_30) */
    Tensor<float> var_33(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_33) */
    Tensor<float> var_38(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 256ll>(_var_38) */
    Tensor<float> var_39(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 256ll>(_var_39) */
    Tensor<float> var_42(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_42) */
    Tensor<float> var_45(C, Rand{gen_weights, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 1ll,1ll,1ll,256ll>(_var_45) */
    Tensor<float> var_54(N * H * W * C, Rand{gen_input, 0.01, 1}); /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var_54) */

    Tensor<half> var_29(N * H * W * C, Zero<half>{}); /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var) */
    Tensor<half> var_60(N * H * W * C, Zero<half>{}); /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var) */

    Tensor<half> var_29_new(N * H * W * C, Zero<half>{}); /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var) */
    Tensor<half> var_60_new(N * H * W * C, Zero<half>{}); /* _Buffer_<cinn_buffer_t*: 128ll,56ll,56ll,256ll>(_var) */


    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::cout << "Launching kernel..." << std::endl;

    kernel<<<100352, 256>>>(
      var,
      var_2,
      var_7,
      var_8,
      var_11,
      var_14,
      var_23,
      var_30,
      var_33,
      var_38,
      var_39,
      var_42,
      var_45,
      var_54,
      var_29,
      var_60
    );
    constexpr size_t smem_size = 128 * 4 * sizeof(float) + 128 * 4 * sizeof(half);
    kernel_improved<<<dim3(100352, 2), 128, smem_size>>>(
      var,
      var_2,
      var_7,
      var_8,
      var_11,
      var_14,
      var_23,
      var_30,
      var_33,
      var_38,
      var_39,
      var_42,
      var_45,
      var_54,
      var_29_new,
      var_60_new
    );
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::cout << "Comparing results..." << std::endl;
    var_29.to_host();
    var_29_new.to_host();
    var_29.compare(var_29_new);
    std::cout << "Comparison completed." << std::endl;

    return 0;
}