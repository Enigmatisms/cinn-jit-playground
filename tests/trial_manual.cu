#include "include/tensor.cuh"
#include "include/reduce.cuh"
#include "include/semaphore.cuh"
#include <thread>
#include <chrono>

template <int N, int H, int W, int C>
__global__
void __launch_bounds__(C) kernel_improved(
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
  __builtin_assume(((int)blockIdx.x < (N * H * W / 4)));
  __builtin_assume(((int)threadIdx.x < C));
  float* var_27 = var_16;
  float* var_28 = var_21;
  constexpr float NHW = N * H * W;
  if (((int)blockIdx.x == 0)) {
    var_27[threadIdx.x] = ((0.899999976f * var_16[threadIdx.x]) + (0.100000024f * (var_0[threadIdx.x] / NHW)));
    var_37[threadIdx.x] = rsqrtf((((var_3[threadIdx.x] / NHW) - ((var_0[threadIdx.x] / NHW) * (var_0[threadIdx.x] / NHW))) + 9.99999975e-06f));
  };
  constexpr int b_proc = C * 4;
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    float var_11_local = var_11[(((i_j_k_a_fused_6 * C) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc))];
    float var_0_local = var_0[threadIdx.x];
    float var_3_local = var_3[threadIdx.x];
    float var_29_local = var_29[threadIdx.x];
    float var_32_local = var_32[threadIdx.x];
    var_35[(((i_j_k_a_fused_6 * C) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc))] = ((((var_11_local - (var_0_local / NHW)) * rsqrtf((((var_3_local / NHW) - ((var_0_local / NHW) * (var_0_local / NHW))) + 9.99999975e-06f))) * var_29_local) + var_32_local);
  };
  if (((int)blockIdx.x == 0)) {
    var_28[threadIdx.x] = ((0.899999976f * var_21[threadIdx.x]) + (0.100000024f * ((var_3[threadIdx.x] / NHW) - ((var_0[threadIdx.x] / NHW) * (var_0[threadIdx.x] / NHW)))));
  };
}

template <int N, int H, int W, int C, int TNum = 256>
__global__
void __launch_bounds__(TNum) kernel_origin(
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
  __builtin_assume(((int)blockIdx.x < (N * W * H * C / (4 * TNum))));
  __builtin_assume(((int)threadIdx.x < TNum));
  float* var_27 = var_16;
  float* var_28 = var_21;
  constexpr int HWC = H * W * C, WC = W * C, HW = H * W;
  constexpr float NHW = N * H * W;
  constexpr int b_proc = TNum * 4;
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_0 = 0; append_var_0_append_var_1_append_var_2_i_fused_0 < 4; append_var_0_append_var_1_append_var_2_i_fused_0 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * TNum) + (int)threadIdx.x) % HWC) / WC) * static_cast<float>(H)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * TNum) + (int)threadIdx.x) % WC) / static_cast<float>(C))) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * TNum) + (int)threadIdx.x) / HWC) * HW)) == 0ll)) {
      var_27[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * TNum)) % C)] = ((0.899999976f * var_16[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * TNum)) % C)]) + (0.100000024f * (var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * TNum)) % C)] / NHW)));
    };
  };
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_3 = 0; append_var_0_append_var_1_append_var_2_i_fused_3 < 4; append_var_0_append_var_1_append_var_2_i_fused_3 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * TNum) + (int)threadIdx.x) % HWC) / WC) * static_cast<float>(H)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * TNum) + (int)threadIdx.x) % WC) / static_cast<float>(C))) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * TNum) + (int)threadIdx.x) / HWC) * HW)) == 0ll)) {
      var_37[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * TNum)) % C)] = rsqrtf((((var_3[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * TNum)) % C)] / NHW) - ((var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * TNum)) % C)] / NHW) * (var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * TNum)) % C)] / NHW))) + 9.99999975e-06f));
    };
  };

  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    float var_11_local = var_11[(((i_j_k_a_fused_6 * TNum) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc))];
    float var_0_local = var_0[((((i_j_k_a_fused_6 * TNum) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc)) % C)];
    float var_3_local = var_3[((((i_j_k_a_fused_6 * TNum) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc)) % C)];
    float var_29_local = var_29[((((i_j_k_a_fused_6 * TNum) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc)) % C)];
    float var_32_local = var_32[((((i_j_k_a_fused_6 * TNum) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc)) % C)];
    var_35[(((i_j_k_a_fused_6 * TNum) + (int)threadIdx.x) + ((int)blockIdx.x * b_proc))] = ((((var_11_local - (var_0_local / NHW)) * rsqrtf((((var_3_local / NHW) - ((var_0_local / NHW) * (var_0_local / NHW))) + 9.99999975e-06f))) * var_29_local) + var_32_local);
  };
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_6 = 0; append_var_0_append_var_1_append_var_2_i_fused_6 < 4; append_var_0_append_var_1_append_var_2_i_fused_6 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * TNum) + (int)threadIdx.x) % HWC) / WC) * static_cast<float>(H)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * TNum) + (int)threadIdx.x) % WC) / static_cast<float>(C))) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * TNum) + (int)threadIdx.x) / HWC) * HW)) == 0ll)) {
      var_28[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * TNum)) % C)] = ((0.899999976f * var_21[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * TNum)) % C)]) + (0.100000024f * ((var_3[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * TNum)) % C)] / NHW) - ((var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * TNum)) % C)] / NHW) * (var_0[(((((int)blockIdx.x * b_proc) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * TNum)) % C)] / NHW)))));
    };
  };
}

__global__
void __launch_bounds__(256) kernel_64_48_224(
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
  __builtin_assume(((int)blockIdx.x < 32256));
  __builtin_assume(((int)threadIdx.x < 256));
  float* var_27 = var_16;
  float* var_28 = var_21;
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_0 = 0; append_var_0_append_var_1_append_var_2_i_fused_0 < 4; append_var_0_append_var_1_append_var_2_i_fused_0 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) % 516096ll) / 10752ll) * 48ll) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) % 10752ll) / 224ll)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_0) * 256ll) + (int)threadIdx.x) / 516096ll) * 2304ll)) == 0ll)) {
      var_27[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 224)] = ((0.899999976f * var_16[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 224)]) + (0.100000024f * (var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_0 * 256)) % 224)] / 147456.000f)));
    };
  };
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_3 = 0; append_var_0_append_var_1_append_var_2_i_fused_3 < 4; append_var_0_append_var_1_append_var_2_i_fused_3 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) % 516096ll) / 10752ll) * 48ll) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) % 10752ll) / 224ll)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_3) * 256ll) + (int)threadIdx.x) / 516096ll) * 2304ll)) == 0ll)) {
      var_37[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 224)] = rsqrtf((((var_3[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 224)] / 147456.000f) - ((var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 224)] / 147456.000f) * (var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_3 * 256)) % 224)] / 147456.000f))) + 9.99999975e-06f));
    };
  };
  for (int32_t i_j_k_a_fused_6 = 0; i_j_k_a_fused_6 < 4; i_j_k_a_fused_6 += 1) {
    float var_11_local = var_11[(((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024))];
    float var_0_local = var_0[((((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024)) % 224)];
    float var_3_local = var_3[((((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024)) % 224)];
    float var_29_local = var_29[((((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024)) % 224)];
    float var_32_local = var_32[((((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024)) % 224)];
    var_35[(((i_j_k_a_fused_6 * 256) + (int)threadIdx.x) + ((int)blockIdx.x * 1024))] = ((((var_11_local - (var_0_local / 147456.000f)) * rsqrtf((((var_3_local / 147456.000f) - ((var_0_local / 147456.000f) * (var_0_local / 147456.000f))) + 9.99999975e-06f))) * var_29_local) + var_32_local);
  };
  for (int32_t append_var_0_append_var_1_append_var_2_i_fused_6 = 0; append_var_0_append_var_1_append_var_2_i_fused_6 < 4; append_var_0_append_var_1_append_var_2_i_fused_6 += 1) {
    if ((((((((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) % 516096ll) / 10752ll) * 48ll) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) % 10752ll) / 224ll)) + (((((((int)blockIdx.x * 4ll) + append_var_0_append_var_1_append_var_2_i_fused_6) * 256ll) + (int)threadIdx.x) / 516096ll) * 2304ll)) == 0ll)) {
      var_28[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 224)] = ((0.899999976f * var_21[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 224)]) + (0.100000024f * ((var_3[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 224)] / 147456.000f) - ((var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 224)] / 147456.000f) * (var_0[(((((int)blockIdx.x * 1024) + (int)threadIdx.x) + (append_var_0_append_var_1_append_var_2_i_fused_6 * 256)) % 224)] / 147456.000f)))));
    };
  };
}

int main()
{
    std::cout << "Initializing data..." << std::endl;
    constexpr size_t N = 64, H = 48, W = 48, C = 224, TNum = 256;   // input shapes

    Tensor<curandState> gen_input(256, RandInit{2024});
    Tensor<curandState> gen_w1(C, RandInit{2025});

    Tensor<float> X(N * H * W * C, Randn{gen_input}, true);

    Tensor<float> var1(C, Rand{gen_w1, 0, 1});
    Tensor<float> var0(C, Rand{gen_w1, N*H*W - 2, N*H*W - 1});
    Tensor<float> var3(C, Rand{gen_w1, N*H*W, N*H*W + 1});
    Tensor<float> w1(C, Rand{gen_w1, 0.01, 1}); 
    Tensor<float> w2(C, Rand{gen_w1, 0.01, 1}); 

    Tensor<float> out2(C, Zero<float>{});
    Tensor<float> out3(N * H * W * C, Zero<float>{});
    Tensor<float> out4(C, Zero<float>{});

    // used for comparison
    Tensor<float> var1_cp = var1;
    Tensor<float> out2_cp = out2;
    Tensor<float> out3_cp = out3;
    Tensor<float> out4_cp = out4;

    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    printf("Launching kernel (%d, %d, %d, %d)...\n", N, H, W, C);

    constexpr int num_blocks = N * W * H / 4, num_blocks_origin = N * H * W * C / (TNum * 4);
    // constexpr int dyna_shared = C * 4 * sizeof(float);

    kernel_improved<N, H, W, C><<<num_blocks, C>>>(var0, var3, X, var1, out2, w1, w2, out3, out4);
    kernel_origin<N, H, W, C, TNum><<<num_blocks_origin, TNum>>>(var0, var3, X, var1_cp, out2_cp, w1, w2, out3_cp, out4_cp);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::cout << "Comparing results..." << std::endl;
    out3_cp.to_host();
    out3.to_host();
    out3_cp.compare(out3);
    std::cout << "Comparison completed." << std::endl;

    return 0;
}