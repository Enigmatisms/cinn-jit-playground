#include <array>
#include "include/bn_registry.cuh"
#include "include/bn_template.cuh"

struct KernelEntry {
	size_t N = 64;
	size_t H = 56;
	size_t W = 56;
	size_t C = 192;
	void* func_ptr;

	KernelEntry(size_t N, size_t H, size_t W, size_t C) : N(N), H(H), W(W), C(C) {}

	KernelEntry() = default;

	size_t size() const { return N * H * W * C; }
	size_t num_pixels() const { return N * H * W; }
};

static std::array<KernelEntry, 60> kernel_original_table;
static std::array<KernelEntry, 61> kernel_improved_table;

#define BN_IMPROVED_KERNEL(N, H, W, C)        \
	template __global__                       \
		__launch_bounds__(C) void             \
		kernel_improved<N, H, W, C>(          \
			const float *__restrict__ var_0,  \
			const float *__restrict__ var_3,  \
			const float *__restrict__ var_11, \
			float *__restrict__ var_16,       \
			float *__restrict__ var_21,       \
			const float *__restrict__ var_29, \
			const float *__restrict__ var_32, \
			float *__restrict__ var_35,       \
			float *__restrict__ var_37);

#define BN_ORIGIN_KERNEL(N, H, W, C, TNum)    \
	template __global__                       \
		__launch_bounds__(TNum) void          \
		kernel_origin<N, H, W, C, TNum>(      \
			const float *__restrict__ var_0,  \
			const float *__restrict__ var_3,  \
			const float *__restrict__ var_11, \
			float *__restrict__ var_16,       \
			float *__restrict__ var_21,       \
			const float *__restrict__ var_29, \
			const float *__restrict__ var_32, \
			float *__restrict__ var_35,       \
			float *__restrict__ var_37);

#define GENERATE_KERNEL_FIXED_N_W(N, H, W, C) \
	BN_IMPROVED_KERNEL(N, H, W, C)            \
	BN_ORIGIN_KERNEL(N, H, W, C, 256)

#define GENERATE_KERNEL_FIXED_N(N, W)       \
	GENERATE_KERNEL_FIXED_N_W(N, W, W, 384) \
	GENERATE_KERNEL_FIXED_N_W(N, W, W, 224) \
	GENERATE_KERNEL_FIXED_N_W(N, W, W, 192)

#define GENERATE_KERNEL(N)         \
	GENERATE_KERNEL_FIXED_N(N, 72) \
	GENERATE_KERNEL_FIXED_N(N, 64) \
	GENERATE_KERNEL_FIXED_N(N, 56) \
	GENERATE_KERNEL_FIXED_N(N, 48)

GENERATE_KERNEL(400)
GENERATE_KERNEL(256)
GENERATE_KERNEL(200)
GENERATE_KERNEL(128)
GENERATE_KERNEL(64)

#define PUT_IMPROVED_KERNEL(n, h, w, c, index) \
    { auto& entry_improved = kernel_improved_table[index]; \
    entry_improved.N = n;   \
    entry_improved.H = h;   \
    entry_improved.W = w;   \
    entry_improved.C = c;   \
    entry_improved.func_ptr = reinterpret_cast<void*>(kernel_improved<n, h, w, c>);}

#define PUT_ORIGINAL_KERNEL(n, h, w, c, index) \
    { auto& entry_original = kernel_original_table[index]; \
    entry_original.N = n;   \
    entry_original.H = h;   \
    entry_original.W = w;   \
    entry_original.C = c;   \
    entry_original.func_ptr = reinterpret_cast<void*>(kernel_origin<n, h, w, c, 256>);}

#define PUT_KERNEL_FIXED_N_W(N, H, W, C, index) \
    PUT_IMPROVED_KERNEL(N, H, W, C, index) \
    PUT_ORIGINAL_KERNEL(N, H, W, C, index)

#define PUT_KERNEL_FIXED_N(N, W, base, num_c) \
    PUT_KERNEL_FIXED_N_W(N, W, W, 384, base * num_c) \
    PUT_KERNEL_FIXED_N_W(N, W, W, 224, base * num_c + 1) \
    PUT_KERNEL_FIXED_N_W(N, W, W, 192, base * num_c + 2)

#define PUT_KERNEL(N, n_index, num_w, num_c) \
    printf("[BN Registry] Storing kernels (batch %d)...\n", N); \
    PUT_KERNEL_FIXED_N(N, 72, n_index * num_w,     num_c) \
    PUT_KERNEL_FIXED_N(N, 64, n_index * num_w + 1, num_c) \
    PUT_KERNEL_FIXED_N(N, 56, n_index * num_w + 2, num_c) \
    PUT_KERNEL_FIXED_N(N, 48, n_index * num_w + 3, num_c) \
    printf("[BN Registry] Kernels (batch %d) are successfully stored.\n", N);

__global__ void foo_kernel(float* a) {
    a[threadIdx.x + blockIdx.x * blockDim.x]++;
}

void init_kernel_table() {
    PUT_KERNEL(400, 0, 4, 3)
    PUT_KERNEL(256, 1, 4, 3)
    PUT_KERNEL(200, 2, 4, 3)
    PUT_KERNEL(128, 3, 4, 3)
    PUT_KERNEL(64,  4, 4, 3)

    auto& entry_original = kernel_original_table[60];
    entry_original.N = 256;
    entry_original.H = 1;
    entry_original.W = 1;
    entry_original.C = 256;
    entry_original.func_ptr = reinterpret_cast<void*>(foo_kernel);
}

void lookup_kernel_call(int n, int w, int c) {
    for (size_t i = 0; i < kernel_improved_table.size(); i++) {
        const auto& entry_improved = kernel_improved_table[i];
        if (entry_improved.N == n && entry_improved.W == w && entry_improved.C == c) {
            const auto& entry_original = kernel_original_table[i];
            constexpr int TNum = 256;

            Tensor<curandState> gen_input(TNum, RandInit{2024});
            Tensor<curandState> gen_w1(entry_improved.C, RandInit{2025});

            Tensor<float> X(entry_improved.size(), Randn{gen_input}, true);

            size_t num_pixels = entry_improved.num_pixels();
            Tensor<float> var1(entry_improved.C, Rand{gen_w1, 0, 1});
            Tensor<float> var0(entry_improved.C, Rand{gen_w1, float(num_pixels - 2), float(num_pixels - 1)});
            Tensor<float> var3(entry_improved.C, Rand{gen_w1, float(num_pixels), float(num_pixels + 1)});
            Tensor<float> w1(entry_improved.C, Rand{gen_w1, 0.01, 1}); 
            Tensor<float> w2(entry_improved.C, Rand{gen_w1, 0.01, 1}); 

            Tensor<float> out2(entry_improved.C, Zero<float>{});
            Tensor<float> out3(entry_improved.size(), Zero<float>{});
            Tensor<float> out4(entry_improved.C, Zero<float>{});

            // used for comparison
            Tensor<float> var1_cp = var1;
            Tensor<float> out2_cp = out2;
            Tensor<float> out3_cp = out3;
            Tensor<float> out4_cp = out4;

            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            printf("Launching kernel (%d, %d, %d, %d)...\n", 
                entry_improved.N, entry_improved.H, entry_improved.W, entry_improved.C
            );

            const int num_blocks = num_pixels / 4, num_blocks_origin = entry_improved.size() / (TNum * 4);
            // constexpr int dyna_shared = C * 4 * sizeof(float);
            dim3 grid_improved(num_blocks, 1, 1), block_improved(entry_improved.C, 1, 1);
            dim3 grid_original(num_blocks_origin, 1, 1), block_original(TNum, 1, 1);

            void* args_improved[] = {
                &var0, &var3, &X, &var1, &out2, &w1, &w2, &out3, &out4
            };

            void* args_original[] = {
                &var0, &var3, &X, &var1_cp, &out2_cp, &w1, &w2, &out3_cp, &out4_cp
            };

            std::cout << entry_improved.func_ptr << std::endl;
            CUDA_CHECK_RETURN(cudaLaunchKernel(entry_improved.func_ptr, grid_improved, block_improved, args_improved));
            CUDA_CHECK_RETURN(cudaLaunchKernel(entry_original.func_ptr, grid_original, block_original, args_original));
            CUDA_CHECK_RETURN(cudaDeviceSynchronize());

            std::cout << "Comparing results..." << std::endl;
            out3_cp.to_host();
            out3.to_host();
            out3_cp.compare(out3);
            std::cout << "Comparison completed." << std::endl;
            return;
        }
    }
}