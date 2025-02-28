#pragma once
#include <iostream>
#include <curand_kernel.h>
#include "cuda_utils.cuh"

template <typename T>
struct Zero {
    __device__ T operator()(int) { return T(0); }
};
template <typename T>
struct One {
    __device__ T operator()(int) { return T(1); }
};
struct Arange {
    __device__ int operator()(int i) { return i; }
};
struct RandInit {
    unsigned long long seed;
    __device__ curandState operator()(int i) {
        curandState state;
        curand_init(seed, i, 0, &state);
        return state;
    }
};
struct Randn {
    curandState *state;
};

struct Rand {
    curandState *state;
    float mini;
    float maxi;

    Rand(curandState *_state, float _mini = 0, float _maxi = 1) : state(_state), mini(_mini), maxi(_maxi) {}
};

template <typename T, typename F>
__global__ void vector_map(T *p, int n, F f)
{
    for (int k = 0; k < n; k += gridDim.x * blockDim.x)
    {
        int i = k + blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            p[i] = f(i);
        }
    }
}
template <typename T>
__global__ void vector_map(T *p, int n, Randn f) {
    int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_id < n) {
        curandState state = f.state[seq_id];
        for (int k = 0; k < n; k += gridDim.x * blockDim.x) {
            int i = k + seq_id;
            if (i < n) {
                p[i] = curand_normal(&state);
            }
        }
        f.state[seq_id] = state;
    }
}

template <typename T>
__global__ void vector_map(T *p, int n, Rand f) {
    int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_id < n) {
        curandState state = f.state[seq_id];
        for (int k = 0; k < n; k += gridDim.x * blockDim.x) {
            int i = k + seq_id;
            if (i < n) {
                p[i] = curand_uniform(&state) * (f.maxi - f.mini) + f.mini;
            }
        }
        f.state[seq_id] = state;
    }
}

template <typename T>
__global__ void vector_map_lite(T *p, int n, Randn f) {
    curandState state = f.state[threadIdx.x];
    curand_init(blockIdx.x, 0, 0, &state);
    int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < n; k += gridDim.x * blockDim.x) {
        int i = k + seq_id;
        if (i < n) {
            p[i] = curand_normal(&state);
        }
    }
    // no write back
}

template <typename T>
__global__ void vector_map_lite(T *p, int n, Rand f) {
    curandState state = f.state[threadIdx.x];
    curand_init(blockIdx.x, 0, 0, &state);
    int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < n; k += gridDim.x * blockDim.x) {
        int i = k + seq_id;
        if (i < n) {
            p[i] = curand_uniform(&state) * (f.maxi - f.mini) + f.mini;
        }
    }
    // no write back
}

template <typename T, typename F>
__global__ void vector_map_lite(T *p, int n, F f) {
    int seq_id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = 0; k < n; k += gridDim.x * blockDim.x) {
        int i = k + seq_id;
        if (i < n) {
            p[i] = f(i);
        }
    }
}

template <typename T>
struct Tensor {
    size_t numel;
    T *ptr;

    bool is_host;

    Tensor(size_t numel) : numel(numel), ptr(nullptr), is_host(false) {
        // printf("Allocating: %d, type: %s, %d\n", numel, typeid(T).name(), sizeof(T));
        CUDA_CHECK_RETURN(cudaMalloc((void **)&ptr, sizeof(T) * numel));
    }

    Tensor(const Tensor<T>& ts) {
        numel = ts.numel;
        is_host = false;
        if (ts.is_host) {
            ptr = new T[numel];
            CUDA_CHECK_RETURN(cudaMemcpy(ptr, ts.ptr, sizeof(T) * numel, cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK_RETURN(cudaMalloc((void **)&ptr, sizeof(T) * numel));
            CUDA_CHECK_RETURN(cudaMemcpy(ptr, ts.ptr, sizeof(T) * numel, cudaMemcpyDeviceToDevice));
        }
    }

    Tensor(Tensor<T>&& ts) {
        numel = ts.numel;
        is_host = ts.is_host;
        ptr = ts.ptr;

        ts.ptr   = nullptr;
        ts.numel = 0;
    }

    template <typename Init>
    Tensor(size_t numel, Init init, bool lite = false) : Tensor(numel) { 
        apply(init, lite); 
    }

    void to_host() {
        T *host_ptr = new T[numel];
        CUDA_CHECK_RETURN(cudaMemcpy(host_ptr, ptr, sizeof(T) * numel, cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaFree(ptr));

        ptr = host_ptr;
        is_host = true;
    }

    ~Tensor() {
        if (ptr != nullptr) {
            if (is_host) {
                delete[] ptr;
            } else {
                CUDA_CHECK_RETURN(cudaFree(ptr));
            }
            ptr = nullptr;
        }
    }

    void clear() { apply(Zero<T>()); }

    template <typename F, size_t NBLOCK = 64, size_t NTHREAD = 256>
    void apply(F f, bool lite = false) {
        if (lite) {
            vector_map_lite<<<NBLOCK, NTHREAD>>>(ptr, numel, f);
        } else {
            vector_map<<<NBLOCK, NTHREAD>>>(ptr, numel, f);
        }
    }

    operator T *() { return ptr; }
    operator const T *() const { return ptr; }
    T **operator&() { return &ptr; }
    T operator[](int i) { return ptr[i]; }


    void compare(const Tensor<T>& o, float eps = 1e-6f, float scale = 1e-4f) const {
        if (numel != o.numel) {
            std::cerr << "cannot compare tensors with different numels" << std::endl;
            exit(1);
        }
        int errcnt = 0;
        for (int i = 0; i < numel; i++) {
            float desired = ptr[i];
            float actual = o.ptr[i];
            if (abs(actual - desired) > eps + scale * abs(desired)) {
                if (++errcnt < 12) {
                    std::cerr << i << " = " << desired << " " << actual << std::endl;
                }
            }
        }
        if (errcnt == 0) {
            std::cout << "ok" << std::endl;
        } else {
            float rate = ((float)errcnt / std::max(numel, size_t(1)));
            std::cerr << "num diff: " << errcnt << " / " << numel << " ("
                << rate * 100 << "%)" << std::endl;
        }
    }
};