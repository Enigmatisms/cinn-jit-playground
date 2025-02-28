#pragma once
#include <cuda_runtime.h>

__device__ inline bool cinn_grid_reduce_update_semaphore(int *semaphores) {
  __shared__ bool done;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    int old = atomicAdd(&semaphores[blockIdx.x], 1);
    done = (old == (gridDim.y - 1));
  }
  __syncthreads();
  return done;
}