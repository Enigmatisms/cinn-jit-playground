#include "include/bn_registry.cuh"
#include "include/cuda_utils.cuh"

int main(int argc, char** argv) {
    CUDA_CHECK_RETURN(cudaFree(nullptr));       // initialize CUDA

    int N = 64, W = 56, C = 192;
    if (argc < 2) {
        printf("Usage: %s <N> [<W>] [<C>]\n", argv[0]);
        printf("Using default N W C: (%d, %d, %d)\n", N, W, C);
    }
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        W = atoi(argv[2]);
    }
    if (argc > 3) {
        C = atoi(argv[3]);
    }

    init_kernel_table();
    lookup_kernel_call(N, W, C);

    return 0;
}