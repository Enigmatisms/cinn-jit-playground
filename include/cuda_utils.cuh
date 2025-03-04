#pragma once
#include <iostream>
#include <cuda_runtime.h>

// #define VERBOSE_DEBUG
#ifdef VERBOSE_DEBUG
	#define DEBUG_LOG(fmt, ...)	printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
	#define DEBUG_LOG(fmt, ...)	// do nothing
#endif

#ifndef NO_CUDA
__host__ static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__host__ static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err), err, file, line);
	exit (1);
}
#endif