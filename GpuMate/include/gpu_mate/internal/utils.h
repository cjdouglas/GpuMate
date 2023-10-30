#ifndef GPU_MATE_INTERNAL_UTILS_H
#define GPU_MATE_INTERNAL_UTILS_H

#include <cstdio>
#include <cstdlib>

#include "gpu_mate/gpu_runtime.h"

using gpu_mate::runtime::gpuError_t;
using gpu_mate::runtime::gpuGetErrorString;

#ifdef __GM_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifndef GPU_CHECK
#define GPU_CHECK(error)                                          \
  if (error != gpuError_t::gpuSuccess) {                          \
    fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",             \
            gpuGetErrorString(error), error, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                           \
  }
#endif

#elif defined __GM_PLATFORM_NVIDIA__

#include <cublas.h>
#include <cuda_runtime.h>

#ifndef GPU_CHECK
#define GPU_CHECK(error)                                          \
  if (error != gpuError_t::gpuSuccess) {                          \
    fprintf(stderr, "Cuda error: '%s'(%d) at %s:%d\n",            \
            gpuGetErrorString(error), error, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                           \
  }
#endif

#else
#error unsupported platform detected
#endif

#endif  // GPU_MATE_INTERNAL_UTILS_H
