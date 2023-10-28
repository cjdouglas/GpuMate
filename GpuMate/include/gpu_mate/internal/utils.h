#ifndef GPU_MATE_INTERNAL_UTILS_H
#define GPU_MATE_INTERNAL_UTILS_H

#include <cstdio>
#include <cstdlib>

#ifdef __GM_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifndef RUNTIME_CHECK
#define RUNTIME_CHECK(error)                                      \
  if (error != hipSuccess) {                                      \
    fprintf(stderr, "Hip error: '%s'(%d) at %s:%d\n",             \
            hipGetErrorString(error), error, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                           \
  }
#endif

#ifndef ROCBLAS_CHECK
#define ROCBLAS_CHECK(error)             \
  if (error != rocblas_status_success) { \
    fprintf(stderr, "rocBLAS error");    \
    exit(EXIT_FAILURE);                  \
  }
#endif

#elif defined __GM_PLATFORM_NVIDIA__

#include <cublas.h>
#include <cuda_runtime.h>

#ifndef RUNTIME_CHECK
#define RUNTIME_CHECK(error)                                       \
  if (error != cudaSuccess) {                                      \
    fprintf(stderr, "Cuda error: '%s'(%d) at %s:%d\n",             \
            cudaGetErrorString(error), error, __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                            \
  }
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(error)             \
  if (error != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error");    \
    exit(EXIT_FAILURE);                 \
  }
#endif

#else
#error unsupported platform detected
#endif

#endif  // GPU_MATE_INTERNAL_UTILS_H
