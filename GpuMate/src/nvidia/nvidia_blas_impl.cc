#include <cublas_v2.h>

#include <cstdlib>
#include <iostream>

#include "gpu_mate/gpu_blas.h"
#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace blas {
namespace {
static cublasOperation_t GpuToCudaOperation(const GpuOperation op) {
  switch (op) {
    case GpuOperation::none:
      return CUBLAS_OP_N;
    case GpuOperation::transpose:
      return CUBLAS_OP_T;
    case GpuOperation::conjugateTranspose:
      return CUBLAS_OP_C;
    default:
      return CUBLAS_OP_N;
  }
}

static GpuBlasStatus CudaToGpuStatus(const cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return success;
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return notInitialized;
    case CUBLAS_STATUS_ALLOC_FAILED:
      return allocFailed;
    case CUBLAS_STATUS_INVALID_VALUE:
      return invalidValue;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return archMismatch;
    case CUBLAS_STATUS_MAPPING_ERROR:
      return mappingError;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return executionFailed;
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return internalError;
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return notSupported;
    default:
      return unknown;
  }
}
}  // namespace

// Start GpuBlasHandle implementation

GpuBlasHandle::GpuBlasHandle() {
  cublasHandle_t handle = cublasHandle_t();
  const cublasStatus_t status = cublasCreate_v2(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error initializing GpuBlasHandle - aborting" << std::endl;
    exit(EXIT_FAILURE);
  }
  handle_ = static_cast<void*>(handle);
}

GpuBlasHandle::~GpuBlasHandle() {
  cublasHandle_t handle = static_cast<cublasHandle_t>(handle_);
  cublasDestroy_v2(handle);
}

void GpuBlasHandle::SetStream(const gpu_mate::runtime::GpuStream& stream) {
  cublasHandle_t blas_handle = static_cast<cublasHandle_t>(handle_);
  cudaStream_t stream_handle = static_cast<cudaStream_t>(*stream);
  cublasSetStream(blas_handle, stream_handle);
}

// Start BLAS functions

GpuBlasStatus gm_sgemm(const GpuBlasHandle& handle, GpuOperation transA,
                       GpuOperation transB, int m, int n, int k,
                       const float* alpha, const float* A, int lda,
                       const float* B, int ldb, const float* beta, float* C,
                       int ldc) {
  const cublasHandle_t _handle = static_cast<cublasHandle_t>(*handle);
  const cublasOperation_t _transA = GpuToCudaOperation(transA);
  const cublasOperation_t _transB = GpuToCudaOperation(transB);
  const cublasStatus_t status = cublasSgemm(
      _handle, _transA, _transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return CudaToGpuStatus(status);
}

}  // namespace blas
}  // namespace gpu_mate
