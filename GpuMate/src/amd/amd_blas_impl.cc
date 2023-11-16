#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>

#include <cstdlib>
#include <iostream>

#include "gpu_mate/gpu_blas.h"
#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace blas {
namespace {
static rocblas_operation GpuToRocOperation(const GpuOperation op) {
  switch (op) {
    case GpuOperation::none:
      return rocblas_operation_none;
    case GpuOperation::transpose:
      return rocblas_operation_transpose;
    case GpuOperation::conjugateTranspose:
      return rocblas_operation_conjugate_transpose;
    default:
      return rocblas_operation_none;
  }
}

static GpuBlasStatus RocToGpuStatus(const rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return GpuBlasStatus::success;
    case rocblas_status_invalid_handle:
      return GpuBlasStatus::notInitialized;
    case rocblas_status_memory_error:
      return GpuBlasStatus::allocFailed;
    case rocblas_status_invalid_value:
      return GpuBlasStatus::invalidValue;
    case rocblas_status_internal_error:
      return GpuBlasStatus::internalError;
    default:
      return GpuBlasStatus::unknown;
  }
}
}  // namespace

// Start GpuBlasHandle implementation

GpuBlasHandle::GpuBlasHandle() {
  rocblas_handle handle = rocblas_handle();
  const rocblas_status status = rocblas_create_handle(&handle);
  if (status != rocblas_status_success) {
    std::cerr << "Error initializing GpuBlasHandle - aborting" << std::endl;
    exit(EXIT_FAILURE);
  }
  handle_ = static_cast<void*>(handle);
}

GpuBlasHandle::~GpuBlasHandle() {
  rocblas_handle handle = static_cast<rocblas_handle>(handle_);
  rocblas_destroy_handle(handle);
}

GpuBlasHandle GpuBlasHandle::Create() { return GpuBlasHandle(); }

void* GpuBlasHandle::operator*() { return handle_; }

// Start BLAS functions

GpuBlasStatus sgemm(GpuBlasHandle& handle, GpuOperation transA,
                    GpuOperation transB, int m, int n, int k,
                    const float* alpha, const float* A, int lda, const float* B,
                    int ldb, const float* beta, float* C, int ldc) {
  const rocblas_handle _handle = reinterpret_cast<rocblas_handle>(*handle);
  const rocblas_operation _transA = GpuToRocOperation(transA);
  const rocblas_operation _transB = GpuToRocOperation(transB);
  const rocblas_status status = rocblas_sgemm(
      _handle, _transA, _transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  return RocToGpuStatus(status);
}
}  // namespace blas
}  // namespace gpu_mate
