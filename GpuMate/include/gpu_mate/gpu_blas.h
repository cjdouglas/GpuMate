#pragma once

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/defines.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace blas {
class GpuBlasHandle {
 public:
  static GpuBlasHandle Create();
  ~GpuBlasHandle();
  GpuBlasHandle(const GpuBlasHandle&) = delete;
  GpuBlasHandle& operator=(const GpuBlasHandle&) = delete;
  GpuBlasHandle(GpuBlasHandle&&) = default;
  GpuBlasHandle& operator=(GpuBlasHandle&&) = default;
  void SetStream(const gpu_mate::runtime::GpuStream& stream);
  void* operator*() const { return handle_; }

 private:
  explicit GpuBlasHandle();
  void* handle_;
};

enum class GpuOperation {
  none = 0,
  transpose = 1,
  conjugateTranspose = 2,
};

enum GpuBlasStatus {
  success = 0,
  notInitialized = 1,
  allocFailed = 2,
  invalidValue = 3,
  archMismatch = 4,
  mappingError = 5,
  executionFailed = 6,
  internalError = 7,
  notSupported = 8,
  unknown = 99,
};

GPU_MATE_API GpuBlasStatus sgemm(GpuBlasHandle& handle, GpuOperation transA,
                                 GpuOperation transB, int m, int n, int k,
                                 const float* alpha, const float* A, int lda,
                                 const float* B, int ldb, const float* beta,
                                 float* C, int ldc);
}  // namespace blas
}  // namespace gpu_mate
