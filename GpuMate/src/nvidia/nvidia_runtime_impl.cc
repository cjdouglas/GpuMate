#include <cuda_runtime.h>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace runtime {
namespace {

static cudaMemcpyKind MapMemcpyKind(const GpuMemcpyKind copy_kind) {
  switch (copy_kind) {
    case GpuMemcpyKind::kHostToHost:
      return cudaMemcpyHostToHost;
    case GpuMemcpyKind::kHostToDevice:
      return cudaMemcpyHostToDevice;
    case GpuMemcpyKind::kDeviceToHost:
      return cudaMemcpyDeviceToHost;
    case GpuMemcpyKind::kDeviceToDevice:
      return cudaMemcpyDeviceToDevice;
    case GpuMemcpyKind::kDefault:
    default:
      return cudaMemcpyDefault;
  }
}
}  // namespace

GpuMateError_t GpuMalloc(void** ptr, const size_t size) {
  return cudaMalloc(ptr, size);
}

GpuMateError_t GpuMemcpy(void* dst, const void* src, const size_t size,
                         const GpuMemcpyKind copy_kind) {
  return cudaMemcpy(dst, src, size, MapMemcpyKind(copy_kind));
}

GpuMateError_t GpuFree(void* ptr) { return cudaFree(ptr); }

}  // namespace runtime
}  // namespace gpu_mate
