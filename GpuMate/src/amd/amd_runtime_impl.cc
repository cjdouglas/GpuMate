#include <hip/hip_runtime.h>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace runtime {
namespace {

static hipMemcpyKind MapMemcpyKind(const GpuMemcpyKind copy_kind) {
  switch (copy_kind) {
    case GpuMemcpyKind::kHostToHost:
      return hipMemcpyHostToHost;
    case GpuMemcpyKind::kHostToDevice:
      return hipMemcpyHostToDevice;
    case GpuMemcpyKind::kDeviceToHost:
      return hipMemcpyDeviceToHost;
    case GpuMemcpyKind::kDeviceToDevice:
      return hipMemcpyDeviceToDevice;
    case GpuMemcpyKind::kDefault:
    default:
      return hipMemcpyDefault;
  }
}
}  // namespace

GpuMateError_t GpuMalloc(void** ptr, const size_t size) {
  return hipMalloc(ptr, size);
}

GpuMateError_t GpuMemcpy(void* dst, const void* src, const size_t size,
                         const GpuMemcpyKind copy_kind) {
  return hipMemcpy(dst, src, size, MapMemcpyKind(copy_kind));
}

GpuMateError_t GpuFree(void* ptr) { return hipFree(ptr); }

}  // namespace runtime
}  // namespace gpu_mate
