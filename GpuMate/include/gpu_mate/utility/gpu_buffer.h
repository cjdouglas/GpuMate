#ifndef GPU_MATE_UTILITIES_GPU_BUFFER_H
#define GPU_MATE_UTILITIES_GPU_BUFFER_H

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

using gpu_mate::runtime::gpuMemcpyKind;

namespace gpu_mate {
namespace utility {
class GpuBuffer {
 public:
  GpuBuffer(const void* data = nullptr, size_t size = 0);
  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;
  GpuBuffer(GpuBuffer&&) noexcept;
  GpuBuffer& operator=(GpuBuffer&&) noexcept;
  ~GpuBuffer();

  void CopyFrom(const void* data, size_t size, gpuMemcpyKind copy_kind);
  void CopyTo(void* dst, gpuMemcpyKind copy_kind);
  void* operator*();

 private:
  void Free();
  void* handle_ = nullptr;
  size_t size_ = 0;
  bool initialized_ = false;
};
}  // namespace utility
}  // namespace gpu_mate

#endif  // GPU_MATE_UTILITIES_GPU_BUFFER_H
