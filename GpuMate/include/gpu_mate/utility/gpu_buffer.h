#pragma once

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

using gpu_mate::runtime::GpuMemcpyKind;

namespace gpu_mate {
namespace utility {
class GpuBuffer {
 public:
  explicit GpuBuffer();
  GpuBuffer(const void* data, size_t size);
  GpuBuffer(const GpuBuffer&) = delete;
  GpuBuffer& operator=(const GpuBuffer&) = delete;
  GpuBuffer(GpuBuffer&&) noexcept;
  GpuBuffer& operator=(GpuBuffer&&) noexcept;
  ~GpuBuffer();

  void CopyFrom(const void* data, size_t size, GpuMemcpyKind copy_kind);
  void CopyTo(void* dst, GpuMemcpyKind copy_kind);
  void* operator*();

 private:
  void Free();
  void* handle_ = nullptr;
  size_t size_ = 0;
  bool initialized_ = false;
};
}  // namespace utility
}  // namespace gpu_mate
