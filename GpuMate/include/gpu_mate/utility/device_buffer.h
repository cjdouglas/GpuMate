#pragma once

#include "gpu_mate/gpu_runtime.h"

namespace gpu_mate {
namespace utility {
class DeviceBuffer {
 public:
  static DeviceBuffer FromHost(const void* data, size_t size);

  explicit DeviceBuffer();
  explicit DeviceBuffer(size_t size);
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&&);
  DeviceBuffer& operator=(DeviceBuffer&&);
  ~DeviceBuffer();

  [[nodiscard]] runtime::GpuError CopyFrom(const void* src, size_t size,
                                           runtime::GpuMemcpyKind copy_kind);
  [[nodiscard]] runtime::GpuError CopyTo(
      void* dst, runtime::GpuMemcpyKind copy_kind) const;

  size_t Size() const { return size_; }
  bool Initialized() const { return initialized_; }
  void* Handle() { return handle_; }
  const void* Handle() const { return handle_; }

  template <typename T>
  T* TypedHandle() {
    return static_cast<T*>(handle_);
  }

  template <typename T>
  const T* TypedHandle() const {
    return static_cast<T*>(handle_);
  }

 private:
  void Alloc(size_t size);
  void Free();
  void* handle_;
  size_t size_;
  bool initialized_;
};
}  // namespace utility
}  // namespace gpu_mate
