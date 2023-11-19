#pragma once

#include "gpu_mate/gpu_runtime.h"

namespace gpu_mate {
namespace utility {
class DeviceBuffer {
 public:
  static DeviceBuffer FromHost(const void* data, size_t size) noexcept;
  static DeviceBuffer FromHostAsync(const void* data, size_t size,
                                    const runtime::GpuStream& stream) noexcept;
  static DeviceBuffer FromDevice(const void* data, size_t size) noexcept;
  static DeviceBuffer FromDeviceAsync(
      const void* data, size_t size, const runtime::GpuStream& stream) noexcept;

  explicit DeviceBuffer();
  explicit DeviceBuffer(size_t size);
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&&) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
  ~DeviceBuffer();

  [[nodiscard]] runtime::GpuError CopyToHost(void* dst) const;
  [[nodiscard]] runtime::GpuError CopyToHostAsync(
      void* dst, const runtime::GpuStream& stream) const;
  [[nodiscard]] runtime::GpuError CopyToDevice(void* dst) const;
  [[nodiscard]] runtime::GpuError CopyToDeviceAsync(
      void* dst, const runtime::GpuStream& stream) const;

  [[nodiscard]] runtime::GpuError CopyFromHost(const void* src, size_t size);
  [[nodiscard]] runtime::GpuError CopyFromHostAsync(
      const void* src, size_t size, const runtime::GpuStream& stream);
  [[nodiscard]] runtime::GpuError CopyFromDevice(const void* src, size_t size);
  [[nodiscard]] runtime::GpuError CopyFromDeviceAsync(
      const void* src, size_t size, const runtime::GpuStream& stream);

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
    return static_cast<const T*>(handle_);
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
