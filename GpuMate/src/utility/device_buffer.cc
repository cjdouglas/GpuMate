#include "gpu_mate/utility/device_buffer.h"

#include <cassert>
#include <utility>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

using gpu_mate::runtime::GpuMemcpyKind;
using gpu_mate::runtime::GpuStream;

namespace gpu_mate {
namespace utility {

DeviceBuffer DeviceBuffer::FromHost(const void* data,
                                    const size_t size) noexcept {
  DeviceBuffer buffer(size);
  GPU_CHECK(runtime::GpuMemcpy(buffer.handle_, data, size,
                               GpuMemcpyKind::hostToDevice));
  return buffer;
}

DeviceBuffer DeviceBuffer::FromHostAsync(const void* data, const size_t size,
                                         const GpuStream& stream) noexcept {
  DeviceBuffer buffer(size);
  GPU_CHECK(runtime::GpuMemcpyAsync(buffer.handle_, data, size,
                                    GpuMemcpyKind::hostToDevice, stream));
  return buffer;
}

DeviceBuffer DeviceBuffer::FromDevice(const void* data,
                                      const size_t size) noexcept {
  DeviceBuffer buffer(size);
  GPU_CHECK(runtime::GpuMemcpy(buffer.handle_, data, size,
                               GpuMemcpyKind::deviceToDevice));
  return buffer;
}

DeviceBuffer DeviceBuffer::FromDeviceAsync(const void* data, const size_t size,
                                           const GpuStream& stream) noexcept {
  DeviceBuffer buffer(size);
  GPU_CHECK(runtime::GpuMemcpyAsync(buffer.handle_, data, size,
                                    GpuMemcpyKind::deviceToDevice, stream));
  return buffer;
}

DeviceBuffer::DeviceBuffer()
    : handle_(nullptr), size_(0), initialized_(false) {}

DeviceBuffer::DeviceBuffer(const size_t size) {
  assert(size > 0);
  Alloc(size);
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  if (initialized_) {
    Free();
  }

  handle_ = std::exchange(other.handle_, nullptr);
  size_ = std::exchange(other.size_, 0);
  initialized_ = std::exchange(other.initialized_, false);

  return *this;
}

DeviceBuffer::~DeviceBuffer() { Free(); }

GpuError DeviceBuffer::CopyToHost(void* dst) const {
  if (!initialized_) {
    return GpuError::notInitialized;
  }

  return runtime::GpuMemcpy(dst, handle_, size_, GpuMemcpyKind::deviceToHost);
}

GpuError DeviceBuffer::CopyToHostAsync(void* dst,
                                       const GpuStream& stream) const {
  if (!initialized_) {
    return GpuError::notInitialized;
  }

  return runtime::GpuMemcpyAsync(dst, handle_, size_,
                                 GpuMemcpyKind::deviceToHost, stream);
}

GpuError DeviceBuffer::CopyToDevice(void* dst) const {
  if (!initialized_) {
    return GpuError::notInitialized;
  }

  return runtime::GpuMemcpy(dst, handle_, size_, GpuMemcpyKind::deviceToDevice);
}

GpuError DeviceBuffer::CopyToDeviceAsync(void* dst,
                                         const GpuStream& stream) const {
  if (!initialized_) {
    return GpuError::notInitialized;
  }

  return runtime::GpuMemcpyAsync(dst, handle_, size_,
                                 GpuMemcpyKind::deviceToDevice, stream);
}

GpuError DeviceBuffer::CopyFromHost(const void* src, const size_t size) {
  if (!initialized_) {
    Alloc(size);
  }

  if (size > size_) {
    return GpuError::invalidValue;
  }

  return runtime::GpuMemcpy(handle_, src, size_, GpuMemcpyKind::hostToDevice);
}

GpuError DeviceBuffer::CopyFromHostAsync(const void* src, const size_t size,
                                         const GpuStream& stream) {
  if (!initialized_) {
    Alloc(size);
  }

  if (size > size_) {
    return GpuError::invalidValue;
  }

  return runtime::GpuMemcpyAsync(handle_, src, size_,
                                 GpuMemcpyKind::hostToDevice, stream);
}

GpuError DeviceBuffer::CopyFromDevice(const void* src, const size_t size) {
  if (!initialized_) {
    Alloc(size);
  }

  if (size > size_) {
    return GpuError::invalidValue;
  }

  return runtime::GpuMemcpy(handle_, src, size_, GpuMemcpyKind::deviceToDevice);
}

GpuError DeviceBuffer::CopyFromDeviceAsync(const void* src, const size_t size,
                                           const GpuStream& stream) {
  if (!initialized_) {
    Alloc(size);
  }

  if (size > size_) {
    return GpuError::invalidValue;
  }

  return runtime::GpuMemcpyAsync(handle_, src, size_,
                                 GpuMemcpyKind::deviceToDevice, stream);
}

void DeviceBuffer::Alloc(const size_t size) {
  GPU_CHECK(runtime::GpuMalloc(&handle_, size));
  size_ = size;
  initialized_ = true;
}

void DeviceBuffer::Free() {
  if (initialized_) {
    GPU_CHECK(runtime::GpuFree(handle_));
    handle_ = nullptr;
    size_ = 0;
    initialized_ = false;
  }
}
}  // namespace utility
}  // namespace gpu_mate
