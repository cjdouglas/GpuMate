#include "gpu_mate/utility/device_buffer.h"

#include <cassert>
#include <utility>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace utility {
using runtime::GpuMemcpyKind;

DeviceBuffer DeviceBuffer::FromHost(const void* data, const size_t size) {
  DeviceBuffer buffer(size);
  GPU_CHECK(runtime::GpuMemcpy(buffer.handle_, data, size,
                               GpuMemcpyKind::hostToDevice));
  return buffer;
}

DeviceBuffer::DeviceBuffer()
    : handle_(nullptr), size_(0), initialized_(false) {}

DeviceBuffer::DeviceBuffer(const size_t size) {
  assert(size > 0);
  Alloc(size);
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other)
    : handle_(std::exchange(other.handle_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      initialized_(std::exchange(other.initialized_, false)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) {
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

GpuError DeviceBuffer::CopyFrom(const void* src, const size_t size,
                                const GpuMemcpyKind copy_kind) {
  if (!initialized_) {
    Alloc(size);
  }

  if (size > size_) {
    return GpuError::invalidValue;
  }

  return runtime::GpuMemcpy(handle_, src, size, copy_kind);
}

GpuError DeviceBuffer::CopyTo(void* dst, const GpuMemcpyKind copy_kind) const {
  if (!initialized_) {
    return GpuError::notInitialized;
  }

  return runtime::GpuMemcpy(dst, handle_, size_, copy_kind);
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
