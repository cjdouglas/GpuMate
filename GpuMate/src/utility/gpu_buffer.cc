#include "gpu_mate/utility/gpu_buffer.h"

#include <stdexcept>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace utility {
GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : handle_(other.handle_),
      size_(other.size_),
      initialized_(other.initialized_) {
  other.handle_ = nullptr;
  other.size_ = 0;
  other.initialized_ = false;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  if (initialized_) {
    Free();
  }

  handle_ = other.handle_;
  size_ = other.size_;
  initialized_ = other.initialized_;

  other.handle_ = nullptr;
  other.size_ = 0;
  other.initialized_ = false;

  return *this;
}

GpuBuffer::~GpuBuffer() { Free(); }

void GpuBuffer::CopyFrom(const void* data, const size_t size,
                         const GpuMemcpyKind copy_kind) {
  if (initialized_) {
    Free();
  }

  RUNTIME_CHECK(runtime::GpuMalloc(&handle_, size));
  RUNTIME_CHECK(runtime::GpuMemcpy(handle_, data, size, copy_kind));
  size_ = size;
  initialized_ = true;
}

void GpuBuffer::CopyTo(void* dst, const GpuMemcpyKind copy_kind) {
  if (!initialized_) {
    throw std::logic_error("cannot copy uninitialized buffer");
  }

  RUNTIME_CHECK(runtime::GpuMemcpy(dst, handle_, size_, copy_kind));
}

void GpuBuffer::Free() {
  if (initialized_) {
    RUNTIME_CHECK(runtime::GpuFree(handle_));
    handle_ = nullptr;
    size_ = 0;
    initialized_ = false;
  }
}
}  // namespace utility
}  // namespace gpu_mate
