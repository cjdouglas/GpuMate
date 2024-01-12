#pragma once

#include <cstdlib>
#include <iostream>
#include <optional>
#include <utility>

#include "gpu_mate/gpu_runtime.h"

namespace gpu_mate {
namespace utility {
template <typename T>
class GpuBuffer {
 public:
  /// @brief Creates a new empty buffer, with no allocated memory.
  /// @return A new empty buffer, with no allocatd memory.
  static GpuBuffer<T> create() noexcept { return GpuBuffer(); }

  /// @brief Allocates an empty buffer of size size_bytes.
  /// @param size_bytes The requested allocation size, in bytes.
  /// @return An optional possibly containing a buffer with size_bytes
  /// allocated.
  static std::optional<GpuBuffer<T>> create(std::size_t size_bytes) noexcept {
    if (!size_bytes) {
      return GpuBuffer<T>();
    }

    GpuBuffer<T> buf(size_bytes);
    if (buf.initialized()) {
      return buf;
    }
    return std::nullopt;
  }

  /// @brief Deleted copy constructor.
  GpuBuffer(const GpuBuffer&) = delete;

  /// @brief Deleted copy assignment operator.
  GpuBuffer& operator=(const GpuBuffer&) = delete;

  /// @brief Default move constructor.
  GpuBuffer(GpuBuffer&&) = default;

  /// @brief Move assignment operator. Moves the resources over & uninitializes
  /// the other buffer.
  /// @warning This deallocates any currently held resources before moving the
  /// other buffer over.
  /// @param other The incoming buffer.
  /// @return this, with ownership of the resources from other.
  GpuBuffer& operator=(GpuBuffer&& other) {
    if (this == &other) {
      return *this;
    }

    if (initialized_) {
      deallocate();
    }

    handle_ = std::exchange(other.handle_, nullptr);
    size_ = std::exchange(other.size_, 0);
    initialized_ = std::exchange(other.initialized_, false);
    return *this;
  }

  /// @brief Deallocating destructor.
  ~GpuBuffer() { deallocate(); }

  [[nodiscard]] runtime::GpuError copy_from_host(const void* src,
                                                 std::size_t size) {
    if (!initialized_) {
      allocate(size);
    }
  }

  /// @brief Returns a pointer to the underlying handle.
  /// @return A pointer to the underlying handle.
  inline T* handle() { return handle_; }

  /// @brief Returns a const pointer to the underlying handle.
  /// @return A const pointer to the underlying handle.
  inline const T* handle() const { return handle_; }

  /// @brief Returns the current allocated sized of the buffer.
  /// @return The current allocated size of the buffer.
  inline std::size_t size() const { return size_; }

  /// @brief Checks if the buffer is initialized or not.
  /// @return true if the buffer is initialized, false otherwise.
  inline bool initialized() const { return initialized_; }

 private:
  T* handle_;
  std::size_t size_;
  bool initialized_;

  explicit GpuBuffer() noexcept
      : handle_(nullptr), size_(0), initialized_(false) {}

  explicit GpuBuffer(std::size_t size) noexcept { allocate(size); }

  void allocate(std::size_t size) {
    void* ptr = static_cast<void*>(handle_);
    const runtime::GpuError err = runtime::GpuMalloc(&ptr, size);

    if (err == runtime::GpuError::success) {
      handle_ = static_cast<T*>(ptr);
      size_ = size;
      initialized_ = true;
    } else {
      handle_ = nullptr;
      size_ = 0;
      initialized_ = false;
    }
  }

  void deallocate() {
    if (initialized_) {
      runtime::GpuFree(handle_);
      handle_ = nullptr;
      size_ = 0;
      initialized_ = false;
    }
  }
};
}  // namespace utility
}  // namespace gpu_mate
