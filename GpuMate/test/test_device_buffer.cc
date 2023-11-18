#include <gtest/gtest.h>

#include "gpu_mate/utility/device_buffer.h"

using gpu_mate::runtime::GpuError;
using gpu_mate::runtime::GpuMemcpyKind;
using gpu_mate::utility::DeviceBuffer;

TEST(DeviceBufferTest, TestConstructors) {
  DeviceBuffer buffer;
  EXPECT_FALSE(buffer.Initialized());
  EXPECT_EQ(buffer.Handle(), nullptr);
  EXPECT_EQ(buffer.Size(), 0);

  DeviceBuffer allocated(4);
  EXPECT_TRUE(allocated.Initialized());
  EXPECT_NE(allocated.Handle(), nullptr);
  EXPECT_EQ(allocated.Size(), 4);

  DeviceBuffer moved(DeviceBuffer(8));
  EXPECT_TRUE(moved.Initialized());
  EXPECT_NE(moved.Handle(), nullptr);
  EXPECT_EQ(moved.Size(), 8);
}

TEST(DeviceBufferTest, TestMove) {
  DeviceBuffer dst;
  EXPECT_FALSE(dst.Initialized());
  EXPECT_EQ(dst.Handle(), nullptr);
  EXPECT_EQ(dst.Size(), 0);

  DeviceBuffer src(4);
  EXPECT_TRUE(src.Initialized());
  EXPECT_NE(src.Handle(), nullptr);
  EXPECT_EQ(src.Size(), 4);

  dst = std::move(src);
  EXPECT_TRUE(dst.Initialized());
  EXPECT_NE(dst.Handle(), nullptr);
  EXPECT_EQ(dst.Size(), 4);

  EXPECT_FALSE(src.Initialized());
  EXPECT_EQ(src.Handle(), nullptr);
  EXPECT_EQ(src.Size(), 0);
}

TEST(DeviceBufferTest, TestFromHost) {
  const size_t size = 4;
  const float arr[size] = {1.0f, 2.0f, 3.0f, 4.0f};
  const size_t arr_size = size * sizeof(float);

  const DeviceBuffer& buffer = DeviceBuffer::FromHost(arr, arr_size);
  EXPECT_TRUE(buffer.Initialized());
  EXPECT_NE(buffer.Handle(), nullptr);
  EXPECT_EQ(buffer.Size(), arr_size);
}

TEST(DeviceBufferTest, TestCopyFrom) {
  const size_t size = 4;
  const float arr[size] = {1.0f, 2.0f, 3.0f, 4.0f};
  const size_t arr_size = size * sizeof(float);

  DeviceBuffer empty;
  EXPECT_FALSE(empty.Initialized());
  EXPECT_EQ(empty.Handle(), nullptr);
  EXPECT_EQ(empty.Size(), 0);

  EXPECT_EQ(empty.CopyFrom(arr, arr_size, GpuMemcpyKind::hostToDevice),
            GpuError::success);
  EXPECT_TRUE(empty.Initialized());
  EXPECT_NE(empty.Handle(), nullptr);
  EXPECT_EQ(empty.Size(), arr_size);

  DeviceBuffer allocated(arr_size);
  EXPECT_EQ(allocated.CopyFrom(arr, arr_size, GpuMemcpyKind::hostToDevice),
            GpuError::success);
}

TEST(DeviceBufferTest, TestCopyFromFails) {
  const size_t size = 4;
  const float arr[size] = {1.0f, 2.0f, 3.0f, 4.0f};
  const size_t arr_size = size * sizeof(float);

  DeviceBuffer dst(2);
  EXPECT_EQ(dst.CopyFrom(arr, arr_size, GpuMemcpyKind::hostToDevice),
            GpuError::invalidValue);
}

TEST(DeviceBufferTest, TestCopyTo) {
  const size_t size = 4;
  const float arr[size] = {1.0f, 2.0f, 3.0f, 4.0f};
  const size_t arr_size = size * sizeof(float);
  float arr_dst[size] = {};

  EXPECT_NE(arr_dst[0], arr[0]);
  EXPECT_NE(arr_dst[1], arr[1]);
  EXPECT_NE(arr_dst[2], arr[2]);
  EXPECT_NE(arr_dst[3], arr[3]);

  DeviceBuffer src = DeviceBuffer::FromHost(arr, arr_size);
  EXPECT_EQ(src.CopyTo(arr_dst, GpuMemcpyKind::deviceToHost),
            GpuError::success);
  EXPECT_FLOAT_EQ(arr_dst[0], arr[0]);
  EXPECT_FLOAT_EQ(arr_dst[1], arr[1]);
  EXPECT_FLOAT_EQ(arr_dst[2], arr[2]);
  EXPECT_FLOAT_EQ(arr_dst[3], arr[3]);
}

TEST(DeviceBufferTest, TestCopyToFails) {
  const size_t size = 4;
  float arr[size] = {};
  const size_t arr_size = size * sizeof(float);

  DeviceBuffer empty;
  EXPECT_EQ(empty.CopyTo(arr, GpuMemcpyKind::deviceToHost),
            GpuError::notInitialized);
}