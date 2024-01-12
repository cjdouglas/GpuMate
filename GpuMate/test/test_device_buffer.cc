#include <gtest/gtest.h>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/utility/device_buffer.h"

using gpu_mate::runtime::GpuError;
using gpu_mate::runtime::GpuMemcpyKind;
using gpu_mate::runtime::GpuStream;
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

TEST(DeviceBufferTest, TestFromHostAsync) {
  const size_t size = 4;
  const float arr[size] = {1.0f, 2.0f, 3.0f, 4.0f};
  const size_t arr_size = size * sizeof(float);

  GpuStream stream;
  const DeviceBuffer& buffer =
      DeviceBuffer::FromHostAsync(arr, arr_size, stream);
  EXPECT_TRUE(buffer.Initialized());
  EXPECT_NE(buffer.Handle(), nullptr);
  EXPECT_EQ(buffer.Size(), arr_size);
}

// TODO: tests for all copy methods
