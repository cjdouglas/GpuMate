#include <gtest/gtest.h>

#include "gpu_mate/gpu_runtime.h"

using namespace gpu_mate::runtime;

TEST(TestRuntimeDevice, TestGetDevice) {
  int device_id = -1, count;
  EXPECT_EQ(GpuGetDeviceCount(&count), GpuError::success);
  EXPECT_EQ(GpuGetDevice(&device_id), GpuError::success);
  EXPECT_GE(device_id, 0);
  EXPECT_LT(device_id, count);
}

TEST(TestRuntimeDevice, TestBadSetDevice) {
  int device_id = -1, count = 0;
  EXPECT_EQ(GpuGetDeviceCount(&count), GpuError::success);
  EXPECT_EQ(GpuSetDevice(device_id), GpuError::invalidDevice);
  device_id = count;
  EXPECT_EQ(GpuSetDevice(device_id), GpuError::invalidDevice);
}

TEST(TestRuntimeDevice, TestSetDevice) {
  int device_id = -1;
  EXPECT_EQ(GpuGetDevice(&device_id), GpuError::success);
  EXPECT_EQ(GpuSetDevice(device_id), GpuError::success);
}

TEST(TestRuntimeDevice, TestGetDeviceCount) {
  int count;
  EXPECT_EQ(GpuGetDeviceCount(&count), GpuError::success);
  EXPECT_GE(count, 1);
}

TEST(TestRuntimeDevice, TestDeviceSynchronize) {
  EXPECT_EQ(GpuDeviceSynchronize(), GpuError::success);
}

TEST(TestRuntimeDevice, TestDeviceReset) {
  void* ptr;
  const size_t size = 4;
  EXPECT_EQ(GpuMalloc(&ptr, size), GpuError::success);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(GpuDeviceReset(), GpuError::success);
}
