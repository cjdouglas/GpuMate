#include <gtest/gtest.h>

#include "gpu_mate/gpu_runtime.h"

using namespace gpu_mate::runtime;

TEST(TestRuntimeDevice, TestGetDevice) {
  int device_id = -1, count;
  EXPECT_EQ(gpuGetDeviceCount(&count), gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuGetDevice(&device_id), gpuError_t::gpuSuccess);
  EXPECT_GE(device_id, 0);
  EXPECT_LT(device_id, count);
}

TEST(TestRuntimeDevice, TestBadSetDevice) {
  int device_id = -1, count = 0;
  EXPECT_EQ(gpuGetDeviceCount(&count), gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuSetDevice(device_id), gpuError_t::gpuErrorInvalidDevice);
  device_id = count;
  EXPECT_EQ(gpuSetDevice(device_id), gpuError_t::gpuErrorInvalidDevice);
}

TEST(TestRuntimeDevice, TestSetDevice) {
  int device_id = -1;
  EXPECT_EQ(gpuGetDevice(&device_id), gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuSetDevice(device_id), gpuError_t::gpuSuccess);
}

TEST(TestRuntimeDevice, TestGetDeviceCount) {
  int count;
  EXPECT_EQ(gpuGetDeviceCount(&count), gpuError_t::gpuSuccess);
  EXPECT_GE(count, 1);
}

TEST(TestRuntimeDevice, TestDeviceSynchronize) {
  EXPECT_EQ(gpuDeviceSynchronize(), gpuError_t::gpuSuccess);
}

TEST(TestRuntimeDevice, TestDeviceReset) {
  void* ptr;
  const size_t size = 4;
  EXPECT_EQ(gpuMalloc(&ptr, size), gpuError_t::gpuSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(gpuDeviceReset(), gpuError_t::gpuSuccess);
}
