#include <gtest/gtest.h>

#include <climits>

#include "gpu_mate/gpu_runtime.h"

using namespace gpu_mate::runtime;

TEST(TestRuntimeMemory, TestSimpleMalloc) {
  void* ptr;
  const size_t size = 4;
  EXPECT_EQ(GpuMalloc(&ptr, size), GpuError::success);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(GpuFree(ptr), GpuError::success);
}

TEST(TestRuntimeMemory, TestZeroMalloc) {
  void* ptr;
  const size_t size = 0;
  EXPECT_EQ(GpuMalloc(&ptr, size), GpuError::success);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_EQ(GpuFree(ptr), GpuError::success);
}

TEST(TestRuntimeMemory, TestLoopMalloc) {
  void* ptr;
  const size_t size = 4;
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(GpuMalloc(&ptr, size), GpuError::success);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(GpuFree(ptr), GpuError::success);
  }
}

TEST(TestRuntimeMemory, TestBadMalloc) {
  void* ptr;
  const size_t size = ULLONG_MAX;
  EXPECT_EQ(GpuMalloc(&ptr, size), GpuError::outOfMemory);
}

TEST(TestRuntimeMemory, TestNullFree) {
  void* ptr = nullptr;
  EXPECT_EQ(GpuFree(ptr), GpuError::success);
}

TEST(TestRuntimeMemory, TestDoubleFree) {
  void* ptr;
  const size_t size = 4;
  EXPECT_EQ(GpuMalloc(&ptr, size), GpuError::success);
  EXPECT_EQ(GpuFree(ptr), GpuError::success);
  EXPECT_EQ(GpuFree(ptr), GpuError::invalidValue);
}

TEST(TestRuntimeMemory, TestMemcpyH2H) {
  float* ptr = new float(0);
  const float f = 6.9f;
  const size_t size = sizeof(float);
  EXPECT_EQ(GpuMemcpy(ptr, &f, size, GpuMemcpyKind::hostToHost),
            GpuError::success);
  EXPECT_EQ(*ptr, f);
  delete ptr;
}

TEST(TestRuntimeMemory, TestMemcpyH2D) {
  float* ptr;
  const float f1 = 6.9f;
  float f2 = 0;
  const size_t size = sizeof(float);

  EXPECT_NE(f1, f2);
  EXPECT_EQ(GpuMalloc(reinterpret_cast<void**>(&ptr), size),
            GpuError::success);
  EXPECT_EQ(GpuMemcpy(ptr, &f1, size, GpuMemcpyKind::hostToDevice),
            GpuError::success);
  EXPECT_EQ(GpuMemcpy(&f2, ptr, size, GpuMemcpyKind::deviceToHost),
            GpuError::success);
  EXPECT_EQ(f1, f2);
  EXPECT_EQ(GpuFree(ptr), GpuError::success);
}

TEST(TestRuntimeMemory, TestMemcpyD2D) {
  float *ptr1, *ptr2;
  const float f1 = 6.9f;
  float f2 = 0;
  const size_t size = sizeof(float);

  EXPECT_NE(f1, f2);
  EXPECT_EQ(GpuMalloc(reinterpret_cast<void**>(&ptr1), size),
            GpuError::success);
  EXPECT_EQ(GpuMalloc(reinterpret_cast<void**>(&ptr2), size),
            GpuError::success);
  EXPECT_EQ(GpuMemcpy(ptr1, &f1, size, GpuMemcpyKind::hostToDevice),
            GpuError::success);
  EXPECT_EQ(GpuMemcpy(ptr2, ptr1, size, GpuMemcpyKind::deviceToDevice),
            GpuError::success);
  EXPECT_EQ(GpuMemcpy(&f2, ptr2, size, GpuMemcpyKind::deviceToHost),
            GpuError::success);
  EXPECT_EQ(f1, f2);
  EXPECT_EQ(GpuFree(ptr1), GpuError::success);
  EXPECT_EQ(GpuFree(ptr2), GpuError::success);
}
