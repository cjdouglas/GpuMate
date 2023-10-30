#include <gtest/gtest.h>

#include "gpu_mate/gpu_runtime.h"

using namespace gpu_mate::runtime;

TEST(TestRuntime, TestSimpleMalloc) {
  void* ptr;
  const size_t size = 4;
  EXPECT_EQ(gpuMalloc(&ptr, size), gpuError_t::gpuSuccess);
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(gpuFree(ptr), gpuError_t::gpuSuccess);
}

TEST(TestRuntime, TestZeroMalloc) {
  void* ptr;
  const size_t size = 0;
  EXPECT_EQ(gpuMalloc(&ptr, size), gpuError_t::gpuSuccess);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_EQ(gpuFree(ptr), gpuError_t::gpuSuccess);
}

TEST(TestRuntime, TestLoopMalloc) {
  void* ptr;
  const size_t size = 4;
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(gpuMalloc(&ptr, size), gpuError_t::gpuSuccess);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(gpuFree(ptr), gpuError_t::gpuSuccess);
  }
}

TEST(TestRuntime, TestBadMalloc) {
  void* ptr;
  const size_t size = ULLONG_MAX;
  EXPECT_EQ(gpuMalloc(&ptr, size), gpuError_t::gpuErrorOutOfMemory);
}

TEST(TestRuntime, TestNullFree) {
  void* ptr = nullptr;
  EXPECT_EQ(gpuFree(ptr), gpuError_t::gpuSuccess);
}

TEST(TestRuntime, TestDoubleFree) {
  void* ptr;
  const size_t size = 4;
  EXPECT_EQ(gpuMalloc(&ptr, size), gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuFree(ptr), gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuFree(ptr), gpuError_t::gpuErrorInvalidValue);
}

TEST(TestRuntime, TestMemcpyH2H) {
  float* ptr = new float(0);
  const float f = 6.9f;
  const size_t size = sizeof(float);
  EXPECT_EQ(gpuMemcpy(ptr, &f, size, gpuMemcpyKind::gpuMemcpyHostToHost),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(*ptr, f);
  delete ptr;
}

TEST(TestRuntime, TestMemcpyH2D) {
  float* ptr;
  const float f1 = 6.9f;
  float f2 = 0;
  const size_t size = sizeof(float);

  EXPECT_NE(f1, f2);
  EXPECT_EQ(gpuMalloc(reinterpret_cast<void**>(&ptr), size),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuMemcpy(ptr, &f1, size, gpuMemcpyKind::gpuMemcpyHostToDevice),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuMemcpy(&f2, ptr, size, gpuMemcpyKind::gpuMemcpyDeviceToHost),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(f1, f2);
  EXPECT_EQ(gpuFree(ptr), gpuError_t::gpuSuccess);
}

TEST(TestRuntime, TestMemcpyD2D) {
  float *ptr1, *ptr2;
  const float f1 = 6.9f;
  float f2 = 0;
  const size_t size = sizeof(float);

  EXPECT_NE(f1, f2);
  EXPECT_EQ(gpuMalloc(reinterpret_cast<void**>(&ptr1), size),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuMalloc(reinterpret_cast<void**>(&ptr2), size),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuMemcpy(ptr1, &f1, size, gpuMemcpyKind::gpuMemcpyHostToDevice),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuMemcpy(ptr2, ptr1, size, gpuMemcpyKind::gpuMemcpyDeviceToDevice),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuMemcpy(&f2, ptr2, size, gpuMemcpyKind::gpuMemcpyDeviceToHost),
            gpuError_t::gpuSuccess);
  EXPECT_EQ(f1, f2);
  EXPECT_EQ(gpuFree(ptr1), gpuError_t::gpuSuccess);
  EXPECT_EQ(gpuFree(ptr2), gpuError_t::gpuSuccess);
}
