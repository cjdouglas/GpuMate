#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "gpu_mate/gpu_blas.h"
#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/utility/device_buffer.h"

using namespace gpu_mate::blas;
using gpu_mate::runtime::GpuError;
using gpu_mate::runtime::GpuMemcpyKind;
using gpu_mate::utility::DeviceBuffer;

TEST(TestBlas, TestSgemm) {
  GpuBlasHandle handle = GpuBlasHandle::Create();

  const int m = 2;
  const int n = 3;
  const int k = 3;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int lda = m;
  const int ldb = k;
  const int ldc = m;
  const float A[m * k] = {1, 4, 2, 5, 3, 6};
  const float B[k * n] = {1, 4, 7, 2, 5, 8, 3, 6, 9};
  float C[m * n] = {};

  const DeviceBuffer& gpu_A = DeviceBuffer::FromHost(A, m * k * sizeof(float));
  const DeviceBuffer& gpu_B = DeviceBuffer::FromHost(B, k * n * sizeof(float));
  DeviceBuffer gpu_C(m * n * sizeof(float));

  const float* da = gpu_A.TypedHandle<float>();
  const float* db = gpu_B.TypedHandle<float>();
  float* dc = gpu_C.TypedHandle<float>();

  const GpuBlasStatus status =
      sgemm(handle, GpuOperation::none, GpuOperation::none, m, n, k, &alpha, da,
            lda, db, ldb, &beta, dc, ldc);
  EXPECT_EQ(status, GpuBlasStatus::success);
  EXPECT_EQ(gpu_C.CopyTo(C, GpuMemcpyKind::deviceToHost), GpuError::success);

  const float expected[m * n] = {30, 66, 36, 81, 42, 96};
  for (size_t i = 0; i < m * n; ++i) {
    EXPECT_FLOAT_EQ(C[i], expected[i]);
  }
}
