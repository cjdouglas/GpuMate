#pragma once

#include <cstdlib>
#include <iostream>

#include "gpu_mate/gpu_runtime.h"

using gpu_mate::runtime::GpuError;
using gpu_mate::runtime::GpuGetErrorString;

#ifndef GPU_CHECK
#define GPU_CHECK(error)                                               \
  if (error != GpuError::success) {                                    \
    const int e = static_cast<int>(error);                             \
    std::cerr << "Gpu error: " << GpuGetErrorString(error) << "(" << e \
              << ") at " << __FILE__ << ":" << __LINE__ << std::endl;  \
    exit(EXIT_FAILURE);                                                \
  }
#endif
