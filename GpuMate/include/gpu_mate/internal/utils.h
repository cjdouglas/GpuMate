#pragma once

#include <cstdlib>
#include <iostream>

#include "gpu_mate/gpu_runtime.h"

using gpu_mate::runtime::gpuError_t;
using gpu_mate::runtime::gpuGetErrorString;

#ifndef GPU_CHECK
#define GPU_CHECK(error)                                               \
  if (error != gpuError_t::gpuSuccess) {                               \
    const int e = static_cast<int>(error);                             \
    std::cerr << "Gpu error: " << gpuGetErrorString(error) << "(" << e \
              << ") at " << __FILE__ << ":" << __LINE__ << std::endl;  \
    exit(EXIT_FAILURE);                                                \
  }
#endif
