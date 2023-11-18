#pragma once

#include <cstddef>

#include "gpu_mate/internal/defines.h"

namespace gpu_mate {
namespace runtime {

enum class GpuError {
  success = 0,
  invalidValue = 1,
  outOfMemory = 2,
  notInitialized = 3,
  deinitialized = 4,
  profilerDisabled = 5,
  profilerNotInitialized = 6,
  profilerAlreadyStarted = 7,
  profilerAlreadyStopped = 8,
  invalidConfiguration = 9,
  invalidPitchValue = 10,
  invalidSymbol = 11,
  invalidDevicePointer = 12,
  invalidMemcpyDirection = 13,
  insufficientDriver = 14,
  missingConfiguration = 15,
  priorLaunchFailure = 16,
  invalidDeviceFunction = 17,
  noDevice = 18,
  invalidDevice = 19,
  invalidImage = 20,
  invalidContext = 21,
  contextAlreadyCurrent = 22,
  mapFailed = 23,
  unmapFailed = 24,
  arrayIsMapped = 25,
  alreadyMapped = 26,
  noBinaryForGpu = 27,
  alreadyAcquired = 28,
  notMapped = 29,
  notMappedAsArray = 30,
  notMappedAsPointer = 31,
  eccNotCorrectable = 32,
  unsupportedLimit = 33,
  contextAlreadyInUse = 34,
  peerAccessUnsupported = 35,
  invalidKernelFile = 36,
  invalidGraphicsContext = 37,
  invalidSource = 38,
  fileNotFound = 39,
  sharedObjectSymbolNotFound = 40,
  sharedObjectInitFailed = 41,
  operatingSystem = 42,
  invalidHandle = 43,
  illegalState = 44,
  notFound = 45,
  notReady = 46,
  illegalAddress = 47,
  launchOutOfResources = 48,
  launchTimeOut = 49,
  peerAccessAlreadyEnabled = 50,
  peerAccessNotEnabled = 51,
  setOnActiveProcess = 52,
  contextIsDestroyed = 53,
  assert = 54,
  hostMemoryAlreadyRegistered = 55,
  hostMemoryNotRegistered = 56,
  launchFailure = 57,
  cooperativeLaunchTooLarge = 58,
  notSupported = 59,
  streamCaptureUnsupported = 60,
  streamCaptureInvalidated = 61,
  streamCaptureMerge = 62,
  streamCaptureUnmatched = 63,
  streamCaptureUnjoined = 64,
  streamCaptureIsolation = 65,
  streamCaptureImplicit = 66,
  capturedEvent = 67,
  streamCaptureWrongThread = 68,
  graphExecUpdateFailure = 69,

  gpuErrorUnknown = 999
};

enum class GpuMemcpyKind {
  hostToHost = 0,
  hostToDevice = 1,
  deviceToHost = 2,
  deviceToDevice = 3,
  memcpyDefault = 4,
};

class GpuStream {
 public:
  explicit GpuStream();
  ~GpuStream();
  GpuStream(const GpuStream&) = delete;
  GpuStream& operator=(const GpuStream&) = delete;
  GpuStream(GpuStream&&) = default;
  GpuStream& operator=(GpuStream&&) = default;
  void* operator*() const { return handle_; };

 private:
  void* handle_;
};

// Device management

GPU_MATE_API GpuError GpuGetDevice(int* id);

GPU_MATE_API GpuError GpuSetDevice(int id);

GPU_MATE_API GpuError GpuGetDeviceCount(int* count);

GPU_MATE_API GpuError GpuDeviceSynchronize();

GPU_MATE_API GpuError GpuDeviceReset();

// Error handling

GPU_MATE_API GpuError GpuGetLastError();

GPU_MATE_API GpuError GpuPeekAtLastError();

GPU_MATE_API const char* GpuGetErrorName(GpuError error);

GPU_MATE_API const char* GpuGetErrorString(GpuError error);

// Memory management

GPU_MATE_API GpuError GpuMalloc(void** ptr, size_t size);

GPU_MATE_API GpuError GpuMemcpy(void* dst, const void* src, size_t size,
                                GpuMemcpyKind copy_kind);

GPU_MATE_API GpuError GpuFree(void* ptr);

}  // namespace runtime
}  // namespace gpu_mate
