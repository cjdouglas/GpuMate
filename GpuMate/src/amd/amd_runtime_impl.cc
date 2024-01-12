#include <hip/hip_runtime.h>

#include <iostream>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace runtime {
namespace {

static GpuError HipToGpuError(const hipError_t error) {
  switch (error) {
    case hipSuccess:
      return GpuError::success;
    case hipErrorInvalidValue:
      return GpuError::invalidValue;
    case hipErrorOutOfMemory:
      return GpuError::outOfMemory;
    case hipErrorNotInitialized:
      return GpuError::notInitialized;
    case hipErrorDeinitialized:
      return GpuError::deinitialized;
    case hipErrorProfilerDisabled:
      return GpuError::profilerDisabled;
    case hipErrorProfilerNotInitialized:
      return GpuError::profilerNotInitialized;
    case hipErrorProfilerAlreadyStarted:
      return GpuError::profilerAlreadyStarted;
    case hipErrorProfilerAlreadyStopped:
      return GpuError::profilerAlreadyStopped;
    case hipErrorInvalidConfiguration:
      return GpuError::invalidConfiguration;
    case hipErrorInvalidPitchValue:
      return GpuError::invalidPitchValue;
    case hipErrorInvalidSymbol:
      return GpuError::invalidSymbol;
    case hipErrorInvalidDevicePointer:
      return GpuError::invalidDevicePointer;
    case hipErrorInvalidMemcpyDirection:
      return GpuError::invalidMemcpyDirection;
    case hipErrorInsufficientDriver:
      return GpuError::insufficientDriver;
    case hipErrorMissingConfiguration:
      return GpuError::missingConfiguration;
    case hipErrorPriorLaunchFailure:
      return GpuError::priorLaunchFailure;
    case hipErrorInvalidDeviceFunction:
      return GpuError::invalidDeviceFunction;
    case hipErrorNoDevice:
      return GpuError::noDevice;
    case hipErrorInvalidDevice:
      return GpuError::invalidDevice;
    case hipErrorInvalidImage:
      return GpuError::invalidImage;
    case hipErrorInvalidContext:
      return GpuError::invalidContext;
    case hipErrorContextAlreadyCurrent:
      return GpuError::contextAlreadyCurrent;
    case hipErrorMapFailed:
      return GpuError::mapFailed;
    case hipErrorUnmapFailed:
      return GpuError::unmapFailed;
    case hipErrorArrayIsMapped:
      return GpuError::arrayIsMapped;
    case hipErrorAlreadyMapped:
      return GpuError::alreadyMapped;
    case hipErrorNoBinaryForGpu:
      return GpuError::noBinaryForGpu;
    case hipErrorAlreadyAcquired:
      return GpuError::alreadyAcquired;
    case hipErrorNotMapped:
      return GpuError::notMapped;
    case hipErrorNotMappedAsArray:
      return GpuError::notMappedAsArray;
    case hipErrorNotMappedAsPointer:
      return GpuError::notMappedAsPointer;
    case hipErrorECCNotCorrectable:
      return GpuError::eccNotCorrectable;
    case hipErrorUnsupportedLimit:
      return GpuError::unsupportedLimit;
    case hipErrorContextAlreadyInUse:
      return GpuError::contextAlreadyInUse;
    case hipErrorPeerAccessUnsupported:
      return GpuError::peerAccessUnsupported;
    case hipErrorInvalidKernelFile:
      return GpuError::invalidKernelFile;
    case hipErrorInvalidGraphicsContext:
      return GpuError::invalidGraphicsContext;
    case hipErrorInvalidSource:
      return GpuError::invalidSource;
    case hipErrorFileNotFound:
      return GpuError::fileNotFound;
    case hipErrorSharedObjectSymbolNotFound:
      return GpuError::sharedObjectSymbolNotFound;
    case hipErrorSharedObjectInitFailed:
      return GpuError::sharedObjectInitFailed;
    case hipErrorOperatingSystem:
      return GpuError::operatingSystem;
    case hipErrorInvalidHandle:
      return GpuError::invalidHandle;
    case hipErrorIllegalState:
      return GpuError::illegalState;
    case hipErrorNotFound:
      return GpuError::notFound;
    case hipErrorNotReady:
      return GpuError::notReady;
    case hipErrorIllegalAddress:
      return GpuError::illegalAddress;
    case hipErrorLaunchOutOfResources:
      return GpuError::launchOutOfResources;
    case hipErrorLaunchTimeOut:
      return GpuError::launchTimeOut;
    case hipErrorPeerAccessAlreadyEnabled:
      return GpuError::peerAccessAlreadyEnabled;
    case hipErrorPeerAccessNotEnabled:
      return GpuError::peerAccessNotEnabled;
    case hipErrorSetOnActiveProcess:
      return GpuError::setOnActiveProcess;
    case hipErrorContextIsDestroyed:
      return GpuError::contextIsDestroyed;
    case hipErrorAssert:
      return GpuError::assert;
    case hipErrorHostMemoryAlreadyRegistered:
      return GpuError::hostMemoryAlreadyRegistered;
    case hipErrorHostMemoryNotRegistered:
      return GpuError::hostMemoryNotRegistered;
    case hipErrorLaunchFailure:
      return GpuError::launchFailure;
    case hipErrorCooperativeLaunchTooLarge:
      return GpuError::cooperativeLaunchTooLarge;
    case hipErrorNotSupported:
      return GpuError::notSupported;
    case hipErrorStreamCaptureUnsupported:
      return GpuError::streamCaptureUnsupported;
    case hipErrorStreamCaptureInvalidated:
      return GpuError::streamCaptureInvalidated;
    case hipErrorStreamCaptureMerge:
      return GpuError::streamCaptureMerge;
    case hipErrorStreamCaptureUnmatched:
      return GpuError::streamCaptureUnmatched;
    case hipErrorStreamCaptureUnjoined:
      return GpuError::streamCaptureUnjoined;
    case hipErrorStreamCaptureIsolation:
      return GpuError::streamCaptureIsolation;
    case hipErrorStreamCaptureImplicit:
      return GpuError::streamCaptureImplicit;
    case hipErrorCapturedEvent:
      return GpuError::capturedEvent;
    case hipErrorStreamCaptureWrongThread:
      return GpuError::streamCaptureWrongThread;
    case hipErrorGraphExecUpdateFailure:
      return GpuError::graphExecUpdateFailure;
    case hipErrorUnknown:
    default:
      return GpuError::gpuErrorUnknown;
  }
}

static hipError_t GpuToHipError(const GpuError error) {
  switch (error) {
    case GpuError::success:
      return hipSuccess;
    case GpuError::invalidValue:
      return hipErrorInvalidValue;
    case GpuError::outOfMemory:
      return hipErrorOutOfMemory;
    case GpuError::notInitialized:
      return hipErrorNotInitialized;
    case GpuError::deinitialized:
      return hipErrorDeinitialized;
    case GpuError::profilerDisabled:
      return hipErrorProfilerDisabled;
    case GpuError::profilerNotInitialized:
      return hipErrorProfilerNotInitialized;
    case GpuError::profilerAlreadyStarted:
      return hipErrorProfilerAlreadyStarted;
    case GpuError::profilerAlreadyStopped:
      return hipErrorProfilerAlreadyStopped;
    case GpuError::invalidConfiguration:
      return hipErrorInvalidConfiguration;
    case GpuError::invalidPitchValue:
      return hipErrorInvalidPitchValue;
    case GpuError::invalidSymbol:
      return hipErrorInvalidSymbol;
    case GpuError::invalidDevicePointer:
      return hipErrorInvalidDevicePointer;
    case GpuError::invalidMemcpyDirection:
      return hipErrorInvalidMemcpyDirection;
    case GpuError::insufficientDriver:
      return hipErrorInsufficientDriver;
    case GpuError::missingConfiguration:
      return hipErrorMissingConfiguration;
    case GpuError::priorLaunchFailure:
      return hipErrorPriorLaunchFailure;
    case GpuError::invalidDeviceFunction:
      return hipErrorInvalidDeviceFunction;
    case GpuError::noDevice:
      return hipErrorNoDevice;
    case GpuError::invalidDevice:
      return hipErrorInvalidDevice;
    case GpuError::invalidImage:
      return hipErrorInvalidImage;
    case GpuError::invalidContext:
      return hipErrorInvalidContext;
    case GpuError::contextAlreadyCurrent:
      return hipErrorContextAlreadyCurrent;
    case GpuError::mapFailed:
      return hipErrorMapFailed;
    case GpuError::unmapFailed:
      return hipErrorUnmapFailed;
    case GpuError::arrayIsMapped:
      return hipErrorArrayIsMapped;
    case GpuError::alreadyMapped:
      return hipErrorAlreadyMapped;
    case GpuError::noBinaryForGpu:
      return hipErrorNoBinaryForGpu;
    case GpuError::alreadyAcquired:
      return hipErrorAlreadyAcquired;
    case GpuError::notMapped:
      return hipErrorNotMapped;
    case GpuError::notMappedAsArray:
      return hipErrorNotMappedAsArray;
    case GpuError::notMappedAsPointer:
      return hipErrorNotMappedAsPointer;
    case GpuError::eccNotCorrectable:
      return hipErrorECCNotCorrectable;
    case GpuError::unsupportedLimit:
      return hipErrorUnsupportedLimit;
    case GpuError::contextAlreadyInUse:
      return hipErrorContextAlreadyInUse;
    case GpuError::peerAccessUnsupported:
      return hipErrorPeerAccessUnsupported;
    case GpuError::invalidKernelFile:
      return hipErrorInvalidKernelFile;
    case GpuError::invalidGraphicsContext:
      return hipErrorInvalidGraphicsContext;
    case GpuError::invalidSource:
      return hipErrorInvalidSource;
    case GpuError::fileNotFound:
      return hipErrorFileNotFound;
    case GpuError::sharedObjectSymbolNotFound:
      return hipErrorSharedObjectSymbolNotFound;
    case GpuError::sharedObjectInitFailed:
      return hipErrorSharedObjectInitFailed;
    case GpuError::operatingSystem:
      return hipErrorOperatingSystem;
    case GpuError::invalidHandle:
      return hipErrorInvalidHandle;
    case GpuError::illegalState:
      return hipErrorIllegalState;
    case GpuError::notFound:
      return hipErrorNotFound;
    case GpuError::notReady:
      return hipErrorNotReady;
    case GpuError::illegalAddress:
      return hipErrorIllegalAddress;
    case GpuError::launchOutOfResources:
      return hipErrorLaunchOutOfResources;
    case GpuError::launchTimeOut:
      return hipErrorLaunchTimeOut;
    case GpuError::peerAccessAlreadyEnabled:
      return hipErrorPeerAccessAlreadyEnabled;
    case GpuError::peerAccessNotEnabled:
      return hipErrorPeerAccessNotEnabled;
    case GpuError::setOnActiveProcess:
      return hipErrorSetOnActiveProcess;
    case GpuError::contextIsDestroyed:
      return hipErrorContextIsDestroyed;
    case GpuError::assert:
      return hipErrorAssert;
    case GpuError::hostMemoryAlreadyRegistered:
      return hipErrorHostMemoryAlreadyRegistered;
    case GpuError::hostMemoryNotRegistered:
      return hipErrorHostMemoryNotRegistered;
    case GpuError::launchFailure:
      return hipErrorLaunchFailure;
    case GpuError::cooperativeLaunchTooLarge:
      return hipErrorCooperativeLaunchTooLarge;
    case GpuError::notSupported:
      return hipErrorNotSupported;
    case GpuError::streamCaptureUnsupported:
      return hipErrorStreamCaptureUnsupported;
    case GpuError::streamCaptureInvalidated:
      return hipErrorStreamCaptureInvalidated;
    case GpuError::streamCaptureMerge:
      return hipErrorStreamCaptureMerge;
    case GpuError::streamCaptureUnmatched:
      return hipErrorStreamCaptureUnmatched;
    case GpuError::streamCaptureUnjoined:
      return hipErrorStreamCaptureUnjoined;
    case GpuError::streamCaptureIsolation:
      return hipErrorStreamCaptureIsolation;
    case GpuError::streamCaptureImplicit:
      return hipErrorStreamCaptureImplicit;
    case GpuError::capturedEvent:
      return hipErrorCapturedEvent;
    case GpuError::streamCaptureWrongThread:
      return hipErrorStreamCaptureWrongThread;
    case GpuError::graphExecUpdateFailure:
      return hipErrorGraphExecUpdateFailure;
    case GpuError::gpuErrorUnknown:
    default:
      return hipErrorUnknown;
  }
}

static hipMemcpyKind MapMemcpyKind(const GpuMemcpyKind copy_kind) {
  switch (copy_kind) {
    case GpuMemcpyKind::hostToHost:
      return hipMemcpyHostToHost;
    case GpuMemcpyKind::hostToDevice:
      return hipMemcpyHostToDevice;
    case GpuMemcpyKind::deviceToHost:
      return hipMemcpyDeviceToHost;
    case GpuMemcpyKind::deviceToDevice:
      return hipMemcpyDeviceToDevice;
    case GpuMemcpyKind::memcpyDefault:
    default:
      return hipMemcpyDefault;
  }
}
}  // namespace

// GpuStream implementation

GpuStream::GpuStream() {
  hipStream_t handle;
  GPU_CHECK(HipToGpuError(hipStreamCreate(&handle)));
  handle_ = static_cast<void*>(handle);
}

GpuStream::~GpuStream() {
  hipStream_t handle = static_cast<hipStream_t>(handle_);
  GPU_CHECK(HipToGpuError(hipStreamDestroy(handle)));
}

// Device management

GpuError GpuGetDevice(int* id) { return HipToGpuError(hipGetDevice(id)); }

GpuError GpuSetDevice(int id) { return HipToGpuError(hipSetDevice(id)); }

GpuError GpuGetDeviceCount(int* count) {
  return HipToGpuError(hipGetDeviceCount(count));
}

GpuError GpuDeviceSynchronize() {
  return HipToGpuError(hipDeviceSynchronize());
}

GpuError GpuDeviceReset() { return HipToGpuError(hipDeviceReset()); }

// Error handling

GpuError GpuGetLastError() { return HipToGpuError(hipGetLastError()); }

GpuError GpuPeekAtLastError() { return HipToGpuError(hipPeekAtLastError()); }

const char* GpuGetErrorName(GpuError error) {
  return hipGetErrorName(GpuToHipError(error));
}

const char* GpuGetErrorString(GpuError error) {
  return hipGetErrorString(GpuToHipError(error));
}

// Stream management

GpuError GpuStreamSynchronize(const GpuStream& stream) {
  hipStream_t handle = static_cast<hipStream_t>(*stream);
  return HipToGpuError(hipStreamSynchronize(handle));
}

// Memory management

GpuError GpuMalloc(void** ptr, const size_t size) {
  return HipToGpuError(hipMalloc(ptr, size));
}

GpuError GpuMallocHost(void** ptr, const size_t size) {
  hipError_t err = hipHostMalloc(ptr, size);
  return HipToGpuError(err);
}

GpuError GpuMemcpy(void* dst, const void* src, const size_t size,
                   const GpuMemcpyKind copy_kind) {
  return HipToGpuError(hipMemcpy(dst, src, size, MapMemcpyKind(copy_kind)));
}

GpuError GpuMemcpyAsync(void* dst, const void* src, size_t size,
                        GpuMemcpyKind copy_kind, const GpuStream& stream) {
  hipStream_t handle = static_cast<hipStream_t>(*stream);
  return HipToGpuError(
      hipMemcpyAsync(dst, src, size, MapMemcpyKind(copy_kind), handle));
}

GpuError GpuFree(void* ptr) { return HipToGpuError(hipFree(ptr)); }

GpuError GpuFreeHost(void* ptr) { return HipToGpuError(hipHostFree(ptr)); }

}  // namespace runtime
}  // namespace gpu_mate
