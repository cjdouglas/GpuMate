#include <hip/hip_runtime.h>

#include "gpu_mate/gpu_runtime.h"
#include "gpu_mate/internal/utils.h"

namespace gpu_mate {
namespace runtime {
namespace {

static gpuError_t hipErrorToGpuError(const hipError_t error) {
  switch (error) {
    case hipSuccess:
      return gpuError_t::gpuSuccess;
    case hipErrorInvalidValue:
      return gpuError_t::gpuErrorInvalidValue;
    case hipErrorOutOfMemory:
      return gpuError_t::gpuErrorOutOfMemory;
    case hipErrorNotInitialized:
      return gpuError_t::gpuErrorNotInitialized;
    case hipErrorDeinitialized:
      return gpuError_t::gpuErrorDeinitialized;
    case hipErrorProfilerDisabled:
      return gpuError_t::gpuErrorProfilerDisabled;
    case hipErrorProfilerNotInitialized:
      return gpuError_t::gpuErrorProfilerNotInitialized;
    case hipErrorProfilerAlreadyStarted:
      return gpuError_t::gpuErrorProfilerAlreadyStarted;
    case hipErrorProfilerAlreadyStopped:
      return gpuError_t::gpuErrorProfilerAlreadyStopped;
    case hipErrorInvalidConfiguration:
      return gpuError_t::gpuErrorInvalidConfiguration;
    case hipErrorInvalidPitchValue:
      return gpuError_t::gpuErrorInvalidPitchValue;
    case hipErrorInvalidSymbol:
      return gpuError_t::gpuErrorInvalidSymbol;
    case hipErrorInvalidDevicePointer:
      return gpuError_t::gpuErrorInvalidDevicePointer;
    case hipErrorInvalidMemcpyDirection:
      return gpuError_t::gpuErrorInvalidMemcpyDirection;
    case hipErrorInsufficientDriver:
      return gpuError_t::gpuErrorInsufficientDriver;
    case hipErrorMissingConfiguration:
      return gpuError_t::gpuErrorMissingConfiguration;
    case hipErrorPriorLaunchFailure:
      return gpuError_t::gpuErrorPriorLaunchFailure;
    case hipErrorInvalidDeviceFunction:
      return gpuError_t::gpuErrorInvalidDeviceFunction;
    case hipErrorNoDevice:
      return gpuError_t::gpuErrorNoDevice;
    case hipErrorInvalidDevice:
      return gpuError_t::gpuErrorInvalidDevice;
    case hipErrorInvalidImage:
      return gpuError_t::gpuErrorInvalidImage;
    case hipErrorInvalidContext:
      return gpuError_t::gpuErrorInvalidContext;
    case hipErrorContextAlreadyCurrent:
      return gpuError_t::gpuErrorContextAlreadyCurrent;
    case hipErrorMapFailed:
      return gpuError_t::gpuErrorMapFailed;
    case hipErrorUnmapFailed:
      return gpuError_t::gpuErrorUnmapFailed;
    case hipErrorArrayIsMapped:
      return gpuError_t::gpuErrorArrayIsMapped;
    case hipErrorAlreadyMapped:
      return gpuError_t::gpuErrorAlreadyMapped;
    case hipErrorNoBinaryForGpu:
      return gpuError_t::gpuErrorNoBinaryForGpu;
    case hipErrorAlreadyAcquired:
      return gpuError_t::gpuErrorAlreadyAcquired;
    case hipErrorNotMapped:
      return gpuError_t::gpuErrorNotMapped;
    case hipErrorNotMappedAsArray:
      return gpuError_t::gpuErrorNotMappedAsArray;
    case hipErrorNotMappedAsPointer:
      return gpuError_t::gpuErrorNotMappedAsPointer;
    case hipErrorECCNotCorrectable:
      return gpuError_t::gpuErrorECCNotCorrectable;
    case hipErrorUnsupportedLimit:
      return gpuError_t::gpuErrorUnsupportedLimit;
    case hipErrorContextAlreadyInUse:
      return gpuError_t::gpuErrorContextAlreadyInUse;
    case hipErrorPeerAccessUnsupported:
      return gpuError_t::gpuErrorPeerAccessUnsupported;
    case hipErrorInvalidKernelFile:
      return gpuError_t::gpuErrorInvalidKernelFile;
    case hipErrorInvalidGraphicsContext:
      return gpuError_t::gpuErrorInvalidGraphicsContext;
    case hipErrorInvalidSource:
      return gpuError_t::gpuErrorInvalidSource;
    case hipErrorFileNotFound:
      return gpuError_t::gpuErrorFileNotFound;
    case hipErrorSharedObjectSymbolNotFound:
      return gpuError_t::gpuErrorSharedObjectSymbolNotFound;
    case hipErrorSharedObjectInitFailed:
      return gpuError_t::gpuErrorSharedObjectInitFailed;
    case hipErrorOperatingSystem:
      return gpuError_t::gpuErrorOperatingSystem;
    case hipErrorInvalidHandle:
      return gpuError_t::gpuErrorInvalidHandle;
    case hipErrorIllegalState:
      return gpuError_t::gpuErrorIllegalState;
    case hipErrorNotFound:
      return gpuError_t::gpuErrorNotFound;
    case hipErrorNotReady:
      return gpuError_t::gpuErrorNotReady;
    case hipErrorIllegalAddress:
      return gpuError_t::gpuErrorIllegalAddress;
    case hipErrorLaunchOutOfResources:
      return gpuError_t::gpuErrorLaunchOutOfResources;
    case hipErrorLaunchTimeOut:
      return gpuError_t::gpuErrorLaunchTimeOut;
    case hipErrorPeerAccessAlreadyEnabled:
      return gpuError_t::gpuErrorPeerAccessAlreadyEnabled;
    case hipErrorPeerAccessNotEnabled:
      return gpuError_t::gpuErrorPeerAccessNotEnabled;
    case hipErrorSetOnActiveProcess:
      return gpuError_t::gpuErrorSetOnActiveProcess;
    case hipErrorContextIsDestroyed:
      return gpuError_t::gpuErrorContextIsDestroyed;
    case hipErrorAssert:
      return gpuError_t::gpuErrorAssert;
    case hipErrorHostMemoryAlreadyRegistered:
      return gpuError_t::gpuErrorHostMemoryAlreadyRegistered;
    case hipErrorHostMemoryNotRegistered:
      return gpuError_t::gpuErrorHostMemoryNotRegistered;
    case hipErrorLaunchFailure:
      return gpuError_t::gpuErrorLaunchFailure;
    case hipErrorCooperativeLaunchTooLarge:
      return gpuError_t::gpuErrorCooperativeLaunchTooLarge;
    case hipErrorNotSupported:
      return gpuError_t::gpuErrorNotSupported;
    case hipErrorStreamCaptureUnsupported:
      return gpuError_t::gpuErrorStreamCaptureUnsupported;
    case hipErrorStreamCaptureInvalidated:
      return gpuError_t::gpuErrorStreamCaptureInvalidated;
    case hipErrorStreamCaptureMerge:
      return gpuError_t::gpuErrorStreamCaptureMerge;
    case hipErrorStreamCaptureUnmatched:
      return gpuError_t::gpuErrorStreamCaptureUnmatched;
    case hipErrorStreamCaptureUnjoined:
      return gpuError_t::gpuErrorStreamCaptureUnjoined;
    case hipErrorStreamCaptureIsolation:
      return gpuError_t::gpuErrorStreamCaptureIsolation;
    case hipErrorStreamCaptureImplicit:
      return gpuError_t::gpuErrorStreamCaptureImplicit;
    case hipErrorCapturedEvent:
      return gpuError_t::gpuErrorCapturedEvent;
    case hipErrorStreamCaptureWrongThread:
      return gpuError_t::gpuErrorStreamCaptureWrongThread;
    case hipErrorGraphExecUpdateFailure:
      return gpuError_t::gpuErrorGraphExecUpdateFailure;
    case hipErrorUnknown:
    default:
      return gpuError_t::gpuErrorUnknown;
  }
}

static hipError_t gpuErrorToHipError(const gpuError_t error) {
  switch (error) {
    case gpuError_t::gpuSuccess:
      return hipSuccess;
    case gpuError_t::gpuErrorInvalidValue:
      return hipErrorInvalidValue;
    case gpuError_t::gpuErrorOutOfMemory:
      return hipErrorOutOfMemory;
    case gpuError_t::gpuErrorNotInitialized:
      return hipErrorNotInitialized;
    case gpuError_t::gpuErrorDeinitialized:
      return hipErrorDeinitialized;
    case gpuError_t::gpuErrorProfilerDisabled:
      return hipErrorProfilerDisabled;
    case gpuError_t::gpuErrorProfilerNotInitialized:
      return hipErrorProfilerNotInitialized;
    case gpuError_t::gpuErrorProfilerAlreadyStarted:
      return hipErrorProfilerAlreadyStarted;
    case gpuError_t::gpuErrorProfilerAlreadyStopped:
      return hipErrorProfilerAlreadyStopped;
    case gpuError_t::gpuErrorInvalidConfiguration:
      return hipErrorInvalidConfiguration;
    case gpuError_t::gpuErrorInvalidPitchValue:
      return hipErrorInvalidPitchValue;
    case gpuError_t::gpuErrorInvalidSymbol:
      return hipErrorInvalidSymbol;
    case gpuError_t::gpuErrorInvalidDevicePointer:
      return hipErrorInvalidDevicePointer;
    case gpuError_t::gpuErrorInvalidMemcpyDirection:
      return hipErrorInvalidMemcpyDirection;
    case gpuError_t::gpuErrorInsufficientDriver:
      return hipErrorInsufficientDriver;
    case gpuError_t::gpuErrorMissingConfiguration:
      return hipErrorMissingConfiguration;
    case gpuError_t::gpuErrorPriorLaunchFailure:
      return hipErrorPriorLaunchFailure;
    case gpuError_t::gpuErrorInvalidDeviceFunction:
      return hipErrorInvalidDeviceFunction;
    case gpuError_t::gpuErrorNoDevice:
      return hipErrorNoDevice;
    case gpuError_t::gpuErrorInvalidDevice:
      return hipErrorInvalidDevice;
    case gpuError_t::gpuErrorInvalidImage:
      return hipErrorInvalidImage;
    case gpuError_t::gpuErrorInvalidContext:
      return hipErrorInvalidContext;
    case gpuError_t::gpuErrorContextAlreadyCurrent:
      return hipErrorContextAlreadyCurrent;
    case gpuError_t::gpuErrorMapFailed:
      return hipErrorMapFailed;
    case gpuError_t::gpuErrorUnmapFailed:
      return hipErrorUnmapFailed;
    case gpuError_t::gpuErrorArrayIsMapped:
      return hipErrorArrayIsMapped;
    case gpuError_t::gpuErrorAlreadyMapped:
      return hipErrorAlreadyMapped;
    case gpuError_t::gpuErrorNoBinaryForGpu:
      return hipErrorNoBinaryForGpu;
    case gpuError_t::gpuErrorAlreadyAcquired:
      return hipErrorAlreadyAcquired;
    case gpuError_t::gpuErrorNotMapped:
      return hipErrorNotMapped;
    case gpuError_t::gpuErrorNotMappedAsArray:
      return hipErrorNotMappedAsArray;
    case gpuError_t::gpuErrorNotMappedAsPointer:
      return hipErrorNotMappedAsPointer;
    case gpuError_t::gpuErrorECCNotCorrectable:
      return hipErrorECCNotCorrectable;
    case gpuError_t::gpuErrorUnsupportedLimit:
      return hipErrorUnsupportedLimit;
    case gpuError_t::gpuErrorContextAlreadyInUse:
      return hipErrorContextAlreadyInUse;
    case gpuError_t::gpuErrorPeerAccessUnsupported:
      return hipErrorPeerAccessUnsupported;
    case gpuError_t::gpuErrorInvalidKernelFile:
      return hipErrorInvalidKernelFile;
    case gpuError_t::gpuErrorInvalidGraphicsContext:
      return hipErrorInvalidGraphicsContext;
    case gpuError_t::gpuErrorInvalidSource:
      return hipErrorInvalidSource;
    case gpuError_t::gpuErrorFileNotFound:
      return hipErrorFileNotFound;
    case gpuError_t::gpuErrorSharedObjectSymbolNotFound:
      return hipErrorSharedObjectSymbolNotFound;
    case gpuError_t::gpuErrorSharedObjectInitFailed:
      return hipErrorSharedObjectInitFailed;
    case gpuError_t::gpuErrorOperatingSystem:
      return hipErrorOperatingSystem;
    case gpuError_t::gpuErrorInvalidHandle:
      return hipErrorInvalidHandle;
    case gpuError_t::gpuErrorIllegalState:
      return hipErrorIllegalState;
    case gpuError_t::gpuErrorNotFound:
      return hipErrorNotFound;
    case gpuError_t::gpuErrorNotReady:
      return hipErrorNotReady;
    case gpuError_t::gpuErrorIllegalAddress:
      return hipErrorIllegalAddress;
    case gpuError_t::gpuErrorLaunchOutOfResources:
      return hipErrorLaunchOutOfResources;
    case gpuError_t::gpuErrorLaunchTimeOut:
      return hipErrorLaunchTimeOut;
    case gpuError_t::gpuErrorPeerAccessAlreadyEnabled:
      return hipErrorPeerAccessAlreadyEnabled;
    case gpuError_t::gpuErrorPeerAccessNotEnabled:
      return hipErrorPeerAccessNotEnabled;
    case gpuError_t::gpuErrorSetOnActiveProcess:
      return hipErrorSetOnActiveProcess;
    case gpuError_t::gpuErrorContextIsDestroyed:
      return hipErrorContextIsDestroyed;
    case gpuError_t::gpuErrorAssert:
      return hipErrorAssert;
    case gpuError_t::gpuErrorHostMemoryAlreadyRegistered:
      return hipErrorHostMemoryAlreadyRegistered;
    case gpuError_t::gpuErrorHostMemoryNotRegistered:
      return hipErrorHostMemoryNotRegistered;
    case gpuError_t::gpuErrorLaunchFailure:
      return hipErrorLaunchFailure;
    case gpuError_t::gpuErrorCooperativeLaunchTooLarge:
      return hipErrorCooperativeLaunchTooLarge;
    case gpuError_t::gpuErrorNotSupported:
      return hipErrorNotSupported;
    case gpuError_t::gpuErrorStreamCaptureUnsupported:
      return hipErrorStreamCaptureUnsupported;
    case gpuError_t::gpuErrorStreamCaptureInvalidated:
      return hipErrorStreamCaptureInvalidated;
    case gpuError_t::gpuErrorStreamCaptureMerge:
      return hipErrorStreamCaptureMerge;
    case gpuError_t::gpuErrorStreamCaptureUnmatched:
      return hipErrorStreamCaptureUnmatched;
    case gpuError_t::gpuErrorStreamCaptureUnjoined:
      return hipErrorStreamCaptureUnjoined;
    case gpuError_t::gpuErrorStreamCaptureIsolation:
      return hipErrorStreamCaptureIsolation;
    case gpuError_t::gpuErrorStreamCaptureImplicit:
      return hipErrorStreamCaptureImplicit;
    case gpuError_t::gpuErrorCapturedEvent:
      return hipErrorCapturedEvent;
    case gpuError_t::gpuErrorStreamCaptureWrongThread:
      return hipErrorStreamCaptureWrongThread;
    case gpuError_t::gpuErrorGraphExecUpdateFailure:
      return hipErrorGraphExecUpdateFailure;
    case gpuError_t::gpuErrorUnknown:
    default:
      return hipErrorUnknown;
  }
}

static hipMemcpyKind mapMemcpyKind(const gpuMemcpyKind copy_kind) {
  switch (copy_kind) {
    case gpuMemcpyKind::gpuMemcpyHostToHost:
      return hipMemcpyHostToHost;
    case gpuMemcpyKind::gpuMemcpyHostToDevice:
      return hipMemcpyHostToDevice;
    case gpuMemcpyKind::gpuMemcpyDeviceToHost:
      return hipMemcpyDeviceToHost;
    case gpuMemcpyKind::gpuMemcpyDeviceToDevice:
      return hipMemcpyDeviceToDevice;
    case gpuMemcpyKind::gpuMemcpyDefault:
    default:
      return hipMemcpyDefault;
  }
}
}  // namespace

// Device management

gpuError_t gpuGetDevice(int* id) {
  return hipErrorToGpuError(hipGetDevice(id));
}

gpuError_t gpuSetDevice(int id) { return hipErrorToGpuError(hipSetDevice(id)); }

gpuError_t gpuGetDeviceCount(int* count) {
  return hipErrorToGpuError(hipGetDeviceCount(count));
}

gpuError_t gpuDeviceSynchronize() {
  return hipErrorToGpuError(hipDeviceSynchronize());
}

gpuError_t gpuDeviceReset() { return hipErrorToGpuError(hipDeviceReset()); }

// Error handling

gpuError_t gpuGetLastError() { return hipErrorToGpuError(hipGetLastError()); }

gpuError_t gpuPeekAtLastError() {
  return hipErrorToGpuError(hipPeekAtLastError());
}

const char* gpuGetErrorName(gpuError_t error) {
  return hipGetErrorName(gpuErrorToHipError(error));
}

const char* gpuGetErrorString(gpuError_t error) {
  return hipGetErrorString(gpuErrorToHipError(error));
}

// Memory management

gpuError_t gpuMalloc(void** ptr, const size_t size) {
  return hipErrorToGpuError(hipMalloc(ptr, size));
}

gpuError_t gpuMemcpy(void* dst, const void* src, const size_t size,
                     const gpuMemcpyKind copy_kind) {
  return hipErrorToGpuError(
      hipMemcpy(dst, src, size, mapMemcpyKind(copy_kind)));
}

gpuError_t gpuFree(void* ptr) { return hipErrorToGpuError(hipFree(ptr)); }

}  // namespace runtime
}  // namespace gpu_mate
