#ifndef GPU_MATE_INTERNAL_DEFINES_H
#define GPU_MATE_INTERNAL_DEFINES_H

#if defined __GM_PLATFORM_AMD__ && defined __GM_PLATFORM_NVIDIA__
#error only one of __GM_PLATFORM_AMD__ and __GM_PLATFORM_NVIDIA__ can be defined!
#elif !defined __GM_PLATFORM_AMD__ && !defined __GM_PLATFORM_NVIDIA__
#error none of __GM_PLATFORM_AMD__ or __GM_PLATFORM_NVIDIA__ are defined!
#endif

#if defined(_WIN32) || defined(_WIN64)
#ifdef GPU_MATE_BUILD_SHARED
#define GPU_MATE_API __declspec(dllimport)
#elif defined(GPU_MATE_USE_SHARED)
#define GPU_MATE_API __declspec(dllexport)
#else
#define GPU_MATE_API
#endif
#else
#define GPU_MATE_API
#endif

#endif  // GPU_MATE_INTERNAL_DEFINES_H
