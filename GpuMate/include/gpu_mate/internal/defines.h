#pragma once

#if defined __GM_PLATFORM_AMD__ && defined __GM_PLATFORM_NVIDIA__
#error only one of __GM_PLATFORM_AMD__ and __GM_PLATFORM_NVIDIA__ can be defined!
#elif !defined __GM_PLATFORM_AMD__ && !defined __GM_PLATFORM_NVIDIA__
#error none of __GM_PLATFORM_AMD__ or __GM_PLATFORM_NVIDIA__ are defined!
#endif

#if defined(_WIN32) || defined(_WIN64)
#ifdef GM_DLL_EXPORT
#define GPU_MATE_API __declspec(dllexport)
#elif defined(BUILD_SHARED_LIBS)
#define GPU_MATE_API __declspec(dllimport)
#else
#define GPU_MATE_API
#endif
#else
#define GPU_MATE_API
#endif
