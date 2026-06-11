/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif

#ifdef __CUDACC__
#define HOST __host__
#else
#define HOST
#endif

#ifdef __CUDACC__
#define GLOBAL __global__
#else
#define GLOBAL
#endif

#if defined(__CUDACC__) && __CUDACC_VER_MAJOR__ < 8
#define STATIC_QUAL
#else
#define STATIC_QUAL static
#endif

#ifdef __CUDACC__
#define FORCE_INLINE __forceinline__
#elif defined(_WIN32)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#if defined(__CUDACC__) || (defined(__GNUC__) && defined(__SANITIZE_THREAD__)) || \
    defined(WITH_JIT_DEBUG)
#define ALWAYS_INLINE
#elif defined(ENABLE_EMBEDDED_DATABASE) && !defined(_WIN32)
#define ALWAYS_INLINE __attribute__((inline)) __attribute__((__visibility__("protected")))
#elif defined(_WIN32)
#define ALWAYS_INLINE __inline
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#ifdef __CUDACC__
#define NEVER_INLINE
#elif defined(_WIN32)
#define NEVER_INLINE __declspec(noinline)
#else
#define NEVER_INLINE __attribute__((noinline))
#endif

#ifdef __CUDACC__
#define SUFFIX(name) name##_gpu
#else
#define SUFFIX(name) name
#endif

#ifdef _WIN32
#define RUNTIME_EXPORT __declspec(dllexport)
#else
#define RUNTIME_EXPORT
#endif
