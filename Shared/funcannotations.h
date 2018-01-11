/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#else
#define FORCE_INLINE inline __attribute__((always_inline))
#endif

#ifdef __CUDACC__
#define ALWAYS_INLINE
#else
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

#ifdef __CUDACC__
#define NEVER_INLINE
#else
#define NEVER_INLINE __attribute__((noinline))
#endif

#ifdef __CUDACC__
#define SUFFIX(name) name##_gpu
#else
#define SUFFIX(name) name
#endif
