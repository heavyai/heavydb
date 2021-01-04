/*
 * Copyright (c) 2020 OmniSci, Inc.
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

// Functions in gpu_enabled handle two cases:
//  * __CUDACC__ is defined and function call is made from device.
//  * __CUDACC__ is not defined and function call is made from host.
// These do NOT work when __CUDACC__ is defined and call is made from host.

#pragma once

#include "funcannotations.h"

#include <utility>  // std::forward

#ifdef __CUDACC__
#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/reverse.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#else
#include <algorithm>
#include <numeric>
#endif

namespace gpu_enabled {

template <typename... ARGS>
DEVICE auto accumulate(ARGS&&... args) {
#ifdef __CUDACC__
  return thrust::reduce(thrust::device, std::forward<ARGS>(args)...);
#else
  return std::accumulate(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE auto copy(ARGS&&... args) {
#ifdef __CUDACC__
  return thrust::copy(thrust::device, std::forward<ARGS>(args)...);
#else
  return std::copy(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE void fill(ARGS&&... args) {
#ifdef __CUDACC__
  thrust::fill(thrust::device, std::forward<ARGS>(args)...);
#else
  std::fill(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE void iota(ARGS&&... args) {
#ifdef __CUDACC__
  thrust::sequence(thrust::device, std::forward<ARGS>(args)...);
#else
  std::iota(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE auto lower_bound(ARGS&&... args) {
#ifdef __CUDACC__
  return thrust::lower_bound(thrust::device, std::forward<ARGS>(args)...);
#else
  return std::lower_bound(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE void partial_sum(ARGS&&... args) {
#ifdef __CUDACC__
  thrust::inclusive_scan(thrust::device, std::forward<ARGS>(args)...);
#else
  std::partial_sum(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE void reverse(ARGS&&... args) {
#ifdef __CUDACC__
  thrust::reverse(thrust::device, std::forward<ARGS>(args)...);
#else
  std::reverse(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE void sort(ARGS&&... args) {
#ifdef __CUDACC__
  thrust::sort(thrust::device, std::forward<ARGS>(args)...);
#else
  std::sort(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE void swap(ARGS&&... args) {
#ifdef __CUDACC__
  thrust::swap(std::forward<ARGS>(args)...);
#else
  std::swap(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
DEVICE auto upper_bound(ARGS&&... args) {
#ifdef __CUDACC__
  return thrust::upper_bound(thrust::device, std::forward<ARGS>(args)...);
#else
  return std::upper_bound(std::forward<ARGS>(args)...);
#endif
}

}  // namespace gpu_enabled
