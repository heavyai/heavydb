/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

inline std::vector<size_t> get_nvidia_compute_capability() {
  using namespace std::string_literals;
  std::vector<size_t> ret;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    throw std::runtime_error("cudaGetDeviceCount failed: "s + std::to_string(error_id) +
                             ": "s + cudaGetErrorString(error_id));
  }

  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    if (deviceProp.major <= 0) {
      throw std::runtime_error("unexpected cuda compute capability: major "s +
                               std::to_string(deviceProp.major));
    }
    if (deviceProp.minor < 0) {
      throw std::runtime_error("unexpected cuda compute capability: minor "s +
                               std::to_string(deviceProp.minor));
    }

    if (deviceProp.major >= 10) {
      throw std::runtime_error("unexpected cuda compute capability: major "s +
                               std::to_string(deviceProp.major));
    }
    if (deviceProp.minor >= 10) {
      throw std::runtime_error("unexpected cuda compute capability: minor "s +
                               std::to_string(deviceProp.minor));
    }

    ret.push_back((deviceProp.major * 10U) + deviceProp.minor);
  }

  return ret;
}
