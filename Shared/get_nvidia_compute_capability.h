/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
    if (deviceProp.minor >= 10) {
      throw std::runtime_error("unexpected cuda compute capability: minor "s +
                               std::to_string(deviceProp.minor));
    }

    ret.push_back((deviceProp.major * 10U) + deviceProp.minor);
  }

  return ret;
}
