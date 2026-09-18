/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ExecutorDeviceType.h"

#include <sstream>

std::ostream& operator<<(std::ostream& os, ExecutorDeviceType device_type) {
  constexpr size_t array_size{2};
  constexpr char const* strings[array_size]{"CPU", "GPU"};
  auto index = static_cast<size_t>(device_type);
  CHECK_LT(index, array_size);
  return os << strings[index];
}

std::string toString(ExecutorDeviceType device_type) {
  std::stringstream ss;
  ss << device_type;
  return ss.str();
}
