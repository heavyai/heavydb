/*
 * Copyright 2023 HEAVY.AI, Inc.
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
