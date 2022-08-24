/*
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "QueryEngine/CompilationOptions.h"

namespace costmodel {

// TODO: add templates
enum AnalyticalTemplate {
  GroupBy,
  Scan,
  Join,
  Reduce,
};

std::string templateToString(AnalyticalTemplate templ);

namespace Detail {

struct Measurement {
  size_t bytes;
  size_t milliseconds;
};

struct BytesOrder {
  bool operator()(const Measurement& m1, const Measurement& m2) {
    return m1.bytes < m2.bytes;
  }
};

using TemplateMeasurements =
    std::unordered_map<AnalyticalTemplate, std::vector<Measurement>>;
using DeviceMeasurements = std::unordered_map<ExecutorDeviceType, TemplateMeasurements>;

}  // namespace Detail

}  // namespace costmodel
