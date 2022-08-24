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

#include <functional>
#include <unordered_map>

#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/CostModel/Measurements.h"

namespace costmodel {

class ExtrapolationModel {
 public:
  ExtrapolationModel(const std::vector<Detail::Measurement>& measurement)
      : measurement(measurement) {}
  ExtrapolationModel(std::vector<Detail::Measurement>&& measurement)
      : measurement(std::move(measurement)) {}
  virtual ~ExtrapolationModel() = default;

  virtual size_t getExtrapolatedData(size_t bytes) = 0;

 protected:
  std::vector<Detail::Measurement> measurement;
};

}  // namespace costmodel
