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

#include <memory>
#include <mutex>

#include "DataSources/DataSource.h"
#include "ExtrapolationModels/ExtrapolationModel.h"
#include "Measurements.h"

#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"
#include "QueryEngine/Dispatchers/ExecutionPolicy.h"

namespace costmodel {

struct CaibrationConfig {
  std::vector<ExecutorDeviceType> devices;
};

using TemplatePredictions =
    std::unordered_map<AnalyticalTemplate, std::unique_ptr<ExtrapolationModel>>;
using DevicePredictions = std::unordered_map<ExecutorDeviceType, TemplatePredictions>;

class CostModel {
 public:
  CostModel(std::unique_ptr<DataSource> _dataSource);
  virtual ~CostModel() = default;

  virtual void calibrate(const CaibrationConfig& conf);
  virtual std::unique_ptr<policy::ExecutionPolicy> predict(
      const RelAlgExecutionUnit& queryDag) = 0;

 protected:
  std::unique_ptr<DataSource> dataSource;

  DevicePredictions dp;

  static const std::vector<AnalyticalTemplate> templates;

  std::vector<ExecutorDeviceType> devices = {ExecutorDeviceType::CPU,
                                             ExecutorDeviceType::GPU};

  std::mutex latch;
};

class CostModelException : std::runtime_error {
 public:
  CostModelException(const std::string& msg)
      : std::runtime_error("CostModel exception: " + msg){};
};

}  // namespace costmodel
