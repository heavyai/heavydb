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

#include "DummyCostModel.h"

#include "QueryEngine/Dispatchers/DefaultExecutionPolicy.h"
#include "QueryEngine/Dispatchers/ProportionBasedExecutionPolicy.h"
#include "QueryEngine/Dispatchers/RRExecutionPolicy.h"

namespace costmodel {

std::unique_ptr<policy::ExecutionPolicy> DummyCostModel::predict(
    const RelAlgExecutionUnit& queryDag) {
  std::unique_ptr<policy::ExecutionPolicy> exe_policy;
  if (cfg_.enable_heterogeneous_execution) {
    if (cfg_.forced_heterogeneous_distribution) {
      std::map<ExecutorDeviceType, unsigned> distribution{
          {ExecutorDeviceType::CPU, cfg_.forced_cpu_proportion},
          {ExecutorDeviceType::GPU, cfg_.forced_gpu_proportion}};
      exe_policy = std::make_unique<policy::ProportionBasedExecutionPolicy>(
          std::move(distribution));
    } else {
      exe_policy = std::make_unique<policy::RoundRobinExecutionPolicy>();
    }
  } else {
    exe_policy = std::make_unique<policy::FragmentIDAssignmentExecutionPolicy>(dt_);
  }
  return exe_policy;
}

}  // namespace costmodel
