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

#include "ExecutionPolicy.h"

namespace policy {
/**
 * @brief Allocates kernels to devices according to the specified proportion. Example:
 * proportion = {{CPU, 3}, {GPU, 7}}; would assign 30% of fragments to CPU and 70% to GPU.
 * Within the device type scheduling follows storage-defined strategy.
 *
 */
class ProportionBasedExecutionPolicy : public ExecutionPolicy {
 public:
  ProportionBasedExecutionPolicy(std::map<ExecutorDeviceType, unsigned>&& proportion);
  SchedulingAssignment scheduleSingleFragment(const FragmentInfo&,
                                              size_t frag_id,
                                              size_t frag_num) const override;

 private:
  std::map<ExecutorDeviceType, unsigned> proportion_;
  unsigned total_parts_;
};
}  // namespace policy
