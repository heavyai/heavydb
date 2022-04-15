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

#include "DataProvider/TableFragmentsInfo.h"
#include "QueryEngine/CompilationOptions.h"

namespace policy {
using TableFragments = std::vector<FragmentInfo>;

struct SchedulingAssignment {
  ExecutorDeviceType dt;
  int device_id;
};

class ExecutionPolicy {
 public:
  virtual SchedulingAssignment scheduleSingleFragment(const FragmentInfo&,
                                                      size_t frag_id,
                                                      size_t frag_num) const = 0;
  virtual std::vector<ExecutorDeviceType> devices() const {
    return {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU};
  }

  virtual ~ExecutionPolicy() = default;
};

}  // namespace policy
