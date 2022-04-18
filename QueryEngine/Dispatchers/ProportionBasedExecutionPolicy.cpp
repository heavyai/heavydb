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

#include "QueryEngine/Dispatchers/ProportionBasedExecutionPolicy.h"
#include "DataMgr/MemoryLevel.h"

#include <numeric>

namespace policy {

ProportionBasedExecutionPolicy::ProportionBasedExecutionPolicy(
    std::map<ExecutorDeviceType, unsigned>&& propotion) {
  CHECK_GT(propotion.size(), 0u);
  proportion_.merge(propotion);
  total_parts_ = std::accumulate(
      proportion_.begin(), proportion_.end(), 0u, [](unsigned acc, auto& cur) {
        return acc + cur.second;
      });
  if (total_parts_ == 0u) {
    throw std::runtime_error(
        "Invalid proportion based execution policy. At least one portion must be greater "
        "than zero.");
  }
}

SchedulingAssignment ProportionBasedExecutionPolicy::scheduleSingleFragment(
    const FragmentInfo& fragment,
    size_t frag_id,
    size_t frag_num) const {
  unsigned scheduled_portion = 0;
  for (auto& [device_type, portion] : proportion_) {
    if (frag_id * total_parts_ < (portion + scheduled_portion) * frag_num) {
      auto memory_level = device_type == ExecutorDeviceType::GPU
                              ? Data_Namespace::GPU_LEVEL
                              : Data_Namespace::CPU_LEVEL;
      int device_id = fragment.deviceIds[static_cast<int>(memory_level)];
      return {device_type, device_id};
    }
    scheduled_portion += portion;
  }
  UNREACHABLE();
  return {};
}
}  // namespace policy
