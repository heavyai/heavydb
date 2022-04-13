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

#include "QueryEngine/Dispatchers/DefaultExecutionPolicy.h"
#include "DataMgr/MemoryLevel.h"

namespace policy {
SchedulingAssignment FragmentIDAssignmentExecutionPolicy::scheduleSingleFragment(
    const Fragmenter_Namespace::FragmentInfo& fragment,
    size_t frag_id,
    size_t frag_num) const {
  auto memory_level = dt_ == ExecutorDeviceType::GPU ? Data_Namespace::GPU_LEVEL
                                                     : Data_Namespace::CPU_LEVEL;
  unsigned device_id = fragment.deviceIds[static_cast<unsigned>(memory_level)];
  return {dt_, device_id};
}
std::vector<ExecutorDeviceType> FragmentIDAssignmentExecutionPolicy::devices() const {
  return {dt_};
}
}  // namespace policy
