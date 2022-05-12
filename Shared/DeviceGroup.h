/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include <vector>

#include "Shared/uuid.h"

namespace heavyai {
struct DeviceIdentifier {
  const int32_t index;   //!< index into device group (currently num_gpus - start_gpu)
  const int32_t gpu_id;  //!< Gpu Id for device (ignores start_gpu). Assigned by CudaMgr
                         //!< or GfxDriver
  const UUID uuid;       //!< UUID for device (hardware invariant)
};

using DeviceGroup = std::vector<DeviceIdentifier>;
}  // namespace heavyai
