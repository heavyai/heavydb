#pragma once

#include <vector>

#include "Shared/uuid.h"

namespace omnisci {
struct DeviceIdentifier {
  const int32_t index;   //!< index into device group (currently num_gpus - start_gpu)
  const int32_t gpu_id;  //!< Gpu Id for device (ignores start_gpu). Assigned by CudaMgr
                         //!< or GfxDriver
  const UUID uuid;       //!< UUID for device (hardware invariant)
};

using DeviceGroup = std::vector<DeviceIdentifier>;
}  // namespace omnisci
