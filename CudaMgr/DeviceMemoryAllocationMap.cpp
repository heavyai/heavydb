/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include <utility>

#include "CudaMgr/DeviceMemoryAllocationMap.h"
#include "Logger/Logger.h"

namespace CudaMgr_Namespace {

DeviceMemoryAllocationMap::DeviceMemoryAllocationMap() : last_map_changed_cbid_{0u} {}

const DeviceMemoryAllocationMap::Map& DeviceMemoryAllocationMap::getMap() const {
  return map_;
}

const bool DeviceMemoryAllocationMap::mapEmpty() const {
  return map_.empty();
}

void DeviceMemoryAllocationMap::addAllocation(const DevicePtr device_ptr,
                                              const uint64_t size,
                                              const uint64_t handle,
                                              const heavyai::UUID uuid,
                                              const int device_num,
                                              const bool is_slab) {
  Allocation allocation{size, handle, uuid, device_num, is_slab};
  CHECK(map_.try_emplace(device_ptr, std::move(allocation)).second);
}

DeviceMemoryAllocationMap::Allocation DeviceMemoryAllocationMap::removeAllocation(
    const DevicePtr device_ptr) {
  // find the exact match
  auto const itr = map_.find(device_ptr);
  CHECK(itr != map_.end());
  // copy out the allocation
  Allocation allocation = itr->second;
  // remove from map
  map_.erase(itr);
  // return the allocation that was
  return allocation;
}

std::pair<DeviceMemoryAllocationMap::DevicePtr, DeviceMemoryAllocationMap::Allocation>
DeviceMemoryAllocationMap::getAllocation(const DevicePtr device_ptr) {
  // find the map entry above this address
  auto itr = map_.upper_bound(device_ptr);
  CHECK(itr != map_.begin());
  // return the previous entry
  --itr;
  return std::make_pair(itr->first, itr->second);
}

const DeviceMemoryAllocationMap::MapChangedCBID
DeviceMemoryAllocationMap::registerMapChangedCB(MapChangedCB cb) {
  auto const cbid = ++last_map_changed_cbid_;
  CHECK(map_changed_cbs_.emplace(cbid, cb).second) << "Repeat registration";
  return cbid;
}

void DeviceMemoryAllocationMap::unregisterMapChangedCB(const MapChangedCBID cbid) {
  auto itr = map_changed_cbs_.find(cbid);
  CHECK(itr != map_changed_cbs_.end()) << "Failed to unregister";
  map_changed_cbs_.erase(itr);
}

void DeviceMemoryAllocationMap::notifyMapChanged(const heavyai::UUID device_uuid,
                                                 const bool is_slab) const {
  for (auto const& cb : map_changed_cbs_) {
    cb.second(device_uuid, is_slab);
  }
}

}  // namespace CudaMgr_Namespace
