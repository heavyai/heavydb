/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <unordered_map>

#include "Shared/uuid.h"

namespace CudaMgr_Namespace {

class DeviceMemoryAllocationMap {
 public:
  using DevicePtr = uint64_t;
  struct Allocation {
    uint64_t size = 0u;
    uint64_t handle = 0u;
    heavyai::UUID device_uuid;
    int device_num = -1;
    bool is_slab = false;
  };
  using Map = std::map<DevicePtr, Allocation>;
  using MapChangedCBID = uint32_t;
  using MapChangedCB = std::function<void(const heavyai::UUID, const bool)>;

  DeviceMemoryAllocationMap();
  ~DeviceMemoryAllocationMap() = default;

  const Map& getMap() const;
  const bool mapEmpty() const;

  void addAllocation(const DevicePtr device_ptr,
                     const uint64_t size,
                     const uint64_t handle,
                     const heavyai::UUID uuid,
                     const int device_num,
                     const bool is_slab);
  Allocation removeAllocation(const DevicePtr device_ptr);
  std::pair<DevicePtr, Allocation> getAllocation(const DevicePtr device_ptr);

  const MapChangedCBID registerMapChangedCB(MapChangedCB cb);
  void unregisterMapChangedCB(const MapChangedCBID cbid);
  void notifyMapChanged(const heavyai::UUID device_uuid, const bool is_slab) const;

 private:
  Map map_;
  MapChangedCBID last_map_changed_cbid_;
  std::unordered_map<MapChangedCBID, MapChangedCB> map_changed_cbs_;
};

using DeviceMemoryAllocationMapUqPtr = std::unique_ptr<DeviceMemoryAllocationMap>;

}  // namespace CudaMgr_Namespace
