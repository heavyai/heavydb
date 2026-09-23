/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/noncopyable.hpp>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

class Resource : boost::noncopyable {
 public:
  Resource() = delete;
  explicit Resource(const DeviceContext& device_ctx,
                    std::string_view resource_tracking_string,
                    ResourceType type);
  virtual ~Resource() = default;

  const DeviceContext& getDeviceContext() const;
  ResourceType getResourceType() const;
  UniqueResourceId getUniqueResourceId() const;

  virtual ResourceHandle getResourceHandle() const = 0;

  // this will always report the memory usage regardless of where
  // and how that memory is allocated (Vulkan, CUDA, slab)
  // filter by type elsewhere as required for statistics purposes
  virtual uint64_t getGpuAllocationSize() const { return 0ULL; }

  bool isUsable() const;
  void isUsable(const char* filename, int lineno) const;
  void validateUsability(const char* filename,
                         int lineno,
                         bool check_thread = true) const;

  const ResourceTrackingData& getTrackingData() const;

  void cleanupResource();

 protected:
  void setUsable();
  void setUnusable();
  void setResourceId(const ResourceId resource_id);

 private:
  virtual void cleanupResourceBase() = 0;
  virtual void makeEmpty() = 0;

  const DeviceContext& device_ctx_;
  ResourceType resource_type_;
  ResourceId resource_id_;
  bool usable_;
  ResourceTrackingData tracking_data_;

  friend class ResourceManager;
};

}  // namespace gfx
