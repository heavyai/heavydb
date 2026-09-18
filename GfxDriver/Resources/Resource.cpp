/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/Resource.h"

#include "GfxDriver/DeviceContext.h"
#include "Shared/StackTrace.h"

#define STORE_STACK_TRACE 0

namespace gfx {

Resource::Resource(const DeviceContext& device_ctx,
                   std::string_view resource_tracking_string,
                   ResourceType resource_type)
    : device_ctx_{device_ctx}
    , resource_type_{resource_type}
    , resource_id_{0}
    , usable_{false} {
#if STORE_STACK_TRACE
  tracking_data_ = {std::string(resource_tracking_string), getCurrentStackTrace()};
#else
  tracking_data_ = {std::string(resource_tracking_string), ""};
#endif
}

const DeviceContext& Resource::getDeviceContext() const {
  return device_ctx_;
}

ResourceType Resource::getResourceType() const {
  return resource_type_;
}

UniqueResourceId Resource::getUniqueResourceId() const {
  return {device_ctx_.getGpuId(), resource_id_};
}

bool Resource::isUsable() const {
  return usable_;
}

void Resource::isUsable(const char* filename, int lineno) const {
  RUNTIME_EX_ASSERT(
      usable_ == true,
      std::string(filename) + ":" + std::to_string(lineno) + " The resource of type " +
          to_string(getResourceType()) +
          " is not usable. Cannot use resource. This could be a result of the resource "
          "being uninitialized, an error occurred during initialization, or the resource "
          "was cleaned up or flagged unusable by an outside source.");
}

void Resource::validateUsability(const char* filename,
                                 int lineno,
                                 bool check_thread) const {
  isUsable(filename, lineno);
}

const ResourceTrackingData& Resource::getTrackingData() const {
  CHECK(tracking_data_.origin.size()) << "Tracking Data accessed but was never set!";
  return tracking_data_;
}

void Resource::cleanupResource() {
  if (isUsable()) {
    cleanupResourceBase();
  } else {
    makeEmpty();
  }
  setUnusable();
}

void Resource::setUsable() {
  usable_ = true;
}

void Resource::setUnusable() {
  usable_ = false;
}

void Resource::setResourceId(const ResourceId resource_id) {
  resource_id_ = resource_id;
}

}  // namespace gfx
