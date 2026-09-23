/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/Buffer.h"
#include "GfxDriver/Resources/Enums.h"

namespace gfx {

Buffer::Buffer(const DeviceContext& device_ctx,
               std::string_view resource_tracking_string,
               const BufferCreateInfo& create_info)
    : Resource(device_ctx,
               resource_tracking_string,
               resource_type_from_buffer_type(create_info.buffer_type))
    , resource_handle_{0}
    , state_{create_info} {
  // need to set the size to 0 here as nothing is actually allocated yet. If the
  // create_info has a size specified, derived classes should do the appropriate
  // allocation in their own constructors and set the resulting size post the allocation.
  state_.size = 0;
}

Buffer::~Buffer() {}

}  // namespace gfx
