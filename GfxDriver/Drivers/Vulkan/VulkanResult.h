/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string_view>

#include <vulkan/vulkan.h>

#include "GfxDriver/DeviceContext.h"

namespace gfx {

std::string_view vulkan_result_to_string(VkResult result);
std::string_view vulkan_object_type_to_str(VkObjectType type);

void handle_vulkan_error(VkResult result,
                         std::string_view msg,
                         const char* file,
                         int line);

void check_vulkan_oom(VkResult result,
                      std::string_view msg,
                      std::string_view resource_name,
                      uint64_t size,
                      const DeviceContext& device,
                      std::optional<LoggingCallback> callback,
                      const char* file,
                      int line);

// There is little to no overhead in checking for errors since we're just checking
// function return status. Robust error checking is handled by validation layers which are
// disabled in optimized builds.

// Minimal logging
#define CHECK_VKRESULT(result, msg)                              \
  if (result != VK_SUCCESS) {                                    \
    ::gfx::handle_vulkan_error(result, msg, __FILE__, __LINE__); \
  }

// Error check with extended out of memory logging
#define CHECK_OOM_VKRESULT(result, msg, name, size, device, callback)   \
  if (result != VK_SUCCESS) {                                           \
    ::gfx::check_vulkan_oom(                                            \
        result, msg, name, size, device, callback, __FILE__, __LINE__); \
  }
}  // namespace gfx
