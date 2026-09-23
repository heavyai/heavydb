/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandExecutionContext.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandExecutor.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"

namespace gfx {

VulkanCommandExecutionContext::VulkanCommandExecutionContext(
    const VulkanDeviceContext& device,
    uint32_t command_timeout_ms)
    : device_{device} {
  cmd_pool_ = std::make_unique<VulkanCommandPool>(
      const_cast<VulkanDeviceContext&>(device),
      device.getGraphicsQueue(),
      VulkanDeviceContext::CommandPoolSelector::kExecutor,
      command_timeout_ms);
  cmd_executor_ = std::make_unique<VulkanCommandExecutor>(device, *cmd_pool_);
  cmd_list_ = std::make_unique<CommandList>(*cmd_executor_);
}

VulkanCommandExecutionContext::~VulkanCommandExecutionContext() {
  // destroy stuff
}

const DeviceContext& VulkanCommandExecutionContext::getDeviceContext() const {
  return device_;
}

CommandExecutor& VulkanCommandExecutionContext::getCommandExecutor() const {
  return *cmd_executor_;
}

CommandList& VulkanCommandExecutionContext::getCommandList() const {
  return *cmd_list_;
}

void VulkanCommandExecutionContext::resetPool() {
  cmd_pool_->resetPool();
}

}  // namespace gfx
