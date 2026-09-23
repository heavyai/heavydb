/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Commands/CommandExecutionContext.h"

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"

namespace gfx {
class VulkanCommandExecutionContext : public CommandExecutionContext {
 public:
  explicit VulkanCommandExecutionContext(const VulkanDeviceContext& device,
                                         uint32_t command_timeout_ms);
  ~VulkanCommandExecutionContext() override;

  const DeviceContext& getDeviceContext() const override;
  CommandExecutor& getCommandExecutor() const override;
  CommandList& getCommandList() const override;

  void resetPool() override;

 private:
  const DeviceContext& device_;
  CommandExecutorUqPtr cmd_executor_;
  CommandListUqPtr cmd_list_;
  VulkanCommandPoolUqPtr cmd_pool_;
};

}  // namespace gfx
