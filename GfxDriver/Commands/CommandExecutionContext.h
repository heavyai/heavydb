/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace gfx {

class DeviceContext;
class CommandList;
class CommandExecutor;

// Interface class managing instances of components required for command
// recording and submission: CommandList, CommandExecutor, and in the
// Vulkan implementation, a VulkanCommandQueue
//
// For multi-threaded command recording, create a CommandExecutionContext for
// each thread
class CommandExecutionContext {
 public:
  virtual ~CommandExecutionContext() = default;

  virtual const DeviceContext& getDeviceContext() const = 0;
  virtual CommandList& getCommandList() const = 0;
  virtual CommandExecutor& getCommandExecutor() const = 0;
  virtual void resetPool() = 0;
};

using CommandExecutionContextUqPtr = std::unique_ptr<CommandExecutionContext>;

}  // namespace gfx
