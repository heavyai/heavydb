/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/DeviceContext.h"

#include "GfxDriver/DriverInstance.h"

namespace gfx {

DeviceContext::DeviceContext(const BaseDriver& driver) : driver_(driver) {}

CommandList& DeviceContext::getCommandList() const {
  CHECK(command_list_);
  return *command_list_;
}

CommandExecutor& DeviceContext::getCommandExecutor() const {
  CHECK(command_executor_);
  return *command_executor_;
}

void DeviceContext::destructBase() {
  command_list_ = nullptr;
  command_executor_ = nullptr;
}

DriverType DeviceContext::getDriverType() const {
  return driver_.getDriverType();
}

}  // namespace gfx
