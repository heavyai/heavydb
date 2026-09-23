/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <vector>

class TestProcessSignalHandler {
 public:
  static void registerSignalHandler();
  static void addShutdownCallback(std::function<void()> shutdown_callback);

 private:
  static void shutdownSubsystemsAndExit(int signal_number);

  static bool has_registered_signal_handler_;
  static std::vector<std::function<void()>> shutdown_callbacks_;
};
