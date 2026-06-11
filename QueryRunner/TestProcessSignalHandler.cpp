/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TestProcessSignalHandler.h"

#include <csignal>
#include <cstdlib>
#include <iostream>

#include "Logger/Logger.h"

void TestProcessSignalHandler::registerSignalHandler() {
  if (!has_registered_signal_handler_) {
    std::signal(SIGTERM, shutdownSubsystemsAndExit);
    std::signal(SIGSEGV, shutdownSubsystemsAndExit);
    std::signal(SIGABRT, shutdownSubsystemsAndExit);
    has_registered_signal_handler_ = true;
  }
}

void TestProcessSignalHandler::addShutdownCallback(
    std::function<void()> shutdown_callback) {
  shutdown_callbacks_.emplace_back(shutdown_callback);
}

void TestProcessSignalHandler::shutdownSubsystemsAndExit(int signal_number) {
  std::cerr << __func__ << ": Interrupt signal (" << signal_number << ") received."
            << std::endl;

  // Perform additional shutdowns
  for (auto& callback : shutdown_callbacks_) {
    callback();
  }

  // Shutdown logging force a flush
  logger::shutdown();

  // Terminate program
  // TODO: Why convert SIGTERM to EXIT_SUCCESS?
  int const exit_code = signal_number == SIGTERM ? EXIT_SUCCESS : signal_number;
#ifdef __APPLE__
  std::exit(exit_code);
#else
  std::quick_exit(exit_code);
#endif
}

bool TestProcessSignalHandler::has_registered_signal_handler_{false};
std::vector<std::function<void()>> TestProcessSignalHandler::shutdown_callbacks_{};
