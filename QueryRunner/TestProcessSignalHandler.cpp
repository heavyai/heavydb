/*
 * Copyright 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "TestProcessSignalHandler.h"

#include <csignal>
#include <cstdlib>

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
  LOG(ERROR) << "Interrupt signal (" << signal_number << ") received.";

  // Perform additional shutdowns
  for (auto& callback : shutdown_callbacks_) {
    callback();
  }

  // Shutdown logging force a flush
  logger::shutdown();

  // Terminate program
  if (signal_number == SIGTERM) {
    std::exit(EXIT_SUCCESS);
  } else {
    std::exit(signal_number);
  }
}

bool TestProcessSignalHandler::has_registered_signal_handler_{false};
std::vector<std::function<void()>> TestProcessSignalHandler::shutdown_callbacks_{};
