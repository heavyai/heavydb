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

#include <thread>

#include "Asio.h"
#include "Shared/Logger.h"

std::atomic<bool> Asio::running{true};
boost::asio::io_service Asio::io_context;
boost::asio::signal_set Asio::signals{Asio::io_context};

void Asio::start() {
  // start boost signal handler thread
  std::thread t([] { io_context.run(); });
  t.detach();
}
