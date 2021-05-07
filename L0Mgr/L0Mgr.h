/*
 * Copyright 2021 OmniSci, Inc.
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
#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "L0Mgr/L0Exception.h"
#include "L0Mgr/Utils.h"

#include <level_zero/ze_api.h>

namespace l0 {
class L0Device;
class L0Driver {
 private:
  ze_context_handle_t context_;
  ze_driver_handle_t driver_;

  std::vector<std::shared_ptr<L0Device>> devices_;

 public:
  explicit L0Driver(ze_driver_handle_t handle);
  ~L0Driver();
  ze_context_handle_t ctx() const;
  ze_driver_handle_t driver() const;

  const std::vector<std::shared_ptr<L0Device>>& devices() const;
};

class L0Device {
 private:
  ze_device_handle_t device_;
  ze_command_queue_handle_t command_queue_;

  const L0Driver& driver_;

 public:
  L0Device(const L0Driver& driver, ze_device_handle_t device);

  ze_device_handle_t device() const;
  ze_command_queue_handle_t command_queue() const;
  ze_command_list_handle_t create_command_list() const;
  ze_context_handle_t ctx() const;

  ~L0Device();
};

class L0Manager {
 private:
  std::vector<std::shared_ptr<L0Driver>> drivers_;

 public:
  L0Manager();
  const std::vector<std::shared_ptr<L0Driver>>& drivers() const;
};

void copy_host_to_device(int8_t* device_ptr,
                         const int8_t* host_ptr,
                         const size_t num_bytes,
                         ze_command_list_handle_t command_list);

void copy_device_to_host(int8_t* host_ptr,
                         const int8_t* device_ptr,
                         const size_t num_bytes,
                         ze_command_list_handle_t command_list);

int8_t* allocate_device_mem(const size_t num_bytes,
                            ze_command_list_handle_t command_list);

}  // namespace l0