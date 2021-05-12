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

#include "L0Mgr/L0Mgr.h"
#include <level_zero/ze_api.h>
#include <limits>
#include "Logger/Logger.h"
#include "Utils.h"

namespace l0 {

std::vector<std::shared_ptr<L0Device>> get_devices() {
  zeInit(0);
  uint32_t driver_count = 0;
  zeDriverGet(&driver_count, nullptr);

  std::vector<ze_driver_handle_t> drivers(driver_count);
  zeDriverGet(&driver_count, drivers.data());

  std::vector<std::shared_ptr<L0Device>> res;
  for (auto driver : drivers) {
    uint32_t device_count = 0;
    zeDeviceGet(driver, &device_count, nullptr);

    std::vector<ze_device_handle_t> devices(device_count);
    zeDeviceGet(driver, &device_count, devices.data());

    for (auto device : devices) {
      ze_device_properties_t device_properties;
      L0_SAFE_CALL(zeDeviceGetProperties(device, &device_properties));
      if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
        res.push_back(std::make_shared<L0Device>(driver, device));
      }
    }
  }
  return res;
}

void copy_host_to_device(int8_t* device_ptr,
                         const int8_t* host_ptr,
                         const size_t num_bytes,
                         ze_command_list_handle_t command_list) {
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(
      command_list, device_ptr, host_ptr, num_bytes, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendBarrier(command_list, nullptr, 0, nullptr));
}

void copy_device_to_host(int8_t* host_ptr,
                         const int8_t* device_ptr,
                         const size_t num_bytes,
                         const ze_command_list_handle_t command_list) {
  L0_SAFE_CALL(zeCommandListAppendMemoryCopy(
      command_list, host_ptr, device_ptr, num_bytes, nullptr, 0, nullptr));
  L0_SAFE_CALL(zeCommandListAppendBarrier(command_list, nullptr, 0, nullptr));
}

int8_t* allocate_device_mem(const size_t num_bytes, L0Device& device) {
  ze_device_mem_alloc_desc_t alloc_desc;
  alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  alloc_desc.pNext = nullptr;
  alloc_desc.flags = 0;
  alloc_desc.ordinal = 0;

  void* mem;
  L0_SAFE_CALL(zeMemAllocDevice(
      device.ctx(), &alloc_desc, num_bytes, 0 /*align*/, device.device(), &mem));
  return (int8_t*)mem;
}

L0Device::L0Device(ze_driver_handle_t driver, ze_device_handle_t device)
    : device_(device) {
  ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
  L0_SAFE_CALL(zeContextCreate(driver, &ctx_desc, &context_));
  ze_command_queue_desc_t command_queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                                nullptr,
                                                0,
                                                0,
                                                0,
                                                ZE_COMMAND_QUEUE_MODE_DEFAULT,
                                                ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
  L0_SAFE_CALL(
      zeCommandQueueCreate(context_, device_, &command_queue_desc, &command_queue_));
}

L0Device::~L0Device() {
  L0_SAFE_CALL(zeContextDestroy(context_));
  L0_SAFE_CALL(zeCommandQueueDestroy(command_queue_));
}

ze_context_handle_t L0Device::ctx() const {
  return context_;
}
ze_device_handle_t L0Device::device() const {
  return device_;
}
ze_command_queue_handle_t L0Device::command_queue() const {
  return command_queue_;
}

ze_command_list_handle_t L0Device::create_command_list() const {
  ze_command_list_desc_t desc = {
      ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
      nullptr,
      0,
      0  // flags
  };
  ze_command_list_handle_t res;
  zeCommandListCreate(context_, device_, &desc, &res);
  return res;
}
}  // namespace l0