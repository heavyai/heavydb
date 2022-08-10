/*
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "QueryEngine/L0Kernel.h"

#include "L0Mgr/L0Exception.h"
#include "L0Mgr/Utils.h"
#include "Logger/Logger.h"  // CHECK

#include <iostream>

L0BinResult spv_to_bin(const std::string& spv,
                       const std::string& name,
                       const unsigned block_size,
                       const l0::L0Manager* mgr) {
  CHECK(!spv.empty());
  CHECK(mgr);

  auto driver = mgr->drivers()[0];
  auto device = driver->devices()[0];

  CHECK(driver);
  CHECK(device);

  auto module = device->create_module((uint8_t*)spv.data(), spv.size(), true);
  auto kernel = module->create_kernel(name.c_str(), 1, 1, 1);

  return {device, kernel, module};
}

L0DeviceCompilationContext::L0DeviceCompilationContext(
    std::shared_ptr<l0::L0Device> device,
    std::shared_ptr<l0::L0Kernel> kernel,
    std::shared_ptr<l0::L0Module> module,
    const l0::L0Manager* l0_mgr,
    const int device_id,
    unsigned int num_options,
    void** option_vals)
    : device_(device)
    , kernel_(kernel)
    , module_(module)
    , l0_mgr_(l0_mgr)
    , device_id_(device_id) {}

L0DeviceCompilationContext::~L0DeviceCompilationContext() {}
