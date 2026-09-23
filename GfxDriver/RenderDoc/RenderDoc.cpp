/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/RenderDoc/RenderDoc.h"

#if ENABLE_RENDERDOC

#include <dlfcn.h>
#include <iostream>

#include "Logger/Logger.h"

// edit this
#include "/path/to/your/renderdoc/include/renderdoc.h"

namespace renderdoc {

// edit this
static std::string rdoc_path{"/path/to/your/renderdoc/lib/librenderdoc.so"};

static void* rdoc_lib_handle = nullptr;
static RENDERDOC_API_1_4_1* rdoc_api = nullptr;
static RENDERDOC_DevicePointer rdoc_device_ptr = nullptr;

void load() {
  LOG(WARNING) << "ENABLING RENDERDOC INTEGRATION!";
  std::cerr << "ENABLING RENDERDOC INTEGRATION!" << std::endl;

  rdoc_lib_handle = dlopen(rdoc_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  CHECK(rdoc_lib_handle) << "Failed to load RenderDoc DSO from " << rdoc_path;
  pRENDERDOC_GetAPI RENDERDOC_GetAPI =
      (pRENDERDOC_GetAPI)dlsym(rdoc_lib_handle, "RENDERDOC_GetAPI");
  int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_4_1, (void**)&rdoc_api);
  CHECK_EQ(ret, 1);
}

void set_vulkan_device(void* vk_instance) {
  CHECK(vk_instance);
  if (rdoc_device_ptr == nullptr) {
    rdoc_device_ptr = RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(vk_instance);
  } else {
    LOG(WARNING) << "RenderDoc configured for first GPU only";
  }
}

void begin_frame_capture() {
  CHECK(rdoc_api);
  CHECK(rdoc_device_ptr);
  rdoc_api->StartFrameCapture(rdoc_device_ptr, NULL);
}

void end_frame_capture() {
  CHECK(rdoc_api);
  CHECK(rdoc_device_ptr);
  rdoc_api->EndFrameCapture(rdoc_device_ptr, NULL);
}

void unload() {
  if (rdoc_lib_handle) {
    dlclose(rdoc_lib_handle);
    rdoc_lib_handle = nullptr;
  }
  rdoc_api = nullptr;
  rdoc_device_ptr = nullptr;
}

}  // namespace renderdoc

#endif  // ENABLE_RENDERDOC
