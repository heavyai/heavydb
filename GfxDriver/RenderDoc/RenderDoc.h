/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// to use RenderDoc for a local build
// change this #define to 1
// and edit the paths in RenderDoc.cpp
// to point to your local install

// edit this
#define ENABLE_RENDERDOC 0

#if ENABLE_RENDERDOC

namespace renderdoc {

void load();
void set_vulkan_device(void* vk_instance);
void begin_frame_capture();
void end_frame_capture();
void unload();

}  // namespace renderdoc

#endif  // ENABLE_RENDERDOC
