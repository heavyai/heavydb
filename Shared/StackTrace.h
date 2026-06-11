/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

std::string getCurrentStackTrace(uint32_t num_frames_to_skip = 1,
                                 const char* stop_at_this_frame = nullptr,
                                 bool skip_void_and_stl_frames = true);
