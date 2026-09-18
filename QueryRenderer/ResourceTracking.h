/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>

namespace QueryRenderer {

std::string ResourceTrackingString(std::string_view resource_creator,
                                   const int resource_index = -1);
}
