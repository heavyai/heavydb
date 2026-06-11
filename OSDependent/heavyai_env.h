/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace heavyai {
void setenv(const std::string& var,
            const std::string& value,
            const bool overwrite = true);
std::string env_path_separator();
}  // namespace heavyai
