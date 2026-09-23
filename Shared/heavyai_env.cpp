/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/heavyai_env.h"

namespace heavyai {

void setenv(const std::string& var, const std::string& value, const bool overwrite) {
  ::setenv(var.c_str(), value.c_str(), overwrite);
}

std::string env_path_separator() {
  return ":";
}

}  // namespace heavyai
