/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OSDependent/heavyai_env.h"

#include "Logger/Logger.h"
#include "Shared/clean_windows.h"

namespace heavyai {

void setenv(const std::string& var, const std::string& value, const bool overwrite) {
  CHECK(overwrite) << "setenv without overwrite not supported on Windows";
  _putenv_s(var.c_str(), value.c_str());
}

std::string env_path_separator() {
  return ";";
}

}  // namespace heavyai
