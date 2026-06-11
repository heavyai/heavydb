/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OSDependent/heavyai_glob.h"

#include <glob.h>
#include <string>
#include <vector>

namespace heavyai {
std::vector<std::string> glob(const std::string& pattern) {
  std::vector<std::string> results;
  glob_t glob_result;
  ::glob(pattern.c_str(), GLOB_BRACE | GLOB_TILDE, nullptr, &glob_result);
  for (size_t i = 0; i < glob_result.gl_pathc; i++) {
    results.emplace_back(glob_result.gl_pathv[i]);
  }
  globfree(&glob_result);
  return results;
}
}  // namespace heavyai
