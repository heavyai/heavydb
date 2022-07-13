/*
 * Copyright 2022 HEAVY.AI, Inc.
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
