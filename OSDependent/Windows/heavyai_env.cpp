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
