/*
 * Copyright 2020 OmniSci, Inc.
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

#include "OSDependent/omnisci_path.h"

#include <boost/filesystem/path.hpp>

#include "Logger/Logger.h"

#include <windows.h>  // Must be last

namespace {
std::string get_root_abs_path() {
  char abs_exe_path[MAX_PATH];
  auto path_len = GetModuleFileNameA(NULL, abs_exe_path, MAX_PATH);
  CHECK_GT(path_len, 0u);
  CHECK_LT(static_cast<size_t>(path_len), sizeof(abs_exe_path));
  boost::filesystem::path abs_exe_dir(std::string(abs_exe_path, path_len));
  abs_exe_dir.remove_filename();
  const auto mapd_root = abs_exe_dir.parent_path();

  return mapd_root.string();
}
}  // namespace
