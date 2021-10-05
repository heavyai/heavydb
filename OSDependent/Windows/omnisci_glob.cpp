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

#include "OSDependent/omnisci_glob.h"

#include <boost/filesystem.hpp>

#include "Shared/clean_windows.h"

#include <string>
#include <vector>

namespace fs = boost::filesystem;

namespace {

bool has_wildcard(const std::string& name) {
  if (name.find('*') != std::string::npos) {
    return true;
  }
  if (name.find('?') != std::string::npos) {
    return true;
  }
  return false;
}

void glob(const fs::path& base, const fs::path& pattern, std::vector<std::string>& out) {
  if (pattern.empty()) {
    out.push_back(base.string());
    return;
  }

  auto it = pattern.begin();
  auto next_part = *(it++);
  fs::path next_pattern;
  for (; it != pattern.end(); ++it) {
    next_pattern /= *it;
  }

  if (has_wildcard(next_part.string())) {
    WIN32_FIND_DATA file_data;
    auto search = base / next_part;
#ifdef _UNICODE
    auto handle = FindFirstFile(search.wstring().data(), &file_data);
#else
    auto handle = FindFirstFile(search.string().data(), &file_data);
#endif
    if (handle != INVALID_HANDLE_VALUE) {
      do {
        fs::path found_part(file_data.cFileName);
        if (!found_part.filename_is_dot() && !found_part.filename_is_dot_dot()) {
          glob(base / found_part, next_pattern, out);
        }
      } while (FindNextFile(handle, &file_data) != 0);
      FindClose(handle);
    }
  } else {
    glob(base / next_part, next_pattern, out);
  }
}

}  // namespace

namespace omnisci {

std::vector<std::string> glob(const std::string& pattern) {
  std::vector<std::string> results;
  fs::path pattern_path(pattern);
  if (!pattern_path.empty()) {
    ::glob(pattern_path.root_path(), pattern_path.relative_path(), results);
  }
  return results;
}

}  // namespace omnisci
