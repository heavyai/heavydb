/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OSDependent/heavyai_glob.h"

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

namespace heavyai {

std::vector<std::string> glob(const std::string& pattern) {
  std::vector<std::string> results;
  fs::path pattern_path(pattern);
  if (!pattern_path.empty()) {
    ::glob(pattern_path.root_path(), pattern_path.relative_path(), results);
  }
  return results;
}

}  // namespace heavyai
