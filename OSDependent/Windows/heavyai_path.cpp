/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OSDependent/heavyai_path.h"

#include <boost/filesystem/path.hpp>

#include "Logger/Logger.h"

#include "Shared/clean_windows.h"

namespace heavyai {

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

}  // namespace heavyai
