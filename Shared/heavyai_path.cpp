/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Shared/heavyai_path.h"

#include <linux/limits.h>
#include <unistd.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "Logger/Logger.h"

namespace heavyai {

std::string get_root_abs_path() {
  char abs_exe_path[PATH_MAX] = {0};
  auto path_len = readlink("/proc/self/exe", abs_exe_path, sizeof(abs_exe_path));
  CHECK_GT(path_len, 0);
  CHECK_LT(static_cast<size_t>(path_len), sizeof(abs_exe_path));
  boost::filesystem::path abs_exe_dir(std::string(abs_exe_path, path_len));
  abs_exe_dir.remove_filename();
  const auto mapd_root = abs_exe_dir.parent_path();
  return mapd_root.string();
}

}  // namespace heavyai
