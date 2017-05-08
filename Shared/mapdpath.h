/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef _MAPDPATH_H
#define _MAPDPATH_H

#include <boost/filesystem/path.hpp>
#include <glog/logging.h>

#ifdef __APPLE__
#include <libproc.h>
#else
#include <linux/limits.h>
#endif
#include <unistd.h>

inline std::string mapd_root_abs_path() {
#ifdef __APPLE__
  char abs_exe_path[PROC_PIDPATHINFO_MAXSIZE] = {0};
  auto path_len = proc_pidpath(getpid(), abs_exe_path, sizeof(abs_exe_path));
#else
  char abs_exe_path[PATH_MAX] = {0};
  auto path_len = readlink("/proc/self/exe", abs_exe_path, sizeof(abs_exe_path));
#endif
  CHECK_GT(path_len, 0);
  CHECK_LT(static_cast<size_t>(path_len), sizeof(abs_exe_path));
  boost::filesystem::path abs_exe_dir(std::string(abs_exe_path, path_len));
  abs_exe_dir.remove_filename();
#ifdef XCODE
  const auto mapd_root = abs_exe_dir.parent_path().parent_path();
#else
  const auto mapd_root = abs_exe_dir.parent_path();
#endif
  return mapd_root.string();
}

#endif  // _MAPDPATH_H
