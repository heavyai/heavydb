/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OSDependent/heavyai_path.h"

#ifdef __APPLE__
#include <libproc.h>
#else
#include <linux/limits.h>
#endif
#include <unistd.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include "Logger/Logger.h"
#ifdef ENABLE_EMBEDDED_DATABASE
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <link.h>
#endif

namespace heavyai {

std::string get_root_abs_path() {
#ifdef ENABLE_EMBEDDED_DATABASE
  void* const handle = dlopen(DBEngine_LIBNAME, RTLD_LAZY | RTLD_NOLOAD);
  if (handle) {
    /* Non-zero handle means that libDBEngine.so has been loaded and
       the omnisci root path will be determined with respect to the
       location of the shared library rather than the appliction
       `/proc/self/exe` path. */
    const struct link_map* link_map = 0;
    const int ret = dlinfo(handle, RTLD_DI_LINKMAP, &link_map);
    CHECK_EQ(ret, 0);
    CHECK(link_map);
    /* Despite the dlinfo man page claim that l_name is absolute path,
       it is so only when the location path to the library is absolute,
       say, as specified in LD_LIBRARY_PATH. */
    boost::filesystem::path abs_exe_dir(boost::filesystem::absolute(
        boost::filesystem::canonical(std::string(link_map->l_name))));
    abs_exe_dir.remove_filename();
#ifdef XCODE
    const auto mapd_root = abs_exe_dir.parent_path().parent_path();
#else
    const auto mapd_root = abs_exe_dir.parent_path();
#endif
    return mapd_root.string();
  }
#endif
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

}  // namespace heavyai
