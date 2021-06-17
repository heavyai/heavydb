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

#ifdef __APPLE__
#include <libproc.h>
#else
#include <linux/limits.h>
#endif
#include <unistd.h>

#include <boost/filesystem/path.hpp>

#include "Logger/Logger.h"
#ifdef ENABLE_EMBEDDED_DATABASE
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <link.h>
#endif

namespace omnisci {

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

}  // namespace omnisci
