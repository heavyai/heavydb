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
  const auto mapd_root = abs_exe_dir.parent_path();
  return mapd_root.string();
}

#endif  // _MAPDPATH_H
