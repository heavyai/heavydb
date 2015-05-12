#ifndef _MAPDPATH_H
#define _MAPDPATH_H

#include <boost/filesystem/path.hpp>
#include <glog/logging.h>

#include <cstdlib>


inline std::string mapd_root_from_exe_path(const char* exe_path) {
  auto abs_exe_path = realpath(exe_path, nullptr);
  CHECK(abs_exe_path);
  boost::filesystem::path abs_exe_dir(abs_exe_path);
  free(abs_exe_path);
  abs_exe_dir.remove_filename();
  const auto mapd_root = abs_exe_dir.parent_path();
  return mapd_root.string();
}

#endif  // _MAPDPATH_H
