/*
 * Copyright 2021 OmniSci, Inc.
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

/**
 * @file    glob_local_recursive_files.cpp
 * @author  andrew.do@omnisci.com>
 * @brief   shared utility for globbing files, paths can be specified as either a single
 * file, directory or wildcards
 *
 */
#include "Shared/glob_local_recursive_files.h"

#include <boost/filesystem.hpp>

#include "OSDependent/omnisci_glob.h"

namespace shared {

std::set<std::string> glob_local_recursive_files(const std::string& file_path) {
  std::set<std::string> file_paths;

  if (boost::filesystem::is_regular_file(file_path)) {
    file_paths.insert(file_path);
  } else if (boost::filesystem::is_directory(file_path)) {
    for (boost::filesystem::recursive_directory_iterator
             it(file_path, boost::filesystem::symlink_option::recurse),
         eit;
         it != eit;
         ++it) {
      if (!boost::filesystem::is_directory(it->path())) {
        file_paths.insert(it->path().string());
      }
    }
    // empty directories will not throw an error
  } else {
    auto glob_results = omnisci::glob(file_path);
    for (const auto& path : glob_results) {
      if (boost::filesystem::is_directory(path)) {
        auto expanded_paths = glob_local_recursive_files(path);
        file_paths.insert(expanded_paths.begin(), expanded_paths.end());
      } else {
        file_paths.insert(path);
      }
    }
    if (file_paths.empty()) {
      throw_no_match_found(file_path);
    }
  }
  return file_paths;
}

}  // namespace shared