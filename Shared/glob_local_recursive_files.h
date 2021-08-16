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
 * @file    glob_local_recursive_files.h
 * @author  andrew.do@omnisci.com>
 * @brief   shared utility for globbing files, paths can be specified as either a single
 * file, directory or wildcards
 *
 */

#pragma once
#include <set>
#include <stdexcept>

namespace shared {

class FileNotFoundException : public std::runtime_error {
 public:
  FileNotFoundException(const std::string& error_message)
      : std::runtime_error(error_message) {}
};

inline void throw_no_match_found(const std::string& file_path) {
  throw FileNotFoundException{"File or directory \"" + file_path + "\" does not exist."};
}

std::set<std::string> glob_local_recursive_files(const std::string& file_path);

}  // namespace shared