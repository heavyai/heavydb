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
 * @file    file_glob.h
 * @author  andrew.do@omnisci.com>
 * @brief   shared utility for globbing files, paths can be specified as either a single
 * file, directory or wildcards
 *
 */

#pragma once
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

namespace shared {

class FileNotFoundException : public std::runtime_error {
 public:
  FileNotFoundException(const std::string& error_message)
      : std::runtime_error(error_message) {}
};

inline void throw_file_not_found(const std::string& file_path) {
  throw FileNotFoundException{"File or directory \"" + file_path + "\" does not exist."};
}

class NoRegexFilterMatchException : public std::runtime_error {
 public:
  NoRegexFilterMatchException(const std::string& error_message)
      : std::runtime_error(error_message) {}
};

inline void throw_no_filter_match(const std::string& pattern) {
  throw NoRegexFilterMatchException{"No files matched the regex file path \"" + pattern +
                                    "\"."};
}

std::set<std::string> glob_local_recursive_files(const std::string& file_path);

std::set<std::string> regex_file_filter(const std::string& pattern,
                                        const std::set<std::string>& file_paths);
std::set<std::pair<std::string, size_t>> regex_file_filter(
    const std::string& pattern,
    const std::set<std::pair<std::string, size_t>>& file_infos);

std::set<std::string> glob_recursive_and_filter_local_files(
    const std::string& file_path,
    const std::optional<std::string>& filter);

}  // namespace shared
