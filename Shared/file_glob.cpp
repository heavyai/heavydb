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
 * @file    file_glob.cpp
 * @author  andrew.do@omnisci.com>
 * @brief   shared utility for globbing files, paths can be specified as either a single
 * file, directory or wildcards
 *
 */
#include "Shared/file_glob.h"

#include <regex>

#include <boost/filesystem.hpp>

#include "Logger/Logger.h"
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
      throw_file_not_found(file_path);
    }
  }
  return file_paths;
}

std::set<std::string> regex_file_filter(const std::string& pattern,
                                        const std::set<std::string>& file_paths) {
  std::regex regex_pattern(pattern);
  std::set<std::string> matched_file_paths;
  for (const auto& path : file_paths) {
    if (std::regex_match(path, regex_pattern)) {
      matched_file_paths.insert(path);
    }
  }
  if (matched_file_paths.empty()) {
    throw_no_filter_match(pattern);
  }
  return matched_file_paths;
}

std::set<std::pair<std::string, size_t>> regex_file_filter(
    const std::string& pattern,
    const std::set<std::pair<std::string, size_t>>& file_infos) {
  std::regex regex_pattern(pattern);
  std::set<std::pair<std::string, size_t>> matched_file_infos;
  for (const auto& info : file_infos) {
    if (std::regex_match(info.first, regex_pattern)) {
      matched_file_infos.insert(info);
    }
  }
  if (matched_file_infos.empty()) {
    throw_no_filter_match(pattern);
  }
  return matched_file_infos;
}

#ifdef HAVE_AWS_S3
std::vector<Aws::S3::Model::Object> regex_file_filter(
    const std::string& pattern,
    const std::vector<Aws::S3::Model::Object>& objects_list) {
  std::regex regex_pattern(pattern);
  std::vector<Aws::S3::Model::Object> matched_objects_list;
  for (const auto& object : objects_list) {
    if (std::regex_match(object.GetKey(), regex_pattern)) {
      matched_objects_list.push_back(object);
    }
  }
  if (matched_objects_list.empty()) {
    throw_no_filter_match(pattern);
  }
  return matched_objects_list;
}
#endif  // HAVE_AWS_S3

std::set<std::string> glob_recursive_and_filter_local_files(
    const std::string& file_path,
    const std::optional<std::string>& filter) {
  auto glob_result = glob_local_recursive_files(file_path);
  if (filter.has_value()) {
    return regex_file_filter(filter.value(), glob_result);
  }
  return glob_result;
}

}  // namespace shared