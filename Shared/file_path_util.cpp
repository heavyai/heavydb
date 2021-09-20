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
 * @file    file_path_util.cpp
 * @author  andrew.do@omnisci.com>
 * @brief   shared utility for globbing files, paths can be specified as either a single
 * file, directory or wildcards
 *
 */
#include "Shared/file_path_util.h"

#include "Logger/Logger.h"
#include "OSDependent/omnisci_glob.h"
#include "Shared/misc.h"

namespace shared {

void validate_sort_options(const std::optional<std::string>& sort_by,
                           const std::optional<std::string>& sort_regex) {
  const auto sort_by_str = to_upper(sort_by.value_or(PATHNAME_ORDER_TYPE));

  if (!shared::contains(supported_file_sort_order_types, sort_by_str)) {
    throw std::runtime_error{FILE_SORT_ORDER_BY_KEY +
                             " must be one of the following options: " +
                             join(supported_file_sort_order_types, ", ") + "."};
  }

  if (shared::contains(non_regex_sort_order_types, sort_by_str) &&
      sort_regex.has_value()) {
    throw std::runtime_error{"Option \"" + FILE_SORT_REGEX_KEY +
                             "\" must not be set for selected option \"" +
                             FILE_SORT_ORDER_BY_KEY + "='" + sort_by_str + "'\"."};
  }

  if (shared::contains(regex_sort_order_types, sort_by_str) && !sort_regex.has_value()) {
    throw std::runtime_error{"Option \"" + FILE_SORT_REGEX_KEY +
                             "\" must be set for selected option \"" +
                             FILE_SORT_ORDER_BY_KEY + "='" + sort_by_str + "'\"."};
  }
}

namespace {

std::vector<std::string> glob_local_recursive_files(const std::string& file_path) {
  std::vector<std::string> file_paths;

  if (boost::filesystem::is_regular_file(file_path)) {
    file_paths.emplace_back(file_path);
  } else if (boost::filesystem::is_directory(file_path)) {
    for (boost::filesystem::recursive_directory_iterator
             it(file_path, boost::filesystem::symlink_option::recurse),
         eit;
         it != eit;
         ++it) {
      if (!boost::filesystem::is_directory(it->path())) {
        file_paths.emplace_back(it->path().string());
      }
    }
    // empty directories will not throw an error
  } else {
    auto glob_results = omnisci::glob(file_path);
    for (const auto& path : glob_results) {
      if (boost::filesystem::is_directory(path)) {
        auto expanded_paths = glob_local_recursive_files(path);
        file_paths.insert(file_paths.end(), expanded_paths.begin(), expanded_paths.end());
      } else {
        file_paths.emplace_back(path);
      }
    }
    if (file_paths.empty()) {
      throw_file_not_found(file_path);
    }
  }
  return file_paths;
}

std::vector<std::string> regex_file_filter(const std::string& pattern,
                                           const std::vector<std::string>& file_paths) {
  boost::regex regex_pattern(pattern);
  std::vector<std::string> matched_file_paths;
  for (const auto& path : file_paths) {
    if (boost::regex_match(path, regex_pattern)) {
      matched_file_paths.emplace_back(path);
    }
  }
  if (matched_file_paths.empty()) {
    throw_no_filter_match(pattern);
  }
  return matched_file_paths;
}

}  // namespace

std::vector<std::string> local_glob_filter_sort_files(
    const std::string& file_path,
    const std::optional<std::string>& filter_regex,
    const std::optional<std::string>& sort_by,
    const std::optional<std::string>& sort_regex) {
  auto result_files = glob_local_recursive_files(file_path);
  if (filter_regex.has_value()) {
    result_files = regex_file_filter(filter_regex.value(), result_files);
  }
  // initial lexicographical order ensures a determinisitc ordering for files not matching
  // sort_regex
  auto initial_file_order = FileOrderLocal(std::nullopt, PATHNAME_ORDER_TYPE);
  auto lexi_comp = initial_file_order.getFileComparator();
  std::stable_sort(result_files.begin(), result_files.end(), lexi_comp);

  auto file_order = FileOrderLocal(sort_regex, sort_by);
  auto comp = file_order.getFileComparator();
  std::stable_sort(result_files.begin(), result_files.end(), comp);
  return result_files;
}

#ifdef HAVE_AWS_S3
namespace {

std::vector<arrow::fs::FileInfo> arrow_fs_regex_file_filter(
    const std::string& pattern,
    const std::vector<arrow::fs::FileInfo>& file_info_list) {
  boost::regex regex_pattern(pattern);
  std::vector<arrow::fs::FileInfo> matched_file_info_list;
  for (const auto& file_info : file_info_list) {
    if (boost::regex_match(file_info.path(), regex_pattern)) {
      matched_file_info_list.emplace_back(file_info);
    }
  }
  if (matched_file_info_list.empty()) {
    throw_no_filter_match(pattern);
  }
  return matched_file_info_list;
}

}  // namespace

std::vector<arrow::fs::FileInfo> arrow_fs_filter_sort_files(
    const std::vector<arrow::fs::FileInfo>& file_paths,
    const std::optional<std::string>& filter_regex,
    const std::optional<std::string>& sort_by,
    const std::optional<std::string>& sort_regex) {
  auto result_files = filter_regex.has_value()
                          ? arrow_fs_regex_file_filter(filter_regex.value(), file_paths)
                          : file_paths;
  // initial lexicographical order ensures a determinisitc ordering for files not matching
  // sort_regex
  auto initial_file_order = FileOrderArrow(std::nullopt, PATHNAME_ORDER_TYPE);
  auto lexi_comp = initial_file_order.getFileComparator();
  std::stable_sort(result_files.begin(), result_files.end(), lexi_comp);

  auto file_order = FileOrderArrow(sort_regex, sort_by);
  auto comp = file_order.getFileComparator();
  std::stable_sort(result_files.begin(), result_files.end(), comp);
  return result_files;
}

#endif  // HAVE_AWS_S3

}  // namespace shared