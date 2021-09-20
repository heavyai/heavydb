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
 * @file    file_path_util.h
 * @author  andrew.do@omnisci.com>
 * @brief   shared utility for globbing files, paths can be specified as either a single
 * file, directory or wildcards
 *
 */

#pragma once
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_AWS_S3
#include <arrow/filesystem/filesystem.h>
#endif  // HAVE_AWS_S3
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include "Shared/DateTimeParser.h"
#include "Shared/StringTransform.h"

namespace shared {

using LocalFileComparator = std::function<bool(const std::string&, const std::string&)>;
#ifdef HAVE_AWS_S3
using ArrowFsComparator =
    std::function<bool(const arrow::fs::FileInfo&, const arrow::fs::FileInfo&)>;
#endif  // HAVE_AWS_S3

inline const std::string FILE_SORT_ORDER_BY_KEY = "FILE_SORT_ORDER_BY";
inline const std::string FILE_SORT_REGEX_KEY = "FILE_SORT_REGEX";

inline const std::string PATHNAME_ORDER_TYPE = "PATHNAME";
inline const std::string DATE_MODIFIED_ORDER_TYPE = "DATE_MODIFIED";
inline const std::string REGEX_ORDER_TYPE = "REGEX";
inline const std::string REGEX_DATE_ORDER_TYPE = "REGEX_DATE";
inline const std::string REGEX_NUMBER_ORDER_TYPE = "REGEX_NUMBER";

inline const std::array<std::string, 5> supported_file_sort_order_types{
    PATHNAME_ORDER_TYPE,
    DATE_MODIFIED_ORDER_TYPE,
    REGEX_ORDER_TYPE,
    REGEX_DATE_ORDER_TYPE,
    REGEX_NUMBER_ORDER_TYPE};

inline const std::array<std::string, 2> non_regex_sort_order_types{
    PATHNAME_ORDER_TYPE,
    DATE_MODIFIED_ORDER_TYPE};

inline const std::array<std::string, 3> regex_sort_order_types{REGEX_ORDER_TYPE,
                                                               REGEX_DATE_ORDER_TYPE,
                                                               REGEX_NUMBER_ORDER_TYPE};

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

void validate_sort_options(const std::optional<std::string>& sort_by,
                           const std::optional<std::string>& sort_regex);

std::vector<std::string> local_glob_filter_sort_files(
    const std::string& file_path,
    const std::optional<std::string>& filter_regex,
    const std::optional<std::string>& sort_by,
    const std::optional<std::string>& sort_regex);

#ifdef HAVE_AWS_S3
std::vector<arrow::fs::FileInfo> arrow_fs_filter_sort_files(
    const std::vector<arrow::fs::FileInfo>& file_paths,
    const std::optional<std::string>& filter_regex,
    const std::optional<std::string>& sort_by,
    const std::optional<std::string>& sort_regex);

const std::function<bool(const std::string&, const std::string&)>
    common_regex_date_comp_ = [](const std::string& lhs, const std::string& rhs) -> bool {
  int64_t lhs_t;
  int64_t rhs_t;
  try {
    lhs_t = dateTimeParse<kDATE>(lhs, 0);
  } catch (const std::exception& e) {
    lhs_t = 0;
  }
  try {
    rhs_t = dateTimeParse<kDATE>(rhs, 0);
  } catch (const std::exception& e) {
    rhs_t = 0;
  }
  return lhs_t < rhs_t;
};
const std::function<bool(const std::string&, const std::string&)>
    common_regex_number_comp_ =
        [](const std::string& lhs, const std::string& rhs) -> bool {
  int64_t lhs_i;
  int64_t rhs_i;
  try {
    lhs_i = stoll(lhs, 0);
  } catch (const std::exception& e) {
    lhs_i = 0;
  }
  try {
    rhs_i = stoll(rhs, 0);
  } catch (const std::exception& e) {
    rhs_i = 0;
  }
  return lhs_i < rhs_i;
};
#endif  // HAVE_AWS_S3

template <class T>
class FileOrderBase {
 public:
  inline FileOrderBase(const std::optional<std::string>& sort_regex,
                       const std::optional<std::string>& sort_by)
      : sort_regex_(sort_regex), sort_by_(sort_by) {}

  virtual inline std::string concatCaptureGroups(const std::string& file_name) const {
    CHECK(sort_regex_.has_value());
    boost::match_results<std::string::const_iterator> capture_groups;
    boost::regex regex_pattern(sort_regex_.value());

    if (boost::regex_search(file_name, capture_groups, regex_pattern)) {
      std::stringstream ss;
      for (size_t i = 1; i < capture_groups.size(); i++) {
        ss << capture_groups[i];
      }
      return ss.str();
    }
    return "";  // Empty strings sorted to beginning
  }

  virtual inline std::string getSortBy() {
    return to_upper(sort_by_.value_or(PATHNAME_ORDER_TYPE));
  }

  virtual T getFileComparator() = 0;

 protected:
  std::optional<std::string> sort_regex_;
  std::optional<std::string> sort_by_;
};

class FileOrderLocal : public FileOrderBase<LocalFileComparator> {
 public:
  FileOrderLocal(const std::optional<std::string>& sort_regex,
                 const std::optional<std::string>& sort_by)
      : FileOrderBase<LocalFileComparator>(sort_regex, sort_by) {}

  virtual inline LocalFileComparator getFileComparator() {
    auto comparator_pair = comparator_map_.find(getSortBy());
    CHECK(comparator_pair != comparator_map_.end());
    return comparator_pair->second;
  }

 protected:
  const std::map<std::string, LocalFileComparator> comparator_map_{
      {PATHNAME_ORDER_TYPE,
       [](const std::string& lhs, const std::string& rhs) -> bool { return lhs < rhs; }},
      {DATE_MODIFIED_ORDER_TYPE,
       [](const std::string& lhs, const std::string& rhs) -> bool {
         return boost::filesystem::last_write_time(lhs) <
                boost::filesystem::last_write_time(rhs);
       }},
      {REGEX_ORDER_TYPE,
       [this](const std::string& lhs, const std::string& rhs) -> bool {
         return this->concatCaptureGroups(lhs) < this->concatCaptureGroups(rhs);
       }},
      {REGEX_DATE_ORDER_TYPE,
       [this](const std::string& lhs, const std::string& rhs) -> bool {
         return common_regex_date_comp_(this->concatCaptureGroups(lhs),
                                        this->concatCaptureGroups(rhs));
       }},
      {REGEX_NUMBER_ORDER_TYPE,
       [this](const std::string& lhs, const std::string& rhs) -> bool {
         return common_regex_number_comp_(this->concatCaptureGroups(lhs),
                                          this->concatCaptureGroups(rhs));
       }}};
};

#ifdef HAVE_AWS_S3

class FileOrderArrow : public FileOrderBase<ArrowFsComparator> {
 public:
  FileOrderArrow(const std::optional<std::string>& sort_regex,
                 const std::optional<std::string>& sort_by)
      : FileOrderBase<ArrowFsComparator>(sort_regex, sort_by) {}

  virtual inline ArrowFsComparator getFileComparator() {
    auto comparator_pair = comparator_map_.find(getSortBy());
    CHECK(comparator_pair != comparator_map_.end());
    return comparator_pair->second;
  }

 protected:
  const std::map<std::string, ArrowFsComparator> comparator_map_{
      {PATHNAME_ORDER_TYPE,
       [](const arrow::fs::FileInfo& lhs, const arrow::fs::FileInfo& rhs) -> bool {
         return lhs.path() < rhs.path();
       }},
      {DATE_MODIFIED_ORDER_TYPE,
       [](const arrow::fs::FileInfo& lhs, const arrow::fs::FileInfo& rhs) -> bool {
         return lhs.mtime() < rhs.mtime();
       }},
      {REGEX_ORDER_TYPE,
       [this](const arrow::fs::FileInfo& lhs, const arrow::fs::FileInfo& rhs) -> bool {
         auto lhs_name = lhs.path();
         auto rhs_name = rhs.path();
         return this->concatCaptureGroups(lhs_name) < this->concatCaptureGroups(rhs_name);
       }},
      {REGEX_DATE_ORDER_TYPE,
       [this](const arrow::fs::FileInfo& lhs, const arrow::fs::FileInfo& rhs) -> bool {
         return common_regex_date_comp_(this->concatCaptureGroups(lhs.path()),
                                        this->concatCaptureGroups(rhs.path()));
       }},
      {REGEX_NUMBER_ORDER_TYPE,
       [this](const arrow::fs::FileInfo& lhs, const arrow::fs::FileInfo& rhs) -> bool {
         return common_regex_number_comp_(this->concatCaptureGroups(lhs.path()),
                                          this->concatCaptureGroups(rhs.path()));
       }}};
};

#endif  // HAVE_AWS_S3

}  // namespace shared
