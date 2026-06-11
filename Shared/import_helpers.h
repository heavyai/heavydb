/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMPORT_HELPERS_H_
#define IMPORT_HELPERS_H_

#include "Shared/clean_boost_regex.hpp"

#include <Parser/ReservedKeywords.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace ImportHelpers {

inline bool is_reserved_name(const std::string& name) {
  return reserved_keywords.find(boost::to_upper_copy<std::string>(name)) !=
         reserved_keywords.end();
}

inline std::string sanitize_name(const std::string& name, const bool underscore = false) {
  boost::regex invalid_chars{R"([^0-9a-z_])",
                             boost::regex::extended | boost::regex::icase};
  std::string sanitized_name =
      boost::regex_replace(name, invalid_chars, underscore ? "_" : "");
  boost::regex starts_with_digit{R"(^[0-9].*)"};
  if (boost::regex_match(sanitized_name, starts_with_digit)) {
    sanitized_name = "_" + sanitized_name;
  }
  if (is_reserved_name(sanitized_name)) {
    sanitized_name += "_";
  }
  return sanitized_name;
}

template <typename DatumStringType>
inline bool is_null_datum(const DatumStringType& datum,
                          const std::string& null_indicator) {
  return datum == null_indicator || datum == "NULL" || datum == "\\N";
}

}  // namespace ImportHelpers

#endif  // IMPORT_HELPERS_H_
