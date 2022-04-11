/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef IMPORT_HELPERS_H_
#define IMPORT_HELPERS_H_

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/regex.hpp>

namespace ImportHelpers {

inline bool is_reserved_name(const std::string& name) {
  UNREACHABLE();
  return false;
}

inline std::string sanitize_name(const std::string& name) {
  boost::regex invalid_chars{R"([^0-9a-z_])",
                             boost::regex::extended | boost::regex::icase};
  std::string sanitized_name = boost::regex_replace(name, invalid_chars, "");
  if (is_reserved_name(sanitized_name)) {
    sanitized_name += "_";
  }
  return sanitized_name;
}

}  // namespace ImportHelpers

#endif  // IMPORT_HELPERS_H_
