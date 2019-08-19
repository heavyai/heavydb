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

#ifndef _STRING_CONVERSIONS_H_
#define _STRING_CONVERSIONS_H_
#include <boost/utility/string_view.hpp>
#include <string>

namespace StringConversions {

const inline std::string& to_string(const std::string& s) {
  return s;
}

std::string inline to_string(const boost::string_view& s) {
  return s.to_string();
}

template <typename String>
int64_t strtol(const String& s) {
    int64_t res = 0;
    bool neg = false;
    auto iter = s.begin();
    if (*iter == '-') {
        neg = true;
        ++iter;
    }
    for (; iter != s.end() && std::isdigit(*iter); iter++) {
      res = res * 10 + (*iter - '0');
    }
    if (neg) {
        res = -res;
    }
    return res;
}


template <typename String>
double strtod(const String& s) {
  auto st = to_string(s);
  return std::stod(st);
}

template <typename String>
float strtof(const String& s) {
  auto st = to_string(s);
  return std::stof(st);
}
}  // namespace StringConversions
#endif /* _STRING_CONVERSIONS_H_ */