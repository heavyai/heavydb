/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace QueryRenderer {

std::string makeLowerCase(const std::string& str);
std::string makeUpperCase(const std::string& str);

template <typename T>
std::string container_to_str(const T& cont,
                             const char start_char,
                             const char end_char,
                             const char* sep) {
  std::stringstream ss;
  ss << start_char;
  if (cont.size()) {
    auto itr = cont.begin();
    ss << *itr;
    while (++itr != cont.end()) {
      ss << sep << *itr;
    }
  }
  ss << end_char;
  return ss.str();
}

template <typename T>
std::string to_string(const std::vector<T>& v) {
  return container_to_str(v, '[', ']', ", ");
}

template <typename T>
std::string to_string(const std::set<T>& s) {
  return container_to_str(s, '{', '}', ", ");
}

template <typename EnumType>
std::string enum_to_string(const EnumType start,
                           const EnumType end,
                           std::function<std::string(const EnumType)> to_str_func,
                           std::function<bool(const EnumType)> cond_func = nullptr) {
  std::string rtn = "[";
  if (!cond_func) {
    for (auto i = static_cast<int>(start); i < static_cast<int>(end); ++i) {
      rtn += "\"" + to_str_func(static_cast<EnumType>(i)) + "\", ";
    }
  } else {
    for (auto i = static_cast<int>(start); i < static_cast<int>(end); ++i) {
      auto enumval = static_cast<EnumType>(i);
      if (cond_func(enumval)) {
        rtn += "\"" + to_str_func(enumval) + "\", ";
      }
    }
  }
  if (rtn.size() > 1) {
    // remove last ", "
    rtn.pop_back();
    rtn.pop_back();
  }
  rtn += "]";
  return rtn;
}

}  // namespace QueryRenderer
