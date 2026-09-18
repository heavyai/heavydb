/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <string>

namespace shared {

inline std::string decode_base64(const std::string& val, bool trim_nulls) {
  // For some strings, signatature particualary it is important no to trim
  // '\0' characters
  using namespace boost::archive::iterators;
  using It = transform_width<binary_from_base64<std::string::const_iterator>, 8, 6>;

  if (!trim_nulls) {
    return std::string(It(val.begin()), It(val.end()));
  }
  return boost::algorithm::trim_right_copy_if(
      std::string(It(std::begin(val)), It(std::end(val))),
      [](char c) { return c == '\0'; });
}

inline std::string decode_base64(const std::string& val) {
  return decode_base64(val, true);
}

static inline std::string encode_base64(const std::string& val) {
  using namespace boost::archive::iterators;
  using It = base64_from_binary<transform_width<std::string::const_iterator, 6, 8>>;
  auto tmp = std::string(It(std::begin(val)), It(std::end(val)));
  return tmp.append((3 - val.size() % 3) % 3, '=');
}

std::string decode_base64_uri(const std::string& value, bool trim_nulls = true);

}  // namespace shared
