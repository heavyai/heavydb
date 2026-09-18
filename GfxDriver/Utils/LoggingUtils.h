/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <ostream>
#include <string_view>
#include <variant>

#include "Logger/Logger.h"

namespace gfx {

//
// class NullOStream
//
// ostream overload acts as a non-outputing sink for ostream operations
class NullOStream : public std::ostream {
 public:
  NullOStream() : std::ostream(nullptr) {}
  NullOStream(const NullOStream&) : std::ostream(nullptr) {}
};

template <class T>
const NullOStream& operator<<(NullOStream&& os, const T& value) {
  return os;
}

// global instance of NullOStream
extern NullOStream null_ostream;

//
// class StreamStatFormatter
//
// Formats "stats" into consistent name / value columns
class StreamStatFormatter {
 public:
  enum { kDefaultNameWidth = 25, kDefaultValueWidth = 11, kDefaultPrecision = 3 };

  explicit StreamStatFormatter(std::ostream& os,
                               int name_width = kDefaultNameWidth,
                               int value_width = kDefaultValueWidth,
                               int precision = kDefaultPrecision);
  using stat_value_type =
      std::variant<uint32_t, int32_t, uint64_t, int64_t, float, double, bool>;

  void operator()(const std::string_view name,
                  const stat_value_type value,
                  std::string_view suffix = "");

  std::ostream& operator<<(const std::string& s);

  void memory_stat(const std::string_view name, uint64_t bytes);

 private:
  std::ostream& os_;
  int name_width_;
  int value_width_;
  int precision_;
};

// Format memory output into human readable form (e.g. "23.15 Mb");
void pretty_print_bytes(std::ostream& os, uint64_t bytes, int precision = 3);

}  // namespace gfx
