/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Utils/LoggingUtils.h"

#include <iomanip>

namespace gfx {

NullOStream null_ostream;

StreamStatFormatter::StreamStatFormatter(std::ostream& os,
                                         int name_width,
                                         int value_width,
                                         int precision)
    : os_{os}
    , name_width_{name_width}
    , value_width_{value_width}
    , precision_{precision} {}

void StreamStatFormatter::operator()(const std::string_view name,
                                     const stat_value_type value,
                                     std::string_view suffix) {
  os_ << std::left << std::setw(name_width_) << name << std::right
      << std::setw(value_width_);
  if (const auto* pvalue = std::get_if<uint32_t>(&value)) {
    os_ << *pvalue;
  } else if (const auto* pvalue = std::get_if<int32_t>(&value)) {
    os_ << *pvalue;
  } else if (const auto* pvalue = std::get_if<uint64_t>(&value)) {
    os_ << *pvalue;
  } else if (const auto* pvalue = std::get_if<int64_t>(&value)) {
    os_ << *pvalue;
  } else if (const auto* pvalue = std::get_if<float>(&value)) {
    os_ << std::fixed << std::setprecision(precision_) << *pvalue;
  } else if (const auto* pvalue = std::get_if<double>(&value)) {
    os_ << std::fixed << std::setprecision(precision_) << *pvalue;
  } else if (const auto* pvalue = std::get_if<bool>(&value)) {
    os_ << std::boolalpha << *pvalue;
  } else {
    CHECK(false) << "Invalid value type in FormatStreamItem";
  }
  os_ << suffix << "\n";
}

std::ostream& StreamStatFormatter::operator<<(const std::string& s) {
  os_ << s;
  return os_;
}

void StreamStatFormatter::memory_stat(const std::string_view name, uint64_t bytes) {
  os_ << std::left << std::setw(name_width_) << name << std::right
      << std::setw(value_width_);

  pretty_print_bytes(os_, bytes, precision_);
  os_ << "\n";
}

void pretty_print_bytes(std::ostream& os, uint64_t bytes, int precision) {
  constexpr std::array<std::string_view, 7> suffixes = {
      "B", "KB", "MB", "GB", "TB", "PB", "EB"};

  uint s = 0;  // which suffix to use
  double count = bytes;
  while (count >= 1024 && s < 7) {
    s++;
    count /= 1024;
  }
  if (s == 0) {
    os << count << " " << suffixes[0];
  } else {
    os << std::fixed << std::setprecision(precision) << count << " " << suffixes[s];
  }
}

}  // namespace gfx
