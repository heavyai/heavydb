/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "Shared/sqltypes.h"

namespace foreign_storage {
using SampleRows = std::vector<std::vector<std::string>>;
struct DataPreview {
  std::vector<std::string> column_names;
  std::vector<SQLTypeInfo> column_types;
  SampleRows sample_rows;
  size_t num_rejected_rows;

  bool operator==(const DataPreview& o) const {
    return column_names == o.column_names && column_types == o.column_types &&
           sample_rows == o.sample_rows && num_rejected_rows == o.num_rejected_rows;
  }
};

inline std::ostream& operator<<(std::ostream& out, const DataPreview& data) {
  out << "column_names: {";
  std::string separator;
  for (const auto& i : data.column_names) {
    out << separator << i;
    separator = ", ";
  }
  out << "}, column_types: {";
  separator = "";
  for (const auto& i : data.column_types) {
    out << separator << i;
    separator = ", ";
  }
  out << "}, sample_rows: {";
  separator = "";
  for (const auto& i : data.sample_rows) {
    out << separator << "{";
    std::string inner_separator;
    for (const auto& j : i) {
      out << inner_separator << j;
      inner_separator = ", ";
    }
    out << "}";
    separator = ", ";
  }
  out << "} num_rejected_rows: " << data.num_rejected_rows;
  return out;
}

std::optional<SQLTypes> detect_geo_type(const SampleRows& sample_rows,
                                        size_t column_index);

}  // namespace foreign_storage
