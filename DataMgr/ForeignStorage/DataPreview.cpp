/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DataPreview.h"

#include <regex>

namespace foreign_storage {
std::optional<SQLTypes> detect_geo_type(const SampleRows& sample_rows,
                                        size_t column_index) {
  std::optional<SQLTypes> tentative_geo_type{};
  for (const auto& row : sample_rows) {
    static std::regex geo_regex{
        "\\s*(POINT|MULTIPOINT|LINESTRING|MULTILINESTRING|POLYGON|MULTIPOLYGON)\\s*\\(.+"
        "\\)\\s*"};
    std::smatch match;
    CHECK_LT(column_index, row.size());
    if (std::regex_match(row[column_index], match, geo_regex)) {
      CHECK_EQ(match.size(), static_cast<size_t>(2));
      SQLTypes geo_type{kNULLT};
      const auto& geo_type_str = match[1];
      if (geo_type_str == "POINT") {
        geo_type = kPOINT;
      } else if (geo_type_str == "MULTIPOINT") {
        geo_type = kMULTIPOINT;
      } else if (geo_type_str == "LINESTRING") {
        geo_type = kLINESTRING;
      } else if (geo_type_str == "MULTILINESTRING") {
        geo_type = kMULTILINESTRING;
      } else if (geo_type_str == "POLYGON") {
        geo_type = kPOLYGON;
      } else if (geo_type_str == "MULTIPOLYGON") {
        geo_type = kMULTIPOLYGON;
      } else {
        UNREACHABLE() << "Unexpected geo type match: " << geo_type_str;
      }
      if (tentative_geo_type.has_value()) {
        if (tentative_geo_type.value() != geo_type) {
          return {};  // geo type does not match between rows, can not be imported
        }
      } else {
        tentative_geo_type = geo_type;
      }
    }
  }
  return tentative_geo_type;
}
}  // namespace foreign_storage
