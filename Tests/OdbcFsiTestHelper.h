/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "Logger/Logger.h"

namespace {
using ColumnPair = std::pair<std::string, std::string>;

inline std::string quoted_identifier(const std::string& dsn,
                                     const std::string& identifier) {
  if (dsn == "bigquery" || dsn == "hive") {  // BigQuery & Hive uses a different set of
                                             // quotes for quoted identifiers
    return "`" + identifier + "`";
  }
  return "\"" + identifier + "\"";
}

inline bool is_quoted_identifier(const std::string& identifier) {
  if ((identifier[0] == '"' && identifier[identifier.length() - 1] == '"') ||
      (identifier[0] == '`' && identifier[identifier.length() - 1] == '`')) {
    return true;
  }
  return false;
}

inline std::string strip_quoted_identifier(const std::string& identifier) {
  CHECK_GE(identifier.length(), 3UL);
  auto new_name =
      identifier.substr(1, identifier.length() - 2);  // remove quotes for comparison
  return new_name;
}

inline std::string parse_next_field(const std::string& record, size_t& pos) {
  while (pos < record.size() && record[pos] == ' ') {
    pos++;
  }
  if (pos == record.size()) {
    return "NULL";
  }

  // determine wrapper for the next data field
  bool is_wrapped = false;
  char field_wrapper;
  if (record[pos] == '\'' || record[pos] == '"') {
    is_wrapped = true;
    field_wrapper = record[pos];
    pos++;
  }

  // determine the length of the next data field
  size_t field_length, end_pos;
  if (is_wrapped) {
    end_pos = record.find(field_wrapper, pos);
    CHECK(end_pos < record.size()) << "Expected a terminating " << field_wrapper
                                   << " in the record '" << record << "'.";
  } else {
    end_pos = record.find(',', pos);
  }
  field_length = end_pos - pos;

  auto field_value = record.substr(pos, field_length);
  pos = end_pos + (is_wrapped ? 2 : 1);

  // determine if the data is nulled
  if ((!is_wrapped && field_value.empty()) || field_value == "NULL" ||
      field_value == "\\N") {
    field_value = "NULL";
  }

  return field_value;
}

inline void apply_mods_to_insert_record(std::string& record,
                                        const std::vector<ColumnPair>& column_pairs,
                                        const std::string& data_wrapper_type) {
  const std::vector<std::string> data_wrapper_requires_all_wrapped_fields = {
      "sqlite", "postgres", "redshift", "snowflake", "hive"};
  const std::vector<std::string> column_type_requires_wrapped_field = {
      "timestamp", "string", "time", "date", "char", "string(12600)"};
  const std::vector<std::string> column_type_requires_upper_case_field = {"boolean"};
  const std::vector<std::string> is_geography_column_type = {"point",
                                                             "multipoint",
                                                             "linestring",
                                                             "multilinestring",
                                                             "polygon",
                                                             "multipolygon",
                                                             "geography",
                                                             "geometry"};
  const std::map<std::string, std::string> data_wrapper_to_geography_function = {
      {"bigquery", "ST_GEOGFROMTEXT"}, {"redshift", "ST_GeomFromText"}};
  const std::vector<std::string> is_bignumeric_column_type = {
      "bignumeric(38)", "bigdecimal(38)", "bignumeric (38)", "bigdecimal (38)"};
  const std::map<std::string, std::string> data_wrapper_to_bignumeric_literal_function = {
      {"bigquery", "BIGNUMERIC"}};

  std::stringstream ss;
  size_t pos = 0;
  for (size_t i = 0; i < column_pairs.size(); i++) {
    auto field_value = parse_next_field(record, pos);
    auto col_type = to_lower(column_pairs[i].second);

    if (shared::contains(column_type_requires_upper_case_field, col_type)) {
      field_value = to_upper(field_value);
    }
    auto do_not_wrap_field_in_quotes =
        field_value == "NULL" || field_value == "TRUE" || field_value == "FALSE";

    if (!do_not_wrap_field_in_quotes &&
        shared::contains(is_geography_column_type, col_type)) {
      if (data_wrapper_to_geography_function.find(data_wrapper_type) !=
          data_wrapper_to_geography_function.end()) {
        ss << data_wrapper_to_geography_function.at(data_wrapper_type) << "('"
           << field_value << "')";
      } else {
        ss << "'" << field_value << "'";
      }
    } else if (!do_not_wrap_field_in_quotes &&
               shared::contains(is_bignumeric_column_type, col_type)) {
      if (data_wrapper_to_bignumeric_literal_function.find(data_wrapper_type) !=
          data_wrapper_to_bignumeric_literal_function.end()) {
        ss << data_wrapper_to_bignumeric_literal_function.at(data_wrapper_type) << " '"
           << field_value << "'";
      } else {
        ss << field_value;
      }

    } else if (!do_not_wrap_field_in_quotes &&
               (shared::contains(data_wrapper_requires_all_wrapped_fields,
                                 data_wrapper_type) ||
                shared::contains(column_type_requires_wrapped_field, col_type))) {
      ss << "'" << field_value << "'";
    } else {
      ss << field_value;
    }

    if (i < column_pairs.size() - 1) {
      ss << ",";
    }
  }
  record = ss.str();
}

inline bool does_wrapper_support_geo_type(const std::string& wrapper_type) {
  if (wrapper_type == "sqlite" || wrapper_type == "hive") {
    return false;
  }
  return true;
}

inline boost::regex make_regex(const std::string& pattern) {
  std::string whitespace_wrapper = "\\s*" + pattern + "\\s*";
  return boost::regex(whitespace_wrapper, boost::regex::icase);
}

inline const std::map<std::string, std::map<boost::regex, std::string>>
    k_rdms_column_type_substitutes = {
        {"sqlite",
         {{make_regex("TEXT.*"), "text"},
          {make_regex("DECIMAL\\s*\\(\\d+,\\s*\\d+\\)\\s*(\\[\\d*\\])?"), "double"},
          {make_regex("FLOAT"), "double"}}},
        {"postgres",
         {{make_regex("TEXT.*"), "text"},
          {make_regex("FLOAT"), "real"},
          {make_regex("DOUBLE"), "double precision"},
          {make_regex("TINYINT"), "smallint"},
          {make_regex("TIME\\b"), "time(0)"},
          {make_regex("TIMESTAMP"), "timestamp(0)"},
          {make_regex("TIMESTAMP\\s*\\(6\\)"), "timestamp"},
          {make_regex("(MULTI)?POINT"), "geometry"},
          {make_regex("(MULTI)?LINESTRING"), "geometry"},
          {make_regex("(MULTI)?POLYGON"), "geometry"}}},
        {"redshift",
         {{make_regex("TEXT.*"), "text"},
          {make_regex("FLOAT"), "real"},
          {make_regex("DOUBLE"), "double precision"},
          {make_regex("TIMESTAMP\\s*\\(\\d+\\)"), "timestamp"},
          {make_regex("TINYINT"), "smallint"},
          {make_regex("(MULTI)?POINT"), "geometry"},
          {make_regex("(MULTI)?LINESTRING"), "geometry"},
          {make_regex("(MULTI)?POLYGON"), "geometry"}}},
        {"snowflake",
         {{make_regex("TEXT.*"), "text"},
          {make_regex("TIME\\b"), "time(0)"},
          {make_regex("(MULTI)?POINT"), "geography"},
          {make_regex("(MULTI)?LINESTRING"), "geography"},
          {make_regex("(MULTI)?POLYGON"), "geography"}}},
        {"bigquery",
         {{make_regex("TEXT.*"), "string"},
          {make_regex("TIMESTAMP\\s*\\(\\d+\\)"), "timestamp"},
          {make_regex("TIME\\s*\\(\\d+\\)"), "time"},
          {make_regex("FLOAT"), "float64"},
          {make_regex("DOUBLE"), "float64"},
          {make_regex("(MULTI)?POINT"), "geography"},
          {make_regex("(MULTI)?LINESTRING"), "geography"},
          {make_regex("(MULTI)?POLYGON"), "geography"}}},
        {"hive",
         {{make_regex("TEXT.*"), "string"},
          {make_regex("CHAR"), "CHAR(1)"},
          {make_regex("TIMESTAMP\\s*\\(\\d+\\)"), "timestamp"},
          {make_regex("TIME\\b"), "string"},
          {make_regex("(MULTI)?POINT"), "string"},
          {make_regex("(MULTI)?LINESTRING"), "string"},
          {make_regex("(MULTI)?POLYGON"), "string"}}}};

inline bool is_odbc(const std::string& data_wrapper_type) {
  const std::vector<std::string> odbc_wrappers{
      "sqlite", "postgres", "redshift", "snowflake", "bigquery", "hive"};
  return std::find(odbc_wrappers.begin(), odbc_wrappers.end(), data_wrapper_type) !=
         odbc_wrappers.end();
}
}  // namespace
