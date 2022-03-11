/*
 * Copyright 2022 OmniSci, Inc.
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

/*
 * @file MetadataColumn.cpp
 * @author Simon Eves <simon.eves@omnisci.com>
 * @brief Metadata Column info struct and parser
 */

#include "ImportExport/MetadataColumn.h"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include "ImportExport/ExpressionParser.h"
#include "Shared/DateTimeParser.h"
#include "Shared/StringTransform.h"
#include "Shared/clean_boost_regex.hpp"

namespace import_export {

MetadataColumnInfos parse_add_metadata_columns(const std::string& add_metadata_columns,
                                               const std::string& file_path) {
  //
  // each string is "column_name,column_type,expression"
  //
  // column_type can be:
  //   tinyint
  //   smallint
  //   int
  //   bigint
  //   float
  //   double
  //   date
  //   time
  //   timestamp
  //   text
  //
  // expression can be in terms of:
  //   filename
  //   filedir
  //   filepath
  //   etc.
  //

  // anything to do?
  if (add_metadata_columns.length() == 0u) {
    return {};
  }

  // split by ";"
  // @TODO(se) is this safe?
  // probably won't appear in a file name/path or a date/time string
  std::vector<std::string> add_metadata_column_strings;
  boost::split(add_metadata_column_strings, add_metadata_columns, boost::is_any_of(";"));
  if (add_metadata_column_strings.size() == 0u) {
    return {};
  }

  ExpressionParser parser;

  // known string constants
  auto const fn = boost::filesystem::path(file_path).filename().string();
  auto const fd = boost::filesystem::path(file_path).parent_path().string();
  auto const fp = file_path;
  parser.setStringConstant("filename", fn);
  parser.setStringConstant("filedir", fd);
  parser.setStringConstant("filepath", fp);

  MetadataColumnInfos metadata_column_infos;

  // for each requested column...
  for (auto const& add_metadata_column_string : add_metadata_column_strings) {
    // strip
    auto const add_metadata_column = strip(add_metadata_column_string);

    // tokenize and extract
    std::vector<std::string> tokens;
    boost::split(tokens, add_metadata_column, boost::is_any_of(","));
    if (tokens.size() < 3u) {
      throw std::runtime_error("Invalid metadata column info '" + add_metadata_column +
                               "' (must be of the form 'name,type,expression')");
    }
    auto token_itr = tokens.begin();
    auto const column_name = strip(*token_itr++);
    auto const data_type = strip(to_lower(*token_itr++));
    tokens.erase(tokens.begin(), token_itr);
    auto const expression = strip(boost::join(tokens, ","));

    // get column type
    SQLTypes sql_type{kNULLT};
    double range_min{0.0}, range_max{0.0};
    if (data_type == "tinyint") {
      sql_type = kTINYINT;
      range_min = static_cast<double>(std::numeric_limits<int8_t>::min());
      range_max = static_cast<double>(std::numeric_limits<int8_t>::max());
    } else if (data_type == "smallint") {
      sql_type = kSMALLINT;
      range_min = static_cast<double>(std::numeric_limits<int16_t>::min());
      range_max = static_cast<double>(std::numeric_limits<int16_t>::max());
    } else if (data_type == "int") {
      sql_type = kINT;
      range_min = static_cast<double>(std::numeric_limits<int32_t>::min());
      range_max = static_cast<double>(std::numeric_limits<int32_t>::max());
    } else if (data_type == "bigint") {
      sql_type = kBIGINT;
      range_min = static_cast<double>(std::numeric_limits<int64_t>::min());
      range_max = static_cast<double>(std::numeric_limits<int64_t>::max());
    } else if (data_type == "float") {
      sql_type = kFLOAT;
      range_min = static_cast<double>(std::numeric_limits<float>::min());
      range_max = static_cast<double>(std::numeric_limits<float>::max());
    } else if (data_type == "double") {
      sql_type = kDOUBLE;
      range_min = static_cast<double>(std::numeric_limits<double>::min());
      range_max = static_cast<double>(std::numeric_limits<double>::max());
    } else if (data_type == "date") {
      sql_type = kDATE;
    } else if (data_type == "time") {
      sql_type = kTIME;
    } else if (data_type == "timestamp") {
      sql_type = kTIMESTAMP;
    } else if (data_type == "text") {
      sql_type = kTEXT;
    } else {
      throw std::runtime_error("Invalid metadata column data type '" + data_type +
                               "' for column '" + column_name + "'");
    }

    // set expression with force cast back to string
    parser.setExpression("str(" + expression + ")");

    // evaluate
    auto value = parser.evalAsString();

    // validate date/time/timestamp value now
    // @TODO(se) do we need to provide for non-zero dimension?
    try {
      if (sql_type == kDATE) {
        dateTimeParse<kDATE>(value, 0);
      } else if (sql_type == kTIME) {
        dateTimeParse<kTIME>(value, 0);
      } else if (sql_type == kTIMESTAMP) {
        dateTimeParse<kTIMESTAMP>(value, 0);
      }
    } catch (std::runtime_error& e) {
      throw std::runtime_error("Invalid metadata column " + to_string(sql_type) +
                               " value '" + value + "' for column '" + column_name + "'");
    }

    // validate int/float/double
    try {
      if (IS_INTEGER(sql_type) || sql_type == kFLOAT || sql_type == kDOUBLE) {
        size_t num_chars{0u};
        auto const v = static_cast<double>(std::stod(value, &num_chars));
        if (v < range_min || v > range_max) {
          throw std::out_of_range(to_string(sql_type));
        }
        if (num_chars == 0u) {
          throw std::invalid_argument("empty value");
        }
      }
    } catch (std::invalid_argument& e) {
      throw std::runtime_error("Invalid metadata column " + to_string(sql_type) +
                               " value '" + value + "' for column '" + column_name +
                               "' (" + e.what() + ")");
    } catch (std::out_of_range& e) {
      throw std::runtime_error("Out-of-range metadata column " + to_string(sql_type) +
                               " value '" + value + "' for column '" + column_name +
                               "' (" + e.what() + ")");
    }

    // build column descriptor
    ColumnDescriptor cd;
    cd.columnName = cd.sourceName = column_name;
    cd.columnType.set_type(sql_type);
    cd.columnType.set_fixed_size();
    if (sql_type == kTEXT) {
      cd.columnType.set_compression(kENCODING_DICT);
      cd.columnType.set_comp_param(0);
    }

    // add to result
    metadata_column_infos.push_back({std::move(cd), std::move(value)});
  }

  // done
  return metadata_column_infos;
}

}  // namespace import_export
