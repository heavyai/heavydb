/*
 * Copyright 2021 OmniSci, Inc.
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

#include "RegexParserDataWrapper.h"

#include <boost/regex.hpp>

#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"

namespace foreign_storage {
RegexParserDataWrapper::RegexParserDataWrapper()
    : AbstractTextFileDataWrapper(), regex_file_buffer_parser_{nullptr} {}

RegexParserDataWrapper::RegexParserDataWrapper(const int db_id,
                                               const ForeignTable* foreign_table)
    : AbstractTextFileDataWrapper(db_id, foreign_table)
    , regex_file_buffer_parser_{foreign_table} {}

namespace {
void validate_regex(const std::string& regex, const std::string& option_name) {
  try {
    boost::regex{regex};
  } catch (const std::exception& e) {
    throw ForeignStorageException{"Error parsing " + option_name + " \"" + regex +
                                  "\": " + e.what()};
  }
}
}  // namespace

void RegexParserDataWrapper::validateTableOptions(
    const ForeignTable* foreign_table) const {
  AbstractTextFileDataWrapper::validateTableOptions(foreign_table);
  auto line_regex_it = foreign_table->options.find(RegexFileBufferParser::LINE_REGEX_KEY);
  if (line_regex_it == foreign_table->options.end() || line_regex_it->second.empty()) {
    throw ForeignStorageException{
        "Foreign table options must contain a non-empty value for \"" +
        RegexFileBufferParser::LINE_REGEX_KEY + "\"."};
  }
  validate_regex(line_regex_it->second, RegexFileBufferParser::LINE_REGEX_KEY);

  auto line_start_regex_it =
      foreign_table->options.find(RegexFileBufferParser::LINE_START_REGEX_KEY);
  if (line_start_regex_it != foreign_table->options.end()) {
    if (line_start_regex_it->second.empty()) {
      throw ForeignStorageException{"Foreign table option \"" +
                                    RegexFileBufferParser::LINE_START_REGEX_KEY +
                                    "\", when set, must contain a non-empty value."};
    }
    validate_regex(line_start_regex_it->second,
                   RegexFileBufferParser::LINE_START_REGEX_KEY);
  }
}

const std::set<std::string_view>& RegexParserDataWrapper::getSupportedTableOptions()
    const {
  static const auto supported_table_options = getAllRegexTableOptions();
  return supported_table_options;
}

std::set<std::string_view> RegexParserDataWrapper::getAllRegexTableOptions() const {
  std::set<std::string_view> supported_table_options(
      AbstractFileStorageDataWrapper::getSupportedTableOptions().begin(),
      AbstractFileStorageDataWrapper::getSupportedTableOptions().end());
  supported_table_options.insert(regex_table_options_.begin(),
                                 regex_table_options_.end());
  return supported_table_options;
}

const TextFileBufferParser& RegexParserDataWrapper::getFileBufferParser() const {
  return regex_file_buffer_parser_;
}

const std::set<std::string_view> RegexParserDataWrapper::regex_table_options_{
    RegexFileBufferParser::LINE_REGEX_KEY,
    RegexFileBufferParser::LINE_START_REGEX_KEY,
    RegexFileBufferParser::BUFFER_SIZE_KEY};
}  // namespace foreign_storage
