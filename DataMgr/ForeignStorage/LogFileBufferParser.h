/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include "DataMgr/ForeignStorage/RegexFileBufferParser.h"

namespace foreign_storage {
class LogFileBufferParser : public RegexFileBufferParser {
 public:
  LogFileBufferParser(const ForeignTable* foreign_table, int32_t db_id);

 protected:
  bool regexMatchColumns(const std::string& row_str,
                         const boost::regex& line_regex,
                         size_t logical_column_count,
                         std::vector<std::string>& parsed_columns_str,
                         std::vector<std::string_view>& parsed_columns_sv,
                         const std::string& file_path) const override;

  bool shouldRemoveNonMatches() const override;

  bool shouldTruncateStringValues() const override;

 private:
  const ForeignTable* foreign_table_;
  const int32_t db_id_;
};
}  // namespace foreign_storage
