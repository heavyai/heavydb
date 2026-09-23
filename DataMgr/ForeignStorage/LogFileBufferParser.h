/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/ForeignStorage/RegexFileBufferParser.h"

namespace foreign_storage {
class LogFileBufferParser : public RegexFileBufferParser {
 public:
  LogFileBufferParser(const ForeignTable* foreign_table, int32_t db_id);

  void optionallyRemoveBadFiles(MultiFileReader* mfr) const override;

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
