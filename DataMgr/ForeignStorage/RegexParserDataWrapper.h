/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "AbstractTextFileDataWrapper.h"
#include "DataMgr/ForeignStorage/RegexFileBufferParser.h"

namespace foreign_storage {
class RegexParserDataWrapper : public AbstractTextFileDataWrapper {
 public:
  RegexParserDataWrapper();

  RegexParserDataWrapper(const int db_id, const ForeignTable* foreign_table);

  RegexParserDataWrapper(const int db_id,
                         const ForeignTable* foreign_table,
                         const UserMapping* user_mapping,
                         const bool disable_cache = false);

  void validateTableOptions(const ForeignTable* foreign_table) const override;

  const std::set<std::string_view>& getSupportedTableOptions() const override;

 protected:
  const TextFileBufferParser& getFileBufferParser() const override;

 private:
  std::set<std::string_view> getAllRegexTableOptions() const;
  static const std::set<std::string_view> regex_table_options_;
  const RegexFileBufferParser regex_file_buffer_parser_;
};
}  // namespace foreign_storage
