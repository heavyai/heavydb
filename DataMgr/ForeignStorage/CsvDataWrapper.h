/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "AbstractTextFileDataWrapper.h"
#include "DataMgr/ForeignStorage/CsvFileBufferParser.h"

namespace foreign_storage {
class CsvDataWrapper : public AbstractTextFileDataWrapper {
 public:
  CsvDataWrapper();

  CsvDataWrapper(const int db_id, const ForeignTable* foreign_table);

  CsvDataWrapper(const int db_id,
                 const ForeignTable* foreign_table,
                 const UserMapping* user_mapping,
                 const bool disable_cache = false);

  void validateTableOptions(const ForeignTable* foreign_table) const override;

  const std::set<std::string_view>& getSupportedTableOptions() const override;

  static bool validateAndGetIsS3Select(const ForeignTable* foreign_table);

 protected:
  const TextFileBufferParser& getFileBufferParser() const override;

 private:
  std::set<std::string_view> getAllCsvTableOptions() const;
  static const std::set<std::string_view> csv_table_options_;
  static const CsvFileBufferParser csv_file_buffer_parser_;
};
}  // namespace foreign_storage
