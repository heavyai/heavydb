/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/ForeignStorage/TextFileBufferParser.h"
#include "Shared/clean_boost_regex.hpp"

namespace foreign_storage {
class RegexFileBufferParser : public TextFileBufferParser {
 public:
  RegexFileBufferParser(const ForeignTable* foreign_table);

  ParseBufferResult parseBuffer(ParseBufferRequest& request,
                                bool convert_data_blocks,
                                bool columns_are_pre_filtered = false,
                                bool skip_dict_encoding = false) const override;

  import_export::CopyParams validateAndGetCopyParams(
      const ForeignTable* foreign_table) const override;

  size_t findRowEndPosition(size_t& alloc_size,
                            std::unique_ptr<char[]>& buffer,
                            size_t& buffer_size,
                            const import_export::CopyParams& copy_params,
                            const size_t buffer_first_row_index,
                            unsigned int& num_rows_in_buffer,
                            FileReader* file_reader) const override;

  void validateFiles(const FileReader* file_reader,
                     const ForeignTable* foreign_table) const override;

  // For testing purposes only
  static void setMaxBufferResize(size_t max_buffer_resize);

  inline static const std::string LINE_REGEX_KEY = "LINE_REGEX";
  inline static const std::string LINE_START_REGEX_KEY = "LINE_START_REGEX";
  inline static const std::string HEADER_KEY = "HEADER";

 protected:
  virtual bool regexMatchColumns(const std::string& row_str,
                                 const boost::regex& line_regex,
                                 size_t logical_column_count,
                                 std::vector<std::string>& parsed_columns_str,
                                 std::vector<std::string_view>& parsed_columns_sv,
                                 const std::string& file_path) const;

  virtual bool shouldRemoveNonMatches() const;

  virtual bool shouldTruncateStringValues() const;

 private:
  static size_t getMaxBufferResize();

  inline static size_t max_buffer_resize_{
      import_export::max_import_buffer_resize_byte_size};

  // Flag added for testing purposes only
  inline static bool skip_first_line_{false};

  boost::regex line_regex_;

 protected:
  std::optional<boost::regex> line_start_regex_;
};

inline bool line_starts_with_regex(const char* buffer,
                                   size_t start,
                                   size_t end,
                                   const boost::regex& line_start_regex) {
  return boost::regex_search(std::string{buffer + start, end - start + 1},
                             line_start_regex,
                             boost::regex_constants::match_continuous);
}
}  // namespace foreign_storage
