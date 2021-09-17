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

#pragma once

#include <boost/regex.hpp>

#include "DataMgr/ForeignStorage/TextFileBufferParser.h"

namespace foreign_storage {
class RegexFileBufferParser : public TextFileBufferParser {
 public:
  RegexFileBufferParser(const ForeignTable* foreign_table);

  ParseBufferResult parseBuffer(ParseBufferRequest& request,
                                bool convert_data_blocks,
                                bool columns_are_pre_filtered = false) const override;

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
  static void setSkipFirstLineForTesting(bool skip);
  static void setMaxBufferResize(size_t max_buffer_resize);

  inline static const std::string LINE_REGEX_KEY = "LINE_REGEX";
  inline static const std::string LINE_START_REGEX_KEY = "LINE_START_REGEX";
  inline static const std::string BUFFER_SIZE_KEY = "BUFFER_SIZE";

 private:
  static size_t getMaxBufferResize();

  inline static size_t max_buffer_resize_{
      import_export::max_import_buffer_resize_byte_size};

  // Flag added for testing purposes only
  inline static bool skip_first_line_{false};

  boost::regex line_regex_;
  std::optional<boost::regex> line_start_regex_;
};
}  // namespace foreign_storage
