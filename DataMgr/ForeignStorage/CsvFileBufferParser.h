/*
 * Copyright 2020 OmniSci, Inc.
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

#include "DataMgr/ForeignStorage/TextFileBufferParser.h"

namespace foreign_storage {
class CsvFileBufferParser : public TextFileBufferParser {
 public:
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
                            foreign_storage::FileReader* file_reader) const override;

  void validateExpectedColumnCount(const std::string& row,
                                   const import_export::CopyParams& copy_params,
                                   size_t num_cols,
                                   int point_cols,
                                   const std::string& file_name) const;
};
}  // namespace foreign_storage
