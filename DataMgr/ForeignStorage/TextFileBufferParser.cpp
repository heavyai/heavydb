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

#include "DataMgr/ForeignStorage/TextFileBufferParser.h"

namespace foreign_storage {
ParseBufferRequest::ParseBufferRequest(size_t buffer_size,
                                       const import_export::CopyParams& copy_params,
                                       int db_id,
                                       const ForeignTable* foreign_table,
                                       std::set<int> column_filter_set,
                                       const std::string& full_path)
    : buffer_size(buffer_size)
    , buffer_alloc_size(buffer_size)
    , copy_params(copy_params)
    , db_id(db_id)
    , foreign_table_schema(std::make_unique<ForeignTableSchema>(db_id, foreign_table))
    , full_path(full_path) {
  if (buffer_size > 0) {
    buffer = std::make_unique<char[]>(buffer_size);
  }
  // initialize import buffers from columns.
  for (const auto column : getColumns()) {
    if (column_filter_set.find(column->columnId) == column_filter_set.end()) {
      import_buffers.emplace_back(nullptr);
    } else {
      StringDictionary* string_dictionary = nullptr;
      if (column->columnType.is_dict_encoded_string() ||
          (column->columnType.is_array() && IS_STRING(column->columnType.get_subtype()) &&
           column->columnType.get_compression() == kENCODING_DICT)) {
        auto dict_descriptor = getCatalog()->getMetadataForDictUnlocked(
            column->columnType.get_comp_param(), true);
        string_dictionary = dict_descriptor->stringDict.get();
      }
      import_buffers.emplace_back(
          std::make_unique<import_export::TypedImportBuffer>(column, string_dictionary));
    }
  }
}
}  // namespace foreign_storage
