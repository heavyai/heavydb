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

#include "DataMgr/ForeignStorage/FileReader.h"
#include "DataMgr/ForeignStorage/ForeignTableSchema.h"

#include "ImportExport/Importer.h"

namespace foreign_storage {
struct ParseBufferRequest {
  ParseBufferRequest(const ParseBufferRequest& request) = delete;
  ParseBufferRequest(ParseBufferRequest&& request) = default;
  ParseBufferRequest(size_t buffer_size,
                     const import_export::CopyParams& copy_params,
                     int db_id,
                     const ForeignTable* foreign_table,
                     const std::set<int> column_filter_set,
                     const std::string& full_path);

  inline std::shared_ptr<Catalog_Namespace::Catalog> getCatalog() const {
    auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
    CHECK(catalog);
    return catalog;
  }

  inline std::list<const ColumnDescriptor*> getColumns() const {
    return foreign_table_schema->getLogicalAndPhysicalColumns();
  }

  inline int32_t getTableId() const {
    return foreign_table_schema->getForeignTable()->tableId;
  }

  inline std::string getTableName() const {
    return foreign_table_schema->getForeignTable()->tableName;
  }

  inline size_t getMaxFragRows() const {
    return foreign_table_schema->getForeignTable()->maxFragRows;
  }

  inline std::string getFilePath() const { return full_path; }

  // These must be initialized at construction (before parsing).
  std::unique_ptr<char[]> buffer;
  size_t buffer_size;
  size_t buffer_alloc_size;
  const import_export::CopyParams copy_params;
  const int db_id;
  std::unique_ptr<ForeignTableSchema> foreign_table_schema;
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;

  // These are set during parsing.
  size_t buffer_row_count;
  size_t begin_pos;
  size_t end_pos;
  size_t first_row_index;
  size_t file_offset;
  size_t process_row_count;
  std::string full_path;
};

struct ParseBufferResult {
  std::map<int, DataBlockPtr> column_id_to_data_blocks_map;
  size_t row_count;
  std::vector<size_t> row_offsets;
};

class TextFileBufferParser {
 public:
  /**
   * Parses a given file buffer and returns data blocks for each column in the
   * file along with metadata related to rows and row offsets within the buffer.
   *  @param convert_data_blocks      - convert import buffers to data blocks
   *  @param columns_are_pre_filtered - file buffer passed into parse_buffer only has the
   * necessary columns that are being requested, not all columns.
   */
  virtual ParseBufferResult parseBuffer(ParseBufferRequest& request,
                                        bool convert_data_blocks,
                                        bool columns_are_pre_filtered = false) const = 0;
  /**
   * Validates foreign table parse options and returns a CopyParams object upon
   * successful validation. An exception is thrown if validation fails.
   */
  virtual import_export::CopyParams validateAndGetCopyParams(
      const ForeignTable* foreign_table) const = 0;

  /**
   * Finds and returns the offset of the end of the last row in the given buffer.
   * If the buffer does not contain at least one row, the buffer is extended with
   * more content from the file until a row is read. An exception is thrown if
   * the buffer is extended to a maximum threshold and at least one row has still
   * not been read.
   */
  virtual size_t findRowEndPosition(size_t& alloc_size,
                                    std::unique_ptr<char[]>& buffer,
                                    size_t& buffer_size,
                                    const import_export::CopyParams& copy_params,
                                    const size_t buffer_first_row_index,
                                    unsigned int& num_rows_in_buffer,
                                    foreign_storage::FileReader* file_reader) const = 0;
};
}  // namespace foreign_storage