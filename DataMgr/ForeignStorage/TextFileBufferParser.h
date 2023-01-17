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
                     const std::string& full_path,
                     const bool track_rejected_rows = false);

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

  // This parameter controls the behaviour of error handling in the data wrapper
  const bool track_rejected_rows;

  // This tracks the number of rows processed, is necessary to identify requests that are
  // not completed
  size_t processed_row_count;
};

struct ParseBufferResult {
  std::map<int, DataBlockPtr> column_id_to_data_blocks_map;
  size_t row_count;
  std::vector<size_t> row_offsets;
  std::set<size_t> rejected_rows;
};

class TextFileBufferParser {
 public:
  /**
   * Parses a given file buffer and returns data blocks for each column in the
   * file along with metadata related to rows and row offsets within the buffer.
   *  @param convert_data_blocks      - convert import buffers to data blocks
   *  @param columns_are_pre_filtered - file buffer passed into parse_buffer only has the
   * necessary columns that are being requested, not all columns.
   *  @param skip_dict_encoding       - skip dictionary encoding for encoded
   *  strings; the encoding will be required to happen later in processing
   */
  virtual ParseBufferResult parseBuffer(ParseBufferRequest& request,
                                        bool convert_data_blocks,
                                        bool columns_are_pre_filtered = false,
                                        bool skip_dict_encoding = false) const = 0;
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
                                    FileReader* file_reader) const = 0;

  /**
   * Performs basic validation of files to be parsed.
   */
  virtual void validateFiles(const FileReader* file_reader,
                             const ForeignTable* foreign_table) const = 0;

  static std::map<int, DataBlockPtr> convertImportBuffersToDataBlocks(
      const std::vector<std::unique_ptr<import_export::TypedImportBuffer>>&
          import_buffers,
      const bool skip_dict_encoding = false);

  static bool isCoordinateScalar(const std::string_view datum);

  static void processGeoColumn(
      std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
      size_t& col_idx,
      const import_export::CopyParams& copy_params,
      std::list<const ColumnDescriptor*>::iterator& cd_it,
      std::vector<std::string_view>& row,
      size_t& import_idx,
      bool is_null,
      size_t first_row_index,
      size_t row_index_plus_one,
      std::shared_ptr<Catalog_Namespace::Catalog> catalog);

  /**
   * Fill the current row of the `request` with invalid (null) data as row will
   * be marked as rejected
   */
  static void fillRejectedRowWithInvalidData(
      const std::list<const ColumnDescriptor*>& columns,
      std::list<const ColumnDescriptor*>::iterator& cd_it,
      const size_t col_idx,
      ParseBufferRequest& request);

  static bool isNullDatum(const std::string_view datum,
                          const ColumnDescriptor* column,
                          const std::string& null_indicator);

  inline static const std::string BUFFER_SIZE_KEY = "BUFFER_SIZE";

 private:
  static void processInvalidGeoColumn(
      std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
      size_t& col_idx,
      const import_export::CopyParams& copy_params,
      const ColumnDescriptor* cd,
      std::shared_ptr<Catalog_Namespace::Catalog> catalog);
};
}  // namespace foreign_storage
