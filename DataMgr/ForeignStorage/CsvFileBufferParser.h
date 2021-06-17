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

#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "DataMgr/ForeignStorage/ForeignTableSchema.h"
#include "Geospatial/Types.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "Shared/misc.h"

namespace foreign_storage {
namespace csv_file_buffer_parser {
static constexpr bool PROMOTE_POLYGON_TO_MULTIPOLYGON = true;

std::map<int, DataBlockPtr> convert_import_buffers_to_data_blocks(
    const std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers);

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
    // MAT are we really doing any good wrapping this?
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

inline bool skip_column_import(ParseBufferRequest& request, int column_idx) {
  return request.import_buffers[column_idx] == nullptr;
}

void parse_and_validate_expected_column_count(
    const std::string& row,
    const import_export::CopyParams& copy_params,
    size_t num_cols,
    int point_cols,
    const std::string& file_name);

/**
 * Parses a given CSV file buffer and returns data blocks for each column in the
 * file along with metadata related to rows and row offsets within the buffer.
 *  @param convert_data_blocks      - convert import buffers to data blocks
 *  @param columns_are_pre_filtered -  CSV buffer passed into parse_buffer only has the
 * necessary columns that are being requested, not all columns.
 */
ParseBufferResult parse_buffer(ParseBufferRequest& request,
                               bool convert_data_blocks,
                               bool columns_are_pre_filtered = false);

}  // namespace csv_file_buffer_parser
}  // namespace foreign_storage
