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

#include "ParquetShared.h"

#include <parquet/column_scanner.h>
#include <parquet/exception.h>
#include <parquet/platform.h>

namespace foreign_storage {

UniqueReaderPtr open_parquet_table(const std::string& file_path,
                                   std::shared_ptr<arrow::fs::FileSystem>& file_system) {
  UniqueReaderPtr reader;
  auto file_result = file_system->OpenInputFile(file_path);
  if (!file_result.ok()) {
    throw std::runtime_error{"Unable to access " + file_system->type_name() + " file: " +
                             file_path + ". " + file_result.status().message()};
  }
  auto infile = file_result.ValueOrDie();
  PARQUET_THROW_NOT_OK(OpenFile(infile, arrow::default_memory_pool(), &reader));
  return reader;
}

std::pair<int, int> get_parquet_table_size(const ReaderPtr& reader) {
  auto file_metadata = reader->parquet_reader()->metadata();
  const auto num_row_groups = file_metadata->num_row_groups();
  const auto num_columns = file_metadata->num_columns();
  return std::make_pair(num_row_groups, num_columns);
}

const parquet::ColumnDescriptor* get_column_descriptor(
    const parquet::arrow::FileReader* reader,
    const int logical_column_index) {
  return reader->parquet_reader()->metadata()->schema()->Column(logical_column_index);
}

parquet::Type::type get_physical_type(ReaderPtr& reader, const int logical_column_index) {
  return reader->parquet_reader()
      ->metadata()
      ->schema()
      ->Column(logical_column_index)
      ->physical_type();
}

void validate_equal_column_descriptor(
    const parquet::ColumnDescriptor* reference_descriptor,
    const parquet::ColumnDescriptor* new_descriptor,
    const std::string& reference_file_path,
    const std::string& new_file_path) {
  if (!reference_descriptor->Equals(*new_descriptor)) {
    throw std::runtime_error{"Parquet file \"" + new_file_path +
                             "\" has a different schema. Please ensure that all Parquet "
                             "files use the same schema. Reference Parquet file: " +
                             reference_file_path +
                             ", column name: " + reference_descriptor->name() +
                             ". New Parquet file: " + new_file_path +
                             ", column name: " + new_descriptor->name() + "."};
  }
}

std::unique_ptr<ColumnDescriptor> get_sub_type_column_descriptor(
    const ColumnDescriptor* column) {
  auto column_type = column->columnType.get_elem_type();
  if (column_type.get_size() == -1 && column_type.is_dict_encoded_string()) {
    column_type.set_size(4);  // override default size of -1
  }
  return std::make_unique<ColumnDescriptor>(
      column->tableId, column->columnId, column->columnName, column_type);
}

std::shared_ptr<parquet::Statistics> validate_and_get_column_metadata_statistics(
    const parquet::ColumnChunkMetaData* column_metadata) {
  CHECK(column_metadata->is_stats_set());
  std::shared_ptr<parquet::Statistics> stats = column_metadata->statistics();
  bool is_all_nulls = stats->null_count() == column_metadata->num_values();
  CHECK(is_all_nulls || stats->HasMinMax());
  return stats;
}

}  // namespace foreign_storage
