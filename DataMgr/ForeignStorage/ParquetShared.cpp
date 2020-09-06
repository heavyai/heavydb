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

void open_parquet_table(const std::string& file_path,
                        std::unique_ptr<parquet::arrow::FileReader>& reader) {
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(file_path));
  PARQUET_THROW_NOT_OK(OpenFile(infile, arrow::default_memory_pool(), &reader));
}

std::pair<int, int> get_parquet_table_size(
    const std::unique_ptr<parquet::arrow::FileReader>& reader) {
  auto file_metadata = reader->parquet_reader()->metadata();
  const auto num_row_groups = file_metadata->num_row_groups();
  const auto num_columns = file_metadata->num_columns();
  return std::make_pair(num_row_groups, num_columns);
}

const parquet::ColumnDescriptor* get_column_descriptor(
    std::unique_ptr<parquet::arrow::FileReader>& reader,
    const int logical_column_index) {
  return reader->parquet_reader()->metadata()->schema()->Column(logical_column_index);
}

parquet::Type::type get_physical_type(std::unique_ptr<parquet::arrow::FileReader>& reader,
                                      const int logical_column_index) {
  return reader->parquet_reader()
      ->metadata()
      ->schema()
      ->Column(logical_column_index)
      ->physical_type();
}

size_t get_physical_type_byte_size(std::unique_ptr<parquet::arrow::FileReader>& reader,
                                   const int logical_column_index) {
  return parquet::GetTypeByteSize(get_physical_type(reader, logical_column_index));
}

}  // namespace foreign_storage
