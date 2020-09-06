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

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/types.h>

namespace foreign_storage {

void open_parquet_table(const std::string& file_path,
                        std::unique_ptr<parquet::arrow::FileReader>& reader);

std::pair<int, int> get_parquet_table_size(
    const std::unique_ptr<parquet::arrow::FileReader>& reader);

const parquet::ColumnDescriptor* get_column_descriptor(
    std::unique_ptr<parquet::arrow::FileReader>& reader,
    const int logical_column_index);

size_t get_physical_type_byte_size(std::unique_ptr<parquet::arrow::FileReader>& reader,
                                   const int logical_column_index);

}  // namespace foreign_storage
