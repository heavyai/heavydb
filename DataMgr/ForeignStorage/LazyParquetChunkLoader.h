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

#include <parquet/schema.h>

#include "DataMgr/Chunk/Chunk.h"
#include "ImportExport/Importer.h"
#include "Interval.h"

namespace foreign_storage {
/**
 * A lazy parquet to chunk loader
 */
class LazyParquetChunkLoader {
 public:
  // The number of elements in a batch that are read from the Parquet file;
  // this number is subject to change with performance tuning.
  // Most filesystems use a default block size of 4096 bytes.
  const static int batch_reader_num_elements = 4096;

  LazyParquetChunkLoader(const std::string& file_name);

  /**
   * Load a number of row groups of a column in a parquet file into a chunk
   *
   * @param row_group_interval - an inclusive interval [start,end] that specifies row
   * groups to load
   * @param parquet_column_index - the logical column index in the parquet file (and
   * omnisci db) of column to load
   * @param chunk - the chunk to load
   *
   * This function expects the chunk buffer to have enough space allocated to hold the
   * contents of the loaded data.
   */
  void loadChunk(const Interval<RowGroupType>& row_group_interval,
                 const int parquet_column_index,
                 Chunk_NS::Chunk& chunk);

  /**
   * Determine if a Parquet to OmniSci column mapping is supported.
   *
   * @param omnisci_column - the column descriptor of the OmniSci column
   * @param parquet_column - the column descriptor of the Parquet column
   *
   * @return true if the column mapping is supported by LazyParquetChunkLoader, false
   * otherwise
   */
  static bool isColumnMappingSupported(const ColumnDescriptor* omnisci_column,
                                       const parquet::ColumnDescriptor* parquet_column);

 private:
  std::string file_name_;
  std::vector<int16_t> def_levels_;
  std::vector<int16_t> rep_levels_;
};

}  // namespace foreign_storage
