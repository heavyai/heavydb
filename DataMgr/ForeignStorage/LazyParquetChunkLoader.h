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

#include <arrow/filesystem/filesystem.h>
#include <parquet/schema.h>

#include "DataMgr/Chunk/Chunk.h"
#include "DataPreview.h"
#include "ForeignTableSchema.h"
#include "Interval.h"
#include "ParquetEncoder.h"
#include "ParquetShared.h"
#include "StringDictionary/StringDictionary.h"

extern size_t g_max_import_threads;

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

  LazyParquetChunkLoader(std::shared_ptr<arrow::fs::FileSystem> file_system,
                         FileReaderMap* file_reader_cache,
                         const RenderGroupAnalyzerMap* render_group_analyzer_map,
                         const std::string& foreign_table_name);

  /**
   * Load a number of row groups of a column in a parquet file into a chunk
   *
   * @param row_group_interval - an inclusive interval [start,end] that specifies row
   * groups to load
   * @param parquet_column_index - the logical column index in the parquet file (and
   * omnisci db) of column to load
   * @param chunks - a list containing the chunks to load
   * @param string_dictionary - a string dictionary for the column corresponding to the
   * column, if applicable
   * @param rejected_row_indices - optional, if specified errors will be
   * tracked in this data structure while loading
   *
   * @return An empty list when no metadata update is applicable, otherwise a
   * list of ChunkMetadata shared pointers with which to update the
   * corresponding column chunk metadata.
   *
   * NOTE: if more than one chunk is supplied, the first chunk is required to
   * be the chunk corresponding to the logical column, while the remaining
   * chunks correspond to physical columns (in ascending order of column id.)
   * Similarly, if a metada update is expected, the list of ChunkMetadata
   * shared pointers returned will correspond directly to the list `chunks`.
   */
  std::list<std::unique_ptr<ChunkMetadata>> loadChunk(
      const std::vector<RowGroupInterval>& row_group_intervals,
      const int parquet_column_index,
      std::list<Chunk_NS::Chunk>& chunks,
      StringDictionary* string_dictionary = nullptr,
      RejectedRowIndices* rejected_row_indices = nullptr);

  /**
   * @brief Perform a metadata scan for the paths specified
   *
   * @param file_paths -  (ordered) files of the metadata scan
   * @param schema - schema of the foreign table to perform metadata scan for
   * @param do_metadata_stats_validation - validate stats in metadata of parquet files if
   * true
   *
   * @return a list of the row group metadata extracted from `file_paths`
   */
  std::list<RowGroupMetadata> metadataScan(
      const std::vector<std::string>& file_paths,
      const ForeignTableSchema& schema,
      const bool do_metadata_stats_validation = true);

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

  /**
   * @brief Load row groups of data into given chunks
   *
   * @param row_group_interval - specifies which row groups to load
   * @param chunks - map of column index to chunk which data will be loaded into
   * @param schema - schema of the foreign table to perform metadata scan for
   * @param column_dictionaries - a map of string dictionaries for columns that require it
   * @param num_threads - number of threads to utilize while reading (if applicale)
   *
   * @return [num_rows_completed,num_rows_rejected] - returns number of rows
   * loaded and rejected while loading
   *
   * Note that only logical chunks are expected because the data is read into
   * an intermediate form into the underlying buffers. This member is intended
   * to be used for import.
   *
   * NOTE: Currently, loading one row group at a time is required.
   */
  std::pair<size_t, size_t> loadRowGroups(
      const RowGroupInterval& row_group_interval,
      const std::map<int, Chunk_NS::Chunk>& chunks,
      const ForeignTableSchema& schema,
      const std::map<int, StringDictionary*>& column_dictionaries,
      const int num_threads = 1);

  /**
   * @brief Preview rows of data and column types in a set of files
   *
   * @param files - files to preview
   * @param max_num_rows - maximum number of rows to preview
   *
   * @return a `DataPreview` instance that contains relevant preview
   * information
   */
  DataPreview previewFiles(const std::vector<std::string>& files,
                           const size_t max_num_rows);

 private:
  /**
   * Suggest a possible Parquet to OmniSci column mapping based on heuristics.
   *
   * @param parquet_column - the column descriptor of the Parquet column
   *
   * @return a supported OmniSci `SQLTypeInfo` given the Parquet column type
   *
   * NOTE: the suggested type may be entirely inappropriate given a specific
   * use-case; however, it is guaranteed to be an allowed mapping. For example,
   * geo-types are never attempted to be detected and instead strings are
   * always suggested in their place.
   */
  static SQLTypeInfo suggestColumnMapping(
      const parquet::ColumnDescriptor* parquet_column);

  std::list<std::unique_ptr<ChunkMetadata>> appendRowGroups(
      const std::vector<RowGroupInterval>& row_group_intervals,
      const int parquet_column_index,
      const ColumnDescriptor* column_descriptor,
      std::list<Chunk_NS::Chunk>& chunks,
      StringDictionary* string_dictionary,
      RejectedRowIndices* rejected_row_indices,
      const bool is_for_detect = false,
      const std::optional<int64_t> max_levels_read = std::nullopt);

  std::shared_ptr<arrow::fs::FileSystem> file_system_;
  FileReaderMap* file_reader_cache_;

  const RenderGroupAnalyzerMap* render_group_analyzer_map_;
  std::string foreign_table_name_;
};
}  // namespace foreign_storage
