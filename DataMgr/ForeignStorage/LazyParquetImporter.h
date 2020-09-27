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

#include <limits>
#include <list>

#include <arrow/filesystem/filesystem.h>

#include "ArrayMetadataStats.h"
#include "ForeignTableSchema.h"
#include "ImportExport/Importer.h"
#include "Interval.h"
#include "ParquetShared.h"

namespace foreign_storage {

/**
 * Metadata used when lazily importing Parquet data.
 */
struct RowGroupMetadata {
  int row_group_index;
  bool metadata_only;
  bool has_nulls;
  size_t num_elements;
  bool is_all_nulls;
  ArrayMetadataStats array_stats;
};

struct ParquetLoaderMetadata {
  std::string file_path;
  std::vector<RowGroupMetadata> row_group_metadata_vector;
};

/**
 * A lazy Parquet file loader
 */
class LazyParquetImporter : public import_export::Importer {
 public:
  LazyParquetImporter(import_export::Loader* provided_loader,
                      const std::set<std::string>& file_paths,
                      std::shared_ptr<arrow::fs::FileSystem> file_system,
                      const import_export::CopyParams& copy_params,
                      ParquetLoaderMetadata& parquet_loader_metadata,
                      ForeignTableSchema& schema);

  /**
   * Partial load of a Parquet file
   *
   * @param row_group_interval - vector of start and end (inclusive) row group intervals
   * to read in a file
   * @param column_interval - start and end (inclusive) interval specifying range of
   * columns to read
   * @param is_metadata_scan - flag indicating whether or not partial import is only for
   * scanning metadata
   */
  void partialImport(const std::vector<RowGroupInterval>& row_group_intervals,
                     const Interval<ColumnType>& column_interval,
                     const bool is_metadata_scan = false);

  /**
   * Scan the parquet file, importing only the metadata
   */
  void metadataScan();

 private:
  ParquetLoaderMetadata& parquet_loader_metadata_;
  const ForeignTableSchema& schema_;
  const std::set<std::string>& file_paths_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;
};

}  // namespace foreign_storage
