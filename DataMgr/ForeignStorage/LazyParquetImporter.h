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
#include "ForeignTableColumnMap.h"
#include "ImportExport/Importer.h"
#include "Interval.h"

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
};

/**
 * A lazy Parquet file loader
 */
class LazyParquetImporter : public import_export::Importer {
 public:
  using RowGroupMetadataVector = std::vector<RowGroupMetadata>;

  LazyParquetImporter(import_export::Loader* provided_loader,
                      const std::string& file_name,
                      const import_export::CopyParams& copy_params,
                      RowGroupMetadataVector& metadata_vector);

  /**
   * Partial load of a Parquet file
   *
   * @param row_group_interval - [start,end] inclusive interval specifying row groups to
   * read
   * @param logical_column_interval - [start,end] inclusive interval specifying logical
   * columns to read
   * @param metadata_scan - if true, a scan is performed over the entire Parquet file to
   * load metadata
   */
  void partialImport(const Interval<RowGroupType>& row_group_interval,
                     const Interval<ColumnType>& logical_column_interval,
                     const bool metadata_scan = false);

  /**
   * Scan the parquet file, importing only the metadata
   */
  void metadataScan() { partialImport({0, 0}, {0, 0}, true); }

 private:
  RowGroupMetadataVector& row_group_metadata_vec_;
  ForeignTableColumnMap foreign_table_column_map_;
};

}  // namespace foreign_storage
