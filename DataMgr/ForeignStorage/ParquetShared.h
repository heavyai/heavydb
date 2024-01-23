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

#include <arrow/api.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/statistics.h>
#include <parquet/types.h>

#include "Catalog/CatalogFwd.h"
#include "DataMgr/ChunkMetadata.h"
#include "Shared/heavyai_shared_mutex.h"

namespace foreign_storage {

using UniqueReaderPtr = std::unique_ptr<parquet::arrow::FileReader>;
using ReaderPtr = parquet::arrow::FileReader*;

struct RowGroupInterval {
  std::string file_path;
  int start_index{-1}, end_index{-1};
};

struct RowGroupMetadata {
  std::string file_path;
  int row_group_index;
  std::list<std::shared_ptr<ChunkMetadata>> column_chunk_metadata;
};

UniqueReaderPtr open_parquet_table(const std::string& file_path,
                                   std::shared_ptr<arrow::fs::FileSystem>& file_system);

std::pair<int, int> get_parquet_table_size(const ReaderPtr& reader);

const parquet::ColumnDescriptor* get_column_descriptor(
    const parquet::arrow::FileReader* reader,
    const int logical_column_index);

void validate_equal_column_descriptor(
    const parquet::ColumnDescriptor* reference_descriptor,
    const parquet::ColumnDescriptor* new_descriptor,
    const std::string& reference_file_path,
    const std::string& new_file_path);

std::unique_ptr<ColumnDescriptor> get_sub_type_column_descriptor(
    const ColumnDescriptor* column);

std::shared_ptr<parquet::Statistics> validate_and_get_column_metadata_statistics(
    const parquet::ColumnChunkMetaData* column_metadata);

// A cache for parquet FileReaders which locks access for parallel use.
class FileReaderMap {
 public:
  const ReaderPtr getOrInsert(const std::string& path,
                              std::shared_ptr<arrow::fs::FileSystem>& file_system) {
    bool path_exists_and_is_open = true;
    {
      heavyai::shared_lock<heavyai::shared_mutex> cache_lock(mutex_);
      if (map_.count(path) < 1 || !(map_.at(path))) {
        path_exists_and_is_open = false;
      }
    }

    if (!path_exists_and_is_open) {
      auto parquet_file_reader = open_parquet_table(path, file_system);
      heavyai::unique_lock<heavyai::shared_mutex> cache_lock(mutex_);
      // Check the `path_exists_and_is_open` condition again, it's possible
      // another thread is competing with this thread in the case of concurrent
      // `getOrInsert`
      if (map_.count(path) < 1 || !(map_.at(path))) {
        map_[path] = std::move(parquet_file_reader);
      }
    }

    heavyai::unique_lock<heavyai::shared_mutex> cache_lock(mutex_);
    return map_.at(path).get();
  }

  const ReaderPtr insert(const std::string& path,
                         std::shared_ptr<arrow::fs::FileSystem>& file_system) {
    auto parquet_file_reader = open_parquet_table(path, file_system);
    heavyai::unique_lock<heavyai::shared_mutex> cache_lock(mutex_);
    map_[path] = std::move(parquet_file_reader);
    return map_.at(path).get();
  }

  void initializeIfEmpty(const std::string& path) {
    heavyai::unique_lock<heavyai::shared_mutex> cache_lock(mutex_);
    if (map_.count(path) < 1) {
      map_.emplace(path, UniqueReaderPtr());
    }
  }

  void clear() {
    heavyai::unique_lock<heavyai::shared_mutex> cache_lock(mutex_);
    map_.clear();
  }

 private:
  mutable heavyai::shared_mutex mutex_;
  std::map<const std::string, UniqueReaderPtr> map_;
};
}  // namespace foreign_storage
