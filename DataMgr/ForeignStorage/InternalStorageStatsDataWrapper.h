/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "DataMgr/FileMgr/FileMgr.h"
#include "InternalSystemDataWrapper.h"

namespace foreign_storage {
struct StorageDetails {
  std::string node{"Server"};
  int32_t database_id;
  int32_t table_id;
  int32_t shard_id;
  uint64_t total_dictionary_data_file_size;
  File_Namespace::StorageStats storage_stats;

  StorageDetails(int32_t database_id,
                 int32_t table_id,
                 int32_t shard_id,
                 int64_t total_dictionary_data_file_size,
                 File_Namespace::StorageStats storage_stats)
      : database_id(database_id)
      , table_id(table_id)
      , shard_id(shard_id)
      , total_dictionary_data_file_size(total_dictionary_data_file_size)
      , storage_stats(storage_stats) {}
};

class InternalStorageStatsDataWrapper : public InternalSystemDataWrapper {
 public:
  InternalStorageStatsDataWrapper();

  InternalStorageStatsDataWrapper(const int db_id, const ForeignTable* foreign_table);

 private:
  void initializeObjectsForTable(const std::string& table_name) override;

  void populateChunkBuffersForTable(
      const std::string& table_name,
      std::map<std::string, import_export::UnmanagedTypedImportBuffer*>& import_buffers)
      override;

  std::vector<StorageDetails> storage_details_;
};
}  // namespace foreign_storage
