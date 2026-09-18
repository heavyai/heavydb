/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
#include "TableDescriptor.h"
struct TableMetadata {
  int32_t table_id;
  std::string table_name;
  int32_t owner_id;
  std::string owner_name;
  bool is_temp_table;
  int32_t num_columns;
  bool is_sharded;
  int32_t num_shards;
  int64_t max_rows;
  int32_t fragment_size;
  int32_t max_rollback_epochs;
  int32_t min_epoch;
  int32_t max_epoch;
  int32_t min_epoch_floor;
  int32_t max_epoch_floor;
  int64_t num_bytes;
  int64_t num_files;
  int64_t num_pages;
  TableMetadata(const TableDescriptor* td)
      : table_id(td->tableId)
      , table_name(td->tableName)
      , owner_id(td->userId)
      , is_temp_table(td->persistenceLevel != Data_Namespace::MemoryLevel::DISK_LEVEL)
      , num_columns(td->nColumns)
      , is_sharded(td->nShards > 0)
      , num_shards(td->nShards)
      , max_rows(td->maxRows)
      , fragment_size(td->maxFragRows)
      , max_rollback_epochs(td->maxRollbackEpochs) {}
};
