/* Copyright 2020 OmniSci, Inc.
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
#include <string>
#include "TableDescriptor.h"
struct TableMetadata {
  int32_t table_id;
  std::string table_name;
  int32_t owner_id;
  std::string owner_name;
  bool is_temp_table;
  int32_t num_columns;
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
      , max_rows(td->maxRows)
      , fragment_size(td->maxFragRows)
      , max_rollback_epochs(td->maxRollbackEpochs) {}
};
