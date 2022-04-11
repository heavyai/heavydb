/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef TABLE_DESCRIPTOR_H
#define TABLE_DESCRIPTOR_H

#include <cstdint>
#include <string>

#include "DataMgr/MemoryLevel.h"
#include "SchemaMgr/TableInfo.h"
#include "Shared/sqldefs.h"

/**
 * @type StorageType
 * @brief Encapsulates an enumeration of table storage type strings
 */
struct StorageType {
  static constexpr char const* LOCAL_TABLE = "LOCAL_TABLE";
};

/**
 * @type TableDescriptor
 * @brief specifies the content in-memory of a row in the table metadata table
 *
 */

#define DEFAULT_MAX_ROLLBACK_EPOCHS 3
struct TableDescriptor {
  int32_t tableId;       /**< tableId starts at 0 for valid tables. */
  std::string tableName; /**< tableName is the name of the table table -must be unique */
  int32_t userId;
  int32_t nColumns;
  bool isView;
  std::string viewSQL;
  std::string fragments;  // placeholder for fragmentation information
  int32_t maxFragRows;    // max number of rows per fragment
  int64_t maxChunkSize;   // max number of rows per fragment
  int32_t fragPageSize;   // page size
  int64_t maxRows;        // max number of rows in the table
  std::string
      keyMetainfo;  // meta-information about shard keys and shared dictionary, as JSON

  int sortedColumnId;  // Id of the column to be sorted on
  Data_Namespace::MemoryLevel persistenceLevel;
  // Spi means Sequential Positional Index which is equivalent to the input index in a
  // RexInput node
  std::vector<int> columnIdBySpi_;  // spi = 1,2,3,...
  std::string storageType;          // foreign/local storage

  int32_t maxRollbackEpochs;

  // write mutex, only to be used inside catalog package
  std::shared_ptr<std::mutex> mutex_;

  TableDescriptor()
      : tableId(-1)
      , sortedColumnId(0)
      , persistenceLevel(Data_Namespace::MemoryLevel::DISK_LEVEL)
      , maxRollbackEpochs(DEFAULT_MAX_ROLLBACK_EPOCHS)
      , mutex_(std::make_shared<std::mutex>()) {}

  virtual ~TableDescriptor() = default;

  inline bool isTemporaryTable() const {
    return persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL;
  }
};

// compare for lowest id
inline bool compare_td_id(const TableDescriptor* first, const TableDescriptor* second) {
  return (first->tableId < second->tableId);
}

inline bool table_is_temporary(const TableDescriptor* const td) {
  return td->persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL;
}

struct TableDescriptorUpdateParams {
  int32_t max_rollback_epochs;
  int64_t max_rows;

  TableDescriptorUpdateParams(const TableDescriptor* td)
      : max_rollback_epochs(td->maxRollbackEpochs), max_rows(td->maxRows) {}

  bool operator==(const TableDescriptor* td) {
    if (max_rollback_epochs != td->maxRollbackEpochs) {
      return false;
    }
    if (max_rows != td->maxRows) {
      return false;
    }
    // Add more tests for additional params as needed
    return true;
  }
};

#endif  // TABLE_DESCRIPTOR
