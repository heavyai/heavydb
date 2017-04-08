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

#include <string>
#include <cstdint>
#include "../Shared/sqldefs.h"
#include "../Fragmenter/AbstractFragmenter.h"

/**
 * @type TableDescriptor
 * @brief specifies the content in-memory of a row in the table metadata table
 *
 */

struct TableDescriptor {
  int32_t tableId; /**< tableId starts at 0 for valid tables. */
  int32_t shard;
  std::string tableName; /**< tableName is the name of the table table -must be unique */
  int32_t nColumns;
  bool isView;
  std::string viewSQL;
  std::string fragments;                          // placeholder for fragmentation information
  Fragmenter_Namespace::FragmenterType fragType;  // fragmentation type. Only INSERT_ORDER is supported now.
  int32_t maxFragRows;                            // max number of rows per fragment
  int64_t maxChunkSize;                           // max number of rows per fragment
  int32_t fragPageSize;                           // page size
  int64_t maxRows;                                // max number of rows in the table
  std::string partitions;                         // distributed partition scheme
  Fragmenter_Namespace::AbstractFragmenter*
      fragmenter;                // point to fragmenter object for the table.  it's instantiated upon first use.
  bool isIdentityBasedSharding;  // sharding based on: a) true  - column value based,
                                 //                    b) false - round robin or murmur hash function based
  bool isLogicalTable;           // true - logical table, false - physical table (default: false)
  int32_t nShards;               // # of shards, i.e. physical tables for this logical table (default: 0)
  int shardedColumnId;           // Id of the column to be sharded on
  TableDescriptor()
      : tableId(-1), shard(-1), isIdentityBasedSharding(true), isLogicalTable(false), nShards(0), shardedColumnId(0) {}
};

#endif  // TABLE_DESCRIPTOR
