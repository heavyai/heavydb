/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DATAFRAME_TABLE_DESCRIPTOR_H
#define DATAFRAME_TABLE_DESCRIPTOR_H

#include <cstdint>
#include <string>

#include "TableDescriptor.h"

/**
 * @type DataframeTableDescriptor
 * @brief specifies the content in-memory of a row in the table metadata table
 *
 */

struct DataframeTableDescriptor : TableDescriptor {
  int64_t skipRows;       // number of skipped rows of data in CSV file
  std::string delimiter;  // delimiter of values in the CSV file
  bool hasHeader;         // does table has a header in CSV file

  DataframeTableDescriptor()
      : TableDescriptor(), skipRows(0), delimiter(","), hasHeader(true) {}

  DataframeTableDescriptor(const TableDescriptor& td) {
    tableId = td.tableId;
    shard = td.shard;
    tableName = td.tableName;
    userId = td.userId;
    nColumns = td.nColumns;
    isView = td.isView;
    viewSQL = td.viewSQL;
    fragments = td.fragments;
    fragType = td.fragType;
    maxFragRows = td.maxFragRows;
    maxChunkSize = td.maxChunkSize;
    fragPageSize = td.fragPageSize;
    maxRows = td.maxRows;
    partitions = td.partitions;
    keyMetainfo = td.keyMetainfo;
    fragmenter = td.fragmenter;
    nShards = td.nShards;
    shardedColumnId = td.shardedColumnId;
    sortedColumnId = td.sortedColumnId;
    persistenceLevel = td.persistenceLevel;
    hasDeletedCol = td.hasDeletedCol;
    columnIdBySpi_ = td.columnIdBySpi_;
    storageType = td.storageType;
    mutex_ = td.mutex_;
    skipRows = 0;
    delimiter = ",";
    hasHeader = true;
  }

  ~DataframeTableDescriptor() override = default;
};

#endif  // DATAFRAME_TABLE_DESCRIPTOR_H
