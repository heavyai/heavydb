/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "InsertOrderFragmenter.h"

namespace Fragmenter_Namespace {

class SortedOrderFragmenter : public InsertOrderFragmenter {
 public:
  SortedOrderFragmenter(
      const std::vector<int> chunkKeyPrefix,
      std::vector<Chunk_NS::Chunk>& chunkVec,
      Data_Namespace::DataMgr* dataMgr,
      Catalog_Namespace::Catalog* catalog,
      const int physicalTableId,
      const int shard,
      const size_t maxFragmentRows = DEFAULT_FRAGMENT_ROWS,
      const size_t maxChunkSize = DEFAULT_MAX_CHUNK_SIZE,
      const size_t pageSize = DEFAULT_PAGE_SIZE /*default 1MB*/,
      const size_t maxRows = DEFAULT_MAX_ROWS,
      const Data_Namespace::MemoryLevel defaultInsertLevel = Data_Namespace::DISK_LEVEL)
      : InsertOrderFragmenter(chunkKeyPrefix,
                              chunkVec,
                              dataMgr,
                              catalog,
                              physicalTableId,
                              shard,
                              maxFragmentRows,
                              maxChunkSize,
                              pageSize,
                              maxRows,
                              defaultInsertLevel) {}

  ~SortedOrderFragmenter() override {}
  void insertData(InsertData& insert_data_struct) override {
    sortData(insert_data_struct);
    InsertOrderFragmenter::insertData(insert_data_struct);
  }

  void insertDataNoCheckpoint(InsertData& insert_data_struct) override {
    sortData(insert_data_struct);
    InsertOrderFragmenter::insertDataNoCheckpoint(insert_data_struct);
  }

  SortedOrderFragmenter(SortedOrderFragmenter&&) = default;
  SortedOrderFragmenter(const SortedOrderFragmenter&) = delete;
  SortedOrderFragmenter& operator=(const SortedOrderFragmenter&) = delete;

 protected:
  virtual void sortData(InsertData& insertDataStruct);
};

}  // namespace Fragmenter_Namespace
