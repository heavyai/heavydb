/*
 * Copyright 2019 OmniSci, Inc.
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
