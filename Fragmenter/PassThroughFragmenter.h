/*
 * Copyright 2023 HEAVY.AI, Inc.
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

/**
 * @file	PassThroughFragmenter.h
 * @brief
 *
 */

#pragma once

#include "InsertOrderFragmenter.h"

namespace Fragmenter_Namespace {

class PassThroughFragmenter : public InsertOrderFragmenter {
 public:
  PassThroughFragmenter(
      const std::vector<int> chunk_key_prefix,
      std::vector<Chunk_NS::Chunk>& chunk_vec,
      Data_Namespace::DataMgr* data_mgr,
      Catalog_Namespace::Catalog* catalog,
      const int physical_table_id,
      const int shard,
      const size_t max_fragment_rows = DEFAULT_FRAGMENT_ROWS,
      const size_t max_chunk_size = DEFAULT_MAX_CHUNK_SIZE,
      const size_t page_size = DEFAULT_PAGE_SIZE /*default 1MB*/,
      const size_t max_rows = DEFAULT_MAX_ROWS,
      const Data_Namespace::MemoryLevel default_insert_level = Data_Namespace::DISK_LEVEL,
      const bool uses_foreign_storage = false)
      : InsertOrderFragmenter(chunk_key_prefix,
                              chunk_vec,
                              data_mgr,
                              catalog,
                              physical_table_id,
                              shard,
                              max_fragment_rows,
                              max_chunk_size,
                              page_size,
                              max_rows,
                              default_insert_level,
                              uses_foreign_storage) {}

  ~PassThroughFragmenter() override {}

 protected:
  void insertChunksImpl(const InsertChunks& insert_chunk) override;
};

}  // namespace Fragmenter_Namespace
