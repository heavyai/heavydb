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

#include "DataMgr/DataMgr.h"

struct BufferPoolStats {
  size_t num_buffers;
  size_t num_bytes;
  size_t num_tables;
  size_t num_columns;
  size_t num_fragments;
  size_t num_chunks;

  void print() const {
    std::cout << std::endl
              << std::endl
              << "------------ Buffer Pool Stats  ------------" << std::endl;
    std::cout << "Num buffers: " << num_buffers << std::endl;
    std::cout << "Num bytes: " << num_bytes << std::endl;
    std::cout << "Num tables: " << num_tables << std::endl;
    std::cout << "Num columns: " << num_columns << std::endl;
    std::cout << "Num fragments: " << num_fragments << std::endl;
    std::cout << "Num chunks: " << num_chunks << std::endl;
    std::cout << "--------------------------------------------" << std::endl << std::endl;
  }
};

inline BufferPoolStats getBufferPoolStats(
    DataMgr* data_mgr,
    const Data_Namespace::MemoryLevel memory_level) {
  auto memory_infos = data_mgr->getMemoryInfo(memory_level);
  if (memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
    CHECK_EQ(memory_infos.size(), static_cast<size_t>(1));
  }
  std::set<std::vector<int32_t>> chunk_keys;
  std::set<std::vector<int32_t>> table_keys;
  std::set<std::vector<int32_t>> column_keys;
  std::set<std::vector<int32_t>> fragment_keys;
  size_t total_num_buffers{
      0};  // can be greater than chunk keys set size due to table replication
  size_t total_num_bytes{0};
  for (auto& pool_memory_info : memory_infos) {
    const auto& memory_data = pool_memory_info.nodeMemoryData;
    for (auto& memory_datum : memory_data) {
      total_num_buffers++;
      const auto& chunk_key = memory_datum.chunk_key;
      if (memory_datum.memStatus == Buffer_Namespace::MemStatus::FREE ||
          chunk_key.size() < 4) {
        continue;
      }
      total_num_bytes += (memory_datum.numPages * pool_memory_info.pageSize);
      table_keys.insert({chunk_key[0], chunk_key[1]});
      column_keys.insert({chunk_key[0], chunk_key[1], chunk_key[2]});
      fragment_keys.insert({chunk_key[0], chunk_key[1], chunk_key[3]});
      chunk_keys.insert(chunk_key);
    }
  }
  return {total_num_buffers,
          total_num_bytes,
          table_keys.size(),
          column_keys.size(),
          fragment_keys.size(),
          chunk_keys.size()};
}
