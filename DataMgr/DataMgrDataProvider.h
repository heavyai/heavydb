/*
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

#include "DataProvider/DataProvider.h"

#include <memory>

namespace DataMgr_Namespace {
class DataMgr;
}

class DataMgrDataProvider : public DataProvider {
 public:
  DataMgrDataProvider(DataMgr* data_mgr);

  std::shared_ptr<Chunk_NS::Chunk> getChunk(
      ColumnInfoPtr col_info,
      const ChunkKey& key,
      const Data_Namespace::MemoryLevel memory_level,
      const int device_id,
      const size_t num_bytes,
      const size_t num_elems) override;

  Fragmenter_Namespace::TableInfo getTableMetadata(int db_id,
                                                   int table_id) const override;

  const DictDescriptor* getDictMetadata(int db_id,
                                        int dict_id,
                                        bool load_dict = true) const override;

 private:
  DataMgr* data_mgr_;
};