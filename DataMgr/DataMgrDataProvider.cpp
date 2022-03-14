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

#include "DataMgr/DataMgrDataProvider.h"

#include "DataMgr/DataMgr.h"

DataMgrDataProvider::DataMgrDataProvider(DataMgr* data_mgr) : data_mgr_(data_mgr) {}

std::shared_ptr<Chunk_NS::Chunk> DataMgrDataProvider::getChunk(
    ColumnInfoPtr col_info,
    const ChunkKey& key,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    const size_t num_bytes,
    const size_t num_elems) {
  return Chunk_NS::Chunk::getChunk(
      col_info, data_mgr_, key, memory_level, device_id, num_bytes, num_elems);
}
Fragmenter_Namespace::TableInfo DataMgrDataProvider::getTableMetadata(
    int db_id,
    int table_id) const {
  return data_mgr_->getTableMetadata(db_id, table_id);
}
const DictDescriptor* DataMgrDataProvider::getDictMetadata(int db_id,
                                                           int dict_id,
                                                           bool load_dict) const {
  return data_mgr_->getDictMetadata(db_id, dict_id, load_dict);
}
