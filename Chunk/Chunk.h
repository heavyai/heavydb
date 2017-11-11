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

/**
 * @file Chunk.h
 * @author Wei Hong <wei@mapd.com>
 *
 */
#ifndef _CHUNK_H_
#define _CHUNK_H_

#include <list>
#include <memory>
#include "../Shared/sqltypes.h"
#include "../DataMgr/AbstractBuffer.h"
#include "../DataMgr/ChunkMetadata.h"
#include "../DataMgr/DataMgr.h"
#include "../Catalog/ColumnDescriptor.h"
#include "../Utils/ChunkIter.h"

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;
using Data_Namespace::MemoryLevel;

namespace Chunk_NS {
class Chunk;
};

namespace Chunk_NS {

class Chunk {
 public:
  Chunk() : buffer(nullptr), index_buf(nullptr), column_desc(nullptr) {}
  explicit Chunk(const ColumnDescriptor* td) : buffer(nullptr), index_buf(nullptr), column_desc(td) {}
  Chunk(AbstractBuffer* b, AbstractBuffer* ib, const ColumnDescriptor* td)
      : buffer(b), index_buf(ib), column_desc(td){};
  ~Chunk() { unpin_buffer(); }
  const ColumnDescriptor* get_column_desc() const { return column_desc; }
  static void translateColumnDescriptorsToChunkVec(const std::list<const ColumnDescriptor*>& colDescs,
                                                   std::vector<Chunk>& chunkVec) {
    for (auto cd : colDescs)
      chunkVec.push_back(Chunk(cd));
  }
  ChunkIter begin_iterator(const ChunkMetadata&, int start_idx = 0, int skip = 1) const;
  size_t getNumElemsForBytesInsertData(const DataBlockPtr& src_data,
                                       const size_t num_elems,
                                       const size_t start_idx,
                                       const size_t byte_limit);
  ChunkMetadata appendData(DataBlockPtr& srcData, const size_t numAppendElems, const size_t startIdx);
  void createChunkBuffer(DataMgr* data_mgr,
                         const ChunkKey& key,
                         const MemoryLevel mem_level,
                         const int deviceId = 0,
                         const size_t page_size = 0);
  void getChunkBuffer(DataMgr* data_mgr,
                      const ChunkKey& key,
                      const MemoryLevel mem_level,
                      const int deviceId = 0,
                      const size_t num_bytes = 0,
                      const size_t num_elems = 0);
  static std::shared_ptr<Chunk> getChunk(const ColumnDescriptor* cd,
                                         DataMgr* data_mgr,
                                         const ChunkKey& key,
                                         const MemoryLevel mem_level,
                                         const int deviceId,
                                         const size_t num_bytes,
                                         const size_t num_elems);
  bool isChunkOnDevice(DataMgr* data_mgr, const ChunkKey& key, const MemoryLevel mem_level, const int device_id);

  // protected:
  AbstractBuffer* get_buffer() const { return buffer; }
  AbstractBuffer* get_index_buf() const { return index_buf; }
  void set_buffer(AbstractBuffer* b) { buffer = b; }
  void set_index_buf(AbstractBuffer* ib) { index_buf = ib; }
  void init_encoder();
  void decompress(int8_t* compressed, VarlenDatum* result, Datum* datum) const;

 private:
  AbstractBuffer* buffer;
  AbstractBuffer* index_buf;
  const ColumnDescriptor* column_desc;
  void unpin_buffer();
};
}

#endif  // _CHUNK_H_
