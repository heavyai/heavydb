/*
 * Copyright 2020 OmniSci, Inc.
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

#pragma once

#include <list>
#include <memory>

#include "Catalog/CatalogFwd.h"
#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/ChunkMetadata.h"
#include "DataMgr/DataMgr.h"
#include "Shared/sqltypes.h"
#include "Utils/ChunkIter.h"

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;
using Data_Namespace::MemoryLevel;

namespace Chunk_NS {

class Chunk {
 public:
  Chunk(bool pinnable = true)
      : buffer_(nullptr)
      , index_buf_(nullptr)
      , column_desc_(nullptr)
      , pinnable_(pinnable) {}

  explicit Chunk(const ColumnDescriptor* td)
      : buffer_(nullptr), index_buf_(nullptr), column_desc_(td), pinnable_(true) {}

  Chunk(const ColumnDescriptor* td, bool pinnable)
      : buffer_(nullptr), index_buf_(nullptr), column_desc_(td), pinnable_(pinnable) {}

  Chunk(AbstractBuffer* b,
        AbstractBuffer* ib,
        const ColumnDescriptor* td,
        bool pinnable = true)
      : buffer_(b), index_buf_(ib), column_desc_(td), pinnable_(pinnable) {}

  ~Chunk() { unpinBuffer(); }

  void setPinnable(bool pinnable) { pinnable_ = pinnable; }

  const ColumnDescriptor* getColumnDesc() const { return column_desc_; }

  void setColumnDesc(const ColumnDescriptor* cd) { column_desc_ = cd; }

  static void translateColumnDescriptorsToChunkVec(
      const std::list<const ColumnDescriptor*>& colDescs,
      std::vector<Chunk>& chunkVec);

  ChunkIter begin_iterator(const std::shared_ptr<ChunkMetadata>&,
                           int start_idx = 0,
                           int skip = 1) const;

  size_t getNumElemsForBytesEncodedDataAtIndices(const int8_t* index_data,
                                                 const std::vector<size_t>& selected_idx,
                                                 const size_t byte_limit);

  size_t getNumElemsForBytesInsertData(const DataBlockPtr& src_data,
                                       const size_t num_elems,
                                       const size_t start_idx,
                                       const size_t byte_limit,
                                       const bool replicating = false);

  std::shared_ptr<ChunkMetadata> appendData(DataBlockPtr& srcData,
                                            const size_t numAppendElems,
                                            const size_t startIdx,
                                            const bool replicating = false);

  std::shared_ptr<ChunkMetadata> appendEncodedDataAtIndices(
      const Chunk& src_chunk,
      const std::vector<size_t>& selected_idx);

  std::shared_ptr<ChunkMetadata> appendEncodedData(const Chunk& src_chunk,
                                                   const size_t num_elements,
                                                   const size_t start_idx);

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
                                         const size_t num_elems,
                                         const bool pinnable = true);

  /**
   * @brief Compose a chunk from components and return it
   *
   * @param cd - the column descriptor for the chunk
   * @param data_buffer - the data buffer for the chunk
   * @param index_buffer - the (optional) index buffer for the chunk
   * @param pinnable - sets the chunk as pinnable (or not)
   *
   * @return a chunk composed of supplied components
   *
   * Note, the `index_buffer` is only applicable if the column is a variable
   * length column. If the column type is not variable length, this parameter
   * is ignored.
   */
  static std::shared_ptr<Chunk> getChunk(const ColumnDescriptor* cd,
                                         AbstractBuffer* data_buffer,
                                         AbstractBuffer* index_buffer,
                                         const bool pinnable = true);

  bool isChunkOnDevice(DataMgr* data_mgr,
                       const ChunkKey& key,
                       const MemoryLevel mem_level,
                       const int device_id);

  AbstractBuffer* getBuffer() const { return buffer_; }

  AbstractBuffer* getIndexBuf() const { return index_buf_; }

  void setBuffer(AbstractBuffer* b) { buffer_ = b; }

  void setIndexBuffer(AbstractBuffer* ib) { index_buf_ = ib; }

  void initEncoder();

  void decompress(int8_t* compressed, VarlenDatum* result, Datum* datum) const;

  std::string toString() const;

 private:
  void setChunkBuffer(AbstractBuffer* buffer, AbstractBuffer* index_buffer);

  AbstractBuffer* buffer_;
  AbstractBuffer* index_buf_;
  const ColumnDescriptor* column_desc_;
  // When using Chunk as a buffer wrapper, disable pinnable_ to avoid assymetric pin/unPin
  // of the buffers
  bool pinnable_;

  void unpinBuffer();
};

}  // namespace Chunk_NS
