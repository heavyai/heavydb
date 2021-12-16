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

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/ChunkMetadata.h"
#include "DataMgr/DataMgr.h"
#include "SchemaMgr/ColumnInfo.h"
#include "Shared/sqltypes.h"
#include "Shared/toString.h"
#include "Utils/ChunkIter.h"

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;
using Data_Namespace::MemoryLevel;

namespace Chunk_NS {

class Chunk {
 public:
  Chunk() : buffer_(nullptr), index_buf_(nullptr), column_info_(nullptr) {}

  explicit Chunk(ColumnInfoPtr col_info)
      : buffer_(nullptr), index_buf_(nullptr), column_info_(col_info) {}

  Chunk(AbstractBuffer* b, AbstractBuffer* ib, ColumnInfoPtr col_info)
      : buffer_(b), index_buf_(ib), column_info_(col_info) {}

  ~Chunk() { unpinBuffer(); }

  ColumnInfoPtr getColumnInfo() const { return column_info_; }

  void setColumnInfo(ColumnInfoPtr col_info) { column_info_ = col_info; }

  int getTableId() const { return column_info_->table_id; }

  int getColumnId() const { return column_info_->column_id; }

  const SQLTypeInfo& getColumnType() const { return column_info_->type; }

  const std::string& getColumnName() const { return column_info_->name; }

  ChunkIter begin_iterator(const std::shared_ptr<ChunkMetadata>&,
                           int start_idx = 0,
                           int skip = 1) const;

  size_t getNumElemsForBytesInsertData(const DataBlockPtr& src_data,
                                       const size_t num_elems,
                                       const size_t start_idx,
                                       const size_t byte_limit,
                                       const bool replicating = false);

  std::shared_ptr<ChunkMetadata> appendData(DataBlockPtr& srcData,
                                            const size_t numAppendElems,
                                            const size_t startIdx,
                                            const bool replicating = false);

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

  static std::shared_ptr<Chunk> getChunk(ColumnInfoPtr col_info,
                                         DataMgr* data_mgr,
                                         const ChunkKey& key,
                                         const MemoryLevel mem_level,
                                         const int deviceId,
                                         const size_t num_bytes,
                                         const size_t num_elems);

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

  std::string toString() const {
    return ::typeName(this) + "(buffer=" + ::toString(buffer_) +
           ", index_buf=" + ::toString(index_buf_) +
           ", column_info=" + ::toString(*column_info_) + ")";
  }

 private:
  AbstractBuffer* buffer_;
  AbstractBuffer* index_buf_;
  ColumnInfoPtr column_info_;

  void unpinBuffer();
};

}  // namespace Chunk_NS
