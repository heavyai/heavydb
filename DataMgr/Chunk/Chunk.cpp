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

/*
 * @file Chunk.cpp
 * @author Wei Hong <wei@mapd.com>
 */

#include "DataMgr/Chunk/Chunk.h"
#include "DataMgr/ArrayNoneEncoder.h"
#include "DataMgr/FixedLengthArrayNoneEncoder.h"
#include "DataMgr/StringNoneEncoder.h"

namespace Chunk_NS {
std::shared_ptr<Chunk> Chunk::getChunk(ColumnInfoPtr col_info,
                                       DataMgr* data_mgr,
                                       const ChunkKey& key,
                                       const MemoryLevel memoryLevel,
                                       const int deviceId,
                                       const size_t numBytes,
                                       const size_t numElems) {
  std::shared_ptr<Chunk> chunkp = std::make_shared<Chunk>(Chunk(col_info));
  chunkp->getChunkBuffer(data_mgr, key, memoryLevel, deviceId, numBytes, numElems);
  return chunkp;
}

bool Chunk::isChunkOnDevice(DataMgr* data_mgr,
                            const ChunkKey& key,
                            const MemoryLevel mem_level,
                            const int device_id) {
  if (column_info_->type.is_varlen() && !column_info_->type.is_fixlen_array()) {
    ChunkKey subKey = key;
    ChunkKey indexKey(subKey);
    indexKey.push_back(1);
    ChunkKey dataKey(subKey);
    dataKey.push_back(2);
    return data_mgr->isBufferOnDevice(indexKey, mem_level, device_id) &&
           data_mgr->isBufferOnDevice(dataKey, mem_level, device_id);
  } else {
    return data_mgr->isBufferOnDevice(key, mem_level, device_id);
  }
}

void Chunk::getChunkBuffer(DataMgr* data_mgr,
                           const ChunkKey& key,
                           const MemoryLevel mem_level,
                           const int device_id,
                           const size_t num_bytes,
                           const size_t num_elems) {
  if (column_info_->type.is_varlen() && !column_info_->type.is_fixlen_array()) {
    ChunkKey subKey = key;
    subKey.push_back(1);  // 1 for the main buffer_
    buffer_ = data_mgr->getChunkBuffer(subKey, mem_level, device_id, num_bytes);
    subKey.pop_back();
    subKey.push_back(2);  // 2 for the index buffer_
    index_buf_ = data_mgr->getChunkBuffer(
        subKey,
        mem_level,
        device_id,
        (num_elems + 1) * sizeof(StringOffsetT));  // always record n+1 offsets so string
                                                   // length can be calculated
    switch (column_info_->type.get_type()) {
      case kARRAY: {
        auto array_encoder = dynamic_cast<ArrayNoneEncoder*>(buffer_->getEncoder());
        CHECK(array_encoder);
        array_encoder->setIndexBuffer(index_buf_);
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        CHECK_EQ(kENCODING_NONE, column_info_->type.get_compression());
        auto str_encoder = dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
        CHECK(str_encoder);
        str_encoder->setIndexBuffer(index_buf_);
        break;
      }
      default:
        UNREACHABLE();
    }
  } else {
    buffer_ = data_mgr->getChunkBuffer(key, mem_level, device_id, num_bytes);
  }
}

void Chunk::createChunkBuffer(DataMgr* data_mgr,
                              const ChunkKey& key,
                              const MemoryLevel mem_level,
                              const int device_id,
                              const size_t page_size) {
  if (column_info_->type.is_varlen() && !column_info_->type.is_fixlen_array()) {
    ChunkKey subKey = key;
    subKey.push_back(1);  // 1 for the main buffer_
    buffer_ = data_mgr->createChunkBuffer(subKey, mem_level, device_id, page_size);
    subKey.pop_back();
    subKey.push_back(2);  // 2 for the index buffer_
    index_buf_ = data_mgr->createChunkBuffer(subKey, mem_level, device_id, page_size);
  } else {
    buffer_ = data_mgr->createChunkBuffer(key, mem_level, device_id, page_size);
  }
}

size_t Chunk::getNumElemsForBytesInsertData(const DataBlockPtr& src_data,
                                            const size_t num_elems,
                                            const size_t start_idx,
                                            const size_t byte_limit,
                                            const bool replicating) {
  CHECK(column_info_->type.is_varlen());
  switch (column_info_->type.get_type()) {
    case kARRAY: {
      if (column_info_->type.get_size() > 0) {
        FixedLengthArrayNoneEncoder* array_encoder =
            dynamic_cast<FixedLengthArrayNoneEncoder*>(buffer_->getEncoder());
        return array_encoder->getNumElemsForBytesInsertData(
            src_data.arraysPtr, start_idx, num_elems, byte_limit, replicating);
      }
      ArrayNoneEncoder* array_encoder =
          dynamic_cast<ArrayNoneEncoder*>(buffer_->getEncoder());
      return array_encoder->getNumElemsForBytesInsertData(
          src_data.arraysPtr, start_idx, num_elems, byte_limit, replicating);
    }
    case kTEXT:
    case kVARCHAR:
    case kCHAR: {
      CHECK_EQ(kENCODING_NONE, column_info_->type.get_compression());
      StringNoneEncoder* str_encoder =
          dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
      return str_encoder->getNumElemsForBytesInsertData(
          src_data.stringsPtr, start_idx, num_elems, byte_limit, replicating);
    }
    default:
      CHECK(false);
      return 0;
  }
}

std::shared_ptr<ChunkMetadata> Chunk::appendData(DataBlockPtr& src_data,
                                                 const size_t num_elems,
                                                 const size_t start_idx,
                                                 const bool replicating) {
  const auto& ti = column_info_->type;
  if (ti.is_varlen()) {
    switch (ti.get_type()) {
      case kARRAY: {
        if (ti.get_size() > 0) {
          FixedLengthArrayNoneEncoder* array_encoder =
              dynamic_cast<FixedLengthArrayNoneEncoder*>(buffer_->getEncoder());
          return array_encoder->appendData(
              src_data.arraysPtr, start_idx, num_elems, replicating);
        }
        ArrayNoneEncoder* array_encoder =
            dynamic_cast<ArrayNoneEncoder*>(buffer_->getEncoder());
        return array_encoder->appendData(
            src_data.arraysPtr, start_idx, num_elems, replicating);
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        CHECK_EQ(kENCODING_NONE, ti.get_compression());
        StringNoneEncoder* str_encoder =
            dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
        return str_encoder->appendData(
            src_data.stringsPtr, start_idx, num_elems, replicating);
      }
      default:
        CHECK(false);
    }
  }
  return buffer_->getEncoder()->appendData(
      src_data.numbersPtr, num_elems, ti, replicating);
}

void Chunk::unpinBuffer() {
  if (buffer_) {
    buffer_->unPin();
  }
  if (index_buf_) {
    index_buf_->unPin();
  }
}

void Chunk::initEncoder() {
  buffer_->initEncoder(column_info_->type);
  if (column_info_->type.is_varlen() && !column_info_->type.is_fixlen_array()) {
    switch (column_info_->type.get_type()) {
      case kARRAY: {
        ArrayNoneEncoder* array_encoder =
            dynamic_cast<ArrayNoneEncoder*>(buffer_->getEncoder());
        array_encoder->setIndexBuffer(index_buf_);
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        CHECK_EQ(kENCODING_NONE, column_info_->type.get_compression());
        StringNoneEncoder* str_encoder =
            dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
        str_encoder->setIndexBuffer(index_buf_);
        break;
      }
      default:
        CHECK(false);
    }
  }
}

ChunkIter Chunk::begin_iterator(const std::shared_ptr<ChunkMetadata>& chunk_metadata,
                                int start_idx,
                                int skip) const {
  ChunkIter it;
  it.type_info = column_info_->type;
  it.skip = skip;
  it.skip_size = column_info_->type.get_size();
  if (it.skip_size < 0) {  // if it's variable length
    it.current_pos = it.start_pos =
        index_buf_->getMemoryPtr() + start_idx * sizeof(StringOffsetT);
    it.end_pos = index_buf_->getMemoryPtr() + index_buf_->size() - sizeof(StringOffsetT);
    it.second_buf = buffer_->getMemoryPtr();
  } else {
    it.current_pos = it.start_pos = buffer_->getMemoryPtr() + start_idx * it.skip_size;
    it.end_pos = buffer_->getMemoryPtr() + buffer_->size();
    it.second_buf = nullptr;
  }
  it.num_elems = chunk_metadata->numElements;
  return it;
}
}  // namespace Chunk_NS
