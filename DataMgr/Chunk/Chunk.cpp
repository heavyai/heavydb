/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @file Chunk.cpp
 * @brief
 *
 */

#include "DataMgr/Chunk/Chunk.h"
#include "Catalog/ColumnDescriptor.h"
#include "DataMgr/ArrayNoneEncoder.h"
#include "DataMgr/FixedLengthArrayNoneEncoder.h"
#include "DataMgr/StringNoneEncoder.h"
#include "Shared/toString.h"

namespace Chunk_NS {
std::shared_ptr<Chunk> Chunk::getChunk(const ColumnDescriptor* cd,
                                       DataMgr* data_mgr,
                                       const ChunkKey& key,
                                       const MemoryLevel memoryLevel,
                                       const int deviceId,
                                       const size_t numBytes,
                                       const size_t numElems,
                                       const bool pinnable) {
  std::shared_ptr<Chunk> chunkp = std::make_shared<Chunk>(Chunk(cd, pinnable));
  chunkp->getChunkBuffer(data_mgr, key, memoryLevel, deviceId, numBytes, numElems);
  return chunkp;
}

std::shared_ptr<Chunk> Chunk::getChunk(const ColumnDescriptor* cd,
                                       AbstractBuffer* data_buffer,
                                       AbstractBuffer* index_buffer,
                                       const bool pinnable) {
  std::shared_ptr<Chunk> chunkp = std::make_shared<Chunk>(Chunk(cd, pinnable));
  chunkp->setChunkBuffer(data_buffer, index_buffer);
  return chunkp;
}

bool Chunk::isChunkOnDevice(DataMgr* data_mgr,
                            const ChunkKey& key,
                            const MemoryLevel mem_level,
                            const int device_id) {
  if (column_desc_->columnType.is_varlen() &&
      !column_desc_->columnType.is_fixlen_array()) {
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

void Chunk::setChunkBuffer(AbstractBuffer* buffer, AbstractBuffer* index_buffer) {
  if (column_desc_->columnType.is_varlen() &&
      !column_desc_->columnType.is_fixlen_array()) {
    CHECK(index_buffer);
    buffer_ = buffer;
    index_buf_ = index_buffer;
    switch (column_desc_->columnType.get_type()) {
      case kARRAY: {
        auto array_encoder = dynamic_cast<ArrayNoneEncoder*>(buffer_->getEncoder());
        CHECK(array_encoder);
        array_encoder->setIndexBuffer(index_buf_);
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        CHECK_EQ(kENCODING_NONE, column_desc_->columnType.get_compression());
        auto str_encoder = dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
        CHECK(str_encoder);
        str_encoder->setIndexBuffer(index_buf_);
        break;
      }
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON: {
        auto str_encoder = dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
        CHECK(str_encoder);
        str_encoder->setIndexBuffer(index_buf_);
        break;
      }
      default:
        UNREACHABLE();
    }
  } else {
    buffer_ = buffer;
  }
}

void Chunk::getChunkBuffer(DataMgr* data_mgr,
                           const ChunkKey& key,
                           const MemoryLevel mem_level,
                           const int device_id,
                           const size_t num_bytes,
                           const size_t num_elems) {
  if (column_desc_->columnType.is_varlen() &&
      !column_desc_->columnType.is_fixlen_array()) {
    ChunkKey data_key = key;
    data_key.push_back(1);
    ChunkKey index_key = key;
    index_key.push_back(2);
    setChunkBuffer(
        data_mgr->getChunkBuffer(data_key, mem_level, device_id, num_bytes),
        data_mgr->getChunkBuffer(
            index_key, mem_level, device_id, (num_elems + 1) * sizeof(StringOffsetT)));

  } else {
    setChunkBuffer(data_mgr->getChunkBuffer(key, mem_level, device_id, num_bytes),
                   nullptr);
  }
}

void Chunk::createChunkBuffer(DataMgr* data_mgr,
                              const ChunkKey& key,
                              const MemoryLevel mem_level,
                              const int device_id,
                              const size_t page_size) {
  if (column_desc_->columnType.is_varlen() &&
      !column_desc_->columnType.is_fixlen_array()) {
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

size_t Chunk::getNumElemsForBytesEncodedDataAtIndices(
    const int8_t* index_data,
    const std::vector<size_t>& selected_idx,
    const size_t byte_limit) {
  CHECK(column_desc_->columnType.is_varlen());
  CHECK(buffer_->getEncoder());
  return buffer_->getEncoder()->getNumElemsForBytesEncodedDataAtIndices(
      index_data, selected_idx, byte_limit);
}

size_t Chunk::getNumElemsForBytesInsertData(const DataBlockPtr& src_data,
                                            const size_t num_elems,
                                            const size_t start_idx,
                                            const size_t byte_limit,
                                            const bool replicating) {
  CHECK(column_desc_->columnType.is_varlen());
  switch (column_desc_->columnType.get_type()) {
    case kARRAY: {
      if (column_desc_->columnType.get_size() > 0) {
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
      CHECK_EQ(kENCODING_NONE, column_desc_->columnType.get_compression());
      StringNoneEncoder* str_encoder =
          dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
      return str_encoder->getNumElemsForBytesInsertData(
          src_data.stringsPtr, start_idx, num_elems, byte_limit, replicating);
    }
    case kPOINT:
    case kMULTIPOINT:
    case kLINESTRING:
    case kMULTILINESTRING:
    case kPOLYGON:
    case kMULTIPOLYGON: {
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

std::shared_ptr<ChunkMetadata> Chunk::appendEncodedDataAtIndices(
    const Chunk& src_chunk,
    const std::vector<size_t>& selected_idx) {
  const auto& ti = column_desc_->columnType;
  int8_t* data_buffer_ptr = src_chunk.getBuffer()->getMemoryPtr();
  const int8_t* index_buffer_ptr =
      ti.is_varlen_indeed() ? src_chunk.getIndexBuf()->getMemoryPtr() : nullptr;
  CHECK(buffer_->getEncoder());
  return buffer_->getEncoder()->appendEncodedDataAtIndices(
      index_buffer_ptr, data_buffer_ptr, selected_idx);
}

std::shared_ptr<ChunkMetadata> Chunk::appendEncodedData(const Chunk& src_chunk,
                                                        const size_t num_elements,
                                                        const size_t start_idx) {
  const auto& ti = column_desc_->columnType;
  int8_t* data_buffer_ptr = src_chunk.getBuffer()->getMemoryPtr();
  const int8_t* index_buffer_ptr =
      ti.is_varlen_indeed() ? src_chunk.getIndexBuf()->getMemoryPtr() : nullptr;
  CHECK(buffer_->getEncoder());
  return buffer_->getEncoder()->appendEncodedData(
      index_buffer_ptr, data_buffer_ptr, start_idx, num_elements);
}

std::shared_ptr<ChunkMetadata> Chunk::appendData(DataBlockPtr& src_data,
                                                 const size_t num_elems,
                                                 const size_t start_idx,
                                                 const bool replicating) {
  const auto& ti = column_desc_->columnType;
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
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON: {
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
  if (pinnable_) {
    if (buffer_) {
      buffer_->unPin();
    }
    if (index_buf_) {
      index_buf_->unPin();
    }
  }
}

void Chunk::initEncoder() {
  buffer_->initEncoder(column_desc_->columnType);
  if (column_desc_->columnType.is_varlen() &&
      !column_desc_->columnType.is_fixlen_array()) {
    switch (column_desc_->columnType.get_type()) {
      case kARRAY: {
        ArrayNoneEncoder* array_encoder =
            dynamic_cast<ArrayNoneEncoder*>(buffer_->getEncoder());
        array_encoder->setIndexBuffer(index_buf_);
        break;
      }
      case kTEXT:
      case kVARCHAR:
      case kCHAR: {
        CHECK_EQ(kENCODING_NONE, column_desc_->columnType.get_compression());
        StringNoneEncoder* str_encoder =
            dynamic_cast<StringNoneEncoder*>(buffer_->getEncoder());
        str_encoder->setIndexBuffer(index_buf_);
        break;
      }
      case kPOINT:
      case kMULTIPOINT:
      case kLINESTRING:
      case kMULTILINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON: {
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
  it.type_info = column_desc_->columnType;
  it.skip = skip;
  it.skip_size = column_desc_->columnType.get_size();
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

void Chunk::translateColumnDescriptorsToChunkVec(
    const std::list<const ColumnDescriptor*>& colDescs,
    std::vector<Chunk>& chunkVec) {
  for (auto cd : colDescs) {
    chunkVec.emplace_back(cd);
  }
}

std::string Chunk::toString() const {
  return ::typeName(this) + "(buffer=" + ::toString(buffer_) +
         ", index_buf=" + ::toString(index_buf_) +
         ", column_desc=" + ::toString(column_desc_) + ")";
}
}  // namespace Chunk_NS
