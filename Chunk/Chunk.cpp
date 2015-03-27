/*
 * @file Chunk.cpp
 * @author Wei Hong <wei@mapd.com>
 */

#include "Chunk.h"
#include "../DataMgr/StringNoneEncoder.h"
#include "../DataMgr/StringTokDictEncoder.h"

namespace Chunk_NS {
  std::shared_ptr<Chunk>
  Chunk::getChunk(const ColumnDescriptor *cd, DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel memoryLevel, const int deviceId, const size_t numBytes, const size_t numElems) {
      std::shared_ptr<Chunk> chunkp = std::make_shared<Chunk>(Chunk(cd));
      chunkp->getChunkBuffer(data_mgr, key, memoryLevel, deviceId, numBytes, numElems);
      return chunkp;
    }

  bool
  Chunk::isChunkOnDevice(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int device_id) {
    if (column_desc->columnType.is_varlen()) {
      ChunkKey subKey = key;
      ChunkKey indexKey (subKey);
      indexKey.push_back(1);
      ChunkKey dataKey (subKey);
      dataKey.push_back(2);
      return data_mgr->isBufferOnDevice(indexKey,mem_level,device_id) && data_mgr->isBufferOnDevice(dataKey,mem_level,device_id);
    }
    else {
      return data_mgr->isBufferOnDevice(key,mem_level,device_id);
    }
  }

  void 
  Chunk::getChunkBuffer(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int device_id, const size_t num_bytes, const size_t num_elems)
  {
    if (column_desc->columnType.is_varlen()) {
      ChunkKey subKey = key;
      subKey.push_back(1); // 1 for the main buffer
      buffer = data_mgr->getChunkBuffer(subKey, mem_level, device_id, num_bytes);
      subKey.pop_back();
      subKey.push_back(2); // 2 for the index buffer
      index_buf = data_mgr->getChunkBuffer(subKey, mem_level, device_id, (num_elems + 1) * sizeof(StringOffsetT)); // always record n+1 offsets so string length can be calculated
      if (column_desc->columnType.get_compression() == kENCODING_NONE) {
        StringNoneEncoder *str_encoder = dynamic_cast<StringNoneEncoder*>(buffer->encoder);
        str_encoder->set_index_buf(index_buf);
      } else if (column_desc->columnType.get_compression() == kENCODING_TOKDICT) {
        switch (column_desc->columnType.get_elem_size()) {
          case 1: {
            StringTokDictEncoder<int8_t> *str_encoder = dynamic_cast<StringTokDictEncoder<int8_t>*>(buffer->encoder);
            str_encoder->set_index_buf(index_buf);
            break;
          }
          case 2: {
            StringTokDictEncoder<int16_t> *str_encoder = dynamic_cast<StringTokDictEncoder<int16_t>*>(buffer->encoder);
            str_encoder->set_index_buf(index_buf);
            break;
          }
          case 4: {
            StringTokDictEncoder<int32_t> *str_encoder = dynamic_cast<StringTokDictEncoder<int32_t>*>(buffer->encoder);
            str_encoder->set_index_buf(index_buf);
            break;
          }
          default:
            assert(false);
        }
      }
    } else
      buffer = data_mgr->getChunkBuffer(key, mem_level, device_id, num_bytes);
  }

  void
  Chunk::createChunkBuffer(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int device_id)
  {
    if (column_desc->columnType.is_varlen()) {
      ChunkKey subKey = key;
      subKey.push_back(1); // 1 for the main buffer
      buffer = data_mgr->createChunkBuffer(subKey, mem_level, device_id);
      subKey.pop_back();
      subKey.push_back(2); // 2 for the index buffer
      index_buf = data_mgr->createChunkBuffer(subKey, mem_level, device_id);
    } else
      buffer = data_mgr->createChunkBuffer(key, mem_level, device_id);
  }

  ChunkMetadata
  Chunk::appendData(DataBlockPtr &src_data, const size_t num_elems, const size_t start_idx)
  {
    if (column_desc->columnType.is_varlen()) {
      if (column_desc->columnType.get_compression() == kENCODING_NONE) {
        StringNoneEncoder *str_encoder = dynamic_cast<StringNoneEncoder*>(buffer->encoder);
        return str_encoder->appendData(src_data.stringsPtr, start_idx, num_elems);
      } else if (column_desc->columnType.get_compression() == kENCODING_TOKDICT) {
        switch (column_desc->columnType.get_elem_size()) {
          case 1: {
            StringTokDictEncoder<int8_t> *tokdict_encoder = dynamic_cast<StringTokDictEncoder<int8_t>*>(buffer->encoder);
            return tokdict_encoder->appendData(src_data.tok8dictPtr, start_idx, num_elems);
          }
          case 2: {
            StringTokDictEncoder<int16_t> *tokdict_encoder = dynamic_cast<StringTokDictEncoder<int16_t>*>(buffer->encoder);
            return tokdict_encoder->appendData(src_data.tok16dictPtr, start_idx, num_elems);
          }
          case 4: {
            StringTokDictEncoder<int32_t> *tokdict_encoder = dynamic_cast<StringTokDictEncoder<int32_t>*>(buffer->encoder);
            return tokdict_encoder->appendData(src_data.tok32dictPtr, start_idx, num_elems);
          }
          default:
            assert(false);
        }
      }
    }
    return buffer->encoder->appendData(src_data.numbersPtr, num_elems);
  }

  void
  Chunk::unpin_buffer()
  {
    if (buffer != nullptr)
      buffer->unPin();
    if (index_buf != nullptr)
      index_buf->unPin();
  }

  void
  Chunk::init_encoder()
  {
    buffer->initEncoder(column_desc->columnType);
    if (column_desc->columnType.is_varlen()) {
      if (column_desc->columnType.get_compression() == kENCODING_NONE) {
        StringNoneEncoder *str_encoder = dynamic_cast<StringNoneEncoder*>(buffer->encoder);
        str_encoder->set_index_buf(index_buf);
      } else if (column_desc->columnType.get_compression() == kENCODING_TOKDICT) {
        switch (column_desc->columnType.get_elem_size()) {
          case 1: {
            StringTokDictEncoder<int8_t> *str_encoder = dynamic_cast<StringTokDictEncoder<int8_t>*>(buffer->encoder);
            str_encoder->set_index_buf(index_buf);
            break;
          }
          case 2: {
            StringTokDictEncoder<int16_t> *str_encoder = dynamic_cast<StringTokDictEncoder<int16_t>*>(buffer->encoder);
            str_encoder->set_index_buf(index_buf);
            break;
          }
          case 4: {
            StringTokDictEncoder<int32_t> *str_encoder = dynamic_cast<StringTokDictEncoder<int32_t>*>(buffer->encoder);
            str_encoder->set_index_buf(index_buf);
            break;
          }
          default:
            assert(false);
        }
      }
    }
  }

  ChunkIter
  Chunk::begin_iterator(int start_idx, int skip) const
  {
    ChunkIter it;
    it.type_info = column_desc->columnType;
    it.skip = skip;
    it.skip_size = column_desc->columnType.get_size();
    if (it.skip_size < 0) { // if it's variable length
      it.current_pos = it.start_pos = index_buf->getMemoryPtr() + start_idx * sizeof(StringOffsetT);
      it.end_pos = index_buf->getMemoryPtr() + index_buf->size() - sizeof(StringOffsetT);;
      it.second_buf = buffer->getMemoryPtr();
    } else {
      it.current_pos = it.start_pos = buffer->getMemoryPtr() + start_idx * it.skip_size;
      it.end_pos = buffer->getMemoryPtr() + buffer->size();
      it.second_buf = nullptr;
    }
    ChunkMetadata chunkMetadata;
    buffer->encoder->getMetadata(chunkMetadata);
    it.num_elems = chunkMetadata.numElements;
    return it;
  }
}
