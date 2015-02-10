/*
 * @file Chunk.cpp
 * @author Wei Hong <wei@mapd.com>
 */

#include "Chunk.h"

namespace Chunk_NS {
	Chunk
	Chunk::getChunk(const ColumnDescriptor *cd, DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel memoryLevel, const int deviceId, const size_t numBytes) {
			Chunk chunk(cd);
			chunk.getChunkBuffer(data_mgr, key, memoryLevel, deviceId, numBytes);
			return chunk;
		}

	void 
	Chunk::getChunkBuffer(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int device_id, const size_t num_bytes)
	{
		// @TODO add logic here to handle string case
		buffer = data_mgr->getChunkBuffer(key, mem_level, device_id, num_bytes);
	}

	void
	Chunk::createChunkBuffer(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int device_id)
	{
		buffer = data_mgr->createChunkBuffer(key, mem_level, device_id);
	}

	ChunkMetadata
	Chunk::appendData(int8_t *&src_data, const size_t num_elems)
	{
		return buffer->encoder->appendData(src_data, num_elems);
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
	Chunk::pin_buffer()
	{
		buffer->pin();
		if (index_buf != nullptr)
			index_buf->pin();
	}

	void
	Chunk::init_encoder()
	{
		buffer->initEncoder(column_desc->columnType, column_desc->compression, column_desc->comp_param);
	}

	ChunkIter
	Chunk::begin_iterator(int start_idx, int skip) const
	{
		ChunkIter it;
		it.chunk = this;
		it.skip = skip;
		it.skip_size = column_desc->getStorageSize();
		if (it.skip_size < 0) { // if it's variable length
			it.current_pos = it.start_pos = index_buf->getMemoryPtr() + start_idx * sizeof(int32_t);
			it.end_pos = index_buf->getMemoryPtr() + index_buf->size();
		} else {
			it.current_pos = it.start_pos = buffer->getMemoryPtr() + start_idx * it.skip_size;
			it.end_pos = buffer->getMemoryPtr() + buffer->size();
		}
		return it;
	}

	void
	ChunkIter_reset(ChunkIter *it)
	{
		it->current_pos = it->start_pos;
	}

	void
	ChunkIter_get_next(ChunkIter *it, bool uncompress, VarlenDatum *result, bool *is_end)
	{
		if (it->current_pos >= it->end_pos) {
			*is_end = true;
			result->length = 0;
			result->pointer = nullptr;
			result->is_null = true;
			return;
		}
		*is_end = false;
			
		if (it->skip_size > 0) {
			// for fixed-size
			if (uncompress && it->chunk->get_column_desc()->compression != kENCODING_NONE) {
				assert(false);
			} else {
				result->length = it->skip_size;
				result->pointer = it->current_pos;
				result->is_null = false;
			}
			it->current_pos += it->skip * it->skip_size;
		} else {
			// @TODO(wei) ignore uncompress flag for variable length?
			int offset = *(int32_t*)it->current_pos;
			result->length = *((int32_t*)it->current_pos + 1) - offset;
			result->pointer = it->chunk->get_buffer()->getMemoryPtr() + offset;
			result->is_null = false;
			it->current_pos += it->skip * sizeof(int32_t);
		}
	}

}
