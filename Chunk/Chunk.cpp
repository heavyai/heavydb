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
		// @TODO calculate start position
		return ChunkIter(*this, 0, start_idx, skip);
	}

	VarlenDatum
	ChunkIter::get_next(bool uncompress, bool &is_end)
	{
		is_end = true;
		return VarlenDatum();
	}

	Datum
	ChunkIter::get_next_value(bool &is_null, bool &is_end)
	{
		is_null = true;
		is_end = true;
		Datum d;
		d.bigintval = 0;
		return d;
	}
}
