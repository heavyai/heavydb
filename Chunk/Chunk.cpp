/*
 * @file Chunk.cpp
 * @author Wei Hong <wei@mapd.com>
 */

#include "Chunk.h"
#include "../DataMgr/StringNoneEncoder.h"

namespace Chunk_NS {
	Chunk
	Chunk::getChunk(const ColumnDescriptor *cd, DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel memoryLevel, const int deviceId, const size_t numBytes, const size_t numElems) {
			Chunk chunk(cd);
			chunk.getChunkBuffer(data_mgr, key, memoryLevel, deviceId, numBytes, numElems);
			return chunk;
		}

	void 
	Chunk::getChunkBuffer(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int device_id, const size_t num_bytes, const size_t num_elems)
	{
		if (column_desc->is_varlen()) {
			ChunkKey subKey = key;
			subKey.push_back(1); // 1 for the main buffer
			buffer = data_mgr->getChunkBuffer(subKey, mem_level, device_id, num_bytes);
			subKey.pop_back();
			subKey.push_back(2); // 2 for the index buffer
			index_buf = data_mgr->getChunkBuffer(subKey, mem_level, device_id, (num_elems + 1) * sizeof(StringOffsetT)); // always record n+1 offsets so string length can be calculated
		} else
			buffer = data_mgr->getChunkBuffer(key, mem_level, device_id, num_bytes);
	}

	void
	Chunk::createChunkBuffer(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int device_id)
	{
		if (column_desc->is_varlen()) {
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
		if (column_desc->is_varlen()) {
			StringNoneEncoder *str_encoder = dynamic_cast<StringNoneEncoder*>(buffer->encoder);
			return str_encoder->appendData(src_data.stringsPtr, start_idx, num_elems);

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
		if (column_desc->is_varlen()) {
			StringNoneEncoder *str_encoder = dynamic_cast<StringNoneEncoder*>(buffer->encoder);
			str_encoder->set_index_buf(index_buf);
		}
	}

	ChunkIter
	Chunk::begin_iterator(int start_idx, int skip) const
	{
		ChunkIter it;
		it.chunk = this;
		it.skip = skip;
		it.skip_size = column_desc->getStorageSize();
		if (it.skip_size < 0) { // if it's variable length
			it.current_pos = it.start_pos = index_buf->getMemoryPtr() + start_idx * sizeof(StringOffsetT);
			it.end_pos = index_buf->getMemoryPtr() + index_buf->size() - sizeof(StringOffsetT);;
		} else {
			it.current_pos = it.start_pos = buffer->getMemoryPtr() + start_idx * it.skip_size;
			it.end_pos = buffer->getMemoryPtr() + buffer->size();
		}
		return it;
	}

	void
	Chunk::decompress(int8_t *compressed, VarlenDatum *result, Datum *datum) const
	{
		result->is_null = false;
		switch (column_desc->columnType.type) {
			case kSMALLINT:
				result->length = sizeof(int16_t);
				result->pointer = (int8_t*)&datum->smallintval;
				switch (column_desc->compression) {
					case kENCODING_FIXED:
						assert(column_desc->comp_param == 8);
						datum->smallintval = (int16_t)*(int8_t*)compressed;
						break;
					case kENCODING_RL:
					case kENCODING_DIFF:
					case kENCODING_DICT:
					case kENCODING_SPARSE:
					case kENCODING_NONE:
						assert(false);
					break;
				}
				break;
			case kINT:
				result->length = sizeof(int32_t);
				result->pointer = (int8_t*)&datum->intval;
				switch (column_desc->compression) {
					case kENCODING_FIXED:
						switch (column_desc->comp_param) {
							case 8:
								datum->intval = (int32_t)*(int8_t*)compressed;
								break;
							case 16:
								datum->intval = (int32_t)*(int16_t*)compressed;
								break;
							default:
								assert(false);
						}
						break;
					case kENCODING_RL:
					case kENCODING_DIFF:
					case kENCODING_DICT:
					case kENCODING_SPARSE:
					case kENCODING_NONE:
						assert(false);
					break;
				}
				break;
			case kBIGINT:
			case kNUMERIC:
			case kDECIMAL:
				result->length = sizeof(int64_t);
				result->pointer = (int8_t*)&datum->bigintval;
				switch (column_desc->compression) {
					case kENCODING_FIXED:
						switch (column_desc->comp_param) {
							case 8:
								datum->bigintval = (int64_t)*(int8_t*)compressed;
								break;
							case 16:
								datum->bigintval = (int64_t)*(int16_t*)compressed;
								break;
							case 32:
								datum->bigintval = (int64_t)*(int32_t*)compressed;
								break;
							default:
								assert(false);
						}
						break;
					case kENCODING_RL:
					case kENCODING_DIFF:
					case kENCODING_DICT:
					case kENCODING_SPARSE:
					case kENCODING_NONE:
						assert(false);
					break;
				}
				break;
			default:
				assert(false);
		}
	}
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
			it->chunk->decompress(it->current_pos, result, &it->datum);
		} else {
			result->length = it->skip_size;
			result->pointer = it->current_pos;
			result->is_null = false;
		}
		it->current_pos += it->skip * it->skip_size;
	} else {
		// @TODO(wei) ignore uncompress flag for variable length?
		StringOffsetT offset = *(StringOffsetT*)it->current_pos;
		result->length = *((StringOffsetT*)it->current_pos + 1) - offset;
		result->pointer = it->chunk->get_buffer()->getMemoryPtr() + offset;
		result->is_null = false;
		it->current_pos += it->skip * sizeof(StringOffsetT);
	}
}

