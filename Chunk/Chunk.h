/**
 * @file Chunk.h
 * @author Wei Hong <wei@mapd.com>
 *
 */
#ifndef _CHUNK_H_
#define _CHUNK_H_

#include <list>
#include "../Shared/sqltypes.h"
#include "../DataMgr/AbstractBuffer.h"
#include "../DataMgr/ChunkMetadata.h"
#include "../DataMgr/DataMgr.h"
#include "../Catalog/ColumnDescriptor.h"

using Data_Namespace::AbstractBuffer;
using Data_Namespace::DataMgr;
using Data_Namespace::MemoryLevel;

namespace Chunk_NS {
	class Chunk;
};

struct ChunkIter {
		const Chunk_NS::Chunk *chunk;
		int8_t *current_pos;
		int8_t *start_pos;
		int8_t *end_pos;
		int skip;
		int skip_size;
		Datum datum; // used to hold uncompressed value
};

void ChunkIter_reset(ChunkIter *it);
VarlenDatum ChunkIter_get_next(ChunkIter *it, bool uncompress, bool *is_end);


namespace Chunk_NS {

	class Chunk {
		public:
			Chunk() : buffer(nullptr), index_buf(nullptr), column_desc(nullptr) {}
			Chunk(const ColumnDescriptor *td) : buffer(nullptr), index_buf(nullptr), column_desc(td) {}
			Chunk(AbstractBuffer *b, AbstractBuffer *ib, const ColumnDescriptor *td) : buffer(b), index_buf(ib), column_desc(td) {};
			const ColumnDescriptor *get_column_desc() const { return column_desc; }
			static void translateColumnDescriptorsToChunkVec(const std::list<const ColumnDescriptor*> &colDescs, std::vector<Chunk> &chunkVec) {
				for (auto cd : colDescs)
					chunkVec.push_back(Chunk(cd));
			}
			ChunkIter begin_iterator(int start_idx, int skip) const;
			ChunkMetadata appendData(int8_t *&srcData, const size_t numAppendElems);
			void createChunkBuffer(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int deviceId = 0);
			void getChunkBuffer(DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int deviceId = 0, const size_t num_bytes = 0);
			static Chunk getChunk(const ColumnDescriptor *cd, DataMgr *data_mgr, const ChunkKey &key, const MemoryLevel mem_level, const int deviceId, const size_t num_bytes);

		// protected:
			AbstractBuffer *get_buffer() const { return buffer; }
			AbstractBuffer *get_index_buf() const { return index_buf; }
			void set_buffer(AbstractBuffer *b) { buffer = b; }
			void set_index_buf(AbstractBuffer *ib) { index_buf = ib; }
			void unpin_buffer();
			void pin_buffer();
			void init_encoder();
		private:
			AbstractBuffer *buffer;
			AbstractBuffer *index_buf;
			const ColumnDescriptor *column_desc;
	};

}

#endif // _CHUNK_H_
