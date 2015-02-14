/**
 * @file		StringNoneEncoder.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		For unencoded strings
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <algorithm>
#include <cstdlib>
#include <memory>
#include "StringNoneEncoder.h"

using Data_Namespace::AbstractBuffer;

// default max input buffer size to 1MB
#define MAX_INPUT_BUF_SIZE		1048576

ChunkMetadata
StringNoneEncoder::appendData(const std::vector<std::string> *srcData, const int start_idx, const size_t numAppendElems)
{
	assert(index_buf != nullptr); // index_buf must be set before this.
	size_t index_size = numAppendElems * sizeof(StringOffsetT);
	if (numElems == 0)
		index_size += sizeof(StringOffsetT); // plus one for the initial offset of 0.
	index_buf->reserve(index_size);
	StringOffsetT offset = 0;
	if (numElems == 0) {
		index_buf->append((int8_t*)&offset, sizeof(StringOffsetT));  // write the inital 0 offset
		last_offset = 0;
	} else {
		if (last_offset < 0) {
			// need to read the last offset from buffer/disk
			index_buf->read((int8_t*)&last_offset, sizeof(StringOffsetT), Data_Namespace::CPU_BUFFER, index_buf->size() - sizeof(StringOffsetT));
			assert(last_offset >= 0);
		}
	}
	size_t data_size = 0;
	for (int n = start_idx; n < start_idx + numAppendElems; n++) {
		size_t len = (*srcData)[n].length();
		data_size += len;
	}
	buffer_->reserve(data_size);

	size_t inbuf_size = std::min(std::max(index_size, data_size), (size_t)MAX_INPUT_BUF_SIZE);
	int8_t *inbuf = (int8_t*)malloc(inbuf_size);
	std::unique_ptr<int8_t> gc_inbuf(inbuf);
	for (size_t num_appended = 0; num_appended < numAppendElems; ) {
		StringOffsetT *p = (StringOffsetT*)inbuf;
		int i;
		for (i = 0; num_appended < numAppendElems && i < inbuf_size/sizeof(StringOffsetT); i++, num_appended++) {
			p[i] = last_offset + (*srcData)[num_appended + start_idx].length();
			last_offset = p[i];
		}
		index_buf->append(inbuf, i * sizeof(StringOffsetT));
	}

	for (size_t num_appended = 0; num_appended < numAppendElems; ) {
		size_t size = 0;
		for (int i = start_idx + num_appended; num_appended < numAppendElems && size < inbuf_size;  i++, num_appended++) {
			size_t len = (*srcData)[i].length();
			if (len > inbuf_size) {
				// for large strings, append on its own
				if (size > 0)
					buffer_->append(inbuf, size);
				size = 0;
				buffer_->append((int8_t*)(*srcData)[i].data(), len);
				num_appended++;
				break;
			} else if (size + len > inbuf_size)
				break;
			char *dest = (char*)inbuf + size;
			(*srcData)[i].copy(dest, len);
			size += len;
		}
		if (size > 0)
			buffer_->append(inbuf, size);
	}

	numElems += numAppendElems;
	ChunkMetadata chunkMetadata;
	getMetadata(chunkMetadata);
	return chunkMetadata;
}
