/**
 * @file		StringNoneEncoder.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		For unencoded strings
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "StringNoneEncoder.h"

using Data_Namespace::AbstractBuffer;

ChunkMetadata
StringNoneEncoder::appendData(const std::vector<std::string> *srcData, const int start_idx, const size_t numAppendElems)
{
	assert(index_buf != nullptr); // index_buf must be set before this.
	size_t index_size = numAppendElems * sizeof(int64_t);
	if (numElems == 0)
		index_size += sizeof(int64_t); // plus one for the initial offset of 0.
	index_buf->reserve(index_size);
	// @TODO worry about locking 
	int64_t *index = (int64_t*)(index_buf->getMemoryPtr() + index_buf->size());
	int i = 0;
	if (numElems == 0) {
		index[0] = 0; // write the inital 0 offset
		i = 1;
	}
	size_t data_size = 0;
	for (int n = start_idx, c = 0; c < numAppendElems; n++, c++, i++) {
		size_t len = (*srcData)[n].length();
		index[i] = index[i - 1] + len;
		data_size += len;
	}
	index_buf->setSize(index_buf->size() + index_size);
	index_buf->setAppended();
	buffer_->reserve(data_size);
	for (int n = start_idx, c = 0, i = 0 ; c < numAppendElems; n++, c++, i++) {
		size_t len = (*srcData)[n].length();
		char *dest = (char*)(buffer_->getMemoryPtr() + index[i]);
		(*srcData)[n].copy(dest, len);
	}
	buffer_->setSize(buffer_->size() + data_size);
	buffer_->setAppended();

	numElems += numAppendElems;
	ChunkMetadata chunkMetadata;
	getMetadata(chunkMetadata);
	return chunkMetadata;
}
