/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @file		StringNoneEncoder.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		For unencoded strings
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <algorithm>
#include <cstdlib>
#include <memory>
#include "MemoryLevel.h"
#include "StringNoneEncoder.h"

using Data_Namespace::AbstractBuffer;

size_t StringNoneEncoder::getNumElemsForBytesInsertData(const std::vector<std::string>* srcData,
                                                        const int start_idx,
                                                        const size_t numAppendElems,
                                                        const size_t byteLimit) {
  size_t dataSize = 0;
  size_t n = start_idx;
  for (; n < start_idx + numAppendElems; n++) {
    size_t len = (*srcData)[n].length();
    if (dataSize + len > byteLimit)
      break;
    dataSize += len;
  }
  return n - start_idx;
}

ChunkMetadata StringNoneEncoder::appendData(const std::vector<std::string>* srcData,
                                            const int start_idx,
                                            const size_t numAppendElems) {
  assert(index_buf != nullptr);  // index_buf must be set before this.
  size_t index_size = numAppendElems * sizeof(StringOffsetT);
  if (numElems == 0)
    index_size += sizeof(StringOffsetT);  // plus one for the initial offset of 0.
  index_buf->reserve(index_size);
  StringOffsetT offset = 0;
  if (numElems == 0) {
    index_buf->append((int8_t*)&offset, sizeof(StringOffsetT));  // write the inital 0 offset
    last_offset = 0;
  } else {
    if (last_offset < 0) {
      // need to read the last offset from buffer/disk
      index_buf->read((int8_t*)&last_offset,
                      sizeof(StringOffsetT),
                      index_buf->size() - sizeof(StringOffsetT),
                      Data_Namespace::CPU_LEVEL);
      assert(last_offset >= 0);
    }
  }
  size_t data_size = 0;
  for (size_t n = start_idx; n < start_idx + numAppendElems; n++) {
    size_t len = (*srcData)[n].length();
    data_size += len;
  }
  buffer_->reserve(data_size);

  size_t inbuf_size = std::min(std::max(index_size, data_size), (size_t)MAX_INPUT_BUF_SIZE);
  auto inbuf = new int8_t[inbuf_size];
  std::unique_ptr<int8_t[]> gc_inbuf(inbuf);
  for (size_t num_appended = 0; num_appended < numAppendElems;) {
    StringOffsetT* p = (StringOffsetT*)inbuf;
    size_t i;
    for (i = 0; num_appended < numAppendElems && i < inbuf_size / sizeof(StringOffsetT); i++, num_appended++) {
      p[i] = last_offset + (*srcData)[num_appended + start_idx].length();
      last_offset = p[i];
    }
    index_buf->append(inbuf, i * sizeof(StringOffsetT));
  }

  for (size_t num_appended = 0; num_appended < numAppendElems;) {
    size_t size = 0;
    for (int i = start_idx + num_appended; num_appended < numAppendElems && size < inbuf_size; i++, num_appended++) {
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
      char* dest = (char*)inbuf + size;
      if (len > 0) {
        (*srcData)[i].copy(dest, len);
        size += len;
      } else
        has_nulls = true;
    }
    if (size > 0)
      buffer_->append(inbuf, size);
  }
  // make sure buffer_ is flushed even if no new data is appended to it
  // (e.g. empty strings) because the metadata needs to be flushed.
  if (!buffer_->isDirty())
    buffer_->setDirty();

  numElems += numAppendElems;
  ChunkMetadata chunkMetadata;
  getMetadata(chunkMetadata);
  return chunkMetadata;
}
