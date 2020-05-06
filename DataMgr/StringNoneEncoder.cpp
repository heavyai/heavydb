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

#include "StringNoneEncoder.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include "MemoryLevel.h"

using Data_Namespace::AbstractBuffer;

size_t StringNoneEncoder::getNumElemsForBytesInsertData(
    const std::vector<std::string>* srcData,
    const int start_idx,
    const size_t numAppendElems,
    const size_t byteLimit,
    const bool replicating) {
  size_t dataSize = 0;
  size_t n = start_idx;
  for (; n < start_idx + numAppendElems; n++) {
    size_t len = (*srcData)[replicating ? 0 : n].length();
    if (dataSize + len > byteLimit) {
      break;
    }
    dataSize += len;
  }
  return n - start_idx;
}

std::shared_ptr<ChunkMetadata> StringNoneEncoder::appendData(
    const std::vector<std::string>* srcData,
    const int start_idx,
    const size_t numAppendElems,
    const bool replicating) {
  CHECK(index_buf);  // index_buf must be set before this.
  size_t index_size = numAppendElems * sizeof(StringOffsetT);
  if (num_elems_ == 0) {
    index_size += sizeof(StringOffsetT);  // plus one for the initial offset of 0.
  }
  index_buf->reserve(index_size);
  StringOffsetT offset = 0;
  if (num_elems_ == 0) {
    index_buf->append((int8_t*)&offset,
                      sizeof(StringOffsetT));  // write the inital 0 offset
    last_offset = 0;
  } else {
    // always need to read a valid last offset from buffer/disk
    // b/c now due to vacuum "last offset" may go backward and if
    // index chunk was not reloaded last_offset would go way off!
    index_buf->read((int8_t*)&last_offset,
                    sizeof(StringOffsetT),
                    index_buf->size() - sizeof(StringOffsetT),
                    Data_Namespace::CPU_LEVEL);
    CHECK_GE(last_offset, 0);
  }
  size_t data_size = 0;
  for (size_t n = start_idx; n < start_idx + numAppendElems; n++) {
    size_t len = (*srcData)[replicating ? 0 : n].length();
    data_size += len;
  }
  buffer_->reserve(data_size);

  size_t inbuf_size =
      std::min(std::max(index_size, data_size), (size_t)MAX_INPUT_BUF_SIZE);
  auto inbuf = std::make_unique<int8_t[]>(inbuf_size);
  for (size_t num_appended = 0; num_appended < numAppendElems;) {
    StringOffsetT* p = reinterpret_cast<StringOffsetT*>(inbuf.get());
    size_t i;
    for (i = 0; num_appended < numAppendElems && i < inbuf_size / sizeof(StringOffsetT);
         i++, num_appended++) {
      p[i] =
          last_offset + (*srcData)[replicating ? 0 : num_appended + start_idx].length();
      last_offset = p[i];
    }
    index_buf->append(inbuf.get(), i * sizeof(StringOffsetT));
  }

  for (size_t num_appended = 0; num_appended < numAppendElems;) {
    size_t size = 0;
    for (int i = start_idx + num_appended;
         num_appended < numAppendElems && size < inbuf_size;
         i++, num_appended++) {
      size_t len = (*srcData)[replicating ? 0 : i].length();
      if (len > inbuf_size) {
        // for large strings, append on its own
        if (size > 0) {
          buffer_->append(inbuf.get(), size);
        }
        size = 0;
        buffer_->append((int8_t*)(*srcData)[replicating ? 0 : i].data(), len);
        num_appended++;
        break;
      } else if (size + len > inbuf_size) {
        break;
      }
      char* dest = reinterpret_cast<char*>(inbuf.get()) + size;
      if (len > 0) {
        (*srcData)[replicating ? 0 : i].copy(dest, len);
        size += len;
      } else {
        has_nulls = true;
      }
    }
    if (size > 0) {
      buffer_->append(inbuf.get(), size);
    }
  }
  // make sure buffer_ is flushed even if no new data is appended to it
  // (e.g. empty strings) because the metadata needs to be flushed.
  if (!buffer_->isDirty()) {
    buffer_->setDirty();
  }

  num_elems_ += numAppendElems;
  auto chunk_metadata = std::make_shared<ChunkMetadata>();
  getMetadata(chunk_metadata);
  return chunk_metadata;
}
