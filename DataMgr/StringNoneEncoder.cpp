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
 * @file		StringNoneEncoder.cpp
 * @brief		For unencoded strings
 *
 */

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

size_t StringNoneEncoder::getNumElemsForBytesEncodedDataAtIndices(
    const int8_t* index_data,
    const std::vector<size_t>& selected_idx,
    const size_t byte_limit) {
  size_t num_elements = 0;
  size_t data_size = 0;
  for (const auto& offset_index : selected_idx) {
    auto element_size = getStringSizeAtIndex(index_data, offset_index);
    if (data_size + element_size > byte_limit) {
      break;
    }
    data_size += element_size;
    num_elements++;
  }
  return num_elements;
}

std::shared_ptr<ChunkMetadata> StringNoneEncoder::appendEncodedDataAtIndices(
    const int8_t* index_data,
    int8_t* data,
    const std::vector<size_t>& selected_idx) {
  std::vector<std::string_view> data_subset;
  data_subset.reserve(selected_idx.size());
  for (const auto& offset_index : selected_idx) {
    data_subset.emplace_back(getStringAtIndex(index_data, data, offset_index));
  }
  return appendData(&data_subset, 0, selected_idx.size(), false);
}

std::shared_ptr<ChunkMetadata> StringNoneEncoder::appendEncodedData(
    const int8_t* index_data,
    int8_t* data,
    const size_t start_idx,
    const size_t num_elements) {
  std::vector<std::string_view> data_subset;
  data_subset.reserve(num_elements);
  for (size_t count = 0; count < num_elements; ++count) {
    auto current_index = start_idx + count;
    data_subset.emplace_back(getStringAtIndex(index_data, data, current_index));
  }
  return appendData(&data_subset, 0, num_elements, false);
}

template <typename StringType>
std::shared_ptr<ChunkMetadata> StringNoneEncoder::appendData(
    const std::vector<StringType>* srcData,
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
      }
      update_elem_stats((*srcData)[replicating ? 0 : i]);
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

void StringNoneEncoder::updateStats(const std::vector<std::string>* const src_data,
                                    const size_t start_idx,
                                    const size_t num_elements) {
  for (size_t n = start_idx; n < start_idx + num_elements; n++) {
    update_elem_stats((*src_data)[n]);
    if (has_nulls) {
      break;
    }
  }
}

template <typename StringType>
void StringNoneEncoder::update_elem_stats(const StringType& elem) {
  if (!has_nulls && elem.empty()) {
    has_nulls = true;
  }
}

std::pair<StringOffsetT, StringOffsetT> StringNoneEncoder::getStringOffsets(
    const int8_t* index_data,
    size_t index) {
  auto string_offsets = reinterpret_cast<const StringOffsetT*>(index_data);
  auto current_index = index + 1;
  auto offset = string_offsets[current_index];
  CHECK(offset >= 0);
  int64_t last_offset = string_offsets[current_index - 1];
  CHECK(last_offset >= 0 && last_offset <= offset);
  return {offset, last_offset};
}

size_t StringNoneEncoder::getStringSizeAtIndex(const int8_t* index_data, size_t index) {
  auto [offset, last_offset] = getStringOffsets(index_data, index);
  size_t string_byte_size = offset - last_offset;
  return string_byte_size;
}

std::string_view StringNoneEncoder::getStringAtIndex(const int8_t* index_data,
                                                     const int8_t* data,
                                                     size_t index) {
  auto [offset, last_offset] = getStringOffsets(index_data, index);
  size_t string_byte_size = offset - last_offset;
  auto current_data = reinterpret_cast<const char*>(data + last_offset);
  return std::string_view{current_data, string_byte_size};
}

template std::shared_ptr<ChunkMetadata> StringNoneEncoder::appendData<std::string>(
    const std::vector<std::string>* srcData,
    const int start_idx,
    const size_t numAppendElems,
    const bool replicating);

template std::shared_ptr<ChunkMetadata> StringNoneEncoder::appendData<std::string_view>(
    const std::vector<std::string_view>* srcData,
    const int start_idx,
    const size_t numAppendElems,
    const bool replicating);

template void StringNoneEncoder::update_elem_stats<std::string>(const std::string& elem);
template void StringNoneEncoder::update_elem_stats<std::string_view>(
    const std::string_view& elem);
