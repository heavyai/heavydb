/*
 * Copyright 2020 OmniSci, Inc.
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

#include "DummyForeignStorage.h"

void DummyPersistentForeignStorage::append(
    const std::vector<ForeignStorageColumnBuffer>& column_buffers) {
  std::lock_guard<std::mutex> files_lock(files_mutex_);
  for (const auto& column_buffer : column_buffers) {
    append(column_buffer.chunk_key,
           column_buffer.sql_type,
           &column_buffer.buff[0],
           column_buffer.buff.size());
  }
}

void DummyPersistentForeignStorage::read(const ChunkKey& chunk_key,
                                         const SQLTypeInfo& sql_type,
                                         int8_t* dest,
                                         const size_t numBytes) {
  std::lock_guard<std::mutex> files_lock(files_mutex_);
  const auto it = files_.find(chunk_key);
  CHECK(it != files_.end());
  const auto& src = it->second;
  CHECK_EQ(numBytes, src.size());
  memcpy(dest, &src[0], numBytes);
}

std::string DummyPersistentForeignStorage::getType() const {
  return "DUMMY";
}

void DummyPersistentForeignStorage::append(const ChunkKey& chunk_key,
                                           const SQLTypeInfo& sql_type,
                                           const int8_t* src,
                                           const size_t numBytes) {
  files_[chunk_key].insert(files_[chunk_key].end(), src, src + numBytes);
}
