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

/*
 * File:        types.h
 * Author(s):   steve@map-d.com
 *
 * Created on June 19, 2014, 4:29 PM
 */

#ifndef _TYPES_H
#define _TYPES_H

#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "Logger/Logger.h"

// The ChunkKey is a unique identifier for chunks in the database file.
// The first element of the underlying vector for ChunkKey indicates the type of
// ChunkKey (also referred to as the keyspace id)
using ChunkKey = std::vector<int>;

#define CHUNK_KEY_DB_IDX 0
#define CHUNK_KEY_TABLE_IDX 1
#define CHUNK_KEY_COLUMN_IDX 2
#define CHUNK_KEY_FRAGMENT_IDX 3
#define CHUNK_KEY_VARLEN_IDX 4

inline bool is_table_key(const ChunkKey& key) {
  return key.size() == 2;
}

inline bool has_table_prefix(const ChunkKey& key) {
  return key.size() >= 2;
}

inline ChunkKey get_table_key(const ChunkKey& key) {
  CHECK(has_table_prefix(key));
  return ChunkKey{key[CHUNK_KEY_DB_IDX], key[CHUNK_KEY_TABLE_IDX]};
}

inline std::pair<int, int> get_table_prefix(const ChunkKey& key) {
  CHECK(has_table_prefix(key));
  return std::pair<int, int>{key[CHUNK_KEY_DB_IDX], key[CHUNK_KEY_TABLE_IDX]};
}

inline bool is_column_key(const ChunkKey& key) {
  return key.size() == 3;
}

inline bool is_varlen_key(const ChunkKey& key) {
  return key.size() == 5;
}

inline bool is_varlen_data_key(const ChunkKey& key) {
  return key.size() == 5 && key[4] == 1;
}

inline bool is_varlen_index_key(const ChunkKey& key) {
  return key.size() == 5 && key[4] == 2;
}

inline bool in_same_table(const ChunkKey& left_key, const ChunkKey& right_key) {
  CHECK(has_table_prefix(left_key));
  CHECK(has_table_prefix(right_key));
  return (left_key[CHUNK_KEY_DB_IDX] == right_key[CHUNK_KEY_DB_IDX] &&
          left_key[CHUNK_KEY_TABLE_IDX] == right_key[CHUNK_KEY_TABLE_IDX]);
}

inline std::string show_chunk(const ChunkKey& key) {
  std::ostringstream tss;
  for (auto vecIt = key.begin(); vecIt != key.end(); ++vecIt) {
    tss << *vecIt << ",";
  }
  return tss.str();
}

#endif /* _TYPES_H */
