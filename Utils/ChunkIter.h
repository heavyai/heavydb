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
 * @file ChunkIter.h
 * @author Wei Hong <wei@mapd.com>
 *
 */
#ifndef _CHUNK_ITER_H_
#define _CHUNK_ITER_H_

#include "../Shared/funcannotations.h"
#include "../Shared/sqltypes.h"

class ChunkIter {
 public:
  SQLTypeInfo type_info;
  int8_t* second_buf;
  int8_t* current_pos;
  int8_t* start_pos;
  int8_t* end_pos;
  int skip;
  int skip_size;
  size_t num_elems;
  Datum datum;  // used to hold uncompressed value
};

void ChunkIter_reset(ChunkIter* it);
DEVICE void ChunkIter_get_next(ChunkIter* it, bool uncompress, VarlenDatum* vd, bool* is_end);
// @brief get nth element in Chunk.  Does not change ChunkIter state
DEVICE void ChunkIter_get_nth(ChunkIter* it, int nth, bool uncompress, VarlenDatum* vd, bool* is_end);
DEVICE void ChunkIter_get_nth(ChunkIter* it, int nth, ArrayDatum* vd, bool* is_end);

#endif  // _CHUNK_ITER_H_
