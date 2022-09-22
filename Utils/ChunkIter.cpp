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
 * @file ChunkIter.cpp
 * @author Wei Hong <wei@mapd.com>
 */

#include "ChunkIter.h"

#include "IR/Type.h"

#include <cstdlib>

HOST DEVICE inline bool isNull(const int8_t* val, const ChunkIter* it) {
  if (it->type_id == hdk::ir::Type::kFloatingPoint) {
    switch (it->type_size) {
      case 4:
        return *(float*)val == inline_null_value<float>();
      case 8:
        return *(double*)val == inline_null_value<double>();
      default:
        assert(false);
    }
  }
  switch (it->type_size) {
    case 0:
      // For Null type.
      return true;
    case 1:
      return *val == inline_null_value<int8_t>();
    case 2:
      return *(int16_t*)val == inline_null_value<int16_t>();
    case 4:
      return *(int32_t*)val == inline_null_value<int32_t>();
    case 8:
      return *(int64_t*)val == inline_null_value<int64_t>();
    default:
      break;
  }
  return false;
}

HOST DEVICE inline bool isNullFixedLenArray(const int8_t* val,
                                            int array_size,
                                            const ChunkIter* it) {
  // Check if fixed length array has a NULL_ARRAY sentinel as the first element
  if (it->type_id == hdk::ir::Type::kFixedLenArray && val && array_size > 0 &&
      array_size == it->type_size) {
    if (it->elem_type_id == hdk::ir::Type::kFloatingPoint) {
      switch (it->elem_type_size) {
        case 4:
          return *(float*)val == inline_null_array_value<float>();
        case 8:
          return *(double*)val == inline_null_array_value<double>();
        default:
          assert(false);
      }
    }
    switch (it->elem_type_size) {
      case 1:
        return *val == inline_null_array_value<int8_t>();
      case 2:
        return *(int16_t*)val == inline_null_array_value<int16_t>();
      case 4:
        return *(int32_t*)val == inline_null_array_value<int32_t>();
      case 8:
        return *(int64_t*)val == inline_null_array_value<int64_t>();
      default:
        return false;
    }
  }
  return false;
}

DEVICE static bool needDecompression(const ChunkIter* it) {
  if ((it->type_id == hdk::ir::Type::kDate || it->type_id == hdk::ir::Type::kTimestamp ||
       it->type_id == hdk::ir::Type::kTime) &&
      it->type_size != 8) {
    return true;
  }
  return false;
}

DEVICE static void decompress(const ChunkIter* it,
                              int8_t* compressed,
                              VarlenDatum* result,
                              Datum* datum) {
  switch (it->type_id) {
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp: {
      result->length = sizeof(int64_t);
      result->pointer = (int8_t*)&datum->bigintval;
      switch (it->type_size) {
        case 1:
          datum->bigintval = (int64_t) * (int8_t*)compressed;
          break;
        case 2:
          datum->bigintval = (int64_t) * (int16_t*)compressed;
          break;
        case 4:
          datum->bigintval = (int64_t) * (int32_t*)compressed;
          break;
        default:
          assert(false);
      }
      result->is_null = datum->bigintval == inline_null_value<int64_t>();
    } break;
    default:
      assert(false);
  }
}

void ChunkIter_reset(ChunkIter* it) {
  it->current_pos = it->start_pos;
}

DEVICE void ChunkIter_get_next(ChunkIter* it,
                               bool uncompress,
                               VarlenDatum* result,
                               bool* is_end) {
  if (it->current_pos >= it->end_pos) {
    *is_end = true;
    result->length = 0;
    result->pointer = NULL;
    result->is_null = true;
    return;
  }
  *is_end = false;

  if (it->skip_size > 0) {
    // for fixed-size
    if (uncompress && needDecompression(it)) {
      decompress(it, it->current_pos, result, &it->datum);
    } else {
      result->length = static_cast<size_t>(it->skip_size);
      result->pointer = it->current_pos;
      result->is_null = isNull(result->pointer, it);
    }
    it->current_pos += it->skip * it->skip_size;
  } else {
    StringOffsetT offset = *(StringOffsetT*)it->current_pos;
    result->length = static_cast<size_t>(*((StringOffsetT*)it->current_pos + 1) - offset);
    result->pointer = it->second_buf + offset;
    // @TODO(wei) treat zero length as null for now
    result->is_null = (result->length == 0);
    it->current_pos += it->skip * sizeof(StringOffsetT);
  }
}

// @brief get nth element in Chunk.  Does not change ChunkIter state
DEVICE void ChunkIter_get_nth(ChunkIter* it,
                              int n,
                              bool uncompress,
                              VarlenDatum* result,
                              bool* is_end) {
  if (static_cast<size_t>(n) >= it->num_elems || n < 0) {
    *is_end = true;
    result->length = 0;
    result->pointer = NULL;
    result->is_null = true;
    return;
  }
  *is_end = false;

  if (it->skip_size > 0) {
    // for fixed-size
    int8_t* current_pos = it->start_pos + n * it->skip_size;
    if (uncompress && needDecompression(it)) {
      decompress(it, current_pos, result, &it->datum);
    } else {
      result->length = static_cast<size_t>(it->skip_size);
      result->pointer = current_pos;
      result->is_null = isNull(result->pointer, it);
    }
  } else {
    int8_t* current_pos = it->start_pos + n * sizeof(StringOffsetT);
    StringOffsetT offset = *(StringOffsetT*)current_pos;
    result->length = static_cast<size_t>(*((StringOffsetT*)current_pos + 1) - offset);
    result->pointer = it->second_buf + offset;
    // @TODO(wei) treat zero length as null for now
    result->is_null = (result->length == 0);
  }
}

// @brief get nth element in Chunk.  Does not change ChunkIter state
DEVICE void ChunkIter_get_nth(ChunkIter* it, int n, ArrayDatum* result, bool* is_end) {
  if (static_cast<size_t>(n) >= it->num_elems || n < 0) {
    *is_end = true;
    result->length = 0;
    result->pointer = NULL;
    result->is_null = true;
    return;
  }
  *is_end = false;

  if (it->skip_size > 0) {
    // for fixed-size
    int8_t* current_pos = it->start_pos + n * it->skip_size;
    result->length = static_cast<size_t>(it->skip_size);
    result->pointer = current_pos;
    bool is_null = false;
    if (it->type_nullable) {
      // Nulls can only be recognized when iterating over a nullable-typed chunk
      is_null = isNullFixedLenArray(result->pointer, result->length, it);
    }
    result->is_null = is_null;
  } else {
    int8_t* current_pos = it->start_pos + n * sizeof(ArrayOffsetT);
    int8_t* next_pos = current_pos + sizeof(ArrayOffsetT);
    ArrayOffsetT offset = *(ArrayOffsetT*)current_pos;
    ArrayOffsetT next_offset = *(ArrayOffsetT*)next_pos;
    if (next_offset < 0) {  // Encoded NULL array
      result->length = 0;
      result->pointer = NULL;
      result->is_null = true;
    } else {
      if (offset < 0) {
        offset = -offset;  // Previous array may have been NULL, remove negativity
      }
      result->length = static_cast<size_t>(next_offset - offset);
      result->pointer = it->second_buf + offset;
      result->is_null = false;
    }
  }
}

// @brief get nth varlen array element in Chunk.  Does not change ChunkIter state
DEVICE void ChunkIter_get_nth_varlen(ChunkIter* it,
                                     int n,
                                     ArrayDatum* result,
                                     bool* is_end) {
  *is_end = (static_cast<size_t>(n) >= it->num_elems || n < 0);

  if (!*is_end) {
    int8_t* current_pos = it->start_pos + n * sizeof(ArrayOffsetT);
    int8_t* next_pos = current_pos + sizeof(ArrayOffsetT);
    ArrayOffsetT offset = *(ArrayOffsetT*)current_pos;
    ArrayOffsetT next_offset = *(ArrayOffsetT*)next_pos;

    if (next_offset >= 0) {
      // Previous array may have been NULL, remove offset negativity
      if (offset < 0) {
        offset = -offset;
      }
      result->length = static_cast<size_t>(next_offset - offset);
      result->pointer = it->second_buf + offset;
      result->is_null = false;
      return;
    }
  }
  // Encoded NULL array or out of bounds
  result->length = 0;
  result->pointer = NULL;
  result->is_null = true;
}

// @brief get nth varlen notnull array element in Chunk.  Does not change ChunkIter state
DEVICE void ChunkIter_get_nth_varlen_notnull(ChunkIter* it,
                                             int n,
                                             ArrayDatum* result,
                                             bool* is_end) {
  *is_end = (static_cast<size_t>(n) >= it->num_elems || n < 0);

  int8_t* current_pos = it->start_pos + n * sizeof(ArrayOffsetT);
  int8_t* next_pos = current_pos + sizeof(ArrayOffsetT);
  ArrayOffsetT offset = *(ArrayOffsetT*)current_pos;
  ArrayOffsetT next_offset = *(ArrayOffsetT*)next_pos;

  result->length = static_cast<size_t>(next_offset - offset);
  result->pointer = it->second_buf + offset;
  result->is_null = false;
}
