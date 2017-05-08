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

DEVICE static void decompress(const SQLTypeInfo& ti, int8_t* compressed, VarlenDatum* result, Datum* datum) {
  switch (ti.get_type()) {
    case kSMALLINT:
      result->length = sizeof(int16_t);
      result->pointer = (int8_t*)&datum->smallintval;
      switch (ti.get_compression()) {
        case kENCODING_FIXED:
          assert(ti.get_comp_param() == 8);
          datum->smallintval = (int16_t) * (int8_t*)compressed;
          break;
        case kENCODING_RL:
        case kENCODING_DIFF:
        case kENCODING_SPARSE:
          assert(false);
          break;
        default:
          assert(false);
      }
      break;
    case kINT:
      result->length = sizeof(int32_t);
      result->pointer = (int8_t*)&datum->intval;
      switch (ti.get_compression()) {
        case kENCODING_FIXED:
          switch (ti.get_comp_param()) {
            case 8:
              datum->intval = (int32_t) * (int8_t*)compressed;
              break;
            case 16:
              datum->intval = (int32_t) * (int16_t*)compressed;
              break;
            default:
              assert(false);
          }
          break;
        case kENCODING_RL:
        case kENCODING_DIFF:
        case kENCODING_SPARSE:
          assert(false);
          break;
        default:
          assert(false);
      }
      break;
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
      result->length = sizeof(int64_t);
      result->pointer = (int8_t*)&datum->bigintval;
      switch (ti.get_compression()) {
        case kENCODING_FIXED:
          switch (ti.get_comp_param()) {
            case 8:
              datum->bigintval = (int64_t) * (int8_t*)compressed;
              break;
            case 16:
              datum->bigintval = (int64_t) * (int16_t*)compressed;
              break;
            case 32:
              datum->bigintval = (int64_t) * (int32_t*)compressed;
              break;
            default:
              assert(false);
          }
          break;
        case kENCODING_RL:
        case kENCODING_DIFF:
        case kENCODING_SPARSE:
          assert(false);
          break;
        default:
          assert(false);
      }
      break;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      switch (ti.get_compression()) {
        case kENCODING_FIXED:
          datum->timeval = (time_t) * (int32_t*)compressed;
          break;
        case kENCODING_RL:
        case kENCODING_DIFF:
        case kENCODING_DICT:
        case kENCODING_SPARSE:
        case kENCODING_NONE:
          assert(false);
          break;
        default:
          assert(false);
      }
      result->length = sizeof(time_t);
      result->pointer = (int8_t*)&datum->timeval;
      break;
    default:
      assert(false);
  }
  result->is_null = ti.is_null(*datum);
}

void ChunkIter_reset(ChunkIter* it) {
  it->current_pos = it->start_pos;
}

DEVICE void ChunkIter_get_next(ChunkIter* it, bool uncompress, VarlenDatum* result, bool* is_end) {
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
    if (uncompress && it->type_info.get_compression() != kENCODING_NONE) {
      decompress(it->type_info, it->current_pos, result, &it->datum);
    } else {
      result->length = it->skip_size;
      result->pointer = it->current_pos;
      result->is_null = it->type_info.is_null(result->pointer);
    }
    it->current_pos += it->skip * it->skip_size;
  } else {
    StringOffsetT offset = *(StringOffsetT*)it->current_pos;
    result->length = *((StringOffsetT*)it->current_pos + 1) - offset;
    result->pointer = it->second_buf + offset;
    // @TODO(wei) treat zero length as null for now
    result->is_null = (result->length == 0);
    it->current_pos += it->skip * sizeof(StringOffsetT);
  }
}

// @brief get nth element in Chunk.  Does not change ChunkIter state
DEVICE void ChunkIter_get_nth(ChunkIter* it, int n, bool uncompress, VarlenDatum* result, bool* is_end) {
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
    if (uncompress && it->type_info.get_compression() != kENCODING_NONE) {
      decompress(it->type_info, current_pos, result, &it->datum);
    } else {
      result->length = it->skip_size;
      result->pointer = current_pos;
      result->is_null = it->type_info.is_null(result->pointer);
    }
  } else {
    int8_t* current_pos = it->start_pos + n * sizeof(StringOffsetT);
    StringOffsetT offset = *(StringOffsetT*)current_pos;
    result->length = *((StringOffsetT*)current_pos + 1) - offset;
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
    result->length = it->skip_size;
    result->pointer = current_pos;
    result->is_null = it->type_info.is_null(result->pointer);
  } else {
    int8_t* current_pos = it->start_pos + n * sizeof(StringOffsetT);
    StringOffsetT offset = *(StringOffsetT*)current_pos;
    result->length = *((StringOffsetT*)current_pos + 1) - offset;
    result->pointer = it->second_buf + offset;
    // @TODO(wei) treat zero length as null for now
    result->is_null = (result->length == 0);
  }
}
