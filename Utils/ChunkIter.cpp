/*
 * @file ChunkIter.cpp
 * @author Wei Hong <wei@mapd.com>
 */

#include "ChunkIter.h"

DEVICE static void
decompress(const SQLTypeInfo &ti, int8_t *compressed, VarlenDatum *result, Datum *datum)
{
  result->is_null = false;
  switch (ti.get_type()) {
    case kSMALLINT:
      result->length = sizeof(int16_t);
      result->pointer = (int8_t*)&datum->smallintval;
      switch (ti.get_compression()) {
        case kENCODING_FIXED:
          assert(ti.get_comp_param() == 8);
          datum->smallintval = (int16_t)*(int8_t*)compressed;
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
              datum->intval = (int32_t)*(int8_t*)compressed;
              break;
            case 16:
              datum->intval = (int32_t)*(int16_t*)compressed;
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
              datum->bigintval = (int64_t)*(int8_t*)compressed;
              break;
            case 16:
              datum->bigintval = (int64_t)*(int16_t*)compressed;
              break;
            case 32:
              datum->bigintval = (int64_t)*(int32_t*)compressed;
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
        result->length = sizeof(time_t);
        result->pointer = (int8_t*)&datum->timeval;
        switch (ti.get_compression()) {
          case kENCODING_FIXED:
          case kENCODING_RL:
          case kENCODING_DIFF:
          case kENCODING_DICT:
          case kENCODING_TOKDICT:
          case kENCODING_SPARSE:
          case kENCODING_NONE:
            assert(false);
            break;
          default:
            assert(false);
        }
        break;
    default:
      assert(false);
  }
}

void
ChunkIter_reset(ChunkIter *it)
{
  it->current_pos = it->start_pos;
}

DEVICE void
ChunkIter_get_next(ChunkIter *it, bool uncompress, VarlenDatum *result, bool *is_end)
{
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
      result->is_null = false;
    }
    it->current_pos += it->skip * it->skip_size;
  } else {
    // @TODO(wei) ignore uncompress flag for variable length?
    StringOffsetT offset = *(StringOffsetT*)it->current_pos;
    result->length = *((StringOffsetT*)it->current_pos + 1) - offset;
    result->pointer = it->second_buf + offset;
    result->is_null = false;
    it->current_pos += it->skip * sizeof(StringOffsetT);
  }
}

// @brief get nth element in Chunk.  Does not change ChunkIter state
DEVICE void
ChunkIter_get_nth(ChunkIter *it, int n, bool uncompress, VarlenDatum *result, bool *is_end)
{
  if (n >= it->num_elems || n < 0) {
    *is_end = true;
    result->length = 0;
    result->pointer = NULL;
    result->is_null = true;
    return;
  }
  *is_end = false;
    
  if (it->skip_size > 0) {
    // for fixed-size
    int8_t *current_pos = it->start_pos + n * it->skip_size;
    if (uncompress && it->type_info.get_compression() != kENCODING_NONE) {
      decompress(it->type_info, current_pos, result, &it->datum);
    } else {
      result->length = it->skip_size;
      result->pointer = current_pos;
      result->is_null = false;
    }
  } else {
    // @TODO(wei) ignore uncompress flag for variable length?
    int8_t *current_pos = it->start_pos + n * sizeof(StringOffsetT);
    StringOffsetT offset = *(StringOffsetT*)current_pos;
    result->length = *((StringOffsetT*)current_pos + 1) - offset;
    result->pointer = it->second_buf + offset;
    result->is_null = false;
  }
}
