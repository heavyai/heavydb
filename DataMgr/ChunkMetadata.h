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

#pragma once

#include <cstddef>
#include <iostream>

#include "Logger/Logger.h"
#include "Shared/StringTransform.h"
#include "Shared/sqltypes.h"
#include "Shared/types.h"

struct ChunkStats {
  Datum min;
  Datum max;
  bool has_nulls;
};

struct ChunkMetadata {
  SQLTypeInfo sqlType;
  size_t numBytes;
  size_t numElements;
  ChunkStats chunkStats;

  ChunkMetadata(const SQLTypeInfo& sql_type,
                const size_t num_bytes,
                const size_t num_elements,
                const ChunkStats& chunk_stats)
      : sqlType(sql_type)
      , numBytes(num_bytes)
      , numElements(num_elements)
      , chunkStats(chunk_stats) {}

  ChunkMetadata() {}

  template <typename T>
  void fillChunkStats(const T min, const T max, const bool has_nulls) {
    chunkStats.has_nulls = has_nulls;
    switch (sqlType.get_type()) {
      case kBOOLEAN: {
        chunkStats.min.tinyintval = min;
        chunkStats.max.tinyintval = max;
        break;
      }
      case kTINYINT: {
        chunkStats.min.tinyintval = min;
        chunkStats.max.tinyintval = max;
        break;
      }
      case kSMALLINT: {
        chunkStats.min.smallintval = min;
        chunkStats.max.smallintval = max;
        break;
      }
      case kINT: {
        chunkStats.min.intval = min;
        chunkStats.max.intval = max;
        break;
      }
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL: {
        chunkStats.min.bigintval = min;
        chunkStats.max.bigintval = max;
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        chunkStats.min.bigintval = min;
        chunkStats.max.bigintval = max;
        break;
      }
      case kFLOAT: {
        chunkStats.min.floatval = min;
        chunkStats.max.floatval = max;
        break;
      }
      case kDOUBLE: {
        chunkStats.min.doubleval = min;
        chunkStats.max.doubleval = max;
        break;
      }
      case kVARCHAR:
      case kCHAR:
      case kTEXT:
        if (sqlType.get_compression() == kENCODING_DICT) {
          chunkStats.min.intval = min;
          chunkStats.max.intval = max;
        }
        break;
      default: {
        break;
      }
    }
  }

  void fillChunkStats(const Datum min, const Datum max, const bool has_nulls) {
    chunkStats.has_nulls = has_nulls;
    chunkStats.min = min;
    chunkStats.max = max;
  }

  bool operator==(const ChunkMetadata& that) const {
    return sqlType == that.sqlType && numBytes == that.numBytes &&
           numElements == that.numElements &&
           DatumEqual(chunkStats.min,
                      that.chunkStats.min,
                      sqlType.is_array() ? sqlType.get_elem_type() : sqlType) &&
           DatumEqual(chunkStats.max,
                      that.chunkStats.max,
                      sqlType.is_array() ? sqlType.get_elem_type() : sqlType) &&
           chunkStats.has_nulls == that.chunkStats.has_nulls;
  }

  bool isPlaceholder() const {
    // Currently needed because a lot of our Datum operations (in this case
    // extract_int_type_from_datum()) are not safe for all types.
    const auto type =
        sqlType.is_decimal() ? decimal_to_int_type(sqlType) : sqlType.get_type();
    switch (type) {
      case kCHAR:
      case kVARCHAR:
      case kTEXT:
        if (sqlType.get_compression() != kENCODING_DICT) {
          return false;
        }
      case kBOOLEAN:
      case kTINYINT:
      case kSMALLINT:
      case kINT:
      case kBIGINT:
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        auto min = extract_int_type_from_datum(chunkStats.min, sqlType);
        auto max = extract_int_type_from_datum(chunkStats.max, sqlType);
        return (numElements > 0 && !chunkStats.has_nulls && (min > max));
      }
      default:
        return false;
    }
    return false;
  }
};

inline std::ostream& operator<<(std::ostream& out, const ChunkMetadata& chunk_metadata) {
  auto type = chunk_metadata.sqlType.is_array() ? chunk_metadata.sqlType.get_elem_type()
                                                : chunk_metadata.sqlType;
  // Unencoded strings have no min/max.
  std::string min, max;
  if (type.is_string() && type.get_compression() == kENCODING_NONE) {
    min = "<invalid>";
    max = "<invalid>";
  } else if (type.is_string()) {
    min = to_string(chunk_metadata.chunkStats.min.intval);
    max = to_string(chunk_metadata.chunkStats.max.intval);
  } else {
    min = DatumToString(chunk_metadata.chunkStats.min, type);
    max = DatumToString(chunk_metadata.chunkStats.max, type);
  }
  out << "type: " << chunk_metadata.sqlType.get_type_name()
      << " numBytes: " << chunk_metadata.numBytes << " numElements "
      << chunk_metadata.numElements << " min: " << min << " max: " << max
      << " has_nulls: " << std::to_string(chunk_metadata.chunkStats.has_nulls);
  return out;
}

inline int64_t extract_min_stat_int_type(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_int_type_from_datum(stats.min, ti);
}

inline int64_t extract_max_stat_int_type(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_int_type_from_datum(stats.max, ti);
}

inline double extract_min_stat_fp_type(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_fp_type_from_datum(stats.min, ti);
}

inline double extract_max_stat_fp_type(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_fp_type_from_datum(stats.max, ti);
}

using ChunkMetadataMap = std::map<int, std::shared_ptr<ChunkMetadata>>;
using ChunkMetadataVector =
    std::vector<std::pair<ChunkKey, std::shared_ptr<ChunkMetadata>>>;
