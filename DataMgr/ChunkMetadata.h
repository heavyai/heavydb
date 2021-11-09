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

#pragma once

#include <cstddef>
#include "../Shared/sqltypes.h"
#include "Shared/types.h"

#include <map>

#include "Logger/Logger.h"

struct ChunkStats {
  Datum min;
  Datum max;
  bool has_nulls;
};

template <typename T>
void fillChunkStats(ChunkStats& stats,
                    const SQLTypeInfo& type,
                    const T min,
                    const T max,
                    const bool has_nulls) {
  stats.has_nulls = has_nulls;
  switch (type.get_type()) {
    case kBOOLEAN: {
      stats.min.tinyintval = min;
      stats.max.tinyintval = max;
      break;
    }
    case kTINYINT: {
      stats.min.tinyintval = min;
      stats.max.tinyintval = max;
      break;
    }
    case kSMALLINT: {
      stats.min.smallintval = min;
      stats.max.smallintval = max;
      break;
    }
    case kINT: {
      stats.min.intval = min;
      stats.max.intval = max;
      break;
    }
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL: {
      stats.min.bigintval = min;
      stats.max.bigintval = max;
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      stats.min.bigintval = min;
      stats.max.bigintval = max;
      break;
    }
    case kFLOAT: {
      stats.min.floatval = min;
      stats.max.floatval = max;
      break;
    }
    case kDOUBLE: {
      stats.min.doubleval = min;
      stats.max.doubleval = max;
      break;
    }
    case kVARCHAR:
    case kCHAR:
    case kTEXT:
      if (type.get_compression() == kENCODING_DICT) {
        stats.min.intval = min;
        stats.max.intval = max;
      }
      break;
    default: {
      break;
    }
  }
}

inline void mergeStats(ChunkStats& lhs, const ChunkStats& rhs, const SQLTypeInfo& type) {
  lhs.has_nulls |= rhs.has_nulls;
  switch (type.is_array() ? type.get_subtype() : type.get_type()) {
    case kBOOLEAN: {
      lhs.min.tinyintval = std::min(lhs.min.tinyintval, rhs.min.tinyintval);
      lhs.max.tinyintval = std::max(lhs.max.tinyintval, rhs.max.tinyintval);
      break;
    }
    case kTINYINT: {
      lhs.min.tinyintval = std::min(lhs.min.tinyintval, rhs.min.tinyintval);
      lhs.max.tinyintval = std::max(lhs.max.tinyintval, rhs.max.tinyintval);
      break;
    }
    case kSMALLINT: {
      lhs.min.smallintval = std::min(lhs.min.smallintval, rhs.min.smallintval);
      lhs.max.smallintval = std::max(lhs.max.smallintval, rhs.max.smallintval);
      break;
    }
    case kINT: {
      lhs.min.intval = std::min(lhs.min.intval, rhs.min.intval);
      lhs.max.intval = std::max(lhs.max.intval, rhs.max.intval);
      break;
    }
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL: {
      lhs.min.bigintval = std::min(lhs.min.bigintval, rhs.min.bigintval);
      lhs.max.bigintval = std::max(lhs.max.bigintval, rhs.max.bigintval);
      break;
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      lhs.min.bigintval = std::min(lhs.min.bigintval, rhs.min.bigintval);
      lhs.max.bigintval = std::max(lhs.max.bigintval, rhs.max.bigintval);
      break;
    }
    case kFLOAT: {
      lhs.min.floatval = std::min(lhs.min.floatval, rhs.min.floatval);
      lhs.max.floatval = std::max(lhs.max.floatval, rhs.max.floatval);
      break;
    }
    case kDOUBLE: {
      lhs.min.doubleval = std::min(lhs.min.doubleval, rhs.min.doubleval);
      lhs.max.doubleval = std::max(lhs.max.doubleval, rhs.max.doubleval);
      break;
    }
    case kVARCHAR:
    case kCHAR:
    case kTEXT:
      if (type.get_compression() == kENCODING_DICT) {
        lhs.min.intval = std::min(lhs.min.intval, rhs.min.intval);
        lhs.max.intval = std::max(lhs.max.intval, rhs.max.intval);
      }
      break;
    default: {
      break;
    }
  }
}

struct ChunkMetadata {
  SQLTypeInfo sqlType;
  size_t numBytes;
  size_t numElements;
  ChunkStats chunkStats;

#ifndef __CUDACC__
  std::string dump() const {
    auto type = sqlType.is_array() ? sqlType.get_elem_type() : sqlType;
    // Unencoded strings have no min/max.
    if (type.is_string() && type.get_compression() == kENCODING_NONE) {
      return "type: " + sqlType.get_type_name() + " numBytes: " + to_string(numBytes) +
             " numElements " + to_string(numElements) + " min: <invalid>" +
             " max: <invalid>" + " has_nulls: " + to_string(chunkStats.has_nulls);
    } else if (type.is_string()) {
      return "type: " + sqlType.get_type_name() + " numBytes: " + to_string(numBytes) +
             " numElements " + to_string(numElements) +
             " min: " + to_string(chunkStats.min.intval) +
             " max: " + to_string(chunkStats.max.intval) +
             " has_nulls: " + to_string(chunkStats.has_nulls);
    } else {
      return "type: " + sqlType.get_type_name() + " numBytes: " + to_string(numBytes) +
             " numElements " + to_string(numElements) +
             " min: " + DatumToString(chunkStats.min, type) +
             " max: " + DatumToString(chunkStats.max, type) +
             " has_nulls: " + to_string(chunkStats.has_nulls);
    }
  }

  std::string toString() const { return dump(); }
#endif

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
    ::fillChunkStats(chunkStats, sqlType, min, max, has_nulls);
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
};

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
