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

#include "Shared/Logger.h"

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

  std::string dump() {
    return "numBytes: " + to_string(numBytes) + " numElements " + to_string(numElements) +
           " min: " + DatumToString(chunkStats.min, sqlType) +
           " max: " + DatumToString(chunkStats.max, sqlType) +
           " has_nulls: " + to_string(chunkStats.has_nulls);
  }

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
           DatumEqual(chunkStats.min, that.chunkStats.min, sqlType) &&
           DatumEqual(chunkStats.max, that.chunkStats.max, sqlType) &&
           chunkStats.has_nulls == that.chunkStats.has_nulls;
  }
};

using ChunkMetadataMap = std::map<int, std::shared_ptr<ChunkMetadata>>;
using ChunkMetadataVector =
    std::vector<std::pair<ChunkKey, std::shared_ptr<ChunkMetadata>>>;
