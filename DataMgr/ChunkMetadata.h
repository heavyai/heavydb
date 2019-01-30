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

#ifndef CHUNKMETADATA_H
#define CHUNKMETADATA_H

#include <stddef.h>
#include "../Shared/sqltypes.h"

#include <glog/logging.h>

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
        chunkStats.min.timeval = min;
        chunkStats.max.timeval = max;
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
      default: { break; }
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

#endif  // CHUNKMETADATA_H
