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

// A way to represent a Tile's position relative to other tiles in the original file.
struct FileLocalCoords {
  int32_t file_id = -1, x = -1, y = -1;

  bool operator==(const FileLocalCoords& o) const {
    return file_id == o.file_id && x == o.x && y == o.y;
  }
};

inline std::ostream& operator<<(std::ostream& out, const FileLocalCoords& flc) {
  out << flc.file_id << ": " << flc.x << ", " << flc.y;
  return out;
}

struct RasterTileInfo {
  int32_t width = 0, height = 0;
  FileLocalCoords local_coords;

  bool operator==(const RasterTileInfo& o) const {
    return width == o.width && height == o.height && local_coords == o.local_coords;
  }
};

inline std::ostream& operator<<(std::ostream& out, const RasterTileInfo& tile) {
  out << "width: " << tile.width << ", height: " << tile.height << " LocalCoords: {"
      << tile.local_coords << "}";
  return out;
}

struct ChunkStats {
  Datum min;
  Datum max;
  bool has_nulls;

  ChunkStats() {}

  ChunkStats(const Datum new_min, const Datum new_max, const bool new_has_nulls)
      : min(new_min), max(new_max), has_nulls(new_has_nulls) {}

  template <typename T>
  ChunkStats(const T new_min,
             const T new_max,
             const bool new_has_nulls,
             const SQLTypeInfo& ti) {
    has_nulls = new_has_nulls;
    switch (ti.get_type()) {
      case kBOOLEAN: {
        min.tinyintval = new_min;
        max.tinyintval = new_max;
        break;
      }
      case kTINYINT: {
        min.tinyintval = new_min;
        max.tinyintval = new_max;
        break;
      }
      case kSMALLINT: {
        min.smallintval = new_min;
        max.smallintval = new_max;
        break;
      }
      case kINT: {
        min.intval = new_min;
        max.intval = new_max;
        break;
      }
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL: {
        min.bigintval = new_min;
        max.bigintval = new_max;
        break;
      }
      case kTIME:
      case kTIMESTAMP:
      case kDATE: {
        min.bigintval = new_min;
        max.bigintval = new_max;
        break;
      }
      case kFLOAT: {
        min.floatval = new_min;
        max.floatval = new_max;
        break;
      }
      case kDOUBLE: {
        min.doubleval = new_min;
        max.doubleval = new_max;
        break;
      }
      case kVARCHAR:
      case kCHAR:
      case kTEXT:
        if (ti.get_compression() == kENCODING_DICT) {
          min.intval = new_min;
          max.intval = new_max;
        }
        break;
      default: {
        UNREACHABLE() << "Unknown type";
        break;
      }
    }
  }
};

struct ChunkMetadata {
  SQLTypeInfo sqlType;
  size_t numBytes;
  size_t numElements;
  ChunkStats chunkStats;
  RasterTileInfo rasterTile;

  ChunkMetadata(const SQLTypeInfo& sql_type,
                const size_t num_bytes,
                const size_t num_elements,
                const ChunkStats& chunk_stats,
                const RasterTileInfo& raster_tile)
      : sqlType(sql_type)
      , numBytes(num_bytes)
      , numElements(num_elements)
      , chunkStats(chunk_stats)
      , rasterTile(raster_tile) {}

  ChunkMetadata(const SQLTypeInfo& sql_type,
                const size_t num_bytes,
                const size_t num_elements)
      : sqlType(sql_type), numBytes(num_bytes), numElements(num_elements) {}

  ChunkMetadata() {}

  template <typename T>
  void fillChunkStats(const T min, const T max, const bool has_nulls) {
    chunkStats = ChunkStats(min, max, has_nulls, sqlType);
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
           chunkStats.has_nulls == that.chunkStats.has_nulls &&
           rasterTile == that.rasterTile;
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
  } else if (type.get_type() == kPOINT) {
    min = "<NULL>";
    max = "<NULL>";
  } else {
    min = DatumToString(chunk_metadata.chunkStats.min, type);
    max = DatumToString(chunk_metadata.chunkStats.max, type);
  }
  out << "type: " << chunk_metadata.sqlType.toString()
      << ", numBytes: " << chunk_metadata.numBytes << ", numElements "
      << chunk_metadata.numElements << ", min: " << min << ", max: " << max
      << ", has_nulls: " << std::to_string(chunk_metadata.chunkStats.has_nulls)
      << ", rasterTile: (" << chunk_metadata.rasterTile << ")";
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

inline std::ostream& operator<<(std::ostream& os, const ChunkMetadataVector& meta_vec) {
  os << "chunk metadata vector:\n";
  for (const auto& chunk_meta : meta_vec) {
    const auto& [key, meta] = chunk_meta;
    os << "{" << show_chunk(key) << "}: " << *meta << "\n";
  }
  return os;
}
