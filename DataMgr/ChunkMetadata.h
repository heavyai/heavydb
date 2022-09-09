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
#include "IR/Type.h"
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
                    const hdk::ir::Type* type,
                    const T min,
                    const T max,
                    const bool has_nulls) {
  stats.has_nulls = has_nulls;
  switch (type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
      switch (type->size()) {
        case 1:
          stats.min.tinyintval = min;
          stats.max.tinyintval = max;
          break;
        case 2:
          stats.min.smallintval = min;
          stats.max.smallintval = max;
          break;
        case 4:
          stats.min.intval = min;
          stats.max.intval = max;
          break;
        case 8:
          stats.min.bigintval = min;
          stats.max.bigintval = max;
          break;
        default:
          abort();
      }
      break;
    case hdk::ir::Type::kExtDictionary:
      stats.min.intval = min;
      stats.max.intval = max;
      break;
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kInterval:
      stats.min.bigintval = min;
      stats.max.bigintval = max;
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          stats.min.floatval = min;
          stats.max.floatval = max;
          break;
        case hdk::ir::FloatingPointType::kDouble:
          stats.min.doubleval = min;
          stats.max.doubleval = max;
          break;
        default:
          abort();
      }
      break;
    default:
      break;
  }
}

inline void mergeStats(ChunkStats& lhs,
                       const ChunkStats& rhs,
                       const hdk::ir::Type* type) {
  auto elem_type =
      type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type;
  lhs.has_nulls |= rhs.has_nulls;
  switch (elem_type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kDecimal:
      switch (elem_type->size()) {
        case 1:
          lhs.min.tinyintval = std::min(lhs.min.tinyintval, rhs.min.tinyintval);
          lhs.max.tinyintval = std::max(lhs.max.tinyintval, rhs.max.tinyintval);
          break;
        case 2:
          lhs.min.smallintval = std::min(lhs.min.smallintval, rhs.min.smallintval);
          lhs.max.smallintval = std::max(lhs.max.smallintval, rhs.max.smallintval);
          break;
        case 4:
          lhs.min.intval = std::min(lhs.min.intval, rhs.min.intval);
          lhs.max.intval = std::max(lhs.max.intval, rhs.max.intval);
          break;
        case 8:
          lhs.min.bigintval = std::min(lhs.min.bigintval, rhs.min.bigintval);
          lhs.max.bigintval = std::max(lhs.max.bigintval, rhs.max.bigintval);
          break;
        default:
          abort();
      }
      break;
    case hdk::ir::Type::kExtDictionary:
      lhs.min.intval = std::min(lhs.min.intval, rhs.min.intval);
      lhs.max.intval = std::max(lhs.max.intval, rhs.max.intval);
      break;
    case hdk::ir::Type::kDate:
    case hdk::ir::Type::kTime:
    case hdk::ir::Type::kTimestamp:
    case hdk::ir::Type::kInterval:
      lhs.min.bigintval = std::min(lhs.min.bigintval, rhs.min.bigintval);
      lhs.max.bigintval = std::max(lhs.max.bigintval, rhs.max.bigintval);
      break;
    case hdk::ir::Type::kFloatingPoint:
      switch (elem_type->as<hdk::ir::FloatingPointType>()->precision()) {
        case hdk::ir::FloatingPointType::kFloat:
          lhs.min.floatval = std::min(lhs.min.floatval, rhs.min.floatval);
          lhs.max.floatval = std::max(lhs.max.floatval, rhs.max.floatval);
          break;
        case hdk::ir::FloatingPointType::kDouble:
          lhs.min.doubleval = std::min(lhs.min.doubleval, rhs.min.doubleval);
          lhs.max.doubleval = std::max(lhs.max.doubleval, rhs.max.doubleval);
          break;
        default:
          abort();
      }
      break;
    default:
      break;
  }
}

struct ChunkMetadata {
  const hdk::ir::Type* type;
  size_t numBytes;
  size_t numElements;
  ChunkStats chunkStats;

#ifndef __CUDACC__
  std::string dump() const {
    auto elem_type =
        type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type;
    // Unencoded strings have no min/max.
    if (elem_type->isString()) {
      return "type: " + type->toString() + " numBytes: " + to_string(numBytes) +
             " numElements " + to_string(numElements) + " min: <invalid>" +
             " max: <invalid>" + " has_nulls: " + to_string(chunkStats.has_nulls);
    } else if (elem_type->isExtDictionary()) {
      return "type: " + type->toString() + " numBytes: " + to_string(numBytes) +
             " numElements " + to_string(numElements) +
             " min: " + to_string(chunkStats.min.intval) +
             " max: " + to_string(chunkStats.max.intval) +
             " has_nulls: " + to_string(chunkStats.has_nulls);
    } else {
      return "type: " + type->toString() + " numBytes: " + to_string(numBytes) +
             " numElements " + to_string(numElements) +
             " min: " + DatumToString(chunkStats.min, elem_type) +
             " max: " + DatumToString(chunkStats.max, elem_type) +
             " has_nulls: " + to_string(chunkStats.has_nulls);
    }
  }

  std::string toString() const { return dump(); }
#endif

  ChunkMetadata(const hdk::ir::Type* type_,
                const size_t num_bytes,
                const size_t num_elements,
                const ChunkStats& chunk_stats)
      : type(type_)
      , numBytes(num_bytes)
      , numElements(num_elements)
      , chunkStats(chunk_stats) {}

  ChunkMetadata() {}

  template <typename T>
  void fillChunkStats(const T min, const T max, const bool has_nulls) {
    ::fillChunkStats(chunkStats, type, min, max, has_nulls);
  }

  void fillChunkStats(const Datum min, const Datum max, const bool has_nulls) {
    chunkStats.has_nulls = has_nulls;
    chunkStats.min = min;
    chunkStats.max = max;
  }

  bool operator==(const ChunkMetadata& that) const {
    return type->equal(that.type) && numBytes == that.numBytes &&
           numElements == that.numElements &&
           DatumEqual(
               chunkStats.min,
               that.chunkStats.min,
               type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type) &&
           DatumEqual(
               chunkStats.max,
               that.chunkStats.max,
               type->isArray() ? type->as<hdk::ir::ArrayBaseType>()->elemType() : type) &&
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
