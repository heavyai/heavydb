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

#include <parquet/schema.h>
#include <parquet/types.h>
#include "GeospatialEncoder.h"
#include "ParquetEncoder.h"

namespace foreign_storage {

class ParquetGeospatialEncoder : public ParquetEncoder, public GeospatialEncoder {
 public:
  ParquetGeospatialEncoder(const bool geo_validate_geometry)
      : ParquetEncoder(nullptr), GeospatialEncoder(geo_validate_geometry) {}

  ParquetGeospatialEncoder(std::list<Chunk_NS::Chunk>& chunks,
                           std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
                           const bool geo_validate_geometry)
      : ParquetEncoder(nullptr)
      , GeospatialEncoder(chunks, chunk_metadata, geo_validate_geometry) {}

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  int8_t* values) override {
    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);

    clearDatumBuffers();

    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      clearParseBuffers();
      if (def_levels[i] == 0) {
        if (is_error_tracking_enabled_ && column_type_.get_notnull()) {
          // mark as invalid due to a null in a NOT NULL column
          invalid_indices_.insert(current_chunk_offset_ + i);
        }
        processNullGeoElement();
      } else {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        auto geo_string_view = std::string_view{
            reinterpret_cast<const char*>(byte_array.ptr), byte_array.len};
        try {
          processGeoElement(geo_string_view);
        } catch (const std::runtime_error& error) {
          if (is_error_tracking_enabled_) {
            invalid_indices_.insert(current_chunk_offset_ + i);
            /// add null if failed
            clearParseBuffers();
            processNullGeoElement();
          } else {
            throw;
          }
        }
      }
    }

    appendArrayDatumsToBufferAndUpdateMetadata();

    appendBaseDataAndUpdateMetadata(levels_read);

    if (is_error_tracking_enabled_) {
      current_chunk_offset_ += levels_read;
    }
  }

  void appendDataTrackErrors(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values) override {
    CHECK(is_error_tracking_enabled_);
    // `appendData` modifies its behaviour based on the
    // `is_error_tracking_enabled_` flag to handle this case
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }
};

template <class T>
class ParquetLatLonGeospatialEncoder : public ParquetGeospatialEncoder {
 public:
  ParquetLatLonGeospatialEncoder(const bool geo_validate_geometry)
      : ParquetGeospatialEncoder(geo_validate_geometry) {}

  ParquetLatLonGeospatialEncoder(
      std::list<Chunk_NS::Chunk>& chunks,
      std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
      const bool geo_validate_geometry)
      : ParquetGeospatialEncoder(chunks, chunk_metadata, geo_validate_geometry) {}

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  int8_t* values) override {
    UNREACHABLE()
        << "ParquetLatLonGeospatialEncoder requires multiple inputs for compression";
  }

  void appendDataTrackErrors(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values) override {
    UNREACHABLE()
        << "ParquetLatLonGeospatialEncoder requires multiple inputs for compression";
  }

  void appendData(const std::vector<int16_t*>& def_levels,
                  const std::vector<int16_t*>& rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  std::vector<int8_t*>& coords) override {
    CHECK_EQ(def_levels.size(), 2U);
    CHECK_EQ(rep_levels.size(), 2U);
    CHECK_EQ(coords.size(), 2U);
    clearDatumBuffers();

    for (int64_t i = 0; i < levels_read; ++i) {
      clearParseBuffers();
      if (def_levels[0][i] == 0 || def_levels[1][i] == 0) {
        // Check if either lon or lat values are null.
        if (is_error_tracking_enabled_ && column_type_.get_notnull()) {
          // mark as invalid due to a null in a NOT NULL column
          invalid_indices_.insert(current_chunk_offset_ + i);
        }
        processNullGeoElement();
      } else {
        std::vector<double> uncompressed_coords = {reinterpret_cast<T*>(coords[0])[i],
                                                   reinterpret_cast<T*>(coords[1])[i]};
        std::vector<uint8_t> compressed_coords = Geospatial::compress_coords(
            uncompressed_coords, geo_column_descriptor_->columnType);
        coords_datum_buffer_.emplace_back(encode_as_array_datum(compressed_coords));
      }
    }

    appendArrayDatumsToBufferAndUpdateMetadata();

    appendBaseDataAndUpdateMetadata(levels_read);

    if (is_error_tracking_enabled_) {
      current_chunk_offset_ += levels_read;
    }
  }

  void appendDataTrackErrors(const std::vector<int16_t*>& def_levels,
                             const std::vector<int16_t*>& rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             std::vector<int8_t*>& values) override {
    CHECK(is_error_tracking_enabled_);
    // `appendData` modifies its behaviour based on the
    // `is_error_tracking_enabled_` flag to handle this case
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }
};

}  // namespace foreign_storage
