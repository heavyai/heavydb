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

#include "DataMgr/Chunk/Chunk.h"
#include "Geospatial/Compression.h"
#include "Geospatial/Types.h"

#include "ParquetEncoder.h"

#include "ImportExport/Importer.h"
#include "LazyParquetChunkLoader.h"

#include "DataMgr/ArrayNoneEncoder.h"
#include "DataMgr/FixedLengthArrayNoneEncoder.h"
#include "DataMgr/StringNoneEncoder.h"

#include <parquet/schema.h>
#include <parquet/types.h>

namespace foreign_storage {

template <typename T>
inline ArrayDatum encode_as_array_datum(const std::vector<T>& data) {
  const size_t num_bytes = data.size() * sizeof(T);
  std::shared_ptr<int8_t> buffer(new int8_t[num_bytes], std::default_delete<int8_t[]>());
  memcpy(buffer.get(), data.data(), num_bytes);
  return ArrayDatum(num_bytes, buffer, false);
}

class ParquetGeospatialEncoder : public ParquetEncoder {
 public:
  ParquetGeospatialEncoder() : ParquetEncoder(nullptr) {}

  ParquetGeospatialEncoder(const parquet::ColumnDescriptor* parquet_column_descriptor,
                           std::list<Chunk_NS::Chunk>& chunks,
                           std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata)
      : ParquetEncoder(nullptr)
      , geo_column_descriptor_(chunks.begin()->getColumnDesc())
      , base_column_encoder_(nullptr)
      , coords_column_encoder_(nullptr)
      , bounds_column_encoder_(nullptr)
      , ring_sizes_column_encoder_(nullptr)
      , poly_rings_column_encoder_(nullptr)
      , render_group_column_encoder_(nullptr)
      , base_column_metadata_(nullptr)
      , coords_column_metadata_(nullptr)
      , bounds_column_metadata_(nullptr)
      , ring_sizes_column_metadata_(nullptr)
      , poly_rings_column_metadata_(nullptr)
      , render_group_column_metadata_(nullptr)
      , render_group_values_(LazyParquetChunkLoader::batch_reader_num_elements, 0)
      , base_values_(LazyParquetChunkLoader::batch_reader_num_elements)
      , row_count_(0)
      , parquet_column_name_(parquet_column_descriptor->name()) {
    CHECK(geo_column_descriptor_->columnType.is_geometry());

    validateChunksAndMetadataSizing(chunks, chunk_metadata);

    const auto geo_column_type = geo_column_descriptor_->columnType.get_type();
    std::shared_ptr<ParquetScalarEncoder> null_scalar_encoder;

    // initialize base column encoder
    auto base_chunk = chunks.begin();
    base_chunk->initEncoder();
    base_column_encoder_ =
        dynamic_cast<StringNoneEncoder*>(base_chunk->getBuffer()->getEncoder());
    base_column_metadata_ = chunk_metadata.begin()->get();
    CHECK(base_column_encoder_);

    // initialize coords column
    std::tie(coords_column_encoder_, coords_column_metadata_, coords_column_descriptor_) =
        initEncoderAndGetEncoderAndMetadata(
            chunks, chunk_metadata, geo_column_type, COORDS);

    // initialize bounds column
    if (geo_column_type == kLINESTRING || geo_column_type == kPOLYGON ||
        geo_column_type == kMULTIPOLYGON) {
      std::tie(
          bounds_column_encoder_, bounds_column_metadata_, bounds_column_descriptor_) =
          initEncoderAndGetEncoderAndMetadata(
              chunks, chunk_metadata, geo_column_type, BOUNDS);
    }

    // initialize ring sizes column & render group column
    if (geo_column_type == kPOLYGON || geo_column_type == kMULTIPOLYGON) {
      std::tie(ring_sizes_column_encoder_,
               ring_sizes_column_metadata_,
               ring_sizes_column_descriptor_) =
          initEncoderAndGetEncoderAndMetadata(
              chunks, chunk_metadata, geo_column_type, RING_SIZES);
      std::tie(render_group_column_encoder_,
               render_group_column_metadata_,
               render_group_column_descriptor_) =
          initEncoderAndGetEncoderAndMetadata(
              chunks, chunk_metadata, geo_column_type, RENDER_GROUP);
    }

    // initialize poly rings column
    if (geo_column_type == kMULTIPOLYGON) {
      std::tie(poly_rings_column_encoder_,
               poly_rings_column_metadata_,
               poly_rings_column_descriptor_) =
          initEncoderAndGetEncoderAndMetadata(
              chunks, chunk_metadata, geo_column_type, POLY_RINGS);
    }
  }

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  const bool is_last_batch,
                  int8_t* values) override {
    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);

    clearDatumBuffers();

    for (int64_t i = 0, j = 0; i < levels_read; ++i, ++row_count_) {
      clearParseBuffers();
      if (def_levels[i] == 0) {
        processNullGeoElement();
      } else {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        auto geo_string_view = std::string_view{
            reinterpret_cast<const char*>(byte_array.ptr), byte_array.len};
        processGeoElement(geo_string_view);
      }
    }

    appendArrayDatumsToBufferAndUpdateMetadata();

    appendBaseAndRenderGroupDataAndUpdateMetadata(levels_read);
  }

 private:
  void appendBaseAndRenderGroupDataAndUpdateMetadata(const int64_t levels_read) {
    // add nulls to base column & zeros to render group (if applicable)
    *base_column_metadata_ =
        *base_column_encoder_->appendData(&base_values_, 0, levels_read);
    if (render_group_column_encoder_) {
      auto data_ptr = reinterpret_cast<int8_t*>(render_group_values_.data());
      *render_group_column_metadata_ = *render_group_column_encoder_->appendData(
          data_ptr, levels_read, render_group_column_descriptor_->columnType);
    }
  }

  void validateChunksAndMetadataSizing(
      std::list<Chunk_NS::Chunk>& chunks,
      std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata) const {
    const auto geo_column_type = geo_column_descriptor_->columnType.get_type();
    if (geo_column_type == kPOINT) {
      CHECK(chunk_metadata.size() == 2);
      CHECK(chunks.size() == 2);
    } else if (geo_column_type == kLINESTRING) {
      CHECK(chunk_metadata.size() == 3);
      CHECK(chunks.size() == 3);
    } else if (geo_column_type == kPOLYGON) {
      CHECK(chunk_metadata.size() == 5);
      CHECK(chunks.size() == 5);
    } else if (geo_column_type == kMULTIPOLYGON) {
      CHECK(chunk_metadata.size() == 6);
      CHECK(chunks.size() == 6);
    }
  }

  void appendArrayDatumsToBufferAndUpdateMetadata() {
    appendToArrayEncoderAndUpdateMetadata(
        coords_datum_buffer_, coords_column_encoder_, coords_column_metadata_);
    appendToArrayEncoderAndUpdateMetadata(
        bounds_datum_buffer_, bounds_column_encoder_, bounds_column_metadata_);
    appendToArrayEncoderAndUpdateMetadata(ring_sizes_datum_buffer_,
                                          ring_sizes_column_encoder_,
                                          ring_sizes_column_metadata_);
    appendToArrayEncoderAndUpdateMetadata(poly_rings_datum_buffer_,
                                          poly_rings_column_encoder_,
                                          poly_rings_column_metadata_);
  }

  void appendToArrayEncoderAndUpdateMetadata(
      const std::vector<ArrayDatum>& datum_parse_buffer,
      Encoder* encoder,
      ChunkMetadata* chunk_metadata) const {
    if (!encoder) {
      CHECK(!chunk_metadata);
      return;
    }
    if (auto fixed_len_array_encoder =
            dynamic_cast<FixedLengthArrayNoneEncoder*>(encoder)) {
      auto new_chunk_metadata = fixed_len_array_encoder->appendData(
          &datum_parse_buffer, 0, datum_parse_buffer.size());
      *chunk_metadata = *new_chunk_metadata;
    } else if (auto array_encoder = dynamic_cast<ArrayNoneEncoder*>(encoder)) {
      auto new_chunk_metadata = array_encoder->appendData(
          &datum_parse_buffer, 0, datum_parse_buffer.size(), false);
      *chunk_metadata = *new_chunk_metadata;
    } else {
      UNREACHABLE();
    }
  }

  void processGeoElement(std::string_view geo_string_view) {
    SQLTypeInfo import_ti{geo_column_descriptor_->columnType};
    if (!Geospatial::GeoTypesFactory::getGeoColumns(std::string(geo_string_view),
                                                    import_ti,
                                                    coords_parse_buffer_,
                                                    bounds_parse_buffer_,
                                                    ring_sizes_parse_buffer_,
                                                    poly_rings_parse_buffer_,
                                                    PROMOTE_POLYGON_TO_MULTIPOLYGON)) {
      std::string error_message =
          "Failed to extract valid geometry from row " + std::to_string(row_count_) +
          " for OmniSci column " + geo_column_descriptor_->columnName +
          " importing from parquet column " + parquet_column_name_;
      throw std::runtime_error(error_message);
    }

    // validate types
    if (geo_column_descriptor_->columnType.get_type() != import_ti.get_type()) {
      if (!PROMOTE_POLYGON_TO_MULTIPOLYGON ||
          !(import_ti.get_type() == SQLTypes::kPOLYGON &&
            geo_column_descriptor_->columnType.get_type() == SQLTypes::kMULTIPOLYGON)) {
        throw std::runtime_error("Imported geometry from parquet column " +
                                 parquet_column_name_ +
                                 " doesn't match the geospatial type of OmniSci column " +
                                 geo_column_descriptor_->columnName);
      }
    }

    // append coords
    std::vector<uint8_t> compressed_coords = Geospatial::compress_coords(
        coords_parse_buffer_, geo_column_descriptor_->columnType);
    coords_datum_buffer_.emplace_back(encode_as_array_datum(compressed_coords));

    // append bounds
    if (bounds_column_encoder_) {
      bounds_datum_buffer_.emplace_back(encode_as_array_datum(bounds_parse_buffer_));
    }

    // append ring sizes
    if (ring_sizes_column_encoder_) {
      ring_sizes_datum_buffer_.emplace_back(
          encode_as_array_datum(ring_sizes_parse_buffer_));
    }

    // append poly rings
    if (poly_rings_column_encoder_) {
      poly_rings_datum_buffer_.emplace_back(
          encode_as_array_datum(poly_rings_parse_buffer_));
    }
  }

  void processNullGeoElement() {
    SQLTypeInfo import_ti{geo_column_descriptor_->columnType};
    Geospatial::GeoTypesFactory::getNullGeoColumns(import_ti,
                                                   coords_parse_buffer_,
                                                   bounds_parse_buffer_,
                                                   ring_sizes_parse_buffer_,
                                                   poly_rings_parse_buffer_,
                                                   PROMOTE_POLYGON_TO_MULTIPOLYGON);
    // POINT columns are represented using fixed length arrays and need
    // special treatment of nulls
    if (geo_column_descriptor_->columnType.get_type() == kPOINT) {
      std::vector<uint8_t> compressed_coords = Geospatial::compress_coords(
          coords_parse_buffer_, geo_column_descriptor_->columnType);
      coords_datum_buffer_.emplace_back(encode_as_array_datum(compressed_coords));
    } else {
      coords_datum_buffer_.emplace_back(import_export::ImporterUtils::composeNullArray(
          coords_column_descriptor_->columnType));
    }
    if (bounds_column_encoder_) {
      bounds_datum_buffer_.emplace_back(import_export::ImporterUtils::composeNullArray(
          bounds_column_descriptor_->columnType));
    }
    if (ring_sizes_column_encoder_) {
      ring_sizes_datum_buffer_.emplace_back(
          import_export::ImporterUtils::composeNullArray(
              ring_sizes_column_descriptor_->columnType));
    }
    if (poly_rings_column_encoder_) {
      poly_rings_datum_buffer_.emplace_back(
          import_export::ImporterUtils::composeNullArray(
              poly_rings_column_descriptor_->columnType));
    }
  }

  void clearParseBuffers() {
    coords_parse_buffer_.clear();
    bounds_parse_buffer_.clear();
    ring_sizes_parse_buffer_.clear();
    poly_rings_parse_buffer_.clear();
  }

  void clearDatumBuffers() {
    coords_datum_buffer_.clear();
    bounds_datum_buffer_.clear();
    ring_sizes_datum_buffer_.clear();
    poly_rings_datum_buffer_.clear();
  }

  enum GeoColumnType { COORDS, BOUNDS, RING_SIZES, POLY_RINGS, RENDER_GROUP };

  template <typename T>
  typename std::list<T>::iterator getIteratorForGeoColumnType(
      std::list<T>& list,
      const SQLTypes column_type,
      const GeoColumnType geo_column) {
    auto list_iter = list.begin();
    list_iter++;  // skip base column
    switch (column_type) {
      case kPOINT: {
        if (geo_column == COORDS) {
          return list_iter;
        }
        UNREACHABLE();
      }
      case kLINESTRING: {
        if (geo_column == COORDS) {
          return list_iter;
        }
        list_iter++;
        if (geo_column == BOUNDS) {
          return list_iter;
        }
        UNREACHABLE();
      }
      case kPOLYGON: {
        if (geo_column == COORDS) {
          return list_iter;
        }
        list_iter++;
        if (geo_column == RING_SIZES) {
          return list_iter;
        }
        list_iter++;
        if (geo_column == BOUNDS) {
          return list_iter;
        }
        list_iter++;
        if (geo_column == RENDER_GROUP) {
          return list_iter;
        }
        UNREACHABLE();
      }
      case kMULTIPOLYGON: {
        if (geo_column == COORDS) {
          return list_iter;
        }
        list_iter++;
        if (geo_column == RING_SIZES) {
          return list_iter;
        }
        list_iter++;
        if (geo_column == POLY_RINGS) {
          return list_iter;
        }
        list_iter++;
        if (geo_column == BOUNDS) {
          return list_iter;
        }
        list_iter++;
        if (geo_column == RENDER_GROUP) {
          return list_iter;
        }
        UNREACHABLE();
      }
      default:
        UNREACHABLE();
    }
    return {};
  }

  std::tuple<Encoder*, ChunkMetadata*, const ColumnDescriptor*>
  initEncoderAndGetEncoderAndMetadata(
      std::list<Chunk_NS::Chunk>& chunks,
      std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
      const SQLTypes sql_type,
      GeoColumnType geo_column_type) {
    auto chunk = getIteratorForGeoColumnType(chunks, sql_type, geo_column_type);
    chunk->initEncoder();
    auto encoder = chunk->getBuffer()->getEncoder();
    auto metadata =
        getIteratorForGeoColumnType(chunk_metadata, sql_type, geo_column_type)->get();
    auto column_descriptor = chunk->getColumnDesc();
    return {encoder, metadata, column_descriptor};
  }

  const ColumnDescriptor* geo_column_descriptor_;

  constexpr static bool PROMOTE_POLYGON_TO_MULTIPOLYGON = true;

  StringNoneEncoder* base_column_encoder_;
  Encoder* coords_column_encoder_;
  Encoder* bounds_column_encoder_;
  Encoder* ring_sizes_column_encoder_;
  Encoder* poly_rings_column_encoder_;
  Encoder* render_group_column_encoder_;

  ChunkMetadata* base_column_metadata_;
  ChunkMetadata* coords_column_metadata_;
  ChunkMetadata* bounds_column_metadata_;
  ChunkMetadata* ring_sizes_column_metadata_;
  ChunkMetadata* poly_rings_column_metadata_;
  ChunkMetadata* render_group_column_metadata_;

  const ColumnDescriptor* coords_column_descriptor_;
  const ColumnDescriptor* bounds_column_descriptor_;
  const ColumnDescriptor* ring_sizes_column_descriptor_;
  const ColumnDescriptor* poly_rings_column_descriptor_;
  const ColumnDescriptor* render_group_column_descriptor_;

  std::vector<int32_t> render_group_values_;
  std::vector<std::string> base_values_;
  size_t row_count_;
  std::string parquet_column_name_;

  // Used repeatedly in parsing geo types, declared as members to prevent
  // deallocation/reallocation costs
  std::vector<double> coords_parse_buffer_;
  std::vector<double> bounds_parse_buffer_;
  std::vector<int> ring_sizes_parse_buffer_;
  std::vector<int> poly_rings_parse_buffer_;

  // Used to buffer array appends in memory for a batch
  std::vector<ArrayDatum> coords_datum_buffer_;
  std::vector<ArrayDatum> bounds_datum_buffer_;
  std::vector<ArrayDatum> ring_sizes_datum_buffer_;
  std::vector<ArrayDatum> poly_rings_datum_buffer_;
};

}  // namespace foreign_storage
