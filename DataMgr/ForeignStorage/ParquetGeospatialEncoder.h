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

#include <parquet/schema.h>
#include <parquet/types.h>
#include "GeospatialEncoder.h"
#include "ParquetEncoder.h"

namespace foreign_storage {

class ParquetGeospatialEncoder : public ParquetEncoder, public GeospatialEncoder {
 public:
  ParquetGeospatialEncoder() : ParquetEncoder(nullptr), GeospatialEncoder() {}

  ParquetGeospatialEncoder(const parquet::ColumnDescriptor* parquet_column_descriptor,
                           std::list<Chunk_NS::Chunk>& chunks,
                           std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata)
      : ParquetEncoder(nullptr), GeospatialEncoder(chunks, chunk_metadata) {}

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  const bool is_last_batch,
                  int8_t* values) override {
    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);

    clearDatumBuffers();

    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
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
};

}  // namespace foreign_storage
