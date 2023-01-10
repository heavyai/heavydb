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

#include "GeospatialEncoder.h"

namespace foreign_storage {

class OdbcGeospatialEncoder : public GeospatialEncoder {
 public:
  OdbcGeospatialEncoder(std::list<Chunk_NS::Chunk>& chunks,
                        std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
                        const bool is_error_tracking_enabled,
                        const SQLTypeInfo& sql_type_info)
      : GeospatialEncoder(chunks, chunk_metadata)
      , type_info_(sql_type_info)
      , is_error_tracking_enabled_(is_error_tracking_enabled) {}

  void appendData(const std::vector<std::string>& geo_strings,
                  std::optional<std::set<size_t>>& rejected_row_local_indices) {
    if (is_error_tracking_enabled_) {
      CHECK(rejected_row_local_indices.has_value());
    }
    clearDatumBuffers();
    int64_t num_rows = geo_strings.size();
    for (int64_t i = 0; i < num_rows; ++i) {
      clearParseBuffers();
      auto const& geo_string = geo_strings[i];
      if (geo_string.size()) {
        try {
          processGeoElement(geo_string);
        } catch (const std::runtime_error& except) {
          if (is_error_tracking_enabled_) {
            rejected_row_local_indices->insert(i);
            clearParseBuffers();
            processNullGeoElement();
          } else {
            throw except;
          }
        }
      } else {
        processNullGeoElement();
        if (is_error_tracking_enabled_ && type_info_.get_notnull()) {
          rejected_row_local_indices->insert(i);
        }
      }
    }
    appendArrayDatumsToBufferAndUpdateMetadata();
    appendBaseDataAndUpdateMetadata(num_rows);
  }

 private:
  const SQLTypeInfo type_info_;
  const bool is_error_tracking_enabled_;
};

}  // namespace foreign_storage
