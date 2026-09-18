/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GeospatialEncoder.h"

namespace foreign_storage {

class OdbcGeospatialEncoder : public GeospatialEncoder {
 public:
  OdbcGeospatialEncoder(std::list<Chunk_NS::Chunk>& chunks,
                        std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
                        const bool is_error_tracking_enabled,
                        const SQLTypeInfo& sql_type_info,
                        const bool geo_validate_geometry)
      : GeospatialEncoder(chunks, chunk_metadata, geo_validate_geometry)
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
