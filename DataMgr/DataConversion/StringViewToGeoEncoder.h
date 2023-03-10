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

#include "BaseConvertEncoder.h"
#include "ImportExport/DelimitedParserUtils.h"

#include "DataMgr/ForeignStorage/GeospatialEncoder.h"

namespace data_conversion {

class StringViewToGeoEncoder : public BaseConvertEncoder,
                               public foreign_storage::GeospatialEncoder {
 public:
  StringViewToGeoEncoder(std::list<Chunk_NS::Chunk>& chunks,
                         std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata,
                         const bool error_tracking_enabled,
                         const bool geo_validate_geometry)
      : BaseConvertEncoder(error_tracking_enabled)
      , foreign_storage::GeospatialEncoder(chunks,
                                           chunk_metadata,
                                           geo_validate_geometry) {}

  void encodeAndAppendData(const int8_t* data, const size_t num_elements) override {
    auto geo_strings = reinterpret_cast<const std::string_view*>(data);
    clearDatumBuffers();

    for (size_t i = 0; i < num_elements; ++i) {
      if (error_tracking_enabled_) {
        delete_buffer_->push_back(false);
      }
      clearParseBuffers();
      auto& geo_string = geo_strings[i];
      if (geo_string.empty()) {
        processNullGeoElement();
      } else {
        try {
          processGeoElement(geo_strings[i]);  // this may throw, need to handle error
        } catch (std::exception& except) {
          if (!error_tracking_enabled_) {
            throw;
          } else {
            clearParseBuffers();
            processNullGeoElement();
            delete_buffer_->back() = true;
          }
        }
      }
    }

    appendArrayDatumsToBufferAndUpdateMetadata();
    appendBaseDataAndUpdateMetadata(num_elements);
  }
};

}  // namespace data_conversion
