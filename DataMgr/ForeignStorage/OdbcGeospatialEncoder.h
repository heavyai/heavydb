/*
 * Copyright 2021 OmniSci, Inc.
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
                        std::list<std::unique_ptr<ChunkMetadata>>& chunk_metadata)
      : GeospatialEncoder(chunks, chunk_metadata) {}

  void appendData(const std::vector<std::string>& geo_strings) {
    clearDatumBuffers();
    int64_t num_rows = geo_strings.size();
    for (int64_t i = 0; i < num_rows; ++i) {
      clearParseBuffers();
      auto const& geo_string = geo_strings[i];
      if (geo_string.size()) {
        processGeoElement(geo_string);
      } else {
        processNullGeoElement();
      }
    }
    appendArrayDatumsToBufferAndUpdateMetadata();
    appendBaseAndRenderGroupDataAndUpdateMetadata(num_rows);
  }
};

}  // namespace foreign_storage
