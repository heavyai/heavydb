/*
 * Copyright 2025 HEAVY.AI, Inc.
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

/**
 * @file	RasterFragmenter.h
 * @brief
 *
 */

#pragma once

#include "InsertOrderFragmenter.h"

namespace Fragmenter_Namespace {

// A recording of which tiles are adjacent to which other tiles.
struct RasterTileNeighbours {
  int32_t left = -1, right = -1, up = -1, down = -1;

  bool operator==(const RasterTileNeighbours& rtn) const {
    return left == rtn.left && right == rtn.right && up == rtn.up && down == rtn.down;
  }
};

inline std::ostream& operator<<(std::ostream& out, const RasterTileNeighbours& rtn) {
  out << rtn.left << ", " << rtn.right << ", " << rtn.up << ", " << rtn.down;
  return out;
}

struct RasterMeshRenderingMetadata {
  int32_t width = 0, height = 0, fragment_id;
  RasterTileNeighbours neighbours;

  bool operator==(const RasterMeshRenderingMetadata& o) const {
    return width == o.width && height == o.height && neighbours == o.neighbours &&
           fragment_id == o.fragment_id;
  }
};

inline std::ostream& operator<<(std::ostream& out,
                                const RasterMeshRenderingMetadata& meta) {
  out << "width: " << meta.width << ", height: " << meta.height
      << ", fragment_id: " << meta.fragment_id << ", Neighbours: {" << meta.neighbours
      << "}";
  return out;
}

class RasterFragmenter : public InsertOrderFragmenter {
 public:
  RasterFragmenter(
      const std::vector<int> chunk_key_prefix,
      std::vector<Chunk_NS::Chunk>& chunk_vec,
      Data_Namespace::DataMgr* data_mgr,
      Catalog_Namespace::Catalog* catalog,
      const int physical_table_id,
      const int shard,
      const size_t max_fragment_rows = DEFAULT_FRAGMENT_ROWS,
      const size_t max_chunk_size = DEFAULT_MAX_CHUNK_SIZE,
      const size_t page_size = DEFAULT_PAGE_SIZE /*default 1MB*/,
      const size_t max_rows = DEFAULT_MAX_ROWS,
      const Data_Namespace::MemoryLevel default_insert_level = Data_Namespace::DISK_LEVEL,
      const bool uses_foreign_storage = false);

  ~RasterFragmenter() override {}

  std::vector<RasterMeshRenderingMetadata> computeRasterMeshRenderingMetadata() const;

  int32_t getLastFragmentFileId() const;

 protected:
  void insertChunksImpl(const InsertChunks& insert_chunk) override;
  void insertDataImpl(InsertData& insert_data) override;
  void dropFragmentsToSizeNoInsertLock(const size_t max_rows) override;
  void populateFileToLocalCoordsMap();

  std::map<int32_t, std::vector<std::vector<int32_t>>> file_to_local_coords_map_;
};

}  // namespace Fragmenter_Namespace
