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

#include "Fragmenter/RasterFragmenter.h"

namespace Fragmenter_Namespace {

RasterFragmenter::RasterFragmenter(const std::vector<int> chunk_key_prefix,
                                   std::vector<Chunk_NS::Chunk>& chunk_vec,
                                   Data_Namespace::DataMgr* data_mgr,
                                   Catalog_Namespace::Catalog* catalog,
                                   const int physical_table_id,
                                   const int shard,
                                   const size_t max_fragment_rows,
                                   const size_t max_chunk_size,
                                   const size_t page_size,
                                   const size_t max_rows,
                                   const Data_Namespace::MemoryLevel default_insert_level,
                                   const bool uses_foreign_storage)
    : InsertOrderFragmenter(chunk_key_prefix,
                            chunk_vec,
                            data_mgr,
                            catalog,
                            physical_table_id,
                            shard,
                            max_fragment_rows,
                            max_chunk_size,
                            page_size,
                            max_rows,
                            default_insert_level,
                            uses_foreign_storage) {
  populateFileToLocalCoordsMap();
}

void RasterFragmenter::insertChunksImpl(const InsertChunks& insert_chunks) {
  auto delete_column_id = findDeleteColumnId();

  // verify that all chunks to be inserted have same number of rows, otherwise the input
  // data is malformed
  auto num_rows = validateSameNumRows(insert_chunks);

  // RasterFragmenter assumes that any call to insertChunksImpl will fit into a
  // single fragment, so we don't have to create more than one fragment or loop over the
  // data.  Every call simply creates a new fragment and inserts the data.
  CHECK_GE(maxFragmentRows_, num_rows.value());

  auto valid_row_indices = insert_chunks.valid_row_indices;
  size_t num_rows_left = valid_row_indices.size();
  size_t num_rows_inserted = 0;
  const size_t num_rows_to_insert = num_rows_left;

  if (num_rows_left == 0) {
    return;
  }

  auto new_fragment = createNewFragment(defaultInsertLevel_);
  CHECK(new_fragment);
  // Set raster tile info in the encoders for the new fragment.
  const auto& tile_info = insert_chunks.chunks.begin()
                              ->second->getBuffer()
                              ->getEncoder()
                              ->getRasterTileInfo();
  for (auto& [col, chunk] : columnMap_) {
    chunk.getBuffer()->getEncoder()->setRasterTileInfo(tile_info);
  }

  size_t fragment_idx = fragmentInfoVec_.size() - 1;

  insertChunksIntoFragment(insert_chunks,
                           delete_column_id,
                           new_fragment,
                           num_rows_to_insert,
                           num_rows_inserted,
                           num_rows_left,
                           valid_row_indices,
                           fragment_idx);

  CHECK_EQ(num_rows_left, 0U);

  numTuples_ += num_rows_inserted;

  int32_t frag_idx = static_cast<int32_t>(fragment_idx);
  const auto& [file_id, x, y] = tile_info.local_coords;
  CHECK_GE(file_id, 0);

  // We don't know the necessary file size in advance, so grow if necessary before
  // assigning.
  if (y >= static_cast<int>(file_to_local_coords_map_[file_id].size())) {
    file_to_local_coords_map_[file_id].resize(y + 1);
  }
  if (x >= static_cast<int>(file_to_local_coords_map_[file_id][y].size())) {
    file_to_local_coords_map_[file_id][y].resize(x + 1, -1);
  }
  file_to_local_coords_map_[file_id][y][x] = frag_idx;
}

std::vector<RasterMeshRenderingMetadata>
RasterFragmenter::computeRasterMeshRenderingMetadata() const {
  std::vector<RasterMeshRenderingMetadata> new_metas;

  for (const auto& frag_info : fragmentInfoVec_) {
    const auto& tile_info =
        shared::get_from_map(frag_info->getChunkMetadataMap(), 1)->rasterTile;
    const auto& [file_id, x, y] = tile_info.local_coords;
    if (file_id < 0) {
      throw std::runtime_error{
          "Table was created using an older raster format that does not support Mesh "
          "Rendering.  Please re-create the table to enable Mesh Rendering."};
    }
    auto& frag_meta = new_metas.emplace_back();
    auto& [left, right, up, down] = frag_meta.neighbours;
    frag_meta.width = tile_info.width;
    frag_meta.height = tile_info.height;
    frag_meta.fragment_id = frag_info->fragmentId;

    const auto& local_map = shared::get_from_map(file_to_local_coords_map_, file_id);
    const int32_t map_height = local_map.size();
    CHECK_GE(map_height, 1);
    const int32_t map_width = local_map[y].size();
    CHECK_GE(map_width, 1);

    if (x == 0) {
      left = -1;
    } else {
      CHECK_GT(map_height, y);
      CHECK_GT(map_width, x - 1);
      left = local_map[y][x - 1];
    }
    if (x == map_width - 1) {
      frag_meta.neighbours.right = -1;
    } else {
      CHECK_GT(map_height, y);
      CHECK_GT(map_width, x + 1);
      frag_meta.neighbours.right = local_map[y][x + 1];
    }
    if (y == 0) {
      frag_meta.neighbours.up = -1;
    } else {
      CHECK_GT(map_height, y - 1);
      CHECK_GT(map_width, x);
      frag_meta.neighbours.up = local_map[y - 1][x];
    }
    if (y == map_height - 1) {
      frag_meta.neighbours.down = -1;
    } else {
      CHECK_GT(map_height, y + 1);
      CHECK_GT(map_width, x);
      frag_meta.neighbours.down = local_map[y + 1][x];
    }
  }
  return new_metas;
}

void RasterFragmenter::populateFileToLocalCoordsMap() {
  // Walk through all fragments, and group them by file identifier.
  // Walk through each group to determine the file's local coordinate space (max height
  // and width).  Create the file-local coords matrix with default values of '-1'. Each
  // file-local coords matrix gets mapped to by a file id (which is now part of the
  // metadata). Iterate through each group again, filling in known file-local entries with
  // the fragment id.

  // Iterate through all fragments, and record the required size of the local coordinate
  // system for each file via the metadata.
  std::map<int32_t, std::pair<int32_t, int32_t>> file_to_coords_size_map;
  for (const auto& frag_info : fragmentInfoVec_) {
    const auto& tile_info =
        shared::get_from_map(frag_info->getChunkMetadataMap(), 1)->rasterTile;
    if (tile_info.local_coords.file_id >= 0) {
      // If file_id < 0, then we are dealing with a table created before local coordinate
      // metadata was supported.
      auto& [max_x, max_y] =
          file_to_coords_size_map[tile_info.local_coords.file_id];  // create a new entry
                                                                    // if none exists.
      max_x = (max_x < tile_info.local_coords.x) ? tile_info.local_coords.x : max_x;
      max_y = (max_y < tile_info.local_coords.y) ? tile_info.local_coords.y : max_y;
    }
  }

  // Create the local coordinate system for each file now that we know the required size.
  for (const auto& [file_id, max_pair] : file_to_coords_size_map) {
    const auto& [max_x, max_y] = max_pair;
    file_to_local_coords_map_[file_id] =
        std::vector<std::vector<int32_t>>(max_y + 1, std::vector<int32_t>(max_x + 1, -1));
  }

  // Now that we know the required size (and have set default values for fragments that
  // might have been clipped out out of the local coordinate system via bounding box
  // clipping) we can assign known fragments to the local coordinate system.
  for (const auto& frag_info : fragmentInfoVec_) {
    const auto& frag_idx = frag_info->fragmentId;
    const auto& tile_info =
        shared::get_from_map(frag_info->getChunkMetadataMap(), 1)->rasterTile;
    const auto& [file_id, x, y] = tile_info.local_coords;
    if (file_id >= 0) {
      // if file_id < = 0 then we are using old metadata that has no coordinate system.
      file_to_local_coords_map_[file_id][y][x] = frag_idx;
    }
  }
}

int32_t RasterFragmenter::getLastFragmentFileId() const {
  if (fragmentInfoVec_.size() > 0) {
    const auto& frag_info = fragmentInfoVec_.back();
    const auto& tile_info =
        shared::get_from_map(frag_info->getChunkMetadataMap(), 1)->rasterTile;
    const auto& [file_id, x, y] = tile_info.local_coords;
    return file_id;
  } else {
    return -1;
  }
}

void RasterFragmenter::insertDataImpl(InsertData& insert_data) {
  UNREACHABLE() << "Insert operation not supported on raster table";
}

void RasterFragmenter::dropFragmentsToSizeNoInsertLock(const size_t max_rows) {
  UNREACHABLE() << "Droping fragments not supported on raster table";
}

}  // namespace Fragmenter_Namespace
