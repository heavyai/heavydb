/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/dynamic_bitset.hpp>
#include <set>
#include <string_view>
#include "DataMgr/ForeignStorage/AbstractFileStorageDataWrapper.h"
#include "DataMgr/ForeignStorage/ForeignDataWrapper.h"
#include "DataMgr/ForeignStorage/ForeignStorageException.h"
#include "ImportExport/RasterImporter.h"
#include "Shared/LonLatBoundingBox.h"

namespace foreign_storage {

using UnprojectedLon = int32_t;
using UnprojectedLat = int32_t;
using UnprojectedPoint = std::pair<UnprojectedLon, UnprojectedLat>;

class RasterDataWrapper : public AbstractFileStorageDataWrapper {
 public:
  inline static const std::string RASTER_WIDTH_KEY = "RASTER_TILE_WIDTH";
  inline static const std::string RASTER_HEIGHT_KEY = "RASTER_TILE_HEIGHT";
  inline static const std::string RASTER_FILTER_BANDS_KEY = "RASTER_FILTER_BANDS";
  inline static const std::string RASTER_POINT_TRANSFORM_KEY = "RASTER_POINT_TRANSFORM";
  inline static const std::string BOUNDING_BOX_CLIP_KEY = "BOUNDING_BOX_CLIP";
  inline static const std::string RASTER_DROP_IF_ALL_NULL_KEY = "RASTER_DROP_IF_ALL_NULL";

  struct Shape {
    Shape(int32_t w, int32_t h) : width(w), height(h) {
      CHECK_GT(width, 0);
      CHECK_GT(height, 0);
    }
    const int32_t width, height;
    int32_t getNumPixels() const { return width * height; }
  };

  // Raster and Chunk dimensions represent the same data, but are unaliased for API type
  // safety.
  struct RasterShape : public Shape {
    using Shape::Shape;
    bool operator==(const RasterShape& other) {
      return (width == other.width) && (height == other.height);
    }
  };

  struct ChunkShape : public Shape {
    using Shape::Shape;
  };

  RasterDataWrapper();
  RasterDataWrapper(const int db_id,
                    const ForeignTable* foreign_table,
                    const UserMapping* user_mapping);
  void populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) override;
  void populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                            const ChunkToBufferMap& optional_buffers,
                            AbstractBuffer* delete_buffer = nullptr) override;
  std::string getSerializedDataWrapper() const override;
  void restoreDataWrapperInternals(const std::string& file_path,
                                   const ChunkMetadataVector& chunk_metadata) override;
  bool isRestored() const override;
  void validateSchema(const std::list<ColumnDescriptor>& columns,
                      const ForeignTable* foreign_table) const override;
  void validateTableOptions(const ForeignTable* foreign_table) const override;
  const std::set<std::string_view>& getSupportedTableOptions() const override;
  int32_t getMaxFragRowsForImport() const;

  // Some functions are protected so that we can mock them.
 protected:
  virtual void clearWrapperData();

 private:
  static const std::set<std::string_view> supported_table_options_;

  void initializeChunkBoundingBoxes(const RasterShape& raster_shape,
                                    const ChunkShape& chunk_shape,
                                    int32_t num_fragments);

  int32_t getChunkWidth(const import_export::RasterImporter&) const;
  int32_t getChunkHeight(const import_export::RasterImporter&) const;
  import_export::RasterImporter::PointType getPointType() const;
  bool hasDropIfAllNull() const;
  bool isPointColumn(int32_t column_idx) const;
  std::string getRasterFilterBands() const;
  import_export::RasterImporter::PointTransform getRasterPointTransform() const;
  std::vector<std::string> getFilesFromPath() const;
  std::vector<std::string> getS3FilteredFiles() const;
  import_export::RasterImporter& getRasterImporter(int32_t frag_id) const;
  std::optional<shared::LonLatBoundingBox> getBoundingBoxClip() const;
  bool hasWrapperData() const;
  bool isAppendMode() const;
  std::set<std::string_view> getAllTableOptions() const;

  void validateHeight(const ForeignTable*) const;
  void validateWidth(const ForeignTable*) const;

  shared::LonLatBoundingBox getPointChunkMinMax(
      const import_export::RasterImporter::ChunkBoundingBox& chunk_box,
      import_export::RasterImporter& raster_importer) const;
  std::pair<std::shared_ptr<ChunkMetadata>, std::shared_ptr<ChunkMetadata>>
  createPointChunkMetadata(const ChunkKey& first_key,
                           const SQLTypeInfo& first_type,
                           const ChunkKey& second_key,
                           const SQLTypeInfo& second_type) const;

  void initializeMetadataMap(int32_t num_fragments, int32_t first_frag);
  void initializeColumns();

  boost::dynamic_bitset<> importBandChunk(AbstractBuffer* buffer,
                                          const ChunkKey& key,
                                          AbstractBuffer* delete_buffer = nullptr);
  void importPointChunk(AbstractBuffer* buffer,
                        const ChunkKey& key,
                        AbstractBuffer* idx_buffer = nullptr);

  import_export::RasterImporter::CoordBuffers& getCoords(
      const import_export::RasterImporter::ChunkBoundingBox& chunk_box,
      const int32_t frag_id);

  void cacheMetadata(const ChunkKey& key, const std::shared_ptr<ChunkMetadata>& meta);

  int32_t mapFilesToWrapper(const std::vector<std::string>& files, int32_t first_frag);
  void initializeRasterImporter(import_export::RasterImporter& raster_importer,
                                const std::string& file);

  import_export::RasterImporter* createNewImporter();
  std::map<UnprojectedPoint, import_export::RasterImporter::CoordBuffers>&
  getCoordinateCache(int32_t frag_id);
  int32_t mapAllTiles(
      const std::vector<import_export::RasterImporter::ChunkBoundingBox>& new_boxes,
      const RasterShape& raster_shape,
      const ChunkShape& chunk_shape);
  int32_t mapClippedTiles(
      import_export::RasterImporter& raster_importer,
      const shared::LonLatBoundingBox& bounding_box_clip,
      const std::vector<import_export::RasterImporter::ChunkBoundingBox>& new_boxes,
      const RasterShape& raster_shape,
      const ChunkShape& chunk_shape);

  // TODO(Misiu): It would be good to turn these all into const references, except
  // that the ForeignDataWrapperFactory expects to create "abstract" wrappers with no
  // foreign table attached.
  const int32_t db_id_;
  const ForeignTable* foreign_table_;
  const UserMapping* user_mapping_;

  std::vector<std::unique_ptr<import_export::RasterImporter>> importers_;
  // Marks at which fragment indexes we start reading from a new file (we need to know
  // this in order to determine which raster importer object to use for each fragment).
  std::vector<int32_t> file_fragment_borders_;
  std::vector<const ColumnDescriptor*> cds_;
  // The index into chunk_bounding_boxes_ and file_local_tiles_ are fragment ids.
  std::vector<import_export::RasterImporter::ChunkBoundingBox> chunk_bounding_boxes_;
  std::vector<FileLocalCoords> file_local_coords_for_frag_;
  std::vector<std::map<UnprojectedPoint, import_export::RasterImporter::CoordBuffers>>
      coordinate_caches_;  // Each raster file has a separate cache, hence vector of map.
  std::map<ChunkKey, std::shared_ptr<ChunkMetadata>> chunk_metadata_map_;

  static constexpr int32_t kNumPointColumns{2};
  bool is_restored_{false};

  enum class BandCombineMode { kNone, kColor };  // extend as required
  struct ColumnBandInfo {
    std::vector<int32_t> band_indices;
    BandCombineMode combine_mode = BandCombineMode::kNone;
  };
  std::vector<ColumnBandInfo> column_band_info_;
};
}  // namespace foreign_storage
