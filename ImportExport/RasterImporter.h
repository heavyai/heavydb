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

/*
 * @file RasterImporter.h
 * @brief GDAL Raster File Importer
 *
 */

#pragma once

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "Geospatial/GDAL.h"
#include "ImportExport/MetadataColumn.h"
#include "Shared/sqltypes.h"

class GDALRasterBand;
class OGRCoordinateTransformation;
class OGRDataSource;
namespace Data_Namespace {
class AbstractBuffer;
}

namespace import_export {
class GCPTransformer {
 public:
  enum class Mode { kPolynomial, kThinPlateSpline };

  explicit GCPTransformer(OGRDataSource* datasource, const Mode mode = Mode::kPolynomial);
  GCPTransformer() = delete;
  ~GCPTransformer();

  void transform(size_t num_elems, double* x, double* y);

 private:
  void* transform_arg_;
  const Mode mode_;
};

class RasterImporter {
 public:
  RasterImporter() = default;
  ~RasterImporter() = default;

  enum class PointType { kNone, kAuto, kSmallInt, kInt, kFloat, kDouble, kPoint };
  enum class PointTransform { kNone, kAuto, kFile, kWorld };

  struct ChunkBoundingBox {
    ChunkBoundingBox(int32_t x, int32_t y, int32_t w, int32_t h)
        : x_offset(x), y_offset(y), width(w), height(h), num_pixels(width * height) {
      CHECK_GT(width, 0);
      CHECK_GT(height, 0);
      CHECK_GE(x_offset, 0);
      CHECK_GE(y_offset, 0);
    }
    int32_t x_offset, y_offset, width, height, num_pixels;
  };

  static PointType createPointType(const std::string& str);

  void detect(const std::string& file_name,
              const std::string& specified_band_names,
              const std::string& specified_band_dimensions,
              const PointType point_type,
              const PointTransform point_transform,
              const bool point_compute_angle,
              const bool throw_on_error,
              const MetadataColumnInfos& metadata_column_infos);

  void import(size_t& max_threads, const bool max_threads_using_default);

  using NamesAndSQLTypes = std::vector<std::pair<std::string, SQLTypes>>;
  using RawPixels = std::vector<std::byte>;
  using NullValue = std::pair<double, bool>;
  using Coords = std::vector<std::tuple<double, double, float>>;
  using CoordBuffers = std::tuple<std::unique_ptr<double[]>,
                                  std::unique_ptr<double[]>,
                                  std::unique_ptr<float[]>>;

  const uint32_t getNumBands() const;
  const PointTransform getPointTransform() const;
  const NamesAndSQLTypes getPointNamesAndSQLTypes() const;
  const NamesAndSQLTypes getBandNamesAndSQLTypes() const;
  const int getBandsWidth() const { return bands_width_; }
  const int getBandsHeight() const { return bands_height_; }
  const int getBlockWidth() const { return block_width_; }
  const int getBlockHeight() const { return block_height_; }
  const NullValue getBandNullValue(const int band_idx) const;
  const Coords getProjectedPixelCoords(const uint32_t thread_idx, const int y) const;
  const std::tuple<double, double> getProjectedPixelCoord(const uint32_t thread_idx,
                                                          const int x,
                                                          const int y) const;
  CoordBuffers getProjectedPixelCoordChunks(
      const ChunkBoundingBox& chunk_size,
      const bool do_point_compute,
      const double* const lons = nullptr,
      const double* const lats = nullptr,
      const ChunkBoundingBox* const prev_chunk_size = nullptr) const;
  void getRawPixels(const uint32_t thread_idx,
                    const uint32_t band_idx,
                    const int y_start,
                    const int num_rows,
                    const SQLTypes column_sql_type,
                    RawPixels& raw_pixel_bytes);
  void getRawPixelsFineGrained(const uint32_t thread_idx,
                               const uint32_t band_idx,
                               const ChunkBoundingBox& chunk_size,
                               const SQLTypes column_sql_type,
                               int8_t* raw_pixel_bytes);

 private:
  struct ImportBandInfo {
    std::string name;
    std::string alt_name;
    SQLTypes sql_type;
    uint32_t datasource_idx;
    int band_idx;
    double null_value;
    bool null_value_valid;
  };

  std::vector<std::string> datasource_names_;
  std::vector<std::vector<std::string>> raw_band_names_;
  std::map<std::string, bool> specified_band_names_map_;
  std::map<std::string, int> column_name_repeats_map_;
  std::vector<ImportBandInfo> import_band_infos_;
  std::array<double, 6> affine_transform_matrix_;
  std::vector<std::vector<Geospatial::GDAL::DataSourceUqPtr>> datasource_handles_;
  int specified_band_width_{-1};
  int specified_band_height_{-1};

  int bands_width_{-1};
  int bands_height_{-1};
  int block_width_{-1};
  int block_height_{-1};
  PointType point_type_{PointType::kNone};
  PointTransform point_transform_{PointTransform::kNone};
  std::vector<Geospatial::GDAL::CoordinateTransformationUqPtr>
      coordinate_transformations_;
  std::vector<std::unique_ptr<GCPTransformer>> gcp_transformers_;
  bool point_compute_angle_{false};

  void getRawBandNamesForFormat(const Geospatial::GDAL::DataSourceUqPtr& datasource);
  void initializeFiltering(const std::string& specified_band_names,
                           const std::string& specified_band_dimensions,
                           const MetadataColumnInfos& metadata_column_infos);
  bool shouldImportBandWithName(const std::string& name);
  bool shouldImportBandWithDimensions(const int width, const int height);
  std::string getBandName(const uint32_t datasource_idx, const int band_idx);
  void checkSpecifiedBandNamesFound() const;
};
}  // namespace import_export

inline std::ostream& operator<<(
    std::ostream& os,
    const import_export::RasterImporter::PointType& point_type) {
  using PT = import_export::RasterImporter::PointType;
  if (point_type == PT::kNone) {
    os << "None";
  } else if (point_type == PT::kAuto) {
    os << "Auto";
  } else if (point_type == PT::kSmallInt) {
    os << "SmallInt";
  } else if (point_type == PT::kInt) {
    os << "Int";
  } else if (point_type == PT::kFloat) {
    os << "Float";
  } else if (point_type == PT::kDouble) {
    os << "Double";
  } else if (point_type == PT::kPoint) {
    os << "Point";
  }
  return os;
}

inline std::ostream& operator<<(
    std::ostream& os,
    const import_export::RasterImporter::PointTransform& point_trans) {
  using PT = import_export::RasterImporter::PointTransform;
  if (point_trans == PT::kNone) {
    os << "None";
  } else if (point_trans == PT::kAuto) {
    os << "Auto";
  } else if (point_trans == PT::kFile) {
    os << "File";
  } else if (point_trans == PT::kWorld) {
    os << "World";
  }
  return os;
}
