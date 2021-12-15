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

/*
 * @file RasterImporter.h
 * @author Simon Eves <simon.eves@omnisci.com>
 * @brief GDAL Raster File Importer
 */

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "Geospatial/GDAL.h"
#include "Shared/sqltypes.h"

class GDALRasterBand;
class OGRCoordinateTransformation;
class OGRDataSource;

namespace import_export {

class RasterImporter {
 public:
  RasterImporter() = default;
  ~RasterImporter() = default;

  enum class PointType { kNone, kAuto, kSmallInt, kInt, kFloat, kDouble, kPoint };
  enum class PointTransform { kNone, kAuto, kFile, kWorld };

  void detect(const std::string& file_name,
              const std::string& specified_band_names,
              const std::string& specified_band_dimensions,
              const PointType point_type,
              const PointTransform point_transform,
              const bool point_compute_angle,
              const bool throw_on_error);

  void import(const uint32_t max_threads);

  using NamesAndSQLTypes = std::vector<std::pair<std::string, SQLTypes>>;
  using RawPixels = std::vector<std::byte>;
  using NullValue = std::pair<double, bool>;
  using Coords = std::vector<std::tuple<double, double, float>>;

  const uint32_t getNumBands() const;
  const PointTransform getPointTransform() const;
  const NamesAndSQLTypes getPointNamesAndSQLTypes() const;
  const NamesAndSQLTypes getBandNamesAndSQLTypes() const;
  const int getBandsWidth() const { return bands_width_; }
  const int getBandsHeight() const { return bands_height_; }
  const NullValue getBandNullValue(const int band_idx) const;
  const Coords getProjectedPixelCoords(const uint32_t thread_idx, const int y) const;
  void getRawPixels(const uint32_t thread_idx,
                    const uint32_t band_idx,
                    const int y_start,
                    const int num_rows,
                    const SQLTypes column_sql_type,
                    RawPixels& raw_pixel_bytes);

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
  PointType point_type_{PointType::kNone};
  PointTransform point_transform_{PointTransform::kNone};
  std::vector<Geospatial::GDAL::CoordinateTransformationUqPtr>
      coordinate_transformations_;
  bool point_compute_angle_{false};

  void getRawBandNamesForFormat(const Geospatial::GDAL::DataSourceUqPtr& datasource);
  void initializeFiltering(const std::string& specified_band_names,
                           const std::string& specified_band_dimensions);
  bool shouldImportBandWithName(const std::string& name);
  bool shouldImportBandWithDimensions(const int width, const int height);
  std::string getBandName(const uint32_t datasource_idx, const int band_idx);
  void checkSpecifiedBandNamesFound() const;
};

}  // namespace import_export
