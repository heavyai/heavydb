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

#ifdef HAVE_PDAL
#ifndef __CUDACC__

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsDataCache.h"
#include "QueryEngine/heavydbTypes.h"

#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/SpatialReference.hpp>
#include <pdal/io/LasReader.hpp>
namespace PdalLoader {

static std::mutex print_mutex;

struct LidarMetadata {
  std::string filename;
  int32_t file_source_id;
  int16_t version_major;
  int16_t version_minor;
  int16_t creation_year;
  bool is_compressed;
  int64_t num_points;
  int16_t num_dims;
  int16_t point_len;
  bool has_time;
  bool has_color;
  bool has_wave;
  bool has_infrared;
  bool has_14_point_format;
  int32_t specified_utm_zone;
  bool is_transformed;
  pdal::SpatialReference source_spatial_reference;
  std::pair<double, double> source_x_bounds;
  std::pair<double, double> source_y_bounds;
  std::pair<double, double> source_z_bounds;
  pdal::SpatialReference transformed_spatial_reference;
  std::pair<double, double> transformed_x_bounds;
  std::pair<double, double> transformed_y_bounds;
  std::pair<double, double> transformed_z_bounds;

  LidarMetadata() : num_points(0) {}

  inline void updateBounds(const double x, const double y, const double z) {
    transformed_x_bounds.first = std::min(transformed_x_bounds.first, x);
    transformed_x_bounds.second = std::max(transformed_x_bounds.second, x);
    transformed_y_bounds.first = std::min(transformed_y_bounds.first, y);
    transformed_y_bounds.second = std::max(transformed_y_bounds.second, y);
    transformed_z_bounds.first = std::min(transformed_z_bounds.first, z);
    transformed_z_bounds.second = std::max(transformed_z_bounds.second, z);
  }

  void print() const {
    std::unique_lock<std::mutex> print_lock(print_mutex);
    std::cout << std::endl << "-----Lidar Metadata-------" << std::endl;
    std::cout << "# Points: " << num_points << std::endl;
    std::cout << "X bounds: (" << transformed_x_bounds.first << ", "
              << transformed_x_bounds.second << ")" << std::endl;
    std::cout << "Y bounds: (" << transformed_y_bounds.first << ", "
              << transformed_y_bounds.second << ")" << std::endl;
    std::cout << "Z bounds: (" << transformed_z_bounds.first << ", "
              << transformed_z_bounds.second << ")" << std::endl;
  }
};

struct LidarData {
  double* x;
  double* y;
  double* z;
  int32_t* intensity;  // unsigned short per standard, but we don't have unsigned types so
                       // need to promote to 4 bytes
  int8_t* return_num;
  int8_t* num_returns;
  int8_t* scan_direction_flag;
  int8_t* edge_of_flight_line;
  int16_t* classification;
  int8_t* scan_angle_rank;
};

static DataBufferCache data_cache;
static DataCache<LidarMetadata> metadata_cache;

inline std::string make_cache_key(const std::string& filename,
                                  const std::string& out_srs,
                                  const std::string& attribute) {
  return filename + " || " + out_srs + "|| " + attribute;
}

std::shared_ptr<LidarMetadata> get_metadata_for_file(const std::string& filename,
                                                     const std::string& out_srs);

std::pair<int64_t, std::vector<std::shared_ptr<LidarMetadata>>> get_metadata_for_files(
    const std::vector<std::filesystem::path>& file_paths,
    const std::string& out_srs);

std::pair<int64_t, std::vector<std::shared_ptr<LidarMetadata>>> filter_files_by_bounds(
    const std::vector<std::shared_ptr<LidarMetadata>>& lidar_metadata_vec,
    const double x_min,
    const double x_max,
    const double y_min,
    const double y_max);

bool keys_are_cached(
    const std::string& filename,
    const std::string& out_srs,
    const size_t num_records,
    const std::vector<std::pair<std::string, size_t>>& sub_keys_and_byte_sizes);

class PdalFileMgr {
 public:
  PdalFileMgr(const std::shared_ptr<LidarMetadata>& lidar_metadata,
              const std::string& out_srs)
      : lidar_metadata_(lidar_metadata), out_srs_(out_srs) {}

  void openAndPrepareFile();

  inline bool isReady() const { return is_ready_; }

  void readData(double* x,
                double* y,
                double* z,
                int32_t* intensity,
                int8_t* return_num,
                int8_t* num_returns,
                int8_t* scan_direction_flag,
                int8_t* edge_of_flight_line_flag,
                int16_t* classification,
                int8_t* scan_angle_rank) {
    readData(x,
             y,
             z,
             intensity,
             return_num,
             num_returns,
             scan_direction_flag,
             edge_of_flight_line_flag,
             classification,
             scan_angle_rank,
             lidar_metadata_->num_points);
  }

  void readData(double* x,
                double* y,
                double* z,
                int32_t* intensity,
                int8_t* return_num,
                int8_t* num_returns,
                int8_t* scan_direction_flag,
                int8_t* edge_of_flight_line_flag,
                int16_t* classification,
                int8_t* scan_angle_rank,
                const int64_t num_points);

 private:
  pdal::LasReader las_reader_;
  pdal::PointTable point_table_;
  pdal::PointViewSet point_view_set_;
  pdal::PointViewPtr point_view_;

  const std::shared_ptr<LidarMetadata> lidar_metadata_;
  const std::string out_srs_;
  bool is_ready_{false};
};

}  // namespace PdalLoader

#endif  // __CUDACC__
#endif  // HAVE_PDAL
