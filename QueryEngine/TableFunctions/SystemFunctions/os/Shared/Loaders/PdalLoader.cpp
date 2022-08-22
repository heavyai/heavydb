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

#ifdef HAVE_PDAL
#ifndef __CUDACC__

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <tbb/parallel_for.h>

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/SpatialReference.hpp>
#include <pdal/filters/ReprojectionFilter.hpp>
#include <pdal/io/LasHeader.hpp>
#include <pdal/io/LasReader.hpp>
#include <pdal/private/SrsTransform.hpp>

#include "PdalLoader.h"

namespace PdalLoader {

#ifdef _WIN32
#pragma comment(linker "/INCLUDE:get_metadata_for_file")
#else
__attribute__((__used__))
#endif
std::shared_ptr<LidarMetadata> get_metadata_for_file(const std::string& filename,
                                                     const std::string& out_srs) {
  std::shared_ptr<LidarMetadata> lidar_metadata = std::make_shared<LidarMetadata>();

  try {
    pdal::Options las_opts;
    pdal::Option las_opt("filename", filename);
    las_opts.add(las_opt);
    pdal::PointTable table;
    pdal::LasReader las_reader;
    las_reader.setOptions(las_opts);
    las_reader.prepare(table);
    pdal::LasHeader las_header = las_reader.header();
    lidar_metadata->filename = filename;
    lidar_metadata->file_source_id = las_header.fileSourceId();
    lidar_metadata->version_major = las_header.versionMajor();
    lidar_metadata->version_minor = las_header.versionMinor();
    lidar_metadata->creation_year = las_header.creationYear();
    lidar_metadata->is_compressed = las_header.compressed();
    lidar_metadata->num_points = las_header.pointCount();
    const auto& dims = las_header.usedDims();
    lidar_metadata->num_dims = dims.size();
    lidar_metadata->has_time = las_header.hasTime();
    lidar_metadata->has_color = las_header.hasColor();
    lidar_metadata->has_wave = las_header.hasWave();
    lidar_metadata->has_infrared = las_header.hasInfrared();
    lidar_metadata->has_14_point_format = las_header.has14PointFormat();

    lidar_metadata->source_spatial_reference = las_header.srs();
    if (out_srs.empty()) {
      lidar_metadata->is_transformed = false;
      lidar_metadata->transformed_spatial_reference = las_header.srs();
    } else {
      lidar_metadata->is_transformed = true;
      lidar_metadata->transformed_spatial_reference = pdal::SpatialReference(out_srs);
    }

    lidar_metadata->specified_utm_zone =
        lidar_metadata->source_spatial_reference.getUTMZone();

    // todo: transform these to out_srs
    lidar_metadata->source_x_bounds =
        std::make_pair(las_header.minX(), las_header.maxX());
    lidar_metadata->source_y_bounds =
        std::make_pair(las_header.minY(), las_header.maxY());
    lidar_metadata->source_z_bounds =
        std::make_pair(las_header.minZ(), las_header.maxZ());

    if (lidar_metadata->is_transformed) {
      pdal::SrsTransform transform =
          pdal::SrsTransform(lidar_metadata->source_spatial_reference,
                             lidar_metadata->transformed_spatial_reference);
      // Need to copy bounds as transform changes them in place
      std::pair<double, double> x_bounds = lidar_metadata->source_x_bounds;
      std::pair<double, double> y_bounds = lidar_metadata->source_y_bounds;
      std::pair<double, double> z_bounds = lidar_metadata->source_z_bounds;
      bool transform_ok =
          transform.transform(x_bounds.first, y_bounds.first, z_bounds.first);
      if (!transform_ok) {
        throw std::runtime_error("Could not transform bounding box coordinate");
      }
      transform_ok =
          transform.transform(x_bounds.second, y_bounds.second, z_bounds.second);
      if (!transform_ok) {
        throw std::runtime_error("Could not transform bounding box coordinate");
      }
      lidar_metadata->transformed_x_bounds = x_bounds;
      lidar_metadata->transformed_y_bounds = y_bounds;
      lidar_metadata->transformed_z_bounds = z_bounds;
    } else {
      lidar_metadata->transformed_x_bounds = lidar_metadata->source_x_bounds;
      lidar_metadata->transformed_y_bounds = lidar_metadata->source_y_bounds;
      lidar_metadata->transformed_z_bounds = lidar_metadata->source_z_bounds;
    }
  } catch (std::exception& e) {
    // std::cout << "Could not read from file: " << filename << " with error: " <<
    // e.what()
    //          << std::endl;
  }
  return lidar_metadata;
}

#ifdef _WIN32
#pragma comment(linker "/INCLUDE:get_metadata_for_files")
#else
__attribute__((__used__))
#endif
std::pair<int64_t, std::vector<std::shared_ptr<LidarMetadata>>> get_metadata_for_files(
    const std::vector<std::filesystem::path>& file_paths,
    const std::string& out_srs) {
  auto timer = DEBUG_TIMER(__func__);
  const size_t num_files(file_paths.size());
  std::vector<std::shared_ptr<LidarMetadata>> lidar_metadata_vec(num_files);
  std::atomic<uint32_t> file_counter = 0;
  std::atomic<int64_t> num_points = 0;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_files), [&](const tbb::blocked_range<size_t>& r) {
        const size_t start_file_idx = r.begin();
        const size_t end_file_idx = r.end();
        for (size_t file_idx = start_file_idx; file_idx != end_file_idx; ++file_idx) {
          const auto& file_path_string = file_paths[file_idx].string();
          std::string cache_key = make_cache_key(file_path_string, out_srs, "metadata");
          std::shared_ptr<LidarMetadata> lidar_metadata;
          if (metadata_cache.isKeyCached(cache_key)) {
            try {
              lidar_metadata = metadata_cache.getDataForKey(cache_key);
            } catch (std::exception& e) {
              lidar_metadata = get_metadata_for_file(file_path_string, out_srs);
            }
          } else {
            lidar_metadata = get_metadata_for_file(file_path_string, out_srs);
            metadata_cache.putDataForKey(cache_key, lidar_metadata);
          }
          if (lidar_metadata->num_points > 0) {
            const auto out_idx = file_counter.fetch_add(1, std::memory_order_relaxed);
            num_points.fetch_add(lidar_metadata->num_points, std::memory_order_relaxed);
            lidar_metadata_vec[out_idx] = lidar_metadata;
          }
        }
      });
  lidar_metadata_vec.resize(file_counter);
  return std::make_pair(num_points.load(), lidar_metadata_vec);
}

#ifdef _WIN32
#pragma comment(linker "/INCLUDE:filter_files_by_bounds")
#else
__attribute__((__used__))
#endif
std::pair<int64_t, std::vector<std::shared_ptr<LidarMetadata>>> filter_files_by_bounds(
    const std::vector<std::shared_ptr<LidarMetadata>>& lidar_metadata_vec,
    const double x_min,
    const double x_max,
    const double y_min,
    const double y_max) {
  std::vector<std::shared_ptr<LidarMetadata>> filtered_lidar_metadata_vec;
  int64_t unfiltered_num_points = 0;
  int64_t filtered_num_points = 0;

  for (const auto& lidar_metadata : lidar_metadata_vec) {
    unfiltered_num_points += lidar_metadata->num_points;

    if (x_max < lidar_metadata->transformed_x_bounds.first ||
        x_min > lidar_metadata->transformed_x_bounds.second ||
        y_max < lidar_metadata->transformed_y_bounds.first ||
        y_min > lidar_metadata->transformed_y_bounds.second) {
      // doesn't pass metadata filter - discard
      continue;
    }
    filtered_num_points += lidar_metadata->num_points;
    filtered_lidar_metadata_vec.emplace_back(lidar_metadata);
  }
  return std::make_pair(filtered_num_points, filtered_lidar_metadata_vec);
}

#ifdef _WIN32
#pragma comment(linker "/INCLUDE:keys_are_cached")
#else
__attribute__((__used__))
#endif
bool keys_are_cached(
    const std::string& filename,
    const std::string& out_srs,
    const size_t num_records,
    const std::vector<std::pair<std::string, size_t>>& sub_keys_and_byte_sizes) {
  for (const auto& sub_key_and_byte_size : sub_keys_and_byte_sizes) {
    const std::string cache_key(
        make_cache_key(filename, out_srs, sub_key_and_byte_size.first));
    const size_t num_bytes = num_records * sub_key_and_byte_size.second;
    if (!data_cache.isKeyCachedAndSameLength(cache_key, num_bytes)) {
      return false;
    }
  }
  return true;
}

#ifdef _WIN32
#pragma comment(linker "/INCLUDE:openAndPrepareFile")
#else
__attribute__((__used__))
#endif
void PdalFileMgr::openAndPrepareFile() {
  auto timer = DEBUG_TIMER(__func__);
  if (is_ready_) {
    return;
  }
  pdal::Option las_reader_filename_opt("filename", lidar_metadata_->filename);
  pdal::Options las_reader_opts;
  las_reader_opts.add(las_reader_filename_opt);
  las_reader_.setOptions(las_reader_opts);

  if (!out_srs_.empty() && out_srs_ != "none") {
    pdal::Options reprojection_opts;
    try {
      reprojection_opts.add("out_srs",
                            lidar_metadata_->transformed_spatial_reference.getWKT());
      pdal::ReprojectionFilter reprojection_filter;
      reprojection_filter.setOptions(reprojection_opts);
      reprojection_filter.setInput(las_reader_);
      reprojection_filter.prepare(point_table_);
      point_view_set_ = reprojection_filter.execute(point_table_);
    } catch (pdal::pdal_error& e) {
      // std::cout << "Encountered error in reprojection: " << e.what() << std::endl;
    }
  } else {
    las_reader_.prepare(point_table_);
    point_view_set_ = las_reader_.execute(point_table_);
  }

  point_view_ = *point_view_set_.begin();
  if (static_cast<int64_t>(las_reader_.header().pointCount()) !=
      lidar_metadata_->num_points) {
    // std::cout << "Encountered mismatch between file header point count and "
    //             "metadata."
    //          << std::endl;
    return;
  }
  is_ready_ = true;
}

#ifdef _WIN32
#pragma comment(linker "/INCLUDE:readData")
#else
__attribute__((__used__))
#endif
void PdalFileMgr::readData(double* x,
                           double* y,
                           double* z,
                           int32_t* intensity,
                           int8_t* return_num,
                           int8_t* num_returns,
                           int8_t* scan_direction_flag,
                           int8_t* edge_of_flight_line_flag,
                           int16_t* classification,
                           int8_t* scan_angle_rank,
                           const int64_t num_points) {
  auto timer = DEBUG_TIMER(__func__);
  for (int64_t p = 0; p < num_points; ++p) {
    x[p] = point_view_->getFieldAs<double>(pdal::Dimension::Id::X, p);
    y[p] = point_view_->getFieldAs<double>(pdal::Dimension::Id::Y, p);
    z[p] = point_view_->getFieldAs<double>(pdal::Dimension::Id::Z, p);
    intensity[p] = point_view_->getFieldAs<int32_t>(pdal::Dimension::Id::Intensity, p);
    return_num[p] = point_view_->getFieldAs<int8_t>(pdal::Dimension::Id::ReturnNumber, p);
    num_returns[p] =
        point_view_->getFieldAs<int8_t>(pdal::Dimension::Id::NumberOfReturns, p);
    scan_direction_flag[p] =
        point_view_->getFieldAs<int8_t>(pdal::Dimension::Id::ScanDirectionFlag, p);
    edge_of_flight_line_flag[p] =
        point_view_->getFieldAs<int8_t>(pdal::Dimension::Id::EdgeOfFlightLine, p);
    classification[p] =
        point_view_->getFieldAs<int16_t>(pdal::Dimension::Id::Classification, p);
    scan_angle_rank[p] =
        point_view_->getFieldAs<int8_t>(pdal::Dimension::Id::ScanAngleRank, p);
  }
}

}  // namespace PdalLoader

#endif  // HAVE_PDAL
#endif  // __CUDACC__
