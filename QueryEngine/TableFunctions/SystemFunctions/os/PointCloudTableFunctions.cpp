/*
 * Copyright 2022 HEAVY.AI, Inc.
 */

#ifdef HAVE_POINT_CLOUD_TFS
#ifndef __CUDACC__

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/Loaders/PdalLoader.h"
#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.hpp"
#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsDataCache.h"

#include "PointCloudTableFunctions.h"

EXTENSION_NOINLINE_HOST
__attribute__((__used__)) int32_t tf_point_cloud_metadata__cpu_(
    TableFunctionManager& mgr,
    const TextEncodingNone& path,
    const double x_min,
    const double x_max,
    const double y_min,
    const double y_max,
    Column<TextEncodingDict>& file_path,
    Column<TextEncodingDict>& file_name,
    Column<int32_t>& file_source_id,
    Column<int16_t>& version_major,
    Column<int16_t>& version_minor,
    Column<int16_t>& creation_year,
    Column<bool>& is_compressed,
    Column<int64_t>& num_points,
    Column<int16_t>& num_dims,
    Column<int16_t>& point_len,
    Column<bool>& has_time,
    Column<bool>& has_color,
    Column<bool>& has_wave,
    Column<bool>& has_infrared,
    Column<bool>& has_14_point_format,
    Column<int32_t>& specified_utm_zone,
    Column<double>& source_x_min,
    Column<double>& source_x_max,
    Column<double>& source_y_min,
    Column<double>& source_y_max,
    Column<double>& source_z_min,
    Column<double>& source_z_max,
    Column<double>& transformed_x_min,
    Column<double>& transformed_x_max,
    Column<double>& transformed_y_min,
    Column<double>& transformed_y_max,
    Column<double>& transformed_z_min,
    Column<double>& transformed_z_max) {
  auto timer = DEBUG_TIMER(__func__);

  if (x_min >= x_max) {
    return mgr.ERROR_MESSAGE("x_min must be less than x_max");
  }

  if (y_min >= y_max) {
    return mgr.ERROR_MESSAGE("y_min must be less than y_max");
  }

  const std::string las_path(path.getString());
  const std::vector<std::filesystem::path> file_paths =
      FileUtilities::get_fs_paths(las_path);
  const std::string out_srs("EPSG:4326");
  const auto& lidar_file_info = PdalLoader::get_metadata_for_files(file_paths, out_srs);
  const auto& filtered_lidar_file_info =
      filter_files_by_bounds(lidar_file_info.second, x_min, x_max, y_min, y_max);
  const size_t num_filtered_files(filtered_lidar_file_info.second.size());
  mgr.set_output_row_size(num_filtered_files);

  for (size_t file_idx = 0; file_idx < num_filtered_files; ++file_idx) {
    const std::shared_ptr<PdalLoader::LidarMetadata>& lidar_metadata =
        filtered_lidar_file_info.second[file_idx];
    file_path[file_idx] =
        file_path.string_dict_proxy_->getOrAddTransient(lidar_metadata->filename);
    auto last_slash_pos =
        lidar_metadata->filename.rfind("/", lidar_metadata->filename.size() - 1UL);
    if (last_slash_pos == std::string::npos) {
      last_slash_pos = 0;
    } else {
      last_slash_pos++;
    }
    file_name[file_idx] =
        file_name.string_dict_proxy_->getOrAddTransient(lidar_metadata->filename.substr(
            last_slash_pos, lidar_metadata->filename.size() - last_slash_pos));
    file_source_id[file_idx] = lidar_metadata->file_source_id;
    version_major[file_idx] = lidar_metadata->version_major;
    version_minor[file_idx] = lidar_metadata->version_minor;
    creation_year[file_idx] = lidar_metadata->creation_year;
    is_compressed[file_idx] = lidar_metadata->is_compressed;
    num_points[file_idx] = lidar_metadata->num_points;
    num_dims[file_idx] = lidar_metadata->num_dims;
    point_len[file_idx] = lidar_metadata->point_len;
    has_time[file_idx] = lidar_metadata->has_time;
    has_color[file_idx] = lidar_metadata->has_color;
    has_wave[file_idx] = lidar_metadata->has_wave;
    has_infrared[file_idx] = lidar_metadata->has_infrared;
    has_14_point_format[file_idx] = lidar_metadata->has_14_point_format;
    specified_utm_zone[file_idx] = lidar_metadata->specified_utm_zone;
    source_x_min[file_idx] = lidar_metadata->source_x_bounds.first;
    source_x_max[file_idx] = lidar_metadata->source_x_bounds.second;
    source_y_min[file_idx] = lidar_metadata->source_y_bounds.first;
    source_y_max[file_idx] = lidar_metadata->source_y_bounds.second;
    source_z_min[file_idx] = lidar_metadata->source_z_bounds.first;
    source_z_max[file_idx] = lidar_metadata->source_z_bounds.second;
    transformed_x_min[file_idx] = lidar_metadata->transformed_x_bounds.first;
    transformed_x_max[file_idx] = lidar_metadata->transformed_x_bounds.second;
    transformed_y_min[file_idx] = lidar_metadata->transformed_y_bounds.first;
    transformed_y_max[file_idx] = lidar_metadata->transformed_y_bounds.second;
    transformed_z_min[file_idx] = lidar_metadata->transformed_z_bounds.first;
    transformed_z_max[file_idx] = lidar_metadata->transformed_z_bounds.second;
  }
  return num_filtered_files;
}

EXTENSION_NOINLINE_HOST
__attribute__((__used__)) int32_t tf_point_cloud_metadata__cpu_2(
    TableFunctionManager& mgr,
    const TextEncodingNone& path,
    Column<TextEncodingDict>& file_path,
    Column<TextEncodingDict>& file_name,
    Column<int32_t>& file_source_id,
    Column<int16_t>& version_major,
    Column<int16_t>& version_minor,
    Column<int16_t>& creation_year,
    Column<bool>& is_compressed,
    Column<int64_t>& num_points,
    Column<int16_t>& num_dims,
    Column<int16_t>& point_len,
    Column<bool>& has_time,
    Column<bool>& has_color,
    Column<bool>& has_wave,
    Column<bool>& has_infrared,
    Column<bool>& has_14_point_format,
    Column<int32_t>& specified_utm_zone,
    Column<double>& source_x_min,
    Column<double>& source_x_max,
    Column<double>& source_y_min,
    Column<double>& source_y_max,
    Column<double>& source_z_min,
    Column<double>& source_z_max,
    Column<double>& transformed_x_min,
    Column<double>& transformed_x_max,
    Column<double>& transformed_y_min,
    Column<double>& transformed_y_max,
    Column<double>& transformed_z_min,
    Column<double>& transformed_z_max) {
  const double x_min = std::numeric_limits<double>::lowest();
  const double x_max = std::numeric_limits<double>::max();
  const double y_min = std::numeric_limits<double>::lowest();
  const double y_max = std::numeric_limits<double>::max();

  return tf_point_cloud_metadata__cpu_(mgr,
                                       path,
                                       x_min,
                                       x_max,
                                       y_min,
                                       y_max,
                                       file_path,
                                       file_name,
                                       file_source_id,
                                       version_major,
                                       version_minor,
                                       creation_year,
                                       is_compressed,
                                       num_points,
                                       num_dims,
                                       point_len,
                                       has_time,
                                       has_color,
                                       has_wave,
                                       has_infrared,
                                       has_14_point_format,
                                       specified_utm_zone,
                                       source_x_min,
                                       source_x_max,
                                       source_y_min,
                                       source_y_max,
                                       source_z_min,
                                       source_z_max,
                                       transformed_x_min,
                                       transformed_x_max,
                                       transformed_y_min,
                                       transformed_y_max,
                                       transformed_z_min,
                                       transformed_z_max);
}

EXTENSION_NOINLINE_HOST
__attribute__((__used__)) int32_t tf_load_point_cloud__cpu_(
    TableFunctionManager& mgr,
    const TextEncodingNone& path,
    const TextEncodingNone& out_srs,
    const bool use_cache,
    const double x_min,
    const double x_max,
    const double y_min,
    const double y_max,
    Column<double>& x,
    Column<double>& y,
    Column<double>& z,
    Column<int32_t>& intensity,
    Column<int8_t>& return_num,
    Column<int8_t>& num_returns,
    Column<int8_t>& scan_direction_flag,
    Column<int8_t>& edge_of_flight_line_flag,
    Column<int16_t>& classification,
    Column<int8_t>& scan_angle_rank) {
  auto timer = DEBUG_TIMER(__func__);

  if (x_min >= x_max) {
    return mgr.ERROR_MESSAGE("x_min must be less than x_max");
  }

  if (y_min >= y_max) {
    return mgr.ERROR_MESSAGE("y_min must be less than y_max");
  }

  const std::string las_path(path.getString());
  const std::string las_out_srs(out_srs.getString());

  const std::vector<std::filesystem::path> file_paths =
      FileUtilities::get_fs_paths(las_path);
  const auto& lidar_file_info =
      PdalLoader::get_metadata_for_files(file_paths, las_out_srs);
  const auto& filtered_lidar_file_info =
      filter_files_by_bounds(lidar_file_info.second, x_min, x_max, y_min, y_max);
  const size_t num_filtered_files(filtered_lidar_file_info.second.size());
  mgr.set_output_row_size(filtered_lidar_file_info.first);  // contains number of rows

  std::atomic<int64_t> row_offset = 0;
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, num_filtered_files),
      [&](const tbb::blocked_range<size_t>& r) {
        const size_t start_file_idx = r.begin();
        const size_t end_file_idx = r.end();
        for (size_t file_idx = start_file_idx; file_idx != end_file_idx; ++file_idx) {
          const std::shared_ptr<PdalLoader::LidarMetadata>& lidar_metadata =
              filtered_lidar_file_info.second[file_idx];
          const int64_t num_points = lidar_metadata->num_points;
          const auto& las_fs_path = std::filesystem::path(lidar_metadata->filename);
          const auto file_status = std::filesystem::status(las_fs_path);
          if (!std::filesystem::is_regular_file(file_status)) {
            continue;
          }

          const std::vector<std::pair<std::string, size_t>> sub_keys_and_byte_sizes = {
              std::make_pair("x", 8),
              std::make_pair("y", 8),
              std::make_pair("z", 8),
              std::make_pair("intensity", 4),
              std::make_pair("return_num", 1),
              std::make_pair("num_returns", 1),
              std::make_pair("scan_direction_flag", 1),
              std::make_pair("edge_of_flight_line_flag", 1),
              std::make_pair("classification", 2),
              std::make_pair("scan_angle_rank", 1)};

          if (use_cache && PdalLoader::keys_are_cached(lidar_metadata->filename,
                                                       las_out_srs,
                                                       num_points,
                                                       sub_keys_and_byte_sizes)) {
            const int64_t offset =
                row_offset.fetch_add(num_points, std::memory_order_relaxed);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(lidar_metadata->filename, las_out_srs, "x"),
                x.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(lidar_metadata->filename, las_out_srs, "y"),
                y.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(lidar_metadata->filename, las_out_srs, "z"),
                z.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(
                    lidar_metadata->filename, las_out_srs, "intensity"),
                intensity.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(
                    lidar_metadata->filename, las_out_srs, "return_num"),
                return_num.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(
                    lidar_metadata->filename, las_out_srs, "num_returns"),
                num_returns.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(
                    lidar_metadata->filename, las_out_srs, "scan_direction_flag"),
                scan_direction_flag.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(
                    lidar_metadata->filename, las_out_srs, "edge_of_flight_line_flag"),
                edge_of_flight_line_flag.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(
                    lidar_metadata->filename, las_out_srs, "classification"),
                classification.ptr_ + offset);
            PdalLoader::data_cache.getDataForKey(
                PdalLoader::make_cache_key(
                    lidar_metadata->filename, las_out_srs, "scan_angle_rank"),
                scan_angle_rank.ptr_ + offset);
            continue;
          }

          PdalLoader::PdalFileMgr pdal_file_mgr(lidar_metadata, las_out_srs);
          pdal_file_mgr.openAndPrepareFile();
          if (!pdal_file_mgr.isReady()) {
            continue;
            // const std::string error_msg = "tf_load_point_cloud: PDAL error opening
            // file " + lidar_metadata->filename; return
            // mgr.ERROR_MESSAGE(error_msg.c_str());
          }
          const int64_t offset =
              row_offset.fetch_add(lidar_metadata->num_points, std::memory_order_relaxed);
          pdal_file_mgr.readData(x.ptr_ + offset,
                                 y.ptr_ + offset,
                                 z.ptr_ + offset,
                                 intensity.ptr_ + offset,
                                 return_num.ptr_ + offset,
                                 num_returns.ptr_ + offset,
                                 scan_direction_flag.ptr_ + offset,
                                 edge_of_flight_line_flag.ptr_ + offset,
                                 classification.ptr_ + offset,
                                 scan_angle_rank.ptr_ + offset);

          if (use_cache) {
            try {
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(lidar_metadata->filename, las_out_srs, "x"),
                  x.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(lidar_metadata->filename, las_out_srs, "y"),
                  y.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(lidar_metadata->filename, las_out_srs, "z"),
                  z.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(
                      lidar_metadata->filename, las_out_srs, "intensity"),
                  intensity.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(
                      lidar_metadata->filename, las_out_srs, "return_num"),
                  return_num.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(
                      lidar_metadata->filename, las_out_srs, "num_returns"),
                  num_returns.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(
                      lidar_metadata->filename, las_out_srs, "scan_direction_flag"),
                  scan_direction_flag.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(
                      lidar_metadata->filename, las_out_srs, "edge_of_flight_line_flag"),
                  edge_of_flight_line_flag.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(
                      lidar_metadata->filename, las_out_srs, "classification"),
                  classification.ptr_ + offset,
                  lidar_metadata->num_points);
              PdalLoader::data_cache.putDataForKey(
                  PdalLoader::make_cache_key(
                      lidar_metadata->filename, las_out_srs, "scan_angle_rank"),
                  scan_angle_rank.ptr_ + offset,
                  lidar_metadata->num_points);
            } catch (std::runtime_error& e) {
              continue;
              // return mgr.ERROR_MESSAGE("tf_tf_load_point_cloud: Error writing data to
              // cache");
            }
          }
        }
      });
  return row_offset;
}

EXTENSION_NOINLINE_HOST
__attribute__((__used__)) int32_t tf_load_point_cloud__cpu_2(
    TableFunctionManager& mgr,
    const TextEncodingNone& filename,
    Column<double>& x,
    Column<double>& y,
    Column<double>& z,
    Column<int32_t>& intensity,
    Column<int8_t>& return_num,
    Column<int8_t>& num_returns,
    Column<int8_t>& scan_direction_flag,
    Column<int8_t>& edge_of_flight_line_flag,
    Column<int16_t>& classification,
    Column<int8_t>& scan_angle_rank) {
  std::string pdal_out_srs("EPSG:4326");
  TextEncodingNone out_srs;
  out_srs.ptr_ = pdal_out_srs.data();
  out_srs.size_ = pdal_out_srs.size();
  const double x_min = std::numeric_limits<double>::lowest();
  const double x_max = std::numeric_limits<double>::max();
  const double y_min = std::numeric_limits<double>::lowest();
  const double y_max = std::numeric_limits<double>::max();
  return tf_load_point_cloud__cpu_(mgr,
                                   filename,
                                   out_srs,
                                   true,
                                   x_min,
                                   x_max,
                                   y_min,
                                   y_max,
                                   x,
                                   y,
                                   z,
                                   intensity,
                                   return_num,
                                   num_returns,
                                   scan_direction_flag,
                                   edge_of_flight_line_flag,
                                   classification,
                                   scan_angle_rank);
}

EXTENSION_NOINLINE_HOST
__attribute__((__used__)) int32_t tf_load_point_cloud__cpu_3(
    TableFunctionManager& mgr,
    const TextEncodingNone& filename,
    const double x_min,
    const double x_max,
    const double y_min,
    const double y_max,
    Column<double>& x,
    Column<double>& y,
    Column<double>& z,
    Column<int32_t>& intensity,
    Column<int8_t>& return_num,
    Column<int8_t>& num_returns,
    Column<int8_t>& scan_direction_flag,
    Column<int8_t>& edge_of_flight_line_flag,
    Column<int16_t>& classification,
    Column<int8_t>& scan_angle_rank) {
  std::string pdal_out_srs("EPSG:4326");
  TextEncodingNone out_srs;
  out_srs.ptr_ = pdal_out_srs.data();
  out_srs.size_ = pdal_out_srs.size();
  return tf_load_point_cloud__cpu_(mgr,
                                   filename,
                                   out_srs,
                                   true,
                                   x_min,
                                   x_max,
                                   y_min,
                                   y_max,
                                   x,
                                   y,
                                   z,
                                   intensity,
                                   return_num,
                                   num_returns,
                                   scan_direction_flag,
                                   edge_of_flight_line_flag,
                                   classification,
                                   scan_angle_rank);
}

EXTENSION_NOINLINE_HOST
__attribute__((__used__)) int32_t tf_load_point_cloud__cpu_4(
    TableFunctionManager& mgr,
    const TextEncodingNone& filename,
    const bool use_cache,
    Column<double>& x,
    Column<double>& y,
    Column<double>& z,
    Column<int32_t>& intensity,
    Column<int8_t>& return_num,
    Column<int8_t>& num_returns,
    Column<int8_t>& scan_direction_flag,
    Column<int8_t>& edge_of_flight_line_flag,
    Column<int16_t>& classification,
    Column<int8_t>& scan_angle_rank) {
  std::string pdal_out_srs("EPSG:4326");
  TextEncodingNone out_srs;
  out_srs.ptr_ = pdal_out_srs.data();
  out_srs.size_ = pdal_out_srs.size();
  const double x_min = std::numeric_limits<double>::lowest();
  const double x_max = std::numeric_limits<double>::max();
  const double y_min = std::numeric_limits<double>::lowest();
  const double y_max = std::numeric_limits<double>::max();
  return tf_load_point_cloud__cpu_(mgr,
                                   filename,
                                   out_srs,
                                   use_cache,
                                   x_min,
                                   x_max,
                                   y_min,
                                   y_max,
                                   x,
                                   y,
                                   z,
                                   intensity,
                                   return_num,
                                   num_returns,
                                   scan_direction_flag,
                                   edge_of_flight_line_flag,
                                   classification,
                                   scan_angle_rank);
}

#endif  // __CUDACC__
#endif  // HAVE_TF_POINT_CLOUD_TFS
