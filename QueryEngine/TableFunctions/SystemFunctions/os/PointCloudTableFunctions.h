/*
 * Copyright 2022 HEAVY.AI, Inc.
 */

#pragma once

#ifdef HAVE_POINT_CLOUD_TFS
#ifndef __CUDACC__

#include "QueryEngine/heavydbTypes.h"

// clang-format off
/*
  UDTF: tf_point_cloud_metadata__cpu_(TableFunctionManager,
  TextEncodingNone path, double x_min, double x_max | require="x_max > x_min",
  double y_min, double y_max | require="y_max > y_min") ->
  Column<TextEncodingDict> file_path | input_id=args<>,
  Column<TextEncodingDict> file_name | input_id=args<>,
  Column<int32_t> file_source_id, Column<int16_t> version_major, Column<int16_t> version_minor,
  Column<int16_t> creation_year, Column<bool> is_compressed, Column<int64_t> num_points,
  Column <int16_t> num_dims, Column<int16_t> point_len, Column<bool> has_time,
  Column<bool> has_color, Column<bool> has_wave, Column<bool> has_infrared,
  Column<bool> has_14_point_format, Column<int32_t> specified_utm_zone,
  Column<double> x_min_source, Column<double> x_max_source, Column<double> y_min_source,
  Column<double> y_max_source, Column<double> z_min_source, Column<double> z_max_source,
  Column<double> x_min_4326, Column<double> x_max_4326, Column<double> y_min_4326,
  Column<double> y_max_4326, Column<double> z_min_4326, Column<double> z_max_4326
 */
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t tf_point_cloud_metadata__cpu_(TableFunctionManager& mgr,
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
                                      Column<double>& transformed_z_max);
// clang-format off
/*
  UDTF: tf_point_cloud_metadata__cpu_2(TableFunctionManager, TextEncodingNone path) ->
  Column<TextEncodingDict> file_path | input_id=args<>,
  Column<TextEncodingDict> file_name | input_id=args<>,
  Column<int32_t> file_source_id, Column<int16_t> version_major, Column<int16_t> version_minor,
  Column<int16_t> creation_year, Column<bool> is_compressed, Column<int64_t> num_points,
  Column <int16_t> num_dims, Column<int16_t> point_len, Column<bool> has_time,
  Column<bool> has_color, Column<bool> has_wave, Column<bool> has_infrared,
  Column<bool> has_14_point_format, Column<int32_t> specified_utm_zone,
  Column<double> x_min_source, Column<double> x_max_source, Column<double> y_min_source,
  Column<double> y_max_source, Column<double> z_min_source, Column<double> z_max_source,
  Column<double> x_min_4326, Column<double> x_max_4326, Column<double> y_min_4326,
  Column<double> y_max_4326, Column<double> z_min_4326, Column<double> z_max_4326
 */
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t tf_point_cloud_metadata__cpu_2(TableFunctionManager& mgr,
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
                                       Column<double>& transformed_z_max);

// clang-format off
/*
  UDTF: tf_load_point_cloud__cpu_ (TableFunctionManager, TextEncodingNone path,
  TextEncodingNone out_srs, bool use_cache,
  double x_min, double x_max | require="x_max > x_min",
  double y_min, double y_max | require="y_max > y_min") ->
  Column<double> x, Column<double> y, Column<double> z, Column<int32_t> intensity,
  Column<int8_t> return_num, Column<int8_t> num_returns, Column<int8_t> scan_direction_flag,
  Column<int8_t> edge_of_flight_line_flag, Column<int16_t> classification,
  Column<int8_t> scan_angle_rank
 */
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t tf_load_point_cloud__cpu_(TableFunctionManager& mgr,
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
                                  Column<int8_t>& scan_angle_rank);

// clang-format off
/*
  UDTF: tf_load_point_cloud__cpu_2 (TableFunctionManager, TextEncodingNone path) ->
  Column<double> x, Column<double> y, Column<double> z, Column<int32_t> intensity,
  Column<int8_t> return_num, Column<int8_t> num_returns, Column<int8_t> scan_direction_flag,
  Column<int8_t> edge_of_flight_line_flag, Column<int16_t> classification,
  Column<int8_t> scan_angle_rank
 */
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t tf_load_point_cloud__cpu_2(TableFunctionManager& mgr,
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
                                   Column<int8_t>& scan_angle_rank);

// clang-format off
/*
  UDTF: tf_load_point_cloud__cpu_3 (TableFunctionManager, TextEncodingNone path,
  double x_min, double x_max | require="x_max > x_min",
  double y_min, double y_max | require="y_max > y_min") ->
  Column<double> x, Column<double> y, Column<double> z, Column<int32_t> intensity,
  Column<int8_t> return_num, Column<int8_t> num_returns,
  Column<int8_t> scan_direction_flag, Column<int8_t> edge_of_flight_line_flag,
  Column<int16_t> classification, Column<int8_t> scan_angle_rank
 */
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t tf_load_point_cloud__cpu_3(TableFunctionManager& mgr,
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
                                   Column<int8_t>& scan_angle_rank);

// clang-format off
/*
  UDTF: tf_load_point_cloud__cpu_4(TableFunctionManager, TextEncodingNone path, bool use_cache) ->
  Column<double> x, Column<double> y, Column<double> z, Column<int32_t> intensity,
  Column<int8_t> return_num, Column<int8_t> num_returns, Column<int8_t> scan_direction_flag,
  Column<int8_t> edge_of_flight_line_flag, Column<int16_t> classification,
  Column<int8_t> scan_angle_rank
 */
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t tf_load_point_cloud__cpu_4(TableFunctionManager& mgr,
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
                                   Column<int8_t>& scan_angle_rank);

#include "PointCloudTableFunctions.h"

#endif  // __CUDACC__
#endif  // HAVE_POINT_CLOUD_TFS
