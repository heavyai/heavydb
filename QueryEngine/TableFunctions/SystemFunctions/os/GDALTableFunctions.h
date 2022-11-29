/*
 * Copyright 2022 HEAVY.AI, Inc.
 */

#pragma once

#ifndef __CUDACC__

#include "QueryEngine/heavydbTypes.h"

// clang-format off
/*
  UDTF: tf_raster_contour_lines__cpu_template(TableFunctionManager mgr,
  Cursor<Column<TLL> lon, Column<TLL> lat, Column<TV> values> raster,
  TextEncodingNone agg_type, float bin_dim_meters | require="bin_dim_meters > 0.0",
  int32_t neighborhood_fill_radius | require="neighborhood_fill_radius >= 0",
  bool fill_only_nulls, TextEncodingNone fill_agg_type, bool flip_latitude, TV contour_interval,
  TV contour_offset) -> Column<GeoLineString> contour_lines, Column<TV> contour_values,
  TLL=[float, double], TV=[float, double]
 */
// clang-format on

template <typename TLL, typename TV>
TEMPLATE_NOINLINE int32_t
tf_raster_contour_lines__cpu_template(TableFunctionManager& mgr,
                                      const Column<TLL>& lon,
                                      const Column<TLL>& lat,
                                      const Column<TV>& values,
                                      const TextEncodingNone& agg_type,
                                      const float bin_dim_meters,
                                      const int32_t neighborhood_fill_radius,
                                      const bool fill_only_nulls,
                                      const TextEncodingNone& fill_agg_type,
                                      const bool flip_latitude,
                                      const TV contour_interval,
                                      const TV contour_offset,
                                      Column<GeoLineString>& contour_lines,
                                      Column<TV>& contour_values);

// clang-format off
/*
  UDTF: tf_raster_contour_lines__cpu_template(TableFunctionManager mgr,
  Cursor<Column<TLL> lon, Column<TLL> lat, Column<TV> values> raster,
  int32_t raster_width | require="raster_width > 0", int32_t raster_height | require="raster_height > 0",
  bool flip_latitude, TV contour_interval, TV contour_offset) -> Column<GeoLineString> contour_lines, Column<TV> contour_values,
  TLL=[float, double], TV=[float, double]
 */
// clang-format on

template <typename TLL, typename TV>
TEMPLATE_NOINLINE int32_t
tf_raster_contour_lines__cpu_template(TableFunctionManager& mgr,
                                      const Column<TLL>& lon,
                                      const Column<TLL>& lat,
                                      const Column<TV>& values,
                                      const int32_t raster_width,
                                      const int32_t raster_height,
                                      const bool flip_latitude,
                                      const TV contour_interval,
                                      const TV contour_offset,
                                      Column<GeoLineString>& contour_lines,
                                      Column<TV>& contour_values);

// clang-format off
/*
  UDTF: tf_raster_contour_polygons__cpu_template(TableFunctionManager mgr,
  Cursor<Column<TLL> lon, Column<TLL> lat, Column<TV> values> raster,
  TextEncodingNone agg_type, float bin_dim_meters | require="bin_dim_meters > 0.0",
  int32_t neighborhood_fill_radius | require="neighborhood_fill_radius >= 0", bool fill_only_nulls, TextEncodingNone fill_agg_type,
  bool flip_latitude, TV contour_interval, TV contour_offset) -> Column<GeoPolygon> contour_polygons, Column<TV> contour_values,
  TLL=[float, double], TV=[float, double]
 */
// clang-format on

template <typename TLL, typename TV>
TEMPLATE_NOINLINE int32_t
tf_raster_contour_polygons__cpu_template(TableFunctionManager& mgr,
                                         const Column<TLL>& lon,
                                         const Column<TLL>& lat,
                                         const Column<TV>& values,
                                         const TextEncodingNone& agg_type,
                                         const float bin_dim_meters,
                                         const int32_t neighborhood_fill_radius,
                                         const bool fill_only_nulls,
                                         const TextEncodingNone& fill_agg_type,
                                         const bool flip_latitude,
                                         const TV contour_interval,
                                         const TV contour_offset,
                                         Column<GeoPolygon>& contour_polygons,
                                         Column<TV>& contour_values);

// clang-format off
/*
  UDTF: tf_raster_contour_polygons__cpu_template(TableFunctionManager mgr,
  Cursor<Column<TLL> lon, Column<TLL> lat, Column<TV> values> raster,
  int32_t raster_width | require="raster_width > 0", int32_t raster_height | require="raster_height > 0",
  bool flip_latitude, TV contour_interval, TV contour_offset) -> Column<GeoPolygon> contour_polygons, Column<TV> contour_values,
  TLL=[float, double], TV=[float, double]
 */
// clang-format on

template <typename TLL, typename TV>
TEMPLATE_NOINLINE int32_t
tf_raster_contour_polygons__cpu_template(TableFunctionManager& mgr,
                                         const Column<TLL>& lon,
                                         const Column<TLL>& lat,
                                         const Column<TV>& values,
                                         const int32_t raster_width,
                                         const int32_t raster_height,
                                         const bool flip_latitude,
                                         const TV contour_interval,
                                         const TV contour_offset,
                                         Column<GeoPolygon>& contour_polygons,
                                         Column<TV>& contour_values);

#include "GDALTableFunctions.cpp"

#endif  // __CUDACC__
