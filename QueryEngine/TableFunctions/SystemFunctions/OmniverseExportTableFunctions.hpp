/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#ifndef __CUDACC__

#include "QueryEngine/TableFunctions/SystemFunctions/OmniverseConnector.h"
#include "QueryEngine/TableFunctions/SystemFunctions/Shared/TableFunctionsCommon.hpp"
#include "QueryEngine/TableFunctions/SystemFunctions/Shared/TableFunctionsStats.hpp"
#include "QueryEngine/heavydbTypes.h"

// clang-format off
/*
  UDTF: tf_export_ov_terrain_texture__cpu_template(TableFunctionManager, 
  Cursor<Column<TXY> x, Column<TXY> y, ColumnList<TA> attrs> data,
  TextEncodingNone stats_requests) ->
  Column<TextEncodingDict> stat_name | input_id=args<>, Column<TA> stat_val, Column<Array<int8_t>> png_bytes, TXY=[double], TA=[float]
*/
// clang-format on

template <typename TXY, typename TA>
NEVER_INLINE HOST int32_t
tf_export_ov_terrain_texture__cpu_template(TableFunctionManager& mgr,
                                           const Column<TXY>& x,
                                           const Column<TXY>& y,
                                           const ColumnList<TA>& attrs,
                                           const TextEncodingNone& stats_requests_json,
                                           Column<TextEncodingDict>& stat_names,
                                           Column<TA>& stat_vals,
                                           Column<Array<int8_t>>& png_bytes) {
  const int64_t num_points = x.size();

  std::vector<const TA*> attr_ptrs;
  for (int64_t attr_idx = 0; attr_idx < attrs.numCols(); ++attr_idx) {
    attr_ptrs.emplace_back(reinterpret_cast<const TA*>(attrs.ptrs_[attr_idx]));
  }

  // fetch values from the key/value store
  int64_t geo_raster_num_x_bins{}, geo_raster_num_y_bins{};
  double geo_raster_x_min{}, geo_raster_y_min{};
  double geo_raster_x_scale_input_to_bin{}, geo_raster_y_scale_input_to_bin{};
  mgr.get_metadata("geo_raster_num_x_bins", geo_raster_num_x_bins);
  mgr.get_metadata("geo_raster_num_y_bins", geo_raster_num_y_bins);
  mgr.get_metadata("geo_raster_x_min", geo_raster_x_min);
  mgr.get_metadata("geo_raster_y_min", geo_raster_y_min);
  mgr.get_metadata("geo_raster_x_scale_input_to_bin", geo_raster_x_scale_input_to_bin);
  mgr.get_metadata("geo_raster_y_scale_input_to_bin", geo_raster_y_scale_input_to_bin);

#if DEBUG_GEO_RASTER
  std::cout << "DEBUG: geo_raster_num_x_bins = " << geo_raster_num_x_bins << std::endl;
  std::cout << "DEBUG: geo_raster_num_y_bins = " << geo_raster_num_y_bins << std::endl;
  std::cout << "DEBUG: geo_raster_x_min = " << geo_raster_x_min << std::endl;
  std::cout << "DEBUG: geo_raster_y_min = " << geo_raster_y_min << std::endl;
  std::cout << "DEBUG: geo_raster_x_scale_input_to_bin = "
            << geo_raster_x_scale_input_to_bin << std::endl;
  std::cout << "DEBUG: geo_raster_y_scale_input_to_bin = "
            << geo_raster_y_scale_input_to_bin << std::endl;
#endif

  try {
    // export texture as PNG into allocated buffer
    int32_t num_png_bytes{};
    int8_t* these_png_bytes = omniverse_connector::export_terrain_texture(
        geo_raster_num_x_bins,
        geo_raster_num_y_bins,
        static_cast<TXY>(geo_raster_x_min),
        static_cast<TXY>(geo_raster_y_min),
        static_cast<TXY>(geo_raster_x_scale_input_to_bin),
        static_cast<TXY>(geo_raster_y_scale_input_to_bin),
        x.ptr_,
        y.ptr_,
        attr_ptrs.data(),
        attr_ptrs.size(),
        num_points,
        num_png_bytes);

    // configure output array columns and row
    // png_bytes is output column 2
    // only the first row has content, so the total is still just num_png_bytes
    mgr.set_output_array_values_total_number(2, num_png_bytes);

    // do stats
    auto stats_requests = parse_stats_requests_json(stats_requests_json, attrs.numCols());
    compute_stats_requests(attrs, stats_requests);
    const int64_t num_stats = stats_requests.size();

    // we need at least one output row
    auto const num_rows = std::max(num_stats, int64_t(1));
    mgr.set_output_row_size(num_rows);

    // populate stats
    if (num_stats > 0) {
      populate_output_stats_cols(stat_names, stat_vals, stats_requests);
    } else {
      // no stats, but put something in row 0 of those columns
      stat_names[0] = 0;
      stat_vals[0] = static_cast<TA>(0);
    }

    // copy PNG bytes from allocated buffer to output column (row 0)
    Array<int8_t> png_bytes_array(these_png_bytes, num_png_bytes);
    png_bytes.setItem(0, png_bytes_array);

    // set empty array in all other rows
    Array<int8_t> empty_bytes_array(nullptr, 0);
    for (int64_t row = 1; row < num_stats; row++) {
      png_bytes.setItem(row, empty_bytes_array);
    }

    // free the allocated buffer
    omniverse_connector::export_free(these_png_bytes);

    // done, return rows
    return num_rows;
  } catch (std::exception& e) {
    const std::string err_msg = e.what();
    return mgr.ERROR_MESSAGE(err_msg);
  }
}

// clang-format off
/*
  UDTF: tf_export_ov_buildings_texture__cpu_template(TableFunctionManager, 
  Cursor<Column<int64_t> rowid, ColumnList<TA> attrs> data) ->
  Column<int32_t> num_patches_xy, Column<Array<int8_t>> png_bytes, TA=[float]
*/
// clang-format on

template <typename TA>
NEVER_INLINE HOST int32_t
tf_export_ov_buildings_texture__cpu_template(TableFunctionManager& mgr,
                                             const Column<int64_t>& rowid,
                                             const ColumnList<TA>& attrs,
                                             Column<int32_t>& num_patches_xy,
                                             Column<Array<int8_t>>& png_bytes) {
  const int64_t num_rows = rowid.size();

  std::vector<const TA*> attr_ptrs;
  for (int64_t attr_idx = 0; attr_idx < attrs.numCols(); ++attr_idx) {
    attr_ptrs.emplace_back(reinterpret_cast<const TA*>(attrs.ptrs_[attr_idx]));
  }

  try {
    // export texture as PNG into allocated buffer
    int32_t out_num_patches_xy{};
    int32_t num_png_bytes{};
    int8_t* these_png_bytes =
        omniverse_connector::export_buildings_texture(rowid.getPtr(),
                                                      attr_ptrs.data(),
                                                      attr_ptrs.size(),
                                                      num_rows,
                                                      out_num_patches_xy,
                                                      num_png_bytes);

    // configure output array columns and row
    mgr.set_output_array_values_total_number(1, num_png_bytes);
    mgr.set_output_row_size(1);

    // copy PNG bytes from allocated buffer to output column
    Array<int8_t> png_bytes_array(these_png_bytes, num_png_bytes);
    png_bytes.setItem(0, png_bytes_array);

    // free the allocated buffer
    omniverse_connector::export_free(these_png_bytes);

    // set the num_patches_xy output value
    num_patches_xy[0] = out_num_patches_xy;

    // done, return one row
    return 1;
  } catch (std::exception& e) {
    const std::string err_msg = e.what();
    return mgr.ERROR_MESSAGE(err_msg);
  }
}

// clang-format off
/*
  UDTF: tf_export_ov_grid_mesh__cpu_template(TableFunctionManager,
  Cursor<Column<TXY> x, Column<TXY> y, Column<TZ> z> data,
  int32_t mesh_type, double tile_origin_x, double tile_origin_y) ->
  Column<Array<int32_t>> mesh_data_out, TXY=[double], TZ=[float]
*/
// clang-format on

template <typename TXY, typename TZ>
NEVER_INLINE HOST int32_t
tf_export_ov_grid_mesh__cpu_template(TableFunctionManager& mgr,
                                     const Column<TXY>& x,
                                     const Column<TXY>& y,
                                     const Column<TZ>& z,
                                     const int32_t mesh_type,
                                     const double tile_origin_x,
                                     const double tile_origin_y,
                                     Column<Array<int32_t>>& mesh_data_out) {
  const int64_t num_points = x.size();

  // fetch values from the key/value store
  int64_t geo_raster_num_x_bins{}, geo_raster_num_y_bins{};
  double geo_raster_x_min{}, geo_raster_y_min{};
  double geo_raster_x_scale_input_to_bin{}, geo_raster_y_scale_input_to_bin{};
  mgr.get_metadata("geo_raster_num_x_bins", geo_raster_num_x_bins);
  mgr.get_metadata("geo_raster_num_y_bins", geo_raster_num_y_bins);
  mgr.get_metadata("geo_raster_x_min", geo_raster_x_min);
  mgr.get_metadata("geo_raster_y_min", geo_raster_y_min);
  mgr.get_metadata("geo_raster_x_scale_input_to_bin", geo_raster_x_scale_input_to_bin);
  mgr.get_metadata("geo_raster_y_scale_input_to_bin", geo_raster_y_scale_input_to_bin);

#if DEBUG_GEO_RASTER
  std::cout << "DEBUG: geo_raster_num_x_bins = " << geo_raster_num_x_bins << std::endl;
  std::cout << "DEBUG: geo_raster_num_y_bins = " << geo_raster_num_y_bins << std::endl;
  std::cout << "DEBUG: geo_raster_x_min = " << geo_raster_x_min << std::endl;
  std::cout << "DEBUG: geo_raster_y_min = " << geo_raster_y_min << std::endl;
  std::cout << "DEBUG: geo_raster_x_scale_input_to_bin = "
            << geo_raster_x_scale_input_to_bin << std::endl;
  std::cout << "DEBUG: geo_raster_y_scale_input_to_bin = "
            << geo_raster_y_scale_input_to_bin << std::endl;
#endif

  try {
    // export texture as PNG into allocated buffer
    int32_t mesh_data_size{};
    int32_t* mesh_data = omniverse_connector::export_grid_mesh(
        static_cast<omniverse_connector::MeshType>(mesh_type),
        tile_origin_x,
        tile_origin_y,
        geo_raster_num_x_bins,
        geo_raster_num_y_bins,
        static_cast<TXY>(geo_raster_x_min),
        static_cast<TXY>(geo_raster_y_min),
        static_cast<TXY>(geo_raster_x_scale_input_to_bin),
        static_cast<TXY>(geo_raster_y_scale_input_to_bin),
        x.ptr_,
        y.ptr_,
        z.ptr_,
        num_points,
        mesh_data_size);

    // configure output column and row
    mgr.set_output_array_values_total_number(0, mesh_data_size);
    mgr.set_output_row_size(1);

    // copy mesh data from allocated buffer to output column
    Array<int32_t> mesh_data_array(mesh_data, mesh_data_size);
    mesh_data_out.setItem(0, mesh_data_array);

    // free the allocated buffer
    omniverse_connector::export_free(mesh_data);

    // done, return one row
    return 1;
  } catch (std::exception& e) {
    const std::string err_msg = e.what();
    return mgr.ERROR_MESSAGE(err_msg);
  }
}

// clang-format off
/*
  UDTF: tf_export_ov_buildings_polygons__cpu_template(TableFunctionManager,
  Cursor<Column<int64_t> rowid, Column<G> polys, Column<double> centroids_x, Column<double> centroids_y,
  Column<float> base_elevations, Column<Array<float>> heights> polygons) ->
  Column<Array<int32>> mesh_data_out, G=[GeoPolygon,GeoMultiPolygon]
*/
// clang-format on

template <typename G>
NEVER_INLINE HOST int32_t
tf_export_ov_buildings_polygons__cpu_template(TableFunctionManager& mgr,
                                              const Column<int64_t>& rowids,
                                              const Column<G>& polys,
                                              const Column<double>& centroids_x,
                                              const Column<double>& centroids_y,
                                              const Column<float>& base_elevations,
                                              const Column<Array<float>>& heights,
                                              Column<Array<int32_t>>& mesh_data_out) {
  try {
    auto const num_rows = rowids.size();
    CHECK_EQ(num_rows, polys.size());
    CHECK_EQ(num_rows, centroids_x.size());
    CHECK_EQ(num_rows, centroids_y.size());
    CHECK_EQ(num_rows, base_elevations.size());
    CHECK_EQ(num_rows, heights.size());

    // convert incoming polygons to simple arrays
    std::vector<std::vector<double>> raw_coords(num_rows);
    std::vector<std::vector<int32_t>> raw_ring_sizes(num_rows);
    std::vector<std::vector<int32_t>> raw_poly_rings(num_rows);
    std::vector<std::vector<float>> raw_heights(num_rows);
    for (int64_t row = 0; row < num_rows; row++) {
      auto rowid = rowids[row];
      if (polys.isNull(row)) {
        auto const err_msg = "Null Poly at rowid " + std::to_string(rowid);
        return mgr.ERROR_MESSAGE(err_msg);
      }
      auto const num_heights = heights[row].getSize();
      if constexpr (std::is_same_v<G, GeoMultiPolygon>) {
        auto const num_polys = polys[row].size();
        if (static_cast<int64_t>(num_polys) != num_heights) {
          auto const err_msg =
              "Poly/Height count mismatch at rowid " + std::to_string(rowid);
          return mgr.ERROR_MESSAGE(err_msg);
        }
        for (size_t i = 0; i < num_polys; i++) {
          auto const poly = polys[row][i];
          auto const num_rings = poly.size();
          for (size_t ring = 0; ring < num_rings; ring++) {
            auto const coords = poly[ring].toCoords();
            auto const num_coords = coords.size();
            raw_coords[row].insert(raw_coords[row].end(), coords.begin(), coords.end());
            raw_ring_sizes[row].push_back(num_coords / 2);
          }
          raw_poly_rings[row].push_back(num_rings);
        }
      } else {
        if (num_heights != 1) {
          auto const err_msg =
              "Heights array must only have 1 element at rowid " + std::to_string(rowid);
          return mgr.ERROR_MESSAGE(err_msg);
        }
        auto const poly = polys[row];
        auto const num_rings = poly.size();
        for (size_t ring = 0; ring < num_rings; ring++) {
          auto const coords = poly[ring].toCoords();
          auto const num_coords = coords.size();
          raw_coords[row].insert(raw_coords[row].end(), coords.begin(), coords.end());
          raw_ring_sizes[row].push_back(num_coords / 2);
        }
        raw_poly_rings[row].push_back(num_rings);
      }
      for (int64_t i = 0; i < num_heights; i++) {
        raw_heights[row].push_back(heights[row][i]);
      }
    }

    // get mesh data
    int32_t** row_mesh_data{};
    int32_t* row_mesh_data_size{};
    omniverse_connector::export_polygons(num_rows,
                                         rowids.getPtr(),
                                         raw_coords,
                                         raw_ring_sizes,
                                         raw_poly_rings,
                                         centroids_x.getPtr(),
                                         centroids_y.getPtr(),
                                         base_elevations.getPtr(),
                                         raw_heights,
                                         row_mesh_data,
                                         row_mesh_data_size);

    // did we get anything?
    if (!row_mesh_data || !row_mesh_data_size) {
      mgr.set_output_array_values_total_number(0, 0);
      mgr.set_output_row_size(0);
      return 0LL;
    }

    // get total size
    int64_t total_mesh_data_size{};
    for (int64_t row = 0; row < num_rows; row++) {
      total_mesh_data_size += row_mesh_data_size[row];
    }

    // allocate space for outputs
    mgr.set_output_array_values_total_number(0, total_mesh_data_size);
    mgr.set_output_row_size(num_rows);

    // copy mesh data to outputs
    for (int64_t row = 0; row < num_rows; row++) {
      Array<int32_t> mesh_data_array(row_mesh_data[row], row_mesh_data_size[row]);
      mesh_data_out.setItem(row, mesh_data_array);
    }

    // free mesh data
    for (int64_t row = 0; row < num_rows; row++) {
      omniverse_connector::export_free(row_mesh_data[row]);
    }
    omniverse_connector::export_free(row_mesh_data);
    omniverse_connector::export_free(row_mesh_data_size);

    // done
    return num_rows;
  } catch (std::exception& e) {
    const std::string err_msg = e.what();
    return mgr.ERROR_MESSAGE(err_msg);
  }
}

// clang-format off
/*
  UDTF: tf_merge_building_polygons__cpu_template(TableFunctionManager,
  Cursor<Column<G> poly, Column<float> base_elevation, Column<float> height, Column<int32_t> group_id> buildings) ->
  Column<GeoMultiPolygon> mpolys_out, Column<float> base_elevations_out, Column<float> roof_elevations_out,
  Column<Array<float>> heights_out, Column<int32_t> group_ids_out, G=[GeoPolygon,GeoMultiPolygon]
*/
// clang-format on

template <typename G>
NEVER_INLINE HOST int32_t
tf_merge_building_polygons__cpu_template(TableFunctionManager& mgr,
                                         const Column<G>& polys,
                                         const Column<float>& base_elevations,
                                         const Column<float>& heights,
                                         const Column<int32_t>& group_ids,
                                         Column<GeoMultiPolygon>& mpolys_out,
                                         Column<float>& base_elevations_out,
                                         Column<float>& roof_elevations_out,
                                         Column<Array<float>>& heights_out,
                                         Column<int32_t>& group_ids_out) {
  try {
    auto const num_input_rows = polys.size();
    CHECK_EQ(num_input_rows, base_elevations.size());
    CHECK_EQ(num_input_rows, heights.size());
    CHECK_EQ(num_input_rows, group_ids.size());

    // gather all input indices for each unique group_id
    // concatenate input polys (or first of multipolys) of each group to multipoly out
    // concatenate input heights of each group to height array out
    // min base_elevation of each group to base elevation out
    // max (base_elevation + height) of each group to roof elevation out
    // copy group_id to out

    // first, build map of group_id to input indices
    std::map<int32_t, std::vector<int64_t>> group_map;
    for (int64_t input_row = 0; input_row < num_input_rows; input_row++) {
      auto const group_id = group_ids[input_row];
      auto itr = group_map.find(group_id);
      if (itr != group_map.end()) {
        itr->second.push_back(input_row);
      } else {
        group_map.try_emplace(group_id, std::vector<int64_t>({input_row}));
      }
    }

    // size of the map is the number of output rows (groups)
    auto const num_output_rows = group_map.size();

    // allocate space for outputs
    mgr.set_output_item_values_total_number(0, polys.getNofValues());
    mgr.set_output_item_values_total_number(1, num_output_rows);
    mgr.set_output_item_values_total_number(2, num_output_rows);
    mgr.set_output_array_values_total_number(3, num_input_rows);
    mgr.set_output_item_values_total_number(4, num_output_rows);
    mgr.set_output_row_size(num_output_rows);

    // populate outputs
    int32_t output_index = 0;
    for (auto itr = group_map.begin(); itr != group_map.end(); itr++, output_index++) {
      auto const& input_indices = itr->second;
      auto const num_input_indices = input_indices.size();

      // multipoly
      std::vector<std::vector<std::vector<double>>> coords;
      coords.reserve(num_input_indices);
      for (uint32_t i = 0; i < num_input_indices; i++) {
        if constexpr (std::is_same_v<G, GeoMultiPolygon>) {
          if (polys[input_indices[i]].size() != 1) {
            return mgr.ERROR_MESSAGE("Input MULTIPOLYGONs must have only one polygon");
          }
          coords.push_back(polys[input_indices[i]][0].toCoords());
        } else {
          coords.push_back(polys[input_indices[i]].toCoords());
        }
      }
      auto status = mpolys_out[output_index].fromCoords(coords);
      if (status != FlatBufferManager::Status::Success) {
        return mgr.ERROR_MESSAGE("fromCoords failed: " + ::toString(status));
      }

      // base elevation (min of base)
      std::vector<float> base_elevations_array(num_input_indices);
      for (uint32_t i = 0; i < num_input_indices; i++) {
        base_elevations_array[i] = base_elevations[input_indices[i]];
      }
      auto const base_elevation_min =
          *std::min_element(base_elevations_array.begin(), base_elevations_array.end());
      base_elevations_out[output_index] = base_elevation_min;

      // roof elevation (max of base + height)
      std::vector<float> roof_elevations_array(num_input_indices);
      for (uint32_t i = 0; i < num_input_indices; i++) {
        roof_elevations_array[i] =
            base_elevations[input_indices[i]] + heights[input_indices[i]];
      }
      auto const roof_elevation_max =
          *std::max_element(roof_elevations_array.begin(), roof_elevations_array.end());
      roof_elevations_out[output_index] = roof_elevation_max;

      // array of heights
      // adjust by base offset from min, such that height is height above base min
      Array<float> height_array(num_input_indices);
      for (uint32_t i = 0; i < num_input_indices; i++) {
        auto const base_offset = base_elevations[input_indices[i]] - base_elevation_min;
        height_array[i] = heights[input_indices[i]] + base_offset;
      }
      heights_out.setItem(output_index, height_array);

      // group_id
      auto const group_id = itr->first;
      group_ids_out[output_index] = group_id;
    }

    // done
    return num_output_rows;
  } catch (std::exception& e) {
    const std::string err_msg = e.what();
    return mgr.ERROR_MESSAGE(err_msg);
  }
}

#endif  // #ifndef __CUDACC__
