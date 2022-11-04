/*
 * Copyright 2022 HEAVY.AI, Inc.
 */

#ifndef __CUDACC__

#include "QueryEngine/TableFunctions/SystemFunctions/os/Shared/TableFunctionsCommon.hpp"

#include <algorithm>
#include <array>

#include "GDALTableFunctions.h"

#include "Geospatial/Compression.h"
#include "Geospatial/GDAL.h"
#include "Shared/scope.h"

#include "cpl_conv.h"
#include "cpl_string.h"
#include "gdal.h"
#include "gdal_alg.h"
#include "gdal_version.h"
#include "ogr_api.h"
#include "ogr_srs_api.h"

#include "ogrsf_frmts.h"

namespace GDALTableFunctions {

//
// general implementation
//

template <typename TV, typename TG>
int32_t tf_raster_contour_impl(TableFunctionManager& mgr,
                               const int32_t raster_width,
                               const int32_t raster_height,
                               const double affine_transform[6],
                               const TV* values,
                               const TV contour_interval,
                               const TV contour_offset,
                               Column<TG>& contour_features,
                               Column<TV>& contour_values) {
  // supported types
  static_assert(std::is_same_v<float, TV> || std::is_same_v<double, TV>);
  static_assert(std::is_same_v<GeoLineString, TG> || std::is_same_v<GeoPolygon, TG>);

  // value type
  std::string value_type_str;
  int pixel_offset{};
  double null_value;
  if constexpr (std::is_same_v<float, TV>) {
    value_type_str = "Float32";
    pixel_offset = sizeof(float);
    null_value = static_cast<double>(std::numeric_limits<float>::min());
  } else if constexpr (std::is_same_v<double, TV>) {
    value_type_str = "Float64";
    pixel_offset = sizeof(double);
    null_value = std::numeric_limits<double>::min();
  }
  CHECK(value_type_str.length());

  auto const num_values = raster_width * raster_height;
  VLOG(1) << "tf_raster_contour: Input raster data has " << num_values << " values, min "
          << *std::min_element(values, values + num_values) << ", max "
          << *std::max_element(values, values + num_values);

  // geo type
  constexpr bool is_polygons = std::is_same_v<GeoPolygon, TG>;

  // construct in-memory-raster string
  std::stringstream mem_ss;
  mem_ss << "MEM:::DATAPOINTER=" << std::hex << values << std::dec
         << ",PIXELS=" << raster_width << ",LINES=" << raster_height
         << ",BANDS=1,DATATYPE=" << value_type_str << ",PIXELOFFSET=" << pixel_offset
         << ",GEOTRANSFORM=" << affine_transform[0] << "/" << affine_transform[1] << "/"
         << affine_transform[2] << "/" << affine_transform[3] << "/"
         << affine_transform[4] << "/" << affine_transform[5];
  std::string mem_str = mem_ss.str();

  // lazy init GDAL
  Geospatial::GDAL::init();

  // things to clean up
  char** options = nullptr;
  GDALDatasetH raster_ds = nullptr;
  GDALDataset* vector_ds = nullptr;

  // auto clean-up on any exit
  ScopeGuard cleanup = [options, raster_ds, vector_ds]() {
    if (options) {
      CSLDestroy(options);
    }
    if (raster_ds) {
      GDALClose(raster_ds);
    }
    if (vector_ds) {
      GDALClose(vector_ds);
    }
  };

  // input dataset
  raster_ds = GDALOpen(mem_str.c_str(), GA_ReadOnly);
  CHECK(raster_ds);

  // input band
  auto raster_band = GDALGetRasterBand(raster_ds, 1);
  CHECK(raster_band);

  // output driver
  auto* memory_driver = GetGDALDriverManager()->GetDriverByName("Memory");
  CHECK(memory_driver);

  // output dataset
  vector_ds = memory_driver->Create("contours", 0, 0, 0, GDT_Unknown, NULL);
  CHECK(vector_ds);

  // output spatial reference
  OGRSpatialReference spatial_reference;
  spatial_reference.importFromEPSG(4326);
  spatial_reference.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

  // output layer
  auto* vector_layer =
      vector_ds->CreateLayer("lines", &spatial_reference, wkbLineString, NULL);
  CHECK(vector_layer);

  // contour values field
  OGRFieldDefn contour_value_field_defn("contour_values", OFTReal);
  vector_layer->CreateField(&contour_value_field_defn);
  auto contour_values_field_index = vector_layer->FindFieldIndex("contour_values", TRUE);
  CHECK_GE(contour_values_field_index, 0);

  // options
  if (contour_values_field_index != -1) {
    if constexpr (is_polygons) {
      options = CSLAppendPrintf(options, "ELEV_FIELD_MIN=%d", contour_values_field_index);
    } else {
      options = CSLAppendPrintf(options, "ELEV_FIELD=%d", contour_values_field_index);
    }
  }
  if (contour_interval != 0.0) {
    options = CSLAppendPrintf(
        options, "LEVEL_INTERVAL=%f", static_cast<float>(contour_interval));
  }
  if (contour_offset != 0.0) {
    options =
        CSLAppendPrintf(options, "LEVEL_BASE=%f", static_cast<float>(contour_offset));
  }
  options = CSLAppendPrintf(options, "NODATA=%.19g", null_value);
  if constexpr (is_polygons) {
    options = CSLAppendPrintf(options, "POLYGONIZE=YES");
  }

  // generate contour
  auto result =
      GDALContourGenerateEx(raster_band, vector_layer, options, nullptr, nullptr);
  if (result != CE_None) {
    return mgr.ERROR_MESSAGE("Contour generation failed");
  }

  // reset the output dataset
  vector_ds->ResetReading();

  // get feature count
  auto const num_features = static_cast<int32_t>(vector_layer->GetFeatureCount());

  VLOG(1) << "tf_raster_contour: GDAL generated " << num_features << " features";

  // did we get any features?
  if (num_features == 0) {
    return mgr.ERROR_MESSAGE("Contour computation did not generate any features");
  }

  // first pass, accumulate total sizes

  int64_t total_num_points{};
  int32_t num_output_features{};

  vector_layer->ResetReading();

  // iterate features
  for (int32_t i = 0; i < num_features; i++) {
    // get feature
    Geospatial::GDAL::FeatureUqPtr feature(vector_layer->GetNextFeature());

    // get geometry
    auto const* geometry = feature->GetGeometryRef();
    CHECK(geometry);

    // get geometry type
    auto const geometry_wkb_type = wkbFlatten(geometry->getGeometryType());

    if constexpr (is_polygons) {
      // check type
      if (geometry_wkb_type != wkbMultiPolygon) {
        return mgr.ERROR_MESSAGE("Geometry WKB type is not MultiPolygon");
      }

      // it's a polygon
      auto const* multipolygon_geometry = dynamic_cast<const OGRMultiPolygon*>(geometry);
      CHECK(multipolygon_geometry);

      // count the points from all rings of all polygons
      auto const num_geometries = multipolygon_geometry->getNumGeometries();
      for (auto p = 0; p < num_geometries; p++) {
        // get this polygon
        auto const* polygon_geometry =
            dynamic_cast<const OGRPolygon*>(multipolygon_geometry->getGeometryRef(p));
        CHECK(polygon_geometry);

        // count points in exterior ring
        auto const* exterior_ring = polygon_geometry->getExteriorRing();
        CHECK(exterior_ring);
        total_num_points += exterior_ring->getNumPoints();

        // count points in interior rings
        for (auto r = 0; r < polygon_geometry->getNumInteriorRings(); r++) {
          auto const* interior_ring = polygon_geometry->getInteriorRing(r);
          CHECK(interior_ring);
          total_num_points += interior_ring->getNumPoints();
        }

        // each polygon is an output feature
        num_output_features++;
      }
    } else {
      // check type
      if (geometry_wkb_type != wkbLineString) {
        return mgr.ERROR_MESSAGE("Geometry WKB type is not Linestring");
      }

      // it's a linestring
      auto const* linestring_geometry = dynamic_cast<const OGRLineString*>(geometry);
      CHECK(linestring_geometry);

      // accumulate these points
      total_num_points += linestring_geometry->getNumPoints();

      // one output feature
      num_output_features++;
    }
  }

  VLOG(1) << "tf_raster_contour: Total points " << total_num_points;

  // size outputs
  mgr.set_output_array_values_total_number(0, total_num_points * 2);
  mgr.set_output_array_values_total_number(1, num_output_features);
  mgr.set_output_row_size(num_output_features);

  // second pass, build output features

  int64_t output_feature_index{};

  vector_layer->ResetReading();

  // iterate features
  for (int32_t i = 0; i < num_features; i++) {
    // get feature
    Geospatial::GDAL::FeatureUqPtr feature(vector_layer->GetNextFeature());

    // get geometry
    auto const* geometry = feature->GetGeometryRef();
    CHECK(geometry);

    if constexpr (is_polygons) {
      // it's a polygon
      auto const* multipolygon_geometry = dynamic_cast<const OGRMultiPolygon*>(geometry);
      CHECK(multipolygon_geometry);

      // process each polygon as an output feature
      auto const num_geometries = multipolygon_geometry->getNumGeometries();
      for (auto p = 0; p < num_geometries; p++) {
        // get this polygon
        auto const* polygon_geometry =
            dynamic_cast<const OGRPolygon*>(multipolygon_geometry->getGeometryRef(p));
        CHECK(polygon_geometry);

        // gather the points from all rings
        std::vector<std::vector<double>> coords;
        {
          std::vector<double> ring_coords;
          auto const* exterior_ring = polygon_geometry->getExteriorRing();
          CHECK(exterior_ring);
          for (int j = 0; j < exterior_ring->getNumPoints(); j++) {
            OGRPoint point;
            exterior_ring->getPoint(j, &point);
            ring_coords.push_back(point.getX());
            ring_coords.push_back(point.getY());
          }
          coords.push_back(ring_coords);
        }
        for (auto r = 0; r < polygon_geometry->getNumInteriorRings(); r++) {
          auto const* interior_ring = polygon_geometry->getInteriorRing(r);
          CHECK(interior_ring);
          std::vector<double> ring_coords;
          for (int j = 0; j < interior_ring->getNumPoints(); j++) {
            OGRPoint point;
            interior_ring->getPoint(j, &point);
            ring_coords.push_back(point.getX());
            ring_coords.push_back(point.getY());
          }
          coords.push_back(ring_coords);
        }

        // set output contour polygon
        auto status = contour_features[output_feature_index].fromCoords(coords);
        if (status != FlatBufferManager::Status::Success) {
          return mgr.ERROR_MESSAGE("fromCoords failed: " + ::toString(status));
        }

        // set output contour value
        contour_values[output_feature_index] =
            static_cast<TV>(feature->GetFieldAsDouble(contour_values_field_index));

        // done
        output_feature_index++;
      }
    } else {
      // it's a linestring
      auto const* linestring_geometry = dynamic_cast<const OGRLineString*>(geometry);
      CHECK(linestring_geometry);

      // unpack linestring
      std::vector<double> coords;
      for (int j = 0; j < linestring_geometry->getNumPoints(); j++) {
        OGRPoint point;
        linestring_geometry->getPoint(j, &point);
        coords.push_back(point.getX());
        coords.push_back(point.getY());
      }

      // set output contour linestring
      auto const* coords_int8_t = reinterpret_cast<const int8_t*>(coords.data());
      contour_features.setItem(
          output_feature_index, coords_int8_t, sizeof(double) * coords.size());

      // set output contour value
      contour_values[output_feature_index] =
          static_cast<TV>(feature->GetFieldAsDouble(contour_values_field_index));

      // done
      output_feature_index++;
    }
  }

  VLOG(1) << "tf_raster_contour: Output " << num_features << " features";

  // done
  return num_output_features;
}

//
// rasterize implementation
//

template <typename TLL, typename TV, typename TG>
int32_t tf_raster_contour_rasterize_impl(TableFunctionManager& mgr,
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
                                         Column<TG>& contour_features,
                                         Column<TV>& contour_values) {
  auto const raster_agg_type = get_raster_agg_type(agg_type, false);
  if (raster_agg_type == RasterAggType::INVALID) {
    auto const error_msg = "Invalid Raster Aggregate Type: " + agg_type.getString();
    return mgr.ERROR_MESSAGE(error_msg);
  }
  auto const raster_fill_agg_type = get_raster_agg_type(fill_agg_type, true);
  if (raster_fill_agg_type == RasterAggType::INVALID) {
    auto const error_msg =
        "Invalid Raster Fill Aggregate Type: " + fill_agg_type.getString();
    return mgr.ERROR_MESSAGE(error_msg);
  }

  GeoRaster<TLL, TV> geo_raster(
      lon, lat, values, raster_agg_type, bin_dim_meters, true, true);

  if (neighborhood_fill_radius > 0) {
    geo_raster.fill_bins_from_neighbors(
        neighborhood_fill_radius, fill_only_nulls, raster_fill_agg_type);
  }

  int64_t raster_width = geo_raster.num_x_bins_;
  int64_t raster_height = geo_raster.num_y_bins_;
  double lon_min = geo_raster.x_min_;
  double lat_min = geo_raster.y_min_;
  double lon_bin_scale = geo_raster.x_scale_input_to_bin_;
  double lat_bin_scale = geo_raster.y_scale_input_to_bin_;

  // validate
  if (raster_width < 2 || raster_height < 2) {
    return mgr.ERROR_MESSAGE("Invalid raster width/height");
  }
  if (lon_bin_scale <= 0.0 || lat_bin_scale <= 0.0) {
    return mgr.ERROR_MESSAGE("Invalid bin scales");
  }

  // build affine transform matrix
  std::array<double, 6> affine_transform;
  affine_transform[0] = lon_min;
  affine_transform[1] = 1.0 / lon_bin_scale;
  affine_transform[2] = 0.0;
  affine_transform[4] = 0.0;
  if (flip_latitude) {
    affine_transform[3] = lat_min + (double(raster_height) / lat_bin_scale);
    affine_transform[5] = -1.0 / lat_bin_scale;
  } else {
    affine_transform[3] = lat_min;
    affine_transform[5] = 1.0 / lat_bin_scale;
  }

  // do it
  return tf_raster_contour_impl<TV, TG>(mgr,
                                        static_cast<int32_t>(raster_width),
                                        static_cast<int32_t>(raster_height),
                                        affine_transform.data(),
                                        geo_raster.z_.data(),
                                        contour_interval,
                                        contour_offset,
                                        contour_features,
                                        contour_values);
}

//
// direct implementation
//

template <typename TLL, typename TV, typename TG>
int32_t tf_raster_contour_direct_impl(TableFunctionManager& mgr,
                                      const Column<TLL>& lon,
                                      const Column<TLL>& lat,
                                      const Column<TV>& values,
                                      const int32_t raster_width,
                                      const int32_t raster_height,
                                      const bool flip_latitude,
                                      const TV contour_interval,
                                      const TV contour_offset,
                                      Column<TG>& contour_features,
                                      Column<TV>& contour_values) {
  // validate
  if (raster_width < 2 || raster_height < 2) {
    return mgr.ERROR_MESSAGE("Invalid raster width/height");
  }

  // expected pixel counts
  auto const num_pixels =
      static_cast<int64_t>(raster_width) * static_cast<int64_t>(raster_height);
  if (lon.size() != num_pixels || lat.size() != num_pixels ||
      values.size() != num_pixels) {
    return mgr.ERROR_MESSAGE("Raster lon/lat/values size mismatch");
  }

  // find lon/lat range
  double lon_min = double(*std::min_element(lon.getPtr(), lon.getPtr() + num_pixels));
  double lon_max = double(*std::max_element(lon.getPtr(), lon.getPtr() + num_pixels));
  double lat_min = double(*std::min_element(lat.getPtr(), lat.getPtr() + num_pixels));
  double lat_max = double(*std::max_element(lat.getPtr(), lat.getPtr() + num_pixels));

  // build affine transform matrix
  // @TODO(se) verify the -1s
  std::array<double, 6> affine_transform;
  affine_transform[0] = lon_min;
  affine_transform[1] = (lon_max - lon_min) / double(raster_width - 1);
  affine_transform[2] = 0.0;
  affine_transform[4] = 0.0;
  if (flip_latitude) {
    affine_transform[3] = lat_max;
    affine_transform[5] = (lat_min - lat_max) / double(raster_height - 1);
  } else {
    affine_transform[3] = lat_min;
    affine_transform[5] = (lat_max - lat_min) / double(raster_height - 1);
  }

  // do it
  return tf_raster_contour_impl<TV, TG>(mgr,
                                        raster_width,
                                        raster_height,
                                        affine_transform.data(),
                                        values.getPtr(),
                                        contour_interval,
                                        contour_offset,
                                        contour_features,
                                        contour_values);
}

}  // namespace GDALTableFunctions

//
// public TFs
//

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
                                      Column<TV>& contour_values) {
  return GDALTableFunctions::tf_raster_contour_rasterize_impl<TLL, TV, GeoLineString>(
      mgr,
      lon,
      lat,
      values,
      agg_type,
      bin_dim_meters,
      neighborhood_fill_radius,
      fill_only_nulls,
      fill_agg_type,
      flip_latitude,
      contour_interval,
      contour_offset,
      contour_lines,
      contour_values);
}

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
                                      Column<TV>& contour_values) {
  return GDALTableFunctions::tf_raster_contour_direct_impl<TLL, TV, GeoLineString>(
      mgr,
      lon,
      lat,
      values,
      raster_width,
      raster_height,
      flip_latitude,
      contour_interval,
      contour_offset,
      contour_lines,
      contour_values);
}

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
                                         Column<TV>& contour_values) {
  return GDALTableFunctions::tf_raster_contour_rasterize_impl<TLL, TV, GeoPolygon>(
      mgr,
      lon,
      lat,
      values,
      agg_type,
      bin_dim_meters,
      neighborhood_fill_radius,
      fill_only_nulls,
      fill_agg_type,
      flip_latitude,
      contour_interval,
      contour_offset,
      contour_polygons,
      contour_values);
}

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
                                         Column<TV>& contour_values) {
  return GDALTableFunctions::tf_raster_contour_direct_impl<TLL, TV, GeoPolygon>(
      mgr,
      lon,
      lat,
      values,
      raster_width,
      raster_height,
      flip_latitude,
      contour_interval,
      contour_offset,
      contour_polygons,
      contour_values);
}

#endif  // __CUDACC__
