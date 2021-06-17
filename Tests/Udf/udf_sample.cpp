#include <cstdint>
#include <limits>
#include <type_traits>

#include "../../QueryEngine/OmniSciTypes.h"

EXTENSION_NOINLINE
bool array_is_null_double(Array<double> arr) {
  return arr.isNull();
}

EXTENSION_NOINLINE
int32_t array_sz_double(Array<double> arr) {
  return arr.getSize();
}

EXTENSION_NOINLINE
double array_at_double(Array<double> arr, std::size_t idx) {
  return arr(idx);
}

EXTENSION_NOINLINE
bool array_is_null_int32(Array<int32_t> arr) {
  return arr.isNull();
}

EXTENSION_NOINLINE
int32_t array_sz_int32(Array<int32_t> arr) {
  return (int32_t)arr.getSize();
}

EXTENSION_NOINLINE
int32_t array_at_int32(Array<int32_t> arr, std::size_t idx) {
  return arr(idx);
}

EXTENSION_NOINLINE
int8_t array_at_int32_is_null(Array<int32_t> arr, std::size_t idx) {
  return (int8_t)(array_at_int32(arr, idx) == arr.null_value());
}

EXTENSION_NOINLINE
bool array_is_null_int64(Array<int64_t> arr) {
  return arr.isNull();
}

EXTENSION_NOINLINE
int64_t array_sz_int64(Array<int64_t> arr) {
  return arr.getSize();
}

EXTENSION_NOINLINE
int64_t array_at_int64(Array<int64_t> arr, std::size_t idx) {
  return arr(idx);
}

EXTENSION_NOINLINE
int8_t array_at_int64_is_null(Array<int64_t> arr, std::size_t idx) {
  return (int8_t)(array_at_int64(arr, idx) == arr.null_value());
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif

EXTENSION_NOINLINE
Array<double> array_ret_udf(const Array<int32_t> arr, double multiplier) {
  Array<double> ret(arr.getSize());
  for (int64_t i = 0; i < arr.getSize(); i++) {
    if (arr(i) == arr.null_value()) {
      ret[i] = ret.null_value();
    } else {
      ret[i] = static_cast<double>(arr(i)) * multiplier;
    }
  }
  return ret;
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

EXTENSION_NOINLINE
int32_t udf_diff(const int32_t x, const int32_t y) {
  return x - y;
}

EXTENSION_NOINLINE
double ST_X_Point(int8_t* p, int64_t psize, int32_t ic, int32_t isr, int32_t osr);

EXTENSION_NOINLINE
double ST_Y_Point(int8_t* p, int64_t psize, int32_t ic, int32_t isr, int32_t osr);

EXTENSION_NOINLINE
double ST_Perimeter_Polygon(int8_t* poly,
                            int64_t polysize,
                            int32_t* poly_ring_sizes,
                            int64_t poly_num_rings,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr);

EXTENSION_NOINLINE
double ST_Perimeter_Polygon_Geodesic(int8_t* poly,
                                     int64_t polysize,
                                     int32_t* poly_ring_sizes,
                                     int64_t poly_num_rings,
                                     int32_t ic,
                                     int32_t isr,
                                     int32_t osr);

EXTENSION_NOINLINE
double ST_Perimeter_MultiPolygon(int8_t* mpoly_coords,
                                 int64_t mpoly_coords_size,
                                 int32_t* mpoly_ring_sizes,
                                 int64_t mpoly_num_rings,
                                 int32_t* mpoly_poly_sizes,
                                 int64_t mpoly_num_polys,
                                 int32_t ic,
                                 int32_t isr,
                                 int32_t osr);

EXTENSION_NOINLINE
double ST_Area_Polygon(int8_t* poly_coords,
                       int64_t poly_coords_size,
                       int32_t* poly_ring_sizes,
                       int64_t poly_num_rings,
                       int32_t ic,
                       int32_t isr,
                       int32_t osr);

EXTENSION_NOINLINE
double ST_Area_Polygon_Geodesic(int8_t* poly_coords,
                                int64_t poly_coords_size,
                                int32_t* poly_ring_sizes,
                                int64_t poly_num_rings,
                                int32_t ic,
                                int32_t isr,
                                int32_t osr);

EXTENSION_NOINLINE
double ST_Area_MultiPolygon(int8_t* mpoly_coords,
                            int64_t mpoly_coords_size,
                            int32_t* mpoly_ring_sizes,
                            int64_t mpoly_num_rings,
                            int32_t* mpoly_poly_sizes,
                            int64_t mpoly_num_polys,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr);

EXTENSION_NOINLINE
double point_x(GeoPoint p) {
  return ST_X_Point(
      p.ptr, p.getSize(), p.getCompression(), p.getInputSrid(), p.getOutputSrid());
}

EXTENSION_NOINLINE
double point_y(GeoPoint p) {
  return ST_Y_Point(
      p.ptr, p.getSize(), p.getCompression(), p.getInputSrid(), p.getOutputSrid());
}

EXTENSION_NOINLINE
int32_t point_compression(GeoPoint p) {
  return p.getCompression();
}

EXTENSION_NOINLINE
int32_t point_input_srid(GeoPoint p) {
  return p.getInputSrid();
}

EXTENSION_NOINLINE
int32_t point_output_srid(GeoPoint p) {
  return p.getOutputSrid();
}

EXTENSION_NOINLINE
double udf_range(const double high_price, const double low_price) {
  return high_price - low_price;
}

EXTENSION_NOINLINE
int64_t udf_range_int(const int64_t high_price, const int64_t low_price) {
  return high_price - low_price;
}

#ifdef UDF_COMPILER_OPTION
EXTENSION_NOINLINE
int64_t udf_range_int2(const int64_t high_price, const int64_t low_price) {
  return high_price - low_price;
}
#endif

EXTENSION_NOINLINE
double udf_truehigh(const double high_price, const double prev_close_price) {
  return (high_price < prev_close_price) ? prev_close_price : high_price;
}

EXTENSION_NOINLINE
double udf_truelow(const double low_price, const double prev_close_price) {
  return !(prev_close_price < low_price) ? low_price : prev_close_price;
}

EXTENSION_NOINLINE
double udf_truerange(const double high_price,
                     const double low_price,
                     const double prev_close_price) {
  return (udf_truehigh(high_price, prev_close_price) -
          udf_truelow(low_price, prev_close_price));
}

EXTENSION_NOINLINE
double ST_X_LineString(int8_t* l,
                       int64_t lsize,
                       int32_t lindex,
                       int32_t ic,
                       int32_t isr,
                       int32_t osr);

EXTENSION_NOINLINE
double ST_Y_LineString(int8_t* l,
                       int64_t lsize,
                       int32_t lindex,
                       int32_t ic,
                       int32_t isr,
                       int32_t osr);

EXTENSION_NOINLINE
double ST_Length_LineString(int8_t* coords,
                            int64_t coords_sz,
                            int32_t ic,
                            int32_t isr,
                            int32_t osr);

// LineString udf

EXTENSION_NOINLINE
double linestring_x(GeoLineString l, int32_t lindex) {
  return ST_X_LineString(l.ptr,
                         l.getSize(),
                         lindex,
                         l.getCompression(),
                         l.getInputSrid(),
                         l.getOutputSrid());
}

EXTENSION_NOINLINE
double linestring_y(GeoLineString l, int32_t lindex) {
  return ST_Y_LineString(l.ptr,
                         l.getSize(),
                         lindex,
                         l.getCompression(),
                         l.getInputSrid(),
                         l.getOutputSrid());
}

EXTENSION_NOINLINE
double linestring_length(GeoLineString l) {
  return ST_Length_LineString(
      l.ptr, l.getSize(), l.getCompression(), l.getInputSrid(), l.getOutputSrid());
}

// Polygon udf

EXTENSION_NOINLINE
double polygon_area(GeoPolygon p) {
  return ST_Area_Polygon(p.ptr_coords,
                         p.getCoordsSize(),
                         p.getRingSizes(),
                         p.getNumRings(),
                         p.getCompression(),
                         p.getInputSrid(),
                         p.getOutputSrid());
}

EXTENSION_NOINLINE
int32_t polygon_compression(GeoPolygon p) {
  return p.getCompression();
}

EXTENSION_NOINLINE
int32_t polygon_input_srid(GeoPolygon p) {
  return p.getInputSrid();
}

EXTENSION_NOINLINE
int32_t polygon_output_srid(GeoPolygon p) {
  return p.getOutputSrid();
}

EXTENSION_NOINLINE
int32_t polygon_num_rings(GeoPolygon p) {
  return p.getNumRings();
}

// MultiPolygon udf

EXTENSION_NOINLINE
double multipolygon_area(GeoMultiPolygon p) {
  return ST_Area_MultiPolygon(p.ptr_coords,
                              p.getCoordsSize(),
                              p.getRingSizes(),
                              p.getNumRings(),
                              p.getPolygonSizes(),
                              p.getNumPolygons(),
                              p.getCompression(),
                              p.getInputSrid(),
                              p.getOutputSrid());
}

EXTENSION_NOINLINE
double multipolygon_perimeter(GeoMultiPolygon p) {
  return ST_Perimeter_MultiPolygon(p.ptr_coords,
                                   p.getCoordsSize(),
                                   p.getRingSizes(),
                                   p.getNumRings(),
                                   p.getPolygonSizes(),
                                   p.getNumPolygons(),
                                   p.getCompression(),
                                   p.getInputSrid(),
                                   p.getOutputSrid());
}

EXTENSION_NOINLINE
int32_t multipolygon_compression(GeoMultiPolygon p) {
  return p.getCompression();
}

EXTENSION_NOINLINE
int32_t multipolygon_input_srid(GeoMultiPolygon p) {
  return p.getInputSrid();
}

EXTENSION_NOINLINE
int32_t multipolygon_output_srid(GeoMultiPolygon p) {
  return p.getOutputSrid();
}

EXTENSION_NOINLINE
int32_t multipolygon_num_rings(GeoMultiPolygon p) {
  return p.getNumRings();
}
