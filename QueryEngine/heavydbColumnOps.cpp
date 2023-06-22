/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include "heavydbTypes.h"

// -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
// expose Column<GeoPoint> methods

extern "C" DEVICE RUNTIME_EXPORT void ColumnGeoPoint_getItem(const Column<GeoPoint>& col,
                                                             int64_t index,
                                                             int32_t output_srid,
                                                             Geo::Point2D& result) {
  result = col.getItem(index, output_srid);
}

extern "C" DEVICE RUNTIME_EXPORT bool ColumnGeoPoint_isNull(const Column<GeoPoint>& col,
                                                            int64_t index) {
  return col.isNull(index);
}

extern "C" DEVICE RUNTIME_EXPORT void ColumnGeoPoint_setNull(Column<GeoPoint>& col,
                                                             int64_t index) {
  col.setNull(index);
}

extern "C" DEVICE RUNTIME_EXPORT void ColumnGeoPoint_setItem(Column<GeoPoint>& col,
                                                             int64_t index,
                                                             const Geo::Point2D& other) {
  col.setItem(index, other);
}

// -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
// Column<Geo*> methods

#define EXPOSE_isNull(RowType)                                                         \
  extern "C" DEVICE RUNTIME_EXPORT bool Column##RowType##_isNull(Column<RowType>& col, \
                                                                 int64_t index) {      \
    return col.isNull(index);                                                          \
  }

#define EXPOSE_setNull(RowType)                                                         \
  extern "C" DEVICE RUNTIME_EXPORT void Column##RowType##_setNull(Column<RowType>& col, \
                                                                  int64_t index) {      \
    col.setNull(index);                                                                 \
  }

// copy elements by hand as operator= expects a valid Geo type as rhs
#define EXPOSE_getItem(RowType, TypeName)                          \
  extern "C" DEVICE RUNTIME_EXPORT void Column##RowType##_getItem( \
      Column<RowType>& col, int64_t index, TypeName& rhs) {        \
    TypeName out = col.getItem(index);                             \
    rhs.flatbuffer_ = out.flatbuffer_;                             \
    rhs.n_ = out.n_;                                               \
    for (int i = 0; i < NESTED_ARRAY_NDIM; i++)                    \
      rhs.index_[i] = out.index_[i];                               \
  }

#define EXPOSE_setItem(RowType, TypeName)                          \
  extern "C" DEVICE RUNTIME_EXPORT void Column##RowType##_setItem( \
      Column<RowType>& col, int64_t index, TypeName& rhs) {        \
    col.setItem(index, rhs);                                       \
  }

#define EXPOSE_getNofValues(RowType)                                       \
  extern "C" DEVICE RUNTIME_EXPORT int64_t Column##RowType##_getNofValues( \
      Column<RowType>& col) {                                              \
    return col.getNofValues();                                             \
  }

#define EXPOSE(RowType, TypeName)    \
  EXPOSE_isNull(RowType);            \
  EXPOSE_setNull(RowType);           \
  EXPOSE_getNofValues(RowType);      \
  EXPOSE_getItem(RowType, TypeName); \
  EXPOSE_setItem(RowType, TypeName);

EXPOSE(GeoLineString, Geo::LineString);
EXPOSE(GeoMultiPoint, Geo::MultiPoint);
EXPOSE(GeoPolygon, Geo::Polygon);
EXPOSE(GeoMultiLineString, Geo::MultiLineString);
EXPOSE(GeoMultiPolygon, Geo::MultiPolygon);
EXPOSE(TextEncodingNone, flatbuffer::TextEncodingNone);

// -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
// Geo* methods

#define EXPOSE_Geo_isNull(BaseType)                                                   \
  extern "C" DEVICE RUNTIME_EXPORT bool Geo##BaseType##_isNull(Geo::BaseType& base) { \
    return base.isNull();                                                             \
  }

#define EXPOSE_Geo_size(BaseType)                                                     \
  extern "C" DEVICE RUNTIME_EXPORT size_t Geo##BaseType##_size(Geo::BaseType& base) { \
    return base.size();                                                               \
  }

extern "C" DEVICE RUNTIME_EXPORT void GeoMultiPolygon_fromCoords(Geo::MultiPolygon& base,
                                                                 double* data,
                                                                 int64_t* dim_x,
                                                                 int64_t* dim_y,
                                                                 int64_t size_dim_x,
                                                                 int64_t size_dim_y) {
  std::vector<std::vector<std::vector<double>>> coords(size_dim_x);
  int l = 0;
  for (int i = 0; i < size_dim_x; i++) {
    coords[i].resize(size_dim_y);
    for (int j = 0; j < size_dim_y; j++) {
      coords[i][j].resize(dim_y[j]);
      for (int k = 0; k < dim_y[j]; k++) {
        coords[i][j][k] = data[l++];
      }
    }
  }
  base.fromCoords(coords);
}

#define EXPOSE_Geo_fromCoords_vec2(BaseType)                                \
  extern "C" DEVICE RUNTIME_EXPORT void Geo##BaseType##_fromCoords(         \
      Geo::BaseType& base, double* data, int64_t* indices, int64_t nrows) { \
    std::vector<std::vector<double>> coords;                                \
    int64_t offset = 0;                                                     \
    for (int64_t i = 0; i < nrows; i++) {                                   \
      int64_t size = indices[i];                                            \
      std::vector<double> row(data + offset, data + offset + size);         \
      coords.push_back(row);                                                \
      offset += size;                                                       \
    }                                                                       \
    base.fromCoords(coords);                                                \
  }

// Can this function receive anything other than Array<double>?
#define EXPOSE_Geo_fromCoords_vec(BaseType)                         \
  extern "C" DEVICE RUNTIME_EXPORT void Geo##BaseType##_fromCoords( \
      Geo::BaseType& base, double* lst, int64_t size) {             \
    std::vector<double> coords(lst, lst + size);                    \
    base.fromCoords(coords);                                        \
  }

#define EXPOSE_Geo_toCoords_vec(BaseType)                                     \
  extern "C" DEVICE RUNTIME_EXPORT double Geo##BaseType##_toCoords_get_value( \
      Geo::BaseType& base, int64_t index) {                                   \
    Geo::Point2D p = base[index / 2];                                         \
    return (index % 2 == 0) ? p.x : p.y;                                      \
  }

#define EXPOSE_Geo_toCoords_vec2(BaseType)                                    \
  extern "C" DEVICE RUNTIME_EXPORT double Geo##BaseType##_toCoords_get_value( \
      const Geo::BaseType& base, int64_t ring_index, int64_t coord_index) {   \
    Geo::Point2D p = base[ring_index][coord_index / 2];                       \
    return (coord_index % 2 == 0) ? p.x : p.y;                                \
  }

#define EXPOSE_Geo_getItem_Point2D(BaseType, ItemType)             \
  extern "C" DEVICE RUNTIME_EXPORT void Geo##BaseType##_getItem(   \
      Geo::BaseType& base, int64_t index, Geo::ItemType& result) { \
    Geo::ItemType out = base.getItem(index);                       \
    result.x = out.x;                                              \
    result.y = out.y;                                              \
  }

#define EXPOSE_Geo_getItem(BaseType, ItemType)                     \
  extern "C" DEVICE RUNTIME_EXPORT void Geo##BaseType##_getItem(   \
      Geo::BaseType& base, int64_t index, Geo::ItemType& result) { \
    Geo::ItemType out = base.getItem(index);                       \
    result.flatbuffer_ = out.flatbuffer_;                          \
    result.n_ = out.n_;                                            \
    for (int i = 0; i < NESTED_ARRAY_NDIM; i++)                    \
      result.index_[i] = out.index_[i];                            \
  }

#define EXPOSE_Geo_Point2D(BaseType, ItemType) \
  EXPOSE_Geo_isNull(BaseType);                 \
  EXPOSE_Geo_size(BaseType);                   \
  EXPOSE_Geo_getItem_Point2D(BaseType, ItemType);

#define EXPOSE_Geo(BaseType, ItemType) \
  EXPOSE_Geo_isNull(BaseType);         \
  EXPOSE_Geo_size(BaseType);           \
  EXPOSE_Geo_getItem(BaseType, ItemType);

EXPOSE_Geo_Point2D(LineString, Point2D);
EXPOSE_Geo_Point2D(MultiPoint, Point2D);
EXPOSE_Geo(MultiLineString, LineString);
EXPOSE_Geo(Polygon, LineString);
EXPOSE_Geo(MultiPolygon, Polygon);

// Expose fromCoords
EXPOSE_Geo_fromCoords_vec(LineString);
EXPOSE_Geo_fromCoords_vec(MultiPoint);
EXPOSE_Geo_fromCoords_vec2(Polygon);
EXPOSE_Geo_fromCoords_vec2(MultiLineString);

// Expose toCoords
EXPOSE_Geo_toCoords_vec(MultiPoint);
EXPOSE_Geo_toCoords_vec(LineString);
EXPOSE_Geo_toCoords_vec2(Polygon);
EXPOSE_Geo_toCoords_vec2(MultiLineString);
// Missing vec3 for MultiPolygon

// -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

extern "C" DEVICE RUNTIME_EXPORT void ColumnTextEncodingNone_setItem_fromBuffer(
    Column<TextEncodingNone>& col,
    int64_t index,
    int8_t* rhs) {
  col.setItem(index, *(reinterpret_cast<TextEncodingNone*>(rhs)));
}

extern "C" DEVICE RUNTIME_EXPORT void ColumnTextEncodingNone_concatItem_fromBuffer(
    Column<TextEncodingNone>& col,
    int64_t index,
    int8_t* rhs) {
  col[index] += *(reinterpret_cast<TextEncodingNone*>(rhs));
}

extern "C" DEVICE RUNTIME_EXPORT void ColumnTextEncodingNone_concatItem(
    Column<TextEncodingNone>& col,
    int64_t index,
    int8_t* rhs) {
  col[index] += *(reinterpret_cast<flatbuffer::TextEncodingNone*>(rhs));
}

// -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

volatile bool avoid_opt_address_geo(void* address) {
  return address != nullptr;
}

#define LIST_FUNCTION_W_SUFFIX(RowType, Suffix) \
  avoid_opt_address_geo(reinterpret_cast<void*>(Column##RowType##_##Suffix))

#define LIST_GEO_fromCoords(RowType) \
  avoid_opt_address_geo(reinterpret_cast<void*>(RowType##_fromCoords))

#define LIST_Geo_toCoords_vec(RowType) \
  avoid_opt_address_geo(reinterpret_cast<void*>(RowType##_toCoords_get_value))

#define LIST_Geo_toCoords_vec2(RowType) \
  avoid_opt_address_geo(reinterpret_cast<void*>(RowType##_toCoords_get_value))

#define LIST_FUNCTIONS(RowType)                   \
  LIST_FUNCTION_W_SUFFIX(RowType, getItem) &&     \
      LIST_FUNCTION_W_SUFFIX(RowType, setItem) && \
      LIST_FUNCTION_W_SUFFIX(RowType, isNull) &&  \
      LIST_FUNCTION_W_SUFFIX(RowType, setNull) && \
      LIST_FUNCTION_W_SUFFIX(RowType, getNofValues)

bool functions_exist_geo_column() {
  bool ret = true;
  ret &=
      (avoid_opt_address_geo(reinterpret_cast<void*>(ColumnGeoPoint_getItem)) &&
       avoid_opt_address_geo(reinterpret_cast<void*>(ColumnGeoPoint_isNull)) &&
       avoid_opt_address_geo(reinterpret_cast<void*>(ColumnGeoPoint_setItem)) &&
       avoid_opt_address_geo(reinterpret_cast<void*>(ColumnGeoPoint_setNull)) &&
       avoid_opt_address_geo(reinterpret_cast<void*>(GeoLineString_toCoords_get_value)) &&
       avoid_opt_address_geo(reinterpret_cast<void*>(ColumnGeoPoint_setNull)));
  ret &= LIST_FUNCTIONS(GeoLineString);
  ret &= LIST_FUNCTIONS(GeoPolygon);
  ret &= LIST_FUNCTIONS(GeoMultiLineString);
  ret &= LIST_FUNCTIONS(GeoMultiPolygon);

  // Geo fromCoords
  ret &= LIST_GEO_fromCoords(GeoLineString);
  ret &= LIST_GEO_fromCoords(GeoMultiLineString);
  ret &= LIST_GEO_fromCoords(GeoPolygon);
  ret &= LIST_GEO_fromCoords(GeoMultiPolygon);
  ret &= LIST_GEO_fromCoords(GeoMultiPoint);

  // Geo toCoords
  ret &= LIST_Geo_toCoords_vec(GeoMultiPoint);
  ret &= LIST_Geo_toCoords_vec(GeoLineString);
  ret &= LIST_Geo_toCoords_vec2(GeoPolygon);
  ret &= LIST_Geo_toCoords_vec2(GeoMultiLineString);

  return ret;
}
