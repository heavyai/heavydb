/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef GEOSPATIAL_TYPES_H_
#define GEOSPATIAL_TYPES_H_

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sqltypes.h"

class OGRGeometry;

namespace Geo_namespace {

class GeoTypesError : public std::runtime_error {
 public:
  explicit GeoTypesError(const std::string& type, const int ogr_err)
      : std::runtime_error("Geo" + type +
                           " Error: " + GeoTypesError::OGRErrorToStr(ogr_err)) {}
  explicit GeoTypesError(const std::string& type, const std::string& err)
      : std::runtime_error("Geo" + type + " Error: " + err) {}

 private:
  static std::string OGRErrorToStr(const int ogr_err);
};

class GeoTypesFactory;

class GeoBase {
 public:
  GeoBase() : owns_geom_obj_(true) {}
  virtual ~GeoBase();

  std::string getWktString() const;
  enum class GeoType { kPOINT, kLINESTRING, kPOLYGON, kMULTIPOLYGON };
  virtual GeoType getType() const = 0;

  virtual bool operator==(const GeoBase& other) const;

 protected:
  GeoBase(OGRGeometry* geom, const bool owns_geom_obj)
      : geom_(geom), owns_geom_obj_(owns_geom_obj) {}
  OGRGeometry* geom_ = nullptr;
  bool owns_geom_obj_;

  static int createFromWktString(const std::string& wkt, OGRGeometry** geom);

  friend class GeoTypesFactory;
};

class GeoPoint : public GeoBase {
 public:
  GeoPoint(const std::vector<double>& coords);
  GeoPoint(const std::string& wkt);

  void getColumns(std::vector<double>& coords) const;
  GeoType getType() const final { return GeoType::kPOINT; }

 protected:
  GeoPoint(OGRGeometry* geom, const bool owns_geom_obj) : GeoBase(geom, owns_geom_obj) {}

  friend class GeoTypesFactory;
};

class GeoLineString : public GeoBase {
 public:
  GeoLineString(const std::vector<double>& coords);
  GeoLineString(const std::string& wkt);

  void getColumns(std::vector<double>& coords, std::vector<double>& bounds) const;
  GeoType getType() const final { return GeoType::kLINESTRING; }

 protected:
  GeoLineString(OGRGeometry* geom, const bool owns_geom_obj)
      : GeoBase(geom, owns_geom_obj) {}

  friend class GeoTypesFactory;
};

class GeoPolygon : public GeoBase {
 public:
  GeoPolygon(const std::vector<double>& coords, const std::vector<int32_t>& ring_sizes);
  GeoPolygon(const std::string& wkt);

  void getColumns(std::vector<double>& coords,
                  std::vector<int32_t>& ring_sizes,
                  std::vector<double>& bounds) const;
  GeoType getType() const final { return GeoType::kPOLYGON; }

  int32_t getNumInteriorRings() const;

 protected:
  GeoPolygon(OGRGeometry* geom, const bool owns_geom_obj)
      : GeoBase(geom, owns_geom_obj) {}

  friend class GeoTypesFactory;
};

class GeoMultiPolygon : public GeoBase {
 public:
  GeoMultiPolygon(const std::vector<double>& coords,
                  const std::vector<int32_t>& ring_sizes,
                  const std::vector<int32_t>& poly_rings);
  GeoMultiPolygon(const std::string& wkt);

  void getColumns(std::vector<double>& coords,
                  std::vector<int32_t>& ring_sizes,
                  std::vector<int32_t>& poly_rings,
                  std::vector<double>& bounds) const;
  GeoType getType() const final { return GeoType::kMULTIPOLYGON; }

 protected:
  GeoMultiPolygon(OGRGeometry* geom, const bool owns_geom_obj)
      : GeoBase(geom, owns_geom_obj) {}

  friend class GeoTypesFactory;
};

class GeoTypesFactory {
 public:
  static std::unique_ptr<GeoBase> createGeoType(const std::string& wkt);
  static std::unique_ptr<GeoBase> createGeoType(OGRGeometry* geom);

  static bool getGeoColumns(const std::string& wkt,
                            SQLTypeInfo& ti,
                            std::vector<double>& coords,
                            std::vector<double>& bounds,
                            std::vector<int>& ring_sizes,
                            std::vector<int>& poly_rings,
                            const bool promote_poly_to_mpoly = false);

  static bool getGeoColumns(OGRGeometry* geom,
                            SQLTypeInfo& ti,
                            std::vector<double>& coords,
                            std::vector<double>& bounds,
                            std::vector<int>& ring_sizes,
                            std::vector<int>& poly_rings,
                            const bool promote_poly_to_mpoly = false);

  static bool getGeoColumns(const std::vector<std::string>* wkt_column,
                            SQLTypeInfo& ti,
                            std::vector<std::vector<double>>& coords_column,
                            std::vector<std::vector<double>>& bounds_column,
                            std::vector<std::vector<int>>& ring_sizes_column,
                            std::vector<std::vector<int>>& poly_rings_column,
                            const bool promote_poly_to_mpoly = false);

 private:
  static std::unique_ptr<Geo_namespace::GeoBase> createGeoTypeImpl(
      OGRGeometry* geom,
      const bool owns_geom_obj = true);
  static void getGeoColumnsImpl(const std::unique_ptr<GeoBase>& geospatial_base,
                                SQLTypeInfo& ti,
                                std::vector<double>& coords,
                                std::vector<double>& bounds,
                                std::vector<int>& ring_sizes,
                                std::vector<int>& poly_rings,
                                const bool promote_poly_to_mpoly = false);
};

}  // namespace Geo_namespace

#endif  // GEOSPATIAL_TYPES_H_
