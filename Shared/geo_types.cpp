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

#include "geo_types.h"
#include "sqltypes.h"

#include <gdal.h>
#include <glog/logging.h>
#include <ogr_geometry.h>
#include <ogrsf_frmts.h>

#include <limits>

/**
 * Note: We use dynamic_cast to convert the OGRGeometry pointer from the base class into
 * appropriate OGR<<type>> objects in derived clases to ensure GDAL is creating the proper
 * geometry type for all possibly inputs. Since we check the output type after creating a
 * new OGRGeometry object via the OGRGeometryFactor, we could theoretically move some
 * dynamic_cast to static_cast. The performance impact and safety of going from RTTI to
 * compile time casting needs to be investigated.
 */

namespace {
constexpr auto DOUBLE_MAX = std::numeric_limits<double>::max();
constexpr auto DOUBLE_MIN = std::numeric_limits<double>::lowest();

struct Coords {
  double x;
  double y;
  Coords(double x, double y) : x(x), y(y) {}
};

struct BoundingBox {
  Coords min;
  Coords max;
  BoundingBox() : min(DOUBLE_MAX, DOUBLE_MAX), max(DOUBLE_MIN, DOUBLE_MIN) {}

  void update(double x, double y) {
    if (x < min.x) {
      min.x = x;
    }
    if (y < min.y) {
      min.y = y;
    }
    if (x > max.x) {
      max.x = x;
    }
    if (y > max.y) {
      max.y = y;
    }
  }
};

int process_poly_ring(OGRLinearRing* ring,
                      std::vector<double>& coords,
                      BoundingBox* bbox) {
  double last_x = DOUBLE_MAX, last_y = DOUBLE_MAX;
  size_t first_index = coords.size();
  int num_points_added = 0;
  int num_points_in_ring = ring->getNumPoints();
  if (num_points_in_ring < 3) {
    throw Geo_namespace::GeoTypesError(
        "PolyRing",
        "All poly rings must have more than 3 points. Found ring with " +
            std::to_string(num_points_in_ring) + " points.");
  }
  for (auto i = 0; i < num_points_in_ring; i++) {
    OGRPoint point;
    ring->getPoint(i, &point);
    last_x = point.getX();
    last_y = point.getY();
    coords.push_back(last_x);
    coords.push_back(last_y);
    if (bbox) {
      bbox->update(last_x, last_y);
    }
    num_points_added++;
  }
  // Store all rings as open rings (implicitly assumes all rings are closed)
  if ((coords[first_index] == last_x) && (coords[first_index + 1] == last_y)) {
    coords.pop_back();
    coords.pop_back();
    num_points_added--;
    if (num_points_added < 3) {
      throw Geo_namespace::GeoTypesError(
          "PolyRing",
          "All exterior rings must have more than 3 points. Found ring with " +
              std::to_string(num_points_added) + " points.");
    }
  }
  return num_points_added;
}

}  // namespace

namespace Geo_namespace {

std::string GeoTypesError::OGRErrorToStr(const int ogr_err) {
  switch (ogr_err) {
    case OGRERR_NOT_ENOUGH_DATA:
      return std::string("not enough input data");
    case OGRERR_NOT_ENOUGH_MEMORY:
      return std::string("not enough memory");
    case OGRERR_UNSUPPORTED_GEOMETRY_TYPE:
      return std::string("unsupported geometry type");
    case OGRERR_UNSUPPORTED_OPERATION:
      return std::string("unsupported operation");
    case OGRERR_CORRUPT_DATA:
      return std::string("corrupt input data");
    case OGRERR_FAILURE:
      return std::string("ogr failure");
    case OGRERR_UNSUPPORTED_SRS:
      return std::string("unsupported spatial reference system");
    case OGRERR_INVALID_HANDLE:
      return std::string("invalid file handle");
#if (GDAL_VERSION_MAJOR > 1)
    case OGRERR_NON_EXISTING_FEATURE:
      return std::string("feature does not exist in input geometry");
#endif
    default:
      return std::string("Unknown OGOR error encountered: ") + std::to_string(ogr_err);
  }
}

GeoBase::~GeoBase() {
  // Note: Removing the geometry object that was pulled from an OGRFeature results in a
  // segfault. If we are wrapping around a pre-existing OGRGeometry object, we let the
  // caller manage the memory.
  if (geom_ && owns_geom_obj_) {
    OGRGeometryFactory::destroyGeometry(geom_);
  }
}

OGRErr GeoBase::createFromWktString(const std::string& wkt, OGRGeometry** geom) {
#if (GDAL_VERSION_MAJOR > 2) || (GDAL_VERSION_MAJOR == 2 && GDAL_VERSION_MINOR >= 3)
  OGRErr ogr_status = OGRGeometryFactory::createFromWkt(wkt.c_str(), nullptr, geom);
#else
  auto data = (char*)wkt.c_str();
  OGRErr ogr_status = OGRGeometryFactory::createFromWkt(&data, NULL, geom);
#endif
  return ogr_status;
}

std::string GeoBase::getWktString() const {
  char* wkt = nullptr;
  geom_->exportToWkt(&wkt);
  CHECK(wkt);
  std::string wkt_str(wkt);
  CPLFree(wkt);
  return wkt_str;
}

bool GeoBase::operator==(const GeoBase& other) const {
  if (!this->geom_ || !other.geom_) {
    return false;
  }
  return this->geom_->Equals(other.geom_);
  // return const_cast<const OGRGeometry*>(this->geom_) == const_cast<const
  // OGRGeometry*>(other.geom_);
}

GeoPoint::GeoPoint(const std::vector<double>& coords) {
  if (coords.size() != 2) {
    throw GeoTypesError("Point",
                        "Incorrect coord size of " + std::to_string(coords.size()) +
                            " supplied. Expected 2.");
  }
  geom_ = OGRGeometryFactory::createGeometry(OGRwkbGeometryType::wkbPoint);
  OGRPoint* point = dynamic_cast<OGRPoint*>(geom_);
  CHECK(point);
  point->setX(coords[0]);
  point->setY(coords[1]);
}

GeoPoint::GeoPoint(const std::string& wkt) {
  const auto err = GeoBase::createFromWktString(wkt, &geom_);
  if (err != OGRERR_NONE) {
    throw GeoTypesError("Point", err);
  }
  CHECK(geom_);
  if (wkbFlatten(geom_->getGeometryType()) != OGRwkbGeometryType::wkbPoint) {
    throw GeoTypesError("Point",
                        "Unexpected geometry type from WKT string: " +
                            std::string(OGRGeometryTypeToName(geom_->getGeometryType())));
  }
}

void GeoPoint::getColumns(std::vector<double>& coords) const {
  const auto point_geom = dynamic_cast<OGRPoint*>(geom_);
  CHECK(point_geom);

  if (point_geom->IsEmpty()) {
    // until the run-time can handle empties
    throw GeoTypesError("Point", "'EMPTY' not supported");
    // we cannot yet handle NULL fixed-length array
    // so we have to store sentinel values instead
    coords.push_back(NULL_DOUBLE);
    coords.push_back(NULL_DOUBLE);
    return;
  }

  coords.push_back(point_geom->getX());
  coords.push_back(point_geom->getY());
}

GeoLineString::GeoLineString(const std::vector<double>& coords) {
  geom_ = OGRGeometryFactory::createGeometry(OGRwkbGeometryType::wkbLineString);
  OGRLineString* line = dynamic_cast<OGRLineString*>(geom_);
  CHECK(line);
  for (size_t i = 0; i < coords.size(); i += 2) {
    line->addPoint(coords[i], coords[i + 1]);
  }
}

GeoLineString::GeoLineString(const std::string& wkt) {
  const auto err = GeoBase::createFromWktString(wkt, &geom_);
  if (err != OGRERR_NONE) {
    throw GeoTypesError("LineString", err);
  }
  CHECK(geom_);
  if (wkbFlatten(geom_->getGeometryType()) != OGRwkbGeometryType::wkbLineString) {
    throw GeoTypesError("LineString",
                        "Unexpected geometry type from WKT string: " +
                            std::string(OGRGeometryTypeToName(geom_->getGeometryType())));
  }
}

void GeoLineString::getColumns(std::vector<double>& coords,
                               std::vector<double>& bounds) const {
  auto linestring_geom = dynamic_cast<OGRLineString*>(geom_);
  CHECK(linestring_geom);

  if (linestring_geom->IsEmpty()) {
    // until the run-time can handle empties
    throw GeoTypesError("LineString", "'EMPTY' not supported");
    // return null bounds
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    return;
  }

  BoundingBox bbox;
  for (auto i = 0; i < linestring_geom->getNumPoints(); i++) {
    OGRPoint point;
    linestring_geom->getPoint(i, &point);
    double x = point.getX();
    double y = point.getY();
    coords.push_back(x);
    coords.push_back(y);
    bbox.update(x, y);
  }
  bounds.push_back(bbox.min.x);
  bounds.push_back(bbox.min.y);
  bounds.push_back(bbox.max.x);
  bounds.push_back(bbox.max.y);
}

GeoPolygon::GeoPolygon(const std::vector<double>& coords,
                       const std::vector<int32_t>& ring_sizes) {
  geom_ = OGRGeometryFactory::createGeometry(OGRwkbGeometryType::wkbPolygon);
  OGRPolygon* poly = dynamic_cast<OGRPolygon*>(geom_);
  CHECK(poly);

  size_t coords_ctr = 0;
  for (size_t r = 0; r < ring_sizes.size(); r++) {
    OGRLinearRing ring;
    const auto ring_sz = ring_sizes[r];
    for (auto i = 0; i < 2 * ring_sz; i += 2) {
      ring.addPoint(coords[coords_ctr + i], coords[coords_ctr + i + 1]);
    }
    ring.addPoint(coords[coords_ctr], coords[coords_ctr + 1]);
    coords_ctr += 2 * ring_sz;
    poly->addRing(&ring);
  }
}

GeoPolygon::GeoPolygon(const std::string& wkt) {
  const auto err = GeoBase::createFromWktString(wkt, &geom_);
  if (err != OGRERR_NONE) {
    throw GeoTypesError("Polygon", err);
  }
  CHECK(geom_);
  if (wkbFlatten(geom_->getGeometryType()) != OGRwkbGeometryType::wkbPolygon) {
    throw GeoTypesError("Polygon",
                        "Unexpected geometry type from WKT string: " +
                            std::string(OGRGeometryTypeToName(geom_->getGeometryType())));
  }
}

void GeoPolygon::getColumns(std::vector<double>& coords,
                            std::vector<int32_t>& ring_sizes,
                            std::vector<double>& bounds) const {
  const auto poly_geom = dynamic_cast<OGRPolygon*>(geom_);
  CHECK(poly_geom);

  if (poly_geom->IsEmpty()) {
    // until the run-time can handle empties
    throw GeoTypesError("Polygon", "'EMPTY' not supported");
    // return null bounds
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    return;
  }

  BoundingBox bbox;
  const auto exterior_ring = poly_geom->getExteriorRing();
  CHECK(exterior_ring);
  // All exterior rings are imported CCW
  if (exterior_ring->isClockwise()) {
    exterior_ring->reverseWindingOrder();
  }
  const auto num_points_added = process_poly_ring(exterior_ring, coords, &bbox);
  ring_sizes.push_back(num_points_added);
  for (auto r = 0; r < poly_geom->getNumInteriorRings(); r++) {
    auto interior_ring = poly_geom->getInteriorRing(r);
    CHECK(interior_ring);
    // All interior rings are imported CW
    if (!interior_ring->isClockwise()) {
      interior_ring->reverseWindingOrder();
    }
    const auto num_points_added = process_poly_ring(interior_ring, coords, nullptr);
    ring_sizes.push_back(num_points_added);
  }
  bounds.push_back(bbox.min.x);
  bounds.push_back(bbox.min.y);
  bounds.push_back(bbox.max.x);
  bounds.push_back(bbox.max.y);
}

int32_t GeoPolygon::getNumInteriorRings() const {
  const auto poly_geom = dynamic_cast<OGRPolygon*>(geom_);
  CHECK(poly_geom);
  return poly_geom->getNumInteriorRings();
}

GeoMultiPolygon::GeoMultiPolygon(const std::vector<double>& coords,
                                 const std::vector<int32_t>& ring_sizes,
                                 const std::vector<int32_t>& poly_rings) {
  geom_ = OGRGeometryFactory::createGeometry(OGRwkbGeometryType::wkbMultiPolygon);
  OGRMultiPolygon* multipoly = dynamic_cast<OGRMultiPolygon*>(geom_);
  CHECK(multipoly);

  size_t ring_ctr = 0;
  size_t coords_ctr = 0;
  for (const auto& rings_in_poly : poly_rings) {
    OGRPolygon poly;
    for (auto r = 0; r < rings_in_poly; r++) {
      OGRLinearRing ring;
      const auto ring_sz = ring_sizes[ring_ctr];
      for (auto i = 0; i < 2 * ring_sz; i += 2) {
        ring.addPoint(coords[coords_ctr + i], coords[coords_ctr + i + 1]);
      }
      ring.addPoint(coords[coords_ctr], coords[coords_ctr + 1]);
      coords_ctr += 2 * ring_sz;
      poly.addRing(&ring);
      ring_ctr++;
    }
    multipoly->addGeometry(&poly);
  }
}

GeoMultiPolygon::GeoMultiPolygon(const std::string& wkt) {
  const auto err = GeoBase::createFromWktString(wkt, &geom_);
  if (err != OGRERR_NONE) {
    throw GeoTypesError("MultiPolygon", err);
  }
  CHECK(geom_);
  if (wkbFlatten(geom_->getGeometryType()) != OGRwkbGeometryType::wkbMultiPolygon) {
    throw GeoTypesError("MultiPolygon",
                        "Unexpected geometry type from WKT string: " +
                            std::string(OGRGeometryTypeToName(geom_->getGeometryType())));
  }
}

void GeoMultiPolygon::getColumns(std::vector<double>& coords,
                                 std::vector<int32_t>& ring_sizes,
                                 std::vector<int32_t>& poly_rings,
                                 std::vector<double>& bounds) const {
  const auto mpoly = dynamic_cast<OGRMultiPolygon*>(geom_);
  CHECK(mpoly);

  if (mpoly->IsEmpty()) {
    // until the run-time can handle empties
    throw GeoTypesError("MultiPolygon", "'EMPTY' not supported");
    // return null bounds
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    bounds.push_back(NULL_DOUBLE);
    return;
  }

  BoundingBox bbox;
  for (auto p = 0; p < mpoly->getNumGeometries(); p++) {
    const auto mpoly_geom = mpoly->getGeometryRef(p);
    CHECK(mpoly_geom);
    const auto poly_geom = dynamic_cast<OGRPolygon*>(mpoly_geom);
    if (!poly_geom) {
      throw GeoTypesError("MultiPolygon",
                          "Failed to read polygon geometry from multipolygon");
    }
    const auto exterior_ring = poly_geom->getExteriorRing();
    CHECK(exterior_ring);
    // All exterior rings are imported CCW
    if (exterior_ring->isClockwise()) {
      exterior_ring->reverseWindingOrder();
    }
    const auto num_points_added = process_poly_ring(exterior_ring, coords, &bbox);
    ring_sizes.push_back(num_points_added);

    for (auto r = 0; r < poly_geom->getNumInteriorRings(); r++) {
      auto interior_ring = poly_geom->getInteriorRing(r);
      CHECK(interior_ring);
      // All interior rings are imported CW
      if (!interior_ring->isClockwise()) {
        interior_ring->reverseWindingOrder();
      }
      const auto num_points_added = process_poly_ring(interior_ring, coords, nullptr);
      ring_sizes.push_back(num_points_added);
    }
    poly_rings.push_back(poly_geom->getNumInteriorRings() + 1);
  }
  bounds.push_back(bbox.min.x);
  bounds.push_back(bbox.min.y);
  bounds.push_back(bbox.max.x);
  bounds.push_back(bbox.max.y);
}

std::unique_ptr<GeoBase> GeoTypesFactory::createGeoType(const std::string& wkt) {
  OGRGeometry* geom = nullptr;
  const auto err = GeoBase::createFromWktString(wkt, &geom);
  if (err != OGRERR_NONE) {
    throw GeoTypesError("GeoFactory", err);
  }
  return GeoTypesFactory::createGeoTypeImpl(geom);
}

std::unique_ptr<GeoBase> GeoTypesFactory::createGeoType(OGRGeometry* geom) {
  return GeoTypesFactory::createGeoTypeImpl(geom, false);
}

bool GeoTypesFactory::getGeoColumns(const std::string& wkt,
                                    SQLTypeInfo& ti,
                                    std::vector<double>& coords,
                                    std::vector<double>& bounds,
                                    std::vector<int>& ring_sizes,
                                    std::vector<int>& poly_rings,
                                    const bool promote_poly_to_mpoly) {
  try {
    const auto geospatial_base = GeoTypesFactory::createGeoType(wkt);

    int srid = 0;
    ti.set_input_srid(srid);
    ti.set_output_srid(srid);

    getGeoColumnsImpl(geospatial_base,
                      ti,
                      coords,
                      bounds,
                      ring_sizes,
                      poly_rings,
                      promote_poly_to_mpoly);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Geospatial Import Error: " << e.what();
    return false;
  }

  return true;
}

bool GeoTypesFactory::getGeoColumns(OGRGeometry* geom,
                                    SQLTypeInfo& ti,
                                    std::vector<double>& coords,
                                    std::vector<double>& bounds,
                                    std::vector<int>& ring_sizes,
                                    std::vector<int>& poly_rings,
                                    const bool promote_poly_to_mpoly) {
  try {
    const auto geospatial_base = GeoTypesFactory::createGeoType(geom);

    int srid = 0;
    ti.set_input_srid(srid);
    ti.set_output_srid(srid);

    getGeoColumnsImpl(geospatial_base,
                      ti,
                      coords,
                      bounds,
                      ring_sizes,
                      poly_rings,
                      promote_poly_to_mpoly);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Geospatial Import Error: " << e.what();
    return false;
  }

  return true;
}

bool GeoTypesFactory::getGeoColumns(const std::vector<std::string>* wkt_column,
                                    SQLTypeInfo& ti,
                                    std::vector<std::vector<double>>& coords_column,
                                    std::vector<std::vector<double>>& bounds_column,
                                    std::vector<std::vector<int>>& ring_sizes_column,
                                    std::vector<std::vector<int>>& poly_rings_column,
                                    const bool promote_poly_to_mpoly) {
  try {
    for (const auto wkt : *wkt_column) {
      std::vector<double> coords;
      std::vector<double> bounds;
      std::vector<int> ring_sizes;
      std::vector<int> poly_rings;

      SQLTypeInfo row_ti;
      getGeoColumns(
          wkt, row_ti, coords, bounds, ring_sizes, poly_rings, promote_poly_to_mpoly);

      if (ti.get_type() != row_ti.get_type()) {
        throw GeoTypesError("GeoFactory", "Columnar: Geometry type mismatch");
      }
      coords_column.push_back(coords);
      bounds_column.push_back(bounds);
      ring_sizes_column.push_back(ring_sizes);
      poly_rings_column.push_back(poly_rings);
    }

  } catch (const std::exception& e) {
    LOG(ERROR) << "Geospatial column Import Error: " << e.what();
    return false;
  }

  return true;
}

std::unique_ptr<GeoBase> GeoTypesFactory::createGeoTypeImpl(OGRGeometry* geom,
                                                            const bool owns_geom_obj) {
  switch (wkbFlatten(geom->getGeometryType())) {
    case wkbPoint:
      return std::unique_ptr<GeoPoint>(new GeoPoint(geom, owns_geom_obj));
    case wkbLineString:
      return std::unique_ptr<GeoLineString>(new GeoLineString(geom, owns_geom_obj));
    case wkbPolygon:
      return std::unique_ptr<GeoPolygon>(new GeoPolygon(geom, owns_geom_obj));
    case wkbMultiPolygon:
      return std::unique_ptr<GeoMultiPolygon>(new GeoMultiPolygon(geom, owns_geom_obj));
    default:
      throw GeoTypesError(
          "GeoTypesFactory",
          "Unrecognized geometry type: " + std::string(geom->getGeometryName()));
  }
}

void GeoTypesFactory::getGeoColumnsImpl(const std::unique_ptr<GeoBase>& geospatial_base,
                                        SQLTypeInfo& ti,
                                        std::vector<double>& coords,
                                        std::vector<double>& bounds,
                                        std::vector<int>& ring_sizes,
                                        std::vector<int>& poly_rings,
                                        const bool promote_poly_to_mpoly) {
  switch (geospatial_base->getType()) {
    case GeoBase::GeoType::kPOINT: {
      const auto geospatial_point = dynamic_cast<GeoPoint*>(geospatial_base.get());
      CHECK(geospatial_point);
      geospatial_point->getColumns(coords);
      ti.set_type(kPOINT);
    } break;
    case GeoBase::GeoType::kLINESTRING: {
      const auto geospatial_linestring =
          dynamic_cast<GeoLineString*>(geospatial_base.get());
      CHECK(geospatial_linestring);
      geospatial_linestring->getColumns(coords, bounds);
      ti.set_type(kLINESTRING);
    } break;
    case GeoBase::GeoType::kPOLYGON: {
      const auto geospatial_poly = dynamic_cast<GeoPolygon*>(geospatial_base.get());
      CHECK(geospatial_poly);
      geospatial_poly->getColumns(coords, ring_sizes, bounds);
      if (promote_poly_to_mpoly) {
        if (ring_sizes.size()) {
          CHECK_GT(coords.size(), 0);
          poly_rings.push_back(1 + geospatial_poly->getNumInteriorRings());
        }
      }
      ti.set_type(kPOLYGON);
    } break;
    case GeoBase::GeoType::kMULTIPOLYGON: {
      const auto geospatial_mpoly = dynamic_cast<GeoMultiPolygon*>(geospatial_base.get());
      CHECK(geospatial_mpoly);
      geospatial_mpoly->getColumns(coords, ring_sizes, poly_rings, bounds);
      ti.set_type(kMULTIPOLYGON);
    } break;
    default:
      throw std::runtime_error("Unrecognized geospatial type");
  }
}

}  // namespace Geo_namespace
