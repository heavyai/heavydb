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

#include "Geospatial/Types.h"

#include <limits>
#include <mutex>

#include <gdal.h>
#include <ogr_geometry.h>
#include <ogrsf_frmts.h>

#include "Logger/Logger.h"
#include "Shared/sqltypes.h"

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
    throw Geospatial::GeoTypesError(
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
      throw Geospatial::GeoTypesError(
          "PolyRing",
          "All exterior rings must have more than 3 points. Found ring with " +
              std::to_string(num_points_added) + " points.");
    }
  }
  return num_points_added;
}

}  // namespace

namespace Geospatial {

std::mutex transformation_map_mutex_;
std::map<std::tuple<int32_t, int32_t>, std::shared_ptr<OGRCoordinateTransformation>>
    transformation_map_;

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

OGRErr GeoBase::createFromWkb(const std::vector<uint8_t>& wkb, OGRGeometry** geom) {
#if (GDAL_VERSION_MAJOR > 2) || (GDAL_VERSION_MAJOR == 2 && GDAL_VERSION_MINOR >= 3)
  OGRErr ogr_status =
      OGRGeometryFactory::createFromWkb(wkb.data(), nullptr, geom, wkb.size());
  return ogr_status;
#else
  CHECK(false);
#endif
}

// GeoBase could also generate GEOS geometries through geom_->exportToGEOS(),
// conversion to WKB and subsequent load of WKB into GEOS could be eliminated.

bool GeoBase::getWkb(std::vector<uint8_t>& wkb) const {
  auto size = geom_->WkbSize();
  if (size > 0) {
    wkb.resize(size);
    geom_->exportToWkb(wkbNDR, wkb.data());
    return true;
  }
  return false;
}

bool GeoBase::isEmpty() const {
  return geom_ && geom_->IsEmpty();
}

bool GeoBase::operator==(const GeoBase& other) const {
  if (!this->geom_ || !other.geom_) {
    return false;
  }
  return this->geom_->Equals(other.geom_);
}

/**  World Mercator, equivalent to EPSG:3395 */
#define SRID_WORLD_MERCATOR 999000
/**  Start of UTM North zone, equivalent to EPSG:32601 */
#define SRID_NORTH_UTM_START 999001
/**  End of UTM North zone, equivalent to EPSG:32660 */
#define SRID_NORTH_UTM_END 999060
/** Lambert Azimuthal Equal Area (LAEA) North Pole, equivalent to EPSG:3574 */
#define SRID_NORTH_LAMBERT 999061
/**  Start of UTM South zone, equivalent to EPSG:32701 */
#define SRID_SOUTH_UTM_START 999101
/**  Start of UTM South zone, equivalent to EPSG:32760 */
#define SRID_SOUTH_UTM_END 999160
/** Lambert Azimuthal Equal Area (LAEA) South Pole, equivalent to EPSG:3409 */
#define SRID_SOUTH_LAMBERT 999161
/** LAEA zones start (6 latitude bands x up to 20 longitude bands) */
#define SRID_LAEA_START 999163
/** LAEA zones end (6 latitude bands x up to 20 longitude bands) */
#define SRID_LAEA_END 999283

int32_t GeoBase::getBestPlanarSRID() const {
  if (!this->geom_) {
    return 0;
  }
  double cx, cy, xwidth, ywidth;
  OGREnvelope envelope;
  geom_->getEnvelope(&envelope);
  // Can't use GDAL's Centroid geom_->Centroid(OGRPoint*): requires GEOS
  // Use center of the bounding box for now.
  // TODO: hook up our own Centroid implementation
  cx = (envelope.MaxX + envelope.MinX) / 2.0;
  cy = (envelope.MaxY + envelope.MinY) / 2.0;
  xwidth = envelope.MaxX - envelope.MinX;
  ywidth = envelope.MaxY - envelope.MinY;

  // Arctic coords: Lambert Azimuthal Equal Area North
  if (cy > 70.0 && ywidth < 45.0) {
    return SRID_NORTH_LAMBERT;
  }
  // Antarctic coords: Lambert Azimuthal Equal Area South
  if (cy < -70.0 && ywidth < 45.0) {
    return SRID_SOUTH_LAMBERT;
  }

  // Can geometry fit into a single UTM zone?
  if (xwidth < 6.0) {
    int zone = floor((cx + 180.0) / 6.0);
    if (zone > 59) {
      zone = 59;
    }
    // Below the equator: UTM South
    // Above the equator: UTM North
    if (cy < 0.0) {
      return SRID_SOUTH_UTM_START + zone;
    } else {
      return SRID_NORTH_UTM_START + zone;
    }
  }

  // TODO: to be removed once we add custom LAEA zone transforms
  // Can geometry still fit into 5 consecutive UTM zones?
  // Then go for the mid-UTM zone, tolerating some limited distortion
  // in the left and right corners. That's still better than Mercator.
  if (xwidth < 30.0) {
    int zone = floor((cx + 180.0) / 6.0);
    if (zone > 59) {
      zone = 59;
    }
    // Below the equator: UTM South
    // Above the equator: UTM North
    if (cy < 0.0) {
      return SRID_SOUTH_UTM_START + zone;
    } else {
      return SRID_NORTH_UTM_START + zone;
    }
  }

  // Can geometry fit into a custom LAEA area 30 degrees high? Allow some overlap.
  if (ywidth < 25.0) {
    int xzone = -1;
    int yzone = 3 + floor(cy / 30.0);  // range of 0-5
    if ((yzone == 2 || yzone == 3) && xwidth < 30.0) {
      // Equatorial band, 12 zones, 30 degrees wide
      xzone = 6 + floor(cx / 30.0);
    } else if ((yzone == 1 || yzone == 4) && xwidth < 45.0) {
      // Temperate band, 8 zones, 45 degrees wide
      xzone = 4 + floor(cx / 45.0);
    } else if ((yzone == 0 || yzone == 5) && xwidth < 90.0) {
      // Arctic band, 4 zones, 90 degrees wide
      xzone = 2 + floor(cx / 90.0);
    }
    // Found an appropriate xzone to fit in?
    if (xzone != -1) {
      return SRID_LAEA_START + 20 * yzone + xzone;
    }
  }

  // Fall-back to Mercator
  return SRID_WORLD_MERCATOR;
}

std::shared_ptr<OGRCoordinateTransformation> GeoBase::getTransformation(int32_t srid0,
                                                                        int32_t srid1) {
  std::lock_guard<std::mutex> guard(transformation_map_mutex_);
  std::tuple<int32_t, int32_t> key{srid0, srid1};
  auto it = transformation_map_.find(key);
  if (it != transformation_map_.end()) {
    return it->second;
  }
  auto setSpatialReference = [&](OGRSpatialReference* sr, int32_t srid) -> bool {
    OGRErr status = OGRERR_NONE;
    if (srid == 4326) {
      status = sr->importFromEPSG(4326);
    } else if (srid == SRID_NORTH_LAMBERT) {
      // +proj=laea +lat_0=90 +lon_0=-40 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m
      // +no_defs
      status = sr->importFromEPSG(3574);
    } else if (srid == SRID_SOUTH_LAMBERT) {
      // +proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m
      // +no_defs
      status = sr->importFromEPSG(3409);
    } else if (SRID_SOUTH_UTM_START <= srid && srid <= SRID_SOUTH_UTM_END) {
      // +proj=utm +zone=%d +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs
      int32_t zone = srid - SRID_SOUTH_UTM_START;
      status = sr->importFromEPSG(32701 + zone);
    } else if (SRID_NORTH_UTM_START <= srid && srid <= SRID_NORTH_UTM_END) {
      // +proj=utm +zone=%d +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
      int32_t zone = srid - SRID_NORTH_UTM_START;
      status = sr->importFromEPSG(32601 + zone);
    } else if (SRID_LAEA_START <= srid && srid <= SRID_LAEA_END) {
      // TODO: add support and coordinate operations for custom Lambert zones,
      // need to calculate lat/lon for the zone, SetCoordinateOperation in options.
      // +proj=laea +ellps=WGS84 +datum=WGS84 +lat_0=%g +lon_0=%g +units=m +no_defs
      // Go with Mercator for now
      status = sr->importFromEPSG(3395);
    } else if (srid == SRID_WORLD_MERCATOR) {
      // +proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m
      // +no_defs
      status = sr->importFromEPSG(3395);
    } else if (srid > 0) {
      // Attempt to import from srid directly
      status = sr->importFromEPSG(srid);
    } else {
      return false;
    }
#if GDAL_VERSION_MAJOR >= 3
    // GDAL 3.x (really Proj.4 6.x) now enforces lat, lon order
    // this results in X and Y being transposed for angle-based
    // coordinate systems. This restores the previous behavior.
    if (status == OGRERR_NONE) {
      sr->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
    }
#endif
    return (status == OGRERR_NONE);
  };

  OGRSpatialReference sr0;
  if (!setSpatialReference(&sr0, srid0)) {
    return nullptr;
  }
  OGRSpatialReference sr1;
  if (!setSpatialReference(&sr1, srid1)) {
    return nullptr;
  }
  // GDAL 3 allows specification of advanced transformations in
  // OGRCoordinateTransformationOptions, including multi-step pipelines.
  // GDAL 3 would be required to handle Lambert zone proj4 strings.
  // Using a simple transform for now.
  std::shared_ptr<OGRCoordinateTransformation> new_transformation;
  new_transformation.reset(OGRCreateCoordinateTransformation(&sr0, &sr1));
  transformation_map_[key] = new_transformation;
  return new_transformation;
}

bool GeoBase::transform(int32_t srid0, int32_t srid1) {
  auto coordinate_transformation = getTransformation(srid0, srid1);
  if (!coordinate_transformation) {
    return false;
  }
  auto ogr_status = geom_->transform(coordinate_transformation.get());
  return (ogr_status == OGRERR_NONE);
}

bool GeoBase::transform(SQLTypeInfo& ti) {
  auto srid1 = ti.get_output_srid();
  if (srid1 == 4326) {
    auto srid0 = ti.get_input_srid();
    if (srid0 > 0 && srid0 != 4326) {
      if (!transform(srid0, srid1)) {
        return false;
      }
    }
  }
  return true;
}

// Run a specific geo operation on this and other geometries
std::unique_ptr<GeoBase> GeoBase::run(GeoBase::GeoOp op, const GeoBase& other) const {
  OGRGeometry* result = nullptr;
  // Invalid geometries are derailing geo op performance,
  // Checking validity before running an operation doesn't have lesser penalty.
  // DOES NOT HELP: if (geom_->IsValid() && other.geom_->IsValid()) {
  switch (op) {
    case GeoBase::GeoOp::kINTERSECTION:
      result = geom_->Intersection(other.geom_);
      break;
    case GeoBase::GeoOp::kDIFFERENCE:
      result = geom_->Difference(other.geom_);
      break;
    case GeoBase::GeoOp::kUNION:
      result = geom_->Union(other.geom_);
      break;
    default:
      break;
  }
  // TODO: Need to handle empty/non-POLYGON result
  if (!result || result->IsEmpty() ||
      !(result->getGeometryType() == wkbPolygon ||
        result->getGeometryType() == wkbMultiPolygon)) {
    throw GeoTypesError(std::string(OGRGeometryTypeToName(geom_->getGeometryType())),
                        "Currenly don't support invalid or empty result");
    // return GeoTypesFactory::createGeoType("POLYGON EMPTY");
    // Not supporting EMPTY polygons, return a dot polygon
    // return GeoTypesFactory::createGeoType(
    //     "MULTIPOLYGON(((0 0,0.0000001 0,0 0.0000001)))");
  }
  return GeoTypesFactory::createGeoType(result);
}

// Run a specific geo operation on this and other geometries
std::unique_ptr<GeoBase> GeoBase::optimized_run(GeoBase::GeoOp op,
                                                const GeoBase& other) const {
  OGRGeometry* result = nullptr;
  // Loop through polys combinations, check validity, do intersections
  // where needed, return a union of all intersections
  auto gc1 = geom_->toGeometryCollection();
  auto gc2 = other.geom_->toGeometryCollection();
  if (!gc1 || !gc2 || gc1->IsEmpty() || gc2->IsEmpty()) {
    return nullptr;
  }
  for (int i1 = 0; i1 < gc1->getNumGeometries(); i1++) {
    auto g1 = gc1->getGeometryRef(i1);
    // Validity check is very slow
    if (!g1 || g1->IsEmpty() /*|| !g1->IsValid()*/) {
      continue;
    }
    OGREnvelope ge1;
    g1->getEnvelope(&ge1);
    for (int i2 = 0; i2 < gc2->getNumGeometries(); i2++) {
      auto g2 = gc2->getGeometryRef(i2);
      // Validity check is very slow
      if (!g2 || g2->IsEmpty() /*|| !g2->IsValid()*/) {
        continue;
      }
      // Check for bounding box overlap
      OGREnvelope ge2;
      g2->getEnvelope(&ge2);
      if (!ge1.Intersects(ge2)) {
        continue;
      }
      // Do intersection
      auto intermediate_result = g1->Intersection(g2);
      if (!intermediate_result || intermediate_result->IsEmpty()) {
        continue;
      }
      if (!result) {
        result = intermediate_result;
      } else {
        result = result->Union(intermediate_result);
      }
    }
  }

  // TODO: Need to handle empty/non-POLYGON result
  if (!result || result->IsEmpty() ||
      !(result->getGeometryType() == wkbPolygon ||
        result->getGeometryType() == wkbMultiPolygon)) {
    throw GeoTypesError(std::string(OGRGeometryTypeToName(geom_->getGeometryType())),
                        "Currenly don't support invalid or empty result");
    // return GeoTypesFactory::createGeoType("POLYGON EMPTY");
    // Not supporting EMPTY polygons, return a dot polygon
    // return GeoTypesFactory::createGeoType(
    //     "MULTIPOLYGON(((0 0,0.0000001 0,0 0.0000001)))");
  }
  return GeoTypesFactory::createGeoType(result);
}

// Run a specific geo operation on this geometry and a double param
std::unique_ptr<GeoBase> GeoBase::run(GeoBase::GeoOp op, double param) const {
  OGRGeometry* result = nullptr;
  switch (op) {
    case GeoBase::GeoOp::kBUFFER:
      result = geom_->Buffer(param);
      break;
    default:
      break;
  }
  // TODO: Need to handle empty/non-POLYGON result
  if (!result || result->IsEmpty() ||
      !(result->getGeometryType() == wkbPolygon ||
        result->getGeometryType() == wkbMultiPolygon)) {
    throw GeoTypesError(std::string(OGRGeometryTypeToName(geom_->getGeometryType())),
                        "Currenly don't support invalid or empty result");
    // return GeoTypesFactory::createGeoType("POLYGON EMPTY");
    // Not supporting EMPTY polygons, return a dot polygon
    // return GeoTypesFactory::createGeoType(
    //     "MULTIPOLYGON(((0 0,0.0000001 0,0 0.0000001)))");
  }
  return GeoTypesFactory::createGeoType(result);
}

// Run a specific predicate operation on this geometry
bool GeoBase::run(GeoBase::GeoOp op) const {
  auto result = false;
  switch (op) {
    case GeoBase::GeoOp::kISVALID:
      result = geom_->IsValid();
      break;
    case GeoBase::GeoOp::kISEMPTY:
      result = isEmpty();
      break;
    default:
      break;
  }
  return result;
}

std::unique_ptr<GeoBase> GeoPoint::clone() const {
  CHECK(geom_);
  return std::unique_ptr<GeoBase>(new GeoPoint(geom_->clone(), true));
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

std::unique_ptr<GeoBase> GeoLineString::clone() const {
  CHECK(geom_);
  return std::unique_ptr<GeoBase>(new GeoLineString(geom_->clone(), true));
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

std::unique_ptr<GeoBase> GeoPolygon::clone() const {
  CHECK(geom_);
  return std::unique_ptr<GeoBase>(new GeoPolygon(geom_->clone(), true));
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

std::unique_ptr<GeoBase> GeoMultiPolygon::clone() const {
  CHECK(geom_);
  return std::unique_ptr<GeoBase>(new GeoMultiPolygon(geom_->clone(), true));
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

std::unique_ptr<GeoBase> GeoGeometry::clone() const {
  CHECK(geom_);
  return std::unique_ptr<GeoBase>(new GeoGeometry(geom_->clone(), true));
}

std::unique_ptr<GeoBase> GeoGeometryCollection::clone() const {
  CHECK(geom_);
  return std::unique_ptr<GeoBase>(new GeoGeometryCollection(geom_->clone(), true));
}

GeoGeometryCollection::GeoGeometryCollection(const std::string& wkt) {
  const auto err = GeoBase::createFromWktString(wkt, &geom_);
  if (err != OGRERR_NONE) {
    throw GeoTypesError("GeometryCollection", err);
  }
  CHECK(geom_);
  if (wkbFlatten(geom_->getGeometryType()) != OGRwkbGeometryType::wkbGeometryCollection) {
    throw GeoTypesError("GeometryCollection",
                        "Unexpected geometry type from WKT string: " +
                            std::string(OGRGeometryTypeToName(geom_->getGeometryType())));
  }
}

namespace {

struct HexDigitToDecimalTable {
  uint8_t table_[128];
  constexpr HexDigitToDecimalTable() : table_{} {
    table_['1'] = 1;
    table_['2'] = 2;
    table_['3'] = 3;
    table_['4'] = 4;
    table_['5'] = 5;
    table_['6'] = 6;
    table_['7'] = 7;
    table_['8'] = 8;
    table_['9'] = 9;
    table_['a'] = 10;
    table_['A'] = 10;
    table_['b'] = 11;
    table_['B'] = 11;
    table_['c'] = 12;
    table_['C'] = 12;
    table_['d'] = 13;
    table_['D'] = 13;
    table_['e'] = 14;
    table_['E'] = 14;
    table_['f'] = 15;
    table_['F'] = 15;
  }
  constexpr uint8_t operator[](const char& hex_digit) const {
    return (hex_digit < 0) ? 0 : table_[static_cast<int>(hex_digit)];
  }
};

constexpr HexDigitToDecimalTable hex_digit_to_decimal_table;

inline uint8_t hex_to_binary(const char& usb, const char& lsb) {
  return (hex_digit_to_decimal_table[usb] << 4) | hex_digit_to_decimal_table[lsb];
}

std::vector<uint8_t> hex_string_to_binary_vector(const std::string& wkb_hex) {
  auto num_bytes = wkb_hex.size() >> 1;
  std::vector<uint8_t> wkb(num_bytes);
  auto* chars = wkb_hex.data();
  auto* bytes = wkb.data();
  for (size_t i = 0; i < num_bytes; i++) {
    auto const& usb = *chars++;
    auto const& lsb = *chars++;
    *bytes++ = hex_to_binary(usb, lsb);
  }
  return wkb;
}

}  // namespace

OGRGeometry* GeoTypesFactory::createOGRGeometry(const std::string& wkt_or_wkb_hex) {
  OGRGeometry* geom = nullptr;
  OGRErr err = OGRERR_NONE;
  if (wkt_or_wkb_hex.empty()) {
    err = OGRERR_NOT_ENOUGH_DATA;
  } else if (wkt_or_wkb_hex[0] == '0') {  // all WKB hex strings start with a 0
    err = GeoBase::createFromWkb(hex_string_to_binary_vector(wkt_or_wkb_hex), &geom);
  } else {
    err = GeoBase::createFromWktString(wkt_or_wkb_hex, &geom);
  }
  if (err != OGRERR_NONE) {
    throw GeoTypesError("GeoFactory", err);
  }
  return geom;
}

std::unique_ptr<GeoBase> GeoTypesFactory::createGeoType(
    const std::string& wkt_or_wkb_hex) {
  return GeoTypesFactory::createGeoTypeImpl(createOGRGeometry(wkt_or_wkb_hex));
}

std::unique_ptr<GeoBase> GeoTypesFactory::createGeoType(const std::vector<uint8_t>& wkb) {
  OGRGeometry* geom = nullptr;
  const auto err = GeoBase::createFromWkb(wkb, &geom);
  if (err != OGRERR_NONE) {
    throw GeoTypesError("GeoFactory", err);
  }
  return GeoTypesFactory::createGeoTypeImpl(geom);
}

std::unique_ptr<GeoBase> GeoTypesFactory::createGeoType(OGRGeometry* geom) {
  return GeoTypesFactory::createGeoTypeImpl(geom, false);
}

bool GeoTypesFactory::getGeoColumns(const std::string& wkt_or_wkb_hex,
                                    SQLTypeInfo& ti,
                                    std::vector<double>& coords,
                                    std::vector<double>& bounds,
                                    std::vector<int>& ring_sizes,
                                    std::vector<int>& poly_rings,
                                    const bool promote_poly_to_mpoly) {
  try {
    if (wkt_or_wkb_hex.empty() || wkt_or_wkb_hex == "NULL") {
      getNullGeoColumns(
          ti, coords, bounds, ring_sizes, poly_rings, promote_poly_to_mpoly);
      return true;
    }

    const auto geospatial_base = GeoTypesFactory::createGeoType(wkt_or_wkb_hex);

    if (!geospatial_base || !geospatial_base->transform(ti)) {
      return false;
    }

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

bool GeoTypesFactory::getGeoColumns(const std::vector<uint8_t>& wkb,
                                    SQLTypeInfo& ti,
                                    std::vector<double>& coords,
                                    std::vector<double>& bounds,
                                    std::vector<int>& ring_sizes,
                                    std::vector<int>& poly_rings,
                                    const bool promote_poly_to_mpoly) {
  try {
    const auto geospatial_base = GeoTypesFactory::createGeoType(wkb);

    if (!geospatial_base || !geospatial_base->transform(ti)) {
      return false;
    }

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

    if (!geospatial_base || !geospatial_base->transform(ti)) {
      return false;
    }

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

bool GeoTypesFactory::getGeoColumns(const std::vector<std::string>* wkt_or_wkb_hex_column,
                                    SQLTypeInfo& ti,
                                    std::vector<std::vector<double>>& coords_column,
                                    std::vector<std::vector<double>>& bounds_column,
                                    std::vector<std::vector<int>>& ring_sizes_column,
                                    std::vector<std::vector<int>>& poly_rings_column,
                                    const bool promote_poly_to_mpoly) {
  try {
    for (const auto& wkt_or_wkb_hex : *wkt_or_wkb_hex_column) {
      std::vector<double> coords;
      std::vector<double> bounds;
      std::vector<int> ring_sizes;
      std::vector<int> poly_rings;

      SQLTypeInfo row_ti{ti};
      getGeoColumns(wkt_or_wkb_hex,
                    row_ti,
                    coords,
                    bounds,
                    ring_sizes,
                    poly_rings,
                    promote_poly_to_mpoly);

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
    case wkbGeometryCollection:
      return std::unique_ptr<GeoGeometryCollection>(
          new GeoGeometryCollection(geom, owns_geom_obj));
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
  UNREACHABLE();
}

void GeoTypesFactory::getNullGeoColumns(SQLTypeInfo& ti,
                                        std::vector<double>& coords,
                                        std::vector<double>& bounds,
                                        std::vector<int>& ring_sizes,
                                        std::vector<int>& poly_rings,
                                        const bool promote_poly_to_mpoly) {
  UNREACHABLE();
}

}  // namespace Geospatial
