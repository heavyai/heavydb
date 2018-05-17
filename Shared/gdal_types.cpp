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

#include "gdal_types.h"

#include <gdal.h>
#include <glog/logging.h>
#include <ogr_geometry.h>
#include <ogrsf_frmts.h>

namespace GDAL_namespace {

GDALBase::~GDALBase() {
  OGRGeometryFactory::destroyGeometry(geom_);
}

std::string GDALBase::getWktString() const {
  char* wkt = nullptr;
  geom_->exportToWkt(&wkt);
  CHECK(wkt);
  std::string wkt_str(wkt);
  CPLFree(wkt);
  return wkt_str;
}

GDALPoint::GDALPoint(const std::vector<double>& coords) {
  CHECK_EQ(coords.size(), size_t(2));
  geom_ = OGRGeometryFactory::createGeometry(OGRwkbGeometryType::wkbPoint);
  OGRPoint* point = dynamic_cast<OGRPoint*>(geom_);
  point->setX(coords[0]);
  point->setY(coords[1]);
}

GDALLineString::GDALLineString(const std::vector<double>& coords) {
  geom_ = OGRGeometryFactory::createGeometry(OGRwkbGeometryType::wkbLineString);
  OGRLineString* line = dynamic_cast<OGRLineString*>(geom_);
  for (size_t i = 0; i < coords.size(); i += 2) {
    line->addPoint(coords[i], coords[i + 1]);
  }
}

GDALPolygon::GDALPolygon(const std::vector<double>& coords, const std::vector<int32_t>& ring_sizes) {
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

GDALMultiPolygon::GDALMultiPolygon(const std::vector<double>& coords,
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

}  // namespace GDAL_namespace
