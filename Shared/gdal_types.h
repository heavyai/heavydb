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

#ifndef GDAL_TYPES_H_
#define GDAL_TYPES_H_

#include <vector>
#include <string>

class OGRGeometry;

namespace GDAL_namespace {

class GDALBase {
 public:
  virtual ~GDALBase();

  std::string getWktString() const;

 protected:
  OGRGeometry* geom_;
};

class GDALPoint : public GDALBase {
 public:
  GDALPoint(const std::vector<double>& coords);
};

class GDALLineString : public GDALBase {
 public:
  GDALLineString(const std::vector<double>& coords);
};

class GDALPolygon : public GDALBase {
 public:
  GDALPolygon(const std::vector<double>& coords, const std::vector<int32_t>& ring_sizes);
};

class GDALMultiPolygon : public GDALBase {
 public:
  GDALMultiPolygon(const std::vector<double>& coords,
                   const std::vector<int32_t>& ring_sizes,
                   const std::vector<int32_t>& poly_rings);
};

}  // namespace GDAL_namespace

#endif  // GDAL_TYPES_H_
