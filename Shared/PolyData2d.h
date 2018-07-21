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

#ifndef SHARED_POLYDATA2D_H_
#define SHARED_POLYDATA2D_H_

#include "Shared/ShapeDrawData.h"

#include <Rendering/Math/AABox.h>
#include <Rendering/Math/Point.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>

template <typename T, bool SEQUENTIAL>
struct PolyData2d {
  typedef T datatype;

  typedef Rendering::Math::AABox<T, 2> Bounds;
  typedef Rendering::Math::Point<T, 2> Point;
  typedef boost::geometry::model::point<T, 2, boost::geometry::cs::cartesian> BoostPoint;
  typedef boost::geometry::model::box<BoostPoint> BoostBox;
  typedef std::pair<BoostBox, PolyData2d<T, SEQUENTIAL>*> BoostRectangle;

  Bounds bounds;
  std::vector<T> coords, x_coords, y_coords;
  std::vector<int> ring_sizes;
  std::vector<int> poly_rings;
  size_t renderGroup;
  std::string wkt;

  PolyData2d() : renderGroup(0U), _inPoly(false), _inRing(false) {}
  ~PolyData2d() {}

  size_t numVerts() const { return SEQUENTIAL ? x_coords.size() : coords.size() / 2U; }
  size_t numRings() const { return ring_sizes.size(); }
  size_t numPolys() const { return poly_rings.size(); }

  void beginPoly() {
    // validate
    CHECK_EQ(_inPoly, false);
    _inPoly = true;

    // start a new poly
    poly_rings.push_back(ring_sizes.size());
  }

  void beginRing() {
    // validate
    CHECK_EQ(_inPoly, true);
    CHECK_EQ(_inRing, false);
    _inRing = true;

    // start a new ring
    ring_sizes.push_back(0);
  }

  void addPoint(T x, T y) {
    // validate
    CHECK_EQ(_inPoly, true);
    CHECK_EQ(_inRing, true);

    // add to coords list
    if (SEQUENTIAL) {
      x_coords.push_back(x);
      y_coords.push_back(y);
    } else {
      coords.push_back(x);
      coords.push_back(y);
    }

    // expand bounds by this point
    bounds.encapsulate({x, y});

    // add the point to the current ring
    ring_sizes.back()++;
  }

  void endRing() {
    // validate
    CHECK_EQ(_inPoly, true);
    CHECK_EQ(_inRing, true);
    _inRing = false;

    // add the ring to the current poly
    poly_rings.back()++;
  }

  void endPoly() {
    // validate
    CHECK_EQ(_inPoly, true);
    _inPoly = false;
  }

  BoostBox getBoostBox() {
    BoostBox box;
    boost::geometry::assign_inverse(box);
    boost::geometry::expand(box,
                            BoostPoint(bounds.getMinPoint()[0], bounds.getMinPoint()[1]));
    boost::geometry::expand(box,
                            BoostPoint(bounds.getMaxPoint()[0], bounds.getMaxPoint()[1]));
    return box;
  }

  void setWKT(const char* wktIn) { wkt = wktIn; }

 private:
  bool _inPoly, _inRing;
};

// explicit types

typedef PolyData2d<float, true> PolyData2dFS;
typedef PolyData2d<float, false> PolyData2dFI;
typedef PolyData2d<double, true> PolyData2dDS;
typedef PolyData2d<double, false> PolyData2dDI;

#endif  // SHARED_POLYDATA2D_H_
