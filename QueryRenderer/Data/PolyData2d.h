/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Math/Point.h"

template <typename T, bool SEQUENTIAL>
struct PolyData2d {
  using datatype = T;

  using Point = gfx::Math::Point<T, 2>;
  std::vector<T> coords, x_coords, y_coords;
  std::vector<int> ring_sizes;
  std::vector<int> poly_rings;

  PolyData2d() : in_poly_{false}, in_ring_{false} {}
  ~PolyData2d() {}

  size_t numVerts() const { return SEQUENTIAL ? x_coords.size() : coords.size() / 2U; }
  size_t numRings() const { return ring_sizes.size(); }
  size_t numPolys() const { return poly_rings.size(); }

  void beginPoly() {
    // validate
    CHECK_EQ(in_poly_, false);
    in_poly_ = true;

    // start a new poly
    poly_rings.push_back(ring_sizes.size());
  }

  void beginRing() {
    // validate
    CHECK_EQ(in_poly_, true);
    CHECK_EQ(in_ring_, false);
    in_ring_ = true;

    // start a new ring
    ring_sizes.push_back(0);
  }

  void addPoint(T x, T y) {
    // validate
    CHECK_EQ(in_poly_, true);
    CHECK_EQ(in_ring_, true);

    // add to coords list
    if (SEQUENTIAL) {
      x_coords.push_back(x);
      y_coords.push_back(y);
    } else {
      coords.push_back(x);
      coords.push_back(y);
    }

    // add the point to the current ring
    ring_sizes.back()++;
  }

  void endRing() {
    // validate
    CHECK_EQ(in_poly_, true);
    CHECK_EQ(in_ring_, true);
    in_ring_ = false;

    // add the ring to the current poly
    poly_rings.back()++;
  }

  void endPoly() {
    // validate
    CHECK_EQ(in_poly_, true);
    in_poly_ = false;
  }

 private:
  bool in_poly_, in_ring_;
};

// explicit types

using PolyData2dFS = PolyData2d<float, true>;
using PolyData2dFI = PolyData2d<float, false>;
using PolyData2dDS = PolyData2d<double, true>;
using PolyData2dDI = PolyData2d<double, false>;
