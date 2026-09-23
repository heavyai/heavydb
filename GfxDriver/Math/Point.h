/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GFXDRIVER_MATH_POINT_H_
#define GFXDRIVER_MATH_POINT_H_

#include <array>
#include <cmath>
#include <string>
#include "../RenderError.h"

namespace gfx {

namespace Math {

enum class Pos { X = 0, Y, Z, W };

template <typename T, int N = 2>
class Point {
 public:
  Point() { _data.fill(T(0)); }
  explicit Point(T filler) { _data.fill(filler); }
  explicit Point(const std::array<T, N>& ini) : _data(ini) {}
  explicit Point(const Point<T, N>& ini) : _data(ini._data) {}

  Point(Point&& ini) noexcept : _data(std::move(ini._data)) {}
  ~Point() {}

  Point<T, N>& set(const Point<T, N>& setter) {
    _data = setter._data;
    return *this;
  }

  Point<T, N>& set(const std::array<T, N>& setter) {
    _data = setter;
    return *this;
  }

  const T& operator[](std::size_t i) const {
    RUNTIME_EX_ASSERT(i < N, "Index " + std::to_string(i) + " out-of-bounds.");

    return _data[i];
  }

  T& operator[](std::size_t i) {
    return const_cast<T&>(static_cast<const Point&>(*this)[i]);
  }

  Point<T, N>& operator=(const Point<T, N>& rhs) { return set(rhs); }
  Point<T, N>& operator=(const std::array<T, N>& rhs) { return set(rhs); }

  // Point<T, N> operator+(const Vector<T, N>& rhs) const;
  // Point<T, N>& operator+=(const Vector<T, N>& rhs);

  // Vector<T, N> operator-(const Point<T, N>& rhs) const;
  // Point<T, N> operator-(const Vector<T, N>& rhs) const;

  // Point<T, N>& operator-=(const Vector<T, N>& rhs);

  // Point<T, N> operator*(const T& scalar) const;
  // friend Point<T, N> operator*(const T& scalar, const Point<T, N>& pt);
  // Point<T, N>& operator*=(const T& scalar);

  // Point<T, N> operator/(const T& scalar) const;
  // Point<T, N>& operator/=(const T& scalar);

  bool operator==(const Point<T, N>& rhs) const { return _data == rhs._data; }
  bool operator!=(const Point<T, N>& rhs) const { return _data != rhs._data; }

  operator std::string() const {
    std::string s = "[";

    if (N > 0) {
      s += std::to_string(_data[0]);
      for (std::size_t i = 1; i < N; ++i) {
        s += ", " + std::to_string(_data[i]);
      }
    }

    s += "]";
    return s;
  }

  operator std::array<T, N>() const { return _data; }

  friend std::ostream& operator<<(std::ostream& ostr, const Point<T, N>& v) {
    return ostr << std::string(v);
  }

  // Point<T, N>& transform(const AffineMatrix<T, N+1>& transformMat);
  // friend Point<T, N> operator*(const AffineMatrix<T, N+1>& transformMat, const Point<T,
  // N>& pt);

  T distance(const Point<T, N>& pt) const { return sqrt(distanceSqr(pt)); }
  T distanceSqr(const Point<T, N>& pt) const {
    T sum(0);
    for (std::size_t i = 0; i < N; ++i) {
      sum += pow(pt[i] - _data[i], 2);
    }
    return sum;
  }

  Point<T, N> lerp(const Point<T, N>& pt, T t) const {
    Point<T, N> rtn;

    for (std::size_t i = 0; i < N; ++i) {
      rtn[i] = _data[i] + t * (pt[i] - _data[i]);
    }

    return rtn;
  }

 private:
  std::array<T, N> _data;
};

template <typename T, int N = 2>
Point<T, N> lerp(const Point<T, N>& start, const Point<T, N>& end, T t) {
  Point<T, N> rtn;
  for (std::size_t i = 0; i < N; ++i) {
    rtn[i] = start[i] + t * (end[i] - start[i]);
  }

  return rtn;
}

}  // namespace Math

}  // namespace gfx

//#include "Point.ipp"  // template implementation

#endif  // GFXDRIVER_MATH_POINT_H_
