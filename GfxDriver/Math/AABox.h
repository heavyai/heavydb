/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GFXDRIVER_MATH_AABOX_H_
#define GFXDRIVER_MATH_AABOX_H_

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include "Point.h"

namespace gfx {

namespace Math {

template <typename T, int DIMS = 2>
class AABox {
 public:
  AABox() { initEmpty(); }
  explicit AABox(const AABox<T, DIMS>& ini) : _data(ini._data) {}  // copy constructor
  explicit AABox(const std::array<T, DIMS * 2>& ini) : _data(ini) {}

  AABox(AABox&& ini) noexcept : _data(std::move(ini._data)) {}  // move constructor
  ~AABox() {}

  /**
   * Initializes the bounding box as empty
   */
  AABox<T, DIMS>& initEmpty() {
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      _data[minIndex] = pos_infinity;
      _data[maxIndex] = neg_infinity;
    }

    return *this;
  }

  /**
   * Initialized the bounding box as infinite. All possible points will be bounded
   * by the box.
   */
  AABox<T, DIMS>& initInfinity() {
    std::size_t minIndex, maxIndex;
    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      _data[minIndex] = neg_infinity;
      _data[maxIndex] = pos_infinity;
    }

    return *this;
  }

  /**
   * Initializes the bounding box starting at the origin and extending outward
   * by a size defined in each dimension.
   */
  AABox<T, DIMS>& initSizeFromOrigin(const std::array<T, DIMS>& sizes) {
    std::size_t i, j;
    for (i = 0, j = DIMS; i < DIMS; ++i, ++j) {
      if (sizes[i] < 0) {
        _data[i] = -sizes[i];
        _data[j] = 0;
      } else {
        _data[i] = 0;
        _data[j] = sizes[i];
      }
    }
    return *this;
  }

  /**
   * Initializes the bounding box starting at a specific pt and extending outward
   * by a size defined in each dimension.
   */
  AABox<T, DIMS>& initSizeFromLocation(const Point<T, DIMS>& pt,
                                       const std::array<T, DIMS>& sizes) {
    std::size_t i, j;
    for (i = 0, j = DIMS; i < DIMS; ++i, ++j) {
      if (sizes[i] < 0) {
        _data[i] = pt[i] - sizes[i];
        _data[j] = pt[i];
      } else {
        _data[i] = pt[i];
        _data[j] = pt[i] + sizes[i];
      }
    }
    return *this;
  }

  AABox<T, DIMS>& initSizeFromLocation(const std::array<T, DIMS>& pt,
                                       const std::array<T, DIMS>& sizes) {
    std::size_t i, j;
    for (i = 0, j = DIMS; i < DIMS; ++i, ++j) {
      if (sizes[i] < 0) {
        _data[i] = pt[i] - sizes[i];
        _data[j] = pt[i];
      } else {
        _data[i] = pt[i];
        _data[j] = pt[i] + sizes[i];
      }
    }
    return *this;
  }

  /**
   * Initializes the bounding box by its center pt and extending outward by an extent size
   * in each dimension. For example, a 2D aabox with the center set to (0, 0), and extents
   * set to (5,5) will result in a bounds centered at the origin with a size of 10 in both
   * the X and Y dimensions.
   */
  AABox<T, DIMS>& initCenterExtents(const Point<T, DIMS>& center,
                                    const std::array<T, DIMS>& extents) {
    std::size_t i, j;
    for (i = 0, j = DIMS; i < DIMS; ++i, ++j) {
      if (extents[i] < 0) {
        _data[i] = center[i] + extents[i];
        _data[j] = center[i] - extents[i];
      } else {
        _data[i] = center[i] - extents[i];
        _data[i] = center[i] + extents[i];
      }
    }
    return *this;
  }

  /**
   * Sets the bounds to another one.
   */
  AABox<T, DIMS>& set(const AABox<T, DIMS>& setter) {
    _data = setter._data;
    return *this;
  }

  /**
   * Returns true if the bounds is considered empty.
   */
  bool isEmpty() const {
    std::size_t i, j;
    for (i = 0, j = DIMS; i < DIMS; ++i, ++j) {
      if (_data[i] > _data[j]) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns true if the bounds is infinte in any dimension.
   */
  bool isInfinite() const {
    std::size_t i, j;
    for (i = 0, j = DIMS; i < DIMS; ++i, ++j) {
      if (_data[i] == neg_infinity || _data[j] == pos_infinity) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns true if the bounds is entirely finite.
   */
  bool isFinite() const {
    std::size_t i, j;
    for (i = 0, j = DIMS; i < DIMS; ++i, ++j) {
      if (!isfinite(_data[i]) || !isfinite(_data[j])) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns an array with the size of the bounds in each
   * dimension
   */
  std::array<T, DIMS> getSize() const {
    std::array<T, DIMS> rtn;
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      rtn[minIndex] = _data[maxIndex] - _data[minIndex];
    }

    return rtn;
  }

  /**
   * Returns the extents of the bounds. The extents are is the size in each dimension
   * from the bounds center. In other words, the size of the bounds in each dimension / 2.
   */
  std::array<T, DIMS> getExtents() const { return getSize() / 2.0; }

  /**
   * Gets the center point of the bounds.
   */
  Point<T, DIMS> getCenter() const {
    Point<T, DIMS> rtn;
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      rtn[minIndex] = _data[minIndex] + (_data[maxIndex] - _data[minIndex]) / 2;
    }

    return rtn;
  }

  /**
   * Gets the minimum point of the bounds.
   */
  Point<T, DIMS> getMinPoint() const {
    Point<T, DIMS> rtn;

    for (std::size_t i = 0; i < DIMS; ++i) {
      rtn[i] = _data[i];
    }

    return rtn;
  }

  Point<T, DIMS> getMaxPoint() const {
    Point<T, DIMS> rtn;

    for (std::size_t i = 0; i < DIMS; ++i) {
      rtn[i] = _data[i + DIMS];
    }

    return rtn;
  }

  /**
   * Gets the area of the bounds. Only applicable for 2D bounds.
   */
  T area() const {
    RUNTIME_EX_ASSERT(DIMS == 2, "Can only compute the area of a 2D box.");

    return volume();
  }

  /**
   * Gets the volume of the entire bounds. This is applicable in all dimensions.
   */
  T volume() const {
    T volume = (DIMS > 1 ? 1 : 0);
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      volume *= (_data[maxIndex] - _data[minIndex]);
    }

    return volume;
  }

  /**
   * Expands this bounding box by the bounds of another.
   */
  AABox<T, DIMS>& hull(const AABox<T, DIMS>& aabox) {
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      _data[minIndex] = std::min(_data[minIndex], aabox[minIndex]);
      _data[maxIndex] = std::max(_data[maxIndex], aabox[maxIndex]);
    }

    return *this;
  }

  /**
   * Returns a new bounding box that defines the intersection of
   * this bounds and another. If the two bounds do not intersect,
   * an empty bounds is returned.
   */
  AABox<T, DIMS> intersection(const AABox<T, DIMS>& aabox) const {
    AABox<T, DIMS> isectBounds;
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      if (_data[maxIndex] < aabox[minIndex] || _data[minIndex] > aabox[maxIndex]) {
        // the two bounds do not intersect.
        break;
      }

      isectBounds[minIndex] = std::max(_data[minIndex], aabox[minIndex]);
      isectBounds[maxIndex] = std::min(_data[maxIndex], aabox[maxIndex]);
    }

    if (minIndex != DIMS) {
      // didn't examine in each dimension, which means we broke the above
      // for loop, which means there wasn't an intersection
      isectBounds.initEmpty();
    }

    return isectBounds;
  }

  /**
   * Returns true if this bounding box overlaps with another one.
   */
  bool overlaps(const AABox<T, DIMS>& aabox) const {
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      if (_data[maxIndex] <= aabox[minIndex] || _data[minIndex] >= aabox[maxIndex]) {
        return false;
      }
    }

    return true;
  }

  /**
   * Returns true if another bounding both is fully contained within
   * this bounding box
   */
  bool contains(const AABox<T, DIMS>& aabox) const {
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      if (aabox[minIndex] < _data[minIndex] || aabox[maxIndex] > _data[maxIndex]) {
        return false;
      }
    }

    return true;
  }

  /**
   * Returns true if a point is contained within the bounds
   */
  bool contains(const Point<T, DIMS>& pt) const {
    std::size_t minIndex, maxIndex;

    for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
      if (pt[minIndex] < _data[minIndex] || pt[minIndex] > _data[maxIndex]) {
        return false;
      }
    }

    return true;
  }

  // AABox<T, DIMS>& transform(const AffineMatrix<T, DIMS+1>& transformMat);
  // friend AABox<T, DIMS> operator*(const AffineMatrix<T, DIMS+1>& transformMat, const
  // AABox<T, DIMS>& aabox);

  /**
   * Extends a bounding box so that it would contain a specific point.
   */
  AABox<T, DIMS>& encapsulate(const Point<T, DIMS>& pt) {
    return encapsulate(std::array<T, DIMS>(pt));
  }

  AABox<T, DIMS>& encapsulate(const std::array<T, DIMS>& pt) {
    std::size_t minIndex, maxIndex;
    if (isEmpty()) {
      for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
        _data[minIndex] = pt[minIndex];
        _data[maxIndex] = pt[minIndex];
      }
    } else {
      for (minIndex = 0, maxIndex = DIMS; minIndex < DIMS; ++minIndex, ++maxIndex) {
        if (pt[minIndex] < _data[minIndex]) {
          _data[minIndex] = pt[minIndex];
        } else if (pt[minIndex] > _data[maxIndex]) {
          _data[maxIndex] = pt[minIndex];
        }
      }
    }

    return *this;
  }

  /**
   * Sets this bounding box to be equal to another
   */
  AABox<T, DIMS>& operator=(const AABox<T, DIMS>& aabox) { return set(aabox); }

  /**
   * Gets the min/max of the bounds at a specific index.
   */
  const T& operator[](std::size_t i) const {
    RUNTIME_EX_ASSERT(i < DIMS * 2, "index " + std::to_string(i) + " is out-of-bounds.");

    return _data[i];
  }
  T& operator[](std::size_t i) {
    return const_cast<T&>(static_cast<const AABox&>(*this)[i]);
  }

  /**
   * Equality operator
   */
  bool operator==(const AABox<T, DIMS>& aabox) const { return _data == aabox._data; }
  bool operator!=(const AABox<T, DIMS>& aabox) { return _data != aabox._data; }

  operator std::string() const {
    std::string s = "[";

    if (DIMS > 0) {
      s += std::to_string(_data[0]);
      for (std::size_t i = 1; i < DIMS * 2; ++i) {
        s += ", " + std::to_string(_data[i]);
      }
    }

    s += "]";

    return s;
  }

  friend std::ostream& operator<<(std::ostream& ostr, const AABox<T, DIMS>& aabox) {
    return ostr << std::string(aabox);
  }

 protected:
  std::array<T, DIMS * 2> _data;

  static const T pos_infinity;
  static const T neg_infinity;
};

template <typename T, int DIMS>
const T AABox<T, DIMS>::pos_infinity = std::numeric_limits<T>::infinity();

template <typename T, int DIMS>
const T AABox<T, DIMS>::neg_infinity = -std::numeric_limits<T>::infinity();

}  // namespace Math

}  // namespace gfx

#endif  // GFXDRIVER_MATH_AABOX_H_
