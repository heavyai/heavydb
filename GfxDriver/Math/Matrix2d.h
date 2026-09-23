/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GFXDRIVER_MATH_MATRIX2D_H_
#define GFXDRIVER_MATH_MATRIX2D_H_

#include <array>
#include <cmath>
#include <string>

namespace gfx {
namespace Math {

/**
 * @class 2x3 Matrix
 *
 * @description
 * A mat2d contains six elements defined as:
 * [a, c, tx,
 *  b, d, ty]
 *
 * This is a short form for the 3x3 matrix:
 * [a, c, tx,
 *  b, d, ty,
 *  0, 0, 1]
 *
 * The last row is ignored for better performance and smaller memory imprint
 */
template <typename T>
class Matrix2d {
 public:
  Matrix2d() : _data({1, 0, 0, 1, 0, 0}) {}
  Matrix2d(const std::array<T, 6>& mat) : _data(mat) {}
  Matrix2d(const Matrix2d& mat) : _data(mat._data) {}

  /**
   * Copies the values from one mat2d to this one
   */
  Matrix2d<T>& operator=(const Matrix2d<T>& mat) { return set(mat); }
  Matrix2d<T>& operator=(const std::array<T, 6>& mat) { return set(mat); }

  /**
   * Sets the matrix to another one.
   */
  Matrix2d<T>& set(const Matrix2d<T>& setter) {
    _data = setter._data;
    return *this;
  }

  /**
   * Set the components of a mat2d to the given values
   */
  Matrix2d<T>& set(const T a, const T b, const T c, const T d, const T tx, const T ty) {
    _data[0] = a;
    _data[1] = b;
    _data[2] = c;
    _data[3] = d;
    _data[4] = tx;
    _data[5] = ty;
    return *this;
  };

  Matrix2d<T>& set(const std::array<T, 6>& mat) {
    _data = mat;
    return *this;
  }

  /**
   * Set a mat2d to the identity matrix
   */
  Matrix2d<T>& identity() {
    _data[0] = 1;
    _data[1] = 0;
    _data[2] = 0;
    _data[3] = 1;
    _data[4] = 0;
    _data[5] = 0;
    return *this;
  };

  /**
   * Create a new mat2d with the given values
   */
  Matrix2d<T>& fromValues(const T a,
                          const T b,
                          const T c,
                          const T d,
                          const T tx,
                          const T ty) {
    _data[0] = a;
    _data[1] = b;
    _data[2] = c;
    _data[3] = d;
    _data[4] = tx;
    _data[5] = ty;
    return *this;
  };

  /**
   * Inverts a mat2d
   */
  Matrix2d<T>& invert() {
    auto aa = _data[0], ab = _data[1], ac = _data[2], ad = _data[3], atx = _data[4],
         aty = _data[5];

    auto det = aa * ad - ab * ac;
    RUNTIME_EX_ASSERT(det != 0, "Matrix is not invertible. It has a determinant of 0.");
    det = 1.0 / det;

    _data[0] = ad * det;
    _data[1] = -ab * det;
    _data[2] = -ac * det;
    _data[3] = aa * det;
    _data[4] = (ac * aty - ad * atx) * det;
    _data[5] = (ab * atx - aa * aty) * det;
    return *this;
  };

  /**
   * Calculates the determinant of a mat2d
   */
  T determinant() const { return _data[0] * _data[3] - _data[1] * _data[2]; };

  /**
   * Multiplies two mat2d's
   * Equates to a * this;
   */
  Matrix2d<T>& multiply(const Matrix2d<T>& a) {
    auto a0 = a._data[0], a1 = a._data[1], a2 = a._data[2], a3 = a._data[3],
         a4 = a._data[4], a5 = a._data[5], b0 = _data[0], b1 = _data[1], b2 = _data[2],
         b3 = _data[3], b4 = _data[4], b5 = _data[5];
    _data[0] = a0 * b0 + a2 * b1;
    _data[1] = a1 * b0 + a3 * b1;
    _data[2] = a0 * b2 + a2 * b3;
    _data[3] = a1 * b2 + a3 * b3;
    _data[4] = a0 * b4 + a2 * b5 + a4;
    _data[5] = a1 * b4 + a3 * b5 + a5;
    return *this;
  };

  /**
   * Rotates the mat2d by the given angle
   */
  Matrix2d<T>& rotate(const T rad) {
    auto a0 = _data[0], a1 = _data[1], a2 = _data[2], a3 = _data[3], s = std::sin(rad),
         c = std::cos(rad);
    _data[0] = a0 * c + a2 * s;
    _data[1] = a1 * c + a3 * s;
    _data[2] = a0 * -s + a2 * c;
    _data[3] = a1 * -s + a3 * c;
    return *this;
  };

  /**
   * Scales the mat2d by the dimensions in the given vec2
   **/
  Matrix2d<T>& scale(const std::array<T, 2>& v) {
    auto a0 = _data[0], a1 = _data[1], a2 = _data[2], a3 = _data[3], v0 = v[0], v1 = v[1];
    _data[0] = a0 * v0;
    _data[1] = a1 * v0;
    _data[2] = a2 * v1;
    _data[3] = a3 * v1;
    return *this;
  };

  /**
   * Translates the mat2d by the dimensions in the given vec2
   **/
  Matrix2d<T>& translate(const std::array<T, 2>& v) {
    auto a0 = _data[0], a1 = _data[1], a2 = _data[2], a3 = _data[3], a4 = _data[4],
         a5 = _data[5], v0 = v[0], v1 = v[1];
    _data[4] = a0 * v0 + a2 * v1 + a4;
    _data[5] = a1 * v0 + a3 * v1 + a5;
    return *this;
  };

  /**
   * Creates a matrix from a given angle
   * This is equivalent to (but much faster than):
   *
   *     mat2d.identity(dest);
   *     mat2d.rotate(dest, dest, rad);
   */
  Matrix2d<T>& fromRotation(const T rad) {
    auto s = std::sin(rad), c = std::cos(rad);
    _data[0] = c;
    _data[1] = s;
    _data[2] = -s;
    _data[3] = c;
    _data[4] = 0;
    _data[5] = 0;
    return *this;
  }

  /**
   * Creates a matrix from a vector scaling
   * This is equivalent to (but much faster than):
   *
   *     mat2d.identity(dest);
   *     mat2d.scale(dest, dest, vec);
   */
  Matrix2d<T>& fromScaling(const std::array<T, 2>& v) {
    _data[0] = v[0];
    _data[1] = 0;
    _data[2] = 0;
    _data[3] = v[1];
    _data[4] = 0;
    _data[5] = 0;
    return *this;
  }

  /**
   * Creates a matrix from a vector translation
   * This is equivalent to (but much faster than):
   *
   *     mat2d.identity(dest);
   *     mat2d.translate(dest, dest, vec);
   */
  Matrix2d<T>& fromTranslation(const std::array<T, 2>& v) {
    _data[0] = 1;
    _data[1] = 0;
    _data[2] = 0;
    _data[3] = 1;
    _data[4] = v[0];
    _data[5] = v[1];
    return *this;
  }

  /**
   * Returns a string representation of a mat2d
   */
  operator std::string() const {
    return "Mat2d(" + std::to_string(_data[0]) + ", " + std::to_string(_data[1]) + ", " +
           std::to_string(_data[2]) + ", " + std::to_string(_data[3]) + ", " +
           std::to_string(_data[4]) + ", " + std::to_string(_data[5]) + ")";
  }

  /**
   * Returns Frobenius norm of a mat2d
   */
  T frob() const {
    return (std::sqrt(std::pow(_data[0], 2) + std::pow(_data[1], 2) +
                      std::pow(_data[2], 2) + std::pow(_data[3], 2) +
                      std::pow(_data[4], 2) + std::pow(_data[5], 2) + 1));
  };

  /**
   * Adds two mat2d's
   */
  Matrix2d<T>& add(const Matrix2d<T>& a) {
    _data[0] = a._data[0] + _data[0];
    _data[1] = a._data[1] + _data[1];
    _data[2] = a._data[2] + _data[2];
    _data[3] = a._data[3] + _data[3];
    _data[4] = a._data[4] + _data[4];
    _data[5] = a._data[5] + _data[5];
    return *this;
  };

  /**
   * Subtracts matrix b from matrix
   */
  Matrix2d<T>& subtract(const Matrix2d<T>& b) {
    _data[0] -= b._data[0];
    _data[1] -= b._data[1];
    _data[2] -= b._data[2];
    _data[3] -= b._data[3];
    _data[4] -= b._data[4];
    _data[5] -= b._data[5];
    return *this;
  };

  /**
   * Multiply each element of the matrix by a scalar.
   */
  Matrix2d<T>& multiplyScalar(const T b) {
    _data[0] *= b;
    _data[1] *= b;
    _data[2] *= b;
    _data[3] *= b;
    _data[4] *= b;
    _data[5] *= b;
    return *this;
  };

  /**
   * Returns whether or not the matrices have exactly the same elements in the same
   * position (when compared with ===)
   */

  // TODO(croot): use epsilon?
  bool operator==(const Matrix2d<T>& mat) const { return _data == mat._data; }
  bool operator!=(const Matrix2d<T>& mat) const { return _data != mat._data; }

  std::array<T, 6> getDataArray() const { return _data; }
  const std::array<T, 6>& getDataArrayRef() const { return _data; }

 private:
  std::array<T, 6> _data;
};

}  // namespace Math
}  // namespace gfx

#endif  // GFXDRIVER_MATH_MATRIX2D_H_
