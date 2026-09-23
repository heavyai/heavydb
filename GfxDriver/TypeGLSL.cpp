/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/TypeGLSL.h"

#include "Logger/Logger.h"

namespace gfx {

BufferAttrType BaseTypeGLSL::baseType() const {
  switch (attr_type_) {
    case BufferAttrType::kUint:
    case BufferAttrType::kVec2ui:
    case BufferAttrType::kVec3ui:
    case BufferAttrType::kVec4ui:
      return BufferAttrType::kUint;
    case BufferAttrType::kInt:
    case BufferAttrType::kVec2i:
    case BufferAttrType::kVec3i:
    case BufferAttrType::kVec4i:
      return BufferAttrType::kInt;
    case BufferAttrType::kFloat:
    case BufferAttrType::kVec2f:
    case BufferAttrType::kVec3f:
    case BufferAttrType::kVec4f:
    case BufferAttrType::kMat3x2f:
      return BufferAttrType::kFloat;
    case BufferAttrType::kDouble:
    case BufferAttrType::kVec2d:
    case BufferAttrType::kVec3d:
    case BufferAttrType::kVec4d:
    case BufferAttrType::kMat3x2d:
      return BufferAttrType::kDouble;
    case BufferAttrType::kUint64:
    case BufferAttrType::kVec2ui64:
    case BufferAttrType::kVec3ui64:
    case BufferAttrType::kVec4ui64:
      return BufferAttrType::kUint64;
    case BufferAttrType::kInt64:
    case BufferAttrType::kVec2i64:
    case BufferAttrType::kVec3i64:
    case BufferAttrType::kVec4i64:
      return BufferAttrType::kInt64;
    default:
      CHECK(false) << "undefined attr type: " << static_cast<unsigned int>(attr_type_);
  }
  return BufferAttrType::kInt;
}

/*****************
 * UNSIGNED INT
 *****************/

template <>
TypeGLSL<uint32_t, 1>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kUint) {}

template <>
TypeGLSL<uint32_t, 2>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec2ui) {}

template <>
TypeGLSL<uint32_t, 3>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec3ui) {}

template <>
TypeGLSL<uint32_t, 4>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec4ui) {}

/*****************
 * INT
 *****************/

template <>
TypeGLSL<int32_t, 1>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kInt) {}

template <>
TypeGLSL<int32_t, 2>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec2i) {}

template <>
TypeGLSL<int32_t, 3>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec3i) {}

template <>
TypeGLSL<int32_t, 4>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec4i) {}

/*****************
 * FLOAT
 *****************/

template <>
TypeGLSL<float, 1>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kFloat) {}

template <>
TypeGLSL<float, 2>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec2f) {}

template <>
TypeGLSL<float, 3>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec3f) {}

template <>
TypeGLSL<float, 4>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec4f) {}

template <>
TypeGLSL<float, 6>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kMat3x2f) {}

/*****************
 * DOUBLE
 *****************/

template <>
TypeGLSL<double, 1>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kDouble) {}

template <>
TypeGLSL<double, 2>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec2d) {}

template <>
TypeGLSL<double, 3>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec3d) {}

template <>
TypeGLSL<double, 4>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec4d) {}

template <>
TypeGLSL<double, 6>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kMat3x2d) {}

/*****************
 * UINT64
 *****************/

template <>
TypeGLSL<uint64_t, 1>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kUint64) {}

template <>
TypeGLSL<uint64_t, 2>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec2ui64) {}

template <>
TypeGLSL<uint64_t, 3>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec3ui64) {}

template <>
TypeGLSL<uint64_t, 4>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec4ui64) {}

/*****************
 * INT64
 *****************/

template <>
TypeGLSL<int64_t, 1>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kInt64) {}

template <>
TypeGLSL<int64_t, 2>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec2i64) {}

template <>
TypeGLSL<int64_t, 3>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec3i64) {}

template <>
TypeGLSL<int64_t, 4>::TypeGLSL() : BaseTypeGLSL(BufferAttrType::kVec4i64) {}

}  // namespace gfx
