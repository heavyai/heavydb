/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

struct BaseTypeGLSL {
  BaseTypeGLSL() = delete;
  virtual ~BaseTypeGLSL() = default;

  virtual TypeGLSLShPtr clone() const = 0;
  virtual uint32_t numComponents() const = 0;
  virtual uint32_t numBytes() const = 0;
  virtual uint32_t numComponentBytes() const = 0;

  BufferAttrType attrType() const { return attr_type_; }
  BufferAttrType baseType() const;
  std::string enumString() const { return to_string(attr_type_); }
  std::string declString() const { return to_string_glsl_decl(attr_type_); }

  inline friend bool operator==(const BaseTypeGLSL& lhs, const BaseTypeGLSL& rhs) {
    return (lhs.attr_type_ == rhs.attr_type_);
  }

  inline friend bool operator!=(const BaseTypeGLSL& lhs, const BaseTypeGLSL& rhs) {
    return (lhs.attr_type_ != rhs.attr_type_);
  }

 protected:
  BufferAttrType attr_type_;

  BaseTypeGLSL(const BufferAttrType attr_type) : attr_type_(attr_type) {}
};

template <typename T, uint32_t num_components>
struct TypeGLSL : BaseTypeGLSL {
  TypeGLSL(const BufferAttrType attr_type) : BaseTypeGLSL(attr_type) {}
  TypeGLSL() = delete;
  ~TypeGLSL() override = default;

  TypeGLSLShPtr clone() const override {
    return std::make_shared<TypeGLSL<T, num_components>>(*this);
  }

  uint32_t numComponents() const override { return num_components; }
  uint32_t numBytes() const override { return sizeof(T) * num_components; }
  uint32_t numComponentBytes() const override { return sizeof(T); }
};

// SPECIALIZATIONS

// UNSIGNED INT:

template <>
TypeGLSL<uint32_t, 1>::TypeGLSL();

template <>
TypeGLSL<uint32_t, 2>::TypeGLSL();

template <>
TypeGLSL<uint32_t, 3>::TypeGLSL();

template <>
TypeGLSL<uint32_t, 4>::TypeGLSL();

// INT:

template <>
TypeGLSL<int32_t, 1>::TypeGLSL();

template <>
TypeGLSL<int32_t, 2>::TypeGLSL();

template <>
TypeGLSL<int32_t, 3>::TypeGLSL();

template <>
TypeGLSL<int32_t, 4>::TypeGLSL();

// FLOAT

template <>
TypeGLSL<float, 1>::TypeGLSL();

template <>
TypeGLSL<float, 2>::TypeGLSL();

template <>
TypeGLSL<float, 3>::TypeGLSL();

template <>
TypeGLSL<float, 4>::TypeGLSL();

template <>
TypeGLSL<float, 6>::TypeGLSL();

// DOUBLE

template <>
TypeGLSL<double, 1>::TypeGLSL();

template <>
TypeGLSL<double, 2>::TypeGLSL();

template <>
TypeGLSL<double, 3>::TypeGLSL();

template <>
TypeGLSL<double, 4>::TypeGLSL();

template <>
TypeGLSL<double, 6>::TypeGLSL();
// UINT64

template <>
TypeGLSL<uint64_t, 1>::TypeGLSL();

template <>
TypeGLSL<uint64_t, 2>::TypeGLSL();

template <>
TypeGLSL<uint64_t, 3>::TypeGLSL();

template <>
TypeGLSL<uint64_t, 4>::TypeGLSL();

// INT64

template <>
TypeGLSL<int64_t, 1>::TypeGLSL();

template <>
TypeGLSL<int64_t, 2>::TypeGLSL();

template <>
TypeGLSL<int64_t, 3>::TypeGLSL();

template <>
TypeGLSL<int64_t, 4>::TypeGLSL();

}  // namespace gfx
