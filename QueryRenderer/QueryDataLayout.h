/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "GfxDriver/Resources/Types.h"
#include "QueryRenderer/Projections/Types.h"
#include "QueryRenderer/Utils/TypeUtils.h"

class TargetMetaInfo;

namespace Analyzer {
class TargetEntry;
}  // namespace Analyzer

namespace Catalog_Namespace {
class Catalog;
}

namespace QueryRenderer {

class QueryDataLayout {
 public:
  struct AttrAliasInfo {
    const std::string attr_alias;
    const SQLTypeInfo type_info;
    const int table_id;
    const int col_id;
    bool is_projected;

    explicit AttrAliasInfo(const std::string& attr_alias,
                           const SQLTypeInfo& type_info,
                           const int table_id = -1,
                           const int col_id = -1)
        : attr_alias{attr_alias}
        , type_info{type_info}
        , table_id{table_id}
        , col_id{col_id}
        , is_projected{false} {};

    ~AttrAliasInfo() = default;

    bool operator==(const AttrAliasInfo& other) const {
      return type_info == other.type_info;
    }
  };
  using TypeConversionFunc =
      std::function<gfx::BufferAttrType(const std::string&, const SQLTypeInfo&)>;

  enum class LayoutType { kVertexInterleaved, kVertexSequential, kStorage };

  // NOTE: conversion_func is lazily called
  explicit QueryDataLayout(
      const std::vector<AttrAliasInfo>& attr_infos,
      const LayoutType layout_type = LayoutType::kVertexInterleaved,
      TypeConversionFunc conversion_func = query_sql_type_to_render_type,
      const int64_t invalid_key = std::numeric_limits<int64_t>::max());

  explicit QueryDataLayout(
      const std::vector<AttrAliasInfo>&& attr_infos,
      const LayoutType layout_type = LayoutType::kVertexInterleaved,
      TypeConversionFunc conversion_func = query_sql_type_to_render_type,
      const int64_t invalid_key = std::numeric_limits<int64_t>::max());

  explicit QueryDataLayout(
      const std::vector<TargetMetaInfo>& targets_meta,
      const LayoutType layout_type = LayoutType::kVertexInterleaved,
      TypeConversionFunc conversion_func = query_sql_type_to_render_type,
      const int64_t invalid_key = std::numeric_limits<int64_t>::max());

  explicit QueryDataLayout(
      const Catalog_Namespace::Catalog* catalog,
      const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets,
      const LayoutType layout_type = LayoutType::kVertexInterleaved,
      TypeConversionFunc conversion_func = query_sql_type_to_render_type,
      const int64_t invalid_key = std::numeric_limits<int64_t>::max());

  ~QueryDataLayout();

  int64_t getInvalidKey() const { return invalid_key_; }

  size_t numAttributes() const;

  gfx::BufferLayoutShPtr getBufferLayout() const { return converted_layout_; }

  bool hasAttribute(const std::string& attr) const;

  const SQLTypeInfo* getAttrSQLTypeInfoPtr(const std::string& attr) const;
  const SQLTypeInfo& getAttrSQLTypeInfoRef(const std::string& attr) const;

  bool isDecimalAttr(const std::string& attr) const;
  uint64_t getDecimalExp(const std::string& attr) const;

  bool isKnownAlias(const std::string& alias) const;
  const AttrAliasInfo& getAliasInfo(const std::string& alias) const;

  std::vector<std::string> getAllAttrNames() const;
  std::vector<AttrAliasInfo> getAllAttrInfo() const;

  bool operator==(const QueryDataLayout& layout) const;
  bool operator!=(const QueryDataLayout& layout) const { return !operator==(layout); }

  void setSpatialProjection(ProjectionType projectionType);
  ProjectionType getSpatialProjection() const;
  bool hasSpatialProjectionForAttr(const std::string& attr) const;
  void setSpatialProjectionForAttr(const std::string& attr, const bool is_projected);

 private:
  const int64_t invalid_key_;
  static const std::string dummy_prefix_;
  gfx::BufferLayoutShPtr converted_layout_;
  ProjectionType projected_state_;

  class AliasInfoMap;
  std::unique_ptr<AliasInfoMap> alias_info_map_;

  LayoutType layout_type_;
  TypeConversionFunc conversion_func_;

  void convertToVBOLayout();
  void convertToSSBOLayout();
  void convertLayout();
};

}  // namespace QueryRenderer
