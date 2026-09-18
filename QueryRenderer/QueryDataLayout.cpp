/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/QueryDataLayout.h"

#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/ShaderBlockLayout.h"
#include "GfxDriver/Resources/Types.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "QueryRenderer/Utils/AnalyzerUtils.h"
#include "Shared/SqlTypesLayout.h"

namespace QueryRenderer {

using gfx::BufferAttrType;
using gfx::BufferLayoutShPtr;
using gfx::InterleavedBufferLayout;
using gfx::SequentialBufferLayout;
using gfx::ShaderBlockLayout;
using gfx::ShaderBlockLayoutShPtr;
using gfx::ShaderBlockType;

const std::string QueryDataLayout::dummy_prefix_ = "_dummy";

class QueryDataLayout::AliasInfoMap {
 public:
  void insert(const AttrAliasInfo& alias_info);
  bool contains(const std::string& alias) const;
  size_t size() const;

  const AttrAliasInfo& getAliasInfo(uint32_t index) const;
  AttrAliasInfo& getAliasInfoForAlias(const std::string& alias);

  std::vector<std::string> getAllNames() const;
  std::vector<QueryDataLayout::AttrAliasInfo> copyInfos() const;
  std::string getAllNamesString() const;

  bool operator==(const AliasInfoMap& other) const;

 private:
  std::vector<AttrAliasInfo> alias_infos_;
  std::map<const std::string, uint32_t> alias_tag_to_alias_info_map_;

  friend class QueryDataLayout;
};

void QueryDataLayout::AliasInfoMap::insert(const AttrAliasInfo& alias_info) {
  alias_infos_.emplace_back(alias_info);
  auto insert_itr = alias_tag_to_alias_info_map_.emplace(alias_info.attr_alias,
                                                         alias_infos_.size() - 1);
  RUNTIME_EX_ASSERT(insert_itr.second,
                    "Cannot add AttrAliasInfo. There are duplicate entries of "
                    "attribute alias \"" +
                        alias_info.attr_alias + "\".");
}

bool QueryDataLayout::AliasInfoMap::contains(const std::string& alias) const {
  return alias_tag_to_alias_info_map_.find(alias) != alias_tag_to_alias_info_map_.end();
}

size_t QueryDataLayout::AliasInfoMap::size() const {
  return alias_infos_.size();
}

const QueryDataLayout::AttrAliasInfo& QueryDataLayout::AliasInfoMap::getAliasInfo(
    uint32_t index) const {
  CHECK_LT(index, alias_infos_.size());
  return alias_infos_[index];
}

QueryDataLayout::AttrAliasInfo& QueryDataLayout::AliasInfoMap::getAliasInfoForAlias(
    const std::string& alias) {
  auto itr = alias_tag_to_alias_info_map_.find(alias);
  RUNTIME_EX_ASSERT(itr != alias_tag_to_alias_info_map_.end(),
                    "The attr alias \"" + alias + "\" is not in the QueryDataLayout.");
  CHECK_LT(itr->second, alias_infos_.size());
  return alias_infos_[itr->second];
}

std::vector<std::string> QueryDataLayout::AliasInfoMap::getAllNames() const {
  std::vector<std::string> rtn;
  for (auto const& info : alias_infos_) {
    rtn.push_back(info.attr_alias);
  }
  return rtn;
}

std::string QueryDataLayout::AliasInfoMap::getAllNamesString() const {
  std::string aliases = "[";
  for (const auto& itr : alias_tag_to_alias_info_map_) {
    if (aliases.size() > 1) {
      aliases += ", ";
    }
    aliases += alias_infos_[itr.second].attr_alias;
  }
  aliases += "]";
  return aliases;
}

std::vector<QueryDataLayout::AttrAliasInfo> QueryDataLayout::AliasInfoMap::copyInfos()
    const {
  std::vector<AttrAliasInfo> rtn;
  for (auto const& info : alias_infos_) {
    rtn.push_back(info);
  }
  return rtn;
}

bool QueryDataLayout::AliasInfoMap::operator==(const AliasInfoMap& other) const {
  return (alias_infos_ == other.alias_infos_) &&
         (alias_tag_to_alias_info_map_ == other.alias_tag_to_alias_info_map_);
}

QueryDataLayout::QueryDataLayout(const std::vector<AttrAliasInfo>& alias_infos,
                                 const LayoutType layout_type,
                                 TypeConversionFunc conversion_func,
                                 const int64_t invalid_key)
    : invalid_key_{invalid_key}
    , projected_state_{ProjectionType::kUndefined}
    , alias_info_map_{std::make_unique<AliasInfoMap>()}
    , layout_type_{layout_type}
    , conversion_func_{conversion_func} {
  for (auto const& alias_info : alias_infos) {
    conversion_func_(alias_info.attr_alias, alias_info.type_info);
    alias_info_map_->insert(alias_info);
  }
  convertLayout();
}

QueryDataLayout::QueryDataLayout(const std::vector<AttrAliasInfo>&& alias_infos,
                                 const LayoutType layout_type,
                                 TypeConversionFunc conversion_func,
                                 const int64_t invalid_key)
    : invalid_key_{invalid_key}
    , projected_state_{ProjectionType::kUndefined}
    , alias_info_map_{std::make_unique<AliasInfoMap>()}
    , layout_type_{layout_type}
    , conversion_func_{conversion_func} {
  decltype(alias_infos) tmpvec(std::move(alias_infos));
  for (auto& alias_info : tmpvec) {
    conversion_func_(alias_info.attr_alias, alias_info.type_info);
    alias_info_map_->insert(alias_info);
  }
  convertLayout();
}

QueryDataLayout::QueryDataLayout(const std::vector<TargetMetaInfo>& targets_meta,
                                 const LayoutType layout_type,
                                 TypeConversionFunc conversion_func,
                                 const int64_t invalid_key)
    : invalid_key_{invalid_key}
    , projected_state_{ProjectionType::kUndefined}
    , alias_info_map_{std::make_unique<AliasInfoMap>()}
    , layout_type_{layout_type}
    , conversion_func_{conversion_func} {
  for (const auto& meta_info : targets_meta) {
    const auto& attr_name = meta_info.get_resname();
    const auto& type_info = meta_info.get_type_info();

    // call the conversion func for validation
    conversion_func_(attr_name, type_info);
    alias_info_map_->insert(AttrAliasInfo(attr_name, type_info));
  }
  convertLayout();
}

QueryDataLayout::QueryDataLayout(
    const Catalog_Namespace::Catalog* catalog,
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets,
    const LayoutType layout_type,
    TypeConversionFunc conversion_func,
    const int64_t invalid_key)
    : invalid_key_{invalid_key}
    , projected_state_{ProjectionType::kUndefined}
    , alias_info_map_{std::make_unique<AliasInfoMap>()}
    , layout_type_{layout_type}
    , conversion_func_{conversion_func} {
  for (const auto& te : targets) {
    int table_id = -1, col_id = -1;
    const auto target_expr = te->get_expr();
    const auto& attr_name = te->get_resname();
    const auto& type_info = target_expr->get_type_info();

    // call the conversion func for validation
    conversion_func_(attr_name, type_info);

    std::tie(table_id, col_id) = get_table_id_col_id_from_target_expr(target_expr);
    alias_info_map_->insert(AttrAliasInfo(attr_name, type_info, table_id, col_id));
  }
  convertLayout();
}

QueryDataLayout::~QueryDataLayout() {}

size_t QueryDataLayout::numAttributes() const {
  return alias_info_map_->size();
}

void QueryDataLayout::convertLayout() {
  RENDER_LOG_SCOPE() << "this=" << this;
  CHECK(!converted_layout_);
  switch (layout_type_) {
    case LayoutType::kVertexInterleaved:
    case LayoutType::kVertexSequential:
      convertToVBOLayout();
      break;
    case LayoutType::kStorage:
      convertToSSBOLayout();
      break;
  }
}

void QueryDataLayout::convertToVBOLayout() {
  RENDER_LOG_SCOPE() << "this=" << this;
  auto populate_layout = [this](auto* layout) {
    for (const auto& alias_info : alias_info_map_->alias_infos_) {
      auto buffer_attr_type =
          conversion_func_(alias_info.attr_alias, alias_info.type_info);
      if (alias_info.type_info.is_array()) {
        CHECK(buffer_attr_type == BufferAttrType::kInt64 ||
              buffer_attr_type == BufferAttrType::kDouble);
        layout->addAttribute(alias_info.attr_alias + "_ptr", BufferAttrType::kUint64);
        layout->addAttribute(alias_info.attr_alias + "_idx", BufferAttrType::kUint64);
      } else if (buffer_attr_type == BufferAttrType::kInt64 &&
                 IS_GEO(alias_info.type_info.get_type())) {
        // geo pointer and count
        layout->addAttribute(alias_info.attr_alias, BufferAttrType::kUint64);
        layout->addAttribute(alias_info.attr_alias + "_num", BufferAttrType::kUint64);
      } else {
        layout->addAttribute(alias_info.attr_alias, buffer_attr_type);
      }
    }
  };
  switch (layout_type_) {
    case LayoutType::kVertexInterleaved: {
      converted_layout_ = std::make_shared<InterleavedBufferLayout>();
      populate_layout(dynamic_cast<InterleavedBufferLayout*>(converted_layout_.get()));
      break;
    }
    case LayoutType::kVertexSequential: {
      converted_layout_ = std::make_shared<SequentialBufferLayout>();
      populate_layout(dynamic_cast<SequentialBufferLayout*>(converted_layout_.get()));
      break;
    }
    default:
      CHECK(false);
      break;
  }
}

void QueryDataLayout::convertToSSBOLayout() {
  RENDER_LOG_SCOPE();
  converted_layout_ =
      std::make_shared<ShaderBlockLayout>(ShaderBlockType::kStorageBuffer);
  ShaderBlockLayout* layout = dynamic_cast<ShaderBlockLayout*>(converted_layout_.get());

  layout->beginAddingAttrs();
  for (const auto& alias_info : alias_info_map_->alias_infos_) {
    auto buffer_attr_type = conversion_func_(alias_info.attr_alias, alias_info.type_info);
    CHECK(!alias_info.type_info.is_array());
    if (buffer_attr_type == BufferAttrType::kInt64) {
      layout->addAttribute(alias_info.attr_alias, BufferAttrType::kInt64);
    } else if (buffer_attr_type == BufferAttrType::kUint64) {
      layout->addAttribute(alias_info.attr_alias, BufferAttrType::kUint64);
    } else {
      layout->addAttribute(alias_info.attr_alias, buffer_attr_type);
    }
  }
  layout->endAddingAttrs();
}

bool QueryDataLayout::hasAttribute(const std::string& attr) const {
  if (converted_layout_) {
    return converted_layout_->hasAttribute(attr);
  }

  return alias_info_map_->contains(attr);
}

const SQLTypeInfo* QueryDataLayout::getAttrSQLTypeInfoPtr(const std::string& attr) const {
  return &alias_info_map_->getAliasInfoForAlias(attr).type_info;
}

const SQLTypeInfo& QueryDataLayout::getAttrSQLTypeInfoRef(const std::string& attr) const {
  return *getAttrSQLTypeInfoPtr(attr);
}

bool QueryDataLayout::isDecimalAttr(const std::string& attr) const {
  if (alias_info_map_->contains(attr)) {
    return alias_info_map_->getAliasInfoForAlias(attr).type_info.is_decimal();
  } else {
    return false;
  }
}

uint64_t QueryDataLayout::getDecimalExp(const std::string& attr) const {
  auto const& alias_info = alias_info_map_->getAliasInfoForAlias(attr);
  RUNTIME_EX_ASSERT(alias_info.type_info.is_decimal(),
                    "The attr \"" + attr + "\" is not a decimal attr.");
  return exp_to_scale(alias_info.type_info.get_scale());
}

bool QueryDataLayout::isKnownAlias(const std::string& alias) const {
  if (alias_info_map_->contains(alias)) {
    return alias_info_map_->getAliasInfoForAlias(alias).col_id >= 0;
  }
  return false;
}

const QueryDataLayout::AttrAliasInfo& QueryDataLayout::getAliasInfo(
    const std::string& alias) const {
  if (!alias_info_map_->contains(alias)) {
    THROW_RUNTIME_EX(
        "Cannot find the column alias \"" + alias +
        "\" in the render data layout map. The map knows of these aliases: " +
        alias_info_map_->getAllNamesString() + ".");
  }
  return alias_info_map_->getAliasInfoForAlias(alias);
}

std::vector<std::string> QueryDataLayout::getAllAttrNames() const {
  return alias_info_map_->getAllNames();
}

std::vector<QueryDataLayout::AttrAliasInfo> QueryDataLayout::getAllAttrInfo() const {
  return alias_info_map_->copyInfos();
}

bool QueryDataLayout::operator==(const QueryDataLayout& layout) const {
  return *converted_layout_ == *(layout.converted_layout_);
}

void QueryDataLayout::setSpatialProjection(ProjectionType projection_type) {
  if (projection_type == projected_state_) {
    return;
  }
  for (auto alias_info : alias_info_map_->alias_infos_) {
    alias_info.is_projected = false;
  }
  projected_state_ = projection_type;
}

ProjectionType QueryDataLayout::getSpatialProjection() const {
  return projected_state_;
}

bool QueryDataLayout::hasSpatialProjectionForAttr(const std::string& attr) const {
  if (projected_state_ == ProjectionType::kUndefined) {
    return false;
  }
  return alias_info_map_->getAliasInfoForAlias(attr).is_projected;
}

void QueryDataLayout::setSpatialProjectionForAttr(const std::string& attr,
                                                  const bool is_projected) {
  RUNTIME_EX_ASSERT(projected_state_ != ProjectionType::kUndefined,
                    "Cannot set spatial projection for \"" + attr +
                        "\". The layout has an undefined projection.");
  auto& alias_info = alias_info_map_->getAliasInfoForAlias(attr);
  alias_info.is_projected = is_projected;
}

}  // namespace QueryRenderer
