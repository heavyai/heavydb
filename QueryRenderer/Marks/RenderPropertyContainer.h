/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <vector>

#include "QueryRenderer/Marks/BaseRenderProperty.h"
#include "Shared/EnumBitmaskOps.h"

namespace QueryRenderer {

class QueryRendererContext;
class BaseQueryDataTableShPtr;

struct RenderPropertyCreateInfo {
  enum class Type {
    kNone = 0,
    kExtendedCoords = 1 << 0,
    kFillColor = 1 << 1,
    kFillOpacity = 1 << 2,
    kStrokeColor = 1 << 3,
    kStrokeOpacity = 1 << 4,
    kStrokeWidth = 1 << 5,
    kLineJoinType = 1 << 6,
    kMiterLimit = 1 << 7
  };

  Type type = Type::kNone;
  RenderPropertyFlagBits flag_bits = RenderPropertyFlagBits::kUnspecified;
  bool is_required = true;
  RenderPropertyValue default_value = {};
  QueryDataType data_type;
  std::function<int(const std::string&)> string_convert_func = nullptr;
};

class RenderPropertyContainer {
 public:
  enum class PropId {
    // Base (all Marks)
    kX,
    kY,
    kOpacity,
    // Extended coords
    kX2,
    kXC,
    kY2,
    kYC,
    // Fill
    kFillColor,
    kFillOpacity,
    // Stroke
    kStrokeColor,
    kStrokeOpacity,
    kStrokeWidth,
    kLineJoinType,
    kMiterLimit,
    kCOUNT
  };

  explicit RenderPropertyContainer(QueryRendererContext& ctx,
                                   BaseRenderProperty::BaseMarkFacade& mark_facade,
                                   std::vector<RenderPropertyCreateInfo>&& create_infos,
                                   bool use_bda_for_coords = false);

  void initFromJSONObj(const JSONLocation& obj_loc, bool data_changed);

  // returns ids.size changed, id values changed
  std::pair<bool, bool> initIds(const bool data_changed, BaseDataTableShPtr data);

  bool isFillActive() const;
  bool isStrokeActive() const;

  // Return property or nullptr
  BaseRenderProperty* getProperty(PropId id) const;

  const std::vector<BaseRenderProperty*>& getProperties() const;
  const std::vector<BaseRenderProperty*>& getCoordProperties() const;

  // all query-based shaders should have a "key"
  BaseRenderPropertyUqPtr key;
  std::vector<BaseRenderPropertyUqPtr> ids;

  using ValidateFunc = std::function<void(const std::string&, const JSONLocation&)>;
  using JSONParseCBFunc = std::function<void(BaseRenderProperty*, const JSONLocation&)>;
  using JSONParseEmptyCBFunc =
      std::function<void(BaseRenderProperty*,
                         const JSONLocation&,
                         const RenderPropertyValue& default_value)>;

 private:
  rapidjson::Pointer properties_json_path_;
  QueryRendererContext& ctx_;
  BaseRenderProperty::BaseMarkFacade& mark_facade_;
  const std::vector<RenderPropertyCreateInfo> create_infos_;
  RenderPropertyCreateInfo::Type prop_types_;

  // properties
  std::map<PropId, BaseRenderPropertyUqPtr> properties_map_;
  std::vector<BaseRenderProperty*> properties_;
  std::vector<BaseRenderProperty*> coord_properties_;

  // Visibility flags (checks if feature may not be fully transparent)
  bool is_fill_active_;
  bool is_stroke_active_;

  enum class CreateType { kFloat, kColor, kEnum, kCOUNT };
  BaseRenderPropertyUqPtr createProperty(const CreateType type,
                                         const std::string& name,
                                         const RenderPropertyCreateInfo& ci);

  JSONLocation initPropFromJSONObj(const bool data_changed,
                                   const JSONLocation& prop_loc,
                                   BaseRenderProperty* prop,
                                   const RenderPropertyValue& default_value,
                                   ValidateFunc validate_type_func,
                                   JSONParseEmptyCBFunc post_empty_func,
                                   JSONParseCBFunc post_update_func = nullptr,
                                   JSONParseCBFunc post_data_update_func = nullptr,
                                   JSONParseCBFunc post_up_to_date_func = nullptr);
};

std::ostream& operator<<(std::ostream& os, RenderPropertyCreateInfo::Type type);
std::ostream& operator<<(std::ostream& os, RenderPropertyContainer::PropId prop_id);

}  // namespace QueryRenderer

ENABLE_BITMASK_OPS(::QueryRenderer::RenderPropertyCreateInfo::Type);
