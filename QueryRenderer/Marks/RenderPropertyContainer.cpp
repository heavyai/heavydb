/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/RenderPropertyContainer.h"

#include <array>

#include "Analyzer/Analyzer.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Cache/RowIdHitTestOffsetData.h"
#include "QueryRenderer/Marks/RenderProperty.h"
#include "QueryRenderer/Marks/Utils.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

namespace {
RenderPropertyContainer::ValidateFunc validateNumPropFunc() {
  return [](const std::string& prop_name, const JSONLocation& json_loc) {
    CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
    RUNTIME_EX_ASSERT(
        (json_loc.isObject() || json_loc.isNumber()),
        RapidJSONUtils::createJsonParseError(
            json_loc,
            "\"" + prop_name +
                "\" mark property must be a scale/data reference or a number."));
  };
}

RenderPropertyContainer::ValidateFunc validateColorPropFunc() {
  return [](const std::string& prop_name, const JSONLocation& json_loc) {
    CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
    RUNTIME_EX_ASSERT((json_loc.isObject() || json_loc.isString()) || json_loc.isInt() ||
                          json_loc.isUint(),
                      RapidJSONUtils::createJsonParseError(
                          json_loc,
                          "\"" + prop_name +
                              "\" mark color property must be a scale/data reference, a "
                              "string, or a color packed into a 32-bit int/uint."));
  };
}

RenderPropertyContainer::ValidateFunc validateEnumPropFunc() {
  return [](const std::string& prop_name, const JSONLocation& json_loc) {
    CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
    RUNTIME_EX_ASSERT(
        json_loc.isInt() || json_loc.isUint64() || json_loc.isString(),
        RapidJSONUtils::createJsonParseError(json_loc,
                                             "\"" + prop_name +
                                                 "\" mark enum property must be an enum "
                                                 "value (i.e. an int or a string)."));
  };
}

}  // namespace

RenderPropertyContainer::RenderPropertyContainer(
    QueryRendererContext& ctx,
    BaseRenderProperty::BaseMarkFacade& mark_facade,
    std::vector<RenderPropertyCreateInfo>&& create_infos,
    bool use_bda_for_coords)
    : key{std::make_unique<KeyRenderProperty>("key", ctx, mark_facade)}
    , ctx_{ctx}
    , mark_facade_{mark_facade}
    , create_infos_{std::move(create_infos)}
    , prop_types_{RenderPropertyCreateInfo::Type::kNone}
    , is_fill_active_{false}
    , is_stroke_active_{false} {
  //
  // Create props
  //
  static constexpr auto default_coord_flag_bits =
      RenderProperty<float>::kDefaultFlagBits | RenderPropertyFlagBits::kIsCoord;
  auto coord_flag_bits = use_bda_for_coords
                             ? default_coord_flag_bits | RenderPropertyFlagBits::kUseBDA
                             : default_coord_flag_bits;

  // Common properties supported by all Marks
  properties_map_[PropId::kX] =
      std::make_unique<RenderProperty<float>>("x", ctx, mark_facade, coord_flag_bits);
  properties_map_[PropId::kY] =
      std::make_unique<RenderProperty<float>>("y", ctx, mark_facade, coord_flag_bits);

  properties_map_[PropId::kOpacity] =
      std::make_unique<RenderProperty<float>>("opacity", ctx, mark_facade);

  using type = RenderPropertyCreateInfo::Type;
  for (auto const& ci : create_infos_) {
    // Check for duplicates
    CHECK(!any_bits_set(prop_types_ & ci.type))
        << "Duplicate RenderPropertyType " << ci.type
        << ". A RenderPropertyType can only be declared once";

    switch (ci.type) {
      case type::kExtendedCoords:
        properties_map_[PropId::kX2] = std::make_unique<RenderProperty<float>>(
            "x2", ctx, mark_facade, coord_flag_bits);
        properties_map_[PropId::kXC] = std::make_unique<RenderProperty<float>>(
            "xc", ctx, mark_facade, coord_flag_bits);
        properties_map_[PropId::kY2] = std::make_unique<RenderProperty<float>>(
            "y2", ctx, mark_facade, coord_flag_bits);
        properties_map_[PropId::kYC] = std::make_unique<RenderProperty<float>>(
            "yc", ctx, mark_facade, coord_flag_bits);
        break;
      case type::kFillColor:
        properties_map_[PropId::kFillColor] =
            createProperty(CreateType::kColor, "fillColor", ci);
        break;
      case type::kFillOpacity:
        properties_map_[PropId::kFillOpacity] =
            createProperty(CreateType::kFloat, "fillOpacity", ci);
        break;
      case type::kStrokeColor:
        properties_map_[PropId::kStrokeColor] =
            createProperty(CreateType::kColor, "strokeColor", ci);
        break;
      case type::kStrokeOpacity:
        properties_map_[PropId::kStrokeOpacity] =
            createProperty(CreateType::kFloat, "strokeOpacity", ci);
        break;
      case type::kStrokeWidth:
        properties_map_[PropId::kStrokeWidth] =
            createProperty(CreateType::kFloat, "strokeWidth", ci);
        break;
      case type::kLineJoinType:
        properties_map_[PropId::kLineJoinType] =
            createProperty(CreateType::kEnum, "lineJoin", ci);
        break;
      case type::kMiterLimit:
        properties_map_[PropId::kMiterLimit] =
            createProperty(CreateType::kFloat, "miterLimit", ci);
        break;
      default:
        UNREACHABLE() << "Unknown RenderPropertyType";
    }
    prop_types_ |= ci.type;
  }

  // Sanity check combinations (relax these?)
  if (any_bits_set(prop_types_ & type::kFillOpacity)) {
    CHECK(any_bits_set(prop_types_ & type::kFillColor))
        << "FillColor must accompany FillOpacity";
  }
  if (any_bits_set(prop_types_ & type::kStrokeOpacity)) {
    CHECK(any_bits_set(prop_types_ & type::kStrokeColor))
        << "StrokeColor must accompany StrokeOpacity";
  }

  // Build secondary lookups
  properties_.reserve(properties_map_.size());
  for (auto const& itr : properties_map_) {
    auto* prop = itr.second.get();
    properties_.emplace_back(prop);
    if (prop->isCoord()) {
      coord_properties_.emplace_back(itr.second.get());
    }
  }
}

JSONLocation RenderPropertyContainer::initPropFromJSONObj(
    const bool data_changed,
    const JSONLocation& prop_loc,
    BaseRenderProperty* prop,
    const RenderPropertyValue& default_value,
    ValidateFunc validate_type_func,
    JSONParseEmptyCBFunc post_empty_func,
    JSONParseCBFunc post_update_func,
    JSONParseCBFunc post_data_update_func,
    JSONParseCBFunc post_up_to_date_func) {
  auto prop_name = prop->getName();
  auto const prop_item_loc = prop_loc.getMember(prop_name);
  auto const& data = mark_facade_.getDataPtr();
  if (prop_item_loc.isValid()) {
    auto const prev_path = prop->getJsonPath();
    auto const prop_json_path = prop_item_loc.getPathRef();
    if (!ctx_.isJSONCacheUpToDate(prev_path, prop_item_loc)) {
      if (validate_type_func) {
        validate_type_func(prop_name, prop_item_loc);
      }
      prop->initializeFromJSONObj(prop_item_loc, data);
      if (post_update_func) {
        post_update_func(prop, prop_item_loc);
      }
    } else if (data_changed) {
      prop->initializeFromJSONObj(prop_item_loc, data);
      if (post_data_update_func) {
        post_data_update_func(prop, prop_item_loc);
      }
    } else {
      prop->updateJsonPath(prop_json_path);
      if (post_up_to_date_func) {
        post_up_to_date_func(prop, prop_item_loc);
      }
    }
  } else {
    // clear the property of all scale/data references
    prop->clear();
    if (post_empty_func) {
      post_empty_func(prop, prop_loc, default_value);
    }
  }

  return prop_item_loc;
}

std::unique_ptr<BaseRenderProperty> RenderPropertyContainer::createProperty(
    const CreateType type,
    const std::string& name,
    const RenderPropertyCreateInfo& ci) {
  static constexpr std::array<RenderPropertyFlagBits,
                              static_cast<int>(CreateType::kCOUNT)>
      kDefaultFlagBits = {RenderProperty<float>::kDefaultFlagBits,
                          ColorRenderProperty::kDefaultFlagBits,
                          EnumRenderProperty::kDefaultFlagBits};

  RenderPropertyFlagBits flag_bits = ci.flag_bits;
  if (flag_bits == RenderPropertyFlagBits::kUnspecified) {
    flag_bits = kDefaultFlagBits[static_cast<int>(type)];
  }
  switch (type) {
    case CreateType::kFloat:
      return std::make_unique<RenderProperty<float>>(name, ctx_, mark_facade_, flag_bits);
    case CreateType::kColor:
      return std::make_unique<ColorRenderProperty>(name, ctx_, mark_facade_, flag_bits);
    case CreateType::kEnum:
      return std::make_unique<EnumRenderProperty>(
          name, ci.data_type, ctx_, mark_facade_, flag_bits, ci.string_convert_func);
    case CreateType::kCOUNT:
      CHECK(false) << "Invalid CreateType: kCOUNT";
  }
  UNREACHABLE();
  return nullptr;
}

void RenderPropertyContainer::initFromJSONObj(const JSONLocation& obj_loc,
                                              bool data_changed) {
  auto const prop_loc = obj_loc.getMember(JSONSchema_v1::Marks::kPropertiesProp);
  auto prev_path = properties_json_path_;
  properties_json_path_ = prop_loc.getPathRef();
  bool did_structure_change = false;

  auto const obj_num_check = validateNumPropFunc();
  auto const obj_color_check = validateColorPropFunc();
  auto const obj_enum_check = validateEnumPropFunc();

  //
  // Callback for default values for optional props that are not in the JSON
  //
  JSONParseEmptyCBFunc default_initializer =
      [](BaseRenderProperty* prop,
         const JSONLocation&,
         const RenderPropertyValue& default_value) {
        prop->initializeValue(default_value);
      };

  // Error callback for required properties that are not in the JSON
  JSONParseEmptyCBFunc required_prop_initializer = [this](BaseRenderProperty* prop,
                                                          const JSONLocation& prop_loc,
                                                          const RenderPropertyValue&) {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        prop_loc,
        "\"" + prop->getName() + "\" mark property must exist for \"" +
            makeLowerCase(to_string(mark_facade_.getType())) + "\" marks."));
  };

  //
  // Funcs to use for initializing properties based on RenderProperty type
  // pass correct empty callback to either error or use default value
  //
  // TODO(scb): Consolidate these methods once bespoke RenderProperties are in place
  auto init_float_prop = [&](auto prop_id, auto is_required, auto const& default_value) {
    initPropFromJSONObj(data_changed,
                        prop_loc,
                        getProperty(prop_id),
                        default_value,
                        obj_num_check,
                        is_required ? required_prop_initializer : default_initializer);
  };

  auto init_color_prop = [&](auto prop_id, auto is_required, auto const& default_value) {
    auto* prop = getProperty(prop_id);
    auto* color_prop = static_cast<ColorRenderProperty*>(prop);
    auto prev_color_type = color_prop->getColorType();
    initPropFromJSONObj(data_changed,
                        prop_loc,
                        prop,
                        default_value,
                        obj_color_check,
                        is_required ? required_prop_initializer : default_initializer);
    return color_prop->getColorType() != prev_color_type;
  };

  auto init_enum_prop = [&](auto prop_id, auto is_required, auto const& default_value) {
    initPropFromJSONObj(data_changed,
                        prop_loc,
                        getProperty(prop_id),
                        default_value,
                        obj_enum_check,
                        is_required ? required_prop_initializer : default_initializer);
  };

  // p1 = (e.g. x)
  // p2 = (e.g. x2)
  // pc = (e.g. xc)
  auto init_extended_coord_props = [&, this](auto p1_id, auto p2_id, auto pc_id) {
    auto* p1 = getProperty(p1_id);
    auto* p2 = getProperty(p2_id);
    auto* pc = getProperty(pc_id);

    auto p1_valid = prop_loc.getMember(p1->getName()).isValid();
    auto p2_valid = prop_loc.getMember(p2->getName()).isValid();
    auto pc_valid = prop_loc.getMember(pc->getName()).isValid();

    // Clear properties that have been removed from the JSON
    if (!p1_valid) {
      p1->clear();
    }
    if (!p2_valid) {
      p2->clear();
    }
    if (!pc_valid) {
      pc->clear();
    }

    // Check if p1 or p2 are valid first before checking pc
    if (p1_valid || p2_valid) {
      if (p1_valid) {
        initPropFromJSONObj(data_changed, prop_loc, p1, 0.0f, obj_num_check, nullptr);
      }
      if (p2_valid) {
        initPropFromJSONObj(data_changed, prop_loc, p2, 0.0f, obj_num_check, nullptr);
      }
    } else if (pc_valid) {
      initPropFromJSONObj(
          data_changed, prop_loc, pc, 0.0f, obj_num_check, required_prop_initializer);
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          prop_loc,
          "An \"" + p1->getName() + "\", \"" + p2->getName() + "\", or \"" +
              pc->getName() + "\" mark property must exist"));
    }
  };

  //
  // Initialize properties
  //

  // Common properties (TODO: use RenderPropertyCreateInfos)
  using type = RenderPropertyCreateInfo::Type;
  // Init coord props
  if (any_bits_set(prop_types_ & type::kExtendedCoords)) {
    init_extended_coord_props(PropId::kX, PropId::kX2, PropId::kXC);
    init_extended_coord_props(PropId::kY, PropId::kY2, PropId::kYC);
  } else {
    init_float_prop(PropId::kX, true, RenderPropertyValue());
    init_float_prop(PropId::kY, true, RenderPropertyValue());
  }

  init_float_prop(PropId::kOpacity, false, RenderPropertyValue(1.0f));

  // Loop over props and update
  for (auto const& ci : create_infos_) {
    switch (ci.type) {
      case type::kNone:
        CHECK(false) << "Invalid RenderPropertyCreateInfo - must specify a type";
        break;
      case type::kExtendedCoords:
        // Do nothing (Mark implementation will handle it for now)
        break;
      case type::kFillColor:
        did_structure_change =
            init_color_prop(PropId::kFillColor, ci.is_required, ci.default_value) ||
            did_structure_change;
        break;
      case type::kFillOpacity:
        init_float_prop(PropId::kFillOpacity, ci.is_required, ci.default_value);
        break;
      case type::kStrokeColor:
        did_structure_change =
            init_color_prop(PropId::kStrokeColor, ci.is_required, ci.default_value) ||
            did_structure_change;
        break;
      case type::kStrokeOpacity:
        init_float_prop(PropId::kStrokeOpacity, ci.is_required, ci.default_value);
        break;
      case type::kStrokeWidth:
        init_float_prop(PropId::kStrokeWidth, ci.is_required, ci.default_value);
        break;
      case type::kLineJoinType:
        init_enum_prop(PropId::kLineJoinType, ci.is_required, ci.default_value);
        break;
      case type::kMiterLimit:
        init_float_prop(PropId::kMiterLimit, ci.is_required, ci.default_value);
        break;
    }
  }

  //
  // Set active flags for Fill and Stroke and validate opacities if accumulation active
  //
  auto validate_color_opacities = [this, &prop_loc](
                                      const BaseRenderProperty* color_prop,
                                      const BaseRenderProperty* color_opacity_prop) {
    // Validate that opacity is 1 if accumulation active for property
    // opacities must be a uniform set to 1.0 if accumulating
    // TODO(croot): use opacity as a weight when accumulating

    auto const* opacity = getProperty(PropId::kOpacity);
    if (color_prop->hasAccumulator()) {
      bool color_opacity_is_one =
          color_opacity_prop ? (!color_opacity_prop->isDataDriven() &&
                                color_opacity_prop->getUniformValue<float>() >= 1.0f)
                             : true;
      bool opacity_is_one =
          !opacity->isDataDriven() && opacity->getUniformValue<float>() >= 1.0f;

      RUNTIME_EX_ASSERT(
          opacity_is_one && color_opacity_is_one,
          RapidJSONUtils::createJsonParseError(
              prop_loc,
              "Color property \"" + color_prop->getName() +
                  "\" is referencing an accumulator scale. The \"" + opacity->getName() +
                  (color_opacity_prop ? "\" and \"" + color_opacity_prop->getName()
                                      : "") +
                  "\" properties need to be explicitly set to 1 (without a scale "
                  "reference) or removed "
                  "from the vega. Opacities < 1 are currently unsupported with "
                  "accumulation rendering."));
    }

    //
    // return true if the color may have any opacity (is not transparent)
    // used to set 'active' flags for fill and stroke
    //
    bool opacity_active = true;
    if (color_opacity_prop) {
      opacity_active = color_opacity_prop->isDataDriven() ||
                       color_opacity_prop->getUniformValue<float>() > 0;
    }
    return opacity_active &&
           (color_prop->isDataDriven() ||
            color_prop->getUniformValue<gfx::ColorUnion>().opacity()) &&
           (opacity->isDataDriven() || opacity->getUniformValue<float>() > 0);
  };

  // Check fill color active
  if (any_bits_set(prop_types_ & type::kFillColor)) {
    auto prev_is_active = is_fill_active_;
    is_fill_active_ = validate_color_opacities(getProperty(PropId::kFillColor),
                                               getProperty(PropId::kFillOpacity));
    did_structure_change = did_structure_change || (prev_is_active != is_fill_active_);
  }

  // Check stroke color active
  if (any_bits_set(prop_types_ & type::kStrokeColor)) {
    auto prev_is_active = is_stroke_active_;
    is_stroke_active_ = validate_color_opacities(getProperty(PropId::kStrokeColor),
                                                 getProperty(PropId::kStrokeOpacity));
    // if color is visible, check if stroke width is > 0
    auto const* stroke_width = getProperty(PropId::kStrokeWidth);
    if (is_stroke_active_ && stroke_width) {
      is_stroke_active_ =
          (stroke_width->isDataDriven() || stroke_width->getUniformValue<float>() > 0);
    }
    did_structure_change = did_structure_change || (prev_is_active != is_stroke_active_);
  }

  // Notify material if it needs to rebuild
  if (did_structure_change) {
    mark_facade_.notifyChanged(BaseRenderProperty::ChangeType::kStructure);
  }
}

bool RenderPropertyContainer::isFillActive() const {
  return is_fill_active_;
}

bool RenderPropertyContainer::isStrokeActive() const {
  return is_stroke_active_;
}

BaseRenderProperty* RenderPropertyContainer::getProperty(PropId id) const {
  if (properties_map_.count(id)) {
    return properties_map_.at(id).get();
  }
  return nullptr;
}

const std::vector<BaseRenderProperty*>& RenderPropertyContainer::getProperties() const {
  return properties_;
}

const std::vector<BaseRenderProperty*>& RenderPropertyContainer::getCoordProperties()
    const {
  return coord_properties_;
}

std::pair<bool, bool> RenderPropertyContainer::initIds(const bool data_changed,
                                                       BaseDataTableShPtr data) {
  RENDER_LOG_SCOPE();
  bool ids_changed = data_changed;
  auto const prev_size = ids.size();
  static constexpr RenderPropertyFlagBits id_prop_flag_bits{
      RenderPropertyFlagBits::kFlexibleType |
      RenderPropertyFlagBits::kResetOnEmptyDataUpdate};

  if (ctx_.doHitTest()) {
    const RowIdHitTestOffsetData* rowid_offsets = nullptr;
    bool rowid_offsets_found = false;
    bool is_hittesting_enabled = true;

    if (!ids_changed) {
      if (!data) {
        // This mark isn't referencing any data tables, so there's no rowids to be found.
        ids.clear();
      } else {
        auto sql_data_table = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data);
        is_hittesting_enabled = sql_data_table ? sql_data_table->getQuerySQL()
                                                     .getRenderQueryOptions()
                                                     .isHitTestingEnabled()
                                               : true;
        rowid_offsets =
            sql_data_table ? sql_data_table->getQuerySQL().getRowIdOffsetData() : nullptr;
        rowid_offsets_found = true;

        if (rowid_offsets && rowid_offsets->rowid_cols.size() != ids.size()) {
          ids_changed = true;
        } else {
          // Double check existing ids for discrepencies between data references. All id
          // properties should reference the same data table.
          for (auto const& id : ids) {
            if (id->getDataTablePtr() != data) {
              ids_changed = true;
              break;
            }
          }
        }
      }
    }

    if (ids_changed) {
      // Set id properties according to all rowids columns from data table - named
      // "rowid", "rowid0", "rowid1".
      CHECK(data);

      if (!rowid_offsets_found) {
        auto sql_data_table = dynamic_cast<BaseQueryDataTableSQLJSON*>(data.get());
        is_hittesting_enabled = sql_data_table ? sql_data_table->getQuerySQL()
                                                     .getRenderQueryOptions()
                                                     .isHitTestingEnabled()
                                               : true;
        rowid_offsets =
            sql_data_table ? sql_data_table->getQuerySQL().getRowIdOffsetData() : nullptr;
      }
      if (rowid_offsets && rowid_offsets->rowid_cols.size()) {
        // NOTE: in the case that an aggregate query is performed, "rowid" is injected
        // into the rendered result set for hit-testing. In this case, the rowid_cols from
        // the offset data should be empty, hence the rowid_cols.size() check above.
        auto const& rowid_attrs = rowid_offsets->rowid_cols;
        auto const num_ids = rowid_attrs.size();

        if (num_ids != ids.size()) {
          // Destroy/create id properties where appropriate
          if (num_ids < ids.size()) {
            // delete unused id properties off the end of the vector
            ids.erase(ids.begin() + num_ids, ids.end());
          } else {
            auto start_idx = ids.size();
            if (!start_idx) {
              // The first id prop must be named "id" to align with shaders.
              ids.push_back(
                  std::make_unique<RenderProperty<uint64_t>>("id",  // property name
                                                             ctx_,  // context
                                                             mark_facade_,
                                                             id_prop_flag_bits));
              start_idx++;
            }
            // Build other ids, with the name id[1-2]
            for (auto i = start_idx; i < num_ids; ++i) {
              ids.push_back(std::make_unique<RenderProperty<uint64_t>>(
                  "id" + std::to_string(i), ctx_, mark_facade_, id_prop_flag_bits));
            }
          }
        }

        for (size_t i = 0; i < num_ids; ++i) {
          ids[i]->initializeFromData(rowid_attrs[i]->get_resname(), data);
        }
      } else if (is_hittesting_enabled) {
        // Look for a "rowid" column in the data.
        auto data_attrs = data->getAllAttrNames();
        bool found_row_id = false;
        for (auto const& attr : data_attrs) {
          if (attr == kDefaultIdColumnName) {
            found_row_id = true;
            break;
          }
        }

        if (found_row_id) {
          if (ids.size() == 0) {
            ids.push_back(std::make_unique<RenderProperty<uint64_t>>(
                "id", ctx_, mark_facade_, id_prop_flag_bits));
          } else {
            ids.erase(ids.begin() + 1, ids.end());
          }

          ids[0]->initializeFromData(kDefaultIdColumnName, data);
        } else {
          ids.clear();
        }
      } else {
        ids.clear();
      }
    }
  } else {
    // Hit-testing is not configured for this user
    ids.clear();
  }

  if (ids.size() != prev_size) {
    return {true, true};
  }
  if (ids_changed) {
    return {false, true};
  }
  return {false, false};
}

std::ostream& operator<<(std::ostream& os, RenderPropertyCreateInfo::Type property_type) {
  using type = RenderPropertyCreateInfo::Type;
  switch (property_type) {
    case type::kNone:
      os << "kNone";
      break;
    case type::kExtendedCoords:
      os << "kExtendedCoords";
      break;
    case type::kFillColor:
      os << "kFillColor";
      break;
    case type::kFillOpacity:
      os << "kFillOpacity";
      break;
    case type::kStrokeColor:
      os << "kStrokeColor";
      break;
    case type::kStrokeOpacity:
      os << "kStrokeOpacity";
      break;
    case type::kStrokeWidth:
      os << "kStrokeWidth";
      break;
    case type::kLineJoinType:
      os << "kLineJoinType";
      break;
    case type::kMiterLimit:
      os << "kMiterLimit";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, RenderPropertyContainer::PropId prop_id) {
  using PropId = RenderPropertyContainer::PropId;
  switch (prop_id) {
    case PropId::kX:
      os << "kX";
      break;
    case PropId::kY:
      os << "kY";
      break;
    case PropId::kX2:
      os << "kX2";
      break;
    case PropId::kXC:
      os << "kXC";
      break;
    case PropId::kY2:
      os << "kY2";
      break;
    case PropId::kYC:
      os << "kYC";
      break;
    case PropId::kOpacity:
      os << "kOpacity";
      break;
    case PropId::kFillColor:
      os << "kFillColor";
      break;
    case PropId::kFillOpacity:
      os << "kFillOpacity";
      break;
    case PropId::kStrokeColor:
      os << "kFillColor";
      break;
    case PropId::kStrokeOpacity:
      os << "kFillOpacity";
      break;
    case PropId::kStrokeWidth:
      os << "kStrokeWidth";
      break;
    case PropId::kLineJoinType:
      os << "kLineJoinType";
      break;
    case PropId::kMiterLimit:
      os << "kMiterLimit";
      break;

    default:
      UNREACHABLE();
  }
  return os;
}

}  // namespace QueryRenderer
