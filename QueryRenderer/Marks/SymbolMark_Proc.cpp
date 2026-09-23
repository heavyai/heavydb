/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/SymbolMark_Proc.h"

#include <regex>

#include <boost/algorithm/string/find.hpp>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/ShaderCompiler/GlslStructBuilder.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/GlobalRenderContext.h"
#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderPropertyContainer.h"
#include "QueryRenderer/Marks/SymbolDefinitions.h"
#include "QueryRenderer/Marks/Utils.h"
#include "Shared/scope.h"

#define PROFILE_MESH_DRAW 0
#if PROFILE_MESH_DRAW
#include "Shared/measure.h"
#endif

namespace QueryRenderer {

using ::gfx::ShaderManager;
using ::gfx::ShaderStage;
using ShaderBuilder = ::gfx::ShaderManager::Builder;

static SymbolDefinitions g_symbol_defs = {};

// This determines if the shape is constant or scale mapped
bool isEnumExplicitlyDefined(const EnumRenderProperty& enum_prop, int gpu_id = -1) {
  if (gpu_id >= 0) {
    return !enum_prop.usesScaleConfig() && !enum_prop.hasVboPtr(gpu_id) &&
           !enum_prop.hasSsboPtr(gpu_id);
  } else {
    return !enum_prop.usesScaleConfig() && !enum_prop.hasVboPtr() &&
           !enum_prop.hasSsboPtr();
  }
}

SymbolMark_Proc::CoordDimensionTypes getPosAndDimensionTypes(
    const rapidjson::Pointer& x_json_path,
    const rapidjson::Pointer& x2_json_path,
    const rapidjson::Pointer& xc_json_path,
    const rapidjson::Pointer& y_json_path,
    const rapidjson::Pointer& y2_json_path,
    const rapidjson::Pointer& yc_json_path) {
  SymbolMark_Proc::CoordinateType x(SymbolMark_Proc::CoordinateType::kPrimary),
      y(SymbolMark_Proc::CoordinateType::kPrimary);
  SymbolMark_Proc::DimensionType width(SymbolMark_Proc::DimensionType::kValue),
      height(SymbolMark_Proc::DimensionType::kValue);

  if (!RapidJSONUtils::isValidPath(x_json_path)) {
    if (RapidJSONUtils::isValidPath(x2_json_path)) {
      x = SymbolMark_Proc::CoordinateType::kSecondary;
    } else if (RapidJSONUtils::isValidPath(xc_json_path)) {
      x = SymbolMark_Proc::CoordinateType::kCenter;
    }
  } else if (RapidJSONUtils::isValidPath(x2_json_path)) {
    width = SymbolMark_Proc::DimensionType::kCoords;
  }

  if (!RapidJSONUtils::isValidPath(y_json_path)) {
    if (RapidJSONUtils::isValidPath(y2_json_path)) {
      y = SymbolMark_Proc::CoordinateType::kSecondary;
    } else if (RapidJSONUtils::isValidPath(yc_json_path)) {
      y = SymbolMark_Proc::CoordinateType::kCenter;
    }
  } else if (RapidJSONUtils::isValidPath(y2_json_path)) {
    height = SymbolMark_Proc::DimensionType::kCoords;
  }

  return {x, y, width, height};
}

const BaseRenderProperty* getPropFromCoordType(
    const SymbolMark_Proc::CoordinateType coord_type,
    const BaseRenderProperty& primary,
    const BaseRenderProperty& secondary,
    const BaseRenderProperty& center) {
  switch (coord_type) {
    case SymbolMark_Proc::CoordinateType::kPrimary:
      return &primary;
    case SymbolMark_Proc::CoordinateType::kSecondary:
      return &secondary;
    case SymbolMark_Proc::CoordinateType::kCenter:
      return &center;
  }
  CHECK(false);
  return nullptr;
}

SymbolMark_Proc::SymbolMark_Proc(const JSONLocation& obj_loc, QueryRendererContext& ctx)
    : BaseMark(GeomType::kSymbols, ctx, obj_loc, DataOutputFormat::kRows, false)
    , shape_("shape",
             QueryDataType::SYMBOL_SHAPE_ENUM,
             ctx,
             *prop_mark_facade_,
             static_cast<RenderPropertyFlagBits>(RenderPropertyFlagBits::kUseScale |
                                                 RenderPropertyFlagBits::kFlexibleType),
             convertStringToSymbolShapeEnum)
    , width_("width", ctx, *prop_mark_facade_)
    , height_("height", ctx, *prop_mark_facade_)
    , angle_("angle", ctx, *prop_mark_facade_)
    , angle_unit_("angleUnit",
                  QueryDataType::ANGLE_UNIT_ENUM,
                  ctx,
                  *prop_mark_facade_,
                  RenderPropertyFlagBits::kFlexibleType,
                  convertStringToAngleUnitEnum)
    , pos_dim_types_({CoordinateType::kPrimary,
                      CoordinateType::kPrimary,
                      DimensionType::kValue,
                      DimensionType::kValue})
    , is_binned_heatmap_{false} {
  using type = RenderPropertyCreateInfo::Type;
  using flag = RenderPropertyFlagBits;
  std::vector<RenderPropertyCreateInfo> render_property_ci = {
      {type::kExtendedCoords},
      {type::kFillColor, flag::kUnspecified, false, gfx::ColorUnion(0.f, 0.f, 0.f, 1.f)},
      {type::kFillOpacity, flag::kUnspecified, false, 1.0f},
      {type::kStrokeColor, flag::kUseScale, false, gfx::ColorUnion(1.f, 1.f, 1.f, 1.f)},
      {type::kStrokeOpacity, flag::kUnspecified, false, 1.0f},
      {type::kStrokeWidth, flag::kUnspecified, false, 0.0f}};

  render_props_ = std::make_unique<RenderPropertyContainer>(
      ctx, *prop_mark_facade_, std::move(render_property_ci));

  auto const& coord_props = render_props_->getCoordProperties();
  projection_policy_ = std::make_unique<MarkProjectionShaderPolicy>(
      MarkProjectionShaderPolicy::PropMap{coord_props.begin(), coord_props.end()});

  initPropertiesFromJSONObj(obj_loc, true, true);
  initTransformsFromJSONObj(obj_loc, getCoordPropAttrInfos());
  json_path_ = obj_loc.getPathRef();
  init_symbol_defs(g_symbol_defs);
}

SymbolMark_Proc::~SymbolMark_Proc() {}

BaseRenderPropertyConstSet SymbolMark_Proc::getUsedProps() const {
  return used_props_const_;
}

void SymbolMark_Proc::initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                                const bool data_changed,
                                                const bool init) {
  RENDER_LOG_SCOPE() << " data_changed: " << data_changed << "  init: " << init;
  auto const prop_loc = obj_loc.getMember(JSONSchema_v1::Marks::kPropertiesProp);
  RUNTIME_EX_ASSERT(prop_loc.isValid(),
                    RapidJSONUtils::createJsonParseError(
                        obj_loc, "Mark objects must have a \"properties\" property."));

  auto const prev_path = properties_json_path_;
  properties_json_path_ = obj_loc.getPathRef();
  if (!ctx_.isJSONCacheUpToDate(prev_path, prop_loc) || data_changed || init) {
    RUNTIME_EX_ASSERT(prop_loc.isObject(),
                      RapidJSONUtils::createJsonParseError(
                          prop_loc, "Property must be a json object."));

    render_props_->initFromJSONObj(obj_loc, data_changed);

    auto const obj_num_check = BaseMark::validateNumPropFunc(*this);
    auto const obj_enum_check = BaseMark::validateEnumPropFunc(*this);
    auto const init_prop_func = [&](const JSONLocation& prop_loc,
                                    BaseRenderProperty* prop,
                                    ValidateFunc validate_type_func = nullptr,
                                    JSONParseCBFunc post_empty_func = nullptr) {
      return BaseMark::initPropFromJSONObj(&ctx_,
                                           data_,
                                           data_changed,
                                           prop_loc,
                                           prop,
                                           properties_json_path_,
                                           validate_type_func,
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           post_empty_func);
    };

    auto const pre_is_static_shape = isEnumExplicitlyDefined(shape_);
    auto pre_shape = static_cast<int>(SymbolShapeType::kCircle);
    if (pre_is_static_shape) {
      pre_shape = shape_.getUniformValue<int>();
    }

    init_prop_func(
        prop_loc,
        &shape_,
        // ValidateFunc validate_type_func
        [](const std::string& prop_name, const JSONLocation& shape_loc) {
          RUNTIME_EX_ASSERT(shape_loc.isObject() || shape_loc.isInt() ||
                                shape_loc.isUint64() || shape_loc.isString(),
                            RapidJSONUtils::createJsonParseError(
                                shape_loc,
                                "\"" + prop_name +
                                    "\" symbol mark property must be a "
                                    "scale/data references or an enum "
                                    "value (i.e. an int or a string)"));
        },
        [this](const JSONLocation&) {
          // set default shape to "square"
          shape_.initializeValue(static_cast<int>(SymbolShapeType::kSquare));
        });

    auto const is_curr_static_shape = isEnumExplicitlyDefined(shape_);
    if (pre_is_static_shape != is_curr_static_shape) {
      setShaderDirty();
    } else if (is_curr_static_shape) {
      auto const curr_shape = shape_.getUniformValue<int>();
      auto const circle_val = static_cast<int>(SymbolShapeType::kCircle);
      if (curr_shape != pre_shape &&
          (curr_shape == circle_val || pre_shape == circle_val)) {
        setShaderDirty();
      }
    }
    using PropId = RenderPropertyContainer::PropId;
    auto* x_prop = render_props_->getProperty(PropId::kX);
    auto* x2_prop = render_props_->getProperty(PropId::kX2);
    auto* xc_prop = render_props_->getProperty(PropId::kXC);
    auto* y_prop = render_props_->getProperty(PropId::kY);
    auto* y2_prop = render_props_->getProperty(PropId::kY2);
    auto* yc_prop = render_props_->getProperty(PropId::kYC);

    if (!RapidJSONUtils::isValidPath(x_prop->getJsonPath()) ||
        !RapidJSONUtils::isValidPath(x2_prop->getJsonPath())) {
      init_prop_func(prop_loc, &width_, obj_num_check, [&, this](const JSONLocation&) {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
            prop_loc,
            "A \"" + width_.getName() +
                "\" mark property must exist for symbol marks if either the \"" +
                x_prop->getName() + "\" or \"" + x2_prop->getName() +
                "\" aren't defined."));
      });
    } else {
      width_.clear();
    }

    if (!RapidJSONUtils::isValidPath(y_prop->getJsonPath()) ||
        !RapidJSONUtils::isValidPath(y2_prop->getJsonPath())) {
      init_prop_func(prop_loc, &height_, obj_num_check, [&, this](const JSONLocation&) {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
            prop_loc,
            "A \"" + height_.getName() +
                "\" mark property must exist for symbol marks if either the \"" +
                y_prop->getName() + "\" or \"" + y2_prop->getName() +
                "\" aren't defined."));
      });
    } else {
      height_.clear();
    }

    bool prev_is_angle_enabled = doAngle();
    init_prop_func(prop_loc, &angle_, obj_num_check, [this](const JSONLocation&) {
      angle_.initializeValue(0.0f);
    });

    bool do_angle = doAngle();
    if (do_angle != prev_is_angle_enabled) {
      setShaderDirty();
    }

    if (do_angle) {
      init_prop_func(prop_loc, &angle_unit_, obj_enum_check, [this](const JSONLocation&) {
        angle_unit_.initializeValue(static_cast<int>(AngleUnit::kDegrees));
      });
    }
    auto pos_dim_types = getPosAndDimensionTypes(x_prop->getJsonPath(),
                                                 x2_prop->getJsonPath(),
                                                 xc_prop->getJsonPath(),
                                                 y_prop->getJsonPath(),
                                                 y2_prop->getJsonPath(),
                                                 yc_prop->getJsonPath());

    if (pos_dim_types != pos_dim_types_) {
      pos_dim_types_ = pos_dim_types;
      setShaderDirty();
    }

    initIds(data_changed);

    auto const& props = render_props_->getProperties();
    BaseRenderPropertySet used_props{props.begin(), props.end()};
    used_props.insert(&shape_);
    used_props.insert(&width_);
    used_props.insert(&height_);

    // we use the mesh shader path for MULTIPOINT, and the vert shader path for POINT
    // for now we need to do all this here too, so that useMeshShader() is valid
    // do it after we've processed all the coord props, obviously
    // @TODO refactor this so we don't have to do it ALL in three different places
    updateGeoPropInfoAndPropCompressionBits({kPOINT, kMULTIPOINT});
    updateUseMeshShader({kMULTIPOINT});

    if (useMeshShader() || do_angle) {
      used_props.insert(&angle_);
      used_props.insert(&angle_unit_);
    }

    bool used_props_changed = used_props_ != used_props;
    if (used_props_changed) {
      used_props_ = std::move(used_props);
      used_props_const_.clear();
      used_props_const_.insert(used_props_.cbegin(), used_props_.cend());
    }

    if (init || data_changed || used_props_changed) {
      prop_buf_loc_state_.clear();
      updateProps(getUsedProps());
    }

    updateVisibility(render_props_->isFillActive() || render_props_->isStrokeActive());
  }
}

bool SymbolMark_Proc::doAngle() {
  return (angle_.isDataDriven() || angle_.getUniformValue<float>() != 0);
}

void SymbolMark_Proc::updateIsBinnedHeatmap() {
  // Check if we're doing a binned heatmap query. We need to check for the rect_pixel_bin
  // and hex_pixel_bin functions in the query and verify that the width matches the mark
  // width. In these cases we need to ensure that mark edges align and change blending
  // mode to Max
  is_binned_heatmap_ = false;  // Assume false to allow early outs when possible
  if (width_.isDataDriven() || height_.isDataDriven()) {
    return;
  }

  auto sql_data_table = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(data_);
  if (sql_data_table) {
    std::string sql_str = sql_data_table->getQuerySQL().getSqlQueryStr();
    // check if we're doing any type of pixel bin query
    std::regex pixel_bin_regex(
        "(?:rect_|reg_hex_(?:horiz_|vert_))?pixel_bin(?:_x|_y|_packed)?");
    if (std::regex_search(sql_str, pixel_bin_regex)) {
      // verify that width in query matches mark width
      std::regex val_regex("(?:\\d|\\.)+");
      std::smatch fn_match;
      std::smatch val_match;
      std::string search_str = sql_str;
      // hex bin (any)
      std::regex fn_regex(
          "reg_hex_(?:horiz_|vert_)pixel_bin(?:_x|_y|_packed)\\s*\\({1}(.*?,){6}");
      while (regex_search(search_str, fn_match, fn_regex)) {
        search_str = fn_match.suffix();
        if (regex_search(search_str, val_match, val_regex)) {
          if (std::stof(val_match[0]) != width_.getUniformValue<float>()) {
            return;
          }
        }
        search_str = val_match.suffix();
        if (regex_search(search_str, val_match, val_regex)) {
          if (std::stof(val_match[0]) != height_.getUniformValue<float>()) {
            return;
          }
        }
      }
      // rect bin packed
      search_str = sql_str;
      fn_regex = "rect_pixel_bin_packed\\s*\\({1}(.*?,){6}";
      while (regex_search(search_str, fn_match, fn_regex)) {
        search_str = fn_match.suffix();
        if (regex_search(search_str, val_match, val_regex)) {
          if (std::stof(val_match[0]) != width_.getUniformValue<float>()) {
            return;
          }
        }
        search_str = val_match.suffix();
        if (regex_search(search_str, val_match, val_regex)) {
          if (std::stof(val_match[0]) != height_.getUniformValue<float>()) {
            return;
          }
        }
      }
      // rect bin x
      search_str = sql_str;
      fn_regex = "rect_pixel_bin_x\\s*\\({1}(.*?,){3}";
      while (regex_search(search_str, fn_match, fn_regex)) {
        search_str = fn_match.suffix();
        if (regex_search(search_str, val_match, val_regex)) {
          if (std::stof(val_match[0]) != width_.getUniformValue<float>()) {
            return;
          }
        }
      }
      // rect bin y
      search_str = sql_str;
      fn_regex = "rect_pixel_bin_y\\s*\\({1}(.*?,){3}";
      while (regex_search(search_str, fn_match, fn_regex)) {
        search_str = fn_match.suffix();
        if (regex_search(search_str, val_match, val_regex)) {
          if (std::stof(val_match[0]) != height_.getUniformValue<float>()) {
            return;
          }
        }
      }
      // Doing a standard uni-variate pixel bin query
      is_binned_heatmap_ = true;
    }
  }
}

void SymbolMark_Proc::updateShader() {
  RENDER_LOG_SCOPE() << "building glsl shaders";

  updateIsBinnedHeatmap();

  auto& shader_mgr = ctx_.getShaderManager();
  auto do_angle = doAngle();
  auto has_accumulator = hasAccumulator();

  // we use the mesh shader path for MULTIPOINT, and the vert shader path for POINT
  updateGeoPropInfoAndPropCompressionBits({kPOINT, kMULTIPOINT});
  updateUseMeshShader({kMULTIPOINT});

  // Create builders and replace tags
  ShaderManager::BuilderUqPtrVector builders;
  if (useMeshShader()) {
    builders = shader_mgr.createBuilderVector(
        {{"Marks/fastSymbolTemplate.mesh"}, {"Marks/fastSymbolTemplate.frag"}});
  } else if (do_angle) {
    builders = shader_mgr.createBuilderVector({{"Marks/fastSymbolTemplate_passthru.vert"},
                                               {"Marks/fastSymbolTemplate.frag"},
                                               {"Marks/fastSymbolTemplate.geom"}});
  } else {
    builders = shader_mgr.createBuilderVector(
        {{"Marks/fastSymbolTemplate.vert"}, {"Marks/fastSymbolTemplate.frag"}});
  }

  std::stringstream get_prop_ss;
  streamPropertyGetters(prop_buf_loc_state_.vbo_props, get_prop_ss);
  streamPropertyGetters(prop_buf_loc_state_.uniform_props, get_prop_ss);

  // Build vertex shader uniform render props
  gfx::GlslStructBuilder ubo_struct_builder("FAST_SYMBOL_VERT_UBO_TYPE");
  if (!useMeshShader() && !do_angle) {
    ubo_struct_builder.addMember("uVPmatrix", gfx::BufferAttrType::kMat3x2f);
    ubo_struct_builder.addMember("uPivotx", gfx::BufferAttrType::kFloat);
    ubo_struct_builder.addMember("uPivoty", gfx::BufferAttrType::kFloat);
  }
  ubo_struct_builder.addMember("invalidKey", gfx::BufferAttrType::kUint64);
  ubo_struct_builder.addMember("propCompressionBits", gfx::BufferAttrType::kUint);
  if (useMeshShader()) {
    ubo_struct_builder.addMember("vboDeviceAddress", gfx::BufferAttrType::kUint64);
  }

  addCommonRenderPropUniforms(ubo_struct_builder);

  // Build fragment shader input interface block
  // Also used for vertex output in point mode
  gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
  std::optional<gfx::GlslStructBuilder> geometry_inputs =
      do_angle ? std::optional<gfx::GlslStructBuilder>{"GeometryShaderInputs"}
               : std::nullopt;

  generate_fast_symbol_interface_blocks(fragment_inputs,
                                        geometry_inputs,
                                        prop_buf_loc_state_.uniform_props.count(&angle_),
                                        has_accumulator,
                                        useMeshShader());

  auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
  builders[1]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

  if (!useMeshShader() && do_angle) {
    builders[0]->replaceFirstTag("GeometryShaderInputs",
                                 geometry_inputs->createInterfaceBlockString(true));
    builders[2]->replaceFirstTag(
        "GeometryShaderInputs",
        geometry_inputs->createInterfaceBlockString(true, std::nullopt, true));
    builders[2]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
  } else {
    builders[0]->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
  }

  // Vertex/mesh shader render prop inputs
  builders[0]->replaceFirstTag(
      "VertexProperties",
      useMeshShader() ? buildVertexDataStruct() : buildVertexShaderInputs());
  builders[0]->replaceFirstTag("UniformProperties",
                               ubo_struct_builder.createStructString());
  builders[0]->replaceFirstTag("PropertyGetters", get_prop_ss.str());

  builders[0]->setExternalUniformBuffers({"SLAB_ADDRESS_TABLE_UBO"});

  builders[0]->replaceFirstTag("computeX",
                               std::to_string(static_cast<int>(pos_dim_types_.x_type)));
  builders[0]->replaceFirstTag("computeY",
                               std::to_string(static_cast<int>(pos_dim_types_.y_type)));
  builders[0]->replaceFirstTag(
      "computeWidth", std::to_string(static_cast<int>(pos_dim_types_.width_type)));
  builders[0]->replaceFirstTag(
      "computeHeight", std::to_string(static_cast<int>(pos_dim_types_.height_type)));

  BaseMark::setKeyInShaderBuilder(*builders[0]);

  // Set symbol definition vertex count
  builders[1]->replaceFirstTag("numSymbolVerts",
                               std::to_string(g_symbol_defs.verts.size()));
  // Set multisampling flag
  // controls use of gl_SamplePosition
  // creates a no-discard zone of 1 pixel from the edge to ensure no contributing samples
  // are also discarded
  builders[1]->replaceFirstTag("isMultiSampling",
                               std::to_string(needsMultisampleEnabled()));

  // Fragment shader needs to use fUV instead of gl_PointCoord when using geometry
  // shader billboards
  const bool using_geom_or_mesh_shader = useMeshShader() || do_angle;
  builders[1]->replaceFirstTag("usingGeomOrMeshShader",
                               std::to_string(using_geom_or_mesh_shader));

  // Vertex / geometry shaders need to expand billboard by 0.1 pixels if rendering a
  // heatmap in order to avoid edge artifacts. This grows the symbol as well, it is
  // not inverse scaled
  builders[0]->replaceFirstTag("doHeatmapEdgePad", std::to_string(is_binned_heatmap_));

  // Insert geometry shader defines
  auto has_fill_accumulator =
      render_props_->getProperty(RenderPropertyContainer::PropId::kFillColor)
          ->hasAccumulator();
  if (!useMeshShader() && do_angle) {
    // Insert accumIdx passthrough in the geometry shader for accumulation support
    builders[2]->replaceFirstTag("doAccumIndex", std::to_string(has_fill_accumulator));
    // Heatmap pad
    builders[2]->replaceFirstTag("doHeatmapEdgePad", std::to_string(is_binned_heatmap_));
    // The geometry shader needs the useUangle flag, but no other defines, so use tag
    // replacement rather than injecting the whole property typeinfo string
    builders[2]->replaceFirstTag(
        "useUangle", prop_buf_loc_state_.uniform_props.count(&angle_) == 0 ? "0" : "1");
  }

  if (useMeshShader()) {
    // subgroup size
    auto const subgroup_size =
        ctx_.getGlobalContext().getGfxContext().getDeviceLimits().subgroup_size;
    auto const subgroup_size_bits = uint32_t(log2(subgroup_size));
    builders[0]->replaceFirstTag("workgroupSize", std::to_string(subgroup_size));
    builders[0]->replaceFirstTag("workgroupSizeBits", std::to_string(subgroup_size_bits));

    // vertex attribute fetches
    // WIP
    auto attr_fetch_str = buildVertexAttributeFetches();
    builders[0]->replaceFirstTag("VertexAttributeFetches", attr_fetch_str);
  }

  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      scale_ref->buildSubroutineBindings(*builders[0], "_" + prop->getName());
      scale_ref->buildSubroutineBindings(*builders[1], "_" + prop->getName());
      if (!useMeshShader() && do_angle) {
        scale_ref->buildSubroutineBindings(*builders[2], "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      scale_ref->buildSubroutineBindings(*builders[0], "_" + prop->getName());
      scale_ref->buildSubroutineBindings(*builders[1], "_" + prop->getName());
      if (!useMeshShader() && do_angle) {
        scale_ref->buildSubroutineBindings(*builders[2], "_" + prop->getName());
      }
    }
  }

  BaseMark::insertPropertyCodeInShaderBuilders(
      builders, used_props_const_, *projection_policy_);

  // color props in either vertex or geometry shader
  BaseMark::setColorConvertSubroutines(
      *builders[0],
      render_props_->getProperty(RenderPropertyContainer::PropId::kFillColor));
  BaseMark::setColorConvertSubroutines(
      *builders[0],
      render_props_->getProperty(RenderPropertyContainer::PropId::kStrokeColor));

  // TODO(scb): add special handlers for stroke width < 1.0 which needs slightly
  // different antialiasing (stroke never fully covers pixel)
  bool do_fill = render_props_->isFillActive();
  bool do_stroke = render_props_->isStrokeActive();

  auto& frag_builder = *builders[1];
  if (has_accumulator) {
    if (do_fill && do_stroke) {
      frag_builder.addSubroutineBinding(
          "maybeDiscardFunc", "maybeDiscardFillAndStroke", true);
    } else if (do_fill) {
      frag_builder.addSubroutineBinding("maybeDiscardFunc", "maybeDiscardFill", true);
    } else if (do_stroke) {
      frag_builder.addSubroutineBinding("maybeDiscardFunc", "maybeDiscardStroke", true);
    }
  } else {
    if (do_fill && do_stroke) {
      frag_builder.addSubroutineBinding(
          "mapDistanceToColor", "mapDistanceToColorFillAndStroke", true);
    } else if (do_fill) {
      frag_builder.addSubroutineBinding(
          "mapDistanceToColor", "mapDistanceToColorFill", true);
    } else if (do_stroke) {
      frag_builder.addSubroutineBinding(
          "mapDistanceToColor", "mapDistanceToColorStroke", true);
    }
  }

  ctx_.clearMarkShaders(*this);
  ctx_.buildMarkShaders(
      *this, MarkGpuResourceSlot::kFill, "SymbolMark Fill", std::move(builders));
  shader_dirty_ = false;

  // set the props dirty to force a rebind with the new shader
  setPropsDirty();
}

void SymbolMark_Proc::buildPipelineDescriptors() {
  if (!pipeline_descriptor_) {
    pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
    CHECK(pipeline_descriptor_);
  }

  pipeline_descriptor_->setRasterSampleCount(getRasterizationSampleCount());

  // TODO(scb): this is a fix for visible symbol boundaries when rendering binned queries
  // (for heatmaps). We switch to using kMax for the blend equation, which prevents the
  // alpha channel from becoming too opaque and the color channel from becoming too bright
  // on the edges of symbols. This is actually only a problem when the symbol is not 100%
  // opaque OR if stroke is being used and not accounted for by the query parameters
  // (stroke grows the symbol beyond its width and height by a factor of strokeWidth*0.5)

  // NOTES:
  // 1/ In the case of bi- or multi-variate queries, where symbol size is also driven by a
  // data field, we need to use standard compositing, since visible overlap of the symbols
  // is expected, and should be captured properly. We also need to check if width and
  // height are constant values, but greater than the step factor specified in the
  // original query. This latter case would be very odd, but is also a case where symbol
  // overlap is expected.
  // 2/ TODO(scb): Having rendering method vary based on query type/parameters is quite
  // reasonable, but this feels very ad hoc. We should define a more explicit API for this
  // going forward.

  pipeline_descriptor_->setBlendEquation(is_binned_heatmap_ ? gfx::BlendEquation::kMax
                                                            : gfx::BlendEquation::kAdd);

  pipeline_descriptor_->getPushConstantRanges().clear();
  pipeline_descriptor_->getPushConstantRanges().insert(
      gfx::ShaderStageBits::kMesh, 0, sizeof(uint32_t));
}

void SymbolMark_Proc::buildPipelines(MarkPerGpuData& per_gpu_data) {
  const gfx::PrimitiveAssembly* primitive_assembly_to_use{};
  if (!useMeshShader()) {
    CHECK(per_gpu_data.fill_primitive_assemblies.size());
    CHECK(per_gpu_data.fill_primitive_assemblies[0]);
    primitive_assembly_to_use = per_gpu_data.fill_primitive_assemblies[0].get();
  }

  CHECK(per_gpu_data.fill_materials.size());
  CHECK(per_gpu_data.fill_materials[0]);

  per_gpu_data.destroyPipelines();
  per_gpu_data.graphics_pipelines.push_back(
      per_gpu_data.getResourceManager().createGraphicsPipeline(
          "SymbolMark Fill",
          *per_gpu_data.fill_materials[0],
          *pipeline_descriptor_,
          primitive_assembly_to_use));
  per_gpu_data.graphics_pipelines[0]->create(
      per_gpu_data.getRootPerGpuData().getCommonRenderPass(
          CommonRenderPassType::kAllAttachments, needsMultisampleEnabled()));
}

void SymbolMark_Proc::buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  if (!useMeshShader()) {
    auto gpu_id = gpu_data.getGpuId();
    CHECK(!gpu_data.fill_materials.empty());

    gfx::PrimitiveAssemblyAttrInfo attr_info;
    int attr_count = 0;
    int vbo_size = 0;
    int prop_size = 0;
    for (auto const* prop : prop_buf_loc_state_.vbo_props) {
      if (!gpu_data.fill_materials[0]->hasVertexAttribute(prop->getName())) {
        continue;
      }
      attr_count++;
      prop_size = prop->size(gpu_id);
      if (attr_count == 1) {
        vbo_size = prop_size;
      } else {
        RUNTIME_EX_ASSERT(prop_size == vbo_size,
                          std::string(*this) +
                              ": Invalid symbol mark. The sizes of the vertex buffer "
                              "attributes do not match for gpuId " +
                              std::to_string(gpu_id) + ". " + std::to_string(vbo_size) +
                              "!=" + std::to_string(prop_size));
      }
      prop->addToPrimitiveAssemblyAttrInfo(gpu_id, attr_info);
    }

    gpu_data.fill_primitive_assemblies.clear();
    gpu_data.fill_primitive_assemblies.push_back(
        gpu_data.getResourceManager().createPrimitiveAssembly(
            "SymbolMark_Proc",
            gfx::PrimitiveTopology::kPointList,
            *gpu_data.fill_materials[0],
            attr_info));
  }
}

void SymbolMark_Proc::setUniformAttributes(MarkPerGpuData& mark_gpu_data) {
  RENDER_LOG_SCOPE();
  auto& active_material = *mark_gpu_data.fill_materials[0];
  auto viewport_width = ctx_.getWidth();
  auto viewport_height = ctx_.getHeight();

  BaseMark::bindKeyPropUniformAttributes(active_material);

  if (hasProjection()) {
    active_material.setViewportAttributes(0, 0, viewport_width, viewport_height);
  }

  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      RENDER_LOG() << "binding VBO property: " << prop->getName();
      scale_ref->bindUniforms(active_material, "_" + prop->getName());
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    auto const& scale_ref = prop->getScaleReference();
    if (scale_ref != nullptr) {
      scale_ref->bindUniforms(active_material, "_" + prop->getName());
    }

    prop->setUniformAttribute(active_material, prop->getName());
  }

  for (auto const* prop : prop_buf_loc_state_.decimal_props) {
    prop->setDecimalScaleUniformAttribute(active_material);
  }

  BaseMark::bindIDPropUniformAttributes(active_material);
  BaseMark::setProjectionUniformAttributes(active_material);

  active_material.setUniformAttribute("uSymbolFlags", g_symbol_defs.flags);
  active_material.setUniformAttribute("uVertCounts", g_symbol_defs.segment_counts);
  active_material.setUniformAttribute("uVertOffsets", g_symbol_defs.vert_offsets);
  active_material.setUniformAttribute("uSymbolVerts", g_symbol_defs.verts);

  float pivotx = (pos_dim_types_.x_type == CoordinateType::kCenter ? 0.0f : 0.5f);
  float pivoty = (pos_dim_types_.y_type == CoordinateType::kCenter ? 0.0f : 0.5f);

  active_material.setUniformAttribute("uVPmatrix",
                                      ctx_.getViewProjMatrix().getDataArrayRef());
  active_material.setUniformAttribute("uPivotx", pivotx);
  active_material.setUniformAttribute("uPivoty", pivoty);

  if (useMeshShader() || doAngle()) {
    active_material.setUniformAttribute("uInvViewportWidth",
                                        1.0f / (float)viewport_width);
    active_material.setUniformAttribute("uInvViewportHeight",
                                        1.0f / (float)viewport_height);
    if (!angle_.isDataDriven()) {
      float angle = -angle_.getUniformValue<float>();
      if (angle_unit_.getUniformValue<int>() == static_cast<int>(AngleUnit::kDegrees)) {
        angle = angle * 3.14159265359f / 180.0f;
      }
      active_material.setUniformAttribute("uSinAngle", std::sin(angle));
      active_material.setUniformAttribute("uCosAngle", std::cos(angle));
    }
  }

  // update prop compression bits again for the case where only the compression changes
  updateGeoPropInfoAndPropCompressionBits({kPOINT, kMULTIPOINT});

  updateSlabAddressTableAndPropCompressionBitsUniforms(mark_gpu_data);
}

void SymbolMark_Proc::updateRenderPropertyGpuResources(
    const std::vector<GpuId>& add_gpus,
    const std::vector<GpuId>& remove_gpus) {
  for (auto const& prop : used_props_) {
    prop->initGpuResources(add_gpus, remove_gpus);
  }
}

bool SymbolMark_Proc::draw(const gfx::DeviceContext& device_ctx,
                           const MarkPerGpuData& mark_gpu_data,
                           gfx::Framebuffer& framebuffer,
                           const int accumulator_index) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());

  // NOTE: shader should have been updated before calling this
  CHECK(!mark_gpu_data.fill_materials.empty());

  auto& render_pass =
      selectDrawRenderPass(mark_gpu_data.getRootPerGpuData(), accumulator_index);

  if (useMeshShader()) {
    drawWithMeshShader(device_ctx, mark_gpu_data, framebuffer, render_pass);
  } else {
    drawWithVertexShader(device_ctx, mark_gpu_data, framebuffer, render_pass);
  }

  return true;
}

void SymbolMark_Proc::drawWithMeshShader(const gfx::DeviceContext& device_ctx,
                                         const MarkPerGpuData& mark_gpu_data,
                                         gfx::Framebuffer& framebuffer,
                                         gfx::RenderPass& render_pass) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());

#if PROFILE_MESH_DRAW
  LOG(INFO) << "DEBUG: starting mesh shader draw";
  auto total_timer = timer_start();
  auto start_mesh_draw_timer = timer_start();
#endif

  // start
  // this does the count pass on the recognized geo prop, builds the work
  // units buffer, and returns the number of work units (mesh shader warps)

  auto const num_work_units = startMeshShaderDraw(mark_gpu_data);
  if (num_work_units == 0U) {
    // nothing to draw
    return;
  }

  ScopeGuard end_mesh_shader_draw = [&]() { endMeshShaderDraw(mark_gpu_data); };

#if PROFILE_MESH_DRAW
  LOG(INFO) << "DEBUG: start mesh draw took " << timer_stop(start_mesh_draw_timer)
            << "ms, num_work_units = " << num_work_units;
#endif

  // finalize
  mark_gpu_data.fill_materials[0]->updateDescriptorSets();

  // draw in batches of max workgroup count / 4
  // this is an arbitrary safety factor, considering the ease of triggering
  // a DL during testing when approaching the full workgroup count
  auto const batch_workgroup_count =
      device_ctx.getLimits().max_mesh_workgroup_count[0] / 4;
  uint32_t first_work_unit = 0u;
  auto& command_list = device_ctx.getCommandList();
  command_list.pushLabel("Symbol_Proc Draw (Mesh Shader)")
      .beginRenderPass(render_pass, framebuffer);
  while (first_work_unit < num_work_units) {
    auto const group_count_x =
        std::min(batch_workgroup_count, num_work_units - first_work_unit);
    command_list
        .setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[0],
                               "firstWorkUnit",
                               gfx::ShaderStageBits::kMesh,
                               first_work_unit)
        .drawMeshTasks(*mark_gpu_data.graphics_pipelines[0], group_count_x, 1u, 1u);
    first_work_unit += group_count_x;
  }
  command_list.endRenderPass().popLabel().flush("Symbol_Proc Draw (Mesh Shader)");

  // can do more draw passes in here with the same work units buffer

#if PROFILE_MESH_DRAW
  LOG(INFO) << "DEBUG: ending mesh shader draw, took " << timer_stop(total_timer) << "ms";
#endif
}

void SymbolMark_Proc::drawWithVertexShader(const gfx::DeviceContext& device_ctx,
                                           const MarkPerGpuData& mark_gpu_data,
                                           gfx::Framebuffer& framebuffer,
                                           gfx::RenderPass& render_pass) {
  RENDER_LOG_SCOPE_P(device_ctx.getGpuId());

  auto& primitive_assembly = mark_gpu_data.fill_primitive_assemblies[0];
  mark_gpu_data.fill_materials[0]->updateDescriptorSets();

  device_ctx.getCommandList()
      .pushLabel("Symbol proc draw")
      .beginRenderPass(render_pass, framebuffer)
      .drawVertices(*mark_gpu_data.graphics_pipelines[0],
                    *primitive_assembly->getVertexBuffer(),
                    primitive_assembly->numVertices(),
                    primitive_assembly->getVertexBufferOffsetBytes())
      .endRenderPass()
      .popLabel()
      .flush("SymbolMark_Proc draw", gfx::CommandList::SubmitType::kImmediateReturn);
}

std::vector<CoordAttrInfo2d> SymbolMark_Proc::getCoordPropAttrInfos() const {
  std::vector<CoordAttrInfo2d> rtn;
  using PropId = RenderPropertyContainer::PropId;
  rtn.emplace_back(getPropFromCoordType(pos_dim_types_.x_type,
                                        *render_props_->getProperty(PropId::kX),
                                        *render_props_->getProperty(PropId::kX2),
                                        *render_props_->getProperty(PropId::kXC)),
                   getPropFromCoordType(pos_dim_types_.y_type,
                                        *render_props_->getProperty(PropId::kY),
                                        *render_props_->getProperty(PropId::kY2),
                                        *render_props_->getProperty(PropId::kYC)));

  if (pos_dim_types_.width_type == DimensionType::kCoords ||
      pos_dim_types_.height_type == DimensionType::kCoords) {
    rtn.emplace_back(render_props_->getProperty(PropId::kX2),
                     render_props_->getProperty(PropId::kY2));
  }

  return rtn;
}

SymbolMark_Proc::operator std::string() const {
  return "SymbolMark_Proc " + std::string(ctx_.getRenderSessionKey());
}

}  // namespace QueryRenderer
