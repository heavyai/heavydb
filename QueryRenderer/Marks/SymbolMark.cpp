/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/SymbolMark.h"

#include <boost/algorithm/string/find.hpp>

#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Marks/LineUtils.h"
#include "QueryRenderer/Marks/MarkProjectionShaderPolicy.h"
#include "QueryRenderer/Marks/RenderPropertyContainer.h"
#include "QueryRenderer/Marks/Utils.h"

namespace QueryRenderer {

using ::gfx::InterleavedBufferLayout;
using ::gfx::InterleavedBufferLayoutShPtr;
using ::gfx::ShaderManager;
using ::gfx::ShaderStage;
using ::gfx::ShaderStageBits;
using ::gfx::VertexBuffer;
using ShaderBuilder = ::gfx::ShaderManager::Builder;

namespace {
bool isEnumExplicitlyDefined(const EnumRenderProperty& enum_prop, int gpu_id = -1) {
  if (gpu_id >= 0) {
    return !enum_prop.usesScaleConfig() && !enum_prop.hasVboPtr(gpu_id) &&
           !enum_prop.hasSsboPtr(gpu_id);
  } else {
    return !enum_prop.usesScaleConfig() && !enum_prop.hasVboPtr() &&
           !enum_prop.hasSsboPtr();
  }
}

void buildInstancedGeomData(gfx::ResourceManager& rsrc_mgr,
                            gfx::BufferWrapperUqPtr& instanced_geom_vbo,
                            gfx::BufferWrapperUqPtr& instanced_geom_ibo,
                            std::vector<gfx::IndirectDrawIndexData>& geom_fill_data,
                            std::vector<gfx::IndirectDrawVertexData>& geom_stroke_data) {
  // clang-format off
  if (!instanced_geom_vbo) {
    // Attributes in 2 pairs [x, y], [u, v]
    // All vertex attributes for stroke shape first, followed by fill shape
    // Repeat the first 3 vertices at the end for the stroke line adjacency
    std::vector<float> rect_data({
        -0.5,   -0.5,  0,       0,     0.5,    -0.5,    1,       0,      0.5,    0.5,     1,      1,      -0.5,
         0.5,    0,     1,  // circle

        -0.5,  -0.5,  0, 0,   0.5, -0.5,  1, 0,   0.5, 0.5,  1, 1,  -0.5,  0.5,   0,  1,
        -0.5,  -0.5,  0, 0,   0.5, -0.5,  1, 0,   0.5, 0.5,  1, 1,  // square

        -0.1667, -0.5,   0.3333,  0.0,   0.1667, -0.5,     0.6667,  0.0,     0.1667, -0.1667,  0.6667, 0.3333, 0.5,
        -0.1667,  1.0,   0.3333,  0.5,   0.1667,  1.0,     0.6667,  0.1667,  0.1667,  0.6667,  0.6667, 0.1667, 0.5,
         0.6667,  1.0,  -0.1667,  0.5,   0.3333,  1.0,    -0.1667,  0.1667,  0.3333,  0.6667, -0.5,    0.1667, 0.0,
         0.6667, -0.5,  -0.1667,  0.0,   0.3333, -0.1667, -0.1667,  0.3333,  0.3333, -0.1667, -0.5,    0.3333, 0.0,
         0.1667, -0.5,   0.6667,  0.0,   0.1667, -0.1667,  0.6667,  0.3333,  // cross

        0.0,     -0.5,  0.5,     0.0,   0.5,    0.0,     1.0,     0.5,    0.0,    0.5,     0.5,    1.0,    -0.5,
        0.0,     0.0,   0.5,     0.0,   -0.5,   0.5,     0.0,     0.5,    0.0,    1.0,     0.5,    0.0,    0.5,
        0.5,     1.0,  // diamond

        -0.5,  -0.5,    0.0,   0.0,    0.5,   -0.5,    1.0,     0.0,    0.0,    0.5,     0.5,    1.0,
        -0.5,  -0.5,    0.0,   0.0,    0.5,   -0.5,    1.0,     0.0,    0.0,    0.5,     0.5,    1.0,  // triangle-up

         0.0,  -0.5,    0.5,   0.0,    0.5,    0.5,     1.0,     1.0,    -0.5,   0.5,     0.0,    1.0,    0.0,
        -0.5,    0.5,   0.0,     0.5,   0.5,    1.0,     1.0,     -0.5,   0.5,    0.0,     1.0,  // triangle-down

        -0.5,    -0.5,  0.0,     0.0,   0.5,    0.0,     1.0,     0.5,    -0.5,   0.5,     0.0,    1.0,    -0.5,
        -0.5,    0.0,   0.0,     0.5,   0.0,    1.0,     0.5,     -0.5,   0.5,    0.0,     1.0,  // triangle-right

        0.5,     -0.5,  1.0,     0.0,   0.5,    0.5,     1.0,     1.0,    -0.5,   0.0,     0.0,    0.5,    0.5,
        -0.5,    1.0,   0.0,     0.5,   0.5,    1.0,     1.0,     -0.5,   0.0,    0.0,     0.5,  // triangle-left

        -0.5,    -0.25, 0.0,     0.25,  0.0,    -0.5,    0.5,     0.0,    0.5,    -0.25,   1.0,    0.25,   0.5,
        0.25,    1.0,   0.75,    0.0,   0.5,    0.5,     1.0,     -0.5,   0.25,   0.0,     0.75,   -0.5,   -0.25,
        0.0,     0.25,  0.0,     -0.5,  0.5,    0.0,     0.5,     -0.25,  1.0,    0.25,  // hexagon-horiz

        -0.25,   -0.5,  0.25,    0.0,   0.25,   -0.5,    0.75,    0.0,    0.5,    0.0,     1.0,    0.5,    0.25,
        0.5,     0.75,  1.0,     -0.25, 0.5,    0.25,    1.0,     -0.5,   0.0,    0.0,     0.5,    -0.25,  -0.5,
        0.25,    0.0,   0.25,    -0.5,  0.75,   0.0,     0.5,     0.0,    1.0,    0.5,  // hexagon-vert

        -0.125, -0.1667,   0.0, 0.0,   0.125, -0.1667,   1.0, 0.0,    0.0, 0.8333,   0.5, 1.0,
        -0.125, -0.1667,   0.0, 0.0,   0.125, -0.1667,   1.0, 0.0,    0.0, 0.8333,   0.5, 1.0, // wedge

         0.0,    0.5,  0.0, 0.0,   0.2,   0.05,  0.0, 0.0,   0.075, 0.05,  0.0, 0.0,   0.075, -0.5,  0.0, 0.0,
        -0.075, -0.5,  0.0, 0.0,  -0.075, 0.05,  0.0, 0.0,  -0.2,   0.05,  0.0, 0.0,
         0.0,    0.5,  0.0, 0.0,   0.2,   0.05,  0.0, 0.0,   0.075, 0.05,  0.0, 0.0, // arrow

        // Airplane [begin]
         0.0,   0.5,   0.0, 0.0,
         0.03,  0.48,  0.0, 0.0,
         0.04,  0.45,  0.0, 0.0,
         0.05,  0.4,   0.0, 0.0,
         0.05,  0.1,   0.0, 0.0,
         0.5,  -0.1,   0.0, 0.0, // right wing tip
         0.5,  -0.18,  0.0, 0.0,
         0.05, -0.1,   0.0, 0.0,
         0.05, -0.35,  0.0, 0.0,
         0.2,  -0.45,  0.0, 0.0, // right stabilizer
         0.19, -0.49,  0.0, 0.0,
         0.0,  -0.46,  0.0, 0.0, // bottom point (12)
        -0.19, -0.49,  0.0, 0.0,
        -0.2,  -0.45,  0.0, 0.0,
        -0.05, -0.35,  0.0, 0.0,
        -0.05, -0.1,   0.0, 0.0,
        -0.5,  -0.18,  0.0, 0.0,
        -0.5,  -0.1,   0.0, 0.0,
        -0.05,  0.1,   0.0, 0.0,
        -0.05,  0.4,   0.0, 0.0,
        -0.04,  0.45,  0.0, 0.0,
        -0.03,  0.48,  0.0, 0.0,
         0.0,   0.5,   0.0, 0.0, // repeat first 3 for stroke
         0.03,  0.48,  0.0, 0.0,
         0.04,  0.45,  0.0, 0.0
        // Airplane [end]
    });

    // Triangle indices for fill
    std::vector<uint32_t> rect_ibo({
        0,  1,  2,  0,  2,  3,                                                   // circle
        4,  5,  6,  4,  6,  7,                                                   // square
        11, 12, 13, 11, 13, 22, 21, 14, 15, 21, 15, 20, 19, 16, 17, 19, 17, 18,  // cross
        26, 27, 28, 26, 28, 29,                                                  // diamond
        33, 34, 35,                                                              // triangle-up
        39, 40, 41,                                                              // triangle-down
        45, 46, 47,                                                              // triangle-right
        51, 52, 53,                                                              // triangle-left
        57, 58, 59, 57, 59, 60, 57, 60, 62, 62, 60, 61,                          // hexagon-horiz
        71, 66, 70, 66, 69, 70, 66, 67, 69, 67, 68, 69,                          // hexagon-vert
        75, 76, 77,                                                              // wedge
        81, 82, 83, 81, 83, 86, 81, 86, 87, 83, 84, 85, 83, 85, 86,              // arrow
        // Airplane [begin]
        91, 112, 92, 92, 112, 111, 92, 111, 93, 93, 111, 110, 93, 110, 94, 94, 110, 109, 94, 109, 95, // plane body fore of wings start
        95, 109, 106, 95, 106, 98, 98, 106, 105, 98, 105, 99, // plane body aft of wing start
        99, 105, 102, // plane body tail triangle
        95, 98, 96, 96, 98, 97, // right wing
        109, 108, 106, 106, 108, 107, // left wing
        99, 102, 100, 100, 102, 101, // right stabilizer
        105, 104, 102, 102, 104, 103 // left stabilizer
        // Airplane [end]
    });
    // clang-format on

    // Num indices and start index for fill triangles
    if (!geom_fill_data.size()) {
      geom_fill_data = {{6, 1, 0},     // circle
                        {6, 1, 6},     // square
                        {18, 1, 12},   // cross
                        {6, 1, 30},    // diamond
                        {3, 1, 36},    // triangle-up
                        {3, 1, 39},    // triangle-down
                        {3, 1, 42},    // triangle-right
                        {3, 1, 45},    // triangle-left
                        {12, 1, 48},   // hexagon-horiz
                        {12, 1, 60},   // hexagon-vert
                        {3, 1, 72},    // wedge
                        {15, 1, 75},   // arrow
                        {60, 1, 90}};  // airplane

      // Num vertices and start vertex for stroke lines
      geom_stroke_data = {{4, 1, 0},     // circle
                          {7, 1, 4},     // square
                          {15, 1, 11},   // cross
                          {7, 1, 26},    // diamond
                          {6, 1, 33},    // triangle-up
                          {6, 1, 39},    // triangle-down
                          {6, 1, 45},    // triangle-right
                          {6, 1, 51},    // triangle-left
                          {9, 1, 57},    // hexgon-horiz
                          {9, 1, 66},    // hexagon-vert
                          {6, 1, 75},    // wedge
                          {10, 1, 81},   // arrow
                          {25, 1, 91}};  // airplane
    }

    auto rect_layout = std::make_shared<InterleavedBufferLayout>();
    rect_layout->addAttribute<float>("x");
    rect_layout->addAttribute<float>("y");
    rect_layout->addAttribute<float>("u");
    rect_layout->addAttribute<float>("v");

    // destroy any existing buffers
    if (instanced_geom_vbo) {
      rsrc_mgr.destroyBuffer(std::move(instanced_geom_vbo));
    }
    if (instanced_geom_ibo) {
      rsrc_mgr.destroyBuffer(std::move(instanced_geom_ibo));
    }

    // create new buffers
    instanced_geom_vbo = rsrc_mgr.createBuffer("SymbolMark (old) VBO",
                                               {gfx::BufferType::kVertexBuffer,
                                                rect_data.size() * sizeof(float),
                                                gfx::BufferUsageBits::kLayoutBufferBit,
                                                gfx::BufferAccessType::kDeviceLocal});
    instanced_geom_vbo->updateSubDataWithLayout(
        &rect_data[0], instanced_geom_vbo->getNumBytes(), 0, rect_layout);
    instanced_geom_ibo = rsrc_mgr.createBuffer("SymbolMark (old) IBO",
                                               {gfx::BufferType::kIndexBuffer,
                                                rect_ibo.size() * sizeof(uint32_t),
                                                gfx::BufferUsageBits::kNone,
                                                gfx::BufferAccessType::kDeviceLocal});
    instanced_geom_ibo->updateSubData(
        rect_ibo.data(), instanced_geom_ibo->getNumBytes(), 0);
  }
}

SymbolMark::CoordDimensionTypes getPosAndDimensionTypes(
    const rapidjson::Pointer& x_json_path,
    const rapidjson::Pointer& x2_json_path,
    const rapidjson::Pointer& xc_json_path,
    const rapidjson::Pointer& y_json_path,
    const rapidjson::Pointer& y2_json_path,
    const rapidjson::Pointer& yc_json_path) {
  SymbolMark::CoordinateType x(SymbolMark::CoordinateType::kPrimary),
      y(SymbolMark::CoordinateType::kPrimary);
  SymbolMark::DimensionType width(SymbolMark::DimensionType::kValue),
      height(SymbolMark::DimensionType::kValue);
  if (!RapidJSONUtils::isValidPath(x_json_path)) {
    if (RapidJSONUtils::isValidPath(x2_json_path)) {
      x = SymbolMark::CoordinateType::kSecondary;
    } else if (RapidJSONUtils::isValidPath(xc_json_path)) {
      x = SymbolMark::CoordinateType::kCenter;
    }
  } else if (RapidJSONUtils::isValidPath(x2_json_path)) {
    width = SymbolMark::DimensionType::kCoords;
  }

  if (!RapidJSONUtils::isValidPath(y_json_path)) {
    if (RapidJSONUtils::isValidPath(y2_json_path)) {
      y = SymbolMark::CoordinateType::kSecondary;
    } else if (RapidJSONUtils::isValidPath(yc_json_path)) {
      y = SymbolMark::CoordinateType::kCenter;
    }
  } else if (RapidJSONUtils::isValidPath(y2_json_path)) {
    height = SymbolMark::DimensionType::kCoords;
  }

  return {x, y, width, height};
}

const BaseRenderProperty* getPropFromCoordType(
    const SymbolMark::CoordinateType coord_type,
    const BaseRenderProperty& primary,
    const BaseRenderProperty& secondary,
    const BaseRenderProperty& center) {
  switch (coord_type) {
    case SymbolMark::CoordinateType::kPrimary:
      return &primary;
    case SymbolMark::CoordinateType::kSecondary:
      return &secondary;
    case SymbolMark::CoordinateType::kCenter:
      return &center;
  }
  CHECK(false);
  return nullptr;
}
}  // namespace

std::vector<::gfx::IndirectDrawIndexData> SymbolMark::geom_fill_data = {};
std::vector<::gfx::IndirectDrawVertexData> SymbolMark::geom_stroke_data = {};

SymbolMark::SymbolMark(const JSONLocation& obj_loc, QueryRendererContext& ctx)
    : BaseMark(GeomType::kLegacySymbols, ctx, obj_loc, DataOutputFormat::kRows, false)
    , shape_("shape",
             QueryDataType::SYMBOL_SHAPE_ENUM,
             ctx,
             *prop_mark_facade_,
             RenderPropertyFlagBits::kUseScale | RenderPropertyFlagBits::kFlexibleType,
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
                      DimensionType::kValue}) {
  using type = RenderPropertyCreateInfo::Type;
  using flag = RenderPropertyFlagBits;
  std::vector<RenderPropertyCreateInfo> render_property_ci{
      {type::kExtendedCoords},
      {type::kFillColor, flag::kUnspecified, false, gfx::ColorUnion(0.f, 0.f, 0.f, 1.f)},
      {type::kFillOpacity, flag::kUnspecified, false, 1.0f},
      {type::kStrokeColor, flag::kUseScale, false, gfx::ColorUnion(1.f, 1.f, 1.f, 1.f)},
      {type::kStrokeOpacity, flag::kUnspecified, false, 1.0f},
      {type::kStrokeWidth, flag::kUnspecified, false, 0.0f},
      {type::kLineJoinType,
       flag::kFlexibleType,
       false,
       static_cast<int>(LineJoinType::kMiter),
       QueryDataType::LINE_JOIN_ENUM,
       convertStringToLineJoinEnum},
      {type::kMiterLimit, flag::kFlexibleType, false, 10.0f}};  // scb: unused?

  render_props_ = std::make_unique<RenderPropertyContainer>(
      ctx, *prop_mark_facade_, std::move(render_property_ci));

  auto const& coord_props = render_props_->getCoordProperties();
  using PropId = RenderPropertyContainer::PropId;
  used_fill_props_ = {&shape_,
                      &width_,
                      &height_,
                      &angle_,
                      &angle_unit_,
                      render_props_->getProperty(PropId::kOpacity),
                      render_props_->getProperty(PropId::kFillColor),
                      render_props_->getProperty(PropId::kFillOpacity)};
  used_fill_props_.insert(coord_props.begin(), coord_props.end());

  used_stroke_props_ = {&shape_,
                        &width_,
                        &height_,
                        &angle_,
                        &angle_unit_,
                        render_props_->getProperty(PropId::kOpacity),
                        render_props_->getProperty(PropId::kStrokeColor),
                        render_props_->getProperty(PropId::kStrokeOpacity),
                        render_props_->getProperty(PropId::kStrokeWidth),
                        render_props_->getProperty(PropId::kLineJoinType),
                        render_props_->getProperty(PropId::kMiterLimit)};
  used_stroke_props_.insert(coord_props.begin(), coord_props.end());

  used_fill_props_const_.insert(used_fill_props_.begin(), used_fill_props_.end());
  used_stroke_props_const_.insert(used_stroke_props_.begin(), used_stroke_props_.end());

  projection_policy_ = std::make_unique<MarkProjectionShaderPolicy>(
      MarkProjectionShaderPolicy::PropMap{coord_props.begin(), coord_props.end()});

  initPropertiesFromJSONObj(obj_loc, true, true);
  initTransformsFromJSONObj(obj_loc, getCoordPropAttrInfos());
  json_path_ = obj_loc.getPathRef();
}

SymbolMark::~SymbolMark() {}

BaseRenderPropertyConstSet SymbolMark::getUsedProps() const {
  BaseRenderPropertyConstSet rtn(used_fill_props_const_);
  rtn.insert(used_stroke_props_const_.begin(), used_stroke_props_const_.end());
  return rtn;
}

void SymbolMark::initPropertiesFromJSONObj(const JSONLocation& obj_loc,
                                           const bool data_changed,
                                           const bool init) {
  RENDER_LOG_SCOPE() << " data_changed: " << data_changed << "  init: " << init;
  auto const prop_loc = obj_loc.getMember(JSONSchema_v1::Marks::kPropertiesProp);
  RUNTIME_EX_ASSERT(
      prop_loc.isValid(),
      RapidJSONUtils::createJsonParseError(
          obj_loc,
          "Mark objects must have a \"" +
              std::string(JSONSchema_v1::Marks::kPropertiesProp) + "\" property."));

  auto const prev_path = properties_json_path_;
  properties_json_path_ = obj_loc.getPathRef();
  if (!ctx_.isJSONCacheUpToDate(prev_path, prop_loc) || data_changed || init) {
    RUNTIME_EX_ASSERT(prop_loc.isObject(),
                      RapidJSONUtils::createJsonParseError(
                          prop_loc, "Property must be a json object."));

    render_props_->initFromJSONObj(obj_loc, data_changed);

    auto const obj_num_check = BaseMark::validateNumPropFunc(*this);
    auto const obj_enum_check = BaseMark::validateEnumPropFunc(*this);
    auto const init_prop_func = [this, data_changed](auto const& prop_loc,
                                                     auto* prop,
                                                     auto validate_type_func,
                                                     auto post_empty_func) {
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
        [](const std::string& prop_name, const JSONLocation& shape_loc) {
          RUNTIME_EX_ASSERT(
              shape_loc.isObject() || shape_loc.isInt() || shape_loc.isUint64() ||
                  shape_loc.isString(),
              RapidJSONUtils::createJsonParseError(
                  shape_loc,
                  "\"" + prop_name +
                      "\" symbol mark property must be a scale/data reference "
                      "or an enum value (i.e. an int or a string)"));
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
      init_prop_func(prop_loc, &width_, obj_num_check, [&](const JSONLocation&) {
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

    init_prop_func(prop_loc, &angle_, obj_num_check, [this](const JSONLocation&) {
      angle_.initializeValue(0.0f);
    });

    init_prop_func(prop_loc, &angle_unit_, obj_enum_check, [this](const JSONLocation&) {
      angle_unit_.initializeValue(static_cast<int>(AngleUnit::kDegrees));
    });

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

    if (init || data_changed) {
      updateProps(getUsedProps());
    }

    updateVisibility(render_props_->isFillActive() || render_props_->isStrokeActive());
  }
}

void SymbolMark::buildShaders(ShaderBuilderVector& builders,
                              const BaseRenderPropertyConstSet& props) {
  std::stringstream get_prop_ss;
  streamPropertyGetters(props, get_prop_ss, projection_policy_.get());
  builders[0]->replaceFirstTag("PropertyGetters", get_prop_ss.str());

  // Inject additional code (prop types, scales, projection, casting, etc.)
  BaseMark::insertPropertyCodeInShaderBuilders(builders, props, *projection_policy_);
}

void SymbolMark::updateShader() {
  RENDER_LOG_SCOPE() << "building glsl shaders";

  ShaderManager::BuilderUqPtrVector circle_fill_builders, geom_fill_builders;
  ShaderManager::BuilderUqPtrVector circle_stroke_builders, geom_stroke_builders;

  updateGeoPropInfoAndPropCompressionBits({kPOINT});

  const bool is_static_shape = isEnumExplicitlyDefined(shape_);
  const bool do_fill = render_props_->isFillActive();
  bool requires_circle_shaders =
      (!is_static_shape ||
       shape_.getUniformValue<int>() == static_cast<int>(SymbolShapeType::kCircle));
  bool requires_non_circle_shaders =
      (!is_static_shape ||
       shape_.getUniformValue<int>() != static_cast<int>(SymbolShapeType::kCircle));

  auto& shader_mgr = ctx_.getShaderManager();

  auto vbo_props_str = buildVertexShaderInputs();

  gfx::GlslStructBuilder ubo_struct_builder("SYMBOL_VERT_UBO_TYPE");
  ubo_struct_builder.addMember("totalNumInstances", gfx::BufferAttrType::kUint);
  ubo_struct_builder.addMember("uViewProjMatrix", gfx::BufferAttrType::kMat3x2f);
  ubo_struct_builder.addMember("uPivotx", gfx::BufferAttrType::kFloat);
  ubo_struct_builder.addMember("uPivoty", gfx::BufferAttrType::kFloat);
  ubo_struct_builder.addMember("uSinAngle", gfx::BufferAttrType::kFloat);
  ubo_struct_builder.addMember("uCosAngle", gfx::BufferAttrType::kFloat);
  ubo_struct_builder.addMember("invalidKey", gfx::BufferAttrType::kUint64);
  ubo_struct_builder.addMember("propCompressionBits", gfx::BufferAttrType::kUint);

  addCommonRenderPropUniforms(ubo_struct_builder);
  auto ubo_props_str = ubo_struct_builder.createStructString();

  auto has_accumulator = hasAccumulator();

  // Fragment shader inputs

  // if isFillPass || isCircle
  //   flat uint64_t fRowId
  //   flat vec4 fColor
  //   flat int accumIdx (accumulation renders only)
  //   if isCircle
  //     flat out float fWidth
  //     flat out float fHeight
  //     vec2 fouterUVCoord
  //     if !isFillPass
  //       vec2 finnerUVCoord
  //       flat float fStrokeWidth
  std::vector<gfx::GlslStructBuilder::Qualifier> quals = {
      gfx::GlslStructBuilder::Qualifier::kFlat};

  gfx::GlslStructBuilder base_fragment_inputs("BaseFragmentInputs");
  base_fragment_inputs.addMember("fRowId", gfx::BufferAttrType::kUint64, quals);
  base_fragment_inputs.addMember("fColor", gfx::BufferAttrType::kVec4f);
  if (has_accumulator) {
    base_fragment_inputs.addMember("accumIdx", gfx::BufferAttrType::kInt, quals);
  }
  auto base_fragment_inputs_str = base_fragment_inputs.createInterfaceBlockString(true);

  if (do_fill) {
    circle_fill_builders = shader_mgr.createBuilderVector(
        {{"Marks/symbolTemplate.vert"}, {"Marks/symbolTemplate.frag"}});

    // vertex shader
    circle_fill_builders[0]->replaceFirstTag("VertexProperties", vbo_props_str);
    circle_fill_builders[0]->replaceFirstTag("UniformProperties", ubo_props_str);

    circle_fill_builders[0]->replaceFirstTag(
        "computeX", std::to_string(static_cast<int>(pos_dim_types_.x_type)));
    circle_fill_builders[0]->replaceFirstTag(
        "computeY", std::to_string(static_cast<int>(pos_dim_types_.y_type)));
    circle_fill_builders[0]->replaceFirstTag(
        "computeWidth", std::to_string(static_cast<int>(pos_dim_types_.width_type)));
    circle_fill_builders[0]->replaceFirstTag(
        "computeHeight", std::to_string(static_cast<int>(pos_dim_types_.height_type)));
    circle_fill_builders[0]->replaceFirstTag("isFillPass", "1");
    BaseMark::setKeyInShaderBuilder(*circle_fill_builders[0]);
    BaseMark::setColorConvertSubroutines(
        *circle_fill_builders[0],
        render_props_->getProperty(RenderPropertyContainer::PropId::kFillColor));

    circle_fill_builders[0]->setExternalUniformBuffers({"SLAB_ADDRESS_TABLE_UBO"});

    // fragment shader
    circle_fill_builders[1]->replaceFirstTag("isFillPass", "1");

    // subroutines
    buildSubroutineBindings(*circle_fill_builders[0], used_fill_props_const_);
    buildSubroutineBindings(*circle_fill_builders[1], used_fill_props_const_);

    if (requires_non_circle_shaders) {
      // TODO(scb) this is inefficient, but this Mark is the only place we try something
      // like this, and in the grand scheme it's not going to be measurable
      geom_fill_builders = shader_mgr.cloneBuilderVector(circle_fill_builders);
      geom_fill_builders[0]->replaceFirstTag("isCircle", "0");
      geom_fill_builders[1]->replaceFirstTag("isCircle", "0");
      geom_fill_builders[0]->replaceFirstTag("FragmentShaderInputs",
                                             base_fragment_inputs_str);
      geom_fill_builders[1]->replaceFirstTag("FragmentShaderInputs",
                                             base_fragment_inputs_str);
      buildShaders(geom_fill_builders, used_fill_props_const_);
    }

    if (requires_circle_shaders) {
      auto circle_fragment_inputs =
          base_fragment_inputs.clone("CircleFillFragmentInputs");
      circle_fragment_inputs.addMember("fWidth", gfx::BufferAttrType::kFloat, quals);
      circle_fragment_inputs.addMember("fHeight", gfx::BufferAttrType::kFloat, quals);
      circle_fragment_inputs.addMember("fouterUVCoord", gfx::BufferAttrType::kVec2f);
      auto circle_fragment_shader_inputs_str =
          circle_fragment_inputs.createInterfaceBlockString(true);
      circle_fill_builders[0]->replaceFirstTag("FragmentShaderInputs",
                                               circle_fragment_shader_inputs_str);
      circle_fill_builders[1]->replaceFirstTag("FragmentShaderInputs",
                                               circle_fragment_shader_inputs_str);

      circle_fill_builders[0]->replaceFirstTag("isCircle", "1");
      circle_fill_builders[1]->replaceFirstTag("isCircle", "1");
      buildShaders(circle_fill_builders, used_fill_props_const_);
    }
  }

  const bool do_stroke = render_props_->isStrokeActive();
  if (do_stroke) {
    auto stroke_vert_builder = shader_mgr.createBuilder("Marks/symbolTemplate.vert");

    stroke_vert_builder->replaceFirstTag("VertexProperties", vbo_props_str);
    stroke_vert_builder->replaceFirstTag("UniformProperties", ubo_props_str);

    stroke_vert_builder->replaceFirstTag(
        "computeX", std::to_string(static_cast<int>(pos_dim_types_.x_type)));
    stroke_vert_builder->replaceFirstTag(
        "computeY", std::to_string(static_cast<int>(pos_dim_types_.y_type)));
    stroke_vert_builder->replaceFirstTag(
        "computeWidth", std::to_string(static_cast<int>(pos_dim_types_.width_type)));
    stroke_vert_builder->replaceFirstTag(
        "computeHeight", std::to_string(static_cast<int>(pos_dim_types_.height_type)));

    BaseMark::setKeyInShaderBuilder(*stroke_vert_builder);

    stroke_vert_builder->replaceFirstTag("isFillPass", "0");
    BaseMark::setColorConvertSubroutines(
        *stroke_vert_builder,
        render_props_->getProperty(RenderPropertyContainer::PropId::kStrokeColor));
    buildSubroutineBindings(*stroke_vert_builder, used_stroke_props_const_);

    stroke_vert_builder->setExternalUniformBuffers({"SLAB_ADDRESS_TABLE_UBO"});

    if (requires_circle_shaders) {
      auto circle_stroke_vert_builder = shader_mgr.cloneBuilder(*stroke_vert_builder);
      auto circle_stroke_frag_builder =
          shader_mgr.createBuilder("Marks/symbolTemplate.frag");

      auto circle_fragment_inputs =
          base_fragment_inputs.clone("CircleStrokeFragmentInputs");
      circle_fragment_inputs.addMember("fWidth", gfx::BufferAttrType::kFloat, quals);
      circle_fragment_inputs.addMember("fHeight", gfx::BufferAttrType::kFloat, quals);
      circle_fragment_inputs.addMember("fouterUVCoord", gfx::BufferAttrType::kVec2f);
      circle_fragment_inputs.addMember("finnerUVCoord", gfx::BufferAttrType::kVec2f);
      circle_fragment_inputs.addMember(
          "fStrokeWidth", gfx::BufferAttrType::kFloat, quals);

      auto circle_fragment_inputs_str =
          circle_fragment_inputs.createInterfaceBlockString("CircleStrokeFragmentInputs");
      circle_stroke_vert_builder->replaceFirstTag("FragmentShaderInputs",
                                                  circle_fragment_inputs_str);
      circle_stroke_frag_builder->replaceFirstTag("FragmentShaderInputs",
                                                  circle_fragment_inputs_str);

      circle_stroke_vert_builder->replaceFirstTag("isCircle", "1");
      circle_stroke_frag_builder->replaceFirstTag("isCircle", "1");
      circle_stroke_frag_builder->replaceFirstTag("isFillPass", "0");

      circle_stroke_builders.push_back(std::move(circle_stroke_vert_builder));
      circle_stroke_builders.push_back(std::move(circle_stroke_frag_builder));

      buildShaders(circle_stroke_builders, used_stroke_props_const_);
    }

    // requires non-circle shaders (stroke is geometry shader driven)
    if (requires_non_circle_shaders) {
      auto stroke_frag_builder = shader_mgr.createBuilder("Marks/lineTemplate.frag");
      auto stroke_geom_builder = shader_mgr.createBuilder("Marks/lineTemplate.geom");
      stroke_vert_builder->replaceFirstTag("isCircle", "0");
      stroke_geom_builder->replaceFirstTag("doStrokeAccum", "0");

      gfx::GlslStructBuilder geometry_inputs("GeometryShaderInputs");
      gfx::GlslStructBuilder fragment_inputs("FragmentShaderInputs");
      generate_line_interface_blocks(geometry_inputs, fragment_inputs, has_accumulator);

      stroke_vert_builder->replaceFirstTag(
          "GeometryShaderInputs", geometry_inputs.createInterfaceBlockString(true, {}));
      stroke_geom_builder->replaceFirstTag(
          "GeometryShaderInputs",
          geometry_inputs.createInterfaceBlockString(true, std::nullopt, true));

      auto fragment_inputs_str = fragment_inputs.createInterfaceBlockString(true);
      stroke_geom_builder->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);
      stroke_frag_builder->replaceFirstTag("FragmentShaderInputs", fragment_inputs_str);

      geom_stroke_builders.push_back(std::move(stroke_vert_builder));
      geom_stroke_builders.push_back(std::move(stroke_frag_builder));
      geom_stroke_builders.push_back(std::move(stroke_geom_builder));

      buildShaders(geom_stroke_builders, used_stroke_props_const_);
    }
  }

  ctx_.clearMarkShaders(*this);
  if (is_static_shape) {
    if (do_fill) {
      ctx_.buildMarkShaders(*this,
                            MarkGpuResourceSlot::kFill,
                            requires_circle_shaders ? "SymbolMark (old) Circle Fill"
                                                    : "SymbolMark (old) Geom Fill",
                            requires_circle_shaders ? std::move(circle_fill_builders)
                                                    : std::move(geom_fill_builders));
    }
    if (do_stroke) {
      ctx_.buildMarkShaders(*this,
                            MarkGpuResourceSlot::kStroke,
                            requires_circle_shaders ? "SymbolMark (old) Circle Stroke"
                                                    : "SymbolMark (old) Geom Stroke",
                            requires_circle_shaders ? std::move(circle_stroke_builders)
                                                    : std::move(geom_stroke_builders));
    }
  } else {
    if (do_fill) {
      ctx_.buildMarkShaders(*this,
                            MarkGpuResourceSlot::kFill,
                            "SymbolMark (old) Circle Fill",
                            std::move(circle_fill_builders));
      ctx_.buildMarkShaders(*this,
                            MarkGpuResourceSlot::kFill,
                            "SymbolMark (old) Geom Fill",
                            std::move(geom_fill_builders));
    }
    if (do_stroke) {
      ctx_.buildMarkShaders(*this,
                            MarkGpuResourceSlot::kStroke,
                            "SymbolMark (old) Circle Stroke",
                            std::move(circle_stroke_builders));
      ctx_.buildMarkShaders(*this,
                            MarkGpuResourceSlot::kStroke,
                            "SymbolMark (old) Geom Stroke",
                            std::move(geom_stroke_builders));
    }
  }

  shader_dirty_ = false;

  // set the props dirty to force a rebind with the new shader
  setPropsDirty();
}

void SymbolMark::buildPipelineDescriptors() {
  //
  // just one shared PD for now
  //

  if (!pipeline_descriptor_) {
    pipeline_descriptor_ = std::make_unique<gfx::PipelineDescriptor>();
    CHECK(pipeline_descriptor_);
  }

  pipeline_descriptor_->setRasterSampleCount(getRasterizationSampleCount());
  pipeline_descriptor_->setEnableDepthTest(true);
  pipeline_descriptor_->setPushConstantRanges(
      {gfx::PushConstantRange{ShaderStageBits::kVertex, 0, sizeof(uint32_t)}});
}

void SymbolMark::buildPipelines(MarkPerGpuData& per_gpu_data) {
  per_gpu_data.destroyPipelines();
  per_gpu_data.graphics_pipelines.resize(4);

  auto const is_static_shape = isEnumExplicitlyDefined(shape_, per_gpu_data.getGpuId());

  auto const& render_pass = per_gpu_data.getRootPerGpuData().getCommonRenderPass(
      CommonRenderPassType::kAllAttachments, needsMultisampleEnabled());

  //
  // fill pipelines
  //

  if (!per_gpu_data.fill_materials.empty()) {
    CHECK(!per_gpu_data.fill_primitive_assemblies.empty());

    // fill 0
    per_gpu_data.graphics_pipelines[kFill0] =
        per_gpu_data.getResourceManager().createGraphicsPipeline(
            "LegacySymbol Fill 0",
            *per_gpu_data.fill_materials[0],
            *pipeline_descriptor_,
            per_gpu_data.fill_primitive_assemblies[0].get());
    per_gpu_data.graphics_pipelines[kFill0]->create(render_pass);

    if (!is_static_shape) {
      CHECK(per_gpu_data.fill_materials.size() == 2 &&
            per_gpu_data.fill_primitive_assemblies.size() == 2);

      // fill 1
      per_gpu_data.graphics_pipelines[kFill1] =
          per_gpu_data.getResourceManager().createGraphicsPipeline(
              "LegacySymbol Fill 1",
              *per_gpu_data.fill_materials[1],
              *pipeline_descriptor_,
              per_gpu_data.fill_primitive_assemblies[1].get());
      per_gpu_data.graphics_pipelines[kFill1]->create(render_pass);
    }
  }

  //
  // stroke pipelines
  //

  if (!per_gpu_data.stroke_materials.empty()) {
    CHECK(!per_gpu_data.stroke_primitive_assemblies.empty());

    // stroke 0
    per_gpu_data.graphics_pipelines[kStroke0] =
        per_gpu_data.getResourceManager().createGraphicsPipeline(
            "LegacySymbol Stroke 0",
            *per_gpu_data.stroke_materials[0],
            *pipeline_descriptor_,
            per_gpu_data.stroke_primitive_assemblies[0].get());
    per_gpu_data.graphics_pipelines[kStroke0]->create(render_pass);

    if (!is_static_shape) {
      CHECK(per_gpu_data.stroke_materials.size() == 2 &&
            per_gpu_data.stroke_primitive_assemblies.size() == 2);

      // stroke 1
      per_gpu_data.graphics_pipelines[kStroke1] =
          per_gpu_data.getResourceManager().createGraphicsPipeline(
              "LegacySymbol Stroke 1",
              *per_gpu_data.stroke_materials[1],
              *pipeline_descriptor_,
              per_gpu_data.stroke_primitive_assemblies[1].get());
      per_gpu_data.graphics_pipelines[kStroke1]->create(render_pass);
    }
  }
}

void SymbolMark::buildFillPrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  CHECK(!gpu_data.fill_materials.empty());
  auto& rsrc_mgr = gpu_data.getResourceManager();
  auto gpu_id = gpu_data.getGpuId();

  buildInstancedGeomData(rsrc_mgr,
                         gpu_data.instanced_geom_vbo,
                         gpu_data.instanced_geom_ibo,
                         geom_fill_data,
                         geom_stroke_data);

  auto vbo = static_cast<gfx::VertexBuffer*>(gpu_data.instanced_geom_vbo.get());
  CHECK(vbo);
  CHECK(vbo->hasLayout());
  CHECK(vbo->getLayoutManager()->getNumBufferLayouts() == 1);

  // this is the same for all shader variants
  gfx::PrimitiveAssemblyAttrInfo instanced_attr_info{
      {vbo, vbo->getLayoutManager()->getBufferLayoutAtIndex(0)},
      {{"x", "shapeposx"}, {"y", "shapeposy"}, {"u", "shapeu"}, {"v", "shapev"}}};

  gfx::PrimitiveAssemblyAttrInfo instances_attr_info;
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
    prop->addToPrimitiveAssemblyAttrInfo(gpu_id, instances_attr_info);
  }

  // cache the references to the two VBOs that will be used
  // this ensures they persist longer than the command execution
  CHECK(instances_attr_info.vbo_and_layout.vertex_buffer) << "No instances VBO";

  auto layout_data =
      instances_attr_info.vbo_and_layout.vertex_buffer->getLayoutManager()
          ->getBufferLayoutDataToUse(instances_attr_info.vbo_and_layout.buffer_layout,
                                     "Cannot get VertexBuffer offset");
  gpu_data.vertex_buffer_refs_cache = {
      {*vbo, 0u},
      {*instances_attr_info.vbo_and_layout.vertex_buffer, layout_data.offset_bytes}};

  gpu_data.fill_primitive_assemblies.clear();
  gpu_data.fill_primitive_assemblies.resize(gpu_data.fill_materials.size());

  if (gpu_data.fill_materials.size() == 1) {
    // single fill PA (any shape)
    gpu_data.fill_primitive_assemblies[0] =
        gpu_data.getResourceManager().createPrimitiveAssembly(
            "SymbolMark (Fill 0)",
            gfx::PrimitiveTopology::kTriangleList,
            *gpu_data.fill_materials[0],
            instanced_attr_info,
            instances_attr_info,
            1,
            static_cast<gfx::IndexBuffer*>(gpu_data.instanced_geom_ibo.get()));
  } else if (gpu_data.fill_materials.size() > 1) {
    CHECK(gpu_data.fill_materials.size() == 2);
    // first fill PA (circle)
    gpu_data.fill_primitive_assemblies[0] =
        gpu_data.getResourceManager().createPrimitiveAssembly(
            "SymbolMark (Fill 0)",
            gfx::PrimitiveTopology::kTriangleList,
            *gpu_data.fill_materials[0],
            instanced_attr_info,
            instances_attr_info,
            1,
            static_cast<gfx::IndexBuffer*>(gpu_data.instanced_geom_ibo.get()));
    // second fill PA (non-circle)
    gpu_data.fill_primitive_assemblies[1] =
        gpu_data.getResourceManager().createPrimitiveAssembly(
            "SymbolMark (Fill 1)",
            gfx::PrimitiveTopology::kTriangleList,
            *gpu_data.fill_materials[1],
            instanced_attr_info,
            instances_attr_info,
            1,
            static_cast<gfx::IndexBuffer*>(gpu_data.instanced_geom_ibo.get()));
  }

  CHECK(!gpu_data.fill_primitive_assemblies.empty());
}

// NOTES: There are 4 possible combinations of stroke shaders:
//  1 - no stroke, therefore no stroke shaders at all
//  2 - circle only. strokeMaterials will be size 1 and it will be a fragment-shader-
//      based circle shader (index instancing)
//  3 - only shapes other than circle. strokeMaterials will be size 1 and it
//      will be a geometry-shader-based stroke shader (vertex instancing, no IBO)
//  4 - both circle and non-circle shapes supported (shape is scale driven). In this
//      case strokeMaterials will be size 2, the first shader will be the circle
//      shader, and the second shader will be the shader that generates geometry for
//      all the other symbol shapes

void SymbolMark::buildStrokePrimitiveAssemblies(MarkPerGpuData& gpu_data) {
  CHECK(!gpu_data.stroke_materials.empty());
  auto& rsrc_mgr = gpu_data.getResourceManager();
  auto gpu_id = gpu_data.getGpuId();

  buildInstancedGeomData(rsrc_mgr,
                         gpu_data.instanced_geom_vbo,
                         gpu_data.instanced_geom_ibo,
                         geom_fill_data,
                         geom_stroke_data);

  auto ibo = static_cast<gfx::IndexBuffer*>(gpu_data.instanced_geom_ibo.get());
  auto vbo = static_cast<gfx::VertexBuffer*>(gpu_data.instanced_geom_vbo.get());
  CHECK(vbo);
  CHECK(vbo->hasLayout());
  CHECK(vbo->getLayoutManager()->getNumBufferLayouts() == 1);

  // this is the same for all shader variants
  gfx::PrimitiveAssemblyAttrInfo instanced_attr_info{
      {vbo, vbo->getLayoutManager()->getBufferLayoutAtIndex(0)},
      {{"x", "shapeposx"}, {"y", "shapeposy"}, {"u", "shapeu"}, {"v", "shapev"}}};

  gfx::PrimitiveAssemblyAttrInfo instances_attr_info;
  int attr_count = 0;
  int vbo_size = 0;
  int prop_size = 0;
  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    if (!gpu_data.stroke_materials[0]->hasVertexAttribute(prop->getName())) {
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
    prop->addToPrimitiveAssemblyAttrInfo(gpu_id, instances_attr_info);
  }

  // cache the references to the two VBOs that will be used
  // this ensures they persist longer than the command execution
  CHECK(instances_attr_info.vbo_and_layout.vertex_buffer) << "No instances VBO";

  auto layout_data =
      instances_attr_info.vbo_and_layout.vertex_buffer->getLayoutManager()
          ->getBufferLayoutDataToUse(instances_attr_info.vbo_and_layout.buffer_layout,
                                     "Cannot get VertexBuffer offset");
  gpu_data.vertex_buffer_refs_cache = {
      {*vbo, 0u},
      {*instances_attr_info.vbo_and_layout.vertex_buffer, layout_data.offset_bytes}};

  gpu_data.stroke_primitive_assemblies.clear();
  gpu_data.stroke_primitive_assemblies.resize(gpu_data.stroke_materials.size());

  CHECK_LE(gpu_data.stroke_materials.size(), 2u);

  if (gpu_data.stroke_materials.size() == 1) {
    // this must be true, as this PA can only be circle or non-circle not both
    CHECK(isEnumExplicitlyDefined(shape_, gpu_id));
    if (shape_.getUniformValue<int>() == static_cast<int>(SymbolShapeType::kCircle)) {
      // single stroke PA (circle only)
      gpu_data.stroke_primitive_assemblies[0] =
          gpu_data.getResourceManager().createPrimitiveAssembly(
              "SymbolMark (Stroke 0, Circle)",
              gfx::PrimitiveTopology::kTriangleList,
              *gpu_data.stroke_materials[0],
              instanced_attr_info,
              instances_attr_info,
              1,
              ibo);
    } else {
      // single stroke PA (non-circle only)
      gpu_data.stroke_primitive_assemblies[0] =
          gpu_data.getResourceManager().createPrimitiveAssembly(
              "SymbolMark (Stroke 0, Non-Circle)",
              gfx::PrimitiveTopology::kLineStripAdjacency,
              *gpu_data.stroke_materials[0],
              instanced_attr_info,
              instances_attr_info,
              1);
    }
  } else if (gpu_data.stroke_materials.size() == 2) {
    // first stroke PA (circle)
    gpu_data.stroke_primitive_assemblies[0] =
        gpu_data.getResourceManager().createPrimitiveAssembly(
            "SymbolMark (Stroke 0, Circle)",
            gfx::PrimitiveTopology::kTriangleList,
            *gpu_data.stroke_materials[0],
            instanced_attr_info,
            instances_attr_info,
            1,
            ibo);
    // second stroke PA (non-circle)
    gpu_data.stroke_primitive_assemblies[1] =
        gpu_data.getResourceManager().createPrimitiveAssembly(
            "SymbolMark (Stroke 1, Non-Circle)",
            gfx::PrimitiveTopology::kLineStripAdjacency,
            *gpu_data.stroke_materials[1],
            instanced_attr_info,
            instances_attr_info,
            1);
  }
  CHECK(!gpu_data.stroke_primitive_assemblies.empty());
}

void SymbolMark::buildSubroutineBindings(ShaderBuilder& builder,
                                         const BaseRenderPropertyConstSet& props) {
  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    if ((props.find(prop) != props.end()) || prop->hasAccumulatorScale()) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->buildSubroutineBindings(builder, "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    if ((props.find(prop) != props.end()) || prop->hasAccumulatorScale()) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->buildSubroutineBindings(builder, "_" + prop->getName());
      }
    }
  }
}

void SymbolMark::setUniformAttributes(const BaseRenderPropertyConstSet& props,
                                      gfx::Material& active_material) {
  RENDER_LOG_SCOPE();
  BaseMark::bindKeyPropUniformAttributes(active_material);
  for (auto const* prop : prop_buf_loc_state_.vbo_props) {
    if ((props.find(prop) != props.end() &&
         active_material.hasVertexAttribute(prop->getName())) ||
        prop->hasAccumulatorScale()) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->bindUniforms(active_material, "_" + prop->getName());
      }
    }
  }

  for (auto const* prop : prop_buf_loc_state_.uniform_props) {
    if ((props.find(prop) != props.end() &&
         active_material.hasUniformAttribute(prop->getName())) ||
        prop->hasAccumulatorScale()) {
      auto const& scale_ref = prop->getScaleReference();
      if (scale_ref != nullptr) {
        scale_ref->bindUniforms(active_material, "_" + prop->getName());
      }
      prop->setUniformAttribute(active_material, prop->getName());
    }
  }

  for (auto const* prop : prop_buf_loc_state_.decimal_props) {
    if (props.find(prop) != props.end()) {
      prop->setDecimalScaleUniformAttribute(active_material);
    }
  }

  BaseMark::bindIDPropUniformAttributes(active_material);
  BaseMark::setProjectionUniformAttributes(active_material);

  float pivotx = (pos_dim_types_.x_type == CoordinateType::kCenter ? 0.0f : 0.5f);
  float pivoty = (pos_dim_types_.y_type == CoordinateType::kCenter ? 0.0f : 0.5f);

  active_material.setUniformAttribute("uViewProjMatrix",
                                      ctx_.getViewProjMatrix().getDataArrayRef());
  active_material.setUniformAttribute("uPivotx", pivotx);
  active_material.setUniformAttribute("uPivoty", pivoty);
  if (!angle_.isDataDriven()) {
    float angle = -angle_.getUniformValue<float>();
    if (angle_unit_.getUniformValue<int>() == static_cast<int>(AngleUnit::kDegrees)) {
      angle = angle * 3.14159265359f / 180.0f;
    }
    active_material.setUniformAttribute("uSinAngle", std::sin(angle));
    active_material.setUniformAttribute("uCosAngle", std::cos(angle));
  }
  if (hasProjection()) {
    active_material.setViewportAttributes(0, 0, ctx_.getWidth(), ctx_.getHeight());
  }
}

void SymbolMark::updateRenderPropertyGpuResources(const std::vector<GpuId>& add_gpus,
                                                  const std::vector<GpuId>& remove_gpus) {
  BaseRenderPropertySet props(used_fill_props_);
  props.insert(used_stroke_props_.begin(), used_stroke_props_.end());
  for (auto const& prop : props) {
    prop->initGpuResources(add_gpus, remove_gpus);
  }
}

void SymbolMark::setUniformAttributes(MarkPerGpuData& gpu_data) {
  // now draw symbols, starting with fill
  auto const gpu_id = gpu_data.getGpuId();
  auto const is_static_shape = isEnumExplicitlyDefined(shape_, gpu_id);

  if (gpu_data.graphics_pipelines[kFill0]) {
    auto& mat0 = gpu_data.fill_materials[0];
    auto& pa0 = gpu_data.fill_primitive_assemblies[0];

    setUniformAttributes(used_fill_props_const_, *mat0);

    auto num_instances = pa0->numInstances();
    mat0->setUniformAttribute("totalNumInstances",
                              std::max(static_cast<int>(num_instances), 1));
    if (!is_static_shape) {
      auto& mat1 = gpu_data.fill_materials[1];
      auto& pa1 = gpu_data.fill_primitive_assemblies[1];

      if (hasProjection()) {
        mat1->setViewportAttributes(0, 0, ctx_.getWidth(), ctx_.getHeight());
      }

      setUniformAttributes(used_fill_props_const_, *mat1);
      num_instances = pa1->numInstances();
      mat1->setUniformAttribute("totalNumInstances",
                                std::max(static_cast<int>(num_instances), 1));
    }
  }

  if (gpu_data.graphics_pipelines[kStroke0]) {
    auto& mat0 = gpu_data.stroke_materials[0];
    auto& pa0 = gpu_data.stroke_primitive_assemblies[0];
    setUniformAttributes(used_stroke_props_const_, *mat0);
    auto const num_instances = pa0->numInstances();
    mat0->setUniformAttribute("totalNumInstances",
                              std::max(static_cast<int>(num_instances), 1));

    if (is_static_shape) {
      mat0->setViewportAttributes(0, 0, ctx_.getWidth(), ctx_.getHeight());
    } else {
      auto& mat1 = gpu_data.stroke_materials[1];
      auto& pa1 = gpu_data.stroke_primitive_assemblies[1];

      setUniformAttributes(used_stroke_props_const_, *mat1);
      auto const num_instances = pa1->numInstances();
      mat1->setUniformAttribute("totalNumInstances",
                                std::max(static_cast<int>(num_instances), 1));

      mat1->setViewportAttributes(0, 0, ctx_.getWidth(), ctx_.getHeight());
    }
  }

  // update prop compression bits again for the case where only the compression changes
  updateGeoPropInfoAndPropCompressionBits({kPOINT});

  updateSlabAddressTableAndPropCompressionBitsUniforms(gpu_data);
}

bool SymbolMark::draw(const gfx::DeviceContext& device_ctx,
                      const MarkPerGpuData& mark_gpu_data,
                      gfx::Framebuffer& framebuffer,
                      const int accumulator_index) {
  auto const gpu_id = device_ctx.getGpuId();
  auto const is_static_shape = isEnumExplicitlyDefined(shape_, gpu_id);

  RENDER_LOG_SCOPE_P(gpu_id);

  static constexpr std::string_view kCurrShapeTypeName{"currShapeType"};

  // NOTE: shader should have been updated before calling this

  auto const& root_gpu_data = mark_gpu_data.getRootPerGpuData();
  auto& cmd_list = root_gpu_data.getCommandList();
  auto& render_pass = root_gpu_data.getCommonRenderPass(
      CommonRenderPassType::kAllAttachments, needsMultisampleEnabled());

  doManualClear(root_gpu_data, accumulator_index, framebuffer);

  auto const& instanced_ibo =
      static_cast<gfx::IndexBuffer&>(*mark_gpu_data.instanced_geom_ibo.get());

  // update all the descriptor sets we're going to use
  if (mark_gpu_data.graphics_pipelines[kFill0]) {
    mark_gpu_data.fill_materials[0]->updateDescriptorSets();
    if (!is_static_shape) {
      mark_gpu_data.fill_materials[1]->updateDescriptorSets();
    }
  }
  if (mark_gpu_data.graphics_pipelines[kStroke0]) {
    mark_gpu_data.stroke_materials[0]->updateDescriptorSets();
    if (!is_static_shape) {
      mark_gpu_data.stroke_materials[1]->updateDescriptorSets();
    }
  }

  // now draw symbols, starting with fill
  if (mark_gpu_data.graphics_pipelines[kFill0]) {
    auto& pa0 = mark_gpu_data.fill_primitive_assemblies[0];
    auto num_instances = pa0->numInstances();

    cmd_list.pushLabel("Symbol Fill");
    cmd_list.beginRenderPass(render_pass, framebuffer);

    if (is_static_shape) {
      // draw any explicit shape
      auto idx = shape_.getUniformValue<int>();
      cmd_list.setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[kFill0],
                                     kCurrShapeTypeName,
                                     ShaderStageBits::kVertex,
                                     idx);
      cmd_list.drawIndexed(*mark_gpu_data.graphics_pipelines[kFill0],
                           mark_gpu_data.vertex_buffer_refs_cache,
                           instanced_ibo,
                           geom_fill_data[idx].index_count,
                           geom_fill_data[idx].first_index,
                           num_instances);
    } else {
      // draw any circles
      auto const circle_idx = static_cast<int>(SymbolShapeType::kCircle);
      cmd_list.setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[kFill0],
                                     kCurrShapeTypeName,
                                     ShaderStageBits::kVertex,
                                     0);
      cmd_list.drawIndexed(*mark_gpu_data.graphics_pipelines[kFill0],
                           mark_gpu_data.vertex_buffer_refs_cache,
                           instanced_ibo,
                           geom_fill_data[circle_idx].index_count,
                           geom_fill_data[circle_idx].first_index,
                           num_instances);

      // draw any other shapes
      auto num_instances = mark_gpu_data.fill_primitive_assemblies[1]->numInstances();
      auto& pipeline = *mark_gpu_data.graphics_pipelines[kFill1];
      for (int i = 0; i < static_cast<int>(SymbolShapeType::kCOUNT); ++i) {
        if (i == circle_idx) {
          continue;
        }
        cmd_list.setPushConstantUInt32(
            pipeline, kCurrShapeTypeName, ShaderStageBits::kVertex, i);
        cmd_list.drawIndexed(pipeline,
                             mark_gpu_data.vertex_buffer_refs_cache,
                             instanced_ibo,
                             geom_fill_data[i].index_count,
                             geom_fill_data[i].first_index,
                             num_instances);
      }
    }
    cmd_list.endRenderPass().popLabel().flush(
        "SymbolMark Fill", gfx::CommandList::SubmitType::kImmediateReturn);
  }

  // now draw outlines
  if (mark_gpu_data.graphics_pipelines[kStroke0]) {
    auto const num_instances =
        mark_gpu_data.stroke_primitive_assemblies[0]->numInstances();

    cmd_list.pushLabel("Symbol Stroke");
    cmd_list.beginRenderPass(render_pass, framebuffer);

    if (is_static_shape) {
      auto idx = shape_.getUniformValue<int>();
      if (idx == static_cast<int>(SymbolShapeType::kCircle)) {
        // draw circles
        cmd_list.setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[kStroke0],
                                       kCurrShapeTypeName,
                                       ShaderStageBits::kVertex,
                                       0);
        cmd_list.drawIndexed(*mark_gpu_data.graphics_pipelines[kStroke0],
                             mark_gpu_data.vertex_buffer_refs_cache,
                             instanced_ibo,
                             geom_fill_data[idx].index_count,
                             geom_fill_data[idx].first_index,
                             num_instances);
      } else {
        // draw non-circles
        cmd_list.setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[kStroke0],
                                       kCurrShapeTypeName,
                                       ShaderStageBits::kVertex,
                                       idx);
        cmd_list.drawVertices(*mark_gpu_data.graphics_pipelines[kStroke0],
                              mark_gpu_data.vertex_buffer_refs_cache,
                              geom_stroke_data[idx].vertex_count,
                              geom_stroke_data[idx].first_vertex,
                              num_instances);
      }
    } else {
      // TODO(croot): if a scale is being used, just traverse the shapes that are actually
      // used in the scale.

      // TODO(croot): do a pre-compute pass that will A) compute all the scales first to
      // avoid duplicating efforts in multiple passes and B) possibly sort by shape type
      // and draw instanced vertex buffers for each sorted shape (with associated instance
      // count). The latter sort and possibly shape counts could be done in sql if we had
      // a structured API

      // draw any circles
      auto const circle_idx = static_cast<uint32_t>(SymbolShapeType::kCircle);
      cmd_list.setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[kStroke0],
                                     kCurrShapeTypeName,
                                     gfx::ShaderStageBits::kVertex,
                                     0);
      cmd_list.drawIndexed(*mark_gpu_data.graphics_pipelines[kStroke0],
                           mark_gpu_data.vertex_buffer_refs_cache,
                           instanced_ibo,
                           geom_fill_data[circle_idx].index_count,
                           geom_fill_data[circle_idx].first_index,
                           num_instances);

      // draw any non-circles
      auto const num_instances =
          mark_gpu_data.stroke_primitive_assemblies[1]->numInstances();
      for (uint32_t i = 0; i < static_cast<uint32_t>(SymbolShapeType::kCOUNT); ++i) {
        if (i == circle_idx) {
          continue;
        }
        cmd_list.setPushConstantUInt32(*mark_gpu_data.graphics_pipelines[kStroke1],
                                       kCurrShapeTypeName,
                                       gfx::ShaderStageBits::kVertex,
                                       i);
        cmd_list.drawVertices(*mark_gpu_data.graphics_pipelines[kStroke1],
                              mark_gpu_data.vertex_buffer_refs_cache,
                              geom_stroke_data[i].vertex_count,
                              geom_stroke_data[i].first_vertex,
                              num_instances);
      }
    }

    cmd_list.endRenderPass().popLabel().flush(
        "SymbolMark Stroke", gfx::CommandList::SubmitType::kImmediateReturn);
  }

  return true;
}

std::vector<CoordAttrInfo2d> SymbolMark::getCoordPropAttrInfos() const {
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

SymbolMark::operator std::string() const {
  return "SymbolMark " + std::string(ctx_.getRenderSessionKey());
}

}  // namespace QueryRenderer
