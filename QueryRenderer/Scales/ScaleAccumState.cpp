/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/ScaleAccumState.h"

#include <boost/algorithm/string/join.hpp>

#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/Scales/BaseScaleRef.h"
#include "QueryRenderer/Scales/Utils.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

using ::gfx::ShaderManager;
using ::gfx::ShaderStage;
using ::gfx::TypeGLSLShPtr;

TypeGLSLShPtr get_percent_type_GLSL(const AnyDataType* pct_cat_val) {
  if (pct_cat_val) {
    switch (pct_cat_val->getType()) {
      case QueryDataType::UINT:
        return TypeToQueryDataTypeSelector<
            QueryDataTypeSelector<QueryDataType::UINT>::type>::getTypeGLSLPtr();
        break;
      case QueryDataType::INT:
        return TypeToQueryDataTypeSelector<
            QueryDataTypeSelector<QueryDataType::INT>::type>::getTypeGLSLPtr();
        break;
      case QueryDataType::FLOAT:
        return TypeToQueryDataTypeSelector<
            QueryDataTypeSelector<QueryDataType::FLOAT>::type>::getTypeGLSLPtr();
        break;
      case QueryDataType::DOUBLE:
        return TypeToQueryDataTypeSelector<
            QueryDataTypeSelector<QueryDataType::DOUBLE>::type>::getTypeGLSLPtr();
        break;
      case QueryDataType::UINT64:
        return TypeToQueryDataTypeSelector<
            QueryDataTypeSelector<QueryDataType::UINT64>::type>::getTypeGLSLPtr();
        break;
      case QueryDataType::INT64:
        return TypeToQueryDataTypeSelector<
            QueryDataTypeSelector<QueryDataType::INT64>::type>::getTypeGLSLPtr();
        break;
      default:
        THROW_RUNTIME_EX("A percent accumulation scale with a category value of type " +
                         to_string(pct_cat_val->getType()) + " is not supported.");
    }
  }
  return nullptr;
}

// TODO(scb): max_textures seems quite arbitrary, and needs to handled at a higher level
// in conjunction with device introspection.
const uint32_t ScaleAccumState::max_textures = 30;

uint32_t ScaleAccumState::convertNumTexturesToNumVals(const uint32_t num_textures,
                                                      const AccumulatorType accum_type) {
  RUNTIME_EX_ASSERT(
      accum_type != AccumulatorType::kDensity && accum_type != AccumulatorType::kPct,
      "Cannot determine the number of accumulator values from a density/pct accumulator "
      "type.");
  return num_textures * 2;
}

std::string ScaleAccumState::getScaleShaderTemplateOverride() const {
  if (accumulator_type_ == AccumulatorType::kPct) {
    return std::string("Scales/accumulatorScalePct_1stPass.vert");
  } else {
    return std::string();
  }
}

ScaleShaderUpdateFlags ScaleAccumState::updateScaleShaderBuilder(
    const BaseScaleRef* scale_ref,
    gfx::ShaderManager::Builder& builder) const {
  ScaleShaderUpdateFlags rtn = ScaleShaderUpdateFlags::kNone;
  if (accumulator_type_ == AccumulatorType::kPct) {
    bool use_ref_pct_cat =
        (scale_ref != nullptr) && (scale_ref->getPctCatValPtr() != nullptr);
    const auto pct_cat_val =
        use_ref_pct_cat ? scale_ref->getPctCatValPtr() : pct_cat_val_.get();
    const auto pct_margin_val =
        use_ref_pct_cat ? scale_ref->getPctMarginValPtr() : pct_margin_val_.get();

    CHECK(pct_cat_val &&
          (!pct_margin_val || (pct_cat_val->getType() == pct_margin_val->getType())));

    auto glsl_type = get_percent_type_GLSL(pct_cat_val);
    builder.replaceFirstTag("domainType", glsl_type->declString());
    builder.replaceAllTags("domainTypeEnum", glsl_type->enumString());
    rtn |= (ScaleShaderUpdateFlags::kDomain | ScaleShaderUpdateFlags::kNumDomains |
            ScaleShaderUpdateFlags::kNumRanges);
  }
  return rtn;
}

ScaleAccumState::ScaleAccumState(BaseScale& parent_scale,
                                 QueryRendererContext& render_context)
    : parent_scale_{parent_scale}
    , accumulator_type_{AccumulatorType::kUndefined}
    , render_context_{render_context}
    , type_changed_{false}
    , scale_props_changed_{false}
    , num_values_{0}
    , num_values_changed_{false}
    , num_textures_changed_{false}
    , is_shader_dirty_{false}
    , num_min_std_dev_{0}
    , min_density_{0}
    , do_find_min_density_{false}
    , num_max_std_dev_{0}
    , max_density_{0}
    , do_find_max_density_{false}
    , do_find_std_dev_{false}
    , pct_accum_changed_{false} {}

bool ScaleAccumState::supportsDomainCoercion() const {
  return (accumulator_type_ != AccumulatorType::kDensity) &&
         (accumulator_type_ != AccumulatorType::kPct);
}

bool ScaleAccumState::supportsCustomNullValue() const {
  return (accumulator_type_ != AccumulatorType::kDensity) &&
         (accumulator_type_ != AccumulatorType::kPct);
}

bool ScaleAccumState::updateFromJSONObj(const JSONLocation& json_loc) {
  // NOTE: obj should be a JSON object by the time it reaches here
  CHECK(json_loc.isValid() && json_loc.isObject())
      << RapidJSONUtils::getPointerPath(json_loc.getPathRef());

  auto type = getScaleAccumulatorTypeFromJSONObj(json_loc);

  CHECK(type != AccumulatorType::kUndefined);

  bool density_accum_changed = false;
  if (type == AccumulatorType::kDensity) {
    bool orig_find_min_density = do_find_min_density_;
    bool orig_find_max_density = do_find_max_density_;
    bool orig_find_std_dev = do_find_std_dev_;

    do_find_std_dev_ = false;
    num_min_std_dev_ = 0;
    num_max_std_dev_ = 0;

    auto density_validator = [](const JSONLocation& json_loc,
                                const char property_name[],
                                uint32_t& density_val,
                                uint8_t& stddev_num,
                                bool& auto_find_density,
                                const std::array<std::string, 3>& valid_strs) {
      constexpr int kMaxIdx = 0;
      constexpr int kStdIdx1 = 1;
      constexpr int kStdIdx2 = 2;
      constexpr char kStdStr1[] = "1st";
      constexpr char kStdStr2[] = "2nd";

      bool use_std_dev = false;
      const auto density_loc = json_loc.getMember(property_name);
      RUNTIME_EX_ASSERT(
          density_loc.isValid(),
          RapidJSONUtils::createJsonParseError(
              json_loc,
              "A \"" + std::string(property_name) +
                  "\" attribute is required for density-based accumulations"));

      const auto item_type = RapidJSONUtils::getDataTypeFromJSONObj(density_loc, true);
      RUNTIME_EX_ASSERT(
          item_type == QueryDataType::UINT || item_type == QueryDataType::INT ||
              item_type == QueryDataType::STRING,
          RapidJSONUtils::createJsonParseError(
              density_loc,
              "Property must be an integer or one of the following strings: " +
                  boost::algorithm::join(valid_strs, ", ") + "."));

      if (item_type == QueryDataType::STRING) {
        const auto val = makeLowerCase(density_loc.getString());
        RUNTIME_EX_ASSERT(
            val == makeLowerCase(valid_strs[kMaxIdx]) ||
                (use_std_dev = (val == makeLowerCase(valid_strs[kStdIdx1]))) ||
                (use_std_dev = (val == makeLowerCase(valid_strs[kStdIdx2]))),
            RapidJSONUtils::createJsonParseError(
                density_loc,
                "Property must be an integer or one of the following strings: " +
                    boost::algorithm::join(valid_strs, ", ") + "."));

        auto_find_density = true;
        if (use_std_dev) {
          if (val.find(kStdStr1) != std::string::npos) {
            stddev_num = 1;
          } else if (val.find(kStdStr2) != std::string::npos) {
            stddev_num = 2;
          } else {
            CHECK(false) << val;
          }
        }
      } else {
        auto_find_density = false;
        if (item_type == QueryDataType::INT) {
          RUNTIME_EX_ASSERT(density_loc.getInt() >= 0,
                            RapidJSONUtils::createJsonParseError(
                                density_loc, "Property must be an integer > 0."));
          density_val = static_cast<uint32_t>(density_loc.getInt());
        } else {
          density_val = static_cast<uint32_t>(density_loc.getUint());
        }
      }

      return use_std_dev;
    };

    do_find_std_dev_ = density_validator(json_loc,
                                         JSONSchema_v1::Scales::kDensityMaxProp,
                                         max_density_,
                                         num_max_std_dev_,
                                         do_find_max_density_,
                                         {"max", "1stStdDev", "2ndStdDev"});

    do_find_std_dev_ = density_validator(json_loc,
                                         JSONSchema_v1::Scales::kDensityMinProp,
                                         min_density_,
                                         num_min_std_dev_,
                                         do_find_min_density_,
                                         {"min", "-1stStdDev", "-2ndStdDev"}) ||
                       do_find_std_dev_;

    density_accum_changed = (orig_find_min_density != do_find_min_density_) ||
                            (orig_find_max_density != do_find_max_density_) ||
                            (orig_find_std_dev != do_find_std_dev_);

    if (density_accum_changed) {
      setFindDensityExtentsChanged();
    }
  } else if (type == AccumulatorType::kPct) {
    const auto pctcat_loc = json_loc.getMember(JSONSchema_v1::Scales::kPctCatProp);
    RUNTIME_EX_ASSERT(pctcat_loc.isValid(),
                      RapidJSONUtils::createJsonParseError(
                          json_loc,
                          "A \"" + std::string(JSONSchema_v1::Scales::kPctCatProp) +
                              "\" attribute is missing which is required for "
                              "percent-based accumulations"));

    if (!pct_cat_val_) {
      pct_cat_val_ = std::make_unique<AnyDataType>();
    }
    const auto num_pct_cat_val = RapidJSONUtils::getAnyDataFromJSONObj(pctcat_loc, true);
    if (num_pct_cat_val != (*pct_cat_val_)) {
      pct_accum_changed_ = true;
      (*pct_cat_val_) = num_pct_cat_val;
    }
    const auto pct_cat_type = pct_cat_val_->getType();
    JSONLocation pctmargin_loc;
    if (isArithmeticQueryDataType(pct_cat_type) &&
        (pctmargin_loc = json_loc.getMember(JSONSchema_v1::Scales::kPctCatMarginProp))
            .isValid()) {
      RUNTIME_EX_ASSERT(
          pctmargin_loc.isNumber(),
          RapidJSONUtils::createJsonParseError(
              pctmargin_loc,
              "Property must be a numeric value for percent-based accumulations"));

      if (!pct_margin_val_) {
        pct_margin_val_ = std::make_unique<AnyDataType>();
      }
      const auto new_pct_margin =
          RapidJSONUtils::getAnyDataFromJSONObj(pctmargin_loc, false);
      if (new_pct_margin != (*pct_margin_val_)) {
        pct_accum_changed_ = true;
        (*pct_margin_val_) = new_pct_margin;
      }

      auto main_type = QueryDataType::UINT;
      RUNTIME_EX_ASSERT(
          RapidJSONUtils::getHigherOrderDataType(
              main_type, *pct_cat_val_, *pct_margin_val_),
          RapidJSONUtils::createJsonParseError(
              pctmargin_loc,
              "The percentage accumulation scale has a \"" +
                  std::string(JSONSchema_v1::Scales::kPctCatProp) +
                  "\" property of type " + to_string(pct_cat_val_->getType()) +
                  " which is not compatible with a \"" +
                  std::string(JSONSchema_v1::Scales::kPctCatMarginProp) +
                  "\" property of type " + to_string(pct_margin_val_->getType()) + "."));
      if (main_type != pct_cat_type) {
        pct_accum_changed_ = true;
        pct_cat_val_->convertToType(main_type);
      }

      if (main_type != pct_margin_val_->getType()) {
        pct_accum_changed_ = true;
        pct_margin_val_->convertToType(main_type);
      }
    } else {
      if (pct_margin_val_) {
        pct_accum_changed_ = true;
      }
      pct_margin_val_.reset();
    }
  }

  RUNTIME_EX_ASSERT(
      (parent_scale_.getValidAccumTypeMask() & type) != AccumulatorType::kUndefined,
      RapidJSONUtils::createJsonParseError(
          json_loc,
          "Scale of type " + to_string(accumulator_type_) +
              " does not support an accumulator of type " + to_string(type) + "."));

  if (type != AccumulatorType::kPct) {
    pct_cat_val_.reset();
    pct_margin_val_.reset();
    pct_accum_changed_ = (accumulator_type_ == AccumulatorType::kPct);
  } else if (type != AccumulatorType::kDensity) {
    accum_stats_.reset();
  }

  CHECK_EQ(type_changed_, false);
  if (type != accumulator_type_) {
    setTypeChanged();
    accumulator_type_ = type;
  }

  return type_changed_ || pct_accum_changed_;
}

void ScaleAccumState::postScaleUpdateFromJSONObj(const bool scale_props_changed) {
  if (scale_props_changed) {
    setScalePropsChanged();
  }
  setNumVals(parent_scale_.getNumValuesForAccumulation());
}

void ScaleAccumState::resetChangedFlags() {
  type_changed_ = false;
  scale_props_changed_ = false;
  num_values_changed_ = false;
  num_textures_changed_ = false;
  pct_accum_changed_ = false;
}

void ScaleAccumState::toJSON(rapidjson::Value& obj,
                             rapidjson::Document::AllocatorType& allocator) const {
  obj.AddMember("accumulator",
                RapidJSONUtils::valToJSON(to_string(accumulator_type_), allocator),
                allocator);
  switch (accumulator_type_) {
    case AccumulatorType::kPct: {
      rapidjson::Value pct;
      if (pct_cat_val_) {
        pct = RapidJSONUtils::valToJSON(*pct_cat_val_, allocator);
      } else {
        pct = rapidjson::Value("undefined");
      }
      obj.AddMember("pctCategory", pct, allocator);
      if (pct_margin_val_) {
        obj.AddMember("pctCategoryMargin",
                      RapidJSONUtils::valToJSON(*pct_margin_val_, allocator),
                      allocator);
      }
      break;
    }
    case AccumulatorType::kDensity: {
      rapidjson::Value densityv;
      if (!do_find_min_density_) {
        densityv = min_density_;
      } else {
        if (accum_stats_) {
          if (num_min_std_dev_) {
            densityv =
                std::max(double(accum_stats_->min),
                         accum_stats_->avg - num_min_std_dev_ * accum_stats_->stddev);
          } else {
            densityv = accum_stats_->min;
          }
        } else {
          densityv = "undefined";
        }
      }
      obj.AddMember("minDensityCnt", densityv, allocator);
      // TODO(croot): make sure this doesn't mess up the min value added above
      if (!do_find_max_density_) {
        densityv = max_density_;
      } else {
        if (accum_stats_) {
          if (num_max_std_dev_) {
            densityv =
                std::min(double(accum_stats_->max),
                         accum_stats_->avg + num_max_std_dev_ * accum_stats_->stddev);
          } else {
            densityv = accum_stats_->max;
          }
        } else {
          densityv = "undefined";
        }
      }
      obj.AddMember("maxDensityCnt", densityv, allocator);
      break;
    }
    case AccumulatorType::kMin:
    case AccumulatorType::kMax:
    case AccumulatorType::kBlend:
    case AccumulatorType::kUndefined:
    case AccumulatorType::kAll:
      // noop
      break;
  }
}

//
// PctAccum
//
TypeGLSLShPtr ScaleAccumState::getPercentTypeGLSL() const {
  return get_percent_type_GLSL(pct_cat_val_.get());
}
const AnyDataType* ScaleAccumState::getPercentCategoryVal() const {
  return pct_cat_val_.get();
}
const AnyDataType* ScaleAccumState::getPercentMargin() const {
  return pct_margin_val_.get();
}
std::string ScaleAccumState::getPercentCategoryUniformName() const {
  return "uPctVal_" + parent_scale_.getName();
}
std::string ScaleAccumState::getPercentMarginUniformName() const {
  return "uPctMargin_" + parent_scale_.getName();
}

//
// Values and Textures
//
namespace {
size_t convertNumValsToNumTextures(const size_t num_vals, const AccumulatorType type) {
  switch (type) {
    case AccumulatorType::kDensity:
      return 1;
    case AccumulatorType::kPct:
      return 2;
    case AccumulatorType::kBlend:
    case AccumulatorType::kMin:
    case AccumulatorType::kMax:
      return (num_vals + 1) / 2;
    case AccumulatorType::kUndefined:
      return 0;
    case AccumulatorType::kAll:
      CHECK(false);
  }
  return 0;
}
}  // namespace

void ScaleAccumState::setNumVals(const uint32_t num_vals) {
  RUNTIME_EX_ASSERT(num_vals > 0, toString() + ", need to have at least 1 accumulator.");
  if (num_vals != num_values_) {
    setNumValsChanged();

    size_t new_num_textures = convertNumValsToNumTextures(num_values_, accumulator_type_);
    RUNTIME_EX_ASSERT(new_num_textures <= max_textures,
                      toString() +
                          " There are too many accumulator values to do a render-based "
                          "accumation as it requires too many textures. There are " +
                          std::to_string(num_vals) +
                          " values requested for accumulation requiring " +
                          std::to_string(new_num_textures) + " but there's a limit of " +
                          std::to_string(max_textures) + " textures");
    num_values_ = num_vals;
    num_textures_changed_ = (num_textures_ != new_num_textures);
    num_textures_ = new_num_textures;
  }
}

//
// Subroutines
//
std::string ScaleAccumState::getSubroutineName() const {
  // TODO(croot): should we verify that subroutines of the
  // same name don't exist? I'm choosing not to as I'm
  // assuming the shader compilation would fail if there are two
  // subroutines of the same name.
  switch (accumulator_type_) {
    case AccumulatorType::kMin:
    case AccumulatorType::kMax:
    case AccumulatorType::kBlend:
      return std::string("minMaxBlendAccumulate");
      break;
    case AccumulatorType::kDensity:
      return std::string("densityAccumulate");
      break;
    case AccumulatorType::kPct: {
      return std::string("pctAccumulate");
      break;
    }
    default:
      THROW_RUNTIME_EX("Accumulator type " + to_string(accumulator_type_) +
                       " is not currently supported for mark rendering.");
      break;
  }
  return std::string();
}

//
// Rendering
//

// Get the number of textures to use per-gpu in the first pass rendering.
uint32_t ScaleAccumState::getNumTextures() {
  return convertNumValsToNumTextures(num_values_, accumulator_type_);
}

gfx::ShaderManager::BuilderUqPtr ScaleAccumState::get1stPassFragSubBuilder() {
  auto sub_builder = render_context_.getShaderManager().createBuilder(
      "Marks/mainTemplate_Accumulation.glsl");
  sub_builder->replaceAllTags("name", parent_scale_.getName());
  sub_builder->replaceAllTags("numAccumTextures", std::to_string(getNumTextures()));
  return sub_builder;
}

void ScaleAccumState::setAccumStats(RenderAccumStatsUqPtr&& accum_stats) {
  accum_stats_ = std::move(accum_stats);
}

void ScaleAccumState::buildSubroutineBindings(ShaderManager::Builder& builder) {
  builder.addSubroutineBinding("getAccumulatedColor", "", true);

  builder.addSubroutineBinding("getMinDensity", "getUniformMinDensity", true);
  builder.addSubroutineBinding("getMaxDensity", "getUniformMaxDensity", true);
  builder.addSubroutineBinding("calcMeanStdDev", "getEmptyMeanStdDev", true);

  if (do_find_min_density_ || do_find_max_density_ || do_find_std_dev_) {
    if (do_find_min_density_) {
      if (num_min_std_dev_ > 0) {
        builder.addSubroutineBinding("getMinDensity", "getStdDevMinDensity", true);
      } else {
        builder.addSubroutineBinding("getMinDensity", "getTextureMinDensity", true);
      }
    }

    if (do_find_max_density_) {
      if (num_max_std_dev_ > 0) {
        builder.addSubroutineBinding("getMaxDensity", "getStdDevMaxDensity", true);
      } else {
        builder.addSubroutineBinding("getMaxDensity", "getTextureMaxDensity", true);
      }
    }

    if (do_find_std_dev_) {
      builder.addSubroutineBinding("calcMeanStdDev", "getTextureMeanStdDev", true);
    }
  }

  std::string accum_function_name;
  switch (accumulator_type_) {
    case AccumulatorType::kMin:
      accum_function_name = "getMinAccumulatedColor";
      break;
    case AccumulatorType::kMax:
      accum_function_name = "getMaxAccumulatedColor";
      break;
    case AccumulatorType::kBlend:
      accum_function_name = "getBlendAccumulatedColor";
      break;
    case AccumulatorType::kDensity:
      accum_function_name = "getDensityAccumulatedColor";
      break;
    case AccumulatorType::kPct:
      accum_function_name = "getPctAccumulatedColor";
      break;
    default:
      THROW_RUNTIME_EX("Accumulator type " + to_string(accumulator_type_) +
                       " is currently unsupported for rendering");
  }
  builder.addSubroutineBinding("getAccumulatedColor", accum_function_name, true);

  parent_scale_.buildSubroutineBindings(
      builder,
      "_ACCUMULATION",  // TODO(croot): expose as a global
      true);
}

gfx::ShaderManager::BuilderUqPtrVector ScaleAccumState::get2ndPassShaderBuilders(
    uint32_t num_textures) {
  auto& shader_mgr = render_context_.getShaderManager();
  auto builders = shader_mgr.createBuilderVector(
      {{"Rendering/fullScreenTriangle.vert"}, {"Scales/accumulatorScale_2ndPass.frag"}});

  builders[1]->replaceAllTags("name", parent_scale_.getName());
  // TODO(scb): resource bindings can be more dynamic (uniforms etc)
  builders[1]->replaceAllTags("numAccumColors", std::to_string(num_values_));
  builders[1]->replaceAllTags("numAccumTextures", std::to_string(num_textures));

  std::string suffix = "_ACCUMULATION";
  builders[1]->replaceFunctionWithSubBuilder(
      "getDensityColor", parent_scale_.getShaderSubBuilder(nullptr, suffix, false), true);

  const auto func_signature = parent_scale_.getScaleGLSLFuncName(suffix, false) + "(pct)";
  builders[1]->replaceAll("getDensityColor(pct)", func_signature);
  builders[1]->replaceAll("getPctColor(pct)", func_signature);

  builders[1]->replaceFirstTag("useExtentsSSBO", getDoFindExtents() ? "1" : "0");

  // handle common subroutines
  buildSubroutineBindings(*builders[1]);

  return builders;
}

void ScaleAccumState::bindScaleSubroutines(ShaderManager::Builder& builder) {
  if (builder.getShaderStage() == ShaderStage::kFragment) {
    builder.addSubroutineBinding("accumulate", getSubroutineName(), true);
  }
}

std::string ScaleAccumState::toString() const {
  return parent_scale_.getName();
}

bool ScaleAccumState::getDoFindExtents() const {
  return (accumulator_type_ == AccumulatorType::kDensity) &&
         (do_find_min_density_ || do_find_max_density_ || do_find_std_dev_);
}

}  // namespace QueryRenderer
