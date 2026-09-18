/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/BaseScale.h"

#include "GfxDriver/RenderLogger.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "QueryRenderer/Scales/ScaleRef.h"
#include "QueryRenderer/Scales/Utils.h"

namespace QueryRenderer {

BaseScale::BaseScale(const JSONLocation& json_loc,
                     QueryRendererContext& ctx,
                     const std::string& name,
                     const ScaleType type)
    : JSONRefObject(ctx, RefType::kScale, name, json_loc.getPathRef())
    , type_{type}
    , dr_changed_flags_{ScaleDRChangedFlags::kNone}
    , accumulator_changed_{false}
    , marked_for_deletion_{false} {}

BaseScale::~BaseScale() {}

void BaseScale::setScaleImpl(ScaleImplUqPtr impl) {
  CHECK(impl_ == nullptr);
  impl_ = std::move(impl);
}

void BaseScale::setDomainData(ScaleDomainRangeDataUqPtr domain_data) {
  domain_data_ = std::move(domain_data);
  domain_data_->setParentScale(this);
  domain_type_glsl_ = domain_data_->getTypeGLSL();
}
void BaseScale::setRangeData(ScaleDomainRangeDataUqPtr range_data) {
  range_data_ = std::move(range_data);
  range_data_->setParentScale(this);
  range_type_glsl_ = range_data_->getTypeGLSL();
}

bool BaseScale::updateFromJSONObj(const JSONLocation& json_loc) {
  bool changed = false;
  if (!ctx_.isJSONCacheUpToDate(json_path_, json_loc)) {
    RUNTIME_EX_ASSERT(
        json_loc.isObject(),
        RapidJSONUtils::createJsonParseError(json_loc, "Scale items must be objects."));

    // Update implementation first - allowing validate and convert functions to
    // properly initialize before updating input DomainRangeData objects
    impl_->updateFromJSONObj(json_loc);

    // Update DomainRangeData inputs
    dr_changed_flags_ |= impl_->updateDRDataFromJSONObj(json_loc);
    changed = dr_changed_flags_ != ScaleDRChangedFlags::kNone;

    if (hasDomainDataChanged()) {
      domain_override_data_.reset();
    }

    if (hasRangeDataChanged()) {
      range_override_data_.reset();
    }

    // Final parse cleanup and validation
    if (changed) {
      impl_->postDRDataJSONUpdate(json_loc);
    }

    // TODO(scb): unify validity tracking
    // There needs to be more clarity over changes that trigger uniform updates and
    // changes that may require shader code changes (eg null presence or clamp enable),
    // though technically clamp could be made a uniform or vulkan push constant.

    // Implementation specific null handling
    impl_->initNullValueFromJSONObj(json_loc);

    // Check if any other special properties have changed (eg clamp)
    bool props_changed = impl_->havePropertiesChanged();

    // Update the accumulator (create/validate/destroy)
    // TODO(scb): Ideally this call can be moved out of BaseScale and handled by QRC
    const bool accum_updated = updateAccumulatorFromJSONObj(json_loc);

    changed = changed || accum_updated || props_changed ||
              impl_->haveUniformPropertiesChanged();
    QueryDataType dtype = getDomainDataType();
    RUNTIME_EX_ASSERT(
        getAccumulatorType() != AccumulatorType::kDensity ||
            dtype == QueryDataType::FLOAT || dtype == QueryDataType::DOUBLE,
        RapidJSONUtils::createJsonParseError(
            json_loc,
            "Density accumulator scales must have floats/doubles as its domain "
            "values which are used as percentages of the final accumulation "
            "counts."));

    if (accum_state_ != nullptr) {
      // Let the accumulation render do any processing/validation post json update.
      // Also let the accumulation render know if any properties changed so it can rebuild
      // the second pass shader
      // TODO(croot): Would this be better if it were a pull system rather than a push?
      accum_state_->postScaleUpdateFromJSONObj(props_changed);
    }
  } else if (json_path_ != json_loc.getPathRef()) {
    dr_changed_flags_ = ScaleDRChangedFlags::kNone;
    domain_data_->updateJSONPath(json_loc.getPathRef(), false);
    range_data_->updateJSONPath(json_loc.getPathRef(), false);
  }
  setJSONPathRef(json_loc.getPathRef());

  return changed;
}

const BaseScaleDomainRangeData* BaseScale::getDomainData(const bool get_orig) const {
  if (!get_orig && domain_override_data_.data) {
    return domain_override_data_.data.get();
  }
  return domain_data_.get();
};

const BaseScaleDomainRangeData* BaseScale::getRangeData(const bool get_orig) const {
  if (!get_orig && range_override_data_) {
    return range_override_data_.get();
  }
  return range_data_.get();
};

QueryDataType BaseScale::getDomainDataType() const {
  return domain_data_->getType();
}
QueryDataType BaseScale::getRangeDataType() const {
  return range_data_->getType();
}

gfx::ColorType BaseScale::getRangeColorType() const {
  return impl_->getRangeColorType();
}

QueryDataType BaseScale::getPrimaryDomainDataType() const {
  // FIXME(scb): Called by RenderProperty in this very specific circumstance. This logic
  // really needs to be elsewhere, since it's a coupling between RenderProperty and
  // accumulation, and Scales are caught in the crossfire.
  if (accum_state_ != nullptr && accum_state_->getType() == AccumulatorType::kPct &&
      accum_state_->getPercentCategoryVal()) {
    return accum_state_->getPercentCategoryVal()->getType();
  }

  return getDomainDataType();
}

const gfx::TypeGLSLShPtr& BaseScale::getDomainTypeGLSL(bool get_orig) const {
  RUNTIME_EX_ASSERT(
      domain_type_glsl_ != nullptr,
      std::string(*this) + " getDomainTypeGLSL(): the domain type is uninitialized.");
  if (!get_orig && domain_override_data_.data) {
    return domain_override_data_.data->getTypeGLSL();
  }
  return domain_type_glsl_;
}

const gfx::TypeGLSLShPtr& BaseScale::getRangeTypeGLSL(bool get_orig) const {
  RUNTIME_EX_ASSERT(
      range_type_glsl_ != nullptr,
      std::string(*this) + " getRangeTypeGLSL(): the range type is uninitialized.");
  if (!get_orig && range_override_data_) {
    return range_override_data_->getTypeGLSL();
  }
  return range_type_glsl_;
}

bool BaseScale::hasDataRef() const {
  auto domain = getDomainData(true);
  auto range = getRangeData(true);
  CHECK(domain && range);
  return domain->hasDataRef() || range->hasDataRef();
}

const std::unordered_set<BaseDataTableShPtr> BaseScale::getDataRefs() const {
  std::unordered_set<BaseDataTableShPtr> rtn;
  auto domain = getDomainData(true);
  auto domain_data_ref = domain->getDataRef();
  if (domain_data_ref) {
    rtn.insert(domain_data_ref);
  }
  auto range = getRangeData(true);
  auto range_data_ref = range->getDataRef();
  if (range_data_ref) {
    rtn.insert(range_data_ref);
  }
  return rtn;
}

BaseScale::DomainTypeUniforms BaseScale::getDomainTypeUniforms(
    const std::string& extra_suffix,
    const SQLTypeInfo* sql_type_info) const {
  const auto type =
      sql_type_info ? convertToQueryDataType(*sql_type_info) : getDomainDataType();
  auto rtn = std::make_pair(type, std::unordered_map<std::string, std::any>());
  if (sql_type_info) {
    if (impl_->hasNullValue()) {
      std::any null_val;
      switch (rtn.first) {
        case QueryDataType::INT:
          null_val =
              getNullValueFromTypeInfo<QueryDataTypeSelector<QueryDataType::INT>::type>(
                  *sql_type_info);
          break;
        case QueryDataType::INT64:
          null_val =
              getNullValueFromTypeInfo<QueryDataTypeSelector<QueryDataType::INT64>::type>(
                  *sql_type_info);
          break;
        case QueryDataType::FLOAT:
          null_val =
              getNullValueFromTypeInfo<QueryDataTypeSelector<QueryDataType::FLOAT>::type>(
                  *sql_type_info);
          break;
        case QueryDataType::DOUBLE:
          null_val = getNullValueFromTypeInfo<
              QueryDataTypeSelector<QueryDataType::DOUBLE>::type>(*sql_type_info);
          break;
        case QueryDataType::STRING:  // strings will be int32_t (dict-encoded). So update
                                     // the return type and null value accordingly
          rtn.first = TypeToQueryDataTypeSelector<TypeToQueryDataTypeSelector<
              QueryDataTypeSelector<QueryDataType::STRING>::type>::BufferType>::
              getQueryDataType();
          null_val = getNullValue<TypeToQueryDataTypeSelector<
              QueryDataTypeSelector<QueryDataType::STRING>::type>::BufferType>();
          break;
        default:
          CHECK(false) << "Unsupported sql type conversion from "
                       << sql_type_info->get_type_name() << " to " << type;
      }
      rtn.second.emplace(getNullGLSLAttrName(extra_suffix), null_val);
    }
  }

  impl_->getDomainTypeUniformsInternal(rtn, extra_suffix, sql_type_info);

  return rtn;
}

BaseScale::RangeTypeUniforms BaseScale::getRangeTypeUniforms(
    const std::string& extra_suffix) const {
  return impl_->getRangeTypeUniforms(extra_suffix);
}

// Creates a mangled name for evaluating the scale.
// Examples: "evalQuantitativeScale_pos_x", "evalOrdinalScale_color_color"
std::string BaseScale::getScaleGLSLFuncName(const std::string& extra_suffix,
                                            const bool use_accumulator) {
  std::string scale_name;

  // FIXME(scb): move to accumulator
  if (use_accumulator && getAccumulatorType() == AccumulatorType::kPct) {
    return "evalPctScale_" + name_ + extra_suffix;
  }

  // FIXME(scb): move to impl
  switch (type_) {
    case ScaleType::kLinear:
    case ScaleType::kLog:
    case ScaleType::kPow:
    case ScaleType::kSqrt:
      scale_name = "Quantitative";
      break;
    case ScaleType::kOrdinal:
      scale_name = "Ordinal";
      break;
    case ScaleType::kQuantize:
      scale_name = "Quantize";
      break;
    case ScaleType::kThreshold:
      scale_name = "Threshold";
      break;
    default:
      THROW_RUNTIME_EX(std::string(*this) +
                       " getScaleGLSLFuncName(): scale type is not supported.");
  }

  return "eval" + scale_name + "Scale_" + name_ + extra_suffix;
}

//
// Accumulation rendering
//
AccumulatorType BaseScale::getValidAccumTypeMask() const {
  return impl_->getValidAccumTypeMask();
}

bool BaseScale::supportsAccumNulls() const {
  return impl_->supportsAccumNulls();
}

ScaleAccumState* BaseScale::getAccumState() const {
  return accum_state_.get();
}

ScaleAccumRenderState* BaseScale::getAccumRenderState() const {
  return accum_render_state_.get();
}

AccumulatorType BaseScale::getAccumulatorType() const {
  return accum_state_ != nullptr ? accum_state_->getType() : AccumulatorType::kUndefined;
}

bool BaseScale::hasAccumulator() const {
  return accum_state_ != nullptr;
}

uint32_t BaseScale::getNumValuesForAccumulation() const {
  return impl_->getNumValuesForAccumulation();
}

bool BaseScale::updateAccumulatorFromJSONObj(const JSONLocation& json_loc) {
  auto type = getScaleAccumulatorTypeFromJSONObj(json_loc);
  if (type != AccumulatorType::kUndefined) {
    if (accum_state_ == nullptr) {
      accum_state_ = std::make_unique<ScaleAccumState>(*this, ctx_);
      CHECK(accum_state_);
      accum_render_state_ = std::make_unique<ScaleAccumRenderState>(*accum_state_);
      CHECK(accum_render_state_);
    }
    accumulator_changed_ = accum_state_->updateFromJSONObj(json_loc);
  } else if (accum_state_ != nullptr) {
    // purge existing accumulation renderer
    accum_render_state_ = nullptr;
    accum_state_ = nullptr;
    accumulator_changed_ = true;
  }
  return accumulator_changed_;
}

void BaseScale::bindAccumulatorColors(gfx::Material& material,
                                      const std::string& attr_name) {
  impl_->bindAccumulatorColors(material, attr_name);
}

void BaseScale::bindUniforms(gfx::Material& material,
                             const std::string& extra_suffix,
                             bool use_domain,
                             bool use_range,
                             bool use_accum,
                             const SQLTypeInfo* sql_type_info) {
  RENDER_LOG_SCOPE();
  // configure binding options
  BindOptions bind_options{
      use_domain, use_range, use_accum && (accum_state_ != nullptr), true};
  impl_->modifyBindOptions(bind_options);
  impl_->bindUniforms(material, extra_suffix, bind_options, sql_type_info);
}

void BaseScale::buildSubroutineBindings(gfx::ShaderManager::Builder& builder,
                                        const std::string& extra_suffix,
                                        bool is_accum_final_pass) {
  // Most scales only get integrated into the vertex (or compute, for PPLL) shader, but we
  // can end up here due to the need for accumulation to manipulate the fragment shader,
  // which is handled in ScaleRef (just up the call stack from here). Marks (top of
  // the stack), have to call down here with both vertex and fragment builders as
  // a result. If a scale ever needs to modify the fragment shader as well we'll need
  // to improve this
  auto shader_stage = builder.getShaderStage();
  if (is_accum_final_pass || shader_stage == gfx::ShaderStage::kVertex ||
      shader_stage == gfx::ShaderStage::kMesh ||
      shader_stage == gfx::ShaderStage::kCompute) {
    impl_->bindImplementationSubroutines(builder, extra_suffix, is_accum_final_pass);

    // Handle null value test
    // useNullValue is only false during the second accumulation pass
    const std::string postfix = name_ + extra_suffix;
    // only allow custom null values if
    // - shader defines null value test AND
    // - one has been set AND
    // - not accumulating OR
    // - accumulator allows it AND NOT
    // - final accumulation pass
    if (!is_accum_final_pass && impl_->hasNullValue() &&
        ((accum_state_ == nullptr) || accum_state_->supportsCustomNullValue())) {
      builder.addSubroutineBinding(
          "isNullValFunc_" + postfix, "isNullVal_" + postfix, false);
    } else {
      builder.addSubroutineBinding(
          "isNullValFunc_" + postfix, "isNullValPassThru_" + postfix, false);
    }
  }
}

namespace {
// FIXME(scb): Push these out into the implementations?
static std::string get_scale_shader_template_name(const BaseScale::ScaleShaderType type) {
  std::string rtn;
  switch (type) {
    case BaseScale::ScaleShaderType::kQuantitative:
      return std::string("Scales/quantitativeScaleTemplate.vert");
    case BaseScale::ScaleShaderType::kOrdinal:
      return std::string("Scales/ordinalScaleTemplate.vert");
    case BaseScale::ScaleShaderType::kQuantize:
      return std::string("Scales/quantizeScaleTemplate.vert");
    case BaseScale::ScaleShaderType::kThreshold:
      return std::string("Scales/thresholdScaleTemplate.vert");
    default:
      THROW_RUNTIME_EX("getScaleShaderTemplate(): scale type is not supported.");
  }
}
}  // namespace

// TODO(scb): This code eventually passes through the ShaderBuilder
// which performs all the tag substitutions. When the builder has
// sub-components (multi-string), we should use that instead to keep
// the Scale code in a separate parse tree.

// NOTE: refPtr will be null when called from ScaleAccumState
gfx::ShaderManager::BuilderShPtr BaseScale::getShaderSubBuilder(
    const BaseScaleRef* ref,
    const std::string& extra_suffix,
    bool use_accum) const {
  const auto scale_shader_type = impl_->getShaderType();

  // FIXME(scb): update this
  CHECK(!domain_override_data_.data ||
        domain_override_data_.data->size() == getDomainData()->size());
  RUNTIME_EX_ASSERT(getDomainData()->size() > 0 && getRangeData()->size() > 0,
                    std::string(*this) +
                        " updateShaderBuilder(): domain/range of scale \"" + name_ +
                        "\" has no value.");

  ScaleShaderUpdateFlags updated_flags = ScaleShaderUpdateFlags::kTemplate;

  bool enable_accum = use_accum && hasAccumulator();
  std::string template_name;
  if (enable_accum) {
    template_name = accum_state_->getScaleShaderTemplateOverride();
  }
  if (template_name.empty()) {
    template_name = get_scale_shader_template_name(scale_shader_type);
  }

  // Explicit type so we can cast from unique ptr to shared
  gfx::ShaderManager::BuilderShPtr sub_builder =
      ctx_.getShaderManager().createBuilder(template_name);

  if (enable_accum) {
    updated_flags |= accum_state_->updateScaleShaderBuilder(ref, *sub_builder);
  }

  gfx::ShaderManager::Builder::NameMap name_map;

  if ((ScaleShaderUpdateFlags::kDomain & updated_flags) ==
      ScaleShaderUpdateFlags::kNone) {
    auto glsl_type_to_use =
        ref != nullptr ? ref->getDomainTypeGLSL() : getDomainTypeGLSL(false);
    CHECK(glsl_type_to_use);
    auto glsl_type = glsl_type_to_use->declString();
    name_map.emplace_back("<domainType>", glsl_type);
    name_map.emplace_back("<domainTypeEnum>", glsl_type_to_use->enumString());
  }

  if ((ScaleShaderUpdateFlags::kRange & updated_flags) == ScaleShaderUpdateFlags::kNone) {
    // NOTE(croot): not using the range override data here because it should
    // be the same type as the original range data
    auto glsl_type_to_use =
        ref != nullptr ? ref->getRangeTypeGLSL() : getRangeTypeGLSL(true);
    CHECK(glsl_type_to_use);
    auto glsl_type = glsl_type_to_use->declString();
    name_map.emplace_back("<rangeType>", glsl_type);
    name_map.emplace_back("<rangeTypeEnum>", glsl_type_to_use->enumString());
  }

  if ((ScaleShaderUpdateFlags::kNumDomains & updated_flags) ==
      ScaleShaderUpdateFlags::kNone) {
    name_map.emplace_back("<numDomains>", std::to_string(getDomainData()->size()));
  }

  if ((ScaleShaderUpdateFlags::kNumRanges & updated_flags) ==
      ScaleShaderUpdateFlags::kNone) {
    name_map.emplace_back("<numRanges>", std::to_string(getRangeData()->size()));
  }

  // TODO(scb): global accum flag? scoped to scale? etc.
  name_map.emplace_back("<doAccum>", std::to_string(enable_accum));
  name_map.emplace_back("<name>", name_ + extra_suffix);

  sub_builder->replaceAllMultiple(std::move(name_map));

  impl_->modifyShaderTemplate(*sub_builder);

  return sub_builder;
}

void BaseScale::postJSONUpdate() {
  // unset from update
  // TODO(croot): should scales have an update function much like a mark
  // that's called in QueryRendererContext::update?
  dr_changed_flags_ = ScaleDRChangedFlags::kNone;
  accumulator_changed_ = false;
  if (accum_state_) {
    accum_state_->resetChangedFlags();
  }

  auto domain = getDomainData();
  if (domain) {
    domain->postJSONUpdate();
  }

  auto range = getRangeData();
  if (range) {
    range->postJSONUpdate();
  }
}

void BaseScale::markForDeletion() {
  auto domain = getDomainData();
  if (domain) {
    domain->markForDeletion();
  }
  auto range = getRangeData();
  if (range) {
    range->markForDeletion();
  }
  marked_for_deletion_ = true;
}

void BaseScale::toJSONInternal(rapidjson::Value& obj,
                               rapidjson::Document::AllocatorType& allocator) const {
  obj.AddMember(
      "type", RapidJSONUtils::valToJSON(to_string(type_), allocator), allocator);

  auto domain = getDomainData(true);
  CHECK(domain);
  obj.AddMember("domain", domain->toJSON(allocator), allocator);
  auto range = getRangeData(true);
  CHECK(range);
  obj.AddMember("range", range->toJSON(allocator), allocator);

  if (accum_state_ != nullptr) {
    accum_state_->toJSON(obj, allocator);
  }

  impl_->toJSONInternal(obj, allocator);
}

void BaseScale::setDomainOverride(const ScaleDomainRangeDataShPtr& domain_override,
                                  const QueryDataTableSQL& domain_override_table) {
  domain_override_data_.data = domain_override;
  domain_override_data_.data_table = &domain_override_table;
}

void BaseScale::setRangeOverride(const ScaleDomainRangeDataShPtr& range_override) {
  if (range_override) {
    auto range_data = getRangeData(true);
    CHECK(range_data->getTypeInfo() == range_override->getTypeInfo() &&
          range_data->size() == range_override->size());
  }
  range_override_data_ = range_override;
}

bool BaseScale::hasDomainOverride() const {
  return domain_override_data_.data != nullptr;
}

bool BaseScale::hasRangeOverride() const {
  return range_override_data_ != nullptr;
}

std::string BaseScale::getDomainOverrideTableName() const {
  return domain_override_data_.data_table
             ? domain_override_data_.data_table->getPrimaryTableName()
             : "";
}

bool BaseScale::isShaderDirty() const {
  bool size_changed = ScaleDRChangedFlags::kNone !=
                      (dr_changed_flags_ & (ScaleDRChangedFlags::kDomainSize |
                                            ScaleDRChangedFlags::kRangeSize));
  if (impl_->havePropertiesChanged() || accumulator_changed_ || size_changed) {
    return true;
  }
  if (accum_state_ && accum_state_->hasNumTexturesChanged()) {
    return true;
  }
  return false;
}

void BaseScale::setDRChangedFlags(ScaleDRChangedFlags flags) {
  dr_changed_flags_ |= flags;
  if ((flags & ScaleDRChangedFlags::kDomainSize) != ScaleDRChangedFlags::kNone) {
    impl_->validateDomainRangeSizes(domain_data_->getJSONLocation());
  }
  if ((flags & ScaleDRChangedFlags::kRangeSize) != ScaleDRChangedFlags::kNone) {
    impl_->validateDomainRangeSizes(range_data_->getJSONLocation());
  }
}

std::string BaseScale::printInfo() const {
  return "(name: " + name_ + ") " + std::string(ctx_.getRenderSessionKey());
}

}  // namespace QueryRenderer
