/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cfenv>

#include <boost/algorithm/string/join.hpp>

#include "QueryRenderer/Scales/Scale.h"

namespace QueryRenderer {

template <typename T>
using EnableIfDouble = std::enable_if_t<std::is_same_v<T, double>>;

template <typename T>
using EnableIfNotDouble = std::enable_if_t<!std::is_same_v<T, double>>;

namespace details {
template <typename T, EnableIfDouble<T>* = nullptr>
ConvertFuncT<T> getQuantConvertFunc(const ScaleType scale_type,
                                    const float exponent = 1.0) {
  switch (scale_type) {
    // NOTE: need to do the static_cast<> in order to grab the right overloaded function
    case ScaleType::kLinear:
      return nullptr;
    case ScaleType::kSqrt:
      return static_cast<T (*)(T)>(std::sqrt);
    case ScaleType::kLog:
      return static_cast<T (*)(T)>(std::log);
    case ScaleType::kPow:
      return [exponent](T value) -> T { return std::pow(value, exponent); };
    default:
      CHECK(false) << "Invalid scale type: " << static_cast<int>(scale_type);
  }
  return nullptr;
}

template <typename T, EnableIfNotDouble<T>* = nullptr>
ConvertFuncT<T> getQuantConvertFunc(const ScaleType scale_type,
                                    const float exponent = 1.0) {
  return nullptr;
}

// WIP(scb): experimental builder code, do not use yet
#if 0
static void validateSqrt(const JSONLocation& json_loc, const double& val) {
  RUNTIME_EX_ASSERT(
      val >= 0,
      RapidJSONUtils::createJsonParseError(
          json_loc,
          std::to_string(val) + " is < 0. sqrt scales only work with positive values"));
}

static void validateLog(const JSONLocation& json_loc, const double& val) {
  RUNTIME_EX_ASSERT(
      val > 0,
      RapidJSONUtils::createJsonParseError(
          json_loc,
          std::to_string(val) + " is <= 0. Log scales only work with positive values."));
}

static void validatePow(const JSONLocation& json_loc,
                        const double& val,
                        const double _exponent) {
  if (math_errhandling & MATH_ERREXCEPT) {
    std::feclearexcept(FE_ALL_EXCEPT);
    std::pow(val, _exponent);
    RUNTIME_EX_ASSERT(
        !std::fetestexcept(FE_DIVBYZERO),
        RapidJSONUtils::createJsonParseError(json_loc,
                                             "pow(" + std::to_string(val) + ", " +
                                                 std::to_string(_exponent) +
                                                 ") results in a divide-by-zero."));
    RUNTIME_EX_ASSERT(!std::fetestexcept(FE_INVALID),
                      RapidJSONUtils::createJsonParseError(
                          json_loc,
                          "pow(" + std::to_string(val) + ", " +
                              std::to_string(_exponent) + ") cannot be evaluated."));
    RUNTIME_EX_ASSERT(
        !std::fetestexcept(FE_OVERFLOW),
        RapidJSONUtils::createJsonParseError(json_loc,
                                             "pow(" + std::to_string(val) + ", " +
                                                 std::to_string(_exponent) +
                                                 ") results in an overflow error."));
    RUNTIME_EX_ASSERT(
        !std::fetestexcept(FE_UNDERFLOW),
        RapidJSONUtils::createJsonParseError(json_loc,
                                             "pow(" + std::to_string(val) + ", " +
                                                 std::to_string(_exponent) +
                                                 ") results in an underflow error."));
  } else {
    errno = 0;
    std::pow(val, _exponent);
    RUNTIME_EX_ASSERT(
        errno != ERANGE,
        RapidJSONUtils::createJsonParseError(json_loc,
                                             "pow(" + std::to_string(val) + ", " +
                                                 std::to_string(_exponent) +
                                                 ") results in a divide-by-zero or "
                                                 "overflow or underflow error."));
    RUNTIME_EX_ASSERT(!std::fetestexcept(EDOM),
                      RapidJSONUtils::createJsonParseError(
                          json_loc,
                          "pow(" + std::to_string(val) + ", " +
                              std::to_string(_exponent) + ") cannot be evaluated."));
  }
}

template <typename T>
class QuantitativeValidator : public ScaleValueValidator<T> {
 public:
  void setExponent(T e) {}
  void operator()(const JSONLocation& json_loc, const T& val) {}
};
template <>
class QuantitativeValidator<double> {
 public:
  QuantitativeValidator(ScaleType scaleType) { _scaleType = scaleType; }
  void setExponent(double e) { _exponent = e; }
  void operator()(const JSONLocation& json_loc, const double& val) {
    switch (_scaleType) {
      case ScaleType::kLinear:
        break;
      case ScaleType::kSqrt:
        validateSqrt(json_loc, val);
        break;
      case ScaleType::kLog:
        validateLog(json_loc, val);
        break;
      case ScaleType::kPow:
        validatePow(json_loc, val, _exponent);
        break;
      default:
        CHECK(false);
    }
  }

 private:
  ScaleType _scaleType = ScaleType::kLinear;
  double _exponent = 1.0;
};

class QuantitativeScaleBuilder : public ScaleBuilder<QuantitativeScaleBuilder> {
  template <typename T>
  ConvertFuncT<T> getDomainConvertFunction() {
    return nullptr;
  }

  template <typename T>
  ScaleValueValidatorUqPtr<T> getDomainValidator() {
    return make_unique<QuantitativeValidator>(scaleType);
  }
};

template <>
ConvertFuncT<double> QuantitativeScaleBuilder::getDomainConvertFunction() {
  return nullptr;
}

template <>
ScaleValueValidatorUqPtr<double, scaleType>
QuantitativeScaleBuilder::getDomainValidator() {
  return std::make_unique<QuantitativeValidator<double, scaleType>>;
}
#endif
}  // namespace details

template <typename DomainType, typename RangeType>
class QuantitativeScale : public Scale<DomainType, RangeType> {
 public:
  QuantitativeScale(const JSONLocation& json_loc,
                    QueryRendererContext& ctx,
                    BaseScale& parent_base_scale,
                    const QueryDataType domain_data_type,
                    const QueryDataType range_data_type,
                    const ScaleInterpType interp_type = ScaleInterpType::kUndefined)
      : Scale<DomainType, RangeType>(json_loc,
                                     ctx,
                                     parent_base_scale,
                                     domain_data_type,
                                     range_data_type)

      , use_clamp_{false}
      , interp_type_{interp_type}
      , domain_validate_func_{nullptr}
      , domain_convert_func_{nullptr}
      , are_props_dirty_{false}
      , pow_exponent_{1.0f} {
    // Map external scale type to internal quantitative type
    auto scale_type = parent_base_scale.getType();

    switch (scale_type) {
      case ScaleType::kLinear:
        quantitative_type_ = QuantitativeType::kLinear;
        break;
      case ScaleType::kLog:
        quantitative_type_ = QuantitativeType::kLog;
        domain_validate_func_ = validateLogFunc<DomainType>;
        break;
      case ScaleType::kSqrt:
        quantitative_type_ = QuantitativeType::kSqrt;
        domain_validate_func_ = validateSqrtFunc<DomainType>;
        break;
      case ScaleType::kPow:
        quantitative_type_ = QuantitativeType::kPow;
        domain_validate_func_ = [this](const JSONLocation& json_loc,
                                       const DomainType& val) {
          validatePowFunc<DomainType>(json_loc, val, pow_exponent_);
        };
        break;
      default:
        THROW_RUNTIME_EX("Invalid QuantitativeScale type");
    }
    domain_convert_func_ = details::getQuantConvertFunc<DomainType>(scale_type);
  }

  ~QuantitativeScale() override {}

  ConvertFuncT<DomainType> getDomainConvertFunction() final {
    return domain_convert_func_;
  }

  operator std::string() const final {
    switch (quantitative_type_) {
      case QuantitativeType::kLinear:
        return "LinearScale " + this->printInfo();
      case QuantitativeType::kLog:
        return "LogScale " + this->printInfo();
      case QuantitativeType::kPow:
        return "PowScale " + this->printInfo();
      case QuantitativeType::kSqrt:
        return "SqrtScale " + this->printInfo();
      default:
        return std::string();
    }
  }

 protected:
  AccumulatorType getValidAccumTypeMask() const final {
    return AccumulatorType::kDensity | AccumulatorType::kPct;
  }

  bool havePropertiesChanged() const final {
    return this->has_null_enabled_changed_ || are_props_dirty_;
  }

  ValidateFuncT<DomainType> getDomainValidateFunction() override {
    return domain_validate_func_;
  }

 private:
  // Internal enum controlling type, isolated from the public enum which spans all
  // scale types.
  enum class QuantitativeType { kLinear, kLog, kPow, kSqrt };

  QuantitativeType quantitative_type_;

  bool use_clamp_;
  ScaleInterpType interp_type_;
  ValidateFuncT<DomainType> domain_validate_func_;
  ConvertFuncT<DomainType> domain_convert_func_;
  bool are_props_dirty_;

  void updateFromJSONObj(const JSONLocation& json_loc) final {
    // Pull special properties
    if (quantitative_type_ == QuantitativeType::kPow) {
      const auto exp_loc = json_loc.getMember(JSONSchema_v1::Scales::kExponentProp);
      if (exp_loc.isValid()) {
        RUNTIME_EX_ASSERT(exp_loc.isNumber(),
                          RapidJSONUtils::createJsonParseError(
                              exp_loc, "Expecting a number for pow scales."));

        pow_exponent_ = RapidJSONUtils::getNumValFromJSONObj<float>(exp_loc);
      } else {
        // TODO(croot): set a const default for _powExponent somewhere
        pow_exponent_ = 1.0f;
      }

      domain_validate_func_ = [this](const JSONLocation& json_loc,
                                     const DomainType& val) {
        validatePowFunc<DomainType>(json_loc, val, pow_exponent_);
      };
      domain_convert_func_ = details::getQuantConvertFunc<DomainType>(
          this->parent_base_scale_.getType(), pow_exponent_);
    }

    // Handle clamp updating
    {
      const bool prev_clamp = use_clamp_;

      const auto clamp_loc = json_loc.getMember(JSONSchema_v1::Scales::kClampProp);
      if (clamp_loc.isValid()) {
        RUNTIME_EX_ASSERT(clamp_loc.isBool(),
                          RapidJSONUtils::createJsonParseError(
                              clamp_loc, "The property must be a boolean."));
        use_clamp_ = clamp_loc.getBool();
      } else {
        // TODO(croot): set a const default for use_clamp_ somewhere
        // Clamping by default makes sense for Density accumulation.
        // TODO(croot): should that be a default for other types?
        use_clamp_ =
            (this->parent_base_scale_.getAccumulatorType() == AccumulatorType::kDensity
                 ? true
                 : false);
      }

      are_props_dirty_ = prev_clamp != use_clamp_;
    }

    {
      auto interp_type = getScaleInterpTypeFromJSONObj(json_loc);
      if (interp_type != ScaleInterpType::kUndefined) {
        RUNTIME_EX_ASSERT(
            validateInterpolator(interp_type),
            RapidJSONUtils::createJsonParseError(
                json_loc,
                "Invalid type " + to_string(interp_type) + ". " +
                    to_string(this->parent_base_scale_.getType()) +
                    " scales with a range of type " +
                    to_string(this->parent_base_scale_.getRangeDataType()) +
                    " only support the following interpolators: [" +
                    boost::algorithm::join(
                        getScaleInterpTypes(getSupportedInterpolators()), ", ") +
                    "]"));
      }
      are_props_dirty_ = are_props_dirty_ || interp_type != interp_type_;
      interp_type_ = interp_type;
    }
  }

  void postDRDataJSONUpdate(const JSONLocation& json_loc) final {}

  uint32_t getNumValuesForAccumulation() const override {
    // null values will be accumulated separately, where appropriate
    // FIXME(scb): allow override here? (base, coerced, override consistency)
    return this->parent_base_scale_.getRangeData(true)->size() +
           (this->null_val_.has_value() ? 1 : 0);
  }

  void toJSONInternal(rapidjson::Value& obj,
                      rapidjson::Document::AllocatorType& allocator) const override {
    Scale<DomainType, RangeType>::toJSONInternal(obj, allocator);
    // TODO(croot): move the "clamp" prop name into a const somewhere.
    obj.AddMember("clamp", use_clamp_, allocator);
    if (quantitative_type_ == QuantitativeType::kPow) {
      obj.AddMember("exponent", pow_exponent_, allocator);
    }
  }

  void modifyBindOptions(BaseScale::BindOptions& opt) final {
    if (opt.use_accum) {
      opt.use_domain = false;
      opt.use_range = false;
      opt.use_null = false;
    } else {
      opt.use_null = true;
    }
  }

  void bindUniforms(gfx::Material& material,
                    const std::string& extra_suffix,
                    const BaseScale::BindOptions& bind_opt,
                    const SQLTypeInfo* sql_type_info) final {
    Scale<DomainType, RangeType>::bindUniforms(
        material, extra_suffix, bind_opt, sql_type_info);
    if (quantitative_type_ == QuantitativeType::kPow) {
      material.setUniformAttribute(
          "uExponent_" + this->parent_base_scale_.getName() + extra_suffix,
          pow_exponent_);
    }
  }

  void bindImplementationSubroutines(gfx::ShaderManager::Builder& builder,
                                     const std::string& extra_suffix,
                                     bool is_accum_final_pass) final {
    // We need to set these subroutines IF:
    //  - no accumulator has been set on this scale
    //  - accumulator is set and it's the final accumulator pass but NOT for the initial
    //  pass
    // FIXME(scb): This is not a sane API, but it is current logical requirement
    if (is_accum_final_pass == this->parent_base_scale_.hasAccumulator()) {
      // TODO(croot): cache these name bindings so we're not building up these strings
      // every time.

      std::string transform_func;
      std::string interp_func;
      switch (quantitative_type_) {
        case QuantitativeType::kLinear:
          transform_func = "passThruTransform";
          break;
        case QuantitativeType::kLog:
          transform_func = "logTransform";
          break;
        case QuantitativeType::kPow:
          transform_func = "powTransform";
          break;
        case QuantitativeType::kSqrt:
          transform_func = "sqrtTransform";
          break;
        default:
          THROW_RUNTIME_EX("ScaleType " + to_string(this->parent_base_scale_.getType()) +
                           " does not have a supported glsl transform func.");
      }

      switch (interp_type_) {
        case ScaleInterpType::kHsl:
        case ScaleInterpType::kHcl:
          interp_func = "colorInterpHslHcl";
          break;
        case ScaleInterpType::kHslLong:
        case ScaleInterpType::kHclLong:
          interp_func = "colorInterpHslHclLong";
          break;
        default:
          interp_func = "defaultInterp";
          break;
      }

      // TODO: string_view
      const std::string name = this->parent_base_scale_.getName();
      transform_func += "_" + name + extra_suffix;
      interp_func += "_" + name + extra_suffix;

      builder.addSubroutineBinding(
          "quantTransform_" + name + extra_suffix, transform_func, true);
      builder.addSubroutineBinding(
          "quantInterp_" + name + extra_suffix, interp_func, true);
    }
  }

  BaseScale::ScaleShaderType getShaderType() final {
    return BaseScale::ScaleShaderType::kQuantitative;
  }

  void modifyShaderTemplate(gfx::ShaderManager::Builder& builder) const final {
    builder.replaceAllTags("useClamp", std::to_string(use_clamp_));
  }

  // Log support (DomainType=double only)
  template <typename T = DomainType, EnableIfDouble<T>* = nullptr>
  static void validateLogFunc(const JSONLocation& json_loc, const T& val) {
    RUNTIME_EX_ASSERT(val > 0,
                      RapidJSONUtils::createJsonParseError(
                          json_loc,
                          std::to_string(val) +
                              " is <= 0. Log scales only work with positive values."));
  }

  template <typename T = DomainType, EnableIfNotDouble<T>* = nullptr>
  static void validateLogFunc(const JSONLocation& json_loc, const T& val) {}

  // Sqrt scale support (DomainType=double only)
  template <typename T = DomainType, EnableIfDouble<T>* = nullptr>
  static void validateSqrtFunc(const JSONLocation& json_loc, const T& val) {
    RUNTIME_EX_ASSERT(
        val >= 0,
        RapidJSONUtils::createJsonParseError(
            json_loc,
            std::to_string(val) + " is < 0. sqrt scales only work with positive values"));
  }

  template <typename T = DomainType, EnableIfNotDouble<T>* = nullptr>
  static void validateSqrtFunc(const JSONLocation& json_loc, const T& val) {}

  // Pow scale support

  // TODO(croot): should this be any type? Right now, it seems we can only do
  // a pow() of a float in glsl. That, and because of polymorphism issues when
  // there's a ScaleRef() object doing type coercion, it's easiest to just
  // keep this as a float.
  float pow_exponent_;

  template <typename T = DomainType, EnableIfDouble<T>* = nullptr>
  static void validatePowFunc(const JSONLocation& json_loc,
                              const DomainType& val,
                              const float exponent) {
    if (math_errhandling & MATH_ERREXCEPT) {
      std::feclearexcept(FE_ALL_EXCEPT);
      std::pow(val, exponent);
      RUNTIME_EX_ASSERT(
          !std::fetestexcept(FE_DIVBYZERO),
          RapidJSONUtils::createJsonParseError(json_loc,
                                               "pow(" + std::to_string(val) + ", " +
                                                   std::to_string(exponent) +
                                                   ") results in a divide-by-zero."));
      RUNTIME_EX_ASSERT(!std::fetestexcept(FE_INVALID),
                        RapidJSONUtils::createJsonParseError(
                            json_loc,
                            "pow(" + std::to_string(val) + ", " +
                                std::to_string(exponent) + ") cannot be evaluated."));
      RUNTIME_EX_ASSERT(
          !std::fetestexcept(FE_OVERFLOW),
          RapidJSONUtils::createJsonParseError(json_loc,
                                               "pow(" + std::to_string(val) + ", " +
                                                   std::to_string(exponent) +
                                                   ") results in an overflow error."));
      RUNTIME_EX_ASSERT(
          !std::fetestexcept(FE_UNDERFLOW),
          RapidJSONUtils::createJsonParseError(json_loc,
                                               "pow(" + std::to_string(val) + ", " +
                                                   std::to_string(exponent) +
                                                   ") results in an underflow error."));
    } else {
      errno = 0;
      std::pow(val, exponent);
      RUNTIME_EX_ASSERT(
          errno != ERANGE,
          RapidJSONUtils::createJsonParseError(
              json_loc,
              "pow(" + std::to_string(val) + ", " + std::to_string(exponent) +
                  ") results in a divide-by-zero or overflow or underflow error."));
      RUNTIME_EX_ASSERT(!std::fetestexcept(EDOM),
                        RapidJSONUtils::createJsonParseError(
                            json_loc,
                            "pow(" + std::to_string(val) + ", " +
                                std::to_string(exponent) + ") cannot be evaluated."));
    }
  }

  template <typename T = DomainType, EnableIfNotDouble<T>* = nullptr>
  static void validatePowFunc(const JSONLocation& json_loc,
                              const DomainType& val,
                              const float exponent) {}

  // Interpolator support
  bool validateInterpolator(const ScaleInterpType interp_type) {
    auto interps = getSupportedInterpolators();
    for (auto& interp : interps) {
      if (interp == interp_type) {
        return true;
      }
    }
    return false;
  }

  template <typename T = RangeType,
            typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
  std::vector<ScaleInterpType> getSupportedInterpolators() const {
    return {};
  }

  template <typename T = RangeType,
            gfx::EnableIfSpecificColorType<T, gfx::ColorRGBA>* = nullptr>
  std::vector<ScaleInterpType> getSupportedInterpolators() const {
    return {ScaleInterpType::kRgb};
  }

  template <typename T = RangeType,
            gfx::EnableIfSpecificColorType<T, gfx::ColorHSL>* = nullptr>
  std::vector<ScaleInterpType> getSupportedInterpolators() const {
    return {ScaleInterpType::kHsl, ScaleInterpType::kHslLong};
  }

  template <typename T = RangeType,
            gfx::EnableIfSpecificColorType<T, gfx::ColorLAB>* = nullptr>
  std::vector<ScaleInterpType> getSupportedInterpolators() const {
    return {ScaleInterpType::kLab};
  }

  template <typename T = RangeType,
            gfx::EnableIfSpecificColorType<T, gfx::ColorHCL>* = nullptr>
  std::vector<ScaleInterpType> getSupportedInterpolators() const {
    return {ScaleInterpType::kHcl, ScaleInterpType::kHclLong};
  }

  void bindAccumulatorColors(gfx::Material& material,
                             const std::string& attr_name) final {
    RUNTIME_EX_ASSERT(this->parent_base_scale_.getRangeDataType() == QueryDataType::COLOR,
                      "Colors are currently the only supported accumulation types.");

    auto data = this->getRangeVectorData();
    if (this->null_val_.has_value()) {
      data.push_back(*this->null_val_);
    }

    CHECK(static_cast<uint32_t>(data.size()) == getNumValuesForAccumulation());
    material.setUniformAttribute(attr_name, data);
  }
};

}  // namespace QueryRenderer
