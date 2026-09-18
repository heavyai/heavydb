/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Pipeline/Material.h"
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/Scales/ScaleDomainRangeData.h"

namespace QueryRenderer {

// WIP(scb): experimental builder code, do not use yet
#if 0
template <typename T>
class ScaleValueValidator {
  void operator()(const JSONLocation& json_loc, const T& val);
};

template <typename T>
using ScaleValueValidatorUqPtr = std::unique_ptr<ScaleValueValidator<T>>;

template <typename Derived>
class ScaleBuilder {
 public:
  template <typename DomainType, typename RangeType>
  ScaleShPtr<DomainType, RangeType> createScale(const JSONLocation& json_loc,
                                                QueryRendererContext& ctx,
                                                const QueryDataType domain_type,
                                                const QueryDataType range_type,
                                                const std::string& scale_name,
                                                const ScaleType scale_type,
                                                const ScaleInterpType interp_type) {
    auto domain_data = std::make_unique<ScaleDomainRangeData<DomainType>>()
    return Derived._createScale<DomainType, RangeType>(
        json_loc, ctx, domain_type, range_type, scale_name, scale_type, interp_type);
  }
  template <typename T>
  ConvertFuncT<T> getDomainConvertFunction() {
    return Derived.getDomainConvertFunction<T>();
  }

  template <typename T, ScaleType scaleType>
  ScaleValueValidatorUqPtr<T, scaleType> getDomainValidator() {
    return Derived.getScaleValueValidatorUqPtr<T, scaleType>();
  }
};
#endif

template <typename DomainType, typename RangeType>
class Scale : public ScaleImplBase {
 public:
  Scale(const JSONLocation& json_loc,
        QueryRendererContext& ctx,
        BaseScale& parent_base_scale,
        const QueryDataType domain_data_type,
        const QueryDataType range_data_type)
      : parent_base_scale_{parent_base_scale}
      , has_null_enabled_changed_{false}
      , has_null_val_changed_{false}
      , null_val_{}
      , ctx_{ctx} {}

  ~Scale() override {}

 protected:
  BaseScale& parent_base_scale_;
  bool has_null_enabled_changed_;
  bool has_null_val_changed_;
  std::optional<RangeType> null_val_;
  const QueryRendererContext& ctx_;

  AccumulatorType getValidAccumTypeMask() const override { return AccumulatorType::kAll; }

  bool supportsAccumNulls() const override { return true; }

  virtual ConvertFuncT<DomainType> getDomainConvertFunction() { return nullptr; }
  virtual ValidateFuncT<DomainType> getDomainValidateFunction() { return nullptr; }

  // Only used during accumulation rendering - called by derived class (Quantize)
  std::vector<RangeType> getRangeVectorData() {
    auto override_data = std::dynamic_pointer_cast<ScaleDomainRangeData<RangeType>>(
        parent_base_scale_.range_override_data_);
    return override_data ? override_data->getVectorData()
                         : getTypedRangeData()->getVectorData();
  }

  void validateDomainRangeSizes(const JSONLocation&) override {
    // Only implemented by ThresholdScale
    // no-op everywhere else
  }

  // TODO(scb): There are currently only two properties we flag here "hasNullVal" and
  // QuantitativeScale's clamp. Changing these values invalidates the shader, causing
  // a rebuild. The current ad-hoc flag system works for now, but as we get more scales
  // and they become more complex, we're going to want a more robust system.
  bool havePropertiesChanged() const override { return has_null_enabled_changed_; }
  bool haveUniformPropertiesChanged() const override { return has_null_val_changed_; }

  ScaleDRChangedFlags updateDRDataFromJSONObj(const JSONLocation& json_loc) final {
    ScaleDRChangedFlags flags(ScaleDRChangedFlags::kNone);
    flags |= getTypedDomainData()->initializeFromJSONObj(
        json_loc, parent_base_scale_.getType(), getDomainValidateFunction());

    // FIXME(scb): Add remaining code path for range validation (nothing needs it yet)
    flags |= getTypedRangeData()->initializeFromJSONObj(
        json_loc, parent_base_scale_.getType(), nullptr);
    return flags;
  }

  void initNullValueFromJSONObj(const JSONLocation& json_loc) override {
    // TODO(scb): use boost::/std:: Optional for null value and remove bool flag
    const auto null_val_loc = json_loc.getMember(JSONSchema_v1::Scales::kNullValueProp);
    bool orig_has_null = null_val_.has_value();

    has_null_val_changed_ = false;
    if (null_val_loc.isValid()) {
      const auto item_type = RapidJSONUtils::getDataTypeFromJSONObj(null_val_loc);
      RUNTIME_EX_ASSERT(
          areTypesCompatible(parent_base_scale_.getRangeDataType(), item_type),
          RapidJSONUtils::createJsonParseError(
              null_val_loc,
              "The scale \"" + parent_base_scale_.getName() + "\" has a range of type " +
                  to_string(parent_base_scale_.getRangeDataType()) +
                  " which is not compatible with a \"" +
                  std::string(JSONSchema_v1::Scales::kNullValueProp) + "\" of type " +
                  to_string(item_type) + "."));

      auto const new_val = getTypedRangeData()->getDataValueFromJSONObj(null_val_loc);
      if (orig_has_null) {
        has_null_val_changed_ = (new_val != *null_val_);
      }
      null_val_ = new_val;
    } else {
      null_val_.reset();
    }
    has_null_enabled_changed_ = (orig_has_null != null_val_.has_value());
  }

  bool hasNullValue() const override { return null_val_.has_value(); }

  // Derived classes must call Scale::bindUniforms() if they override this method
  void bindUniforms(gfx::Material& material,
                    const std::string& extra_suffix,
                    const BaseScale::BindOptions& bind_opt,
                    const SQLTypeInfo* sql_type_info) override {
    const auto domain_val_convert = getDomainConvertFunction();

    const bool is_pct_accum =
        parent_base_scale_.getAccumulatorType() == AccumulatorType::kPct;
    if (bind_opt.use_domain && (!is_pct_accum || !bind_opt.use_accum)) {
      // call appropriate bindDomainUniforms
      auto override = parent_base_scale_.domain_override_data_.data.get();
      if (override) {
        // NOTE: an override should only be set when strings are used in the domain
        // and accumulation rendering is activated. So there's no need to do any value
        // conversion
        CHECK(!domain_val_convert);

        // TODO(scb): replace dynamic_cast<> with type_index lookup (scalemanager)
        if (auto uint_domain =
                dynamic_cast<ScaleDomainRangeData<unsigned int>*>(override)) {
          bindDomainUniforms(
              material, extra_suffix, uint_domain->getVectorDataRef(), sql_type_info);
        } else if (auto int_domain = dynamic_cast<ScaleDomainRangeData<int>*>(override)) {
          bindDomainUniforms(
              material, extra_suffix, int_domain->getVectorDataRef(), sql_type_info);
        } else if (auto float_domain =
                       dynamic_cast<ScaleDomainRangeData<float>*>(override)) {
          bindDomainUniforms(
              material, extra_suffix, float_domain->getVectorDataRef(), sql_type_info);
        } else if (auto double_domain =
                       dynamic_cast<ScaleDomainRangeData<double>*>(override)) {
          bindDomainUniforms(
              material, extra_suffix, double_domain->getVectorDataRef(), sql_type_info);
        } else {
          THROW_RUNTIME_EX("Override domain type " + to_string(override->getType()) +
                           " not supported.");
        }
      } else {
        bindDomainUniforms(material,
                           extra_suffix,
                           domain_val_convert
                               ? getTypedDomainData()->getVectorData(domain_val_convert)
                               : getTypedDomainData()->getVectorDataRef(),
                           sql_type_info);
      }
    }

    // Only true if NOT accumulating and NOT coerced
    if (bind_opt.use_range) {
      if (parent_base_scale_.range_override_data_) {
        auto overrideData = std::dynamic_pointer_cast<ScaleDomainRangeData<RangeType>>(
            parent_base_scale_.range_override_data_);
        CHECK(overrideData);
        material.setUniformAttribute(
            parent_base_scale_.getRangeGLSLUniformName() + extra_suffix,
            overrideData->getVectorDataRef());
      } else {
        material.setUniformAttribute(
            parent_base_scale_.getRangeGLSLUniformName() + extra_suffix,
            getTypedRangeData()->getVectorDataRef());
      }

      if (null_val_.has_value()) {
        material.setUniformAttribute(
            "nullRangeVal_" + parent_base_scale_.getName() + extra_suffix, *null_val_);
      }
    }
  }

  gfx::ColorType getRangeColorType() const final { return getRangeColorTypeInternal(); }

  BaseScale::RangeTypeUniforms getRangeTypeUniforms(
      const std::string& extra_suffix) const override {
    BaseScale::RangeTypeUniforms rtn(parent_base_scale_.getRangeDataType(),
                                     std::unordered_map<std::string, std::any>());

    if (null_val_.has_value()) {
      rtn.second.emplace("nullRangeVal_" + parent_base_scale_.getName() + extra_suffix,
                         *null_val_);
    }

    return rtn;
  }

  // FIXME(scb): External entry point needs to move to BaseScale?
  void toJSONInternal(rapidjson::Value& obj,
                      rapidjson::Document::AllocatorType& allocator) const override {
    if (null_val_.has_value()) {  // TODO(scb): can this all just go in BaseScale?
      // TODO(croot): expose "nullValue" as a constant somewhere;
      obj.AddMember(
          "nullValue", RapidJSONUtils::valToJSON(*null_val_, allocator), allocator);
    }
  }

  std::string printInfo() const {
    return "<" + std::string(typeid(DomainType).name()) + ", " +
           std::string(typeid(RangeType).name()) + ">" + parent_base_scale_.printInfo();
  }

  ScaleDomainRangeData<DomainType>* getTypedDomainData() {
    CHECK(parent_base_scale_.getDomainData());
    return dynamic_cast<ScaleDomainRangeData<DomainType>*>(
        parent_base_scale_.domain_data_.get());
  }
  ScaleDomainRangeData<RangeType>* getTypedRangeData() {
    CHECK(parent_base_scale_.getRangeData());
    return dynamic_cast<ScaleDomainRangeData<RangeType>*>(
        parent_base_scale_.range_data_.get());
  }

 private:
  void getDomainTypeUniformsInternal(BaseScale::DomainTypeUniforms& uniform_data,
                                     const std::string& extra_suffix,
                                     const SQLTypeInfo* sql_type_info) const final {
    if (!sql_type_info) {
      using OutputBufferType =
          typename TypeToQueryDataTypeSelector<DomainType>::BufferType;
      CHECK_EQ(uniform_data.second.size(), 0u);
      uniform_data.first =
          TypeToQueryDataTypeSelector<OutputBufferType>::getQueryDataType();
      if (null_val_.has_value()) {
        uniform_data.second.emplace(parent_base_scale_.getNullGLSLAttrName(extra_suffix),
                                    getNullValue<OutputBufferType>());
      }
    }
  }

  template <typename T = RangeType,
            typename std::enable_if_t<gfx::is_color<T>::value>* = nullptr>
  gfx::ColorType getRangeColorTypeInternal() const {
    return gfx::getColorType<T>();
  }

  template <typename T = RangeType,
            typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
  gfx::ColorType getRangeColorTypeInternal() const {
    THROW_RUNTIME_EX(std::string(*this) + " The range values are not color types.");
    return gfx::ColorType::RGBA;
  }

  // Specialization for non-color types
  template <typename T = DomainType,
            typename std::enable_if_t<std::is_arithmetic_v<T> ||
                                      std::is_same_v<T, std::string>>* = nullptr>
  void bindDomainUniforms(gfx::Material& material,
                          const std::string& extra_suffix,
                          const std::vector<T>& domain,
                          const SQLTypeInfo* sql_type_info) {
    material.setUniformAttribute(
        parent_base_scale_.getDomainGLSLUniformName() + extra_suffix, domain);
    if (null_val_.has_value()) {
      using OutputBufferType = typename TypeToQueryDataTypeSelector<T>::BufferType;
      material.setUniformAttribute(
          parent_base_scale_.getNullGLSLAttrName(extra_suffix),
          sql_type_info ? getNullValueFromTypeInfo<OutputBufferType>(*sql_type_info)
                        : getNullValue<OutputBufferType>());
    }
  }

  template <typename T = DomainType, gfx::EnableIfColorType<T>* = nullptr>
  void bindDomainUniforms(gfx::Material& material,
                          const std::string& extra_suffix,
                          const std::vector<T>& domain,
                          const SQLTypeInfo* sql_type_info) {
    RUNTIME_EX_ASSERT(null_val_.has_value(), "NULL values not yet supported for colors.");
    material.setUniformAttribute(
        parent_base_scale_.getDomainGLSLUniformName() + extra_suffix, domain);
  }

  virtual operator std::string() const = 0;
};

}  // namespace QueryRenderer
