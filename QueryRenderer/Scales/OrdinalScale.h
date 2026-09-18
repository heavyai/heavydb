/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Scales/Scale.h"

namespace QueryRenderer {

template <typename DomainType, typename RangeType>
class OrdinalScale : public Scale<DomainType, RangeType> {
 public:
  OrdinalScale(const JSONLocation& json_loc,
               QueryRendererContext& ctx,
               BaseScale& parent_base_scale,
               const QueryDataType domain_data_type,
               const QueryDataType range_data_type)
      : Scale<DomainType, RangeType>(json_loc,
                                     ctx,
                                     parent_base_scale,
                                     domain_data_type,
                                     range_data_type)
      , default_range_val_() {}

  ~OrdinalScale() override {}

  operator std::string() const final { return "OrdinalScale" + this->printInfo(); }

 protected:
  AccumulatorType getValidAccumTypeMask() const final {
    return (AccumulatorType::kMin | AccumulatorType::kMax | AccumulatorType::kBlend);
  }

  BaseScale::RangeTypeUniforms getRangeTypeUniforms(
      const std::string& extra_suffix) const final {
    auto rtn = Scale<DomainType, RangeType>::getRangeTypeUniforms(extra_suffix);

    rtn.second.emplace(getRangeDefaultGLSLUniformName() + extra_suffix,
                       default_range_val_);

    return rtn;
  }

 private:
  bool has_default_val_changed_;
  RangeType default_range_val_;
  rapidjson::Pointer range_default_prop_json_path_;

  std::string getRangeDefaultGLSLUniformName() const {
    return "uDefault_" + this->parent_base_scale_.getName();
  }

  bool haveUniformPropertiesChanged() const final {
    return has_default_val_changed_ ||
           Scale<DomainType, RangeType>::haveUniformPropertiesChanged();
  }

  void updateFromJSONObj(const JSONLocation& json_loc) final {
    const auto default_loc = json_loc.getMember(JSONSchema_v1::Scales::kDefaultProp);
    RangeType new_default{};
    if (default_loc.isValid()) {
      const auto item_type = RapidJSONUtils::getDataTypeFromJSONObj(default_loc);
      RUNTIME_EX_ASSERT(
          areTypesCompatible(this->parent_base_scale_.getRangeDataType(), item_type),
          RapidJSONUtils::createJsonParseError(
              default_loc,
              "The scale \"" + this->parent_base_scale_.getName() +
                  "\" has a type for its \"" +
                  std::string(JSONSchema_v1::Scales::kDefaultProp) + "\" (" +
                  to_string(item_type) + ") that is not the same type as its range (" +
                  to_string(this->parent_base_scale_.getRangeDataType()) +
                  "). These types must be equal"));

      new_default = this->getTypedRangeData()->getDataValueFromJSONObj(default_loc);
    } else {
      // set an undefined default
      new_default = RangeType();
    }
    has_default_val_changed_ = (default_range_val_ != new_default);
    default_range_val_ = std::move(new_default);
    updateRangeDefaultPropJSONPath(json_loc.getPathRef());
  }

  uint32_t getNumValuesForAccumulation() const override {
    // null values will be accumulated separately, where appropriate
    // TODO(scb): use rangedata override??
    return this->parent_base_scale_.getRangeData(true)->size() + 1 +
           (this->null_val_.has_value() ? 1 : 0);
  }

  //
  // Uniform binding
  //
  void modifyBindOptions(BaseScale::BindOptions& opt) final {
    opt.use_null = true;
    if (opt.use_accum) {
      opt.use_range = false;
    }
  }

  BaseScale::ScaleShaderType getShaderType() final {
    return BaseScale::ScaleShaderType::kOrdinal;
  }

  void bindUniforms(gfx::Material& material,
                    const std::string& extra_suffix,
                    const BaseScale::BindOptions& bind_opt,
                    const SQLTypeInfo* sql_type_info) final {
    Scale<DomainType, RangeType>::bindUniforms(
        material, extra_suffix, bind_opt, sql_type_info);
    if (bind_opt.use_range) {
      material.setUniformAttribute(getRangeDefaultGLSLUniformName() + extra_suffix,
                                   default_range_val_);
    }
  }

  // FIXME(scb): Boilerplate + coupling. Should be containable to BaseScale + AccumRender.
  void bindAccumulatorColors(gfx::Material& material,
                             const std::string& attr_name) final {
    RUNTIME_EX_ASSERT(this->parent_base_scale_.getRangeDataType() == QueryDataType::COLOR,
                      "Colors are currently the only supported accumulation types.");

    auto data = this->getRangeVectorData();
    data.push_back(default_range_val_);

    // nulls will be last here. The ordinal scale shader accounts for this by doing a
    // and the default val by doing a numDomains_<name> + 1
    if (this->null_val_.has_value()) {
      data.push_back(*this->null_val_);
    }

    CHECK(static_cast<uint32_t>(data.size()) == this->getNumValuesForAccumulation());
    material.setUniformAttribute(attr_name, data);
  }

  //
  // JSON parse / serialization
  //
  void updateRangeDefaultPropJSONPath(const rapidjson::Pointer& obj_path) {
    // TODO(croot): expose "default" as a constant somewhere;
    std::string default_str = "default";
    range_default_prop_json_path_ =
        obj_path.Append(default_str.c_str(), default_str.length());
  }

  void toJSONInternal(rapidjson::Value& obj,
                      rapidjson::Document::AllocatorType& allocator) const final {
    Scale<DomainType, RangeType>::toJSONInternal(obj, allocator);
    // TODO(croot): expose "default" as a constant somewhere;
    obj.AddMember(
        "default", RapidJSONUtils::valToJSON(default_range_val_, allocator), allocator);
  }
};

}  // namespace QueryRenderer
