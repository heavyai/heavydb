/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Scales/Scale.h"

namespace QueryRenderer {

template <typename DomainType, typename RangeType>
class ThresholdScale : public Scale<DomainType, RangeType> {
 public:
  ThresholdScale(const JSONLocation& json_loc,
                 QueryRendererContext& ctx,
                 BaseScale& parent_base_scale,
                 const QueryDataType domain_data_type,
                 const QueryDataType range_data_type)
      : Scale<DomainType, RangeType>(json_loc,
                                     ctx,
                                     parent_base_scale,
                                     domain_data_type,
                                     range_data_type) {}

  ~ThresholdScale() override {}

 public:
  operator std::string() const final { return "ThresholdScale" + this->printInfo(); }

 private:
  void updateFromJSONObj(const JSONLocation& json_loc) final {}

  void postDRDataJSONUpdate(const JSONLocation& json_loc) override {
    // FIXME(scb): allow override here? (base, coerced, override consistency)
    validateDomainRangeSizes(
        this->parent_base_scale_.getDomainData(true)->getJSONLocation());
  }

  uint32_t getNumValuesForAccumulation() const override {
    return this->parent_base_scale_.getRangeData()->size() +
           (this->null_val_.has_value() ? 1 : 0);
  }

  void modifyBindOptions(BaseScale::BindOptions& opt) final {
    opt.use_null = true;
    if (opt.use_accum) {
      opt.use_range = false;
    }
  }

  BaseScale::ScaleShaderType getShaderType() final {
    return BaseScale::ScaleShaderType::kThreshold;
  }

  // FIXME(scb): Boilerplate + coupling. Should be containable to BaseScale + AccumRender.
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

  void validateDomainRangeSizes(const JSONLocation& json_loc) final {
    if (this->ctx_.isReadyForRender(this->parent_base_scale_)) {
      // Only doing this validation in the case where we're not distributed or we're not
      // in post-vega transform aggregation mode
      // TODO(croot): Find a better way to control when validation and evaluation is
      // performed. This would be better controlled by a proper dataflow DAG
      // representation

      // Threshold scales require N + 1 range elements where domain is of size N

      // FIXME(scb): allow override here? (base, coerced, override consistency)
      auto domain_size = this->parent_base_scale_.getDomainData()->size();
      auto range_size = this->parent_base_scale_.getRangeData()->size();

      RUNTIME_EX_ASSERT(range_size == domain_size + 1,
                        RapidJSONUtils::createJsonParseError(
                            json_loc,
                            "Range length must be exactly 1 greater than domain length "
                            "for threshold scale. Num domains: " +
                                std::to_string(domain_size) +
                                ", Num Ranges: " + std::to_string(range_size) + "."));
    }
  }
};

}  // namespace QueryRenderer
