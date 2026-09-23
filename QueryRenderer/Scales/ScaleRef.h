/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/algorithm/string/replace.hpp>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/Scales/BaseScaleRef.h"
#include "QueryRenderer/Scales/ScaleAccumRenderState.h"
#include "QueryRenderer/Scales/ScaleAccumState.h"
#include "QueryRenderer/Scales/ScaleDomainRangeData.h"

namespace QueryRenderer {

template <class T, class TT, class Enable = void>
struct ConvertDomainRangeData {
  void operator()(QueryRendererContext& ctx,
                  std::shared_ptr<ScaleDomainRangeData<T>>& dest_data,
                  ScaleDomainRangeData<TT>* src_data) {
    std::vector<TT>& src_vec = src_data->getVectorDataRef();

    dest_data = std::make_shared<ScaleDomainRangeData<T>>(
        ctx,
        src_data->isDomain(),
        src_data->getName(),
        TypeToQueryDataTypeSelector<T>::getQueryDataType(),
        src_vec.size(),
        src_data->useString());
    std::vector<T>& dest_vec = dest_data->getVectorDataRef();
    for (size_t i = 0; i < src_vec.size(); ++i) {
      dest_vec[i] = static_cast<T>(src_vec[i]);
    }
  }
};

template <class T, class TT>
struct ConvertDomainRangeData<
    T,
    TT,
    typename std::enable_if_t<gfx::is_color<T>::value && std::is_arithmetic_v<TT>>> {
  void operator()(QueryRendererContext& ctx,
                  std::shared_ptr<ScaleDomainRangeData<T>>& dest_data,
                  ScaleDomainRangeData<TT>* src_data) {
    THROW_RUNTIME_EX("Cannot convert a numeric value (" + std::string(*src_data) +
                     ") to a color" +
                     (dest_data ? " (" + std::string(*dest_data) + ")." : "."));
  }
};

template <class T, class TT>
struct ConvertDomainRangeData<
    T,
    TT,
    typename std::enable_if_t<std::is_arithmetic_v<T> && gfx::is_color<TT>::value>> {
  void operator()(QueryRendererContext& ctx,
                  std::shared_ptr<ScaleDomainRangeData<T>>& dest_data,
                  ScaleDomainRangeData<TT>* src_data) {
    THROW_RUNTIME_EX("Cannot convert a color (" + std::string(*src_data) +
                     ") to a numeric value" +
                     (dest_data ? " (" + std::string(*dest_data) + ")." : "."));
  }
};

template <class T, class TT>
struct ConvertDomainRangeData<
    T,
    TT,
    typename std::enable_if_t<gfx::is_color<T>::value && gfx::is_color<TT>::value>> {
  void operator()(QueryRendererContext& ctx,
                  std::shared_ptr<ScaleDomainRangeData<T>>& dest_data,
                  ScaleDomainRangeData<TT>* src_data) {
    std::vector<TT>& src_vec = src_data->getVectorDataRef();

    dest_data = std::make_shared<ScaleDomainRangeData<T>>(ctx,
                                                          src_data->isDomain(),
                                                          src_data->getName(),
                                                          QueryDataType::COLOR,
                                                          src_vec.size(),
                                                          src_data->useString());
    std::vector<T>& dest_vec = dest_data->getVectorDataRef();
    for (size_t i = 0; i < src_vec.size(); ++i) {
      convertColor(src_vec[i], dest_vec[i]);
    }
  }
};

template <typename DomainType, typename RangeType>
class ScaleRef : public BaseScaleRef {
 public:
  ScaleRef(QueryRendererContext& ctx,
           const ScaleShPtr& scale,
           BaseRenderProperty* rndr_prop,
           const QueryDataType domain_data_type =
               TypeToQueryDataTypeSelector<DomainType>::getQueryDataType(),
           const QueryDataType range_data_type =
               TypeToQueryDataTypeSelector<RangeType>::getQueryDataType())
      : BaseScaleRef(ctx, scale, rndr_prop)
      , coerced_domain_data_{nullptr}
      , coerced_range_data_{nullptr}
      , coerced_domain_data_type_{domain_data_type}
      , coerced_range_data_type_{range_data_type}
      , sorted_{false} {
    updateDomainRange(true, true, true);
    initScalePtr(coerced_domain_data_, coerced_range_data_);
  }

  ~ScaleRef() override {}

  QueryDataType getDomainDataType() const final {
    verifyScalePointer();
    if (coerced_domain_data_) {
      return coerced_domain_data_->getType();
    }
    return scale_->getDomainDataType();
  }

  QueryDataType getRangeDataType() const final {
    verifyScalePointer();
    if (coerced_range_data_) {
      return coerced_range_data_->getType();
    }
    return scale_->getRangeDataType();
  }

  const gfx::TypeGLSLShPtr& getDomainTypeGLSL() const override {
    verifyScalePointer();
    if (coerced_domain_data_) {
      return coerced_domain_data_->getTypeGLSL();
    }

    return scale_->getDomainTypeGLSL(false);
  }

  const gfx::TypeGLSLShPtr& getRangeTypeGLSL() const override {
    verifyScalePointer();
    if (coerced_range_data_) {
      return coerced_range_data_->getTypeGLSL();
    }

    return scale_->getRangeTypeGLSL(false);
  }

  gfx::ShaderManager::BuilderShPtr getShaderSubBuilder(
      const std::string& extra_suffix) const override {
    verifyScalePointer();
    return scale_->getShaderSubBuilder(this, extra_suffix, true);
  }

  void updateScaleRef(const ScaleShPtr& scale) override {
    // check if we arrived here in response to a RefEventType::kReplace callback from
    // RenderProperty
    if (scale != scale_) {
      deleteScalePtr();
      scale_ = scale;
    }

    updateDomainRange(
        scale_->hasDomainDataChanged(),
        scale_->hasRangeDataChanged(),
        scale_->getAccumState() && scale_->getAccumState()->hasPctAccumChanged());

    // if we get to this point, that means the scale is going to be used by a mark.
    // So we'll initialize any gpu resources the scale may use now.
    initScalePtr(coerced_domain_data_, coerced_range_data_);
  }

  void bindUniforms(gfx::Material& material, const std::string& extra_suffix) override {
    RENDER_LOG_SCOPE();
    verifyScalePointer();

    auto* accum_state = scale_->getAccumState();
    auto* accum_render_state = scale_->getAccumRenderState();
    bool coerce_domain =
        (coerced_domain_data_ != nullptr &&
         (accum_state == nullptr || accum_state->supportsDomainCoercion()));
    bool coerce_range = coerced_range_data_ != nullptr && accum_state == nullptr;

    const SQLTypeInfo* sql_type_info = nullptr;
    const auto column_name = getDataColumnName();
    if (column_name.length()) {
      auto layout_ptr = getDataLayoutForAttribute(getDataTablePtr(), column_name);
      if (layout_ptr) {
        sql_type_info = layout_ptr->getAttrSQLTypeInfoPtr(column_name);
      }
    }

    if (coerce_domain) {
      material.setUniformAttribute(scale_->getDomainGLSLUniformName() + extra_suffix,
                                   coerced_domain_data_->getVectorDataRef());

      auto const domain_uniform_data =
          scale_->getDomainTypeUniforms(extra_suffix, sql_type_info);
      auto const& domain_type = domain_uniform_data.first;
      auto const& uniform_map = domain_uniform_data.second;
      constexpr bool ignore_null_conversion = true;

      for (auto const& [name, value] : uniform_map) {
        material.setUniformAttribute(
            name, convertType<DomainType>(domain_type, value, ignore_null_conversion));
      }
    }

    if (coerce_range) {
      material.setUniformAttribute(scale_->getRangeGLSLUniformName() + extra_suffix,
                                   coerced_range_data_->getVectorDataRef());

      auto const range_uniform_data = scale_->getRangeTypeUniforms(extra_suffix);
      auto const& range_type = range_uniform_data.first;
      auto const& uniform_map = range_uniform_data.second;

      for (auto const& [name, value] : uniform_map) {
        material.setUniformAttribute(name, convertType<RangeType>(range_type, value));
      }
    }

    // extra_suffix comes from the renderProperty allowing the shader to hold more
    // than one instance of the scale (e.g. "_x" and "_y" for quantitative scales)
    scale_->bindUniforms(
        material, extra_suffix, !coerce_domain, !coerce_range, true, sql_type_info);

    if (accum_state && accum_render_state) {
      if (accum_state->getType() == AccumulatorType::kPct && pct_cat_val_) {
        accum_render_state->bindUniforms(
            material, extra_suffix, pct_cat_val_.get(), pct_margin_val_.get());
      } else {
        accum_render_state->bindUniforms(material, extra_suffix, nullptr, nullptr);
      }
    }
  }

  BaseScaleDomainRangeData* getDomainData() final {
    verifyScalePointer();
    if (coerced_domain_data_) {
      return coerced_domain_data_.get();
    }

    return scale_->getDomainData();
  }

  BaseScaleDomainRangeData* getRangeData() final {
    verifyScalePointer();
    if (coerced_range_data_) {
      return coerced_range_data_.get();
    }

    return scale_->getRangeData();
  }

  operator std::string() const final {
    return "ScaleRef<" + std::string(typeid(DomainType).name()) + ", " +
           std::string(typeid(RangeType).name()) + "> " + printInfo();
  }

 private:
  std::shared_ptr<ScaleDomainRangeData<DomainType>> coerced_domain_data_;
  std::shared_ptr<ScaleDomainRangeData<RangeType>> coerced_range_data_;
  const QueryDataType coerced_domain_data_type_;
  const QueryDataType coerced_range_data_type_;

  bool sorted_;

  void updateDomainRange(const bool update_domain,
                         const bool update_range,
                         const bool update_accum = false,
                         const bool force_update = false) {
    CHECK(scale_ != nullptr);

    bool prev_sorted = sorted_, do_sort = false;

    // reset the sort flag
    sorted_ = false;

    const bool is_domain_continuous = isContinuousDomainScale(scale_->getType());
    auto* accum_state = scale_->getAccumState();
    const bool is_pct_accum =
        accum_state && accum_state->getType() == AccumulatorType::kPct;

    if (update_domain && !is_pct_accum) {
      BaseScaleDomainRangeData* domain_data = scale_->getDomainData(true);
      const auto& their_domain_type = domain_data->getTypeInfo();
      const auto& our_domain_type = typeid(DomainType);
      if (force_update ||
          (!is_domain_continuous && their_domain_type != our_domain_type) ||
          (is_domain_continuous &&
           !areTypesCompatible(their_domain_type, our_domain_type))) {
        if (auto* uint_domain =
                dynamic_cast<ScaleDomainRangeData<unsigned int>*>(domain_data)) {
          ConvertDomainRangeData<DomainType, unsigned int>()(
              ctx_, coerced_domain_data_, uint_domain);
        } else if (auto* int_domain =
                       dynamic_cast<ScaleDomainRangeData<int>*>(domain_data)) {
          ConvertDomainRangeData<DomainType, int>()(
              ctx_, coerced_domain_data_, int_domain);
        } else if (auto* float_domain =
                       dynamic_cast<ScaleDomainRangeData<float>*>(domain_data)) {
          ConvertDomainRangeData<DomainType, float>()(
              ctx_, coerced_domain_data_, float_domain);
        } else if (auto* uint64_domain =
                       dynamic_cast<ScaleDomainRangeData<uint64_t>*>(domain_data)) {
          ConvertDomainRangeData<DomainType, uint64_t>()(
              ctx_, coerced_domain_data_, uint64_domain);
        } else if (auto* int64_domain =
                       dynamic_cast<ScaleDomainRangeData<int64_t>*>(domain_data)) {
          ConvertDomainRangeData<DomainType, int64_t>()(
              ctx_, coerced_domain_data_, int64_domain);
        } else if (auto* double_domain =
                       dynamic_cast<ScaleDomainRangeData<double>*>(domain_data)) {
          ConvertDomainRangeData<DomainType, double>()(
              ctx_, coerced_domain_data_, double_domain);
        } else if (auto* string_domain =
                       dynamic_cast<ScaleDomainRangeData<std::string>*>(domain_data)) {
          doStringToDataConversion(string_domain);
          do_sort = true;
        } else {
          THROW_RUNTIME_EX(std::string(*this) +
                           ": Cannot create scale reference - unsupported domain type.");
        }
      } else {
        coerced_domain_data_ = nullptr;
      }
    }

    if (update_accum && is_pct_accum) {
      const auto& their_pct_cat = accum_state->getPercentCategoryVal();
      CHECK(their_pct_cat);
      const auto their_pct_type = their_pct_cat->getType();
      const auto our_pct_type =
          TypeToQueryDataTypeSelector<DomainType>::getQueryDataType();

      if (force_update || (!is_domain_continuous && their_pct_type != our_pct_type) ||
          (is_domain_continuous && !areTypesCompatible(their_pct_type, our_pct_type))) {
        if (!pct_cat_val_) {
          pct_cat_val_ = std::make_unique<AnyDataType>();
        }

        (*pct_cat_val_) = (*their_pct_cat);
        if (their_pct_type == QueryDataType::STRING) {
          doStringToDataConversion(nullptr, pct_cat_val_.get());
        } else {
          pct_cat_val_->convertToType(our_pct_type);
        }

        const auto& their_pct_margin = accum_state->getPercentMargin();
        if (their_pct_margin) {
          if (!pct_margin_val_) {
            pct_margin_val_ = std::make_unique<AnyDataType>();
          }

          (*pct_margin_val_) = (*their_pct_margin);
          pct_margin_val_->convertToType(our_pct_type);
        } else {
          pct_margin_val_.reset();
        }
      }
    } else if (!is_pct_accum) {
      pct_cat_val_.reset();
      pct_margin_val_.reset();
    }

    // make sure to un-sort the range if it was previously sorted
    if (update_range || (!do_sort && prev_sorted)) {
      BaseScaleDomainRangeData* range_data = scale_->getRangeData(true);
      if (force_update || range_data->getTypeInfo() != typeid(RangeType)) {
        if (auto* uint_domain =
                dynamic_cast<ScaleDomainRangeData<unsigned int>*>(range_data)) {
          ConvertDomainRangeData<RangeType, unsigned int>()(
              ctx_, coerced_range_data_, uint_domain);
        } else if (auto* int_domain =
                       dynamic_cast<ScaleDomainRangeData<int>*>(range_data)) {
          ConvertDomainRangeData<RangeType, int>()(ctx_, coerced_range_data_, int_domain);
        } else if (auto* float_domain =
                       dynamic_cast<ScaleDomainRangeData<float>*>(range_data)) {
          ConvertDomainRangeData<RangeType, float>()(
              ctx_, coerced_range_data_, float_domain);
        } else if (auto* uint64_domain =
                       dynamic_cast<ScaleDomainRangeData<uint64_t>*>(range_data)) {
          ConvertDomainRangeData<RangeType, uint64_t>()(
              ctx_, coerced_range_data_, uint64_domain);
        } else if (auto* int64_domain =
                       dynamic_cast<ScaleDomainRangeData<int64_t>*>(range_data)) {
          ConvertDomainRangeData<RangeType, int64_t>()(
              ctx_, coerced_range_data_, int64_domain);
        } else if (auto* double_domain =
                       dynamic_cast<ScaleDomainRangeData<double>*>(range_data)) {
          ConvertDomainRangeData<RangeType, double>()(
              ctx_, coerced_range_data_, double_domain);
        }
        // TODO(croot): support other strings?
        else if (auto* color_rgba_domain =
                     dynamic_cast<ScaleDomainRangeData<gfx::ColorRGBA>*>(range_data)) {
          ConvertDomainRangeData<RangeType, gfx::ColorRGBA>()(
              ctx_, coerced_range_data_, color_rgba_domain);
        } else if (auto* color_hsl_domain =
                       dynamic_cast<ScaleDomainRangeData<gfx::ColorHSL>*>(range_data)) {
          ConvertDomainRangeData<RangeType, gfx::ColorHSL>()(
              ctx_, coerced_range_data_, color_hsl_domain);
        } else if (auto* color_lab_domain =
                       dynamic_cast<ScaleDomainRangeData<gfx::ColorLAB>*>(range_data)) {
          ConvertDomainRangeData<RangeType, gfx::ColorLAB>()(
              ctx_, coerced_range_data_, color_lab_domain);
        } else if (auto* color_hcl_domain =
                       dynamic_cast<ScaleDomainRangeData<gfx::ColorHCL>*>(range_data)) {
          ConvertDomainRangeData<RangeType, gfx::ColorHCL>()(
              ctx_, coerced_range_data_, color_hcl_domain);
        } else {
          THROW_RUNTIME_EX(std::string(*this) +
                           ": Cannot create scale reference - unsupported range type.");
        }
      } else {
        coerced_range_data_ = nullptr;
      }
    }
  }

  void doStringToDataConversion(ScaleDomainRangeData<std::string>* domain_data = nullptr,
                                AnyDataType* pct_accum_data = nullptr) {
    CHECK(domain_data || pct_accum_data);
    auto render_query_runner = ctx_.getRenderQueryRunner();

    RUNTIME_EX_ASSERT(render_query_runner != nullptr,
                      std::string(*this) +
                          ": An render_query_runner is not defined. Cannot numerically "
                          "convert a string column.");

    auto& data_table = getDataTablePtr();
    RUNTIME_EX_ASSERT(
        data_table != nullptr,
        std::string(*this) + ": A data table is not referenced by render property \"" +
            getRenderPropertyName() + "\". Cannot numerically convert a string column.");

    auto sql_data_table = dynamic_cast<BaseQueryDataTableSQLJSON*>(data_table.get());
    RUNTIME_EX_ASSERT(
        sql_data_table != nullptr,
        std::string(*this) + ": The data table referenced by render property \"" +
            getRenderPropertyName() +
            "\" is not an sql data table. Cannot numerically convert a string column");

    const auto& table_info = sql_data_table->getQuerySQL().getAllTableInfoRef();
    RUNTIME_EX_ASSERT(!table_info.phys_tables.empty(),
                      std::string(*this) +
                          "The sql data table referenced by render property \"" +
                          getRenderPropertyName() +
                          "\" is not properly initialized. It is either missing a "
                          "\"dbTableName\" property or the "
                          "query wasn't executed. Cannot numerically "
                          "convert a string column");

    std::string col_name = getDataColumnName();
    RUNTIME_EX_ASSERT(col_name.length() != 0,
                      std::string(*this) + ": The render property \"" +
                          getRenderPropertyName() +
                          "\" is missing a column name to reference in the data. Cannot "
                          "numerically convert a string column.");

    QueryDataLayoutShPtr query_data_layout = sql_data_table->getVboQueryDataLayout();
    // TODO(croot): check whether the colName exists in the layout
    if (!query_data_layout || !query_data_layout->hasAttribute(col_name)) {
      query_data_layout = sql_data_table->getSsboQueryDataLayout();
      if (!query_data_layout) {
        // We can run into a scenario where there still exists a referenced data ptr, but
        // that data ptr has nothing in it. We handle that case by just returning here,
        // but this scale ref should/will not do anything in the future (like init a
        // shader or bind uniforms) We can reach here and ultimately not render thanks to
        // a scale update in QueryRenderer::_initFromJSON
        return;
      }

      RUNTIME_EX_ASSERT(
          query_data_layout->hasAttribute(col_name),
          "The vega data \"" +
              RapidJSONUtils::getPointerPath(sql_data_table->getJsonPathRef()) +
              " does not have a layout that contains the attribute \"" + col_name + "\"");
    }

    if (domain_data) {
      auto& vec = domain_data->getVectorDataRef();
      coerced_domain_data_ =
          std::make_shared<ScaleDomainRangeData<DomainType>>(ctx_,
                                                             true,
                                                             domain_data->getName(),
                                                             coerced_domain_data_type_,
                                                             vec.size(),
                                                             domain_data->useString());

      // get dict-encoded string ids from the render query runner
      std::vector<DomainType>& coerced_vec = coerced_domain_data_->getVectorDataRef();
      auto string_ids = render_query_runner->getStringIds(
          *query_data_layout, col_name, vec, *sql_data_table->getResultSet(), true);
      CHECK(vec.size() == string_ids.size() && vec.size() == coerced_vec.size());
      for (size_t i = 0; i < string_ids.size(); ++i) {
        coerced_vec[i] = static_cast<DomainType>(string_ids[i]);
      }
    }

    if (pct_accum_data) {
      CHECK(pct_accum_data->getType() == QueryDataType::STRING);
      const auto new_type = TypeToQueryDataTypeSelector<DomainType>::getQueryDataType();
      auto string_ids =
          render_query_runner->getStringIds(*query_data_layout,
                                            col_name,
                                            {pct_accum_data->getStringVal()},
                                            *sql_data_table->getResultSet());
      pct_accum_data->set(new_type, static_cast<DomainType>(string_ids[0]));
    }
  }

  void sort(bool domain_updated, bool range_updated) {
    verifyScalePointer();

    bool has_domain = (domain_updated && coerced_domain_data_ != nullptr);
    bool has_range = (range_updated && coerced_range_data_ != nullptr);

    // force a copy of both the domain and range to sort
    updateDomainRange(!has_domain, !has_range, false, true);

    // TODO(croot): somehow do a sort in place? Not sure how to do this without
    // creating an iterator class on the ScaleRef objects (which might be nice
    // to do in the future). So, for now, I'm just copying all the domain/range
    // data as pairs into a vector, sorting that vector based on the domain, and
    // placing the results back. Very hacky, but since domains/ranges should on the
    // whole be small, this shouldn't be a big bottle neck.

    // TODO(croot): Possible bug -- the size of the domains/ranges don't have to
    // be equal. You can have more domains than ranges and vice-versa. So we need
    // to sort by the smaller of the two and leave the hanging items alone.

    auto& domain_vec = coerced_domain_data_->getVectorDataRef();
    auto& range_vec = coerced_range_data_->getVectorDataRef();

    int num_domains = domain_vec.size();
    int num_ranges = range_vec.size();

    if (num_ranges < num_domains) {
      // NOTE: this is a bug as the range colors don't follow along with the domains.
      // The problem is this - _sort() is currently only called when the domain of a
      // scale is driven by a string-encoded dict column, which requires a sort since
      // the GLSL shader used needs the domain in ascending order to do a binary
      // search through the domain values.
      // There is no concept of a hash-table (or map) inherent in GLSL so a
      // binary search is used for O(log n) speed. This means the domain needs
      // to be sorted. But in this case the range values don't follow along.
      // That was decided because usually when the user has more domain values than
      // range values, the intent is to use a subset of colors, but that the color
      // chosen for each category can be random. So not sorting here.
      //
      // NOTE: as of 9/7/17 the shader is not just doing a linear search.
      // It tends to be slightly faster at smaller number of domains and
      // solves this problem so that it works appropriately.
      // We're keeping the _sort function around for possible future use
      // but it is as of now unused.
      std::sort(domain_vec.begin(), domain_vec.end());
    } else {
      std::vector<std::pair<DomainType, RangeType>> sort_vec(num_domains);
      for (int i = 0; i < num_domains; ++i) {
        sort_vec[i] = std::make_pair(domain_vec[i], range_vec[i]);
      }

      std::sort(
          sort_vec.begin(),
          sort_vec.end(),
          [](const std::pair<DomainType, RangeType>& a,
             const std::pair<DomainType, RangeType>& b) { return a.first < b.first; });

      for (int i = 0; i < num_domains; ++i) {
        domain_vec[i] = sort_vec[i].first;
        range_vec[i] = sort_vec[i].second;
      }
    }

    sorted_ = true;
  }
};

}  // namespace QueryRenderer
