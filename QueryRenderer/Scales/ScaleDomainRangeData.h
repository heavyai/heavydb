/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "QueryRenderer/Data/EmbeddedPolyDataTable.h"
#include "QueryRenderer/Data/EmbeddedRowDataTable.h"
#include "QueryRenderer/Data/QuerySourceDataTable.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Marks/Enums.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Scales/BaseScaleDomainRangeData.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Scales/Utils.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/NumericUtils.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

template <typename T>
class ScaleDomainRangeData : public BaseScaleDomainRangeData {
 public:
  ScaleDomainRangeData(QueryRendererContext& ctx,
                       const bool is_domain,
                       const std::string& name,
                       const QueryDataType data_type,
                       const bool use_string = false)
      : BaseScaleDomainRangeData(ctx, is_domain, name, data_type, use_string)
      , cached_type_glsl_{nullptr}
      , prop_cb_subscription_id_{0} {}

  ScaleDomainRangeData(QueryRendererContext& ctx,
                       const bool isDomain,
                       const std::string& name,
                       const QueryDataType data_type,
                       size_t size,
                       bool use_string = false)
      : BaseScaleDomainRangeData(ctx, isDomain, name, data_type, use_string)
      , data_vector_(size)
      , cached_type_glsl_{nullptr}
      , prop_cb_subscription_id_(0) {}

  ~ScaleDomainRangeData() override {
    unsubscribePropCB();
    unsubscribeFromDataEvent();
  }

  ScaleDRChangedFlags initializeFromJSONObj(const JSONLocation& json_loc,
                                            const ScaleType type,
                                            ValidateFuncT<T> validate_val_func) {
    ScaleDRChangedFlags changed_flags = ScaleDRChangedFlags::kNone;
    bool is_object = false;
    bool is_string = false;
    BaseDataTableShPtr table;

    const auto domain_range_loc = json_loc.getMember(name_);
    RUNTIME_EX_ASSERT(
        domain_range_loc.isValid(),
        RapidJSONUtils::createJsonParseError(
            json_loc, "scale objects must have a \"" + name_ + "\" property."));

    if (!ctx_.isJSONCacheUpToDate(json_path_, domain_range_loc)) {
      updateJSONPath(json_loc.getPathRef(), true);
      if (use_string_) {
        RUNTIME_EX_ASSERT(((is_object = domain_range_loc.isObject()) ||
                           (is_string = domain_range_loc.isString()) ||
                           (domain_range_loc.isArray() && domain_range_loc.size())),
                          RapidJSONUtils::createJsonParseError(
                              domain_range_loc,
                              "Invalid type. Must be an object, string, or an array with "
                              "at least 1 element."));
      } else {
        RUNTIME_EX_ASSERT(
            ((is_object = domain_range_loc.isObject()) ||
             (domain_range_loc.isArray() && domain_range_loc.size())),
            RapidJSONUtils::createJsonParseError(
                domain_range_loc,
                "Invalid type. Must be an object or an array with at least 1 element."));
      }

      if (!is_string) {
        unsubscribePropCB();
      }

      const auto prev_size = data_vector_.size();

      validate_val_func_ = validate_val_func;
      if (is_object) {
        RUNTIME_EX_ASSERT(
            is_domain_,
            RapidJSONUtils::createJsonParseError(
                domain_range_loc, "Data references currently only supported by domains"));

        const auto data_loc =
            domain_range_loc.getMember(JSONSchema_v1::Scales::kDataProp);
        RUNTIME_EX_ASSERT(data_loc.isValid() && data_loc.isString(),
                          RapidJSONUtils::createJsonParseError(
                              (data_loc.isValid() ? data_loc : domain_range_loc),
                              "scale data reference must have a \"" +
                                  std::string(JSONSchema_v1::Scales::kDataProp) +
                                  "\" property and it must be a string."));
        table = ctx_.getDataTable(data_loc.getString());

        // TODO(croot): if we support data references in places other than scale
        // domain/ranges, then we should create a data ref object:
        // https://vega.github.io/vega/docs/scales/#dataref It seems that vega only
        // supports data references at the scale level, so this would be ok for now, but
        // it'd be better to create a dataref class/struct.
        unsubscribeFromDataEvent();
        updateDataFromDataRef(type, domain_range_loc, table);
        subscribeToDataEvent(table);
      } else {
        unsubscribeFromDataEvent();
        if (is_string) {
          setFromStringValue(domain_range_loc, type);
        } else {
          data_vector_.resize(domain_range_loc.size());
          data_vector_.shrink_to_fit();

          // gather all the items
          for (size_t i = 0; i < domain_range_loc.size(); ++i) {
            auto const& domain_range_item_loc = domain_range_loc[i];
            data_vector_[i] = getDataValueFromJSONObj(domain_range_item_loc);
            if (validate_val_func_) {
              validate_val_func_(domain_range_item_loc, data_vector_[i]);
            }
          }
        }
      }

      const auto new_size = data_vector_.size();

      // only need to regenerate the shader if the size of the
      // scale data changed, but we need to update any coerced
      // values if the scale's values changed, which means
      // we need to distinguish between when the size of the
      // data changed vs only the values changed.
      changed_flags =
          is_domain_ ? ScaleDRChangedFlags::kDomainVals : ScaleDRChangedFlags::kRangeVals;
      if (prev_size != new_size) {
        changed_flags |= is_domain_ ? ScaleDRChangedFlags::kDomainSize
                                    : ScaleDRChangedFlags::kRangeSize;
      }
    } else {
      updateJSONPath(json_loc.getPathRef(), false);
    }

    return static_cast<ScaleDRChangedFlags>(changed_flags);
  }

  uint32_t size() const override { return static_cast<uint32_t>(data_vector_.size()); }

  double getDifference(const double divisor = 1) const {
    uint32_t sz = size();
    RUNTIME_EX_ASSERT(sz > 0, "Cannot get difference from an empty domain/range.");
    return double(data_vector_[sz - 1] - data_vector_[0]) / divisor;
  }

  std::vector<T>& getVectorDataRef() { return data_vector_; }

  std::vector<T> getVectorData(
      std::function<T(const T&)> domain_val_convert = nullptr) const {
    auto rtn = data_vector_;
    if (domain_val_convert) {
      std::transform(rtn.begin(), rtn.end(), rtn.begin(), domain_val_convert);
    }
    return rtn;
  }

  rapidjson::Value toJSON(rapidjson::Document::AllocatorType& allocator) const final {
    rapidjson::Value arr_val(rapidjson::kArrayType);
    for (const auto& val : data_vector_) {
      arr_val.PushBack(RapidJSONUtils::valToJSON(val, allocator), allocator);
    }
    return arr_val;
  }

  static T getNullValue() { return ::QueryRenderer::getNullValue<T>(); }

  const gfx::TypeGLSLShPtr& getTypeGLSL() override {
    if (!cached_type_glsl_) {
      cached_type_glsl_ = TypeToQueryDataTypeSelector<T>::getTypeGLSLPtr();
    }
    return cached_type_glsl_;
  }

  inline const std::type_info& getTypeInfo() const override { return typeid(T); }

  static T getDataValueFromJSONObj(const JSONLocation& json_loc) {
    CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
    return RapidJSONUtils::getNumValFromJSONObj<T>(json_loc);
  }

  operator std::string() const override {
    return "ScaleDomainRangeData<" + std::string(typeid(T).name()) + ">(" + name_ + ")";
  }

 private:
  std::vector<T> data_vector_;
  gfx::TypeGLSLShPtr cached_type_glsl_;
  PropUpdateCallbackId prop_cb_subscription_id_;
  ValidateFuncT<T> validate_val_func_;

  std::vector<T> getDataFromEmbeddedDataRef(const std::vector<std::string>& column_names,
                                            BaseDataTable* data_ref) {
    CHECK(column_names.size() == 1);
    const auto& column_name = column_names[0];

    DataColumnShPtr column;
    auto data_table = dynamic_cast<EmbeddedRowDataTable*>(data_ref);
    if (data_table) {
      column = data_table->getColumn(column_name);
    } else {
      auto poly_data_table = dynamic_cast<EmbeddedPolyDataTable*>(data_ref);

      if (poly_data_table == nullptr) {
        throw std::runtime_error(
            "Unsupported data reference table type. Data reference "
            "is not a vertex or poly buffer-based data table.");
      }

      column = poly_data_table->getColumn(column_name);
    }

    auto data_column = dynamic_cast<TDataColumn<T>*>(column.get());

    if (data_column == nullptr) {
      throw std::runtime_error("Data column " + column_name + " is of type " +
                               to_string(column->getColumnType()) +
                               " is not compatible with domain/range data of type " +
                               to_string(getType()));
    }

    std::pair<T, T> min_max_domain = data_column->getExtrema();
    return {min_max_domain.first, min_max_domain.second};
  }

  std::vector<T> getDataFromSourcedDataRef(const std::vector<std::string>& column_names,
                                           BaseDataTable* data_ref) {
    auto source_data = dynamic_cast<QuerySourceDataTable*>(data_ref);
    CHECK(source_data);
    const auto array_col =
        BaseScaleDomainRangeData::validateArrayColsFromDataRef(column_names, source_data);
    return (array_col.size() ? source_data->getTypedVectorData<T>(array_col)
                             : source_data->getTypedVectorData<T>(column_names));
  }

  std::pair<bool, bool> updateDataFromDataRef(const ScaleType type,
                                              const JSONLocation& data_loc,
                                              const BaseDataTableShPtr& data_ref) final {
    auto data_input_format = data_ref->getInputFormat();
    RUNTIME_EX_ASSERT(data_input_format == DataInputFormat::kEmbedded ||
                          data_input_format == DataInputFormat::kSourced,
                      RapidJSONUtils::createJsonParseError(
                          data_loc,
                          "scale data references only support JSON-embedded or sourced "
                          "data tables currently."));

    auto column_names = getFieldsFromDataRef(data_loc, ctx_, data_ref);
    CHECK(column_names.size());

    const auto prev_size = data_vector_.size();
    // TODO(scb): replace with
    if (data_input_format == DataInputFormat::kEmbedded) {
      // TODO(scb): replace explicit scale type checks with supports / capabilities flags
      if (type == ScaleType::kLinear || type == ScaleType::kQuantize) {
        try {
          data_vector_ = getDataFromEmbeddedDataRef(column_names, data_ref.get());
        } catch (std::exception& e) {
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(data_loc, e.what()));
        }
      }  // TODO(scb): else??
    } else {
      try {
        data_vector_ = getDataFromSourcedDataRef(column_names, data_ref.get());
      } catch (gfx::OutOfGpuMemoryError& e) {
        // propogate OOM errors
        THROW_RUNTIME_EX(std::make_exception_ptr(e));
      } catch (std::exception& e) {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(data_loc, e.what()));
      }
    }
    if (validate_val_func_) {
      for (const auto& val : data_vector_) {
        validate_val_func_(data_loc, val);
      }
    }
    return std::make_pair(data_vector_.size() != prev_size, true);
  }

  void widthHeightUpdateCB(const uint32_t new_val, const uint32_t old_val) {
    CHECK(data_vector_.size() == 2);
    data_vector_[0] = T(0);
    data_vector_[1] = static_cast<T>(new_val);
    if (validate_val_func_) {
      const JSONLocation tmp_loc = getJSONLocation();
      validate_val_func_(tmp_loc, data_vector_[0]);
      validate_val_func_(tmp_loc, data_vector_[1]);
    }
  }

  void setupWidthHeightPropCB(const std::string& prop_name) {
    // setup callbacks for width/height updates
    auto cb = [this](const uint32_t w, const uint32_t h) { widthHeightUpdateCB(w, h); };
    prop_cb_subscription_id_ = ctx_.subscribeToPropEvent(prop_name, cb);
  }

  void unsubscribePropCB() {
    if (prop_cb_subscription_id_) {
      ctx_.unsubscribeFromPropEvent(prop_cb_subscription_id_);
      prop_cb_subscription_id_ = 0;
    }
  }

  // TODO(scb): Having scale implementations specifics here seems incorrect. This could
  // probably be better better handled using a policy class, since the Scale
  // implementation owns the DomainRangeData instantiation.
  void setFromStringValue(const JSONLocation& str_loc, const ScaleType type) {
    bool is_width = false, is_height = false;
    CHECK(str_loc.isValid() && str_loc.isString());
    auto lower_case = makeLowerCase(str_loc.getString());
    if (isQuantitativeScale(type) && ((is_width = (lower_case == "width")) ||
                                      (is_height = (lower_case == "height")))) {
      T max = 0;
      if (is_width) {
        max = static_cast<T>(ctx_.getWidth());
      } else {
        max = static_cast<T>(ctx_.getHeight());
      }
      data_vector_ = std::vector<T>{0, max};
      setupWidthHeightPropCB(lower_case);
    } else {
      unsubscribePropCB();
      if (type == ScaleType::kOrdinal && (lower_case == "symbol")) {
        data_vector_ = std::vector<T>(static_cast<int>(SymbolShapeType::kCOUNT));
        for (int i = 0; i < static_cast<int>(SymbolShapeType::kCOUNT); ++i) {
          data_vector_[i] = static_cast<T>(i);
        }
      } else {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
            str_loc,
            "\"" + std::string(str_loc.getString()) + "\" for scale type " +
                to_string(type) + " is invalid."));
      }
    }
  }
};

/*
 * RGBA specializations
 */
template <>
double ScaleDomainRangeData<gfx::ColorRGBA>::getDifference(const double divisor) const;

template <>
gfx::ColorRGBA ScaleDomainRangeData<gfx::ColorRGBA>::getDataValueFromJSONObj(
    const JSONLocation& json_loc);

template <>
std::pair<bool, bool> ScaleDomainRangeData<gfx::ColorRGBA>::updateDataFromDataRef(
    const ScaleType type,
    const JSONLocation& data_loc,
    const BaseDataTableShPtr& table);

template <>
void ScaleDomainRangeData<gfx::ColorRGBA>::setFromStringValue(const JSONLocation& str_loc,
                                                              const ScaleType type);
/*
 * HSL specializations
 */
template <>
double ScaleDomainRangeData<gfx::ColorHSL>::getDifference(const double divisor) const;

template <>
gfx::ColorHSL ScaleDomainRangeData<gfx::ColorHSL>::getDataValueFromJSONObj(
    const JSONLocation& json_loc);

template <>
std::pair<bool, bool> ScaleDomainRangeData<gfx::ColorHSL>::updateDataFromDataRef(
    const ScaleType type,
    const JSONLocation& data_loc,
    const BaseDataTableShPtr& table);

template <>
void ScaleDomainRangeData<gfx::ColorHSL>::setFromStringValue(const JSONLocation& str_loc,
                                                             const ScaleType type);

/*
 * LAB specializations
 */
template <>
double ScaleDomainRangeData<gfx::ColorLAB>::getDifference(const double divisor) const;

template <>
gfx::ColorLAB ScaleDomainRangeData<gfx::ColorLAB>::getDataValueFromJSONObj(
    const JSONLocation& json_loc);

template <>
std::pair<bool, bool> ScaleDomainRangeData<gfx::ColorLAB>::updateDataFromDataRef(
    const ScaleType type,
    const JSONLocation& data_loc,
    const BaseDataTableShPtr& table);

template <>
void ScaleDomainRangeData<gfx::ColorLAB>::setFromStringValue(const JSONLocation& str_loc,
                                                             const ScaleType type);

/*
 * HCL specializations
 */
template <>
double ScaleDomainRangeData<gfx::ColorHCL>::getDifference(const double divisor) const;

template <>
gfx::ColorHCL ScaleDomainRangeData<gfx::ColorHCL>::getDataValueFromJSONObj(
    const JSONLocation& json_loc);

template <>
std::pair<bool, bool> ScaleDomainRangeData<gfx::ColorHCL>::updateDataFromDataRef(
    const ScaleType type,
    const JSONLocation& data_loc,
    const BaseDataTableShPtr& table);

template <>
void ScaleDomainRangeData<gfx::ColorHCL>::setFromStringValue(const JSONLocation& str_loc,
                                                             const ScaleType type);

/*
 * string specializations
 */
template <>
double ScaleDomainRangeData<std::string>::getDifference(const double divisor) const;

template <>
std::string ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(
    const JSONLocation& json_loc);

template <>
std::vector<std::string> ScaleDomainRangeData<std::string>::getDataFromEmbeddedDataRef(
    const std::vector<std::string>& column_names,
    BaseDataTable* data_ref);

template <>
std::vector<std::string> ScaleDomainRangeData<std::string>::getDataFromSourcedDataRef(
    const std::vector<std::string>& column_names,
    BaseDataTable* data_ref);

template <>
void ScaleDomainRangeData<std::string>::setFromStringValue(const JSONLocation& str_loc,
                                                           const ScaleType type);

/*
 * unsigned int specializations
 */

/*
 * int specializations
 */

/*
 * float specializations
 */

/*
 * double specializations
 */

/*
 * int64_t specializations
 */

/*
 * uint64_t specializations
 */

}  // namespace QueryRenderer
