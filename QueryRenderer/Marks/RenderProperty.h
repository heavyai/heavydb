/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseRenderProperty.h"

#include "GfxDriver/Colors/ColorUnion.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Data/Utils.h"
#include "QueryRenderer/Marks/Types.h"
#include "QueryRenderer/Scales/BaseScale.h"
#include "QueryRenderer/Scales/ScaleRef.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Types.h"

namespace QueryRenderer {

template <typename T, int NUM_COMPONENTS = 1>
class RenderProperty : public BaseRenderProperty {
 public:
  static constexpr RenderPropertyFlagBits kDefaultFlagBits =
      RenderPropertyFlagBits::kUseScale | RenderPropertyFlagBits::kFlexibleType;

  RenderProperty(const std::string& name,
                 QueryRendererContext& ctx,
                 BaseMarkFacade& mark_facade,
                 const RenderPropertyFlagBits flag_bits = kDefaultFlagBits)
      : BaseRenderProperty(name, ctx, mark_facade, flag_bits), mult_{}, offset_{} {
    resetValue();
    in_type_ = std::make_shared<gfx::TypeGLSL<T, NUM_COMPONENTS>>();
    out_type_ = std::make_shared<gfx::TypeGLSL<T, NUM_COMPONENTS>>();
  }

  ~RenderProperty() override = default;

  QueryDataType getDataType() const override {
    return TypeToQueryDataTypeSelector<T>::getQueryDataType();
  }

  void clearReferences() final {
    clearFieldPath();
    clearScalePtr();
    resetTypes();
    resetValue();
  }

  void initializeValue(const RenderPropertyValue& val) override {
    // TODO: this is a public function.. should I protect from already existing data?
    clearFieldPath();
    clearScalePtr();
    resetTypes();
    std::visit([this](const auto& arg) { uniform_val_.set(getDataType(), arg); }, val);
    validateValue(false, rapidjson::Pointer());
  }

  RenderPropertyValue getUniformValueAsRenderPropertyValue() const override {
    return uniform_val_.getVal<T>();
  }

  void setUniformAttribute(gfx::Material& active_material,
                           const std::string& uniform_attr_name) const override {
    if (in_type_) {
      if (vbo_init_type_ == VboInitType::kFromScaleRef) {
        if (dynamic_cast<gfx::TypeGLSL<int, 1>*>(in_type_.get())) {
          active_material.setUniformAttribute<int>(uniform_attr_name,
                                                   uniform_val_.getVal<int>());
        } else if (dynamic_cast<gfx::TypeGLSL<unsigned int, 1>*>(in_type_.get())) {
          active_material.setUniformAttribute<unsigned int>(
              uniform_attr_name, uniform_val_.getVal<unsigned int>());
        } else if (dynamic_cast<gfx::TypeGLSL<float, 1>*>(in_type_.get())) {
          active_material.setUniformAttribute<float>(uniform_attr_name,
                                                     uniform_val_.getVal<float>());
        } else if (dynamic_cast<gfx::TypeGLSL<double, 1>*>(in_type_.get())) {
          active_material.setUniformAttribute<double>(uniform_attr_name,
                                                      uniform_val_.getVal<double>());
        } else if (dynamic_cast<gfx::TypeGLSL<int64_t, 1>*>(in_type_.get())) {
          active_material.setUniformAttribute<int64_t>(uniform_attr_name,
                                                       uniform_val_.getVal<int64_t>());
        } else if (dynamic_cast<gfx::TypeGLSL<uint64_t, 1>*>(in_type_.get())) {
          active_material.setUniformAttribute<uint64_t>(uniform_attr_name,
                                                        uniform_val_.getVal<uint64_t>());
        } else {
          CHECK(false) << "Unsupported type: " << uniform_val_.getType() << " "
                       << in_type_->declString();
        }
      } else {
        auto checkptr = dynamic_cast<gfx::TypeGLSL<T, 1>*>(in_type_.get());
        CHECK(checkptr);
        active_material.setUniformAttribute<T>(uniform_attr_name,
                                               uniform_val_.getVal<T>());
      }
    }
  }

  operator std::string() const final {
    return "RenderProperty<" + std::string(typeid(T).name()) + ", " +
           std::to_string(NUM_COMPONENTS) + "> " + printInfo();
  }

 protected:
  AnyDataType uniform_val_;

  gfx::TypeGLSLShPtr createDefaultType() const final {
    return std::make_shared<gfx::TypeGLSL<T, NUM_COMPONENTS>>();
  }

 private:
  T mult_;
  T offset_;

  void initScaleFromJSONObj(const JSONLocation& scale_loc) override {
    RUNTIME_EX_ASSERT(
        scale_loc.isString(),
        RapidJSONUtils::createJsonParseError(
            scale_loc,
            "scale reference for mark property \"" + name_ + "\" must be a string."));

    RUNTIME_EX_ASSERT(!scale_config_ && !scale_,
                      RapidJSONUtils::createJsonParseError(
                          scale_loc,
                          "cannot initialize mark property \"" + name_ +
                              "\" from a scale. The context is uninitialized or a scale "
                              "is already being referenced."));

    // TODO(croot): We probably need a better way to deal with types. We've got an inType
    // that is either defined by an incoming data reference or an explicit value (or set
    // of values). The latter is easy. We already have the type in T/NUM_COMPONENTS of the
    // template. But the data reference is trickier.

    ScaleShPtr scale = ctx_.getScale(scale_loc.getString());
    RUNTIME_EX_ASSERT(scale != nullptr,
                      RapidJSONUtils::createJsonParseError(
                          scale_loc,
                          "the scale \"" + std::string(scale_loc.getString()) +
                              "\" does not exist in the json."));

    updateScalePtr(scale);
  }

  void setupScaleCB(const ScaleShPtr& scale) {
    // setup callbacks for scale updates
    CHECK_EQ(scale_ref_subscription_id_, RefCallbackId(0));
    scale_ref_subscription_id_ =
        ctx_.subscribeToRefEvent(RefEventType::kAll, scale, [this](auto&&... args) {
          return this->scaleRefUpdateCB(std::forward<decltype(args)>(args)...);
        });
  }

  void updateScalePtr(const ScaleShPtr& scale) override {
    CHECK(scale);
    const bool is_current_scale = (scale == scale_);

    const bool scale_accumulation = checkAccumulator(scale);
    auto* scale_accum_state = scale->getAccumState();
    auto accum_type = (scale_accum_state == nullptr) ? AccumulatorType::kUndefined
                                                     : scale_accum_state->getType();

    auto in_type_to_use = in_type_;
    if (in_type_to_use) {
      if (!scale_config_) {
        notifyChanged(ChangeType::kStructure);
      }

      if (vbo_init_type_ != VboInitType::kFromDataRef) {
        // FIXME(scb): accum renderer should handle this logic
        if (scale_accumulation && accum_type == AccumulatorType::kPct) {
          in_type_to_use = scale_accum_state->getPercentTypeGLSL();
        } else {
          in_type_to_use = scale->getDomainTypeGLSL();
        }
        CHECK(in_type_to_use);
        if (in_type_ != in_type_to_use && (*in_type_) != (*in_type_to_use)) {
          notifyChanged(ChangeType::kStructure);
        }
        in_type_ = in_type_to_use;
        vbo_init_type_ = VboInitType::kFromScaleRef;
        validateValue(true,
                      RapidJSONUtils::isValidPath(value_json_path_) ? value_json_path_
                                                                    : json_path_);
      }

      if (isDecimal()) {
        scale_config_ = std::make_shared<ScaleRef<double, T>>(ctx_, scale, this);
      } else if (dynamic_cast<gfx::TypeGLSL<unsigned int, 1>*>(in_type_to_use.get())) {
        scale_config_ = std::make_shared<ScaleRef<unsigned int, T>>(ctx_, scale, this);
      } else if (dynamic_cast<gfx::TypeGLSL<int, 1>*>(in_type_to_use.get())) {
        scale_config_ = std::make_shared<ScaleRef<int, T>>(ctx_, scale, this);
      } else if (dynamic_cast<gfx::TypeGLSL<float, 1>*>(in_type_to_use.get())) {
        scale_config_ = std::make_shared<ScaleRef<float, T>>(ctx_, scale, this);
      } else if (dynamic_cast<gfx::TypeGLSL<double, 1>*>(in_type_to_use.get())) {
        scale_config_ = std::make_shared<ScaleRef<double, T>>(ctx_, scale, this);
      } else if (dynamic_cast<gfx::TypeGLSL<int64_t, 1>*>(in_type_to_use.get())) {
        scale_config_ = std::make_shared<ScaleRef<int64_t, T>>(ctx_, scale, this);
      } else if (dynamic_cast<gfx::TypeGLSL<uint64_t, 1>*>(in_type_to_use.get())) {
        scale_config_ = std::make_shared<ScaleRef<uint64_t, T>>(ctx_, scale, this);
      } else {
        const bool scale_density_pct_accumulation =
            (scale_accumulation && (accum_type == AccumulatorType::kDensity ||
                                    accum_type == AccumulatorType::kPct));
        RUNTIME_EX_ASSERT(scale_density_pct_accumulation,
                          std::string(*this) + ": Scale domain with shader type \"" +
                              scale->getDomainTypeGLSL()->declString() +
                              "\" and data with shader type \"" +
                              in_type_to_use->declString() +
                              "\" are not supported to work together.");

        switch (scale->getDomainDataType()) {
          case QueryDataType::DOUBLE:
            scale_config_ = std::make_shared<ScaleRef<double, T>>(ctx_, scale, this);
            break;
          default:
            THROW_RUNTIME_EX(
                std::string(*this) +
                ": Unsupported density accumulator scale with domain of type " +
                to_string(scale->getDomainDataType()));
        }
      }
    } else {
      if (scale_config_) {
        notifyChanged(ChangeType::kStructure);
      }

      scale_config_ = nullptr;
    }

    if (!is_current_scale) {
      clearScalePtrForReplacement(scale_);
    }
    BaseRenderProperty::updateScalePtr(scale);
    if (!is_current_scale) {
      setupScaleCB(scale);
    }
  }

  void initFromJSONObj(const JSONLocation& obj_loc) override {
    // this is internally called at the appropriate time from
    // the base class's initialization function, so there's
    // no need to check that obj is valid since that should've
    // already been done.

    // TODO(croot) - fill this in with mult/offset json path caches.
    // These values will ultimately act as uniforms, so modifications
    // to them should not force a full shader build/compile.
    // if ((mitr = obj.FindMember("mult")) != obj.MemberEnd()) {
    //   _mult = RapidJSONUtils::getNumValFromJSONObj<T>(mitr->value);
    // }

    // if ((mitr = obj.FindMember("offset")) != obj.MemberEnd()) {
    //   _offset = RapidJSONUtils::getNumValFromJSONObj<T>(mitr->value);
    // }
  }

  bool resetTypes(const bool reset_in_type = true,
                  const bool reset_out_type = true) override {
    bool rtn = false;

    if (reset_in_type && vbo_init_type_ != VboInitType::kFromValue) {
      auto new_in_type = createDefaultType();

      for (auto& itr : per_gpu_data_) {
        itr.second.vbo.reset();
        itr.second.ssbo.reset();
      }

      if (vbo_init_type_ == VboInitType::kFromDataRef) {
        notifyChanged(ChangeType::kData);
      }
      vbo_init_type_ = VboInitType::kFromValue;

      if (!in_type_ || (*in_type_) != (*new_in_type)) {
        notifyChanged(ChangeType::kStructure);
        in_type_ = new_in_type;
        rtn = true;
      }
    }

    if (reset_out_type) {
      auto new_out_type = createDefaultType();

      if (!out_type_ || (*out_type_) != (*new_out_type)) {
        notifyChanged(ChangeType::kStructure);
        out_type_ = new_out_type;
        rtn = true;
      }
    }

    return rtn;
  }

  void initValueFromJSONObj(const JSONLocation& obj_loc,
                            const bool has_scale,
                            const bool reset_types = false) final {
    if (reset_types) {
      resetTypes();
    }
    auto old_val = uniform_val_;
    uniform_val_ = RapidJSONUtils::getAnyDataFromJSONObj(
        obj_loc,
        any_bits_set(flag_bits_ & RenderPropertyFlagBits::kAllowNonColorStrings));
    if (uniform_val_ != old_val) {
      notifyChanged(ChangeType::kValues);
    }
    validateValue(has_scale, obj_loc.getPathRef());
  }

  void validateValue(const bool has_scale,
                     const rapidjson::Pointer& value_path) override {}

  void resetValue() final { uniform_val_.set(getDataType(), T()); }

  std::pair<bool, bool> initTypeFromBuffer(const bool has_scale = false) final {
    bool in_changed = false, out_changed = false;
    auto itr = per_gpu_data_.begin();
    if (itr == per_gpu_data_.end()) {
      // there's nothing in the data ptr
      if (in_type_) {
        in_changed = true;
      }

      if (out_type_) {
        out_changed = true;
      }

      in_type_ = nullptr;
      out_type_ = nullptr;

      return std::make_pair(in_changed, out_changed);
    }

    QueryLayoutBufferShPtr buf_to_use =
        (!itr->second.vbo.expired()
             ? std::dynamic_pointer_cast<QueryLayoutBuffer>(itr->second.vbo.lock())
             : std::dynamic_pointer_cast<QueryLayoutBuffer>(itr->second.ssbo.lock()));
    RUNTIME_EX_ASSERT(buf_to_use != nullptr,
                      std::string(*this) +
                          ": Vertex/uniform buffer is uninitialized. Cannot initialize "
                          "type for mark property \"" +
                          name_ + "\".");

    auto layout = getDataLayoutForAttribute(data_, vbo_attr_name_);
    auto vbo_type = buf_to_use->getAttributeTypeGLSL(vbo_attr_name_, *layout);
    auto type_to_use = vbo_type;
    auto prev_is_decimal = isDecimal();

    decimal_exp_scale_ = 0;
    if (!any_bits_set(flag_bits_ & RenderPropertyFlagBits::kFlexibleType) &&
        (!any_bits_set(flag_bits_ & RenderPropertyFlagBits::kUseScale) || !has_scale)) {
      // if flexible type is false, then the render property is rigid,
      // meaning it cannot accept certain types. So validate the type of the attribute
      // in the vbo is appropriate. If validations succeeds, then make sure the
      // in/out types match the vbo
      RUNTIME_EX_ASSERT(validateInType(vbo_type),
                        std::string(*this) + ": The vertex buffer type " +
                            (vbo_type ? vbo_type->declString() : "\"null\"") +
                            " is not a valid input type for the mark property \"" +
                            name_ + ".");
      type_to_use = createDefaultType();
    } else if (layout && layout->isDecimalAttr(vbo_attr_name_)) {
      type_to_use = std::make_shared<gfx::TypeGLSL<double, 1>>();

      // NOTE: decimal types are only determined via data buffers.
      decimal_exp_scale_ = layout->getDecimalExp(vbo_attr_name_);
    }

    if (!in_type_ || *in_type_ != *vbo_type || prev_is_decimal != isDecimal()) {
      in_changed = true;
    }
    in_type_ = vbo_type;

    if (!has_scale) {
      if (!out_type_ || *out_type_ != *type_to_use) {
        out_changed = true;
      }
      out_type_ = type_to_use;
    }

    return std::make_pair(in_changed, out_changed);
  }

  bool validateInType(const gfx::TypeGLSLShPtr& type) override {
    auto type_to_use = (out_type_ ? out_type_ : createDefaultType());
    CHECK(type && type_to_use) << "comparing type: "
                               << (type ? type->declString() : "null") << " = "
                               << (type_to_use ? type_to_use->declString() : "null")
                               << " for " << name_ << " property";
    return (*type_to_use) == (*type);
  }

  bool validateOutType(const gfx::TypeGLSLShPtr& type) override {
    CHECK(type && in_type_) << "comparing type: " << (type ? type->declString() : "null")
                            << " = " << (in_type_ ? in_type_->declString() : "null")
                            << " for " << name_ << " property";

    if (any_bits_set(flag_bits_ & RenderPropertyFlagBits::kFlexibleType)) {
      return (*type) == (*in_type_);
    } else {
      return (*type) == (*createDefaultType());
    }
  }

  void validateScale() override {}

  void scaleRefUpdateCB(RefEventType ref_event_type,
                        const RefObjShPtr& ref_obj) override {
    auto scale = std::dynamic_pointer_cast<BaseScale>(ref_obj);
    CHECK(scale);
    switch (ref_event_type) {
      case RefEventType::kUpdate: {
        RUNTIME_EX_ASSERT(
            any_bits_set(flag_bits_ & RenderPropertyFlagBits::kAllowAccumulator) ||
                !scale->hasAccumulator(),
            std::string(*this) + ": scale \"" + scale->getName() +
                "\" has an accumulator scale, but this property "
                "doesn't allow for accumulator scales.");

        if (scale_config_) {
          scale_config_->updateScaleRef(scale);
        }
        // TODO(croot): should we do something here if there is no _scaleConfigPtr?
        // Should we check if the data has changed or something?
        if (scale_->isShaderDirty()) {
          notifyChanged(ChangeType::kStructure);
        }
        notifyChanged(ChangeType::kValues);
        break;
      }
      case RefEventType::kReplace:
        if (scale != scale_) {
          updateScalePtr(scale);
        }
        break;
      case RefEventType::kRemove:
        THROW_RUNTIME_EX(
            std::string(*this) + ": Error, scale: " + ref_obj->getName() +
            " has been removed but is still being referenced by this render property.")
        break;
      default:
        THROW_RUNTIME_EX(std::string(*this) + ": Ref event type: " +
                         std::to_string(static_cast<int>(ref_event_type)) +
                         " isn't currently supported for scale reference updates.");
        break;
    }
  }
};

template <>
RenderProperty<gfx::ColorUnion, 1>::RenderProperty(
    const std::string& name,
    QueryRendererContext& ctx,
    BaseMarkFacade& mark_facade,
    const RenderPropertyFlagBits flag_bits);

template <>
void RenderProperty<gfx::ColorUnion, 1>::setUniformAttribute(
    gfx::Material& active_material,
    const std::string& uniform_attr_name) const;

template <>
void RenderProperty<gfx::ColorUnion, 1>::updateScalePtr(const ScaleShPtr& scale);

template <>
gfx::TypeGLSLShPtr RenderProperty<gfx::ColorUnion, 1>::createDefaultType() const;

template <>
void RenderProperty<gfx::ColorUnion, 1>::resetValue();

template <>
bool RenderProperty<gfx::ColorUnion, 1>::validateInType(const gfx::TypeGLSLShPtr& type);

template <>
void RenderProperty<gfx::ColorUnion, 1>::validateScale();

class ColorRenderProperty : public RenderProperty<gfx::ColorUnion> {
 public:
  static constexpr RenderPropertyFlagBits kDefaultFlagBits =
      RenderPropertyFlagBits::kUseScale | RenderPropertyFlagBits::kAllowAccumulator;

  ColorRenderProperty(const std::string& name,
                      QueryRendererContext& ctx,
                      BaseMarkFacade& mark_facade,
                      const RenderPropertyFlagBits flag_bits = kDefaultFlagBits)
      : RenderProperty<gfx::ColorUnion>(name, ctx, mark_facade, flag_bits) {
    CHECK(!any_bits_set(flag_bits & kDisallowedFlagBits));
  }

  ~ColorRenderProperty() override {}

  gfx::ColorType getColorType() const;
  bool isColorPacked() const;
  bool hasAccumulator() const override;

 private:
  static constexpr RenderPropertyFlagBits kDisallowedFlagBits =
      ~(RenderPropertyFlagBits::kUseScale | RenderPropertyFlagBits::kAllowAccumulator |
        RenderPropertyFlagBits::kResetOnEmptyDataUpdate);

  enum class ColorInitType { kFromString = 0, kFromPackedUInt };
  ColorInitType color_init_type_;

  static bool isPackedColorType(const SQLTypeInfo& attr_type);

  void initFromJSONObj(const JSONLocation& obj_loc) final;
  void validateValue(const bool has_scale, const rapidjson::Pointer& value_path) final;
  RenderPropertyValue getUniformValueAsRenderPropertyValue() const final;
};

class KeyRenderProperty : public RenderProperty<int64_t> {
 public:
  static constexpr RenderPropertyFlagBits kDefaultFlagBits =
      RenderPropertyFlagBits::kResetOnEmptyDataUpdate;

  KeyRenderProperty(const std::string& name,
                    QueryRendererContext& ctx,
                    BaseMarkFacade& mark_facade,
                    const RenderPropertyFlagBits flag_bits = kDefaultFlagBits)
      : RenderProperty<int64_t>(name, ctx, mark_facade, flag_bits) {
    CHECK(!any_bits_set(flag_bits & kDisallowedFlagBits));
  }

 private:
  static constexpr RenderPropertyFlagBits kDisallowedFlagBits =
      ~RenderPropertyFlagBits::kResetOnEmptyDataUpdate;
  bool validateInType(const gfx::TypeGLSLShPtr& type) final {
    // NOTE(adb): The key property can be either int64_t or int32_t depending on whether
    // the underlying graphics hardware supports 64-bit integers as vertex buffer
    // attributes. Here, we simply ensure that the glsl types match, as an int64_t key
    // will automatically be converted to int32_t when the vertex buffer layout is
    // converted (if the underlying hardware does not support int64_t vertex attributes).
    auto type_to_use = (out_type_ ? out_type_ : createDefaultType());
    return (*type == *type_to_use);
  }
};

class EnumRenderProperty : public RenderProperty<int> {
 public:
  static constexpr RenderPropertyFlagBits kDefaultFlagBits =
      RenderPropertyFlagBits::kUseScale | RenderPropertyFlagBits::kFlexibleType;

  EnumRenderProperty(const std::string& name,
                     const QueryDataType enum_data_type,
                     QueryRendererContext& ctx,
                     BaseMarkFacade& mark_facade,
                     const RenderPropertyFlagBits flag_bits = kDefaultFlagBits,
                     std::function<int(const std::string&)> string_convert_func = nullptr)
      : RenderProperty<int>(name,
                            ctx,
                            mark_facade,
                            flag_bits | RenderPropertyFlagBits::kAllowNonColorStrings)
      , enum_data_type_{enum_data_type}
      , string_convert_func_{string_convert_func} {
    CHECK(!any_bits_set(flag_bits & kDisallowedFlagBits));
  }

  ~EnumRenderProperty() override {}

  QueryDataType getDataType() const final { return enum_data_type_; }

 private:
  static constexpr RenderPropertyFlagBits kDisallowedFlagBits =
      ~(RenderPropertyFlagBits::kUseScale | RenderPropertyFlagBits::kFlexibleType |
        RenderPropertyFlagBits::kResetOnEmptyDataUpdate);
  QueryDataType enum_data_type_;
  std::function<int(const std::string&)> string_convert_func_;
  void validateValue(const bool has_scale, const rapidjson::Pointer& value_path) final;
};

class BoolRenderProperty : public RenderProperty<int> {
 public:
  static constexpr RenderPropertyFlagBits kDefaultFlagBits =
      RenderPropertyFlagBits::kFlexibleType;

  BoolRenderProperty(const std::string& name,
                     const QueryDataType data_type,
                     QueryRendererContext& ctx,
                     BaseMarkFacade& mark_facade,
                     const RenderPropertyFlagBits flag_bits = kDefaultFlagBits,
                     std::function<int(const std::string&)> string_convert_func = nullptr)
      : RenderProperty<int>(name,
                            ctx,
                            mark_facade,
                            flag_bits | RenderPropertyFlagBits::kAllowNonColorStrings)
      , data_type_{data_type}
      , string_convert_func_{string_convert_func} {
    CHECK(!any_bits_set(flag_bits & kDisallowedFlagBits));
  }

  ~BoolRenderProperty() override {}

  QueryDataType getDataType() const final { return data_type_; }

 private:
  static constexpr RenderPropertyFlagBits kDisallowedFlagBits =
      ~(RenderPropertyFlagBits::kUseScale | RenderPropertyFlagBits::kFlexibleType |
        RenderPropertyFlagBits::kResetOnEmptyDataUpdate);
  QueryDataType data_type_;
  std::function<int(const std::string&)> string_convert_func_;
  void validateValue(const bool has_scale, const rapidjson::Pointer& value_path) final;
};

}  // namespace QueryRenderer
