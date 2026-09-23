/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/RenderProperty.h"

#include <boost/algorithm/string/join.hpp>

#include "QueryRenderer/Marks/Utils.h"
#include "Shared/sqltypes.h"

namespace QueryRenderer {

using ::gfx::ColorUnion;
using ::gfx::TypeGLSLShPtr;

template <>
RenderProperty<ColorUnion, 1>::RenderProperty(const std::string& name,
                                              QueryRendererContext& ctx,
                                              BaseMarkFacade& mark_facade,
                                              RenderPropertyFlagBits flag_bits)
    : BaseRenderProperty(name, ctx, mark_facade, flag_bits), mult_{}, offset_{} {
  resetValue();
  in_type_ = ColorUnion::getTypeGLSLPtr();
  out_type_ = ColorUnion::getTypeGLSLPtr();
}

template <>
gfx::TypeGLSLShPtr RenderProperty<ColorUnion, 1>::createDefaultType() const {
  return ColorUnion::getTypeGLSLPtr();
}

template <>
void RenderProperty<gfx::ColorUnion, 1>::resetValue() {
  uniform_val_.set(QueryDataType::COLOR, ColorUnion());
}

RenderPropertyValue ColorRenderProperty::getUniformValueAsRenderPropertyValue() const {
  if (uniform_val_.getType() == QueryDataType::COLOR) {
    return uniform_val_.getColorRef<ColorUnion>();
  }

  THROW_RUNTIME_EX(std::string(*this) +
                   " Cannot get the uniform value of the color property \"" + name_ +
                   "\". The property is referencing a scale with a domain that is not a "
                   "color type and therefore its "
                   "uniform value is not a color.");

  return ColorUnion();
}

template <>
void RenderProperty<ColorUnion, 1>::setUniformAttribute(
    gfx::Material& active_material,
    const std::string& uniform_attr_name) const {
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
      active_material.setUniformAttribute<std::array<float, 4>>(
          uniform_attr_name, uniform_val_.getColorRef<ColorUnion>().getColorArrayRef());
    }
  }
}

bool ColorRenderProperty::isPackedColorType(const SQLTypeInfo& attr_type) {
  return IS_INTEGER(attr_type.get_type());
}

void ColorRenderProperty::initFromJSONObj(const JSONLocation& obj_loc) {
  // TODO: what about offsets / mults for colors?

  // NOTE: this function is called by the base class during initialization, so we
  // know at this point that obj is an json Object

  // TODO(croot): move the following prop strings to a const somewhere
  auto const colorspace_loc = obj_loc.getMember(JSONSchema_v1::Marks::kColorSpaceProp);
  if (colorspace_loc.isValid()) {
    auto color_type = gfx::ColorType::RGBA;

    RUNTIME_EX_ASSERT(colorspace_loc.isString(),
                      RapidJSONUtils::createJsonParseError(colorspace_loc,
                                                           "Property must be a string"));

    auto color_string = std::string(colorspace_loc.getString());
    try {
      color_type = gfx::getColorTypeFromColorPrefix(color_string);
    } catch (gfx::RenderError& err) {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          colorspace_loc,
          "The string \"" + color_string +
              "\" is not a valid color space. The supported color spaces are: [" +
              boost::algorithm::join(gfx::getAllColorPrefixes(), ",") + "]"));
    }

    if (vbo_init_type_ == VboInitType::kFromValue) {
      auto& uniform_color = uniform_val_.getColorRef<ColorUnion>();
      if (color_init_type_ == ColorInitType::kFromPackedUInt) {
        // re-initialize the color based on this new space
        if (color_type != uniform_color.getType()) {
          auto packed_color = uniform_color.getPackedColor();
          try {
            uniform_color.initFromPackedUInt(packed_color, color_type);
          } catch (gfx::RenderError& err) {
            THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
                obj_loc,
                "The packed color " + std::to_string(packed_color) +
                    " is not a valid packed color. " + err.getDetails()));
          }
        }
      } else {
        RUNTIME_EX_ASSERT(
            uniform_color.getType() == color_type,
            RapidJSONUtils::createJsonParseError(
                obj_loc,
                "The color space \"" + color_string + "\" does not match the " +
                    gfx::to_string(uniform_color.getType()) +
                    " color space defined by value"));
      }
    } else if (vbo_init_type_ == VboInitType::kFromDataRef) {
      bool is_packed_type = false;
      if (in_type_) {
        is_packed_type = ColorUnion::isPackedTypeGLSL(in_type_);
      } else {
        auto layout = getDataLayoutForAttribute(data_, vbo_attr_name_);
        if (layout) {
          is_packed_type =
              isPackedColorType(layout->getAttrSQLTypeInfoRef(vbo_attr_name_));
        }
      }
      if (is_packed_type) {
        // verify that the color type is valid
        RUNTIME_EX_ASSERT(
            ColorUnion::isValidPackedType(color_type),
            RapidJSONUtils::createJsonParseError(
                obj_loc,
                "The data attribute \"" + vbo_attr_name_ +
                    "\" is being used as a packed color, but the color space \"" +
                    color_string +
                    "\" is not a valid packed color type. The supported packed color "
                    "types are [" +
                    boost::algorithm::join(ColorUnion::getPackedColorPrefixes(), ",") +
                    "]"));
      }

      // TODO(croot)
      uniform_val_.set(QueryDataType::COLOR,
                       ColorUnion(0.0f, 0.0f, 0.0f, 1.0f, color_type));
    }
  }
}

void ColorRenderProperty::validateValue(const bool has_scale,
                                        const rapidjson::Pointer& value_path) {
  if (!has_scale) {
    auto type = uniform_val_.getType();
    RUNTIME_EX_ASSERT(type == QueryDataType::COLOR || type == QueryDataType::INT ||
                          type == QueryDataType::UINT,
                      JSONRefErrorLogger(
                          ctx_.getRenderSessionKey(),
                          value_path,
                          "value for color property \"" + name_ +
                              "\" must be a string or a color packed into an int/uint."));

    if (type == QueryDataType::INT || type == QueryDataType::UINT) {
      ColorUnion color;
      auto num = uniform_val_.getVal<uint32_t>();
      try {
        color.initFromPackedUInt(num);
      } catch (gfx::RenderError& err) {
        THROW_RUNTIME_EX(JSONRefErrorLogger(ctx_.getRenderSessionKey(),
                                            value_path,
                                            "The packed color " + std::to_string(num) +
                                                " is not a valid packed color. " +
                                                err.what()));
      }
      color_init_type_ = ColorInitType::kFromPackedUInt;
      uniform_val_.set(QueryDataType::COLOR, color);
    }
  }
}

template <typename T>
static ScaleRefShPtr createColorScaleRef(const gfx::ColorType color_type,
                                         QueryRendererContext& ctx,
                                         const ScaleShPtr& scale,
                                         BaseRenderProperty* rndr_prop) {
  switch (color_type) {
    case gfx::ColorType::RGBA:
      return ScaleRefShPtr(new ScaleRef<T, gfx::ColorRGBA>(ctx, scale, rndr_prop));
      break;
    case gfx::ColorType::HSL:
      return ScaleRefShPtr(new ScaleRef<T, gfx::ColorHSL>(ctx, scale, rndr_prop));
      break;
    case gfx::ColorType::LAB:
      return ScaleRefShPtr(new ScaleRef<T, gfx::ColorLAB>(ctx, scale, rndr_prop));
      break;
    case gfx::ColorType::HCL:
      return ScaleRefShPtr(new ScaleRef<T, gfx::ColorHCL>(ctx, scale, rndr_prop));
      break;
    default:
      THROW_RUNTIME_EX(
          "Unsupported color type: " + std::to_string(static_cast<int>(color_type)) +
          ". Cannot create a color scale ref.");
  }

  return nullptr;
}

template <>
void RenderProperty<gfx::ColorUnion, 1>::updateScalePtr(const ScaleShPtr& scale) {
  CHECK(scale);
  const bool is_current_scale = (scale == scale_);

  const bool scale_accumulation = checkAccumulator(scale);
  const bool scale_density_pct_accumulation =
      (scale_accumulation && (scale->getAccumulatorType() == AccumulatorType::kDensity ||
                              scale->getAccumulatorType() == AccumulatorType::kPct));

  auto in_type_to_use = in_type_;
  if (in_type_to_use) {
    if (!scale_config_) {
      notifyChanged(ChangeType::kStructure);
    }

    if (vbo_init_type_ != VboInitType::kFromDataRef) {
      auto* scale_accum_state = scale->getAccumState();
      // FIXME(scb): accum renderer should handle this logic
      if (scale_accumulation && scale_accum_state &&
          scale_accum_state->getType() == AccumulatorType::kPct) {
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
      validateValue(
          true,
          RapidJSONUtils::isValidPath(value_json_path_) ? value_json_path_ : json_path_);
    }

    auto* range_data = scale->getRangeData();

    gfx::ColorType color_type = gfx::ColorType::RGBA;
    if (dynamic_cast<ScaleDomainRangeData<gfx::ColorRGBA>*>(range_data)) {
      color_type = gfx::ColorType::RGBA;
    } else if (dynamic_cast<ScaleDomainRangeData<gfx::ColorHSL>*>(range_data)) {
      color_type = gfx::ColorType::HSL;
    } else if (dynamic_cast<ScaleDomainRangeData<gfx::ColorLAB>*>(range_data)) {
      color_type = gfx::ColorType::LAB;
    } else if (dynamic_cast<ScaleDomainRangeData<gfx::ColorHCL>*>(range_data)) {
      color_type = gfx::ColorType::HCL;
    } else {
      THROW_RUNTIME_EX(
          std::string(*this) +
          ": Trying to add a color scale with an unsupported color type for its range.");
    }

    if (isDecimal()) {
      scale_config_ = createColorScaleRef<double>(color_type, ctx_, scale, this);
    } else if (dynamic_cast<gfx::TypeGLSL<unsigned int, 1>*>(in_type_to_use.get())) {
      scale_config_ = createColorScaleRef<unsigned int>(color_type, ctx_, scale, this);
    } else if (dynamic_cast<gfx::TypeGLSL<int, 1>*>(in_type_to_use.get())) {
      scale_config_ = createColorScaleRef<int>(color_type, ctx_, scale, this);
    } else if (dynamic_cast<gfx::TypeGLSL<float, 1>*>(in_type_to_use.get())) {
      scale_config_ = createColorScaleRef<float>(color_type, ctx_, scale, this);
    } else if (dynamic_cast<gfx::TypeGLSL<double, 1>*>(in_type_to_use.get())) {
      scale_config_ = createColorScaleRef<double>(color_type, ctx_, scale, this);
    } else if (dynamic_cast<gfx::TypeGLSL<int64_t, 1>*>(in_type_to_use.get())) {
      scale_config_ = createColorScaleRef<int64_t>(color_type, ctx_, scale, this);
    } else if (dynamic_cast<gfx::TypeGLSL<uint64_t, 1>*>(in_type_to_use.get())) {
      scale_config_ = createColorScaleRef<uint64_t>(color_type, ctx_, scale, this);
    } else {
      RUNTIME_EX_ASSERT(scale_density_pct_accumulation,
                        std::string(*this) + ": Scale domain with shader type \"" +
                            scale->getDomainTypeGLSL()->declString() +
                            "\" and data with shader type \"" +
                            in_type_to_use->declString() +
                            "\" are not supported to work together.");

      switch (scale->getDomainDataType()) {
        case QueryDataType::DOUBLE:
          scale_config_ = createColorScaleRef<double>(color_type, ctx_, scale, this);
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

template <>
bool RenderProperty<ColorUnion, 1>::validateInType(const gfx::TypeGLSLShPtr& type) {
  auto& uniform_color = uniform_val_.getColorRef<ColorUnion>();
  return uniform_color.isValidTypeGLSL(type);
}

template <>
void RenderProperty<ColorUnion, 1>::validateScale() {
  RUNTIME_EX_ASSERT(scale_config_ != nullptr || scale_ != nullptr,
                    std::string(*this) + ": Cannot verify scale for mark property \"" +
                        name_ + "\". Scale reference is uninitialized.");

  // colors need to be a specific type
  auto* range_data =
      (scale_config_ ? scale_config_->getRangeData() : scale_->getRangeData());

  RUNTIME_EX_ASSERT(
      dynamic_cast<ScaleDomainRangeData<gfx::ColorRGBA>*>(range_data) ||
          dynamic_cast<ScaleDomainRangeData<gfx::ColorHSL>*>(range_data) ||
          dynamic_cast<ScaleDomainRangeData<gfx::ColorLAB>*>(range_data) ||
          dynamic_cast<ScaleDomainRangeData<gfx::ColorHCL>*>(range_data),
      std::string(*this) + ": The scale \"" +
          (scale_config_ ? scale_config_->getName() : scale_->getName()) +
          " does not have a range with a valid color type for mark property \"" + name_ +
          "\".");
}

gfx::ColorType ColorRenderProperty::getColorType() const {
  if (scale_config_ || scale_) {
    auto* range_data =
        (scale_config_ ? scale_config_->getRangeData() : scale_->getRangeData());

    if (dynamic_cast<ScaleDomainRangeData<gfx::ColorRGBA>*>(range_data)) {
      return gfx::ColorType::RGBA;
    } else if (dynamic_cast<ScaleDomainRangeData<gfx::ColorHSL>*>(range_data)) {
      return gfx::ColorType::HSL;
    } else if (dynamic_cast<ScaleDomainRangeData<gfx::ColorLAB>*>(range_data)) {
      return gfx::ColorType::LAB;
    } else if (dynamic_cast<ScaleDomainRangeData<gfx::ColorHCL>*>(range_data)) {
      return gfx::ColorType::HCL;
    } else {
      THROW_RUNTIME_EX(
          std::string(*this) +
          ": Trying to add a color scale with an unsupported color type for its range.");
    }
  } else if (data_) {
    auto embedded_data = std::dynamic_pointer_cast<EmbeddedRowDataTable>(data_);
    if (embedded_data) {
      auto column = embedded_data->getColumn(vbo_attr_name_);
      if (std::dynamic_pointer_cast<TDataColumn<gfx::ColorRGBA>>(column)) {
        return gfx::ColorType::RGBA;
      } else if (std::dynamic_pointer_cast<TDataColumn<gfx::ColorHSL>>(column)) {
        return gfx::ColorType::HSL;
      } else if (std::dynamic_pointer_cast<TDataColumn<gfx::ColorLAB>>(column)) {
        return gfx::ColorType::LAB;
      } else if (std::dynamic_pointer_cast<TDataColumn<gfx::ColorHCL>>(column)) {
        return gfx::ColorType::HCL;
      } else if (std::dynamic_pointer_cast<TDataColumn<int>>(column) ||
                 std::dynamic_pointer_cast<TDataColumn<unsigned int>>(column)) {
        // The color is packed into an int here, so the type of the packed color will be
        // determined by the uniform, which holds the type of the color
        auto& uniform_color = uniform_val_.getColorRef<ColorUnion>();
        return uniform_color.getType();
      } else {
        THROW_RUNTIME_EX(std::string(*this) +
                         ": Trying to use a color embedded in the data with an "
                         "unsupported color type.");
      }
    }

    // NOTE: if a color is provided via a SQL query, it is defined by either a packed
    // uint, or a vec4f. In either case, we don't know what color space this color is
    // defined in. This is determined in Vega like so:
    //
    // fillColor: {
    //     field: "color",    // the name of the attr from the resulting sql
    //     colorSpace: "rgb"  // the space the color is defined in, in this case "rgb".
    //                        // Can also be "hsl", "lab", or "hcl"
    // },
    //
    // So the actual space will be determined in the uniform_val_ that will be set to the
    // appropriate space already.
  }

  auto& uniform_color = uniform_val_.getColorRef<ColorUnion>();
  return uniform_color.getType();
}

bool ColorRenderProperty::isColorPacked() const {
  if (scale_config_ || scale_) {
    // TODO(croot): support packed colors in scales?
    return false;
  }
  CHECK(in_type_);
  return ColorUnion::isPackedTypeGLSL(in_type_);
}

bool ColorRenderProperty::hasAccumulator() const {
  if (scale_config_) {
    return scale_config_->hasAccumulator();
  }
  if (scale_) {
    return scale_->hasAccumulator();
  }
  return false;
}

void EnumRenderProperty::validateValue(const bool has_scale,
                                       const rapidjson::Pointer& value_path) {
  int val = 0;
  if (uniform_val_.getType() == QueryDataType::STRING) {
    RUNTIME_EX_ASSERT(string_convert_func_ != nullptr,
                      JSONRefErrorLogger(ctx_.getRenderSessionKey(),
                                         value_path,
                                         "Enum property does not support strings."));

    auto strval = uniform_val_.getStringVal();
    val = string_convert_func_(strval);

    RUNTIME_EX_ASSERT(val >= 0,
                      JSONRefErrorLogger(ctx_.getRenderSessionKey(),
                                         value_path,
                                         "Enum value " + strval + " is not supported."));
  } else {
    val = uniform_val_.getVal<int>();
  }

  uniform_val_.set(QueryDataType::INT, val);
}

void BoolRenderProperty::validateValue(const bool has_scale,
                                       const rapidjson::Pointer& value_path) {
  int val = 0;
  if (uniform_val_.getType() == QueryDataType::STRING) {
    RUNTIME_EX_ASSERT(string_convert_func_ != nullptr,
                      JSONRefErrorLogger(ctx_.getRenderSessionKey(),
                                         value_path,
                                         "Bool property does not support strings."));

    auto strval = uniform_val_.getStringVal();
    val = string_convert_func_(strval);

    RUNTIME_EX_ASSERT(val >= 0,
                      JSONRefErrorLogger(ctx_.getRenderSessionKey(),
                                         value_path,
                                         "Bool value " + strval + " is not supported."));
  } else {
    val = uniform_val_.getVal<int>();
  }

  uniform_val_.set(QueryDataType::INT, val);
}

}  // namespace QueryRenderer
