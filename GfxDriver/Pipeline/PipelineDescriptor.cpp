/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Pipeline/PipelineDescriptor.h"

namespace gfx {

// these defaults allow a default PD to be used for most of our render operations

PipelineDescriptor::PipelineDescriptor()
    : subpass_index_{0u}
    , enable_blend_{true}
    , raster_sample_count_{RasterSampleCount::k1}
    , enable_depth_test_{false}
    , enable_stencil_test_{false}
    , face_cull_mode_{FaceCullMode::kNone}
    , depth_func_{DepthFunc::kGreaterOrEqual}
    , enable_color_writes_{true}
    , enable_depth_writes_{true}
    , stencil_mask_{0xFFFFFFFFU}
    , blend_color_{{0.0f, 0.0f, 0.0f, 0.0f}}
    , blend_func_src_{BlendFunc::kOne}
    , blend_func_dst_{BlendFunc::kOneMinusSrcAlpha}
    , alpha_blend_func_src_{BlendFunc::kOne}
    , alpha_blend_func_dst_{BlendFunc::kOneMinusSrcAlpha}
    , has_alpha_blend_func_{false}
    , blend_equation_{BlendEquation::kAdd}
    , alpha_blend_equation_{BlendEquation::kAdd}
    , has_alpha_blend_equation_{false}
    , stencil_op_sfail_{StencilOp::kKeep}
    , stencil_op_dpfail_{StencilOp::kKeep}
    , stencil_op_dppass_{StencilOp::kKeep}
    , stencil_func_{StencilFunc::kAlways}
    , stencil_func_ref_{0}
    , stencil_func_mask_{0xFFFFFFFF} {}

}  // namespace gfx

constexpr std::string_view to_string(const gfx::FaceCullMode value) {
  return gfx::face_cull_mode_to_string[static_cast<int>(value)];
}

constexpr std::string_view to_string(const gfx::DepthFunc value) {
  return gfx::depth_func_to_string[static_cast<int>(value)];
}

constexpr std::string_view to_string(const gfx::BlendFunc value) {
  return gfx::blend_func_to_string[static_cast<int>(value)];
}

constexpr std::string_view to_string(const gfx::BlendEquation value) {
  return gfx::blend_equation_to_string[static_cast<int>(value)];
}

constexpr std::string_view to_string(const gfx::StencilOp value) {
  return gfx::stencil_op_to_string[static_cast<int>(value)];
}

constexpr std::string_view to_string(const gfx::StencilFunc value) {
  return gfx::stencil_func_to_string[static_cast<int>(value)];
}

std::ostream& operator<<(std::ostream& os, const gfx::FaceCullMode value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const gfx::DepthFunc value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const gfx::BlendEquation value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const gfx::StencilOp value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const gfx::StencilFunc value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, const gfx::BlendFunc value) {
  os << to_string(value);
  return os;
}
