/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "GfxDriver/Pipeline/PushConstantRanges.h"
#include "GfxDriver/Resources/Enums.h"

namespace gfx {

// add new values as required
// keep in sync with string_view conversions below
enum class FaceCullMode : uint8_t { kNone, kFront, kBack, kFrontAndBack };
enum class DepthFunc : uint8_t { kLessOrEqual, kGreaterOrEqual };
enum class BlendEquation : uint8_t { kAdd, kMax };
enum class StencilOp : uint8_t { kKeep, kInvert, kZero };
enum class StencilFunc : uint8_t { kAlways, kEqual, kNotEqual };
enum class BlendFunc : uint8_t {
  kOne,
  kZero,
  kSrcColor,
  kOneMinusSrcColor,
  kDstColor,
  kOneMinusDstColor,
  kSrcAlpha,
  kOneMinusSrcAlpha,
  kDstAlpha,
  kOneMinusDstAlpha,
  kConstantAlpha,
  kOneMinusConstantAlpha
};

// string_view conversions for enum class above
static constexpr std::array<std::string_view, 4> face_cull_mode_to_string = {
    "NONE",
    "FRONT",
    "BACK",
    "FRONT_AND_BACK"};
static constexpr std::array<std::string_view, 2> depth_func_to_string = {
    "LESS_OR_EQUAL",
    "GREATER_OR_EQUAL"};
static constexpr std::array<std::string_view, 12> blend_func_to_string = {
    "ONE",
    "ZERO",
    "SRC_COLOR",
    "ONE_MINUS_SRC_COLOR",
    "DST_COLOR",
    "ONE_MINUS_DST_COLOR",
    "SRC_ALPHA",
    "ONE_MINUS_SRC_ALPHA",
    "DST_ALPHA",
    "ONE_MINUS_DST_ALPHA"
    "CONSTANT_ALPHA",
    "ONE_MINUS_CONSTANT_ALPHA"};
static constexpr std::array<std::string_view, 2> blend_equation_to_string = {"FUNC_ADD",
                                                                             "MAX"};
static constexpr std::array<std::string_view, 3> stencil_op_to_string = {"KEEP",
                                                                         "INVERT",
                                                                         "ZERO"};
static constexpr std::array<std::string_view, 3> stencil_func_to_string = {"ALWAYS",
                                                                           "EQUAL",
                                                                           "NOT_EQUAL"};

class PipelineDescriptor {
 public:
  PipelineDescriptor();

  void setSubpassIndex(uint32_t value) { subpass_index_ = value; }
  void setEnableBlend(bool value) { enable_blend_ = value; }
  void setRasterSampleCount(RasterSampleCount value) { raster_sample_count_ = value; }
  void setEnableDepthTest(bool value) { enable_depth_test_ = value; }
  void setEnableStencilTest(bool value) { enable_stencil_test_ = value; }
  void setFaceCullMode(FaceCullMode value) { face_cull_mode_ = value; }
  void setDepthFunc(DepthFunc value) { depth_func_ = value; }
  void setEnableColorWrites(bool value) { enable_color_writes_ = value; }
  void setEnableDepthWrites(bool value) { enable_depth_writes_ = value; }
  void setStencilMask(uint32_t value) { stencil_mask_ = value; }
  void setBlendColor(float r, float g, float b, float a) { blend_color_ = {r, g, b, a}; }
  void setBlendFunc(BlendFunc src, BlendFunc dst) {
    blend_func_src_ = src;
    blend_func_dst_ = dst;
  }
  void setAlphaBlendFunc(BlendFunc src, BlendFunc dst) {
    alpha_blend_func_src_ = src;
    alpha_blend_func_dst_ = dst;
    has_alpha_blend_func_ = true;
  }
  void setBlendEquation(BlendEquation eqn) { blend_equation_ = eqn; }
  void setAlphaBlendEquation(BlendEquation eqn) {
    alpha_blend_equation_ = eqn;
    has_alpha_blend_equation_ = true;
  }
  void setStencilOp(StencilOp sfail, StencilOp dpfail, StencilOp dppass) {
    stencil_op_sfail_ = sfail;
    stencil_op_dpfail_ = dpfail;
    stencil_op_dppass_ = dppass;
  }
  void setStencilFunc(StencilFunc func, int32_t ref, uint32_t mask) {
    stencil_func_ = func;
    stencil_func_ref_ = ref;
    stencil_func_mask_ = mask;
  }
  void setPushConstantRanges(const PushConstantRanges& value) {
    push_constant_ranges_ = value;
  }

  inline uint32_t getSubpassIndex() const { return subpass_index_; }
  inline bool getEnableBlend() const { return enable_blend_; }
  inline RasterSampleCount getRasterSampleCount() const { return raster_sample_count_; }
  inline bool getEnableDepthTest() const { return enable_depth_test_; }
  inline bool getEnableStencilTest() const { return enable_stencil_test_; }
  inline FaceCullMode getFaceCullMode() const { return face_cull_mode_; }
  inline DepthFunc getDepthFunc() const { return depth_func_; }
  inline bool getEnableColorWrites() const { return enable_color_writes_; }
  inline bool getEnableDepthWrites() const { return enable_depth_writes_; }
  inline uint32_t getStencilMask() const { return stencil_mask_; }
  inline const std::array<float, 4>& getBlendColor() const { return blend_color_; }
  inline BlendFunc getBlendFuncSrc() const { return blend_func_src_; }
  inline BlendFunc getBlendFuncDst() const { return blend_func_dst_; }
  inline BlendEquation getBlendEquation() const { return blend_equation_; }
  inline bool hasAlphaBlendFunc() const { return has_alpha_blend_func_; }
  inline BlendFunc getAlphaBlendFuncSrc() const { return alpha_blend_func_src_; }
  inline BlendFunc getAlphaBlendFuncDst() const { return alpha_blend_func_dst_; }
  inline bool hasAlphaBlendEquation() const { return has_alpha_blend_equation_; }
  inline BlendEquation getAlphaBlendEquation() const { return alpha_blend_equation_; }
  inline StencilOp getStencilOpStencilFail() const { return stencil_op_sfail_; }
  inline StencilOp getStencilOpDepthFail() const { return stencil_op_dpfail_; }
  inline StencilOp getStencilOpDepthPass() const { return stencil_op_dppass_; }
  inline StencilFunc getStencilFunc() const { return stencil_func_; }
  inline int32_t getStencilFuncRef() const { return stencil_func_ref_; }
  inline uint32_t getStencilFuncMask() const { return stencil_func_mask_; }
  inline const PushConstantRanges& getPushConstantRanges() const {
    return push_constant_ranges_;
  }
  inline PushConstantRanges& getPushConstantRanges() { return push_constant_ranges_; }

 private:
  uint32_t subpass_index_;
  bool enable_blend_;
  RasterSampleCount raster_sample_count_;
  bool enable_depth_test_;
  bool enable_stencil_test_;
  FaceCullMode face_cull_mode_;
  DepthFunc depth_func_;
  bool enable_color_writes_;
  bool enable_depth_writes_;
  uint32_t stencil_mask_;
  std::array<float, 4> blend_color_;
  BlendFunc blend_func_src_;
  BlendFunc blend_func_dst_;
  BlendFunc alpha_blend_func_src_;
  BlendFunc alpha_blend_func_dst_;
  bool has_alpha_blend_func_;
  BlendEquation blend_equation_;
  BlendEquation alpha_blend_equation_;
  bool has_alpha_blend_equation_;
  StencilOp stencil_op_sfail_;
  StencilOp stencil_op_dpfail_;
  StencilOp stencil_op_dppass_;
  StencilFunc stencil_func_;
  int32_t stencil_func_ref_;
  uint32_t stencil_func_mask_;
  PushConstantRanges push_constant_ranges_;
};

}  // namespace gfx

constexpr std::string_view to_string(const gfx::FaceCullMode value);
constexpr std::string_view to_string(const gfx::DepthFunc value);
constexpr std::string_view to_string(const gfx::BlendEquation value);
constexpr std::string_view to_string(const gfx::StencilOp value);
constexpr std::string_view to_string(const gfx::StencilFunc value);
constexpr std::string_view to_string(const gfx::BlendFunc value);

std::ostream& operator<<(std::ostream& os, const gfx::FaceCullMode value);
std::ostream& operator<<(std::ostream& os, const gfx::DepthFunc value);
std::ostream& operator<<(std::ostream& os, const gfx::BlendEquation value);
std::ostream& operator<<(std::ostream& os, const gfx::StencilOp value);
std::ostream& operator<<(std::ostream& os, const gfx::StencilFunc value);
std::ostream& operator<<(std::ostream& os, const gfx::BlendFunc value);
