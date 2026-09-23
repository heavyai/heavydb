/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxWSI/Background.h"

#include <algorithm>
#include <string>
#include <vector>

#include <imgui/imgui.h>
#include <glm/vec2.hpp>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"

using namespace gfx;

static std::vector<std::string> g_type_strings = {"None", "Solid", "Grid", "Symbol Edit"};

Background::Background(const gfx::DeviceContext& device,
                       const std::vector<Type>& supported_types)
    : device_{device}, supported_types_{supported_types} {
  for (auto type : supported_types) {
    supported_type_strings_.push_back(g_type_strings[static_cast<int>(type)]);
  }
}

void Background::init(RasterSampleCount raster_sample_count,
                      const RenderPass& render_pass) {
  auto& resource_mgr = device_.getResourceManager();

  auto caches = resource_mgr.getShaderManager().createCacheVectorFromTemplate(
      {{"WSITests/fullScreenTriangle.vert"}, {"WSITests/background.frag"}});

  // material
  material_ = resource_mgr.createMaterial("Background", caches);
  material_->updateDescriptorSets();

  material_->setUniformAttribute("mode", (int)type_);
  material_->setUniformAttribute("solidColor", solid_color_);
  material_->setUniformAttribute("gridColor", grid_color_);
  material_->setUniformAttribute("symbolFlags", symbol_flag_bits_);

  // pipeline
  PipelineDescriptor pipeline_desc;
  pipeline_desc.setRasterSampleCount(raster_sample_count);
  pipeline_desc.setEnableDepthTest(false);
  pipeline_desc.setEnableDepthWrites(false);
  pipeline_ =
      resource_mgr.createGraphicsPipeline("Background", *material_, pipeline_desc);
  pipeline_->create(render_pass);
}

void Background::shutdown() {
  if (pipeline_) {
    device_.getResourceManager().destroyPipeline(std::move(pipeline_));
  }
}

void Background::setImageSize(uint32_t width, uint32_t height) {
  material_->setUniformAttribute("imageSize", glm::vec2(width, height));
}

void Background::setSymbolFlagBits(uint32_t bits) {
  symbol_flag_bits_ = bits;
  material_->setUniformAttribute("symbolFlags", bits);
}

void Background::drawUI() {
  static constexpr ImGuiColorEditFlags bg_color_edit_flags =
      ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel;

  if (ImGuiBridge::doComboBox<uint32_t>("Type", supported_type_strings_, type_)) {
    setTypeUniform();
  }
  {  // Primary color
    ImGuiBridge::ScopedDisable guard(type_ != Type::kNone);
    if (ImGui::ColorEdit3("Primary Color", (float*)&solid_color_, bg_color_edit_flags)) {
      material_->setUniformAttribute("solidColor", solid_color_);
    }
  }
  {  // Grid color
    ImGuiBridge::ScopedDisable guard(type_ == Type::kGridLines);
    if (ImGui::ColorEdit3("Grid Color", (float*)&grid_color_, bg_color_edit_flags)) {
      material_->setUniformAttribute("gridColor", grid_color_);
    }
  }
}

Background::Type Background::getType() const {
  return supported_types_[type_];
}

void Background::setType(Type type) {
  auto found = std::find(supported_types_.begin(), supported_types_.end(), type);
  CHECK(found != supported_types_.end());
  type_ = found - supported_types_.begin();
  setTypeUniform();
}

void Background::cycleType() {
  if (++type_ >= supported_types_.size()) {
    type_ = 0;
  }
  setTypeUniform();
}

void Background::setTypeUniform() {
  material_->setUniformAttribute("mode", static_cast<uint32_t>(supported_types_[type_]));
}

void Background::draw(CommandList& cmd_list) {
  cmd_list.drawFullscreen(*pipeline_);
}
