/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <glm/vec3.hpp>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/Types.h"

class Background {
 public:
  enum Type { kNone, kSolidColor, kGridLines, kSymbolEdit };

  explicit Background(const gfx::DeviceContext& device,
                      const std::vector<Type>& supported_types);
  void init(gfx::RasterSampleCount raster_sample_count,
            const gfx::RenderPass& render_pass);
  void shutdown();

  void setImageSize(uint32_t width, uint32_t height);

  // Set the symbol flag bits. Used to detect symbol symmetry and shade
  // grid quads to highlight the defining regions
  void setSymbolFlagBits(uint32_t bits);

  // drawUI adds controls to whatever ImGui stack it's in,
  // a main window, collapsable header, tab pane, etc
  void drawUI();

  // draw must be called within a compatible renderpass
  void draw(gfx::CommandList& cmd_list);

  Type getType() const;
  void setType(Type type);

  // Change to the next type in supported_types, cycling back to
  // start if past the end of the vector
  void cycleType();

 private:
  const gfx::DeviceContext& device_;

  std::unique_ptr<gfx::Material> material_;
  gfx::resource_ptr<gfx::GraphicsPipeline> pipeline_;

  uint32_t type_{0};
  std::vector<Type> supported_types_;
  std::vector<std::string> supported_type_strings_;
  uint32_t symbol_flag_bits_{0u};

  glm::vec3 solid_color_{0.4, 0.4, 0.4};
  glm::vec3 grid_color_{0.2, 0.2, 0.2};

  void setTypeUniform();
};
