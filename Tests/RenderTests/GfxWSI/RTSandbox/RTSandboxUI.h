/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"
#include "GfxDriver/Render/Camera.h"
#include "GfxDriver/WSI/EditPanels/EditPanel.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/HeightfieldCallback.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/RTSandbox.h"

class RTSandboxUI {
 public:
  struct State {
    // general dirty flags
    bool viewport_changed{true};
    bool uniforms_changed{true};
    bool matrices_changed{true};
    bool objects_changed{false};
    bool materials_changed{false};
    bool lights_changed{false};

    // General settings
    int max_ray_depth{3};
    int num_aa_samples{4};
    bool do_show_triangles{false};

    bool do_animation{false};
    bool do_glossy_reflections{true};

    // viewport
    uint32_t viewport_width{0};
    uint32_t viewport_height{0};
    uint32_t viewport_offset{0};

    // camera
    float fov{kDefaultFOV};

    // heightfield
    int hf_tesselation{500};
    bool hf_as_spheres{false};
    float hf_as_spheres_scale{1.0f};

    // Environment
    std::array<glm::vec3, 2> bg_gradient_colors{glm::vec3{0.2f, 0.3f, 1.0f},
                                                glm::vec3{0.8f, 0.6f, 0.3f}};
  };

  RTSandboxUI(RTSandbox& rt_sandbox,
              gfx::DeviceContext& device,
              gfx::WindowSystemIntegration& wsi,
              gfx::ImGuiBridge& imgui_bridge,
              gfx::Camera& camera,
              gfx::PBRScene& scene,
              HeightfieldCallback& heightfield,
              const TimingStats& timing_stats,
              const SceneStats& scene_stats);

  State& getState() { return state_; }
  void draw(gfx::Framebuffer& framebuffer);

  void handleWSIEvent(const gfx::WSIEvent& event);

 private:
  RTSandbox& rt_sandbox_;
  gfx::DeviceContext& device_;
  gfx::WindowSystemIntegration& wsi_;
  gfx::ImGuiBridge& imgui_bridge_;
  std::unique_ptr<gfx::ImGuiBridge::Overlay> stats_overlay_;

  // Primary panels
  std::vector<gfx::EditPanelUqPtr> edit_panels_;

  // Scene
  gfx::Camera& camera_;
  gfx::PBRScene& scene_;

  // UI visibility
  bool show_ui_{true};
  bool show_stats_overlay_{true};

  State state_;
  const TimingStats& timing_stats_;
  const SceneStats& scene_stats_;

  // Cursor capture
  enum class MouseCaptureMode { kNone, kOrbitCamera, kPanCamera };
  glm::vec2 last_cursor_;
  MouseCaptureMode mouse_capture_mode_{MouseCaptureMode::kNone};

  void buildDockspace();
  void resizeViewport(uint32_t width, uint32_t height);

  // WSIEvent handlers
  void handleMouseButtonEvent(const gfx::WSIMouseButtonEvent& event);
  void handleMouseCursorEvent(const gfx::WSIMouseCursorEvent& event);
  void handleKeyboardEvent(const gfx::WSIKeyboardEvent& event);
  void handleScrollEvent(const gfx::WSIScrollEvent& event);
};
