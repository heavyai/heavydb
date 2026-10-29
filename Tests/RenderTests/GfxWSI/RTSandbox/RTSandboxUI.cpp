/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxWSI/RTSandbox/RTSandboxUI.h"

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

#include "GfxDriver/Utils/LoggingUtils.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/RTSandboxEditPanels.h"

using namespace gfx;

RTSandboxUI::RTSandboxUI(RTSandbox& rt_sandbox,
                         DeviceContext& device,
                         WindowSystemIntegration& wsi,
                         ImGuiBridge& imgui_bridge,
                         Camera& camera,
                         PBRScene& scene,
                         HeightfieldCallback& heightfield,
                         const TimingStats& timing_stats,
                         const SceneStats& scene_stats)
    : rt_sandbox_{rt_sandbox}
    , device_{device}
    , wsi_{wsi}
    , imgui_bridge_{imgui_bridge}
    , camera_{camera}
    , scene_{scene}
    , timing_stats_{timing_stats}
    , scene_stats_{scene_stats} {
  // Create panels
  edit_panels_.emplace_back(
      std::make_unique<SettingsEditPanel>("Settings", state_, camera_));
  edit_panels_.emplace_back(
      std::make_unique<SceneEditPanel>("Scene", state_, heightfield, scene_));
  edit_panels_.emplace_back(std::make_unique<InfoEditPanel>(
      "Info", state_, imgui_bridge_, device_, scene_stats_, timing_stats_));
  edit_panels_.emplace_back(std::make_unique<HelpEditPanel>("Help", state_));

  // Create stats overlay
  stats_overlay_ = std::make_unique<ImGuiBridge::Overlay>(
      imgui_bridge_,
      "Time",
      ImGuiBridge::Font::kFixedWidth,
      ImGuiBridge::WindowPos::kBottomRight,
      show_stats_overlay_,
      [this]() {
        {
          std::stringstream ss;
          StreamStatFormatter format(ss);
          double avg_time = timing_stats_.total_time_ms /
                            static_cast<double>(timing_stats_.num_renders);
          format("Avg Render Time (ms):", avg_time);
          format("Avg FPS", static_cast<int>(1000.0 / avg_time));
          ImGui::Text("%s", ss.str().c_str());
        }
        ImGui::Separator();
        {
          std::stringstream ss;
          StreamStatFormatter format(ss);
          format("Triangles:", scene_stats_.total_triangles);
          format("AABBs:", scene_stats_.total_aabbs);
          ImGui::Text("%s", ss.str().c_str());
        }
      });
  // Register WSIEventHandler
  wsi_.registerEventHandler([this](const WSIEvent& event) { handleWSIEvent(event); });
}

//
// Dockspace
//
#define SHOW_MENU_BAR false
void RTSandboxUI::buildDockspace() {
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGuiWindowFlags window_flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollWithMouse |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground |
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoDocking;
  if (SHOW_MENU_BAR) {
    window_flags |= ImGuiWindowFlags_MenuBar;
  }

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("RT Dockspace", nullptr, window_flags);
  ImGui::PopStyleVar();
  ImGui::PopStyleVar(2);

  // Submit the DockSpace
  ImGuiID dockspace_id = ImGui::GetID("RTSandboxDockspace");
  ImGui::DockSpace(
      dockspace_id,
      ImVec2(0.0f, 0.0f),
      ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_AutoHideTabBar);

  static bool first_time = true;
  if (first_time) {
    first_time = false;
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(
        dockspace_id,
        ImGuiDockNodeFlags_DockSpace | ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);

    auto edit_panel_node_id =
        ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.2f, nullptr, nullptr);

    ImGui::DockBuilderSetNodeSize(edit_panel_node_id,
                                  ImVec2(kEditPanelWidth, viewport->Size.y));

    // Dock the primary edit panels
    for (auto const& panel : edit_panels_) {
      ImGui::DockBuilderDockWindow(panel->getName().c_str(), edit_panel_node_id);
    }
    ImGui::DockBuilderFinish(dockspace_id);
  }

  auto dock_main = ImGui::DockBuilderGetCentralNode(dockspace_id);
  CHECK(dock_main);
  if (state_.viewport_offset != dock_main->Pos.x) {
    state_.viewport_offset = dock_main->Pos.x;
    state_.uniforms_changed = true;
  }
  if (state_.viewport_width != dock_main->Size.x ||
      state_.viewport_height != dock_main->Size.y) {
    resizeViewport(dock_main->Size.x, dock_main->Size.y);
  }

  //
  // Main menu bar
  //
  if (SHOW_MENU_BAR) {
    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        ImGui::MenuItem("Load", nullptr, nullptr);
        ImGui::MenuItem("Save", nullptr, nullptr);
        ImGui::EndMenu();
      }
      ImGui::EndMenuBar();
    }
  }
  ImGui::End();
}

//
// draw
//
void RTSandboxUI::draw(gfx::Framebuffer& framebuffer) {
  imgui_bridge_.beginRecording();

  if (show_ui_) {
    buildDockspace();

    //
    // Main panel
    //
    for (auto& panel : edit_panels_) {
      panel->draw();
    }
    static bool first_frame = true;
    if (first_frame) {
      ImGui::SetWindowFocus(edit_panels_[0]->getName().c_str());
      first_frame = false;
    }
  }

  //
  // Overlay
  //
  if (show_stats_overlay_) {
    stats_overlay_->draw();
  }

  imgui_bridge_.endRecordingAndDraw(framebuffer);
}

void RTSandboxUI::resizeViewport(uint32_t width, uint32_t height) {
  state_.uniforms_changed = true;
  state_.matrices_changed = true;
  state_.viewport_changed = true;
  state_.viewport_width = width;
  state_.viewport_height = height;
  camera_.setAspect(static_cast<float>(state_.viewport_width) /
                    static_cast<float>(state_.viewport_height));

  std::stringstream ss;
  ss << "RTSandbox "
     << " [" << state_.viewport_width << " x " << state_.viewport_height << "]";
  wsi_.setWindowTitle(ss.str());
}

//
// WSI Events
//
void RTSandboxUI::handleWSIEvent(const WSIEvent& event) {
  using e = WSIEvent::Type;
  switch (event.type()) {
    case e::kKeyboard:
      handleKeyboardEvent(static_cast<const WSIKeyboardEvent&>(event));
      break;
    case e::kMouseButton:
      handleMouseButtonEvent(static_cast<const WSIMouseButtonEvent&>(event));
      break;
    case e::kCursor:
      handleMouseCursorEvent(static_cast<const WSIMouseCursorEvent&>(event));
      break;
    case e::kScroll:
      handleScrollEvent(static_cast<const WSIScrollEvent&>(event));
      break;
    default:
      return;
  }
}

void RTSandboxUI::handleKeyboardEvent(const WSIKeyboardEvent& event) {
  // Handle keys we care about first
  if (event.action() == WSIKeyboardAction::kPress) {
    switch (event.key()) {
      // Animation toggle
      case WSIKeyboardKey::kA: {
        state_.do_animation = !state_.do_animation;
      } break;

      // Reload shaders
      case WSIKeyboardKey::kR:
        rt_sandbox_.reloadShaders();
        break;

      // Toggle stats overlay
      case WSIKeyboardKey::kS:
        show_stats_overlay_ = !show_stats_overlay_;
        break;

      // Toggle UI
      case WSIKeyboardKey::kU:
        show_ui_ = !show_ui_;
        break;

      default:
        break;
    }
  }
}

void RTSandboxUI::handleMouseButtonEvent(const WSIMouseButtonEvent& event) {
  auto& io = ImGui::GetIO();
  auto button = event.button();
  auto action = event.action();
  if ((!io.WantCaptureMouse) && button == WSIMouseButton::kLeft &&
      action == WSIMouseAction::kPress) {
    if (any_bits_set(event.modBits() & WSIKeyboardModBits::kShift)) {
      mouse_capture_mode_ = MouseCaptureMode::kPanCamera;
    } else {
      mouse_capture_mode_ = MouseCaptureMode::kOrbitCamera;
    }
  } else if (button == WSIMouseButton::kRight) {
    if (action == WSIMouseAction::kRelease) {
      mouse_capture_mode_ = MouseCaptureMode::kNone;
    } else {
      mouse_capture_mode_ = MouseCaptureMode::kPanCamera;
    }
  } else if (button == WSIMouseButton::kLeft && action == WSIMouseAction::kRelease) {
    mouse_capture_mode_ = MouseCaptureMode::kNone;
  }
}

void RTSandboxUI::handleMouseCursorEvent(const WSIMouseCursorEvent& event) {
  if (mouse_capture_mode_ != MouseCaptureMode::kNone) {
    glm::vec2 dcursor = (last_cursor_ - glm::vec2(event.x(), event.y())) * 0.002f;
    if (mouse_capture_mode_ == MouseCaptureMode::kOrbitCamera) {
      camera_.orbit(dcursor.x, dcursor.y);
      state_.do_animation = false;
    } else if (mouse_capture_mode_ == MouseCaptureMode::kPanCamera) {
      static constexpr float kPanScale = 0.2f;
      camera_.pan(dcursor.x * kPanScale, dcursor.y * kPanScale);
    }
    state_.matrices_changed = true;
  }
  last_cursor_ = glm::vec2(event.x(), event.y());
}

void RTSandboxUI::handleScrollEvent(const WSIScrollEvent& event) {
  auto& io = ImGui::GetIO();
  if (!io.WantCaptureMouse) {
    camera_.truck(static_cast<float>(event.y()) * 0.1f);
    state_.matrices_changed = true;
  }
}
