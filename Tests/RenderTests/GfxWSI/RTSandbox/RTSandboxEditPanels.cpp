/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxWSI/RTSandbox/RTSandboxEditPanels.h"

#include <glm/gtc/type_ptr.hpp>

#include "GfxDriver/Utils/LoggingUtils.h"
#include "GfxDriver/WSI/EditPanels/CameraEditPanel.h"
#include "GfxDriver/WSI/EditPanels/EditPanelConstants.h"
#include "GfxDriver/WSI/EditPanels/LightEditPanel.h"
#include "GfxDriver/WSI/EditPanels/PBRMaterialEditPanel.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/HeightfieldEditPanel.h"

using namespace gfx;

//
// Settings panel
//
static constexpr CameraEditPanel::Properties camera_props_to_show =
    CameraEditPanel::Properties::kType | CameraEditPanel::Properties::kFOV;

SettingsEditPanel::SettingsEditPanel(const std::string& name,
                                     RTSandboxUI::State& state,
                                     Camera& camera)
    : RTSandboxEditPanel(name, state)
    , camera_edit_panel_{
          std::make_unique<CameraEditPanel>(camera, camera_props_to_show)} {}

bool SettingsEditPanel::draw() {
  if (ImGui::Begin(name_.c_str(), nullptr, ImGuiWindowFlags_NoFocusOnAppearing)) {
    //
    // Rendering
    //
    if (ImGui::CollapsingHeader("Render", ImGuiTreeNodeFlags_DefaultOpen)) {
      state_.uniforms_changed =
          ImGui::SliderInt("AA Samples", &state_.num_aa_samples, 1, 32) ||
          state_.uniforms_changed;
      state_.uniforms_changed =
          ImGui::SliderInt("Max Ray Depth", &state_.max_ray_depth, 1, 16) ||
          state_.uniforms_changed;
      ImGui::Spacing();
    }

    //
    // Camera
    //
    if (ImGui::CollapsingHeader(camera_edit_panel_->getName().c_str())) {
      if (camera_edit_panel_->draw()) {
        state_.matrices_changed = true;
      }
      ImGui::Checkbox("Animation", &state_.do_animation);
    }
  }
  ImGui::End();
  return false;
}

//
// Scene panel
//
static constexpr PBRMaterialEditPanel::Properties material_props_to_show =
    PBRMaterialEditPanel::Properties::kAlbedo | PBRMaterialEditPanel::Properties::kBRDF;
static constexpr LightEditPanel::Properties light_props_to_show =
    LightEditPanel::Properties::kAll;
SceneEditPanel::SceneEditPanel(const std::string& name,
                               RTSandboxUI::State& state,
                               HeightfieldCallback& heightfield,
                               PBRScene& scene)
    : RTSandboxEditPanel(name, state)
    , heightfield_edit_panel_{std::make_unique<HeightfieldEditPanel>(heightfield)}
    , material_edit_panel_{std::make_unique<PBRMaterialEditPanel>(scene,
                                                                  material_props_to_show)}
    , light_edit_panel_{std::make_unique<LightEditPanel>(scene, light_props_to_show)} {}

bool SceneEditPanel::draw() {
  if (ImGui::Begin(name_.c_str(), nullptr, ImGuiWindowFlags_NoFocusOnAppearing)) {
    if (ImGui::BeginTabBar("SceneTabs",
                           ImGuiTabBarFlags_None | ImGuiTabBarFlags_Reorderable)) {
      if (ImGui::BeginTabItem("Objects")) {
        drawObjectsPanel();
        ImGui::EndTabItem();
      }

      //
      // Materials tab
      //
      if (ImGui::BeginTabItem("Materials")) {
        state_.uniforms_changed =
            ImGui::Checkbox("Show Triangles", &state_.do_show_triangles) ||
            state_.uniforms_changed;
        state_.uniforms_changed =
            ImGui::Checkbox("Glossy Reflections", &state_.do_glossy_reflections) ||
            state_.uniforms_changed;

        material_edit_panel_->draw();
        ImGui::EndTabItem();
      }

      //
      // Lights tab
      //
      if (ImGui::BeginTabItem("Lights")) {
        light_edit_panel_->draw();
        ImGui::EndTabItem();
      }

      //
      // Environment tab
      //
      if (ImGui::BeginTabItem("Environment")) {
        bool changed =
            ImGui::ColorEdit3("Color 1",
                              (float*)glm::value_ptr(state_.bg_gradient_colors[0]),
                              kColorEditFlags);
        changed = ImGui::ColorEdit3("Color 2",
                                    (float*)glm::value_ptr(state_.bg_gradient_colors[1]),
                                    kColorEditFlags) ||
                  changed;

        if (changed) {
          state_.uniforms_changed = true;
        }
        ImGui::EndTabItem();
      }

      ImGui::EndTabBar();
    }
  }
  ImGui::End();
  return false;
}

void SceneEditPanel::drawObjectsPanel() {
  if (ImGui::CollapsingHeader("Heightfield", ImGuiTreeNodeFlags_DefaultOpen)) {
    bool changed = false;
    changed = ImGui::Checkbox("As Spheres", &state_.hf_as_spheres);
#if DEMO_MODE
    // ImGui::SameLine(0.0f, viewport_offset_ * 0.2f);
    ImGui::SameLine(0.0f, viewport_offset_ - 300);
    ImGui::PushItemWidth(80);
#else
    ImGui::SameLine(0.0f, 100.0f);
    ImGui::PushItemWidth(120);
#endif
    {
      ImGuiBridge::ScopedDisable guard(state_.hf_as_spheres);
      changed =
          ImGui::DragFloat("Scale", &state_.hf_as_spheres_scale, 0.01f, 0.01f, 5.0f) ||
          changed;
    }
    ImGui::PopItemWidth();

    changed = ImGui::SliderInt("Tesselation", &state_.hf_tesselation, 1, 5000) || changed;

    changed = heightfield_edit_panel_->draw() || changed;

    state_.objects_changed = changed;
  }
}

//
// Info panel
//
InfoEditPanel::InfoEditPanel(const std::string& name,
                             RTSandboxUI::State& state,
                             ImGuiBridge& imgui_bridge,
                             const DeviceContext& device,
                             const SceneStats& scene_stats,
                             const TimingStats& timing_stats)
    : RTSandboxEditPanel(name, state)
    , imgui_bridge_{imgui_bridge}
    , device_{device}
    , scene_stats_{scene_stats}
    , timing_stats_{timing_stats} {}

bool InfoEditPanel::draw() {
  if (ImGui::Begin(name_.c_str(), nullptr, ImGuiWindowFlags_NoFocusOnAppearing)) {
    if (ImGui::BeginTabBar("StatsTabs", ImGuiTabBarFlags_None)) {
      std::stringstream ss;
      StreamStatFormatter log_stat(ss);
      if (ImGui::BeginTabItem("Stats")) {
        imgui_bridge_.pushFont(ImGuiBridge::Font::kFixedWidth);
        ss << "-- Geometry totals --\n";
        log_stat("Vertices", scene_stats_.total_vertices);
        log_stat("Triangles", scene_stats_.total_triangles);
        log_stat("AABBs", scene_stats_.total_aabbs);

        ss << "\n-- Heightfield --\n";
        log_stat("Vertices", scene_stats_.num_hf_vertices);
        log_stat("Triangles", scene_stats_.num_hf_triangles);
        log_stat("Mesh time", timing_stats_.hf_mesh_build_time, " ms");
        log_stat("BLAS time", timing_stats_.hf_blas_build_time, " ms");
        log_stat("Mesh to Spheres time", timing_stats_.hf_mesh_to_spheres_time, " ms");
        ImGui::TextUnformatted(ss.str().c_str());
        imgui_bridge_.popFont();
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Memory")) {
        std::stringstream ss;
        device_.logMemoryBudgetInfo(ss);
        ss << "\n";
        device_.getResourceManager().logMemorySummary(ss);
        imgui_bridge_.pushFont(ImGuiBridge::Font::kFixedWidth);
        ImGui::Text("%s", ss.str().c_str());
        imgui_bridge_.popFont();
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
  }
  ImGui::End();
  return false;
}

//
// Help panel
//
HelpEditPanel::HelpEditPanel(const std::string& name, RTSandboxUI::State& state)
    : RTSandboxEditPanel(name, state) {}

bool HelpEditPanel::draw() {
  if (ImGui::Begin(name_.c_str(), nullptr, ImGuiWindowFlags_NoFocusOnAppearing)) {
    static HelpInputTable hotkey_table = {
        {{"", "A", "Toggle animation"}, {"", "B", "Cycle background"}},
        {{"", "R", "Reload shaders"}},
        {{"", "S", "Show stats overlay"}, {"", "U", "Show UI"}}};

    static HelpInputTableGroup mouse_actions = {{"", "L drag", "Orbit camera"},
                                                {"Shift", "L drag", "Pan camera"},
                                                {"", "R drag", "Pan camera"},
                                                {"", "V scroll", "Truck camera"}};

    ImGui::CollapsingHeader("Hotkeys", ImGuiTreeNodeFlags_Leaf);
    draw_modkey_help_text();
    draw_hotkey_help(hotkey_table, true, true);

    ImGui::CollapsingHeader("Mouse / Trackpad", ImGuiTreeNodeFlags_Leaf);
    draw_mouse_action_help_table(mouse_actions, true, true);
    ImGui::EndTabBar();
  }
  ImGui::End();
  return false;
}
