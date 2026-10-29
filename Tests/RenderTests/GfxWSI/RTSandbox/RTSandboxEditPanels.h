#pragma once

#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"
#include "GfxDriver/Render/Camera.h"
#include "GfxDriver/WSI/EditPanels/EditPanel.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/HeightfieldCallback.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/RTSandbox.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/RTSandboxUI.h"

//
// TEMP middle layer until I figure out what to do with State
//
class RTSandboxEditPanel : public gfx::EditPanel {
 public:
  explicit RTSandboxEditPanel(const std::string& name, RTSandboxUI::State& state)
      : EditPanel(name), state_{state} {}
  ~RTSandboxEditPanel() override = default;

 protected:
  RTSandboxUI::State& state_;
};

//
// Settings panel
//
class SettingsEditPanel : public RTSandboxEditPanel {
 public:
  SettingsEditPanel(const std::string& name,
                    RTSandboxUI::State& state,
                    gfx::Camera& camera);
  ~SettingsEditPanel() override = default;

  bool draw() override;

 private:
  gfx::EditPanelUqPtr camera_edit_panel_;
};

//
// Scene panel
//
class SceneEditPanel : public RTSandboxEditPanel {
 public:
  SceneEditPanel(const std::string& name,
                 RTSandboxUI::State& state,
                 HeightfieldCallback& heightfield,
                 gfx::PBRScene& scene);
  ~SceneEditPanel() override = default;

  bool draw() override;

 private:
  void drawObjectsPanel();

 private:
  gfx::EditPanelUqPtr heightfield_edit_panel_;
  gfx::EditPanelUqPtr material_edit_panel_;
  gfx::EditPanelUqPtr light_edit_panel_;
};

//
// Info panel
//
class InfoEditPanel : public RTSandboxEditPanel {
 public:
  InfoEditPanel(const std::string& name,
                RTSandboxUI::State& state,
                gfx::ImGuiBridge& imgui_bridge,
                const gfx::DeviceContext& device,
                const SceneStats& scene_stats,
                const TimingStats& timing_stats);
  ~InfoEditPanel() override = default;

  bool draw() override;

 private:
  gfx::ImGuiBridge& imgui_bridge_;
  const gfx::DeviceContext& device_;
  const SceneStats& scene_stats_;
  const TimingStats& timing_stats_;
};

//
// Help panel
//
class HelpEditPanel : public RTSandboxEditPanel {
 public:
  HelpEditPanel(const std::string& name, RTSandboxUI::State& state);
  ~HelpEditPanel() override = default;

  bool draw() override;
};
