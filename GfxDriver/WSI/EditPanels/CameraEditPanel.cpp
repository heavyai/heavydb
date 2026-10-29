#include "GfxDriver/WSI/EditPanels/CameraEditPanel.h"

#include <imgui/imgui.h>

namespace gfx {

CameraEditPanel::CameraEditPanel(Camera& camera, Properties props_to_show)
    : EditPanel("Camera"), camera_{camera}, props_to_show_{props_to_show} {}

bool CameraEditPanel::draw() {
  bool did_change = false;
  // Type
  if (any_bits_set(props_to_show_ & Properties::kType)) {
    auto type = camera_.getType();
    if (ImGui::RadioButton("Look At", type == Camera::Type::kLookAt)) {
      camera_.setType(Camera::Type::kLookAt);
      did_change = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Free", type == Camera::Type::kFree)) {
      camera_.setType(Camera::Type::kFree);
      did_change = true;
    }
  }
  // Projection
  if (any_bits_set(props_to_show_ & Properties::kProjection)) {
    auto proj = camera_.getProjection();
    if (ImGui::RadioButton("Perspective", proj == Camera::Projection::kPerspective)) {
      camera_.setProjection(Camera::Projection::kPerspective);
      did_change = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Orthographic", proj == Camera::Projection::kOrthographic)) {
      camera_.setProjection(Camera::Projection::kOrthographic);
      did_change = true;
    }
  }
  // Field-of-view
  if (any_bits_set(props_to_show_ & Properties::kFOV)) {
    auto fov = camera_.getFOV();
    if (ImGui::SliderFloat("FOV", &fov, 1.0f, 120.0f)) {
      camera_.setFOV(fov);
      did_change = true;
    }
  }

  return did_change;
}

}  // namespace gfx
