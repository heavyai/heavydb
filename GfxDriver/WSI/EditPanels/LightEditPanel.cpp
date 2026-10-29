#include "GfxDriver/WSI/EditPanels/LightEditPanel.h"

#include <imgui/imgui.h>
#include <glm/gtc/type_ptr.hpp>

#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"
#include "GfxDriver/WSI/EditPanels/EditPanelConstants.h"

namespace gfx {

static std::vector<std::string> g_light_type_strings = {{"Point"}, {"Parallel"}};

LightEditPanel::LightEditPanel(PBRScene& scene, Properties props_to_show)
    : EditPanel("Lights")
    , scene_{scene}
    , props_to_show_{props_to_show}
    , selected_light_{-1} {}

bool LightEditPanel::draw() {
  bool did_change = false;
  if (auto selected_light = drawSelectTable(); selected_light != -1) {
    did_change = drawEditPanel(selected_light);
  }
  return did_change;
}

int LightEditPanel::drawSelectTable() {
  if (ImGui::CollapsingHeader("Light Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
    auto do_show_color = any_bits_set(props_to_show_ & Properties::kColor);
    int num_columns = do_show_color ? 2 : 1;
    if (ImGui::BeginTable("LightTable", num_columns, kSelectTableFlags)) {
      if (do_show_color) {
        ImGui::TableSetupColumn(
            "Color", ImGuiTableColumnFlags_WidthFixed, kSelectTableColorColumnWidth);
      }
      ImGui::TableSetupColumn("Name", kSelectTableColumnFlags);

      scene_.iterateLightDescs([&](LightDescriptor& light) {
        bool did_change = false;
        int index = static_cast<int>(light.getIndex());
        const bool item_is_selected = index == selected_light_;

        std::string base_name = "Light_" + std::to_string(index);
        ImGui::PushID(index);
        ImGui::TableNextRow(ImGuiTableRowFlags_None);
        ImGui::TableSetColumnIndex(0);
        if (do_show_color) {
          did_change = ImGui::ColorEdit3(std::string("##" + base_name + "Color").c_str(),
                                         (float*)glm::value_ptr(light.color),
                                         kColorEditFlags) ||
                       did_change;
          ImGui::TableSetColumnIndex(1);
        }
        if (ImGui::Selectable(base_name.c_str(),
                              item_is_selected,
                              ImGuiSelectableFlags_SpanAllColumns)) {
          selected_light_ = index;
        }
        ImGui::PopID();
        if (did_change) {
          scene_.updateLightProperties(light);
          scene_.updateLightDataBuffer();
        }
      });
      ImGui::EndTable();
    }
  }
  return selected_light_;
}

bool LightEditPanel::drawEditPanel(int light_index) {
  bool did_change = false;
  auto& light = scene_.getLightDescriptor(light_index);
  std::string base_name = "Light_" + std::to_string(light_index);
  if (ImGui::CollapsingHeader("Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
    // Type
    if (any_bits_set(props_to_show_ & Properties::kType)) {
      did_change = (ImGuiBridge::doComboBox<LightDescriptor::Type>(
                       "Type", g_light_type_strings, light.type)) ||
                   did_change;
    }
    // Color
    if (any_bits_set(props_to_show_ & Properties::kColor)) {
      did_change = ImGui::ColorEdit3(std::string("##" + base_name + " Color").c_str(),
                                     (float*)glm::value_ptr(light.color),
                                     kColorEditFlags) ||
                   did_change;
      ImGui::SameLine();
      ImGui::Text("Color");
    }
    // Multiplier
    if (any_bits_set(props_to_show_ & Properties::kMultiplier)) {
      did_change =
          ImGui::SliderFloat(std::string("##" + base_name + " Multiplier").c_str(),
                             &light.multiplier,
                             0.0f,
                             10.0f) ||
          did_change;
      ImGui::SameLine();
      ImGui::Text("Multiplier");
    }
    // Decay
    if (any_bits_set(props_to_show_ & Properties::kDecay)) {
      did_change = ImGui::Checkbox(std::string("##" + base_name + "Decay").c_str(),
                                   &light.do_decay) ||
                   did_change;
      ImGui::SameLine();
      ImGui::Text("Decay");
    }
  }

  // Transform
  if (any_bits_set(props_to_show_ & Properties::kTransform) &&
      ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (light.type == LightDescriptor::Type::kPoint) {
      did_change = ImGui::DragFloat3(std::string("##" + base_name + "Position").c_str(),
                                     (float*)glm::value_ptr(light.position),
                                     0.1f) ||
                   did_change;
      ImGui::SameLine();
      ImGui::Text("Position");
    } else {
      if (ImGui::SliderFloat3(std::string("##" + base_name + "Direction").c_str(),
                              (float*)glm::value_ptr(light.direction),
                              -1.0f,
                              1.0f)) {
        light.direction = glm::normalize(light.direction);
        did_change = true;
      }
      ImGui::SameLine();
      ImGui::Text("Direction");
    }
  }

  if (did_change) {
    scene_.updateLightProperties(light);
    scene_.updateLightDataBuffer();
  }

  return did_change;
}

}  // namespace gfx
