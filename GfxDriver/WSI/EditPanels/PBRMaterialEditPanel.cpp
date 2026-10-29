#include "GfxDriver/WSI/EditPanels/PBRMaterialEditPanel.h"

#include <imgui/imgui.h>
#include <glm/gtc/type_ptr.hpp>

#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"
#include "GfxDriver/WSI/EditPanels/EditPanelConstants.h"

namespace gfx {

PBRMaterialEditPanel::PBRMaterialEditPanel(PBRScene& scene, Properties props_to_show)
    : EditPanel("Material")
    , scene_{scene}
    , props_to_show_{props_to_show}
    , selected_material_{-1} {}

bool PBRMaterialEditPanel::draw() {
  bool did_change = false;
  if (auto selected_material = drawSelectTable(); selected_material != -1) {
    did_change = drawEditPanel(selected_material);
  }
  return did_change;
}

int PBRMaterialEditPanel::drawSelectTable() {
  if (ImGui::CollapsingHeader("Material Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
    auto do_show_albedo = any_bits_set(props_to_show_ & Properties::kAlbedo);
    int num_columns = do_show_albedo ? 2 : 1;
    if (ImGui::BeginTable("MaterialTable", num_columns, kSelectTableFlags)) {
      if (do_show_albedo) {
        ImGui::TableSetupColumn(
            "Color", ImGuiTableColumnFlags_WidthFixed, kSelectTableColorColumnWidth);
      }
      ImGui::TableSetupColumn("Name", kSelectTableColumnFlags);

      scene_.iterateMaterialDescs([&](PBRMaterialDescriptor& material) {
        bool did_change = false;
        int index = static_cast<int>(material.getIndex());
        const bool item_is_selected = index == selected_material_;

        std::string base_name = "Material_" + std::to_string(index);
        ImGui::PushID(index);
        ImGui::TableNextRow(ImGuiTableRowFlags_None);
        ImGui::TableSetColumnIndex(0);
        if (do_show_albedo) {
          did_change = ImGui::ColorEdit3(std::string("##" + base_name + "Albedo").c_str(),
                                         (float*)glm::value_ptr(material.albedo),
                                         kColorEditFlags) ||
                       did_change;
          ImGui::TableSetColumnIndex(1);
        }
        if (ImGui::Selectable(base_name.c_str(),
                              item_is_selected,
                              ImGuiSelectableFlags_SpanAllColumns)) {
          selected_material_ = index;
        }
        ImGui::PopID();
        if (did_change) {
          scene_.updateMaterialProperties(material);
          scene_.updateMaterialDataBuffer();
        }
      });

      ImGui::EndTable();
    }
  }
  return selected_material_;
}

bool PBRMaterialEditPanel::drawEditPanel(int material_index) {
  static std::vector<std::string> color_source_strings = {{"Constant Color"},
                                                          {"Object Coords"}};
  static std::vector<std::string> shading_model_strings = {
      {"Constant Shading"}, {"Blinn-Phong"}, {"Cook-Torrance"}};

  bool did_change = false;
  if (ImGui::CollapsingHeader("Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
    auto& material_desc = scene_.getMaterialDescriptor(material_index);
    std::string base_name = "Material_" + std::to_string(material_index);

    // Shading Model
    if (any_bits_set(props_to_show_ & Properties::kShadingModel) &&
        ImGuiBridge::doComboBox<PBRMaterialDescriptor::ShadingModel>(
            "Shading Model", shading_model_strings, material_desc.shading_model)) {
      did_change = true;
    }

    ImGui::Spacing();

    // Albedo
    if (any_bits_set(props_to_show_ & Properties::kAlbedo)) {
      if (any_bits_set(props_to_show_ & Properties::kColorSource) &&
          ImGuiBridge::doComboBox<PBRMaterialDescriptor::ColorSource>(
              "Albedo Source", color_source_strings, material_desc.albedo_source)) {
        did_change = true;
      }

      did_change = ImGui::ColorEdit3(std::string("##" + base_name + " Albedo").c_str(),
                                     (float*)glm::value_ptr(material_desc.albedo),
                                     kColorEditFlags) ||
                   did_change;
      ImGui::SameLine();
      ImGui::Text("Albedo");
    }

    // Opacity
    if (any_bits_set(props_to_show_ & Properties::kOpacity)) {
      did_change =
          ImGui::SliderFloat("Opacity", &material_desc.opacity, 0.0f, 1.0f) || did_change;
    }

    // BRDF properties
    if (any_bits_set(props_to_show_ & Properties::kBRDF)) {
      if (material_desc.shading_model ==
          PBRMaterialDescriptor::ShadingModel::kBlinnPhong) {
        did_change =
            ImGui::SliderFloat("Glossiness", &material_desc.glossiness, 0.0f, 100.0f) ||
            did_change;
        did_change =
            ImGui::SliderFloat(
                "Spec Strength", &material_desc.specular_strength, 0.0f, 10.0f) ||
            did_change;
      } else if (material_desc.shading_model ==
                 PBRMaterialDescriptor::ShadingModel::kCookTorrance) {
        did_change =
            ImGui::SliderFloat("Roughness", &material_desc.roughness, 0.0f, 1.0f) ||
            did_change;
        did_change =
            ImGui::SliderFloat("Metallic", &material_desc.metallic, 0.0f, 1.0f) ||
            did_change;
      }
    }

    if (did_change) {
      scene_.updateMaterialProperties(material_desc);
      scene_.updateMaterialDataBuffer();
    }
  }
  return did_change;
}

}  // namespace gfx
