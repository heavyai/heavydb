#include "Tests/RenderTests/GfxWSI/RTSandbox/HeightfieldEditPanel.h"

#include <imgui/imgui.h>

#include "GfxDriver/Drivers/Vulkan/WSI/ImGuiBridge.h"

using namespace gfx;

static std::vector<std::string> g_distance_metric_strings = {{"Euclidean"},
                                                             {"Euclidean Squared"},
                                                             {"Manhatten"},
                                                             {"Euclidean Manhatten"},
                                                             {"Radial Manhatten"}};

static std::vector<std::string> g_hf_generator_strings = {{"Noise"}, {"Cell"}, {"Chips"}};

HeightfieldEditPanel::HeightfieldEditPanel(HeightfieldCallback& hf)
    : EditPanel("Heightfield"), hf_{hf} {}

bool HeightfieldEditPanel::draw() {
  bool did_change = ImGui::SliderFloat("Height", &hf_.height, -1.5f, 1.5f);
  ImGui::Separator();

  ImGui::Text("Noise:");

  if (ImGuiBridge::doComboBox<HeightfieldCallback::GeneratorType>(
          "Generator", g_hf_generator_strings, hf_.generator)) {
    did_change = true;
  }

  ImGui::Spacing();

  did_change = ImGui::Checkbox("Invert", &hf_.invert) || did_change;
  did_change = ImGui::SliderFloat("Low", &hf_.low, 0.0f, 1.0f) || did_change;
  did_change = ImGui::SliderFloat("High", &hf_.high, 0.0f, 1.0f) || did_change;

  ImGui::Spacing();

  did_change =
      ImGui::SliderFloat("Noise Size", &hf_.noise_size, 0.0f, 10.0f) || did_change;
  did_change = ImGui::SliderInt("Noise Levels", &hf_.noise_levels, 1, 12) || did_change;

  ImGui::Spacing();

  // Lacunarity doesn't seem useful. Leaving it out for now
  // did_change = ImGui::SliderFloat("H", &hf_H_, 0.0f, 5.0f) || did_change;
  // did_change =
  //     ImGui::SliderFloat("Lacunarity", &hf_lacunarity_, 0.0f, 5.0f) ||
  //     did_change;
  {
    ImGuiBridge::ScopedDisable guard(
        hf_.generator == HeightfieldCallback::GeneratorType::kCell ||
        hf_.generator == HeightfieldCallback::GeneratorType::kChips);
    if (ImGuiBridge::doComboBox<CellDistanceMetric>(
            "Metric", g_distance_metric_strings, hf_.distance_metric)) {
      did_change = true;
    }
  }

  return did_change;
}
