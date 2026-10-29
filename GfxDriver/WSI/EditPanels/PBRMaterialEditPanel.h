#pragma once

#include "GfxDriver/WSI/EditPanels/EditPanel.h"

#include "GfxDriver/Render/PBRScene.h"
#include "Shared/EnumBitmaskOps.h"

namespace gfx {

class PBRMaterialEditPanel : public EditPanel {
 public:
  enum class Properties : uint32_t {
    kShadingModel = 1 << 0,
    kColorSource = 1 << 1,
    kAlbedo = 1 << 2,
    kOpacity = 1 << 3,
    kBRDF = 1 << 4,
    kAll = 0xFFFFFFFF
  };

  explicit PBRMaterialEditPanel(PBRScene& scene,
                                Properties props_to_show = Properties::kAll);
  ~PBRMaterialEditPanel() override = default;

  bool draw() override;

 protected:
  PBRScene& scene_;
  Properties props_to_show_;
  int selected_material_;

  int drawSelectTable();
  bool drawEditPanel(int material_index);
};

}  // namespace gfx

ENABLE_BITMASK_OPS(::gfx::PBRMaterialEditPanel::Properties);
