#pragma once

#include "GfxDriver/WSI/EditPanels/EditPanel.h"

#include "GfxDriver/Render/PBRScene.h"
#include "Shared/EnumBitmaskOps.h"

namespace gfx {

class LightEditPanel : public EditPanel {
 public:
  enum class Properties : uint32_t {
    kType = 1 << 0,
    kColor = 1 << 1,
    kMultiplier = 1 << 2,
    kDecay = 1 << 3,
    kTransform = 1 << 4,
    kAll = 0XFFFFFFFF
  };

  explicit LightEditPanel(PBRScene& scene, Properties props_to_show = Properties::kAll);
  ~LightEditPanel() override = default;

  bool draw() override;

 protected:
  PBRScene& scene_;
  Properties props_to_show_;
  int selected_light_;

  int drawSelectTable();
  bool drawEditPanel(int light_index);
};

}  // namespace gfx

ENABLE_BITMASK_OPS(::gfx::LightEditPanel::Properties);
