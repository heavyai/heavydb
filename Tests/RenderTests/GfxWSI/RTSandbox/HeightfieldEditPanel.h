#pragma once

#include "GfxDriver/WSI/EditPanels/EditPanel.h"

#include "Tests/RenderTests/GfxWSI/RTSandbox/HeightfieldCallback.h"

class HeightfieldEditPanel : public gfx::EditPanel {
 public:
  explicit HeightfieldEditPanel(HeightfieldCallback& hf);
  ~HeightfieldEditPanel() override = default;

  bool draw() override;

 private:
  HeightfieldCallback& hf_;
};
