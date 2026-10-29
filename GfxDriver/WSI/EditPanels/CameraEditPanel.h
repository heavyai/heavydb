/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/WSI/EditPanels/EditPanel.h"

#include "GfxDriver/Render/Camera.h"
#include "Shared/EnumBitmaskOps.h"

namespace gfx {

class CameraEditPanel : public EditPanel {
 public:
  enum class Properties : uint32_t {
    kType = 1 << 0,
    kProjection = 1 << 1,
    kFOV = 1 << 2,
    kAll = 0xFFFFFFFF
  };

  explicit CameraEditPanel(Camera& camera, Properties props_to_show = Properties::kAll);
  ~CameraEditPanel() override = default;

  bool draw() override;

 private:
  Camera& camera_;
  Properties props_to_show_;
};

}  // namespace gfx

ENABLE_BITMASK_OPS(::gfx::CameraEditPanel::Properties);
