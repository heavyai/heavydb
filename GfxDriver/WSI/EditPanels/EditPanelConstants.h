/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <imgui/imgui.h>

namespace gfx {

static constexpr ImGuiColorEditFlags kColorEditFlags =
    ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel;

static constexpr ImGuiTableFlags kSelectTableFlags =
    ImGuiTableFlags_BordersV | ImGuiTableFlags_BordersOuterH | ImGuiTableFlags_Resizable |
    ImGuiTableFlags_RowBg | ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_Hideable;

static constexpr ImGuiTableColumnFlags kSelectTableColumnFlags =
    ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthStretch |
    ImGuiTableColumnFlags_IndentDisable;

static constexpr float kTextBaseWidth = 10.f;

static constexpr float kSelectTableColorColumnWidth = kTextBaseWidth * 10.0f;

}  // namespace gfx
