/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseRenderProperty.h"

namespace QueryRenderer {

struct RenderPropertyBufferState {
  BaseRenderPropertyConstSet vbo_props;
  BaseRenderPropertyConstSet ssbo_props;
  BaseRenderPropertyConstSet uniform_props;
  BaseRenderPropertyConstSet decimal_props;

  BaseRenderPropertyConstSet getAllUsedProps() const;
  void clear();
};

}  // namespace QueryRenderer
