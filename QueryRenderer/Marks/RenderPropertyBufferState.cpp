/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/RenderPropertyBufferState.h"

namespace QueryRenderer {

//
// PropBufferState
//
BaseRenderPropertyConstSet RenderPropertyBufferState::getAllUsedProps() const {
  BaseRenderPropertyConstSet rtn;
  rtn.insert(vbo_props.begin(), vbo_props.end());
  rtn.insert(ssbo_props.begin(), ssbo_props.end());
  rtn.insert(uniform_props.begin(), uniform_props.end());

  // NOTE: not adding any of the decimal props because all those properties will be found
  // in the vbo, ssbo, or ubo

  return rtn;
}

void RenderPropertyBufferState::clear() {
  vbo_props.clear();
  ssbo_props.clear();
  uniform_props.clear();
  decimal_props.clear();
}

}  // namespace QueryRenderer
