/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/BaseRenderProperty.h"
#include "QueryRenderer/Marks/RenderPropertyBufferState.h"

namespace QueryRenderer {

/**
 * PropBufferStateHandler manages the updating of an active mark property's buffer state,
 * which can be in either a vbo, ssbo, or ubo. It also handles managing props driven by
 * decimal types. Decimals require special handling as they require additional uniform
 * attributes to handle the conversion from decimal to floating-pt in the shaders. When a
 * prop is newly added or removed from a buffer state, the overall state is marked as
 * dirty. The dirty flag can be used to handle state management further up-stream (in this
 * case marking a shader as dirty)
 */
class PropBufferStateHandler {
 public:
  PropBufferStateHandler(const GeomType in_type,
                         RenderPropertyBufferState& in_prop_buf_state,
                         const QueryDataLayout* in_vbo_layout,
                         const QueryDataLayout* in_ssbo_layout);

  // not moveable/copy-constructable.
  PropBufferStateHandler() = delete;
  PropBufferStateHandler(const PropBufferStateHandler&) = delete;

  /**
   * State update handler for used mark properties.
   * This method handles setting the buffer state for the property. A property can only
   * exist in a vbo, ssbo, or ubo. If a property ever switches between buffer state, the
   * shader is marked as dirty. This method also handles validation of property data, for
   * instance, if a property is completely empty, then all properties must be empty.
   */
  void handleProp(const BaseRenderProperty* prop);

  /**
   * Special handler for the key property. The key property is a special-case property
   * that is an externally hidden prop so requires special handling. The key property is
   * also only found in the vbo of in-situ point/symbol renders.
   */
  void handleKey(BaseRenderProperty& key, BaseDataTableShPtr data);

  /**
   * This method is called after all props have been processed. At this point, if there
   * were any props from the original state that were not processed, then those props are
   * removed from the state, and we mark ourselves as dirty.
   */
  void propHandlingFinished();

  bool isEmpty();
  bool isDirty();

 private:
  // Parent mark type, used for type handling
  GeomType type_;

  // True when all props do not have data (i.e. query was empty)
  bool is_empty_;

  // True when at least 1 prop switches state
  bool is_dirty_;

  // Counters used for validating 'empty' state
  uint16_t vbo_count_;
  uint16_t ssbo_count_;

  RenderPropertyBufferState& prop_buf_state_;

  // Parent mark's buffer layouts.
  const QueryDataLayout* vbo_layout_;
  const QueryDataLayout* ssbo_layout_;

  // Captures unhandled props.
  BaseRenderPropertyConstSet unvisited_props_;

  // True when all props are handled.
  bool finished_;

  inline void postPropHandle(const BaseRenderProperty* prop) {
    // set prop as visited.
    unvisited_props_.erase(prop);
  }
};

}  // namespace QueryRenderer
