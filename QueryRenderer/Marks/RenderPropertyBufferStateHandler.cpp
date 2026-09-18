/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/RenderPropertyBufferStateHandler.h"

#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/BaseDataTable.h"

namespace QueryRenderer {

//
// PropBufferStateHandler
//
PropBufferStateHandler::PropBufferStateHandler(
    const GeomType in_type,
    RenderPropertyBufferState& in_prop_buf_state,
    const QueryDataLayout* in_vbo_layout,
    const QueryDataLayout* in_ssbo_layout)
    : type_{in_type}
    , is_empty_{false}
    , is_dirty_{false}
    , vbo_count_{0}
    , ssbo_count_{0}
    , prop_buf_state_{in_prop_buf_state}
    , vbo_layout_{in_vbo_layout}
    , ssbo_layout_{in_ssbo_layout}
    , unvisited_props_{in_prop_buf_state.getAllUsedProps()}
    , finished_{false} {}

/**
 * State update handler for used mark properties.
 * This method handles setting the buffer state for the property. A property can only
 * exist in a vbo, ssbo, or ubo. If a property ever switches between buffer state, the
 * shader is marked as dirty. This method also handles validation of property data, for
 * instance, if a property is completely empty, then all properties must be empty.
 */
void PropBufferStateHandler::handleProp(const BaseRenderProperty* prop) {
  RENDER_LOG_SCOPE() << prop->getName() << "  hasVboPtr=" << prop->hasVboPtr();
  const QueryDataLayout* layout_to_use = nullptr;
  if (prop->hasVboPtr()) {
    // Polys can have vbos even when they're technically "empty" since their vbo
    // data can be cached.
    // TODO(croot): are there other mark types that need this
    // specification? Should we handle this logic in derived classes?
    CHECK(!is_empty_ || type_ == GeomType::kPolys)
        << " (isEmpty? " << is_empty_ << ") - property: " << prop->getName();

    ++vbo_count_;

    if (prop_buf_state_.vbo_props.insert(prop).second) {
      // This prop was newly added to the vbo, so mark ourselves as dirty
      // and remove the prop from the ssbo or ubo.
      is_dirty_ = true;
      if (!prop_buf_state_.ssbo_props.erase(prop)) {
        prop_buf_state_.uniform_props.erase(prop);
      }
    }
    layout_to_use = vbo_layout_;
  } else if (prop->hasSsboPtr()) {
    CHECK(!is_empty_) << " (isEmpty? " << is_empty_
                      << ") - property: " << prop->getName();

    ++ssbo_count_;

    if (prop_buf_state_.ssbo_props.insert(prop).second) {
      // This prop was newly added to the ssbo, so mark ourselves as dirty
      // and remove the prop from the vbo or ubo.
      is_dirty_ = true;
      if (!prop_buf_state_.vbo_props.erase(prop)) {
        prop_buf_state_.uniform_props.erase(prop);
      }
    }
    layout_to_use = ssbo_layout_;
  } else {
    if (prop->isUndefined()) {
      // If we hit this state, then all properties should be marked as undefined.
      // Validating that here.

      // NOTE: Polys can have vbos even when they're technically "empty"
      // since that data can be cached, so we need to handle that special case.
      // TODO(croot): are there other mark types that need this
      // specification? Should we handle this logic in derived classes?
      CHECK(is_empty_ || (!ssbo_count_ && (!vbo_count_ || type_ == GeomType::kPolys)))
          << "(isEmpty? " << is_empty_ << ") - property: \"" << prop->getName()
          << "\", vbo prop size: " << vbo_count_ << ", ssbo prop size: " << ssbo_count_;
      is_empty_ = true;
    }

    if (prop_buf_state_.uniform_props.insert(prop).second) {
      // This prop was newly added to the ubo, so mark ourselves as dirty
      // and remove the prop from the vbo or ssbo.
      is_dirty_ = true;
      if (!prop_buf_state_.vbo_props.erase(prop)) {
        prop_buf_state_.ssbo_props.erase(prop);
      }
    }
  }

  // Handle the 'decimal' state of the driving data ptr. This is required as special
  // uniform attributes are used for decimals to handle the decimal->floating pt
  // conversion in the shaders
  if (layout_to_use && layout_to_use->isDecimalAttr(prop->getDataColumnName())) {
    if (prop_buf_state_.decimal_props.insert(prop).second) {
      is_dirty_ = true;
    }
  } else if (prop_buf_state_.decimal_props.erase(prop)) {
    is_dirty_ = true;
  }

  postPropHandle(prop);
}

/**
 * Special handler for the key property. The key property is a special-case property
 * that is an externally hidden prop so requires special handling. The key property is
 * also only found in the vbo of in-situ point/symbol renders.
 */
void PropBufferStateHandler::handleKey(BaseRenderProperty& key, BaseDataTableShPtr data) {
  bool key_added = false;

  // NOTE: key is only found in vbos and would be present only from in-situ point
  // queries. Insitu lines/polys filter out empty keys in ExecuteRenderInterface when
  // building out the render buffers.
  if (prop_buf_state_.vbo_props.size()) {
    auto key_name = key.getName();
    auto& vbo_props = prop_buf_state_.vbo_props;

    // only look at the first vbo prop that has been visited. Unvisited props are to be
    // removed from the vbo list at a later step.
    auto itr = std::find_if(vbo_props.begin(), vbo_props.end(), [&](auto const prop) {
      return unvisited_props_.find(prop) == unvisited_props_.end();
    });

    if (itr != vbo_props.end()) {
      auto vbo = (*itr)->getVboPtr();
      if (vbo && data && data->getInputFormat() == DataInputFormat::kSQL &&
          vbo->hasAttribute(key_name, *vbo_layout_)) {
        key.initializeFromData(key_name, data);
        if (vbo_props.insert(&key).second) {
          is_dirty_ = true;
        }
        key_added = true;
      }
    }
  }

  if (!key_added) {
    key.initializeValue(0);
    if (prop_buf_state_.vbo_props.erase(&key)) {
      is_dirty_ = true;
    }
  }

  postPropHandle(&key);
}

/**
 * This method is called after all props have been processed. At this point, if there
 * were any props from the original state that were not processed, then those props are
 * removed from the state, and we mark ourselves as dirty.
 */
void PropBufferStateHandler::propHandlingFinished() {
  if (unvisited_props_.size()) {
    for (auto const& prop : unvisited_props_) {
      if (!prop_buf_state_.vbo_props.erase(prop)) {
        if (!prop_buf_state_.ssbo_props.erase(prop)) {
          prop_buf_state_.uniform_props.erase(prop);
        }
      }
      prop_buf_state_.decimal_props.erase(prop);
    }
    unvisited_props_.clear();
    is_dirty_ = true;
  }

  finished_ = true;
}

bool PropBufferStateHandler::isEmpty() {
  CHECK(finished_);
  return is_empty_;
}

bool PropBufferStateHandler::isDirty() {
  CHECK(finished_);
  return is_dirty_;
}

}  // namespace QueryRenderer
