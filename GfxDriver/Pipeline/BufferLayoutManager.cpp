/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Pipeline/BufferLayoutManager.h"

#ifndef NDEBUG
#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#endif  // NDEBUG

#include "GfxDriver/Resources/BufferWrapper.h"

namespace gfx {

uint64_t BufferLayoutManager::getNumUsedBytes(const BufferLayoutShPtr& layout) const {
  if (!layout) {
    uint64_t used_bytes = 0;
    // TODO(croot): cache this and use a dirty flag?
    for (auto& item : layout_map_) {
      used_bytes += item.used_bytes;
    }
    return used_bytes;
  }

  return getBufferLayoutDataToUse(layout, "Cannot get number of used bytes. ").used_bytes;
}

uint32_t BufferLayoutManager::numItems(const BufferLayoutShPtr& layout) const {
  auto& layout_data =
      getBufferLayoutDataToUse(layout, "Cannot calculate number of items. ");
  uint64_t num = layout_data.used_bytes / layout_data.layout->getNumBytesPerItem();
  RUNTIME_EX_ASSERT(num <= std::numeric_limits<uint32_t>::max(),
                    "Too many items in buffer layout.");
  return static_cast<uint32_t>(num);
}

uint64_t BufferLayoutManager::getOffsetBytes(const BufferLayoutShPtr& layout) const {
  return getBufferLayoutDataToUse(layout, "Cannot get byte offset. ").offset_bytes;
}

bool BufferLayoutManager::hasAttribute(const std::string& attr_name,
                                       const BufferLayoutShPtr& layout) const {
  if (!layout_map_.size()) {
    return false;
  }

  return getBufferLayoutToUse(
             layout, "Cannot check the existence of attribute " + attr_name + ". ")
      ->hasAttribute(attr_name);
}

TypeGLSLShPtr BufferLayoutManager::getAttributeTypeGLSL(
    const std::string& attr_name,
    const BufferLayoutShPtr& layout) const {
  return getBufferLayoutToUse(layout,
                              "Cannot get GLSL attribute type for " + attr_name + ". ")
      ->getAttributeTypeGLSL(attr_name);
}

BufferAttrType BufferLayoutManager::getAttributeType(
    const std::string& attr_name,
    const BufferLayoutShPtr& layout) const {
  return getBufferLayoutToUse(layout, "Cannot get attribute type for " + attr_name + ". ")
      ->getAttributeType(attr_name);
}

bool BufferLayoutManager::hasBufferLayout(const BufferLayoutShPtr& layout) const {
  if (!layout) {
    return false;
  }

  const auto& ptr_lookup = layout_map_.get<BufferLayoutTag>();
  return ptr_lookup.find(layout) != ptr_lookup.end();
}

void BufferLayoutManager::replaceBufferLayoutAtOffset(const BufferLayoutShPtr& new_layout,
                                                      const uint64_t num_bytes,
                                                      const uint64_t offset_bytes) {
  RUNTIME_EX_ASSERT(new_layout, "Cannot replace an existing layout with a null");

  validateBufferLayout(num_bytes, offset_bytes, new_layout, true);

  updateBufferLayouts(new_layout, num_bytes, offset_bytes, true);
}

void BufferLayoutManager::deleteAllBufferLayouts() {
  layout_map_.clear();
  layout_mgr_buffer_.markDirty();
}

const BufferLayoutShPtr BufferLayoutManager::getBufferLayoutAtIndex(
    const uint32_t i) const {
  auto data = getBufferLayoutDataAtIndexToUse(i);
  return data.layout;
}

const BufferLayoutShPtr BufferLayoutManager::getBufferLayoutAtOffset(
    const uint64_t offset_bytes) const {
  auto itr = layout_map_.find(offset_bytes);
  return (itr != layout_map_.end() ? itr->layout : nullptr);
}

BufferLayoutManager::LayoutData BufferLayoutManager::getBufferLayoutData(
    const BufferLayoutShPtr& layout) const {
  auto data = getBufferLayoutDataToUse(layout, "Cannot get buffer layout data. ");
  return std::make_pair(data.used_bytes, data.offset_bytes);
}

const BaseBufferLayout* BufferLayoutManager::getBufferLayoutToUse(
    const BufferLayoutShPtr& layout,
    const std::string& err_prefix) const {
  RUNTIME_EX_ASSERT(layout != nullptr || layout_map_.size() == 1,
                    err_prefix +
                        "Buffer layout validation failed. There isn't a layout supplied "
                        "as an argument and the vbo has " +
                        std::to_string(layout_map_.size()) +
                        " attached layouts. Cannot determine which layout to use.");

  const auto& ptr_lookup = layout_map_.get<BufferLayoutTag>();
  RUNTIME_EX_ASSERT(!layout || ptr_lookup.find(layout) != ptr_lookup.end(),
                    err_prefix +
                        "The layout supplied as an argument is not a currently attached "
                        "layout to the vbo.");

  return (layout ? layout.get() : (*layout_map_.begin()).layout.get());
}

const BufferLayoutManager::BoundLayoutData& BufferLayoutManager::getBufferLayoutDataToUse(
    const BufferLayoutShPtr& layout,
    const std::string& err_prefix) const {
  RUNTIME_EX_ASSERT(layout != nullptr || layout_map_.size() == 1,
                    err_prefix +
                        "Buffer layout validation failed. There isn't a layout supplied "
                        "as an argument and the buffer has " +
                        std::to_string(layout_map_.size()) +
                        " attached layouts. Cannot determine which layout to use.");

  const auto& ptr_lookup = layout_map_.get<BufferLayoutTag>();

  auto itr = (layout ? ptr_lookup.find(layout) : ptr_lookup.begin());

  RUNTIME_EX_ASSERT(itr != ptr_lookup.end(),
                    err_prefix +
                        "The layout supplied as an argument is not a currently attached "
                        "layout to the buffer.");

  return *itr;
}

const BufferLayoutManager::BoundLayoutData&
BufferLayoutManager::getBufferLayoutDataAtIndexToUse(const uint32_t idx) const {
  // TODO(croot): perhaps find a way to add a random access index to the
  // boost::multi_index where the indices align with the ordered index
  //
  // Or use iterators and std::set methods such as begin() and end().
  RUNTIME_EX_ASSERT(idx < layout_map_.size(),
                    "Index " + std::to_string(idx) + " out of range. There are only " +
                        std::to_string(layout_map_.size()) + " layouts available.");

  auto itr = layout_map_.begin();
  uint32_t cnt = 0;
  while (++cnt <= idx) {
    itr++;
  }

  CHECK(itr != layout_map_.end());

  return *itr;
}

void BufferLayoutManager::validateBufferLayout(uint64_t num_bytes,
                                               uint64_t offset_bytes,
                                               const BufferLayoutShPtr& layout,
                                               bool replace_existing_layout,
                                               const std::string& err_prefix) {
  RUNTIME_EX_ASSERT(
      offset_bytes + num_bytes <= layout_mgr_buffer_.getNumBytes(),
      "Cannot update buffer with " + std::to_string(num_bytes) +
          " bytes starting at offset_bytes: " + std::to_string(offset_bytes) +
          " as it would overrun the length (" +
          std::to_string(layout_mgr_buffer_.getNumBytes()) + " bytes) of the buffer.");

  if (layout && layout_map_.size()) {
    BufferLayoutShPtr repl_layout;
    if (replace_existing_layout) {
      auto offset_itr = layout_map_.find(offset_bytes);
      if (offset_itr != layout_map_.end()) {
        repl_layout = offset_itr->layout;
      }
    }
    auto bound_itrs = layout_map_.range(offset_bytes <= boost::lambda::_1,
                                        boost::lambda::_1 < offset_bytes + num_bytes);

    auto start_itr = bound_itrs.first;

    RUNTIME_EX_ASSERT(
        start_itr == bound_itrs.second ||
            ((start_itr->layout == layout ||
              (replace_existing_layout && start_itr->layout == repl_layout)) &&
             ++start_itr == bound_itrs.second),
        "Cannot successfully add layout with byte offset: " +
            std::to_string(offset_bytes) +
            " and num bytes: " + std::to_string(num_bytes) +
            " as it overlaps with another layout at byte offset: " +
            std::to_string(bound_itrs.first->layout == layout
                               ? start_itr->offset_bytes
                               : bound_itrs.first->offset_bytes));

    if (bound_itrs.first == layout_map_.begin()) {
      return;
    }

    bound_itrs.first--;
    RUNTIME_EX_ASSERT(
        bound_itrs.first == layout_map_.end() || bound_itrs.first->layout == layout ||
            (replace_existing_layout && bound_itrs.first->layout == repl_layout) ||
            bound_itrs.first->offset_bytes + bound_itrs.first->used_bytes <= offset_bytes,
        "Cannot successfully add layout with byte offset: " +
            std::to_string(offset_bytes) +
            " as it overlaps with another layout at byte offset: " +
            std::to_string(bound_itrs.first->offset_bytes) +
            " and num bytes: " + std::to_string(bound_itrs.first->used_bytes));
  }
}

void BufferLayoutManager::updateBufferLayouts(const BufferLayoutShPtr& layout,
                                              const uint64_t num_bytes,
                                              const uint64_t offset_bytes,
                                              bool replace_existing_layout) {
  BufferLayoutMapByPtr& ptr_lookup = layout_map_.get<BufferLayoutTag>();

  // TODO(croot): might need to come up with a better error
  // resolution scheme here -- i.e. should we restore
  // the state of the layouts and memory on an error?
  // Or mark the vbo as undefined? Need to come up
  // with an appropriate scheme

  auto itr = ptr_lookup.find(layout);
  BufferLayoutMapByOffset::iterator offset_itr;
  bool success = false;
  BoundLayoutData old_data;
  if (itr != ptr_lookup.end()) {
    // first project into a offset_bytes iterator
    offset_itr = layout_map_.project<BufferLayoutOffsetTag>(itr);
    old_data = *offset_itr;
    success = layout_map_.modify(offset_itr,
                                 [num_bytes, offset_bytes](BoundLayoutData& layout_data) {
                                   layout_data.used_bytes = num_bytes;
                                   layout_data.offset_bytes = offset_bytes;
                                 });

    RUNTIME_EX_ASSERT(success,
                      "Cannot successfully add layout to the buffer object. Another "
                      "layout is already bound to the byte offset: " +
                          std::to_string(offset_bytes) + ".");

    offset_itr = layout_map_.find(offset_bytes);
    layout_mgr_buffer_.markDirty();
  } else {
    BufferLayoutShPtr repl_layout;
    if (replace_existing_layout) {
      auto replaceItr = layout_map_.find(offset_bytes);
      if (replaceItr != layout_map_.end()) {
        repl_layout = replaceItr->layout;
        old_data = *replaceItr;
        success = layout_map_.replace(replaceItr,
                                      BoundLayoutData(layout, num_bytes, offset_bytes));
        offset_itr = replaceItr;
      }
    }

    if (!repl_layout) {
      auto insert_pair = layout_map_.emplace(layout, num_bytes, offset_bytes);
      offset_itr = insert_pair.first;
      success = insert_pair.second;
    }

    RUNTIME_EX_ASSERT(success,
                      "Cannot successfully add layout to vbo. Another layout is already "
                      "bound to the byte offset: " +
                          std::to_string(offset_bytes) + ".");

    if (repl_layout) {
      layout_mgr_buffer_.markDirty();
    }
  }

  // now make sure the offset_bytes + useBytes doesn't overlap with a neighboring
  // layout

  // TODO(croot): this might be overkill as the validateBufferLayout checks for
  // overlap
  auto curr_itr = offset_itr;
  if (++offset_itr != layout_map_.end()) {
    if (offset_bytes + num_bytes > offset_itr->offset_bytes) {
      layout_map_.erase(curr_itr);
      THROW_RUNTIME_EX("Cannot successfully add layout with byte offset: " +
                       std::to_string(offset_bytes) +
                       " and num bytes: " + std::to_string(num_bytes) +
                       " as it overlaps with another layout at byte offset: " +
                       std::to_string(offset_itr->offset_bytes));
    }
  }

  offset_itr = curr_itr;
  if (offset_itr != layout_map_.begin()) {
    offset_itr--;
    if (offset_itr->offset_bytes + offset_itr->used_bytes > offset_bytes) {
      layout_map_.erase(curr_itr);
      THROW_RUNTIME_EX("Cannot successfully add layout with byte offset: " +
                       std::to_string(offset_bytes) +
                       " as it overlaps with another layout at byte offset: " +
                       std::to_string(offset_itr->offset_bytes) +
                       " and num bytes: " + std::to_string(offset_itr->used_bytes));
    }
  }
}

void BufferLayoutManager::deleteBufferLayoutsExcept(const BufferLayoutShPtr& layout) {
  if (!layout || !hasBufferLayout(layout)) {
    deleteAllBufferLayouts();
  } else if (layout_map_.size() > 1) {
    // TODO(croot): this is hacky -- may need to revisit
    // Attempting to clear all other layouts except 1.
    // Easiest/fastest way is to clear out all the
    // other layouts and re-add the one in question,
    // but that could have repercusions with the callbacks
    BoundLayoutData ptr_to_reinsert;
    for (auto& layout_data : layout_map_) {
      if (layout_data.layout != layout) {
        // layout_data->clearBuffer();
        layout_mgr_buffer_.markDirty();
      } else {
        ptr_to_reinsert = layout_data;
      }
    }
    layout_map_.clear();
    layout_map_.insert(ptr_to_reinsert);
  }
}

}  // namespace gfx
