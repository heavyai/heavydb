/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/lambda/lambda.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>

#include "GfxDriver/Resources/BufferLayout.h"

namespace gfx {

class BufferLayoutManager {
 public:
  BufferLayoutManager(BufferWrapper& buffer) : layout_mgr_buffer_(buffer) {}
  BufferLayoutManager() = delete;
  ~BufferLayoutManager() = default;

  uint64_t getNumUsedBytes(const BufferLayoutShPtr& layout = nullptr) const;
  uint32_t numItems(const BufferLayoutShPtr& layout = nullptr) const;
  uint64_t getOffsetBytes(const BufferLayoutShPtr& layout = nullptr) const;

  bool hasAttribute(const std::string& attr_name,
                    const BufferLayoutShPtr& layout = nullptr) const;
  TypeGLSLShPtr getAttributeTypeGLSL(const std::string& attr_name,
                                     const BufferLayoutShPtr& layout = nullptr) const;
  BufferAttrType getAttributeType(const std::string& attr_name,
                                  const BufferLayoutShPtr& layout = nullptr) const;

  uint32_t getNumBufferLayouts() const { return layout_map_.size(); }
  bool hasBufferLayout(const BufferLayoutShPtr& layout) const;
  void replaceBufferLayoutAtOffset(const BufferLayoutShPtr& new_layout,
                                   const uint64_t num_bytes,
                                   const uint64_t offset_bytes);
  void deleteAllBufferLayouts();

  const BufferLayoutShPtr getBufferLayoutAtIndex(const uint32_t idx) const;
  const BufferLayoutShPtr getBufferLayoutAtOffset(const uint64_t offset_bytes) const;

  using LayoutData = std::pair<uint64_t /*used_bytes*/, uint64_t /*offset_bytes*/>;
  LayoutData getBufferLayoutData(const BufferLayoutShPtr& layout) const;

  struct BoundLayoutData {
    BufferLayoutShPtr layout;
    uint64_t used_bytes;
    uint64_t offset_bytes;

    BoundLayoutData() : layout(nullptr), used_bytes(0), offset_bytes(0) {}
    BoundLayoutData(const BufferLayoutShPtr& layout,
                    const uint64_t used_bytes,
                    const uint64_t offset_bytes)
        : layout(layout), used_bytes(used_bytes), offset_bytes(offset_bytes) {}
  };

  const BaseBufferLayout* getBufferLayoutToUse(const BufferLayoutShPtr& layout,
                                               const std::string& err_prefix = "") const;

  const BoundLayoutData& getBufferLayoutDataToUse(
      const BufferLayoutShPtr& layout,
      const std::string& err_prefix = "") const;

  const BoundLayoutData& getBufferLayoutDataAtIndexToUse(const uint32_t idx) const;

  void validateBufferLayout(uint64_t num_bytes,
                            uint64_t offset_bytes,
                            const BufferLayoutShPtr& layout,
                            bool replace_existing_layout = false,
                            const std::string& err_prefix = "");
  void updateBufferLayouts(const BufferLayoutShPtr& layout,
                           const uint64_t num_bytes,
                           const uint64_t offset_bytes,
                           bool replace_existing_layout = false);
  void deleteBufferLayoutsExcept(const BufferLayoutShPtr& layout);

 private:
  struct BufferLayoutOffsetTag {};
  struct BufferLayoutTag {};
  using BufferLayoutMap = boost::multi_index_container<
      BoundLayoutData,
      boost::multi_index::indexed_by<
          boost::multi_index::ordered_unique<
              boost::multi_index::tag<BufferLayoutOffsetTag>,
              boost::multi_index::
                  member<BoundLayoutData, size_t, &BoundLayoutData::offset_bytes>>,
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<BufferLayoutTag>,
              boost::multi_index::
                  member<BoundLayoutData, BufferLayoutShPtr, &BoundLayoutData::layout>>>>;

  using BufferLayoutMapByOffset = BufferLayoutMap::index<BufferLayoutOffsetTag>::type;
  using BufferLayoutMapByPtr = BufferLayoutMap::index<BufferLayoutTag>::type;

  BufferWrapper& layout_mgr_buffer_;
  BufferLayoutMap layout_map_;
};

}  // namespace gfx
