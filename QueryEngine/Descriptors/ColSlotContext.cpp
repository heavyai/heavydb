/*
 * Copyright 2019 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file    ColSlotContext.cpp
 * @author  Alex Baden <alex.baden@omnisci.com>
 * @brief   Provides column info and slot info for the output buffer and some metadata
 * helpers
 *
 */

#include "ColSlotContext.h"

#include "../BufferCompaction.h"

#include <Analyzer/Analyzer.h>
#include <Shared/SqlTypesLayout.h>

#include <numeric>

extern bool g_bigint_count;

ColSlotContext::ColSlotContext(const std::vector<Analyzer::Expr*>& col_expr_list,
                               const std::vector<ssize_t>& col_exprs_to_not_project) {
  // Note that non-projected col exprs could be projected cols that we can lazy fetch or
  // grouped cols with keyless hash
  if (!col_exprs_to_not_project.empty()) {
    CHECK_EQ(col_expr_list.size(), col_exprs_to_not_project.size());
  }
  size_t col_expr_idx = 0;
  col_to_slot_map_.resize(col_expr_list.size());
  for (const auto col_expr : col_expr_list) {
    if (!col_exprs_to_not_project.empty() &&
        col_exprs_to_not_project[col_expr_idx] != -1) {
      addSlotForColumn(0, 0, col_expr_idx);
      ++col_expr_idx;
      continue;
    }
    if (!col_expr) {
      // row index
      addSlotForColumn(sizeof(int64_t), col_expr_idx);
    } else {
      const auto agg_info = get_target_info(col_expr, g_bigint_count);
      const auto chosen_type = get_compact_type(agg_info);
      if ((chosen_type.is_string() && chosen_type.get_compression() == kENCODING_NONE) ||
          chosen_type.is_array()) {
        addSlotForColumn(sizeof(int64_t), col_expr_idx);
        addSlotForColumn(sizeof(int64_t), col_expr_idx);
        ++col_expr_idx;
        continue;
      }
      if (chosen_type.is_geometry()) {
        for (auto i = 0; i < chosen_type.get_physical_coord_cols(); ++i) {
          addSlotForColumn(sizeof(int64_t), col_expr_idx);
          addSlotForColumn(sizeof(int64_t), col_expr_idx);
        }
        ++col_expr_idx;
        continue;
      }
      const auto col_expr_bitwidth = get_bit_width(chosen_type);
      CHECK_EQ(size_t(0), col_expr_bitwidth % 8);
      addSlotForColumn(static_cast<int8_t>(col_expr_bitwidth >> 3), col_expr_idx);
      // for average, we'll need to keep the count as well
      if (agg_info.agg_kind == kAVG) {
        CHECK(agg_info.is_agg);
        addSlotForColumn(sizeof(int64_t), col_expr_idx);
      }
    }
    ++col_expr_idx;
  }
}

void ColSlotContext::setAllSlotsSize(const int8_t slot_width_size) {
  const SlotSize ss{slot_width_size, slot_width_size};
  std::fill(slot_sizes_.begin(), slot_sizes_.end(), ss);
}

void ColSlotContext::setAllSlotsPaddedSize(const int8_t padded_size) {
  for (auto& slot_size : slot_sizes_) {
    slot_size.padded_size = padded_size;
  }
}

void ColSlotContext::setAllUnsetSlotsPaddedSize(const int8_t padded_size) {
  for (auto& slot_size : slot_sizes_) {
    if (slot_size.padded_size < 0) {
      slot_size.padded_size = padded_size;
    }
  }
}

void ColSlotContext::setAllSlotsPaddedSizeToLogicalSize() {
  for (auto& slot_size : slot_sizes_) {
    slot_size.padded_size = slot_size.logical_size;
  }
}

void ColSlotContext::validate() const {
  for (const auto& slot_size : slot_sizes_) {
    CHECK_GE(slot_size.logical_size, 0);
    CHECK_LE(slot_size.logical_size, slot_size.padded_size);
  }
}

size_t ColSlotContext::getColCount() const {
  return col_to_slot_map_.size();
}
size_t ColSlotContext::getSlotCount() const {
  return slot_sizes_.size();
}

size_t ColSlotContext::getAllSlotsPaddedSize() const {
  return std::accumulate(slot_sizes_.cbegin(),
                         slot_sizes_.cend(),
                         size_t(0),
                         [](size_t sum, const auto& slot_size) {
                           CHECK_GE(slot_size.padded_size, 0);
                           return sum + static_cast<size_t>(slot_size.padded_size);
                         });
}

size_t ColSlotContext::getAllSlotsAlignedPaddedSize() const {
  return getAlignedPaddedSizeForRange(slot_sizes_.size());
}

size_t ColSlotContext::getAlignedPaddedSizeForRange(const size_t end) const {
  return std::accumulate(slot_sizes_.cbegin(),
                         slot_sizes_.cbegin() + end,
                         size_t(0),
                         [](size_t sum, const auto& slot_size) {
                           CHECK_GE(slot_size.padded_size, 0);
                           const auto chosen_bytes =
                               static_cast<size_t>(slot_size.padded_size);
                           if (chosen_bytes == sizeof(int64_t)) {
                             return align_to_int64(sum) + chosen_bytes;
                           } else {
                             return sum + chosen_bytes;
                           }
                         });
}

size_t ColSlotContext::getTotalBytesOfColumnarBuffers(const size_t entry_count) const {
  const auto total_bytes = std::accumulate(
      slot_sizes_.cbegin(),
      slot_sizes_.cend(),
      size_t(0),
      [entry_count](size_t sum, const auto& slot_size) {
        CHECK_GE(slot_size.padded_size, 0);
        return sum +
               align_to_int64(static_cast<size_t>(slot_size.padded_size) * entry_count);
      });
  return align_to_int64(total_bytes);
}

int8_t ColSlotContext::getMinPaddedByteSize(const int8_t actual_min_byte_width) const {
  if (slot_sizes_.empty()) {
    return actual_min_byte_width;
  }
  const auto min_padded_size = std::min_element(
      slot_sizes_.cbegin(), slot_sizes_.cend(), [](const auto& lhs, const auto& rhs) {
        return lhs.padded_size < rhs.padded_size;
      });
  return std::min(min_padded_size->padded_size, actual_min_byte_width);
}

size_t ColSlotContext::getCompactByteWidth() const {
  if (slot_sizes_.empty()) {
    return 8;
  }
  size_t compact_width{0};
  for (const auto& slot_size : slot_sizes_) {
    if (slot_size.padded_size != 0) {
      compact_width = slot_size.padded_size;
      break;
    }
  }
  if (!compact_width) {
    return 0;
  }
  CHECK_GT(compact_width, size_t(0));
  for (const auto& slot_size : slot_sizes_) {
    if (slot_size.padded_size == 0) {
      continue;
    }
    CHECK_EQ(static_cast<size_t>(slot_size.padded_size), compact_width);
  }
  return compact_width;
}

size_t ColSlotContext::getColOnlyOffInBytes(const size_t slot_idx) const {
  CHECK_LT(slot_idx, slot_sizes_.size());
  auto offset_bytes = getAlignedPaddedSizeForRange(slot_idx);
  if (slot_sizes_[slot_idx].padded_size == sizeof(int64_t)) {
    offset_bytes = align_to_int64(offset_bytes);
  }
  return offset_bytes;
}

bool ColSlotContext::empty() {
  return slot_sizes_.empty();
}

void ColSlotContext::clear() {
  slot_sizes_.clear();
  col_to_slot_map_.clear();
}

void ColSlotContext::alignPaddedSlots(const bool sort_on_gpu) {
  size_t total_bytes{0};
  for (size_t slot_idx = 0; slot_idx < slot_sizes_.size(); slot_idx++) {
    auto chosen_bytes = slot_sizes_[slot_idx].padded_size;
    if (chosen_bytes == sizeof(int64_t)) {
      const auto aligned_total_bytes = align_to_int64(total_bytes);
      CHECK_GE(aligned_total_bytes, total_bytes);
      if (slot_idx >= 1) {
        const auto padding = aligned_total_bytes - total_bytes;
        CHECK(padding == 0 || padding == 4);
        slot_sizes_[slot_idx - 1].padded_size += padding;
      }
      total_bytes = aligned_total_bytes;
    }
    total_bytes += chosen_bytes;
  }
  if (!sort_on_gpu) {
    const auto aligned_total_bytes = align_to_int64(total_bytes);
    CHECK_GE(aligned_total_bytes, total_bytes);
    const auto padding = aligned_total_bytes - total_bytes;
    CHECK(padding == 0 || padding == 4);
    slot_sizes_.back().padded_size += padding;
  }
}

void ColSlotContext::addColumn(
    const std::vector<std::tuple<int8_t, int8_t>>& slots_for_col) {
  const auto col_idx = col_to_slot_map_.size();
  col_to_slot_map_.emplace_back();
  for (const auto& slot_info : slots_for_col) {
    addSlotForColumn(std::get<1>(slot_info), std::get<0>(slot_info), col_idx);
  }
}

void ColSlotContext::addSlotForColumn(const int8_t logical_size,
                                      const size_t column_idx) {
  addSlotForColumn(-1, logical_size, column_idx);
}

void ColSlotContext::addSlotForColumn(const int8_t padded_size,
                                      const int8_t logical_size,
                                      const size_t column_idx) {
  CHECK_LT(column_idx, col_to_slot_map_.size());
  col_to_slot_map_[column_idx].push_back(slot_sizes_.size());
  slot_sizes_.emplace_back(SlotSize{padded_size, logical_size});
}
