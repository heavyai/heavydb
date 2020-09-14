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
 * @file    ColSlotContext.h
 * @author  Alex Baden <alex.baden@omnisci.com>
 * @brief   Provides column info and slot info for the output buffer and some metadata
 * helpers
 *
 */

#pragma once

#include "Logger/Logger.h"

#include <algorithm>
#include <string>
#include <vector>

struct SlotSize {
  int8_t padded_size;   // size of the slot
  int8_t logical_size;  // size of the element in the slot
};

inline bool operator==(const SlotSize& lhs, const SlotSize& rhs) {
  return lhs.padded_size == rhs.padded_size && lhs.logical_size == rhs.logical_size;
}

namespace Analyzer {
class Expr;
}

class ColSlotContext {
 public:
  ColSlotContext() {}

  ColSlotContext(const std::vector<Analyzer::Expr*>& col_expr_list,
                 const std::vector<int64_t>& col_exprs_to_not_project);

  void setAllSlotsSize(const int8_t slot_width_size);

  void setAllSlotsPaddedSize(const int8_t padded_size);

  void setAllUnsetSlotsPaddedSize(const int8_t padded_size);

  void setAllSlotsPaddedSizeToLogicalSize();

  void validate() const;

  size_t getColCount() const;
  size_t getSlotCount() const;

  const SlotSize& getSlotInfo(const size_t slot_idx) const {
    CHECK_LT(slot_idx, slot_sizes_.size());
    return slot_sizes_[slot_idx];
  }

  const std::vector<size_t>& getSlotsForCol(const size_t col_idx) const {
    CHECK_LT(col_idx, col_to_slot_map_.size());
    return col_to_slot_map_[col_idx];
  }

  size_t getAllSlotsPaddedSize() const;

  size_t getAllSlotsAlignedPaddedSize() const;

  size_t getAlignedPaddedSizeForRange(const size_t end) const;

  size_t getTotalBytesOfColumnarBuffers(const size_t entry_count) const;

  int8_t getMinPaddedByteSize(const int8_t actual_min_byte_width) const;

  size_t getCompactByteWidth() const;

  size_t getColOnlyOffInBytes(const size_t slot_idx) const;

  bool empty();

  void clear();

  void addColumn(const std::vector<std::tuple<int8_t, int8_t>>& slots_for_col);

  bool operator==(const ColSlotContext& that) const {
    return std::equal(
               slot_sizes_.cbegin(), slot_sizes_.cend(), that.slot_sizes_.cbegin()) &&
           std::equal(col_to_slot_map_.cbegin(),
                      col_to_slot_map_.cend(),
                      that.col_to_slot_map_.cbegin());
  }

  bool operator!=(const ColSlotContext& that) const { return !(*this == that); }

  void alignPaddedSlots(const bool sort_on_gpu);

  std::string toString() const {
    std::string str{"Col Slot Context State\n"};
    if (slot_sizes_.empty()) {
      str += "\tEmpty";
      return str;
    }
    str += "\tN | P , L\n";
    for (size_t i = 0; i < slot_sizes_.size(); i++) {
      const auto& slot_size = slot_sizes_[i];
      str += "\t" + std::to_string(i) + " | " + std::to_string(slot_size.padded_size) +
             " , " + std::to_string(slot_size.logical_size) + "\n";
    }
    return str;
  }

 private:
  void addSlotForColumn(const int8_t logical_size, const size_t column_idx);

  void addSlotForColumn(const int8_t padded_size,
                        const int8_t logical_size,
                        const size_t column_idx);

  std::vector<SlotSize> slot_sizes_;
  std::vector<std::vector<size_t>> col_to_slot_map_;
};
