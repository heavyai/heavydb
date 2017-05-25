/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "ResultSetTestUtils.h"
#include "../QueryEngine/ResultSetBufferAccessors.h"

void fill_one_entry_baseline(int64_t* value_slots,
                             const int64_t v,
                             const std::vector<TargetInfo>& target_infos,
                             const bool empty /* = false */,
                             const bool null_val /* = false */) {
  size_t target_slot = 0;
  int64_t vv = 0;
  for (const auto& target_info : target_infos) {
    const bool float_argument_input = takes_float_argument(target_info);
    if (target_info.agg_kind == kCOUNT) {
      if (empty || null_val) {
        vv = 0;
      } else {
        vv = v;
      }
    } else {
      bool isNullable = !target_info.sql_type.get_notnull();
      if ((isNullable && target_info.skip_null_val && null_val) || empty) {
        vv = null_val_bit_pattern(target_info.sql_type, float_argument_input);
      } else {
        vv = v;
      }
    }

    switch (target_info.sql_type.get_type()) {
      case kSMALLINT:
      case kINT:
      case kBIGINT:
        value_slots[target_slot] = vv;
        break;
      case kFLOAT:
        if (float_argument_input) {
          float fi = vv;
          int64_t fi_bin = *reinterpret_cast<const int32_t*>(may_alias_ptr(&fi));
          value_slots[target_slot] = null_val ? vv : fi_bin;
          break;
        }
      case kDOUBLE: {
        double di = vv;
        value_slots[target_slot] = null_val ? vv : *reinterpret_cast<const int64_t*>(may_alias_ptr(&di));
        break;
      }
      case kTEXT:
        value_slots[target_slot] = -(vv + 2);
        break;
      default:
        CHECK(false);
    }
    if (target_info.agg_kind == kAVG) {
      value_slots[target_slot + 1] = 1;
    }
    target_slot = advance_slot(target_slot, target_info, false);
  }
}

size_t get_slot_count(const std::vector<TargetInfo>& target_infos) {
  size_t count = 0;
  for (const auto& target_info : target_infos) {
    count = advance_slot(count, target_info, false);
  }
  return count;
}

std::unordered_map<size_t, size_t> get_slot_to_target_mapping(const std::vector<TargetInfo>& target_infos) {
  std::unordered_map<size_t, size_t> mapping;
  size_t target_index = 0;
  size_t slot_index = 0;
  for (const auto& target_info : target_infos) {
    mapping.insert(std::make_pair(slot_index, target_index));
    auto old_slot_index = slot_index;
    slot_index = advance_slot(slot_index, target_info, false);
    if (slot_index == old_slot_index + 2) {
      mapping.insert(std::make_pair(old_slot_index + 1, target_index));
    }
    target_index++;
  }
  return mapping;
}
