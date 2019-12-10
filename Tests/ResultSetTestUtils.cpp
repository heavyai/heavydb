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

// TODO: move this into the ResultSet
int8_t* advance_to_next_columnar_key_buff(int8_t* key_ptr,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          const size_t key_idx) {
  CHECK(!query_mem_desc.hasKeylessHash());
  CHECK_LT(key_idx, query_mem_desc.getGroupbyColCount());
  const auto column_offset =
      query_mem_desc.getEntryCount() * query_mem_desc.groupColWidth(key_idx);
  auto new_key_ptr = align_to_int64(key_ptr + column_offset);
  return new_key_ptr;
}

int64_t get_empty_key_sentinel(int8_t key_bytes) {
  switch (key_bytes) {
    case 8:
      return EMPTY_KEY_64;
    case 4:
      return EMPTY_KEY_32;
    case 2:
      return EMPTY_KEY_16;
    case 1:
      return EMPTY_KEY_8;
    default:
      break;
  }
  UNREACHABLE();
  return 0;
}

void write_key(const int64_t k, int8_t* ptr, const int8_t key_bytes) {
  switch (key_bytes) {
    case 8: {
      *reinterpret_cast<int64_t*>(ptr) = k;
      break;
    }
    case 4: {
      *reinterpret_cast<int32_t*>(ptr) = k;
      break;
    }
    case 2: {
      *reinterpret_cast<int16_t*>(ptr) = k;
      break;
    }
    case 1: {
      *reinterpret_cast<int8_t*>(ptr) = k;
      break;
    }
    default:
      CHECK(false);
  }
}

void write_int(int8_t* slot_ptr, const int64_t v, const size_t slot_bytes) {
  switch (slot_bytes) {
    case 4:
      *reinterpret_cast<int32_t*>(slot_ptr) = v;
      break;
    case 8:
      *reinterpret_cast<int64_t*>(slot_ptr) = v;
      break;
    default:
      CHECK(false);
  }
}

void write_fp(int8_t* slot_ptr, const int64_t v, const size_t slot_bytes) {
  switch (slot_bytes) {
    case 4: {
      float fi = v;
      *reinterpret_cast<int32_t*>(slot_ptr) =
          *reinterpret_cast<const int32_t*>(may_alias_ptr(&fi));
      break;
    }
    case 8: {
      double di = v;
      *reinterpret_cast<int64_t*>(slot_ptr) =
          *reinterpret_cast<const int64_t*>(may_alias_ptr(&di));
      break;
    }
    default:
      CHECK(false);
  }
}

int8_t* fill_one_entry_no_collisions(int8_t* buff,
                                     const QueryMemoryDescriptor& query_mem_desc,
                                     const int64_t v,
                                     const std::vector<TargetInfo>& target_infos,
                                     const bool empty,
                                     const bool null_val /* = false */) {
  size_t target_idx = 0;
  int8_t* slot_ptr = buff;
  int64_t vv = 0;
  for (const auto& target_info : target_infos) {
    const auto slot_bytes = query_mem_desc.getLogicalSlotWidthBytes(target_idx);
    CHECK_LE(target_info.sql_type.get_size(), slot_bytes);
    bool isNullable = !target_info.sql_type.get_notnull();
    if (target_info.agg_kind == kCOUNT) {
      if (empty || null_val) {
        vv = 0;
      } else {
        vv = v;
      }
    } else {
      if (isNullable && target_info.skip_null_val && null_val) {
        vv = inline_int_null_val(target_info.sql_type);
      } else {
        vv = v;
      }
    }
    if (empty) {
      write_int(slot_ptr, query_mem_desc.hasKeylessHash() ? 0 : vv, slot_bytes);
    } else {
      if (target_info.sql_type.is_integer()) {
        write_int(slot_ptr, vv, slot_bytes);
      } else if (target_info.sql_type.is_string()) {
        write_int(slot_ptr, -(vv + 2), slot_bytes);
      } else {
        CHECK(target_info.sql_type.is_fp());
        write_fp(slot_ptr, vv, slot_bytes);
      }
    }
    slot_ptr += slot_bytes;
    if (target_info.agg_kind == kAVG) {
      const auto count_slot_bytes =
          query_mem_desc.getLogicalSlotWidthBytes(target_idx + 1);
      if (empty) {
        write_int(slot_ptr, query_mem_desc.hasKeylessHash() ? 0 : 0, count_slot_bytes);
      } else {
        if (isNullable && target_info.skip_null_val && null_val) {
          write_int(slot_ptr, 0, count_slot_bytes);  // count of elements should be set to
                                                     // 0 for elements with null_val
        } else {
          write_int(slot_ptr, 1, count_slot_bytes);  // count of elements in the group is
                                                     // 1 - good enough for testing
        }
      }
      slot_ptr += count_slot_bytes;
    }
    target_idx = advance_slot(target_idx, target_info, false);
  }
  return slot_ptr;
}

void fill_one_entry_one_col(int8_t* ptr1,
                            const int8_t compact_sz1,
                            int8_t* ptr2,
                            const int8_t compact_sz2,
                            int64_t v,
                            const TargetInfo& target_info,
                            const bool empty_entry,
                            const bool null_val /* = false */) {
  int64_t vv = 0;
  if (target_info.agg_kind == kCOUNT) {
    if (empty_entry || null_val) {
      vv = 0;
    } else {
      vv = v;
    }
  } else {
    bool isNullable = !target_info.sql_type.get_notnull();
    if (isNullable && target_info.skip_null_val && null_val) {
      vv = inline_int_null_val(target_info.sql_type);
    } else {
      if (empty_entry && (target_info.agg_kind == kAVG)) {
        vv = 0;
      } else {
        vv = v;
      }
    }
  }
  CHECK(ptr1);
  switch (compact_sz1) {
    case 8:
      if (target_info.sql_type.is_fp()) {
        double di = vv;
        *reinterpret_cast<int64_t*>(ptr1) =
            *reinterpret_cast<const int64_t*>(may_alias_ptr(&di));
      } else {
        *reinterpret_cast<int64_t*>(ptr1) = vv;
      }
      break;
    case 4:
      if (target_info.sql_type.is_fp()) {
        float fi = vv;
        *reinterpret_cast<int32_t*>(ptr1) =
            *reinterpret_cast<const int32_t*>(may_alias_ptr(&fi));
      } else {
        *reinterpret_cast<int32_t*>(ptr1) = vv;
      }
      break;
    case 2:
      CHECK(!target_info.sql_type.is_fp());
      *reinterpret_cast<int16_t*>(ptr1) = vv;
      break;
    case 1:
      CHECK(!target_info.sql_type.is_fp());
      *reinterpret_cast<int8_t*>(ptr1) = vv;
      break;
    default:
      CHECK(false);
  }
  if (target_info.is_agg && target_info.agg_kind == kAVG) {
    CHECK(ptr2);
    switch (compact_sz2) {
      case 8:
        *reinterpret_cast<int64_t*>(ptr2) = (empty_entry ? *ptr1 : 1);
        break;
      case 4:
        *reinterpret_cast<int32_t*>(ptr2) = (empty_entry ? *ptr1 : 1);
        break;
      default:
        CHECK(false);
    }
  }
}

void fill_one_entry_one_col(int64_t* value_slot,
                            const int64_t v,
                            const TargetInfo& target_info,
                            const size_t entry_count,
                            const bool empty_entry /* = false */,
                            const bool null_val /* = false */) {
  auto ptr1 = reinterpret_cast<int8_t*>(value_slot);
  int8_t* ptr2{nullptr};
  if (target_info.agg_kind == kAVG) {
    ptr2 = reinterpret_cast<int8_t*>(&value_slot[entry_count]);
  }
  fill_one_entry_one_col(ptr1, 8, ptr2, 8, v, target_info, empty_entry, null_val);
}

void fill_storage_buffer_perfect_hash_colwise(int8_t* buff,
                                              const std::vector<TargetInfo>& target_infos,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              NumberGenerator& generator,
                                              const size_t step) {
  const auto key_component_count = query_mem_desc.getKeyCount();
  CHECK(query_mem_desc.didOutputColumnar());
  // initialize the key buffer(s)
  auto col_ptr = buff;
  for (size_t key_idx = 0; key_idx < key_component_count; ++key_idx) {
    auto key_entry_ptr = col_ptr;
    const auto key_bytes = query_mem_desc.groupColWidth(key_idx);

    for (size_t i = 0; i < query_mem_desc.getEntryCount(); ++i) {
      if (i % step == 0) {
        const auto v = generator.getNextValue();
        write_key(v, key_entry_ptr, key_bytes);
      } else {
        write_key(get_empty_key_sentinel(key_bytes), key_entry_ptr, key_bytes);
      }
      key_entry_ptr += key_bytes;
    }
    col_ptr = advance_to_next_columnar_key_buff(col_ptr, query_mem_desc, key_idx);
    generator.reset();
  }
  // initialize the value buffer(s)
  size_t slot_idx = 0;
  for (const auto& target_info : target_infos) {
    auto col_entry_ptr = col_ptr;
    const auto col_bytes = query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
    for (size_t i = 0; i < query_mem_desc.getEntryCount(); ++i) {
      int8_t* ptr2{nullptr};
      const bool read_secondary_buffer{target_info.is_agg &&
                                       target_info.agg_kind == kAVG};
      if (read_secondary_buffer) {
        ptr2 = col_entry_ptr + query_mem_desc.getEntryCount() * col_bytes;
      }
      if (i % step == 0) {
        const auto gen_val = generator.getNextValue();
        const auto val = target_info.sql_type.is_string() ? -(gen_val + 2) : gen_val;
        fill_one_entry_one_col(col_entry_ptr,
                               col_bytes,
                               ptr2,
                               read_secondary_buffer
                                   ? query_mem_desc.getPaddedSlotWidthBytes(slot_idx + 1)
                                   : -1,
                               val,
                               target_info,
                               false);
      } else {
        fill_one_entry_one_col(col_entry_ptr,
                               col_bytes,
                               ptr2,
                               read_secondary_buffer
                                   ? query_mem_desc.getPaddedSlotWidthBytes(slot_idx + 1)
                                   : -1,
                               query_mem_desc.hasKeylessHash() ? 0 : 0xdeadbeef,
                               target_info,
                               true);
      }
      col_entry_ptr += col_bytes;
    }
    col_ptr = advance_to_next_columnar_target_buff(col_ptr, query_mem_desc, slot_idx);
    if (target_info.is_agg && target_info.agg_kind == kAVG) {
      col_ptr =
          advance_to_next_columnar_target_buff(col_ptr, query_mem_desc, slot_idx + 1);
    }
    slot_idx = advance_slot(slot_idx, target_info, false);
    generator.reset();
  }
}

void fill_storage_buffer_perfect_hash_rowwise(int8_t* buff,
                                              const std::vector<TargetInfo>& target_infos,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              NumberGenerator& generator,
                                              const size_t step) {
  const auto key_component_count = query_mem_desc.getKeyCount();
  CHECK(!query_mem_desc.didOutputColumnar());
  auto key_buff = buff;
  for (size_t i = 0; i < query_mem_desc.getEntryCount(); ++i) {
    if (i % step == 0) {
      const auto v = generator.getNextValue();
      auto key_buff_i64 = reinterpret_cast<int64_t*>(key_buff);
      for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
        *key_buff_i64++ = v;
      }
      auto entries_buff = reinterpret_cast<int8_t*>(key_buff_i64);
      key_buff = fill_one_entry_no_collisions(
          entries_buff, query_mem_desc, v, target_infos, false);
    } else {
      auto key_buff_i64 = reinterpret_cast<int64_t*>(key_buff);
      for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
        *key_buff_i64++ = EMPTY_KEY_64;
      }
      auto entries_buff = reinterpret_cast<int8_t*>(key_buff_i64);
      key_buff = fill_one_entry_no_collisions(
          entries_buff, query_mem_desc, 0xdeadbeef, target_infos, true);
    }
  }
}

void fill_storage_buffer_baseline_colwise(int8_t* buff,
                                          const std::vector<TargetInfo>& target_infos,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          NumberGenerator& generator,
                                          const size_t step) {
  CHECK(query_mem_desc.didOutputColumnar());
  const auto key_component_count = query_mem_desc.getKeyCount();
  const auto i64_buff = reinterpret_cast<int64_t*>(buff);
  const auto target_slot_count = get_slot_count(target_infos);
  const auto slot_to_target = get_slot_to_target_mapping(target_infos);
  for (size_t i = 0; i < query_mem_desc.getEntryCount(); ++i) {
    for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
      i64_buff[key_offset_colwise(i, key_comp_idx, query_mem_desc.getEntryCount())] =
          EMPTY_KEY_64;
    }
    for (size_t target_slot = 0; target_slot < target_slot_count; ++target_slot) {
      auto target_it = slot_to_target.find(target_slot);
      CHECK(target_it != slot_to_target.end());
      const auto& target_info = target_infos[target_it->second];
      i64_buff[slot_offset_colwise(
          i, target_slot, key_component_count, query_mem_desc.getEntryCount())] =
          (target_info.agg_kind == kCOUNT ? 0 : 0xdeadbeef);
    }
  }
  for (size_t i = 0; i < query_mem_desc.getEntryCount(); i += step) {
    const auto gen_val = generator.getNextValue();
    std::vector<int64_t> key(key_component_count, gen_val);
    auto value_slots = get_group_value_columnar(
        i64_buff, query_mem_desc.getEntryCount(), &key[0], key.size());
    CHECK(value_slots);
    for (const auto& target_info : target_infos) {
      const auto val = target_info.sql_type.is_string() ? -(gen_val + step) : gen_val;
      fill_one_entry_one_col(
          value_slots, val, target_info, query_mem_desc.getEntryCount());
      value_slots += query_mem_desc.getEntryCount();
      if (target_info.agg_kind == kAVG) {
        value_slots += query_mem_desc.getEntryCount();
      }
    }
  }
}

void fill_storage_buffer_baseline_rowwise(int8_t* buff,
                                          const std::vector<TargetInfo>& target_infos,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          NumberGenerator& generator,
                                          const size_t step) {
  const auto key_component_count = query_mem_desc.getKeyCount();
  const auto i64_buff = reinterpret_cast<int64_t*>(buff);
  const auto target_slot_count = get_slot_count(target_infos);
  const auto slot_to_target = get_slot_to_target_mapping(target_infos);
  for (size_t i = 0; i < query_mem_desc.getEntryCount(); ++i) {
    const auto first_key_comp_offset =
        key_offset_rowwise(i, key_component_count, target_slot_count);
    for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
      i64_buff[first_key_comp_offset + key_comp_idx] = EMPTY_KEY_64;
    }
    for (size_t target_slot = 0; target_slot < target_slot_count; ++target_slot) {
      auto target_it = slot_to_target.find(target_slot);
      CHECK(target_it != slot_to_target.end());
      const auto& target_info = target_infos[target_it->second];
      i64_buff[slot_offset_rowwise(
          i, target_slot, key_component_count, target_slot_count)] =
          (target_info.agg_kind == kCOUNT ? 0 : 0xdeadbeef);
    }
  }
  for (size_t i = 0; i < query_mem_desc.getEntryCount(); i += step) {
    const auto v = generator.getNextValue();
    std::vector<int64_t> key(key_component_count, v);
    auto value_slots = get_group_value(i64_buff,
                                       query_mem_desc.getEntryCount(),
                                       &key[0],
                                       key.size(),
                                       sizeof(int64_t),
                                       key_component_count + target_slot_count,
                                       nullptr);
    CHECK(value_slots);
    fill_one_entry_baseline(value_slots, v, target_infos);
  }
}

void fill_storage_buffer(int8_t* buff,
                         const std::vector<TargetInfo>& target_infos,
                         const QueryMemoryDescriptor& query_mem_desc,
                         NumberGenerator& generator,
                         const size_t step) {
  switch (query_mem_desc.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByPerfectHash: {
      if (query_mem_desc.didOutputColumnar()) {
        fill_storage_buffer_perfect_hash_colwise(
            buff, target_infos, query_mem_desc, generator, step);
      } else {
        fill_storage_buffer_perfect_hash_rowwise(
            buff, target_infos, query_mem_desc, generator, step);
      }
      break;
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      if (query_mem_desc.didOutputColumnar()) {
        fill_storage_buffer_baseline_colwise(
            buff, target_infos, query_mem_desc, generator, step);
      } else {
        fill_storage_buffer_baseline_rowwise(
            buff, target_infos, query_mem_desc, generator, step);
      }
      break;
    }
    default:
      CHECK(false);
  }
  CHECK(buff);
}

// TODO(alex): allow 4 byte keys

/* descriptor with small entry_count to simplify testing and debugging */
QueryMemoryDescriptor perfect_hash_one_col_desc_small(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes) {
  QueryMemoryDescriptor query_mem_desc(
      QueryDescriptionType::GroupByPerfectHash, 0, 19, false, {8});
  for (const auto& target_info : target_infos) {
    const auto slot_bytes =
        std::max(num_bytes, static_cast<int8_t>(target_info.sql_type.get_size()));
    std::vector<std::tuple<int8_t, int8_t>> slots_for_target;
    if (target_info.agg_kind == kAVG) {
      CHECK(target_info.is_agg);
      slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    }
    slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    query_mem_desc.addColSlotInfo(slots_for_target);
  }
  query_mem_desc.setEntryCount(query_mem_desc.getMaxVal() - query_mem_desc.getMinVal() +
                               1);
  return query_mem_desc;
}

QueryMemoryDescriptor perfect_hash_one_col_desc(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes,
    const size_t min_val,
    const size_t max_val,
    std::vector<int8_t> group_column_widths) {
  QueryMemoryDescriptor query_mem_desc(QueryDescriptionType::GroupByPerfectHash,
                                       min_val,
                                       max_val,
                                       false,
                                       group_column_widths);
  for (const auto& target_info : target_infos) {
    const auto slot_bytes =
        std::max(num_bytes, static_cast<int8_t>(target_info.sql_type.get_size()));
    std::vector<std::tuple<int8_t, int8_t>> slots_for_target;
    if (target_info.agg_kind == kAVG) {
      CHECK(target_info.is_agg);
      slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    }
    slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    if (target_info.sql_type.is_geometry()) {
      for (int i = 1; i < 2 * target_info.sql_type.get_physical_coord_cols(); i++) {
        slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
      }
    } else if (target_info.sql_type.is_varlen()) {
      slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    }
    query_mem_desc.addColSlotInfo(slots_for_target);
  }
  query_mem_desc.setEntryCount(query_mem_desc.getMaxVal() - query_mem_desc.getMinVal() +
                               1);
  return query_mem_desc;
}

QueryMemoryDescriptor perfect_hash_two_col_desc(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes) {
  QueryMemoryDescriptor query_mem_desc(
      QueryDescriptionType::GroupByPerfectHash, 0, 36, false, {8, 8});
  for (const auto& target_info : target_infos) {
    const auto slot_bytes =
        std::max(num_bytes, static_cast<int8_t>(target_info.sql_type.get_size()));
    std::vector<std::tuple<int8_t, int8_t>> slots_for_target;
    if (target_info.agg_kind == kAVG) {
      CHECK(target_info.is_agg);
      slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    }
    slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    query_mem_desc.addColSlotInfo(slots_for_target);
  }
  query_mem_desc.setEntryCount(query_mem_desc.getMaxVal());
  return query_mem_desc;
}

QueryMemoryDescriptor baseline_hash_two_col_desc_large(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes) {
  QueryMemoryDescriptor query_mem_desc(
      QueryDescriptionType::GroupByBaselineHash, 0, 19, false, {8, 8});
  for (const auto& target_info : target_infos) {
    const auto slot_bytes =
        std::max(num_bytes, static_cast<int8_t>(target_info.sql_type.get_size()));
    std::vector<std::tuple<int8_t, int8_t>> slots_for_target;
    if (target_info.agg_kind == kAVG) {
      CHECK(target_info.is_agg);
      slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    }
    slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    query_mem_desc.addColSlotInfo(slots_for_target);
  }
  query_mem_desc.setEntryCount(query_mem_desc.getMaxVal() - query_mem_desc.getMinVal() +
                               1);
  return query_mem_desc;
}

QueryMemoryDescriptor baseline_hash_two_col_desc(
    const std::vector<TargetInfo>& target_infos,
    const int8_t num_bytes) {
  QueryMemoryDescriptor query_mem_desc(
      QueryDescriptionType::GroupByBaselineHash, 0, 3, false, {8, 8});
  for (const auto& target_info : target_infos) {
    const auto slot_bytes =
        std::max(num_bytes, static_cast<int8_t>(target_info.sql_type.get_size()));
    std::vector<std::tuple<int8_t, int8_t>> slots_for_target;
    if (target_info.agg_kind == kAVG) {
      CHECK(target_info.is_agg);
      slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    }
    slots_for_target.emplace_back(std::make_tuple(slot_bytes, slot_bytes));
    query_mem_desc.addColSlotInfo(slots_for_target);
  }
  query_mem_desc.setEntryCount(query_mem_desc.getMaxVal() - query_mem_desc.getMinVal() +
                               1);
  return query_mem_desc;
}

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
      case kTINYINT:
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
        value_slots[target_slot] =
            null_val ? vv : *reinterpret_cast<const int64_t*>(may_alias_ptr(&di));
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

std::vector<TargetInfo> generate_custom_agg_target_infos(
    std::vector<int8_t> group_columns,
    std::vector<SQLAgg> sql_aggs,
    std::vector<SQLTypes> agg_types,
    std::vector<SQLTypes> arg_types) {
  const auto num_targets = sql_aggs.size();
  CHECK_EQ(agg_types.size(), num_targets);
  CHECK_EQ(arg_types.size(), num_targets);
  std::vector<TargetInfo> target_infos;
  target_infos.reserve(group_columns.size() + num_targets);
  auto find_group_col_type = [](int8_t group_width) {
    switch (group_width) {
      case 8:
        return kBIGINT;
      case 4:
        return kINT;
      case 2:
        return kSMALLINT;
      case 1:
        return kTINYINT;
      default:
        UNREACHABLE();
    }
    UNREACHABLE();
    return kINT;
  };
  // creating proper TargetInfo to represent group columns:
  for (auto group_size : group_columns) {
    target_infos.push_back(TargetInfo{false,
                                      kMIN,
                                      SQLTypeInfo{find_group_col_type(group_size), false},
                                      SQLTypeInfo{kNULLT, false},
                                      true,
                                      false});
  }
  // creating proper TargetInfo for aggregate columns:
  for (size_t i = 0; i < num_targets; i++) {
    target_infos.push_back(TargetInfo{true,
                                      sql_aggs[i],
                                      SQLTypeInfo{agg_types[i], false},
                                      SQLTypeInfo{arg_types[i], false},
                                      true,
                                      false});
  }
  return target_infos;
}

std::unordered_map<size_t, size_t> get_slot_to_target_mapping(
    const std::vector<TargetInfo>& target_infos) {
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
