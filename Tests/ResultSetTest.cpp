/**
 * @file    ResultSetTest.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Unit tests for the result set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "../QueryEngine/ResultSet.h"
#include "../QueryEngine/RuntimeFunctions.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>

TEST(Construct, Empty) {
  ResultSet result_set;
  ASSERT_TRUE(result_set.isEmptyInitializer());
}

TEST(Construct, Allocate) {
  std::vector<TargetInfo> target_infos;
  QueryMemoryDescriptor query_mem_desc{};
  ResultSet result_set(target_infos, ExecutorDeviceType::CPU, query_mem_desc, std::make_shared<RowSetMemoryOwner>());
  result_set.allocateStorage();
}

namespace {

size_t get_slot_count(const std::vector<TargetInfo>& target_infos) {
  size_t count = 0;
  for (const auto& target_info : target_infos) {
    count = advance_slot(count, target_info);
  }
  return count;
}

class NumberGenerator {
 public:
  virtual int64_t getNextValue() = 0;

  virtual void reset() = 0;
};

class EvenNumberGenerator : public NumberGenerator {
 public:
  EvenNumberGenerator() : crt_(0) {}

  int64_t getNextValue() override {
    const auto crt = crt_;
    crt_ += 2;
    return crt;
  }

  void reset() override { crt_ = 0; }

 private:
  int64_t crt_;
};

class ReverseOddOrEvenNumberGenerator : public NumberGenerator {
 public:
  ReverseOddOrEvenNumberGenerator(const int64_t init) : crt_(init), init_(init) {}

  int64_t getNextValue() override {
    const auto crt = crt_;
    crt_ -= 2;
    return crt;
  }

  void reset() override { crt_ = init_; }

 private:
  int64_t crt_;
  int64_t init_;
};

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
      *reinterpret_cast<int32_t*>(slot_ptr) = *reinterpret_cast<int32_t*>(&fi);
      break;
    }
    case 8: {
      double di = v;
      *reinterpret_cast<int64_t*>(slot_ptr) = *reinterpret_cast<int64_t*>(&di);
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
                                     const bool empty) {
  size_t target_idx = 0;
  int8_t* slot_ptr = buff;
  for (const auto& target_info : target_infos) {
    CHECK_LT(target_idx, query_mem_desc.agg_col_widths.size());
    const auto slot_bytes = query_mem_desc.agg_col_widths[target_idx].actual;
    CHECK_LE(target_info.sql_type.get_size(), slot_bytes);
    if (empty) {
      write_int(slot_ptr, query_mem_desc.keyless_hash ? 0 : v, slot_bytes);
    } else {
      if (target_info.sql_type.is_integer()) {
        write_int(slot_ptr, v, slot_bytes);
      } else if (target_info.sql_type.is_string()) {
        write_int(slot_ptr, -(v + 2), slot_bytes);
      } else {
        CHECK(target_info.sql_type.is_fp());
        write_fp(slot_ptr, v, slot_bytes);
      }
    }
    slot_ptr += slot_bytes;
    if (target_info.agg_kind == kAVG) {
      const auto count_slot_bytes = query_mem_desc.agg_col_widths[target_idx + 1].actual;
      if (empty) {
        write_int(slot_ptr, query_mem_desc.keyless_hash ? 0 : v, count_slot_bytes);
      } else {
        write_int(slot_ptr, 1, count_slot_bytes);
      }
      slot_ptr += count_slot_bytes;
    }
    target_idx = advance_slot(target_idx, target_info);
  }
  return slot_ptr;
}

void fill_one_entry_baseline(int64_t* value_slots,
                             const int64_t v,
                             const std::vector<TargetInfo>& target_infos,
                             const size_t key_component_count,
                             const size_t target_slot_count) {
  size_t target_slot = 0;
  for (const auto& target_info : target_infos) {
    switch (target_info.sql_type.get_type()) {
      case kSMALLINT:
      case kINT:
      case kBIGINT:
        value_slots[target_slot] = v;
        break;
      case kDOUBLE: {
        double di = v;
        value_slots[target_slot] = *reinterpret_cast<int64_t*>(&di);
        break;
      }
      case kTEXT:
        value_slots[target_slot] = -(v + 2);
        break;
      default:
        CHECK(false);
    }
    if (target_info.agg_kind == kAVG) {
      value_slots[target_slot + 1] = 1;
    }
    target_slot = advance_slot(target_slot, target_info);
  }
}

void fill_one_entry_one_col(int8_t* ptr1,
                            const int8_t compact_sz1,
                            int8_t* ptr2,
                            const int8_t compact_sz2,
                            const int64_t v,
                            const TargetInfo& target_info,
                            const bool empty_entry) {
  CHECK(ptr1);
  switch (compact_sz1) {
    case 8:
      if (target_info.sql_type.is_fp()) {
        double di = v;
        *reinterpret_cast<int64_t*>(ptr1) = *reinterpret_cast<int64_t*>(&di);
      } else {
        *reinterpret_cast<int64_t*>(ptr1) = v;
      }
      break;
    case 4:
      CHECK(!target_info.sql_type.is_fp());
      *reinterpret_cast<int32_t*>(ptr1) = v;
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
                            const size_t entry_count) {
  auto ptr1 = reinterpret_cast<int8_t*>(value_slot);
  int8_t* ptr2{nullptr};
  if (target_info.agg_kind == kAVG) {
    ptr2 = reinterpret_cast<int8_t*>(&value_slot[entry_count]);
  }
  fill_one_entry_one_col(ptr1, 8, ptr2, 8, v, target_info, false);
}

int8_t* advance_to_next_columnar_key_buff(int8_t* key_ptr,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          const size_t key_idx) {
  CHECK(!query_mem_desc.keyless_hash);
  CHECK_LT(key_idx, query_mem_desc.group_col_widths.size());
  auto new_key_ptr = key_ptr + query_mem_desc.entry_count * query_mem_desc.group_col_widths[key_idx];
  if (!query_mem_desc.key_column_pad_bytes.empty()) {
    CHECK_LT(key_idx, query_mem_desc.key_column_pad_bytes.size());
    new_key_ptr += query_mem_desc.key_column_pad_bytes[key_idx];
  }
  return new_key_ptr;
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
    default:
      CHECK(false);
  }
}

void fill_storage_buffer_perfect_hash_colwise(int8_t* buff,
                                              const std::vector<TargetInfo>& target_infos,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              NumberGenerator& generator) {
  const auto key_component_count = get_key_count_for_descriptor(query_mem_desc);
  CHECK(query_mem_desc.output_columnar);
  // initialize the key buffer(s)
  auto col_ptr = buff;
  for (size_t key_idx = 0; key_idx < key_component_count; ++key_idx) {
    auto key_entry_ptr = col_ptr;
    const auto key_bytes = query_mem_desc.group_col_widths[key_idx];
    CHECK_EQ(8, key_bytes);
    for (size_t i = 0; i < query_mem_desc.entry_count; ++i) {
      if (i % 2 == 0) {
        const auto v = generator.getNextValue();
        write_key(v, key_entry_ptr, key_bytes);
      } else {
        write_key(EMPTY_KEY_64, key_entry_ptr, key_bytes);
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
    const auto col_bytes = query_mem_desc.agg_col_widths[slot_idx].compact;
    for (size_t i = 0; i < query_mem_desc.entry_count; ++i) {
      int8_t* ptr2{nullptr};
      if (target_info.agg_kind == kAVG) {
        ptr2 = col_entry_ptr + query_mem_desc.entry_count * col_bytes;
      }
      if (i % 2 == 0) {
        const auto gen_val = generator.getNextValue();
        const auto val = target_info.sql_type.is_string() ? -(gen_val + 2) : gen_val;
        fill_one_entry_one_col(col_entry_ptr,
                               col_bytes,
                               ptr2,
                               query_mem_desc.agg_col_widths[slot_idx + 1].compact,
                               val,
                               target_info,
                               false);
      } else {
        fill_one_entry_one_col(col_entry_ptr,
                               col_bytes,
                               ptr2,
                               query_mem_desc.agg_col_widths[slot_idx + 1].compact,
                               query_mem_desc.keyless_hash ? 0 : 0xdeadbeef,
                               target_info,
                               true);
      }
      col_entry_ptr += col_bytes;
    }
    col_ptr = advance_to_next_columnar_target_buff(col_ptr, query_mem_desc, slot_idx);
    if (target_info.is_agg && target_info.agg_kind == kAVG) {
      col_ptr = advance_to_next_columnar_target_buff(col_ptr, query_mem_desc, slot_idx + 1);
    }
    slot_idx = advance_slot(slot_idx, target_info);
    generator.reset();
  }
}

void fill_storage_buffer_perfect_hash_rowwise(int8_t* buff,
                                              const std::vector<TargetInfo>& target_infos,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              NumberGenerator& generator) {
  const auto key_component_count = get_key_count_for_descriptor(query_mem_desc);
  CHECK(!query_mem_desc.output_columnar);
  auto key_buff = buff;
  for (size_t i = 0; i < query_mem_desc.entry_count; ++i) {
    if (i % 2 == 0) {
      const auto v = generator.getNextValue();
      auto key_buff_i64 = reinterpret_cast<int64_t*>(key_buff);
      for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
        *key_buff_i64++ = v;
      }
      auto entries_buff = reinterpret_cast<int8_t*>(key_buff_i64);
      key_buff = fill_one_entry_no_collisions(entries_buff, query_mem_desc, v, target_infos, false);
    } else {
      auto key_buff_i64 = reinterpret_cast<int64_t*>(key_buff);
      for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
        *key_buff_i64++ = EMPTY_KEY_64;
      }
      auto entries_buff = reinterpret_cast<int8_t*>(key_buff_i64);
      key_buff = fill_one_entry_no_collisions(entries_buff, query_mem_desc, 0xdeadbeef, target_infos, true);
    }
  }
}

void fill_storage_buffer_baseline_colwise(int8_t* buff,
                                          const std::vector<TargetInfo>& target_infos,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          NumberGenerator& generator,
                                          const size_t step) {
  CHECK(query_mem_desc.output_columnar);
  const auto key_component_count = get_key_count_for_descriptor(query_mem_desc);
  const auto i64_buff = reinterpret_cast<int64_t*>(buff);
  const auto target_slot_count = get_slot_count(target_infos);
  for (size_t i = 0; i < query_mem_desc.entry_count; ++i) {
    for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
      i64_buff[key_offset_colwise(i, key_comp_idx, query_mem_desc.entry_count)] = EMPTY_KEY_64;
    }
    for (size_t target_slot = 0; target_slot < target_slot_count; ++target_slot) {
      i64_buff[slot_offset_colwise(i, target_slot, key_component_count, query_mem_desc.entry_count)] = 0xdeadbeef;
    }
  }
  for (size_t i = 0; i < query_mem_desc.entry_count; i += step) {
    const auto gen_val = generator.getNextValue();
    std::vector<int64_t> key(key_component_count, gen_val);
    auto value_slots = get_group_value_columnar(i64_buff, query_mem_desc.entry_count, &key[0], key.size());
    CHECK(value_slots);
    for (const auto& target_info : target_infos) {
      const auto val = target_info.sql_type.is_string() ? -(gen_val + step) : gen_val;
      fill_one_entry_one_col(value_slots, val, target_info, query_mem_desc.entry_count);
      value_slots += query_mem_desc.entry_count;
      if (target_info.agg_kind == kAVG) {
        value_slots += query_mem_desc.entry_count;
      }
    }
  }
}

void fill_storage_buffer_baseline_rowwise(int8_t* buff,
                                          const std::vector<TargetInfo>& target_infos,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          NumberGenerator& generator,
                                          const size_t step) {
  const auto key_component_count = get_key_count_for_descriptor(query_mem_desc);
  const auto i64_buff = reinterpret_cast<int64_t*>(buff);
  const auto target_slot_count = get_slot_count(target_infos);
  for (size_t i = 0; i < query_mem_desc.entry_count; ++i) {
    const auto first_key_comp_offset = key_offset_rowwise(i, key_component_count, target_slot_count);
    for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
      i64_buff[first_key_comp_offset + key_comp_idx] = EMPTY_KEY_64;
    }
    for (size_t target_slot = 0; target_slot < target_slot_count; ++target_slot) {
      i64_buff[slot_offset_rowwise(i, target_slot, key_component_count, target_slot_count)] = 0xdeadbeef;
    }
  }
  for (size_t i = 0; i < query_mem_desc.entry_count; i += step) {
    const auto v = generator.getNextValue();
    std::vector<int64_t> key(key_component_count, v);
    auto value_slots = get_group_value(
        i64_buff, query_mem_desc.entry_count, &key[0], key.size(), key_component_count + target_slot_count, nullptr);
    CHECK(value_slots);
    fill_one_entry_baseline(value_slots, v, target_infos, key_component_count, target_slot_count);
  }
}

void fill_storage_buffer(int8_t* buff,
                         const std::vector<TargetInfo>& target_infos,
                         const QueryMemoryDescriptor& query_mem_desc,
                         NumberGenerator& generator,
                         const size_t step) {
  switch (query_mem_desc.hash_type) {
    case GroupByColRangeType::OneColKnownRange:
    case GroupByColRangeType::MultiColPerfectHash: {
      if (query_mem_desc.output_columnar) {
        fill_storage_buffer_perfect_hash_colwise(buff, target_infos, query_mem_desc, generator);
      } else {
        fill_storage_buffer_perfect_hash_rowwise(buff, target_infos, query_mem_desc, generator);
      }
      break;
    }
    case GroupByColRangeType::MultiCol: {
      if (query_mem_desc.output_columnar) {
        fill_storage_buffer_baseline_colwise(buff, target_infos, query_mem_desc, generator, step);
      } else {
        fill_storage_buffer_baseline_rowwise(buff, target_infos, query_mem_desc, generator, step);
      }
      break;
    }
    default:
      CHECK(false);
  }
  CHECK(buff);
}

// TODO(alex): allow 4 byte keys

QueryMemoryDescriptor perfect_hash_one_col_desc(const std::vector<TargetInfo>& target_infos, const int8_t num_bytes) {
  QueryMemoryDescriptor query_mem_desc{};
  query_mem_desc.hash_type = GroupByColRangeType::OneColKnownRange;
  query_mem_desc.min_val = 0;
  query_mem_desc.max_val = 99;
  query_mem_desc.has_nulls = false;
  query_mem_desc.group_col_widths.emplace_back(8);
  for (const auto& target_info : target_infos) {
    const auto slot_bytes = std::max(num_bytes, static_cast<int8_t>(target_info.sql_type.get_size()));
    if (target_info.agg_kind == kAVG) {
      CHECK(target_info.is_agg);
      query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
    }
    query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
  }
  query_mem_desc.entry_count = query_mem_desc.max_val - query_mem_desc.min_val + 1;
  return query_mem_desc;
}

QueryMemoryDescriptor perfect_hash_two_col_desc(const std::vector<TargetInfo>& target_infos, const int8_t num_bytes) {
  QueryMemoryDescriptor query_mem_desc{};
  query_mem_desc.hash_type = GroupByColRangeType::MultiColPerfectHash;
  query_mem_desc.min_val = 0;
  query_mem_desc.max_val = 36;
  query_mem_desc.has_nulls = false;
  query_mem_desc.group_col_widths.emplace_back(8);
  query_mem_desc.group_col_widths.emplace_back(8);
  for (const auto& target_info : target_infos) {
    const auto slot_bytes = std::max(num_bytes, static_cast<int8_t>(target_info.sql_type.get_size()));
    if (target_info.agg_kind == kAVG) {
      CHECK(target_info.is_agg);
      query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
    }
    query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
  }
  query_mem_desc.entry_count = query_mem_desc.max_val;
  return query_mem_desc;
}

QueryMemoryDescriptor baseline_hash_two_col_desc(const std::vector<TargetInfo>& target_infos, const int8_t num_bytes) {
  QueryMemoryDescriptor query_mem_desc{};
  query_mem_desc.hash_type = GroupByColRangeType::MultiCol;
  query_mem_desc.min_val = 0;
  query_mem_desc.max_val = 3;
  query_mem_desc.has_nulls = false;
  query_mem_desc.group_col_widths.emplace_back(8);
  query_mem_desc.group_col_widths.emplace_back(8);
  for (const auto& target_info : target_infos) {
    const auto slot_bytes = std::max(num_bytes, static_cast<int8_t>(target_info.sql_type.get_size()));
    if (target_info.agg_kind == kAVG) {
      CHECK(target_info.is_agg);
      query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
    }
    query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
  }
  query_mem_desc.entry_count = query_mem_desc.max_val - query_mem_desc.min_val + 1;
  return query_mem_desc;
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

template <class T>
const T* vptr(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  return boost::get<T>(scalar_r);
}

bool approx_eq(const double v, const double target, const double eps = 0.01) {
  return target - eps < v && v < target + eps;
}

StringDictionary g_sd("");

void test_iterate(const std::vector<TargetInfo>& target_infos, const QueryMemoryDescriptor& query_mem_desc) {
  SQLTypeInfo double_ti(kDOUBLE, false);
  auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
  row_set_mem_owner->addStringDict(&g_sd, 1);
  ResultSet result_set(target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner);
  for (size_t i = 0; i < query_mem_desc.entry_count; ++i) {
    g_sd.getOrAddTransient(std::to_string(i));
  }
  const auto storage = result_set.allocateStorage();
  EvenNumberGenerator generator;
  fill_storage_buffer(storage->getUnderlyingBuffer(), target_infos, query_mem_desc, generator, 2);
  int64_t ref_val{0};
  while (true) {
    const auto row = result_set.getNextRow(true, false);
    if (row.empty()) {
      break;
    }
    CHECK_EQ(target_infos.size(), row.size());
    for (size_t i = 0; i < target_infos.size(); ++i) {
      const auto& target_info = target_infos[i];
      const auto& ti = target_info.agg_kind == kAVG ? double_ti : target_info.sql_type;
      switch (ti.get_type()) {
        case kSMALLINT:
        case kINT:
        case kBIGINT: {
          const auto ival = v<int64_t>(row[i]);
          ASSERT_EQ(ref_val, ival);
          break;
        }
        case kDOUBLE: {
          const auto dval = v<double>(row[i]);
          ASSERT_TRUE(approx_eq(static_cast<double>(ref_val), dval));
          break;
        }
        case kTEXT: {
          const auto sval = v<NullableString>(row[i]);
          ASSERT_EQ(std::to_string(ref_val), boost::get<std::string>(sval));
          break;
        }
        default:
          CHECK(false);
      }
    }
    ref_val += 2;
  }
}

std::vector<TargetInfo> generate_test_target_infos() {
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo int_ti(kINT, false);
  SQLTypeInfo double_ti(kDOUBLE, false);
  SQLTypeInfo null_ti(kNULLT, false);
  target_infos.push_back(TargetInfo{false, kMIN, int_ti, null_ti, true, false});
  target_infos.push_back(TargetInfo{true, kAVG, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kSUM, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{false, kMIN, double_ti, null_ti, true, false});
  {
    SQLTypeInfo dict_string_ti(kTEXT, false);
    dict_string_ti.set_compression(kENCODING_DICT);
    dict_string_ti.set_comp_param(1);
    target_infos.push_back(TargetInfo{false, kMIN, dict_string_ti, null_ti, true, false});
  }
  return target_infos;
}

typedef std::vector<TargetValue> OneRow;

std::vector<OneRow> get_rows_sorted_by_col(const ResultSet& rs, const size_t col_idx) {
  std::vector<OneRow> result;
  while (true) {
    const auto row = rs.getNextRow(false, false);
    if (row.empty()) {
      break;
    }
    result.push_back(row);
  }
  auto sort_func = [col_idx](const OneRow& lhs, const OneRow& rhs) {
    const auto& lhs_col = lhs[col_idx];
    const auto& rhs_col = rhs[col_idx];
    const auto lhs_iptr = vptr<int64_t>(lhs_col);
    if (lhs_iptr) {
      return *lhs_iptr < v<int64_t>(rhs_col);
    }
    return v<double>(lhs_col) < v<double>(rhs_col);
  };
  std::sort(result.begin(), result.end(), sort_func);
  return result;
}

void test_reduce(const std::vector<TargetInfo>& target_infos,
                 const QueryMemoryDescriptor& query_mem_desc,
                 NumberGenerator& generator1,
                 NumberGenerator& generator2,
                 const int step) {
  SQLTypeInfo double_ti(kDOUBLE, false);
  const ResultSetStorage* storage1{nullptr};
  const ResultSetStorage* storage2{nullptr};
  std::unique_ptr<ResultSet> rs1;
  std::unique_ptr<ResultSet> rs2;
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
  row_set_mem_owner->addStringDict(&g_sd, 1);
  switch (query_mem_desc.hash_type) {
    case GroupByColRangeType::OneColKnownRange:
    case GroupByColRangeType::MultiColPerfectHash: {
      rs1.reset(new ResultSet(target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner));
      storage1 = rs1->allocateStorage();
      fill_storage_buffer(storage1->getUnderlyingBuffer(), target_infos, query_mem_desc, generator1, step);
      rs2.reset(new ResultSet(target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner));
      storage2 = rs2->allocateStorage();
      fill_storage_buffer(storage2->getUnderlyingBuffer(), target_infos, query_mem_desc, generator2, step);
      break;
    }
    case GroupByColRangeType::MultiCol: {
      rs1.reset(new ResultSet(target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner));
      storage1 = rs1->allocateStorage();
      fill_storage_buffer(storage1->getUnderlyingBuffer(), target_infos, query_mem_desc, generator1, step);
      rs2.reset(new ResultSet(target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner));
      storage2 = rs2->allocateStorage();
      fill_storage_buffer(storage2->getUnderlyingBuffer(), target_infos, query_mem_desc, generator2, step);
      break;
    }
    default:
      CHECK(false);
  }
  ResultSetManager rs_manager;
  std::vector<ResultSet*> storage_set{rs1.get(), rs2.get()};
  auto result_rs = rs_manager.reduce(storage_set);
  int64_t ref_val{0};
  const auto result = get_rows_sorted_by_col(*result_rs, 0);
  CHECK(!result.empty());
  for (const auto& row : result) {
    CHECK_EQ(target_infos.size(), row.size());
    for (size_t i = 0; i < target_infos.size(); ++i) {
      const auto& target_info = target_infos[i];
      const auto& ti = target_info.agg_kind == kAVG ? double_ti : target_info.sql_type;
      switch (ti.get_type()) {
        case kSMALLINT:
        case kINT:
        case kBIGINT: {
          const auto ival = v<int64_t>(row[i]);
          ASSERT_EQ((target_info.agg_kind == kSUM || target_info.agg_kind == kCOUNT) ? step * ref_val : ref_val, ival);
          break;
        }
        case kDOUBLE: {
          const auto dval = v<double>(row[i]);
          ASSERT_TRUE(approx_eq(
              static_cast<double>((target_info.agg_kind == kSUM || target_info.agg_kind == kCOUNT) ? step * ref_val
                                                                                                   : ref_val),
              dval));
          break;
        }
        case kTEXT:
          break;
        default:
          CHECK(false);
      }
    }
    ref_val += step;
  }
}

}  // namespace

TEST(Iterate, PerfectHashOneCol) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneCol32) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnar32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4);
  query_mem_desc.output_columnar = true;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8);
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4);
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnarKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnarKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4);
  query_mem_desc.output_columnar = true;
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoCol) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoCol32) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColColumnar32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.output_columnar = true;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColColumnarKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColColumnarKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.output_columnar = true;
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, BaselineHash) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc(target_infos, 8);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, BaselineHashColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  test_iterate(target_infos, query_mem_desc);
}

TEST(Reduce, PerfectHashOneCol) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashOneCol32) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashOneColColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashOneColColumnar32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4);
  query_mem_desc.output_columnar = true;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashOneColKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8);
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashOneColKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4);
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashOneColColumnarKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashOneColColumnarKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4);
  query_mem_desc.output_columnar = true;
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashTwoCol) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashTwoCol32) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashTwoColColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashTwoColColumnar32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.output_columnar = true;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashTwoColKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashTwoColKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashTwoColColumnarKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, PerfectHashTwoColColumnarKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.output_columnar = true;
  query_mem_desc.keyless_hash = true;
  query_mem_desc.idx_target_as_key = 2;
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2);
}

TEST(Reduce, BaselineHash) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc(target_infos, 8);
  EvenNumberGenerator generator1;
  ReverseOddOrEvenNumberGenerator generator2(2 * query_mem_desc.entry_count - 1);
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 1);
}

TEST(Reduce, BaselineHashColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, 8);
  query_mem_desc.output_columnar = true;
  EvenNumberGenerator generator1;
  ReverseOddOrEvenNumberGenerator generator2(2 * query_mem_desc.entry_count - 1);
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 1);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  auto err = RUN_ALL_TESTS();
  return err;
}
