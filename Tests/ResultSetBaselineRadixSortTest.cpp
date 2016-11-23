/**
 * @file    ResultSetBaselineRadixSortTest.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Unit tests for the result set baseline layout sort.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */
#include "ResultSetTestUtils.h"
#include "../QueryEngine/ResultRows.h"
#include "../QueryEngine/ResultSet.h"
#include "../QueryEngine/RuntimeFunctions.h"

#ifdef HAVE_CUDA
#include "../CudaMgr/CudaMgr.h"

extern std::unique_ptr<CudaMgr_Namespace::CudaMgr> g_cuda_mgr;
#endif  // HAVE_CUDA

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <numeric>

namespace {

std::vector<TargetInfo> get_sort_int_target_infos() {
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo int_ti(kINT, false);
  SQLTypeInfo null_ti(kNULLT, false);
  target_infos.push_back(TargetInfo{false, kMIN, int_ti, null_ti, false, false});
  target_infos.push_back(TargetInfo{false, kMIN, int_ti, null_ti, false, false});
  target_infos.push_back(TargetInfo{true, kCOUNT, int_ti, null_ti, false, false});
  return target_infos;
}

QueryMemoryDescriptor baseline_sort_desc(const std::vector<TargetInfo>& target_infos, const size_t hash_entry_count) {
  QueryMemoryDescriptor query_mem_desc{};
  query_mem_desc.hash_type = GroupByColRangeType::MultiCol;
  query_mem_desc.group_col_widths.emplace_back(8);
  query_mem_desc.group_col_widths.emplace_back(8);
  static const size_t slot_bytes = 8;
  for (size_t i = 0; i < target_infos.size(); ++i) {
    query_mem_desc.agg_col_widths.emplace_back(ColWidths{slot_bytes, slot_bytes});
  }
  query_mem_desc.entry_count = hash_entry_count;
  return query_mem_desc;
}

void fill_storage_buffer_baseline_sort_int(int8_t* buff,
                                           const std::vector<TargetInfo>& target_infos,
                                           const QueryMemoryDescriptor& query_mem_desc,
                                           const int64_t upper_bound) {
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
  std::vector<int64_t> values(upper_bound);
  std::iota(values.begin(), values.end(), 1);
  std::random_shuffle(values.begin(), values.end());
  for (const auto val : values) {
    std::vector<int64_t> key(key_component_count, val);
    auto value_slots = get_group_value(
        i64_buff, query_mem_desc.entry_count, &key[0], key.size(), key_component_count + target_slot_count, nullptr);
    CHECK(value_slots);
    fill_one_entry_baseline(value_slots, val, target_infos);
  }
}

void fill_storage_buffer_baseline_sort_fp(int8_t* buff,
                                          const std::vector<TargetInfo>& target_infos,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          const int64_t upper_bound) {
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
  std::vector<int64_t> values(upper_bound);
  std::iota(values.begin(), values.end(), 1);
  const auto null_pattern = null_val_bit_pattern(target_infos.back().sql_type);
  values.push_back(null_pattern);
  std::random_shuffle(values.begin(), values.end());
  for (const auto val : values) {
    std::vector<int64_t> key(key_component_count, val);
    auto value_slots = get_group_value(
        i64_buff, query_mem_desc.entry_count, &key[0], key.size(), key_component_count + target_slot_count, nullptr);
    CHECK(value_slots);
    fill_one_entry_baseline(value_slots, val, target_infos, false, val == null_pattern);
  }
}

template <class T>
void check_sorted(const ResultSet& result_set, const int64_t bound, const size_t top_n, const size_t desc) {
  ASSERT_EQ(top_n, result_set.rowCount());
  T ref_val = bound;
  while (true) {
    const auto row = result_set.getNextRow(true, false);
    if (row.empty()) {
      break;
    }
    ASSERT_EQ(size_t(3), row.size());
    const auto ival = v<T>(row[2]);
    ASSERT_EQ(ref_val, ival);
    if (desc) {
      --ref_val;
    } else {
      ++ref_val;
    }
  }
}

std::vector<TargetInfo> get_sort_fp_target_infos() {
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo null_ti(kNULLT, false);
  SQLTypeInfo fp_ti(kFLOAT, false);
  target_infos.push_back(TargetInfo{false, kMIN, fp_ti, null_ti, false, false});
  target_infos.push_back(TargetInfo{false, kMIN, fp_ti, null_ti, false, false});
  target_infos.push_back(TargetInfo{true, kSUM, fp_ti, fp_ti, true, false});
  return target_infos;
}

}  // namespace

TEST(SortBaseline, Integers) {
  const auto target_infos = get_sort_int_target_infos();
  const auto query_mem_desc = baseline_sort_desc(target_infos, 400);
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
  const int64_t upper_bound = 200;
  std::unique_ptr<ResultSet> rs(
      new ResultSet(target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, nullptr));
  auto storage = rs->allocateStorage();
  fill_storage_buffer_baseline_sort_int(storage->getUnderlyingBuffer(), target_infos, query_mem_desc, upper_bound);
  std::list<Analyzer::OrderEntry> order_entries;
  const bool desc = true;
  order_entries.emplace_back(3, desc, false);
  const size_t top_n = 5;
  rs->sort(order_entries, top_n);
  check_sorted<int64_t>(*rs, upper_bound, top_n, desc);
}

TEST(SortBaseline, Floats) {
  for (const bool desc : {true, false}) {
    for (int tle_no = 1; tle_no <= 3; ++tle_no) {
      const auto target_infos = get_sort_fp_target_infos();
      const auto query_mem_desc = baseline_sort_desc(target_infos, 400);
      const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
      const int64_t upper_bound = 200;
      std::unique_ptr<ResultSet> rs(
          new ResultSet(target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, nullptr));
      auto storage = rs->allocateStorage();
      fill_storage_buffer_baseline_sort_fp(storage->getUnderlyingBuffer(), target_infos, query_mem_desc, upper_bound);
      std::list<Analyzer::OrderEntry> order_entries;
      order_entries.emplace_back(tle_no, desc, false);
      const size_t top_n = 5;
      rs->sort(order_entries, top_n);
      check_sorted<float>(*rs, desc ? upper_bound : 1, top_n, desc);
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
#ifdef HAVE_CUDA
  g_cuda_mgr.reset(new CudaMgr_Namespace::CudaMgr(0));
#endif  // HAVE_CUDA
  auto err = RUN_ALL_TESTS();
#ifdef HAVE_CUDA
  g_cuda_mgr.reset(nullptr);
#endif  // HAVE_CUDA
  return err;
}
