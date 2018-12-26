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

/**
 * @file    ResultSetBaselineRadixSortTest.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Unit tests for the result set baseline layout sort.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */
#include "../QueryEngine/ResultRows.h"
#include "../QueryEngine/ResultSet.h"
#include "../QueryEngine/RuntimeFunctions.h"
#include "ResultSetTestUtils.h"

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

QueryMemoryDescriptor baseline_sort_desc(const std::vector<TargetInfo>& target_infos,
                                         const size_t hash_entry_count,
                                         const size_t key_bytewidth) {
  QueryMemoryDescriptor query_mem_desc(
      QueryDescriptionType::GroupByBaselineHash, 0, 0, false, {8, 8});
  query_mem_desc.setGroupColCompactWidth(key_bytewidth);
  static const size_t slot_bytes = 8;
  for (size_t i = 0; i < target_infos.size(); ++i) {
    query_mem_desc.addAggColWidth(ColWidths{slot_bytes, slot_bytes});
  }
  query_mem_desc.setEntryCount(hash_entry_count);
  return query_mem_desc;
}

template <class K>
void fill_storage_buffer_baseline_sort_int(int8_t* buff,
                                           const std::vector<TargetInfo>& target_infos,
                                           const QueryMemoryDescriptor& query_mem_desc,
                                           const int64_t upper_bound,
                                           const int64_t empty_key) {
  const auto key_component_count = query_mem_desc.getKeyCount();
  const auto target_slot_count = get_slot_count(target_infos);
  const auto slot_to_target = get_slot_to_target_mapping(target_infos);
  const auto row_bytes = get_row_bytes(query_mem_desc);
  for (size_t i = 0; i < query_mem_desc.getEntryCount(); ++i) {
    const auto row_ptr = buff + i * row_bytes;
    for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
      reinterpret_cast<K*>(row_ptr)[key_comp_idx] = empty_key;
    }
    for (size_t target_slot = 0; target_slot < target_slot_count; ++target_slot) {
      auto target_it = slot_to_target.find(target_slot);
      CHECK(target_it != slot_to_target.end());
      const auto& target_info = target_infos[target_it->second];

      const auto cols_ptr = reinterpret_cast<int64_t*>(
          row_ptr + get_slot_off_quad(query_mem_desc) * sizeof(int64_t));
      cols_ptr[target_slot] = (target_info.agg_kind == kCOUNT ? 0 : 0xdeadbeef);
    }
  }
  std::vector<int64_t> values(upper_bound);
  std::iota(values.begin(), values.end(), 1);
  const auto null_pattern = null_val_bit_pattern(target_infos.back().sql_type, false);
  values.push_back(null_pattern);
  std::random_shuffle(values.begin(), values.end());
  CHECK_EQ(size_t(0), row_bytes % 8);
  const auto row_size_quad = row_bytes / 8;
  for (const auto val : values) {
    std::vector<K> key(key_component_count, val);
    auto value_slots = get_group_value(reinterpret_cast<int64_t*>(buff),
                                       query_mem_desc.getEntryCount(),
                                       reinterpret_cast<const int64_t*>(&key[0]),
                                       key.size(),
                                       sizeof(K),
                                       row_size_quad,
                                       nullptr);
    CHECK(value_slots);
    fill_one_entry_baseline(value_slots, val, target_infos);
  }
}

void fill_storage_buffer_baseline_sort_fp(int8_t* buff,
                                          const std::vector<TargetInfo>& target_infos,
                                          const QueryMemoryDescriptor& query_mem_desc,
                                          const int64_t upper_bound) {
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
  std::vector<int64_t> values(upper_bound);
  std::iota(values.begin(), values.end(), 1);
  const auto null_pattern = null_val_bit_pattern(target_infos.back().sql_type, false);
  values.push_back(null_pattern);
  std::random_shuffle(values.begin(), values.end());
  for (const auto val : values) {
    std::vector<int64_t> key(key_component_count, val);
    auto value_slots = get_group_value(i64_buff,
                                       query_mem_desc.getEntryCount(),
                                       &key[0],
                                       key.size(),
                                       sizeof(int64_t),
                                       key_component_count + target_slot_count,
                                       nullptr);
    CHECK(value_slots);
    fill_one_entry_baseline(value_slots, val, target_infos, false, val == null_pattern);
  }
}

template <class T>
void check_sorted(const ResultSet& result_set,
                  const int64_t bound,
                  const size_t top_n,
                  const size_t desc) {
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

template <class K>
int64_t empty_key_val();

template <>
int64_t empty_key_val<int64_t>() {
  return EMPTY_KEY_64;
}

template <>
int64_t empty_key_val<int32_t>() {
  return EMPTY_KEY_32;
}

template <class K>
void SortBaselineIntegersTestImpl(const bool desc) {
  const auto target_infos = get_sort_int_target_infos();
  const auto query_mem_desc = baseline_sort_desc(target_infos, 400, sizeof(K));
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
  const int64_t upper_bound = 200;
  const int64_t lower_bound = 1;
  std::unique_ptr<ResultSet> rs(new ResultSet(
      target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, nullptr));
  auto storage = rs->allocateStorage();
  fill_storage_buffer_baseline_sort_int<K>(storage->getUnderlyingBuffer(),
                                           target_infos,
                                           query_mem_desc,
                                           upper_bound,
                                           empty_key_val<K>());
  std::list<Analyzer::OrderEntry> order_entries;
  order_entries.emplace_back(3, desc, false);
  const size_t top_n = 5;
  rs->sort(order_entries, top_n);
  check_sorted<int64_t>(*rs, desc ? upper_bound : lower_bound, top_n, desc);
}

}  // namespace

TEST(SortBaseline, IntegersKey64) {
  for (const bool desc : {true, false}) {
    SortBaselineIntegersTestImpl<int64_t>(desc);
  }
}

TEST(SortBaseline, IntegersKey32) {
  for (const bool desc : {true, false}) {
    SortBaselineIntegersTestImpl<int32_t>(desc);
  }
}

TEST(SortBaseline, Floats) {
  for (const bool desc : {true, false}) {
    for (int tle_no = 1; tle_no <= 3; ++tle_no) {
      const auto target_infos = get_sort_fp_target_infos();
      const auto query_mem_desc = baseline_sort_desc(target_infos, 400, 8);
      const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>();
      const int64_t upper_bound = 200;
      std::unique_ptr<ResultSet> rs(new ResultSet(target_infos,
                                                  ExecutorDeviceType::CPU,
                                                  query_mem_desc,
                                                  row_set_mem_owner,
                                                  nullptr));
      auto storage = rs->allocateStorage();
      fill_storage_buffer_baseline_sort_fp(
          storage->getUnderlyingBuffer(), target_infos, query_mem_desc, upper_bound);
      std::list<Analyzer::OrderEntry> order_entries;
      order_entries.emplace_back(tle_no, desc, false);
      const size_t top_n = 5;
      rs->sort(order_entries, top_n);
      check_sorted<float>(*rs, desc ? upper_bound : 1, top_n, desc);
    }
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);

#ifdef HAVE_CUDA
  try {
    g_cuda_mgr.reset(new CudaMgr_Namespace::CudaMgr(0));
  } catch (...) {
    LOG(WARNING) << "Could not instantiate CudaMgr, will run on CPU";
  }
#endif  // HAVE_CUDA

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

#ifdef HAVE_CUDA
  g_cuda_mgr.reset(nullptr);
#endif  // HAVE_CUDA

  return err;
}
