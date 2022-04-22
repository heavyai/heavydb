/*
 * Copyright 2020 OmniSci, Inc.
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
 * @file    ResultSetTest.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Unit tests for the result set interface.
 *
 */

#include "ArrowSQLRunner/ArrowSQLRunner.h"
#include "ResultSetTestUtils.h"
#include "TestHelpers.h"

#include "DataMgr/DataMgrDataProvider.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/ResultSetReductionJIT.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "StringDictionary/StringDictionary.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <queue>
#include <random>

using namespace TestHelpers::ArrowSQLRunner;

std::shared_ptr<DataMgrDataProvider> g_data_provider;

extern bool g_is_test_env;

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

TEST(Construct, Allocate) {
  std::vector<TargetInfo> target_infos;
  QueryMemoryDescriptor query_mem_desc;
  ResultSet result_set(target_infos,
                       ExecutorDeviceType::CPU,
                       query_mem_desc,
                       std::make_shared<RowSetMemoryOwner>(g_data_provider.get(),
                                                           Executor::getArenaBlockSize()),
                       nullptr,
                       nullptr,
                       -1,
                       0,
                       0);
  result_set.allocateStorage();
}

namespace {

using OneRow = std::vector<TargetValue>;

/* This class allows to emulate and evaluate ResultSet and it's reduce function.
 * It creates two ResultSet equivalents, populates them with randomly generated data,
 * merges them into one, and provides access to the data contained in the merged set.
 * Comparing these data with the ones received from the ResultSet code reduce procedure
 * run on the same pair of the ResultSet equivalents, allows to evaluate the functionality
 * of the ResultSet code.
 */
class ResultSetEmulator {
 public:
  ResultSetEmulator(int8_t* buff1,
                    int8_t* buff2,
                    const std::vector<TargetInfo>& target_infos,
                    const QueryMemoryDescriptor& query_mem_desc,
                    NumberGenerator& gen1,
                    NumberGenerator& gen2,
                    const size_t perc1,
                    const size_t perc2,
                    const size_t flow,
                    const bool silent)
      :

      rs1_buff(buff1)
      , rs2_buff(buff2)
      , rs_target_infos(target_infos)
      , rs_query_mem_desc(query_mem_desc)
      , rs1_gen(gen1)
      , rs2_gen(gen2)
      , rs1_perc(perc1)
      , rs2_perc(perc2)
      , rs_flow(flow)
      , rs_entry_count(query_mem_desc.getEntryCount())
      , rs_silent(silent) {
    rs_entry_count =
        query_mem_desc.getEntryCount();  // it's set to 10 in "small" query_mem_descriptor
    rs1_groups.resize(rs_entry_count);
    std::fill(rs1_groups.begin(), rs1_groups.end(), false);
    rs2_groups.resize(rs_entry_count);
    std::fill(rs2_groups.begin(), rs2_groups.end(), false);
    rs1_values.resize(rs_entry_count);
    std::fill(rs1_values.begin(), rs1_values.end(), 0);
    rs2_values.resize(rs_entry_count);
    std::fill(rs2_values.begin(), rs2_values.end(), 0);
    rseReducedGroups.resize(rs_entry_count);
    std::fill(rseReducedGroups.begin(), rseReducedGroups.end(), false);

    emulateResultSets();
  }
  ~ResultSetEmulator(){};

  std::queue<std::vector<int64_t>> getReferenceTable() const { return rseReducedTable; }
  std::vector<bool> getReferenceGroupMap() const { return rseReducedGroups; }
  bool getReferenceGroupMapElement(size_t idx) {
    CHECK_LE(idx, rseReducedGroups.size() - 1);
    return rseReducedGroups[idx];
  }
  std::vector<int64_t> getReferenceRow(bool keep_row = false) {
    std::vector<int64_t> rse_reduced_row(rseReducedTable.front());
    rseReducedTable.pop();
    if (keep_row) {
      rseReducedTable.push(rse_reduced_row);
    }
    return rse_reduced_row;
  }
  int64_t rse_get_null_val() {
    int64_t null_val = 0;
    for (const auto& target_info : rs_target_infos) {
      null_val = inline_int_null_val(target_info.sql_type);
      break;  // currently all of TargetInfo's columns used in tests have same type, so
              // the they all share same null_val, and that's why the first column is used
              // here.
    }
    return null_val;
  }
  void print_rse_generated_result_sets() const;
  void print_merged_result_sets(const std::vector<OneRow>& result);

 private:
  void emulateResultSets();
  void createResultSet(size_t rs_perc, std::vector<bool>& rs_groups);
  void mergeResultSets();
  int8_t *rs1_buff, *rs2_buff;
  const std::vector<TargetInfo> rs_target_infos;
  const QueryMemoryDescriptor rs_query_mem_desc;
  NumberGenerator &rs1_gen, &rs2_gen;
  size_t rs1_perc, rs2_perc, rs_flow;
  size_t rs_entry_count;
  bool rs_silent;
  std::vector<bool> rs1_groups;  // true if group is in ResultSet #1
  std::vector<bool> rs2_groups;  // true if group is in ResultSet #2
  std::vector<bool>
      rseReducedGroups;  // true if group is in either ResultSet #1 or ResultSet #2
  std::vector<int64_t> rs1_values;  // generated values for ResultSet #1
  std::vector<int64_t> rs2_values;  // generated values for ResultSet #2
  std::queue<std::vector<int64_t>>
      rseReducedTable;  // combined/reduced values of ResultSet #1 and ResultSet #2

  void rse_fill_storage_buffer_perfect_hash_colwise(int8_t* buff,
                                                    NumberGenerator& generator,
                                                    const std::vector<bool>& rs_groups,
                                                    std::vector<int64_t>& rs_values);
  void rse_fill_storage_buffer_perfect_hash_rowwise(int8_t* buff,
                                                    NumberGenerator& generator,
                                                    const std::vector<bool>& rs_groups,
                                                    std::vector<int64_t>& rs_values);
  void rse_fill_storage_buffer_baseline_colwise(int8_t* buff,
                                                NumberGenerator& generator,
                                                const std::vector<bool>& rs_groups,
                                                std::vector<int64_t>& rs_values);
  void rse_fill_storage_buffer_baseline_rowwise(int8_t* buff,
                                                NumberGenerator& generator,
                                                const std::vector<bool>& rs_groups,
                                                std::vector<int64_t>& rs_values);
  void rse_fill_storage_buffer(int8_t* buff,
                               NumberGenerator& generator,
                               const std::vector<bool>& rs_groups,
                               std::vector<int64_t>& rs_values);
  void print_emulator_diag();
  int64_t rseAggregateKMIN(size_t i);
  int64_t rseAggregateKMAX(size_t i);
  int64_t rseAggregateKAVG(size_t i);
  int64_t rseAggregateKSUM(size_t i);
  int64_t rseAggregateKCOUNT(size_t i);
};

/* top level module to create and fill up ResultSets as well as to generate golden values
 */
void ResultSetEmulator::emulateResultSets() {
  /* generate topology of ResultSet #1 */
  if (!rs_silent) {
    printf("\nResultSetEmulator (ResultSet #1): ");
  }

  createResultSet(rs1_perc, rs1_groups);
  if (!rs_silent) {
    printf("\n");
    for (size_t i = 0; i < rs1_groups.size(); i++) {
      if (rs1_groups[i]) {
        printf("1");
      } else {
        printf("0");
      }
    }
  }

  /* generate topology of ResultSet #2 */
  if (!rs_silent) {
    printf("\nResultSetEmulator (ResultSet #2): ");
  }

  createResultSet(rs2_perc, rs2_groups);
  if (!rs_silent) {
    printf("\n");
    for (size_t i = 0; i < rs2_groups.size(); i++) {
      if (rs2_groups[i]) {
        printf("1");
      } else {
        printf("0");
      }
    }
    printf("\n");
  }

  /* populate both ResultSet's buffers with real data */
  // print_emulator_diag();
  rse_fill_storage_buffer(rs1_buff, rs1_gen, rs1_groups, rs1_values);
  // print_emulator_diag();
  rse_fill_storage_buffer(rs2_buff, rs2_gen, rs2_groups, rs2_values);
  // print_emulator_diag();

  /* merge/reduce data contained in both ResultSets and generate golden values */
  mergeResultSets();
}

/* generate ResultSet topology (create rs_groups and rs_groups_idx vectors) */
void ResultSetEmulator::createResultSet(size_t rs_perc, std::vector<bool>& rs_groups) {
  std::vector<int> rs_groups_idx;
  rs_groups_idx.resize(rs_entry_count);
  std::iota(rs_groups_idx.begin(), rs_groups_idx.end(), 0);
  std::random_device rs_rd;
  std::mt19937 rs_rand_gen(rs_rd());
  std::shuffle(rs_groups_idx.begin(), rs_groups_idx.end(), rs_rand_gen);

  for (size_t i = 0; i < (rs_entry_count * rs_perc / 100); i++) {
    if (!rs_silent) {
      printf(" %i", rs_groups_idx[i]);
    }
    rs_groups[rs_groups_idx[i]] = true;
  }
}

/* merge/reduce data contained in both ResultSets and generate golden values */
void ResultSetEmulator::mergeResultSets() {
  std::vector<int64_t> rse_reduced_row;
  rse_reduced_row.resize(rs_target_infos.size());

  for (size_t j = 0; j < rs_entry_count; j++) {  // iterates through rows
    if (rs1_groups[j] || rs2_groups[j]) {
      rseReducedGroups[j] = true;
      for (size_t i = 0; i < rs_target_infos.size(); i++) {  // iterates through columns
        switch (rs_target_infos[i].agg_kind) {
          case kMIN: {
            rse_reduced_row[i] = rseAggregateKMIN(j);
            break;
          }
          case kMAX: {
            rse_reduced_row[i] = rseAggregateKMAX(j);
            break;
          }
          case kAVG: {
            rse_reduced_row[i] = rseAggregateKAVG(j);
            break;
          }
          case kSUM: {
            rse_reduced_row[i] = rseAggregateKSUM(j);
            break;
          }
          case kCOUNT: {
            rse_reduced_row[i] = rseAggregateKCOUNT(j);
            break;
          }
          default:
            CHECK(false);
        }
      }
      rseReducedTable.push(rse_reduced_row);
    }
  }
}

void ResultSetEmulator::rse_fill_storage_buffer_perfect_hash_colwise(
    int8_t* buff,
    NumberGenerator& generator,
    const std::vector<bool>& rs_groups,
    std::vector<int64_t>& rs_values) {
  const auto key_component_count = rs_query_mem_desc.getKeyCount();
  CHECK(rs_query_mem_desc.didOutputColumnar());
  // initialize the key buffer(s)
  auto col_ptr = buff;
  for (size_t key_idx = 0; key_idx < key_component_count; ++key_idx) {
    auto key_entry_ptr = col_ptr;
    const auto key_bytes = rs_query_mem_desc.groupColWidth(key_idx);
    CHECK_EQ(8, key_bytes);
    for (size_t i = 0; i < rs_entry_count; i++) {
      const auto v = generator.getNextValue();
      if (rs_groups[i]) {
        // const auto v = generator.getNextValue();
        write_key(v, key_entry_ptr, key_bytes);
      } else {
        write_key(EMPTY_KEY_64, key_entry_ptr, key_bytes);
      }
      key_entry_ptr += key_bytes;
    }
    col_ptr = advance_to_next_columnar_key_buff(col_ptr, rs_query_mem_desc, key_idx);
    generator.reset();
  }
  // initialize the value buffer(s)
  size_t slot_idx = 0;
  for (const auto& target_info : rs_target_infos) {
    auto col_entry_ptr = col_ptr;
    const auto col_bytes = rs_query_mem_desc.getPaddedSlotWidthBytes(slot_idx);
    for (size_t i = 0; i < rs_entry_count; i++) {
      int8_t* ptr2{nullptr};
      if (target_info.agg_kind == kAVG) {
        ptr2 = col_entry_ptr + rs_query_mem_desc.getEntryCount() * col_bytes;
      }
      const auto v = generator.getNextValue();
      if (rs_groups[i]) {
        // const auto v = generator.getNextValue();
        rs_values[i] = v;
        if (rs_flow == 2 &&             // null_val test-cases
            i >= rs_entry_count - 4) {  // only the last four rows of RS #1 and RS #2
                                        // exersized for null_val test
          rs_values[i] = -1;
          fill_one_entry_one_col(col_entry_ptr,
                                 col_bytes,
                                 ptr2,
                                 rs_query_mem_desc.getPaddedSlotWidthBytes(slot_idx + 1),
                                 v,
                                 target_info,
                                 false,
                                 true);
        } else {
          fill_one_entry_one_col(col_entry_ptr,
                                 col_bytes,
                                 ptr2,
                                 rs_query_mem_desc.getPaddedSlotWidthBytes(slot_idx + 1),
                                 v,
                                 target_info,
                                 false,
                                 false);
        }
      } else {
        if (rs_flow == 2) {               // null_val test-cases
          if (i >= rs_entry_count - 4) {  // only the last four rows of RS #1 and RS #2
                                          // exersized for null_val test
            rs_values[i] = -1;
          }
          fill_one_entry_one_col(col_entry_ptr,
                                 col_bytes,
                                 ptr2,
                                 rs_query_mem_desc.getPaddedSlotWidthBytes(slot_idx + 1),
                                 rs_query_mem_desc.hasKeylessHash() ? 0 : 0xdeadbeef,
                                 target_info,
                                 true,
                                 true);
        } else {
          fill_one_entry_one_col(col_entry_ptr,
                                 col_bytes,
                                 ptr2,
                                 rs_query_mem_desc.getPaddedSlotWidthBytes(slot_idx + 1),
                                 rs_query_mem_desc.hasKeylessHash() ? 0 : 0xdeadbeef,
                                 target_info,
                                 true,
                                 false);
        }
      }
      col_entry_ptr += col_bytes;
    }
    col_ptr = advance_to_next_columnar_target_buff(col_ptr, rs_query_mem_desc, slot_idx);
    if (target_info.is_agg && target_info.agg_kind == kAVG) {
      col_ptr =
          advance_to_next_columnar_target_buff(col_ptr, rs_query_mem_desc, slot_idx + 1);
    }
    slot_idx = advance_slot(slot_idx, target_info, false);
    generator.reset();
  }
}

void ResultSetEmulator::rse_fill_storage_buffer_perfect_hash_rowwise(
    int8_t* buff,
    NumberGenerator& generator,
    const std::vector<bool>& rs_groups,
    std::vector<int64_t>& rs_values) {
  const auto key_component_count = rs_query_mem_desc.getKeyCount();
  CHECK(!rs_query_mem_desc.didOutputColumnar());
  auto key_buff = buff;
  CHECK_EQ(rs_groups.size(), rs_query_mem_desc.getEntryCount());
  for (size_t i = 0; i < rs_groups.size(); i++) {
    const auto v = generator.getNextValue();
    if (rs_groups[i]) {
      // const auto v = generator.getNextValue();
      rs_values[i] = v;
      auto key_buff_i64 = reinterpret_cast<int64_t*>(key_buff);
      for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
        *key_buff_i64++ = v;
      }
      auto entries_buff = reinterpret_cast<int8_t*>(key_buff_i64);
      if (rs_flow == 2) {               // null_vall test-cases
        if (i >= rs_entry_count - 4) {  // only the last four rows of RS #1 and RS #2
                                        // exersized for null_val test
          rs_values[i] = -1;
          key_buff = fill_one_entry_no_collisions(
              entries_buff, rs_query_mem_desc, v, rs_target_infos, false, true);
        } else {
          key_buff = fill_one_entry_no_collisions(
              entries_buff, rs_query_mem_desc, v, rs_target_infos, false, false);
        }
      } else {
        key_buff = fill_one_entry_no_collisions(
            entries_buff, rs_query_mem_desc, v, rs_target_infos, false);
      }
    } else {
      auto key_buff_i64 = reinterpret_cast<int64_t*>(key_buff);
      for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
        *key_buff_i64++ = EMPTY_KEY_64;
      }
      auto entries_buff = reinterpret_cast<int8_t*>(key_buff_i64);
      if (rs_flow == 2) {               // null_val test-cases
        if (i >= rs_entry_count - 4) {  // only the last four rows of RS #1 and RS #2
                                        // exersized for null_val test
          rs_values[i] = -1;
        }
        key_buff = fill_one_entry_no_collisions(
            entries_buff, rs_query_mem_desc, 0xdeadbeef, rs_target_infos, true, true);
      } else {
        key_buff = fill_one_entry_no_collisions(
            entries_buff, rs_query_mem_desc, 0xdeadbeef, rs_target_infos, true);
      }
    }
  }
}

void ResultSetEmulator::rse_fill_storage_buffer_baseline_colwise(
    int8_t* buff,
    NumberGenerator& generator,
    const std::vector<bool>& rs_groups,
    std::vector<int64_t>& rs_values) {
  CHECK(rs_query_mem_desc.didOutputColumnar());
  const auto key_component_count = rs_query_mem_desc.getKeyCount();
  const auto i64_buff = reinterpret_cast<int64_t*>(buff);
  for (size_t i = 0; i < rs_entry_count; i++) {
    for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
      i64_buff[key_offset_colwise(i, key_comp_idx, rs_entry_count)] = EMPTY_KEY_64;
    }
    size_t target_slot = 0;
    int64_t init_val = 0;
    for (const auto& target_info : rs_target_infos) {
      if (target_info.agg_kind == kCOUNT) {
        init_val = 0;
      } else if (!target_info.sql_type.get_notnull() && target_info.skip_null_val &&
                 (rs_flow == 2)) {  // null_val support
        init_val = inline_int_null_val(target_info.sql_type);
      } else {
        init_val = 0xdeadbeef;
      }
      if (target_info.agg_kind != kAVG) {
        i64_buff[slot_offset_colwise(
            i, target_slot, key_component_count, rs_entry_count)] = init_val;
      } else {
        i64_buff[slot_offset_colwise(
            i, target_slot, key_component_count, rs_entry_count)] = 0;
      }
      target_slot++;
    }
  }
  for (size_t i = 0; i < rs_entry_count; i++) {
    const auto v = generator.getNextValue();
    if (rs_groups[i]) {
      bool null_val = false;
      if ((rs_flow == 2) &&
          (i >= rs_entry_count - 4)) {  // null_val test-cases: last four rows
        rs_values[i] = -1;
        null_val = true;
      } else {
        rs_values[i] = v;
      }
      std::vector<int64_t> key(key_component_count, v);
      auto value_slots =
          get_group_value_columnar(i64_buff, rs_entry_count, &key[0], key.size());
      CHECK(value_slots);
      for (const auto& target_info : rs_target_infos) {
        fill_one_entry_one_col(
            value_slots, v, target_info, rs_entry_count, false, null_val);
        value_slots += rs_entry_count;
        if (target_info.agg_kind == kAVG) {
          value_slots += rs_entry_count;
        }
      }
    }
  }
}

void ResultSetEmulator::rse_fill_storage_buffer_baseline_rowwise(
    int8_t* buff,
    NumberGenerator& generator,
    const std::vector<bool>& rs_groups,
    std::vector<int64_t>& rs_values) {
  CHECK(!rs_query_mem_desc.didOutputColumnar());
  CHECK_EQ(rs_groups.size(), rs_query_mem_desc.getEntryCount());
  const auto key_component_count = rs_query_mem_desc.getKeyCount();
  const auto i64_buff = reinterpret_cast<int64_t*>(buff);
  const auto target_slot_count = get_slot_count(rs_target_infos);
  for (size_t i = 0; i < rs_entry_count; i++) {
    const auto first_key_comp_offset =
        key_offset_rowwise(i, key_component_count, target_slot_count);
    for (size_t key_comp_idx = 0; key_comp_idx < key_component_count; ++key_comp_idx) {
      i64_buff[first_key_comp_offset + key_comp_idx] = EMPTY_KEY_64;
    }
    size_t target_slot = 0;
    int64_t init_val = 0;
    for (const auto& target_info : rs_target_infos) {
      if (target_info.agg_kind == kCOUNT) {
        init_val = 0;
      } else if (!target_info.sql_type.get_notnull() && target_info.skip_null_val &&
                 (rs_flow == 2)) {  // null_val support
        init_val = inline_int_null_val(target_info.sql_type);
      } else {
        init_val = 0xdeadbeef;
      }
      i64_buff[slot_offset_rowwise(
          i, target_slot, key_component_count, target_slot_count)] = init_val;
      target_slot++;
      if (target_info.agg_kind == kAVG) {
        i64_buff[slot_offset_rowwise(
            i, target_slot, key_component_count, target_slot_count)] = 0;
        target_slot++;
      }
    }
  }

  for (size_t i = 0; i < rs_entry_count; i++) {
    const auto v = generator.getNextValue();
    if (rs_groups[i]) {
      std::vector<int64_t> key(key_component_count, v);
      auto value_slots = get_group_value(i64_buff,
                                         rs_entry_count,
                                         &key[0],
                                         key.size(),
                                         sizeof(int64_t),
                                         key_component_count + target_slot_count);
      CHECK(value_slots);
      if ((rs_flow == 2) &&
          (i >= rs_entry_count - 4)) {  // null_val test-cases: last four rows
        rs_values[i] = -1;
        fill_one_entry_baseline(value_slots, v, rs_target_infos, false, true);
      } else {
        rs_values[i] = v;
        fill_one_entry_baseline(value_slots, v, rs_target_infos, false, false);
      }
    }
  }
}

void ResultSetEmulator::rse_fill_storage_buffer(int8_t* buff,
                                                NumberGenerator& generator,
                                                const std::vector<bool>& rs_groups,
                                                std::vector<int64_t>& rs_values) {
  switch (rs_query_mem_desc.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByPerfectHash: {
      if (rs_query_mem_desc.didOutputColumnar()) {
        rse_fill_storage_buffer_perfect_hash_colwise(
            buff, generator, rs_groups, rs_values);
      } else {
        rse_fill_storage_buffer_perfect_hash_rowwise(
            buff, generator, rs_groups, rs_values);
      }
      break;
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      if (rs_query_mem_desc.didOutputColumnar()) {
        rse_fill_storage_buffer_baseline_colwise(buff, generator, rs_groups, rs_values);
      } else {
        rse_fill_storage_buffer_baseline_rowwise(buff, generator, rs_groups, rs_values);
      }
      break;
    }
    default:
      CHECK(false);
  }
  CHECK(buff);
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

void ResultSetEmulator::print_emulator_diag() {
  if (!rs_silent) {
    for (size_t j = 0; j < rs_entry_count; j++) {
      int g1 = 0, g2 = 0;
      if (rs1_groups[j]) {
        g1 = 1;
      }
      if (rs2_groups[j]) {
        g2 = 1;
      }
      printf("\nGroup #%i (%i,%i): Buf1=%lld Buf2=%lld",
             (int)j,
             g1,
             g2,
             static_cast<long long>(rs1_values[j]),
             static_cast<long long>(rs2_values[j]));
    }
  }
}

#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

void ResultSetEmulator::print_rse_generated_result_sets() const {
  printf("\nResultSet #1 Final Groups: ");
  for (size_t i = 0; i < rs1_groups.size(); i++) {
    if (rs1_groups[i]) {
      printf("1");
    } else {
      printf("0");
    }
  }

  printf("\nResultSet #2 Final Groups: ");
  for (size_t i = 0; i < rs2_groups.size(); i++) {
    if (rs2_groups[i]) {
      printf("1");
    } else {
      printf("0");
    }
  }
  printf("\n");
}

void ResultSetEmulator::print_merged_result_sets(const std::vector<OneRow>& result) {
  printf("\n ****** KMIN_DATA_FROM_RS_MERGE_CODE ****** %i", (int)result.size());
  size_t j = 0;
  for (const auto& row : result) {
    const auto ival_0 = v<int64_t>(row[0]);  // kMIN
    const auto ival_1 = v<int64_t>(row[1]);  // kMAX
    const auto ival_2 = v<int64_t>(row[2]);  // kSUM
    const auto ival_3 = v<int64_t>(row[3]);  // kCOUNT
    const auto ival_4 = v<double>(row[4]);   // kAVG
    printf("\n Group #%i KMIN/KMAX/KSUM/KCOUNT from RS_MergeCode: %lld %lld %lld %lld %f",
           (int)j,
           static_cast<long long>(ival_0),
           static_cast<long long>(ival_1),
           static_cast<long long>(ival_2),
           static_cast<long long>(ival_3),
           ival_4);
    j++;
  }

  size_t active_group_count = 0;
  for (size_t j = 0; j < rseReducedGroups.size(); j++) {
    if (rseReducedGroups[j]) {
      active_group_count++;
    }
  }
  printf("\n\n ****** KMIN_DATA_FROM_MERGE_BUFFER_CODE ****** Total: %i, Active: %i",
         (int)rs_entry_count,
         (int)active_group_count);
  size_t num_groups = getReferenceTable().size();
  for (size_t i = 0; i < num_groups; i++) {
    std::vector<int64_t> ref_row = getReferenceRow(true);
    int64_t ref_val_0 = ref_row[0];  // kMIN
    int64_t ref_val_1 = ref_row[1];  // kMAX
    int64_t ref_val_2 = ref_row[2];  // kSUM
    int64_t ref_val_3 = ref_row[3];  // kCOUNT
    int64_t ref_val_4 = ref_row[4];  // kAVG
    printf(
        "\n Group #%i KMIN/KMAX/KSUM/KCOUNT from ReducedBuffer: %lld %lld %lld %lld %f",
        static_cast<int>(i),
        static_cast<long long>(ref_val_0),
        static_cast<long long>(ref_val_1),
        static_cast<long long>(ref_val_2),
        static_cast<long long>(ref_val_3),
        static_cast<double>(ref_val_4));
  }
  printf("\n");
}

int64_t ResultSetEmulator::rseAggregateKMIN(size_t i) {
  int64_t result = rse_get_null_val();

  if (rs1_groups[i] && rs2_groups[i]) {
    if ((rs1_values[i] == -1) && (rs2_values[i] == -1)) {
      return result;
    } else {
      if ((rs1_values[i] != -1) && (rs2_values[i] != -1)) {
        result = std::min(rs1_values[i], rs2_values[i]);
      } else {
        result = std::max(rs1_values[i], rs2_values[i]);
      }
    }
  } else {
    if (rs1_groups[i]) {
      if (rs1_values[i] != -1) {
        result = rs1_values[i];
      }
    } else {
      if (rs2_groups[i]) {
        if (rs2_values[i] != -1) {
          result = rs2_values[i];
        }
      }
    }
  }

  return result;
}

int64_t ResultSetEmulator::rseAggregateKMAX(size_t i) {
  int64_t result = rse_get_null_val();

  if (rs1_groups[i] && rs2_groups[i]) {
    if ((rs1_values[i] == -1) && (rs2_values[i] == -1)) {
      return result;
    } else {
      result = std::max(rs1_values[i], rs2_values[i]);
    }
  } else {
    if (rs1_groups[i]) {
      if (rs1_values[i] != -1) {
        result = rs1_values[i];
      }
    } else {
      if (rs2_groups[i]) {
        if (rs2_values[i] != -1) {
          result = rs2_values[i];
        }
      }
    }
  }

  return result;
}

int64_t ResultSetEmulator::rseAggregateKAVG(size_t i) {
  double result = 0;
  int n1 = 1,
      n2 = 1;  // for test purposes count of elements in each group is 1 (see proc
               // "fill_one_entry_no_collisions")

  if (rs1_groups[i] && rs2_groups[i]) {
    if ((rs1_values[i] == -1) && (rs2_values[i] == -1)) {
      return shared::reinterpret_bits<int64_t>(NULL_DOUBLE);
    }
    int n = 0;
    if (rs1_values[i] != -1) {
      result += rs1_values[i] / n1;
      n++;
    }
    if (rs2_values[i] != -1) {
      result += rs2_values[i] / n2;
      n++;
    }
    if (n > 1) {
      result /= n;
    }
  } else {
    result = NULL_DOUBLE;
    if (rs1_groups[i]) {
      if (rs1_values[i] != -1) {
        result = rs1_values[i] / n1;
      }
    } else {
      if (rs2_groups[i]) {
        if (rs2_values[i] != -1) {
          result = rs2_values[i] / n2;
        }
      }
    }
  }

  return shared::reinterpret_bits<int64_t>(result);
}

int64_t ResultSetEmulator::rseAggregateKSUM(size_t i) {
  int64_t result = 0;

  if (rs1_groups[i] && rs2_groups[i]) {
    if ((rs1_values[i] == -1) && (rs2_values[i] == -1)) {
      return rse_get_null_val();
    }
    if (rs1_values[i] != -1) {
      result += rs1_values[i];
    }
    if (rs2_values[i] != -1) {
      result += rs2_values[i];
    }
  } else {
    result = rse_get_null_val();
    if (rs1_groups[i]) {
      if (rs1_values[i] != -1) {
        result = rs1_values[i];
      }
    } else {
      if (rs2_groups[i]) {
        if (rs2_values[i] != -1) {
          result = rs2_values[i];
        }
      }
    }
  }

  return result;
}

int64_t ResultSetEmulator::rseAggregateKCOUNT(size_t i) {
  int64_t result = 0;

  if (rs1_groups[i] && rs2_groups[i]) {
    if ((rs1_values[i] == -1) && (rs2_values[i] == -1)) {
      return 0;
    }
    if (rs1_values[i] != -1) {
      result += rs1_values[i];
    }
    if (rs2_values[i] != -1) {
      result += rs2_values[i];
    }
  } else {
    if (rs1_groups[i]) {
      if (rs1_values[i] != -1) {
        result = rs1_values[i];
      }
    } else {
      if (rs2_groups[i]) {
        if (rs2_values[i] != -1) {
          result = rs2_values[i];
        }
      }
    }
  }

  return result;
}

constexpr double EPS = 0.01;

bool approx_eq(const double v, const double target, const double eps = EPS) {
  return target - eps < v && v < target + eps;
}

std::shared_ptr<StringDictionary> g_sd =
    std::make_shared<StringDictionary>(DictRef(), "", false, true);

void test_iterate(const std::vector<TargetInfo>& target_infos,
                  const QueryMemoryDescriptor& query_mem_desc) {
  SQLTypeInfo double_ti(kDOUBLE, false);
  auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      g_data_provider.get(), Executor::getArenaBlockSize());
  StringDictionaryProxy* sdp =
      row_set_mem_owner->addStringDict(g_sd, 1, g_sd->storageEntryCount());
  ResultSet result_set(target_infos,
                       ExecutorDeviceType::CPU,
                       query_mem_desc,
                       row_set_mem_owner,
                       nullptr,
                       nullptr,
                       -1,
                       0,
                       0);
  for (size_t i = 0; i < query_mem_desc.getEntryCount(); ++i) {
    sdp->getOrAddTransient(std::to_string(i));
  }
  const auto storage = result_set.allocateStorage();
  EvenNumberGenerator generator;
  fill_storage_buffer(
      storage->getUnderlyingBuffer(), target_infos, query_mem_desc, generator, 2);
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
        case kTINYINT:
        case kSMALLINT:
        case kINT:
        case kBIGINT: {
          const auto ival = v<int64_t>(row[i]);
          ASSERT_EQ(ref_val, ival);
          break;
        }
        case kDOUBLE: {
          const auto dval = v<double>(row[i]);
          ASSERT_NEAR(ref_val, dval, EPS);
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

std::vector<TargetInfo> generate_random_groups_target_infos() {
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo int_ti(kINT, true);
  SQLTypeInfo double_ti(kDOUBLE, true);
  // SQLTypeInfo null_ti(kNULLT, false);
  target_infos.push_back(TargetInfo{true, kMIN, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kMAX, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kSUM, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kCOUNT, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kAVG, int_ti, double_ti, true, false});
  return target_infos;
}

std::vector<TargetInfo> generate_random_groups_nullable_target_infos() {
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo int_ti(kINT, false);
  // SQLTypeInfo null_ti(kNULLT, false);
  SQLTypeInfo double_ti(kDOUBLE, false);
  target_infos.push_back(TargetInfo{true, kMIN, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kMAX, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kSUM, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kCOUNT, int_ti, int_ti, true, false});
  target_infos.push_back(TargetInfo{true, kAVG, int_ti, double_ti, true, false});
  return target_infos;
}

std::vector<OneRow> get_rows_sorted_by_col(ResultSet& rs, const size_t col_idx) {
  std::list<Analyzer::OrderEntry> order_entries;
  order_entries.emplace_back(1, false, false);
  rs.sort(order_entries, 0, nullptr);
  std::vector<OneRow> result;

  while (true) {
    const auto row = rs.getNextRow(false, false);
    if (row.empty()) {
      break;
    }
    result.push_back(row);
  }
  return result;
}

void run_reduction(const std::vector<TargetInfo>& target_infos,
                   const QueryMemoryDescriptor& query_mem_desc,
                   NumberGenerator& generator1,
                   NumberGenerator& generator2,
                   const int step) {
  const ResultSetStorage* storage1{nullptr};
  const ResultSetStorage* storage2{nullptr};
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      g_data_provider.get(), Executor::getArenaBlockSize());
  row_set_mem_owner->addStringDict(g_sd, 1, g_sd->storageEntryCount());
  const auto rs1 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  storage1 = rs1->allocateStorage();
  fill_storage_buffer(
      storage1->getUnderlyingBuffer(), target_infos, query_mem_desc, generator1, step);
  const auto rs2 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  storage2 = rs2->allocateStorage();
  fill_storage_buffer(
      storage2->getUnderlyingBuffer(), target_infos, query_mem_desc, generator2, step);
  ResultSetManager rs_manager;
  std::vector<ResultSet*> storage_set{rs1.get(), rs2.get()};
  rs_manager.reduce(storage_set, Executor::UNITARY_EXECUTOR_ID);
}

void test_reduce(const std::vector<TargetInfo>& target_infos,
                 const QueryMemoryDescriptor& query_mem_desc,
                 NumberGenerator& generator1,
                 NumberGenerator& generator2,
                 const int step,
                 const bool sort) {
  const ResultSetStorage* storage1{nullptr};
  const ResultSetStorage* storage2{nullptr};
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      g_data_provider.get(), Executor::getArenaBlockSize());
  row_set_mem_owner->addStringDict(g_sd, 1, g_sd->storageEntryCount());
  const auto rs1 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  storage1 = rs1->allocateStorage();
  fill_storage_buffer(
      storage1->getUnderlyingBuffer(), target_infos, query_mem_desc, generator1, step);
  const auto rs2 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  storage2 = rs2->allocateStorage();
  fill_storage_buffer(
      storage2->getUnderlyingBuffer(), target_infos, query_mem_desc, generator2, step);
  ResultSetManager rs_manager;
  std::vector<ResultSet*> storage_set{rs1.get(), rs2.get()};
  auto result_rs = rs_manager.reduce(storage_set, Executor::UNITARY_EXECUTOR_ID);

  if (sort) {
    std::list<Analyzer::OrderEntry> order_entries;
    order_entries.emplace_back(1, false, false);
    result_rs->sort(order_entries, 0, nullptr);
  }
  const size_t thread_count = cpu_threads();
  const auto row_count = result_rs->rowCount();
  std::vector<std::future<void>> reduction_threads;
  for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    const auto thread_row_count = (row_count + thread_count - 1) / thread_count;
    const auto start_index = thread_idx * thread_row_count;
    const auto end_index = std::min(start_index + thread_row_count, row_count);
    reduction_threads.emplace_back(std::async(
        std::launch::async, [start_index, end_index, result_rs, &target_infos, step] {
          SQLTypeInfo double_ti(kDOUBLE, false);

          for (size_t row_idx = start_index; row_idx < end_index; ++row_idx) {
            const auto row = result_rs->getRowAtNoTranslations(row_idx);
            if (row.empty()) {
              continue;
            }
            ASSERT_EQ(target_infos.size(), row.size());

            for (size_t i = 0; i < target_infos.size(); ++i) {
              const auto& target_info = target_infos[i];
              const auto& ti =
                  target_info.agg_kind == kAVG ? double_ti : target_info.sql_type;
              switch (ti.get_type()) {
                case kTINYINT:
                case kSMALLINT:
                case kINT:
                case kBIGINT: {
                  const auto ival = v<int64_t>(row[i]);
                  const int64_t ref =
                      (target_info.agg_kind == kSUM || target_info.agg_kind == kCOUNT)
                          ? step * row_idx
                          : row_idx;
                  ASSERT_EQ(ref, ival);
                  break;
                }
                case kDOUBLE: {
                  const auto dval = v<double>(row[i]);
                  ASSERT_DOUBLE_EQ(static_cast<double>((target_info.agg_kind == kSUM ||
                                                        target_info.agg_kind == kCOUNT)
                                                           ? step * row_idx
                                                           : row_idx),
                                   dval);
                  break;
                }
                case kTEXT:
                  break;
                default:
                  CHECK(false);
              }
            }
          }
        }));
  }
  for (auto& t : reduction_threads) {
    t.get();
  }
}

void test_reduce_random_groups(const std::vector<TargetInfo>& target_infos,
                               const QueryMemoryDescriptor& query_mem_desc,
                               NumberGenerator& generator1,
                               NumberGenerator& generator2,
                               const int prct1,
                               const int prct2,
                               bool silent,
                               const int flow = 0) {
  SQLTypeInfo double_ti(kDOUBLE, false);
  const ResultSetStorage* storage1{nullptr};
  const ResultSetStorage* storage2{nullptr};
  std::unique_ptr<ResultSet> rs1;
  std::unique_ptr<ResultSet> rs2;
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      g_data_provider.get(), Executor::getArenaBlockSize());
  switch (query_mem_desc.getQueryDescriptionType()) {
    case QueryDescriptionType::GroupByPerfectHash: {
      rs1.reset(new ResultSet(target_infos,
                              ExecutorDeviceType::CPU,
                              query_mem_desc,
                              row_set_mem_owner,
                              nullptr,
                              nullptr,
                              -1,
                              0,
                              0));
      storage1 = rs1->allocateStorage();
      rs2.reset(new ResultSet(target_infos,
                              ExecutorDeviceType::CPU,
                              query_mem_desc,
                              row_set_mem_owner,
                              nullptr,
                              nullptr,
                              -1,
                              0,
                              0));
      storage2 = rs2->allocateStorage();
      break;
    }
    case QueryDescriptionType::GroupByBaselineHash: {
      rs1.reset(new ResultSet(target_infos,
                              ExecutorDeviceType::CPU,
                              query_mem_desc,
                              row_set_mem_owner,
                              nullptr,
                              nullptr,
                              -1,
                              0,
                              0));
      storage1 = rs1->allocateStorage();
      rs2.reset(new ResultSet(target_infos,
                              ExecutorDeviceType::CPU,
                              query_mem_desc,
                              row_set_mem_owner,
                              nullptr,
                              nullptr,
                              -1,
                              0,
                              0));
      storage2 = rs2->allocateStorage();
      break;
    }
    default:
      CHECK(false);
  }

  ResultSetEmulator* rse = new ResultSetEmulator(storage1->getUnderlyingBuffer(),
                                                 storage2->getUnderlyingBuffer(),
                                                 target_infos,
                                                 query_mem_desc,
                                                 generator1,
                                                 generator2,
                                                 prct1,
                                                 prct2,
                                                 flow,
                                                 silent);
  if (!silent) {
    rse->print_rse_generated_result_sets();
  }

  ResultSetManager rs_manager;
  std::vector<ResultSet*> storage_set{rs1.get(), rs2.get()};
  auto result_rs = rs_manager.reduce(storage_set, Executor::UNITARY_EXECUTOR_ID);
  std::queue<std::vector<int64_t>> ref_table = rse->getReferenceTable();
  std::vector<bool> ref_group_map = rse->getReferenceGroupMap();
  const auto result = get_rows_sorted_by_col(*result_rs, 0);
  CHECK(!result.empty());

  if (!silent) {
    rse->print_merged_result_sets(result);
  }

  size_t row_idx = 0;
  int64_t ref_val = 0;
  for (const auto& row : result) {
    CHECK_EQ(target_infos.size(), row.size());
    while (true) {
      if (row_idx >= rse->getReferenceGroupMap().size()) {
        LOG(FATAL) << "Number of groups in reduced result set is more than expected";
      }
      if (rse->getReferenceGroupMapElement(row_idx)) {
        break;
      } else {
        row_idx++;
        continue;
      }
    }
    std::vector<int64_t> ref_row = rse->getReferenceRow();
    for (size_t i = 0; i < target_infos.size(); ++i) {
      ref_val = ref_row[i];
      const auto& target_info = target_infos[i];
      const auto& ti = target_info.agg_kind == kAVG ? double_ti : target_info.sql_type;
      std::string p_tag("");
      if (flow == 2) {  // null_val test-cases
        p_tag += "kNULLT_";
      }
      switch (ti.get_type()) {
        case kSMALLINT:
        case kINT:
        case kBIGINT: {
          const auto ival = v<int64_t>(row[i]);
          switch (target_info.agg_kind) {
            case kMIN: {
              if (!silent) {
                p_tag += "KMIN";
                printf("\n%s row_idx = %i, ref_val = %lld, ival = %lld",
                       p_tag.c_str(),
                       static_cast<int>(row_idx),
                       static_cast<long long>(ref_val),
                       static_cast<long long>(ival));
                if (ref_val != ival) {
                  printf("%21s%s%s", "", p_tag.c_str(), " TEST FAILED!\n");
                } else {
                  printf("%21s%s%s", "", p_tag.c_str(), " TEST PASSED!\n");
                }
              } else {
                ASSERT_EQ(ref_val, ival);
              }
              break;
            }
            case kMAX: {
              if (!silent) {
                p_tag += "KMAX";
                printf("\n%s row_idx = %i, ref_val = %lld, ival = %lld",
                       p_tag.c_str(),
                       static_cast<int>(row_idx),
                       static_cast<long long>(ref_val),
                       static_cast<long long>(ival));
                if (ref_val != ival) {
                  printf("%21s%s%s", "", p_tag.c_str(), " TEST FAILED!\n");
                } else {
                  printf("%21s%s%s", "", p_tag.c_str(), " TEST PASSED!\n");
                }
              } else {
                ASSERT_EQ(ref_val, ival);
              }
              break;
            }
            case kAVG: {
              if (!silent) {
                p_tag += "KAVG";
                printf("\n%s row_idx = %i, ref_val = %lld, ival = %lld",
                       p_tag.c_str(),
                       static_cast<int>(row_idx),
                       static_cast<long long>(ref_val),
                       static_cast<long long>(ival));
                if (ref_val != ival) {
                  printf("%21s%s%s", "", p_tag.c_str(), " TEST FAILED1!\n");
                } else {
                  printf("%21s%s%s", "", p_tag.c_str(), " TEST PASSED!\n");
                }
              } else {
                ASSERT_EQ(ref_val, ival);
              }
              break;
            }
            case kSUM:
            case kCOUNT: {
              if (!silent) {
                p_tag += "KSUM";
                printf("\n%s row_idx = %i, ref_val = %lld, ival = %lld",
                       p_tag.c_str(),
                       static_cast<int>(row_idx),
                       static_cast<long long>(ref_val),
                       static_cast<long long>(ival));
                if (ref_val != ival) {
                  printf("%21s%s%s", "", p_tag.c_str(), " TEST FAILED!\n");
                } else {
                  printf("%21s%s%s", "", p_tag.c_str(), " TEST PASSED!\n");
                }
              } else {
                ASSERT_EQ(ref_val, ival);
              }
              break;
            }
            default:
              CHECK(false);
          }
          break;
        }
        case kDOUBLE: {
          const auto dval = v<double>(row[i]);
          switch (target_info.agg_kind) {
            case kMIN: {
              if (!silent) {
                p_tag += "KMIN_D";
                printf("\nKMIN_D row_idx = %i, ref_val = %e, dval = %e",
                       static_cast<int>(row_idx),
                       static_cast<double>(ref_val),
                       dval);
                if (!approx_eq(static_cast<double>(ref_val), dval)) {
                  printf("%5s%s%s", "", p_tag.c_str(), " TEST FAILED!\n");
                } else {
                  printf("%5s%s%s", "", p_tag.c_str(), " TEST PASSED!\n");
                }
              } else {
                ASSERT_NEAR(ref_val, dval, EPS) << p_tag << ' ' << row_idx;
              }
              break;
            }
            case kMAX: {
              if (!silent) {
                p_tag += "KMAX_D";
                printf("\n%s row_idx = %i, ref_val = %e, dval = %e",
                       p_tag.c_str(),
                       static_cast<int>(row_idx),
                       static_cast<double>(ref_val),
                       dval);
                if (!approx_eq(static_cast<double>(ref_val), dval)) {
                  printf("%5s%s%s", "", p_tag.c_str(), " TEST FAILED!\n");
                } else {
                  printf("%5s%s%s", "", p_tag.c_str(), " TEST PASSED!\n");
                }
              } else {
                ASSERT_NEAR(ref_val, dval, EPS) << p_tag << ' ' << row_idx;
              }
              break;
            }
            case kAVG: {
              if (!silent) {
                p_tag += "KAVG_D";
                printf("\n%s row_idx = %i, ref_val = %e, dval = %e",
                       p_tag.c_str(),
                       static_cast<int>(row_idx),
                       shared::reinterpret_bits<double>(ref_val),
                       dval);
                if (!approx_eq(shared::reinterpret_bits<double>(ref_val), dval)) {
                  printf("%5s%s%s", "", p_tag.c_str(), " TEST FAILED!\n");
                } else {
                  printf("%5s%s%s", "", p_tag.c_str(), " TEST PASSED!\n");
                }
              } else {
                ASSERT_NEAR(shared::reinterpret_bits<double>(ref_val), dval, EPS)
                    << p_tag << ' ' << row_idx;
                if (dval == NULL_DOUBLE) {
                  ASSERT_EQ(shared::reinterpret_bits<double>(ref_val), dval)
                      << p_tag << ' ' << row_idx;
                }
              }
              break;
            }
            case kSUM:
            case kCOUNT: {
              if (!silent) {
                p_tag += "KSUM_D";
                printf("\n%s row_idx = %i, ref_val = %e, dval = %e",
                       p_tag.c_str(),
                       (int)row_idx,
                       (double)ref_val,
                       dval);
                if (!approx_eq(static_cast<double>(ref_val), dval)) {
                  printf("%5s%s%s", "", p_tag.c_str(), " TEST FAILED!\n");
                } else {
                  printf("%5s%s%s", "", p_tag.c_str(), " TEST PASSED!\n");
                }
              } else {
                ASSERT_NEAR(ref_val, dval, EPS) << p_tag << ' ' << row_idx;
              }
              break;
            }
            default:
              CHECK(false);
          }
          break;
        }
        default:
          CHECK(false);
      }
    }
    row_idx++;
  }

  delete rse;
}
}  // namespace

TEST(Iterate, PerfectHashOneCol) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneCol32) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4, 0, 99);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnar32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnar16) {
  std::vector<int8_t> group_column_widths{8};
  const int8_t suggested_agg_width = 2;
  const auto target_infos = generate_custom_agg_target_infos(
      group_column_widths,
      {kMAX, kMIN, kCOUNT, kSUM, kAVG},
      {kSMALLINT, kSMALLINT, kINT, kINT, kDOUBLE},
      {kSMALLINT, kSMALLINT, kSMALLINT, kSMALLINT, kSMALLINT});
  auto query_mem_desc = perfect_hash_one_col_desc(
      target_infos, suggested_agg_width, 0, 99, group_column_widths);
  query_mem_desc.setOutputColumnar(true);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnar8) {
  std::vector<int8_t> group_column_widths{8};
  const int8_t suggested_agg_width = 1;
  const auto target_infos = generate_custom_agg_target_infos(
      group_column_widths,
      {kMAX, kMIN, kCOUNT, kSUM, kAVG},
      {kTINYINT, kTINYINT, kINT, kINT, kDOUBLE},
      {kTINYINT, kTINYINT, kTINYINT, kTINYINT, kTINYINT});
  auto query_mem_desc = perfect_hash_one_col_desc(
      target_infos, suggested_agg_width, 0, 99, group_column_widths);
  query_mem_desc.setOutputColumnar(true);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4, 0, 99);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnarKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnarKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnarKeyless16) {
  std::vector<int8_t> group_column_widths{8};
  const int8_t suggested_agg_width = 2;
  const auto target_infos = generate_custom_agg_target_infos(
      group_column_widths,
      {kAVG, kSUM, kMIN, kCOUNT, kMAX},
      {kDOUBLE, kINT, kSMALLINT, kINT, kSMALLINT},
      {kSMALLINT, kSMALLINT, kSMALLINT, kSMALLINT, kSMALLINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashOneColColumnarKeyless8) {
  std::vector<int8_t> group_column_widths{8};
  const int8_t suggested_agg_width = 1;
  const auto target_infos = generate_custom_agg_target_infos(
      group_column_widths,
      {kAVG, kSUM, kMIN, kCOUNT, kMAX},
      {kDOUBLE, kINT, kTINYINT, kINT, kTINYINT},
      {kTINYINT, kTINYINT, kTINYINT, kTINYINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
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
  query_mem_desc.setOutputColumnar(true);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColColumnar32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.setOutputColumnar(true);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColColumnarKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Iterate, PerfectHashTwoColColumnarKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
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
  query_mem_desc.setOutputColumnar(true);
  test_iterate(target_infos, query_mem_desc);
}

TEST(Reduce, PerfectHashOneCol) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneCol32) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4, 0, 99);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColColumnar32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColColumnar16) {
  std::vector<int8_t> group_column_widths{8};
  const int8_t suggested_agg_width = 2;
  const auto target_infos = generate_custom_agg_target_infos(
      group_column_widths,
      {kMAX, kMIN, kCOUNT, kSUM, kAVG},
      {kSMALLINT, kSMALLINT, kINT, kBIGINT, kDOUBLE},
      {kSMALLINT, kSMALLINT, kSMALLINT, kSMALLINT, kSMALLINT});
  auto query_mem_desc = perfect_hash_one_col_desc(
      target_infos, suggested_agg_width, 0, 99, group_column_widths);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColColumnar8) {
  std::vector<int8_t> group_column_widths{8};
  const int8_t suggested_agg_width = 1;
  const auto target_infos = generate_custom_agg_target_infos(
      group_column_widths,
      {kMAX, kMIN, kCOUNT, kSUM, kAVG},
      {kTINYINT, kTINYINT, kINT, kBIGINT, kDOUBLE},
      {kTINYINT, kTINYINT, kTINYINT, kTINYINT, kTINYINT});
  auto query_mem_desc = perfect_hash_one_col_desc(
      target_infos, suggested_agg_width, 0, 99, group_column_widths);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4, 0, 99);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColColumnarKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColColumnarKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 4, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColColumnarKeyless16) {
  std::vector<int8_t> group_column_widths{8};
  const int8_t suggested_agg_width = 2;
  const auto target_infos = generate_custom_agg_target_infos(
      group_column_widths,
      {kAVG, kSUM, kMIN, kCOUNT, kMAX},
      {kDOUBLE, kINT, kSMALLINT, kINT, kSMALLINT},
      {kSMALLINT, kSMALLINT, kSMALLINT, kSMALLINT, kSMALLINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashOneColColumnarKeyless8) {
  std::vector<int8_t> group_column_widths{8};
  const int8_t suggested_agg_width = 1;
  const auto target_infos = generate_custom_agg_target_infos(
      group_column_widths,
      {kAVG, kSUM, kMIN, kCOUNT, kMAX},
      {kDOUBLE, kINT, kTINYINT, kINT, kTINYINT},
      {kTINYINT, kTINYINT, kTINYINT, kTINYINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 99);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashTwoCol) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashTwoCol32) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashTwoColColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashTwoColColumnar32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashTwoColKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashTwoColKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashTwoColColumnarKeyless) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, PerfectHashTwoColColumnarKeyless32) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, 4);
  query_mem_desc.setOutputColumnar(true);
  query_mem_desc.setHasKeylessHash(true);
  query_mem_desc.setTargetIdxForKey(2);
  EvenNumberGenerator generator1;
  EvenNumberGenerator generator2;
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 2, false);
}

TEST(Reduce, BaselineHash) {
  const auto target_infos = generate_test_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc(target_infos, 8);
  EvenNumberGenerator generator1;
  ReverseOddOrEvenNumberGenerator generator2(2 * query_mem_desc.getEntryCount() - 1);
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 1, true);
}

TEST(Reduce, BaselineHashColumnar) {
  const auto target_infos = generate_test_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator generator1;
  ReverseOddOrEvenNumberGenerator generator2(2 * query_mem_desc.getEntryCount() - 1);
  test_reduce(target_infos, query_mem_desc, generator1, generator2, 1, true);
}

#ifndef HAVE_TSAN
// The large buffers tests allocate too much memory to instrument under TSAN
TEST(ReduceLargeBuffers, PerfectHashOne_Overflow32) {
  try {
    const auto target_infos = generate_random_groups_nullable_target_infos();
    auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 222208903, {8});
    EvenNumberGenerator gen1;
    EvenNumberGenerator gen2;
    test_reduce(target_infos, query_mem_desc, gen1, gen2, 2, false);
  } catch (std::bad_alloc e) {
    LOG(WARNING) << "Out-of-memory for ReduceLargeBuffers.PerfectHashOne_Overflow32";
    GTEST_SKIP();
  }
}

TEST(ReduceLargeBuffers, PerfectHashColumnarOne_Overflow32) {
  try {
    const auto target_infos = generate_random_groups_nullable_target_infos();
    auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 222208903, {8});
    query_mem_desc.setOutputColumnar(true);
    EvenNumberGenerator gen1;
    EvenNumberGenerator gen2;
    test_reduce(target_infos, query_mem_desc, gen1, gen2, 2, false);
  } catch (std::bad_alloc e) {
    LOG(WARNING)
        << "Out-of-memory for ReduceLargeBuffers.PerfectHashColumnarOne_Overflow32";
    GTEST_SKIP();
  }
}

TEST(ReduceLargeBuffers, BaselineHash_Overflow32) {
  try {
    const auto target_infos = generate_random_groups_nullable_target_infos();
    auto query_mem_desc = baseline_hash_two_col_desc_overflow32(target_infos, 8);
    EvenNumberGenerator gen1;
    EvenNumberGenerator gen2;
    run_reduction(target_infos, query_mem_desc, gen1, gen2, 2);
  } catch (std::bad_alloc e) {
    LOG(WARNING) << "Out-of-memory for ReduceLargeBuffers.BaselineHash_Overflow32";
    GTEST_SKIP();
  }
}

TEST(ReduceLargeBuffers, BaselineHashColumnar_Overflow32) {
  try {
    const auto target_infos = generate_random_groups_nullable_target_infos();
    auto query_mem_desc = baseline_hash_two_col_desc_overflow32(target_infos, 8);
    query_mem_desc.setOutputColumnar(true);
    EvenNumberGenerator gen1;
    EvenNumberGenerator gen2;
    run_reduction(target_infos, query_mem_desc, gen1, gen2, 2);
  } catch (const std::bad_alloc&) {
    LOG(WARNING)
        << "Out-of-memory for ReduceLargeBuffers.BaselineHashColumnar_Overflow32";
    GTEST_SKIP();
  }
}
#endif

TEST(MoreReduce, MissingValues) {
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo bigint_ti(kBIGINT, false);
  SQLTypeInfo null_ti(kNULLT, false);
  target_infos.push_back(TargetInfo{false, kMIN, bigint_ti, null_ti, true, false});
  target_infos.push_back(TargetInfo{true, kCOUNT, bigint_ti, null_ti, true, false});
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 7, 9);
  query_mem_desc.setHasKeylessHash(false);
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      g_data_provider.get(), Executor::getArenaBlockSize());
  const auto rs1 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  const auto storage1 = rs1->allocateStorage();
  const auto rs2 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  const auto storage2 = rs2->allocateStorage();
  {
    auto buff1 = reinterpret_cast<int64_t*>(storage1->getUnderlyingBuffer());
    buff1[0 * 3] = 7;
    buff1[1 * 3] = EMPTY_KEY_64;
    buff1[2 * 3] = EMPTY_KEY_64;
    buff1[0 * 3 + 1] = 7;
    buff1[1 * 3 + 1] = 0;
    buff1[2 * 3 + 1] = 0;
    buff1[0 * 3 + 2] = 15;
    buff1[1 * 3 + 2] = 0;
    buff1[2 * 3 + 2] = 0;
  }
  {
    auto buff2 = reinterpret_cast<int64_t*>(storage2->getUnderlyingBuffer());
    buff2[0 * 3] = EMPTY_KEY_64;
    buff2[1 * 3] = EMPTY_KEY_64;
    buff2[2 * 3] = 9;
    buff2[0 * 3 + 1] = 0;
    buff2[1 * 3 + 1] = 0;
    buff2[2 * 3 + 1] = 9;
    buff2[0 * 3 + 2] = 0;
    buff2[1 * 3 + 2] = 0;
    buff2[2 * 3 + 2] = 5;
  }
  ResultSetReductionJIT reduction_jit(rs1->getQueryMemDesc(),
                                      rs1->getTargetInfos(),
                                      rs1->getTargetInitVals(),
                                      Executor::UNITARY_EXECUTOR_ID);
  const auto reduction_code = reduction_jit.codegen();
  storage1->reduce(*storage2, {}, reduction_code, Executor::UNITARY_EXECUTOR_ID);
  {
    const auto row = rs1->getNextRow(false, false);
    CHECK_EQ(size_t(2), row.size());
    ASSERT_EQ(7, v<int64_t>(row[0]));
    ASSERT_EQ(15, v<int64_t>(row[1]));
  }
  {
    const auto row = rs1->getNextRow(false, false);
    CHECK_EQ(size_t(2), row.size());
    ASSERT_EQ(9, v<int64_t>(row[0]));
    ASSERT_EQ(5, v<int64_t>(row[1]));
  }
  {
    const auto row = rs1->getNextRow(false, false);
    ASSERT_EQ(size_t(0), row.size());
  }
}

TEST(MoreReduce, MissingValuesKeyless) {
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo bigint_ti(kBIGINT, false);
  SQLTypeInfo null_ti(kNULLT, false);
  target_infos.push_back(TargetInfo{false, kMIN, bigint_ti, null_ti, true, false});
  target_infos.push_back(TargetInfo{true, kCOUNT, bigint_ti, null_ti, true, false});
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 7, 9);
  query_mem_desc.setHasKeylessHash(true);
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      g_data_provider.get(), Executor::getArenaBlockSize());
  const auto rs1 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  const auto storage1 = rs1->allocateStorage();
  const auto rs2 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  const auto storage2 = rs2->allocateStorage();
  {
    auto buff1 = reinterpret_cast<int64_t*>(storage1->getUnderlyingBuffer());
    buff1[0 * 2] = 7;
    buff1[1 * 2] = 0;
    buff1[2 * 2] = 0;
    buff1[0 * 2 + 1] = 15;
    buff1[1 * 2 + 1] = 0;
    buff1[2 * 2 + 1] = 0;
  }
  {
    auto buff2 = reinterpret_cast<int64_t*>(storage2->getUnderlyingBuffer());
    buff2[0 * 2] = 0;
    buff2[1 * 2] = 0;
    buff2[2 * 2] = 9;
    buff2[0 * 2 + 1] = 0;
    buff2[1 * 2 + 1] = 0;
    buff2[2 * 2 + 1] = 5;
  }
  ResultSetReductionJIT reduction_jit(rs1->getQueryMemDesc(),
                                      rs1->getTargetInfos(),
                                      rs1->getTargetInitVals(),
                                      Executor::UNITARY_EXECUTOR_ID);
  const auto reduction_code = reduction_jit.codegen();
  storage1->reduce(*storage2, {}, reduction_code, Executor::UNITARY_EXECUTOR_ID);
  {
    const auto row = rs1->getNextRow(false, false);
    CHECK_EQ(size_t(2), row.size());
    ASSERT_EQ(7, v<int64_t>(row[0]));
    ASSERT_EQ(15, v<int64_t>(row[1]));
  }
  {
    const auto row = rs1->getNextRow(false, false);
    CHECK_EQ(size_t(2), row.size());
    ASSERT_EQ(9, v<int64_t>(row[0]));
    ASSERT_EQ(5, v<int64_t>(row[1]));
  }
  {
    const auto row = rs1->getNextRow(false, false);
    ASSERT_EQ(size_t(0), row.size());
  }
}

TEST(MoreReduce, OffsetRewrite) {
  std::vector<TargetInfo> target_infos;
  SQLTypeInfo bigint_ti(kBIGINT, false);
  SQLTypeInfo real_str_ti(kTEXT, true, kENCODING_NONE);
  SQLTypeInfo null_ti(kNULLT, false);

  target_infos.push_back(TargetInfo{false, kMIN, bigint_ti, null_ti, true, false});
  target_infos.push_back(TargetInfo{true, kSAMPLE, real_str_ti, null_ti, true, false});
  auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 7, 9);
  query_mem_desc.setHasKeylessHash(false);
  const auto row_set_mem_owner = std::make_shared<RowSetMemoryOwner>(
      g_data_provider.get(), Executor::getArenaBlockSize());
  const auto rs1 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  const auto storage1 = rs1->allocateStorage();
  const auto rs2 = std::make_unique<ResultSet>(target_infos,
                                               ExecutorDeviceType::CPU,
                                               query_mem_desc,
                                               row_set_mem_owner,
                                               nullptr,
                                               nullptr,
                                               -1,
                                               0,
                                               0);
  const auto storage2 = rs2->allocateStorage();
  std::vector<std::string> serialized_varlen_buffer{"foo", "bar", "hello"};

  {
    auto buff1 = reinterpret_cast<int64_t*>(storage1->getUnderlyingBuffer());
    buff1[0 * 4] = 7;
    buff1[1 * 4] = 8;
    buff1[2 * 4] = 9;
    buff1[0 * 4 + 1] = 7;
    buff1[1 * 4 + 1] = 8;
    buff1[2 * 4 + 1] = 9;
    buff1[0 * 4 + 2] = 0;
    buff1[1 * 4 + 2] = 0;
    buff1[2 * 4 + 2] = 1;
    buff1[0 * 4 + 3] = 0;
    buff1[1 * 4 + 3] = 0;
    buff1[2 * 4 + 3] = 0;
  }
  {
    auto buff2 = reinterpret_cast<int64_t*>(storage2->getUnderlyingBuffer());
    buff2[0 * 4] = 7;
    buff2[1 * 4] = 8;
    buff2[2 * 4] = 9;
    buff2[0 * 4 + 1] = 7;
    buff2[1 * 4 + 1] = 8;
    buff2[2 * 4 + 1] = 9;
    buff2[0 * 4 + 2] = 0;
    buff2[1 * 4 + 2] = 2;
    buff2[2 * 4 + 2] = 1;
    buff2[0 * 4 + 3] = 0;
    buff2[1 * 4 + 3] = 0;
    buff2[2 * 4 + 3] = 0;
  }

  storage1->rewriteAggregateBufferOffsets(serialized_varlen_buffer);
  ResultSetReductionJIT reduction_jit(rs1->getQueryMemDesc(),
                                      rs1->getTargetInfos(),
                                      rs1->getTargetInitVals(),
                                      Executor::UNITARY_EXECUTOR_ID);
  const auto reduction_code = reduction_jit.codegen();
  storage1->reduce(
      *storage2, serialized_varlen_buffer, reduction_code, Executor::UNITARY_EXECUTOR_ID);
  rs1->setSeparateVarlenStorageValid(true);
  {
    const auto row = rs1->getNextRow(false, false);
    CHECK_EQ(size_t(2), row.size());
    ASSERT_EQ(7, v<int64_t>(row[0]));
    ASSERT_EQ("foo", boost::get<std::string>(v<NullableString>(row[1])));
  }
  {
    const auto row = rs1->getNextRow(false, false);
    CHECK_EQ(size_t(2), row.size());
    ASSERT_EQ(8, v<int64_t>(row[0]));
    ASSERT_EQ("hello", boost::get<std::string>(v<NullableString>(row[1])));
  }
  {
    const auto row = rs1->getNextRow(false, false);
    CHECK_EQ(size_t(2), row.size());
    ASSERT_EQ(9, v<int64_t>(row[0]));
    ASSERT_EQ("bar", boost::get<std::string>(v<NullableString>(row[1])));
  }
  {
    const auto row = rs1->getNextRow(false, false);
    ASSERT_EQ(size_t(0), row.size());
  }
}

/* FLOW #1: Perfect_Hash_Row_Based testcases */
TEST(ReduceRandomGroups, PerfectHashOneCol_Small_2525) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  // const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_2575) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  // const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_5050) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  // const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 50, prct2 = 50;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_7525) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  // const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 75, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_25100) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  // const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_10025) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  // const auto query_mem_desc = perfect_hash_one_col_desc(target_infos, 8, 0, 99);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_9505) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 95, prct2 = 5;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_100100) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_2500) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 0;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_Small_0075) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 0, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

/* FLOW #2: Non_Perfect_Hash_Row_Based testcases */
TEST(ReduceRandomGroups, BaselineHash_Large_5050) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 50, prct2 = 50;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHash_Large_7525) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 75, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHash_Large_2575) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHash_Large_1020) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 10, prct2 = 20;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHash_Large_100100) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHash_Large_2500) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 0;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHash_Large_0075) {
  const auto target_infos = generate_random_groups_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 0, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

/*  FLOW #3: Perfect_Hash_Column_Based testcases */
TEST(ReduceRandomGroups, PerfectHashOneColColumnar_Small_5050) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 50, prct2 = 50;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_Small_25100) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_Small_10025) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_Small_100100) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_Small_2500) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 0;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_Small_0075) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 0, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

/* FLOW #4: Non_Perfect_Hash_Column_Based testcases */
TEST(ReduceRandomGroups, BaselineHashColumnar_Large_5050) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 50, prct2 = 50;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_25100) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_10025) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_100100) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_2500) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 0;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_0075) {
  const auto target_infos = generate_random_groups_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 0, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent);
}

/* FLOW #5: Perfect_Hash_Row_Based_NullVal testcases */
TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_2525) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 25;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_2575) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 75;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_5050) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 50, prct2 = 50;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_7525) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 75, prct2 = 25;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_25100) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 100;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_10025) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 25;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_100100) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 100;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_2500) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 0;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneCol_NullVal_0075) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 0, prct2 = 75;
  ASSERT_LE(prct1, 100);
  ASSERT_LE(prct2, 100);
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

/* FLOW #6: Perfect_Hash_Column_Based_NullVal testcases */
TEST(ReduceRandomGroups, PerfectHashOneColColumnar_NullVal_5050) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 50, prct2 = 50;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_NullVal_25100) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_NullVal_10025) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_NullVal_100100) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_NullVal_2500) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 0;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, PerfectHashOneColColumnar_NullVal_0075) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = perfect_hash_one_col_desc_small(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 0, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

/* FLOW #7: Non_Perfect_Hash_Row_Based_NullVal testcases */
TEST(ReduceRandomGroups, BaselineHash_Large_NullVal_5050) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 50, prct2 = 50;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHash_Large_NullVal_7525) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 75, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHash_Large_NullVal_2575) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHash_Large_NullVal_1020) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 10, prct2 = 20;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHash_Large_NullVal_100100) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHash_Large_NullVal_2500) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 0;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHash_Large_NullVal_0075) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  const auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 0, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

/* FLOW #8: Non_Perfect_Hash_Column_Based_NullVal testcases */
TEST(ReduceRandomGroups, BaselineHashColumnar_Large_NullVal_5050) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 50, prct2 = 50;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_NullVal_25100) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_NullVal_10025) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 25;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_NullVal_100100) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 100, prct2 = 100;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_NullVal_2500) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 25, prct2 = 0;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ReduceRandomGroups, BaselineHashColumnar_Large_NullVal_0075) {
  const auto target_infos = generate_random_groups_nullable_target_infos();
  auto query_mem_desc = baseline_hash_two_col_desc_large(target_infos, 8);
  query_mem_desc.setOutputColumnar(true);
  EvenNumberGenerator gen1;
  EvenNumberGenerator gen2;
  const int prct1 = 0, prct2 = 75;
  bool silent = false;  // true/false - don't/do print diagnostic messages
  silent = true;
  test_reduce_random_groups(
      target_infos, query_mem_desc, gen1, gen2, prct1, prct2, silent, 2);
}

TEST(ResultsetConversion, EnforceParallelColumnarConversion) {
  // if we try to columnarize intermediate result which 1) is not truncated and
  // has more than 20000 rows, i.e., rows.entryCount() >= 20000, then
  // we trigger parallel columnarize conversion for SELECT query
  // so, the purpose of this test is to check
  // whether the large columnar conversion is done correctly

  // load 50M rows - single-frag
  createTable("t_large",
              {{"x", SQLTypeInfo(kBIGINT, true)},
               {"y", SQLTypeInfo(kBIGINT, true)},
               {"z", SQLTypeInfo(kBIGINT, true)}},
              {100000000});
  getStorage()->appendParquetFile(
      "../../Tests/Import/datafiles/interrupt_table_very_large.parquet", "t_large");

  // load 50M rows - two frags (use default frag size)
  createTable("t_large_multi_frag",
              {{"x", SQLTypeInfo(kBIGINT, true)},
               {"y", SQLTypeInfo(kBIGINT, true)},
               {"z", SQLTypeInfo(kBIGINT, true)}},
              {32000000});
  getStorage()->appendParquetFile(
      "../../Tests/Import/datafiles/interrupt_table_very_large.parquet",
      "t_large_multi_frag");

  createTable("t_small",
              {{"x", SQLTypeInfo(kBIGINT, true)},
               {"y", SQLTypeInfo(kBIGINT, true)},
               {"z", SQLTypeInfo(kBIGINT, true)}});
  insertCsvValues("t_small", "1,1,1");

  int64_t answer = 9999999;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // single-frag test
    std::shared_ptr<ResultSet> res1 = run_multiple_agg(
        "SELECT COUNT(1) FROM (SELECT x FROM t_large WHERE x < 2) t, t_small r where t.x "
        "= r.x",
        dt,
        false);
    EXPECT_EQ(1, (int64_t)res1.get()->rowCount());
    const auto crt_row1 = res1.get()->getNextRow(false, false);
    EXPECT_EQ(answer, v<int64_t>(crt_row1[0]));

    // multi-frag test
    std::shared_ptr<ResultSet> res2 = run_multiple_agg(
        "SELECT COUNT(1) FROM (SELECT x FROM t_large_multi_frag WHERE x < 2) t, t_small "
        "r where t.x = r.x",
        dt,
        false);
    EXPECT_EQ(1, (int64_t)res2.get()->rowCount());
    const auto crt_row2 = res2.get()->getNextRow(false, false);
    EXPECT_EQ(answer, v<int64_t>(crt_row1[0]));
  }
}

TEST(Util, ReinterpretBits) {
  uint64_t const u64 = 0x0123456789abcdef;
  uint32_t const u32 = 0x89abcdef;
  uint16_t const u16 = 0xcdef;
  uint8_t const u8 = 0xef;
  // downcast
  EXPECT_EQ(u64, shared::reinterpret_bits<uint64_t>(u64));
  EXPECT_EQ(u32, shared::reinterpret_bits<uint32_t>(u64));
  EXPECT_EQ(u16, shared::reinterpret_bits<uint16_t>(u64));
  EXPECT_EQ(u8, shared::reinterpret_bits<uint8_t>(u64));
  // upcast
  EXPECT_EQ(static_cast<uint64_t>(u8), shared::reinterpret_bits<uint64_t>(u8));
  EXPECT_EQ(static_cast<uint64_t>(u16), shared::reinterpret_bits<uint64_t>(u16));
  EXPECT_EQ(static_cast<uint64_t>(u32), shared::reinterpret_bits<uint64_t>(u32));
  EXPECT_EQ(static_cast<uint64_t>(u64), shared::reinterpret_bits<uint64_t>(u64));
  // floats
  EXPECT_EQ(int64_t(1) << 23, (shared::reinterpret_bits<int64_t, float>(FLT_MIN)));
  EXPECT_EQ(int32_t(1) << 23, (shared::reinterpret_bits<int32_t, float>(FLT_MIN)));
  EXPECT_EQ(int64_t(1) << 52, (shared::reinterpret_bits<int64_t, double>(DBL_MIN)));
  EXPECT_EQ(FLT_MIN, shared::reinterpret_bits<float>(int64_t(1) << 23));
  EXPECT_EQ(FLT_MIN, shared::reinterpret_bits<float>(int32_t(1) << 23));
  EXPECT_EQ(DBL_MIN, shared::reinterpret_bits<double>(int64_t(1) << 52));
}

TEST(Util, PairToDouble) {
  const int64_t null_float = shared::reinterpret_bits<int64_t, float>(NULL_FLOAT);
  const int64_t null_double = shared::reinterpret_bits<int64_t, double>(NULL_DOUBLE);
  EXPECT_EQ(int64_t(1) << 23, null_float);
  EXPECT_EQ(int64_t(1) << 52, null_double);
  // Test all 8 combinations of (kFloat,kDouble)x(bool)x(bool).
  // If the denominator is 0, then the return value should always be NULL_DOUBLE.
  EXPECT_EQ(NULL_DOUBLE, pair_to_double({-4, 0}, SQLTypeInfo(kFLOAT, false), false));
  EXPECT_EQ(NULL_DOUBLE, pair_to_double({-3, 0}, SQLTypeInfo(kFLOAT, false), true));
  EXPECT_EQ(NULL_DOUBLE, pair_to_double({-2, 0}, SQLTypeInfo(kFLOAT, true), false));
  EXPECT_EQ(NULL_DOUBLE, pair_to_double({-1, 0}, SQLTypeInfo(kFLOAT, true), true));
  EXPECT_EQ(NULL_DOUBLE, pair_to_double({0, 0}, SQLTypeInfo(kDOUBLE, false), false));
  EXPECT_EQ(NULL_DOUBLE, pair_to_double({1, 0}, SQLTypeInfo(kDOUBLE, false), true));
  EXPECT_EQ(NULL_DOUBLE, pair_to_double({2, 0}, SQLTypeInfo(kDOUBLE, true), false));
  EXPECT_EQ(NULL_DOUBLE, pair_to_double({3, 0}, SQLTypeInfo(kDOUBLE, true), true));
  // Test all 16 combinations of (null_float,null_double)x(kFloat,kDouble)x(bool)x(bool).
  // All should be non-null.
  EXPECT_EQ(shared::reinterpret_bits<double>(null_float) / 2,
            pair_to_double({null_float, 2}, SQLTypeInfo(kFLOAT, false), false));
  EXPECT_EQ(shared::reinterpret_bits<float>(null_float) / 2,
            pair_to_double({null_float, 2}, SQLTypeInfo(kFLOAT, false), true));
  EXPECT_EQ(shared::reinterpret_bits<double>(null_float) / 2,
            pair_to_double({null_float, 2}, SQLTypeInfo(kFLOAT, true), false));
  EXPECT_EQ(shared::reinterpret_bits<float>(null_float) / 2,
            pair_to_double({null_float, 2}, SQLTypeInfo(kFLOAT, true), true));
  EXPECT_EQ(shared::reinterpret_bits<double>(null_float) / 2,
            pair_to_double({null_float, 2}, SQLTypeInfo(kDOUBLE, false), false));
  EXPECT_EQ(shared::reinterpret_bits<double>(null_float) / 2,
            pair_to_double({null_float, 2}, SQLTypeInfo(kDOUBLE, false), true));
  EXPECT_EQ(shared::reinterpret_bits<double>(null_float) / 2,
            pair_to_double({null_float, 2}, SQLTypeInfo(kDOUBLE, true), false));
  EXPECT_EQ(shared::reinterpret_bits<double>(null_float) / 2,
            pair_to_double({null_float, 2}, SQLTypeInfo(kDOUBLE, true), true));
  EXPECT_EQ(NULL_DOUBLE / 2,
            pair_to_double({null_double, 2}, SQLTypeInfo(kFLOAT, false), false));
  EXPECT_EQ(0.0 / 2, pair_to_double({null_double, 2}, SQLTypeInfo(kFLOAT, false), true));
  EXPECT_EQ(NULL_DOUBLE / 2,
            pair_to_double({null_double, 2}, SQLTypeInfo(kFLOAT, true), false));
  EXPECT_EQ(0.0 / 2, pair_to_double({null_double, 2}, SQLTypeInfo(kFLOAT, true), true));
  EXPECT_EQ(NULL_DOUBLE / 2,
            pair_to_double({null_double, 2}, SQLTypeInfo(kDOUBLE, false), false));
  EXPECT_EQ(NULL_DOUBLE / 2,
            pair_to_double({null_double, 2}, SQLTypeInfo(kDOUBLE, false), true));
  EXPECT_EQ(NULL_DOUBLE / 2,
            pair_to_double({null_double, 2}, SQLTypeInfo(kDOUBLE, true), false));
  EXPECT_EQ(NULL_DOUBLE / 2,
            pair_to_double({null_double, 2}, SQLTypeInfo(kDOUBLE, true), true));
  // Misc
  EXPECT_EQ(0.5,
            pair_to_double({shared::reinterpret_bits<int64_t, float>(1.0), 2},
                           SQLTypeInfo(kFLOAT, false),
                           true));
  EXPECT_EQ(-0.5,
            pair_to_double({shared::reinterpret_bits<int64_t, double>(-1.0), 2},
                           SQLTypeInfo(kDOUBLE, false),
                           true));
  EXPECT_EQ(2.5, pair_to_double({1000, 4}, SQLTypeInfo(kDECIMAL, 19, 2), true));
}

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  init();

  g_data_provider = std::make_shared<DataMgrDataProvider>(getDataMgr());

  // instantiate a single executor
  Executor::getExecutor(
      Executor::UNITARY_EXECUTOR_ID, getDataMgr(), getDataMgr()->getBufferProvider());

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  reset();
  return err;
}
