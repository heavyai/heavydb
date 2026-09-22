/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Logger/Logger.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/ColumnarResults.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ResultSet.h"
#include "QueryEngine/TargetValue.h"
#include "Shared/TargetInfo.h"
#include "Tests/DataMgrTestHelpers.h"
#include "Tests/ResultSetTestUtils.h"
#include "Tests/TestHelpers.h"

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

extern bool g_is_test_env;

class ColumnarResultsTester : public ColumnarResults {
 public:
  ColumnarResultsTester(const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                        const ResultSet& rows,
                        const size_t num_columns,
                        const std::vector<SQLTypeInfo>& target_types,
                        const bool is_parallel_execution_enforced = false)
      : ColumnarResults(row_set_mem_owner,
                        rows,
                        num_columns,
                        target_types,
                        Executor::UNITARY_EXECUTOR_ID,
                        0,
                        is_parallel_execution_enforced) {}

  template <typename ENTRY_TYPE>
  ENTRY_TYPE getEntryAt(const size_t row_idx, const size_t column_idx) const {
    CHECK_LT(column_idx, column_buffers_.size());
    CHECK_LT(row_idx, num_rows_);
    return reinterpret_cast<ENTRY_TYPE*>(column_buffers_[column_idx])[row_idx];
  }
};

template <>
float ColumnarResultsTester::getEntryAt<float>(const size_t row_idx,
                                               const size_t column_idx) const {
  CHECK_LT(column_idx, column_buffers_.size());
  CHECK_LT(row_idx, num_rows_);
  return reinterpret_cast<float*>(column_buffers_[column_idx])[row_idx];
}

template <>
double ColumnarResultsTester::getEntryAt<double>(const size_t row_idx,
                                                 const size_t column_idx) const {
  CHECK_LT(column_idx, column_buffers_.size());
  CHECK_LT(row_idx, num_rows_);
  return reinterpret_cast<double*>(column_buffers_[column_idx])[row_idx];
}

void test_columnar_conversion(const std::vector<TargetInfo>& target_infos,
                              const QueryMemoryDescriptor& query_mem_desc,
                              const size_t non_empty_step_size,
                              const bool is_parallel_conversion = false) {
  auto row_set_mem_owner =
      std::make_shared<RowSetMemoryOwner>(Executor::getArenaBlockSize(), 0);
  ResultSet result_set(
      target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, 0, 0);

  // fill the storage
  const auto storage = result_set.allocateStorage();
  EvenNumberGenerator generator;
  fill_storage_buffer(storage->getUnderlyingBuffer(),
                      target_infos,
                      query_mem_desc,
                      generator,
                      non_empty_step_size);

  // Columnar Conversion:
  std::vector<SQLTypeInfo> col_types;
  for (size_t i = 0; i < result_set.colCount(); ++i) {
    col_types.push_back(get_logical_type_info(result_set.getColType(i)));
  }
  ColumnarResultsTester columnar_results(
      row_set_mem_owner, result_set, col_types.size(), col_types, is_parallel_conversion);
  ASSERT_EQ(columnar_results.size(), result_set.rowCount());

  // Validate the results:
  for (size_t rs_row_idx = 0, cr_row_idx = 0; rs_row_idx < query_mem_desc.getEntryCount();
       rs_row_idx++) {
    if (result_set.isRowAtEmpty(rs_row_idx)) {
      // empty entries should be filtered out for conversion:
      continue;
    }
    const auto row = result_set.getRowAt(rs_row_idx);
    if (row.empty()) {
      break;
    }
    CHECK_EQ(target_infos.size(), row.size());
    for (size_t target_idx = 0; target_idx < target_infos.size(); ++target_idx) {
      const auto& target_info = target_infos[target_idx];
      const auto& ti = target_info.agg_kind == kAVG ? SQLTypeInfo{kDOUBLE, false}
                                                    : target_info.sql_type;
      switch (ti.get_type()) {
        case kBIGINT: {
          const auto ival_result_set = v<int64_t>(row[target_idx]);
          const auto ival_converted = static_cast<int64_t>(
              columnar_results.getEntryAt<int64_t>(cr_row_idx, target_idx));
          ASSERT_EQ(ival_converted, ival_result_set);
          break;
        }
        case kINT: {
          const auto ival_result_set = v<int64_t>(row[target_idx]);
          const auto ival_converted = static_cast<int64_t>(
              columnar_results.getEntryAt<int32_t>(cr_row_idx, target_idx));
          ASSERT_EQ(ival_converted, ival_result_set);
          break;
        }
        case kSMALLINT: {
          const auto ival_result_set = v<int64_t>(row[target_idx]);
          const auto ival_converted = static_cast<int64_t>(
              columnar_results.getEntryAt<int16_t>(cr_row_idx, target_idx));
          ASSERT_EQ(ival_result_set, ival_converted);
          break;
        }
        case kTINYINT: {
          const auto ival_result_set = v<int64_t>(row[target_idx]);
          const auto ival_converted = static_cast<int64_t>(
              columnar_results.getEntryAt<int8_t>(cr_row_idx, target_idx));
          ASSERT_EQ(ival_converted, ival_result_set);
          break;
        }
        case kFLOAT: {
          const auto fval_result_set = v<float>(row[target_idx]);
          const auto fval_converted =
              columnar_results.getEntryAt<float>(cr_row_idx, target_idx);
          ASSERT_FLOAT_EQ(fval_result_set, fval_converted);
          break;
        }
        case kDOUBLE: {
          const auto dval_result_set = v<double>(row[target_idx]);
          const auto dval_converted =
              columnar_results.getEntryAt<double>(cr_row_idx, target_idx);
          ASSERT_FLOAT_EQ(dval_result_set, dval_converted);
          break;
        }
        default:
          UNREACHABLE() << "Invalid type info encountered.";
      }
    }
    cr_row_idx++;
  }
}

TEST(Construct, Empty) {
  std::vector<TargetInfo> target_infos;
  std::vector<SQLTypeInfo> sql_type_infos;
  QueryMemoryDescriptor query_mem_desc;
  auto row_set_mem_owner =
      std::make_shared<RowSetMemoryOwner>(Executor::getArenaBlockSize(), 0);
  ResultSet result_set(
      target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, 0, 0);
  ColumnarResultsTester columnar_results(
      row_set_mem_owner, result_set, sql_type_infos.size(), sql_type_infos);
}

TEST(ResultSetColumnCache, MergesColumnDemandAndOwnsResultSet) {
  const SQLTypeInfo int_ti(kINT, false);
  const SQLTypeInfo null_ti(kNULLT, false);
  const std::vector<TargetInfo> target_infos{
      {false, kMIN, int_ti, null_ti, false, false},
      {false, kMIN, int_ti, null_ti, false, false},
      {false, kMIN, int_ti, null_ti, false, false}};
  const QueryMemoryDescriptor query_mem_desc;
  auto row_set_mem_owner =
      std::make_shared<RowSetMemoryOwner>(Executor::getArenaBlockSize(), 0);
  auto result_set = std::make_shared<ResultSet>(
      target_infos, ExecutorDeviceType::CPU, query_mem_desc, row_set_mem_owner, 0, 0);
  std::weak_ptr<ResultSet> result_set_lifetime = result_set;
  ResultSetColumnCache cache;

  EXPECT_FALSE(cache.getColumnSelection(result_set.get()).has_value());
  cache.mergeColumnSelection(result_set, {2, 0, 2});
  ASSERT_TRUE(cache.getColumnSelection(result_set.get()).has_value());
  EXPECT_EQ(*cache.getColumnSelection(result_set.get()), (std::vector<size_t>{0, 2}));

  cache.mergeColumnSelection(result_set, {1});
  EXPECT_FALSE(cache.getColumnSelection(result_set.get()).has_value());
  cache.mergeColumnSelection(result_set, {0});
  EXPECT_FALSE(cache.getColumnSelection(result_set.get()).has_value());

  result_set.reset();
  EXPECT_FALSE(result_set_lifetime.expired());
  cache.clear();
  EXPECT_TRUE(result_set_lifetime.expired());
}

TEST(ResultSetColumnCache, CreatesEachEntryOnceAcrossConcurrentReaders) {
  const QueryMemoryDescriptor query_mem_desc;
  auto row_set_mem_owner =
      std::make_shared<RowSetMemoryOwner>(Executor::getArenaBlockSize(), 0);
  auto result_set = std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                ExecutorDeviceType::CPU,
                                                query_mem_desc,
                                                row_set_mem_owner,
                                                0,
                                                0);
  ResultSetColumnCache cache;
  const ResultSetColumnCache::Key key{
      result_set.get(), 0, Data_Namespace::CPU_LEVEL, 0, -1, true, false, false};
  auto buffer_owner = std::make_shared<int8_t>(42);
  std::atomic<size_t> create_count{0};
  std::atomic<size_t> ready_count{0};
  std::atomic<bool> start{false};
  constexpr size_t thread_count = 16;
  std::vector<const int8_t*> buffers(thread_count);
  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
    threads.emplace_back([&, thread_idx] {
      ready_count.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      buffers[thread_idx] = cache.getOrCreate(key, result_set, buffer_owner, [&] {
        create_count.fetch_add(1, std::memory_order_relaxed);
        return buffer_owner.get();
      });
    });
  }
  while (ready_count.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(create_count.load(), size_t(1));
  EXPECT_TRUE(std::all_of(buffers.begin(), buffers.end(), [&](const auto buffer) {
    return buffer == buffer_owner.get();
  }));
}

// Projections:
// TODO(Saman): add tests for Projections

// Perfect Hash:
TEST(PerfectHashRowWise, OneCol_64Key_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kMIN, kMAX, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_64Key_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kAVG, kMAX, kAVG, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kDOUBLE, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_32Key_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kMIN, kMAX, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 3, 17, 33, 117}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_32Key_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kAVG, kCOUNT, kAVG, kMAX, kMIN},
      {kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kDOUBLE, kBIGINT, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 3, 17, 33, 117}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_64Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE},
      {kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_64Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kAVG, kMAX, kMAX, kAVG, kMAX},
      {kTINYINT, kSMALLINT, kINT, kDOUBLE, kBIGINT, kFLOAT, kDOUBLE, kDOUBLE},
      {kTINYINT, kSMALLINT, kINT, kSMALLINT, kBIGINT, kFLOAT, kINT, kDOUBLE});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_32Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 17, 33, 117}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, OneCol_32Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kAVG, kMAX, kMAX, kMAX},
      {kDOUBLE, kDOUBLE, kFLOAT, kBIGINT, kDOUBLE, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kDOUBLE, kFLOAT, kBIGINT, kFLOAT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 17, 33, 117}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_64Key_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kAVG, kMAX, kAVG, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kDOUBLE, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_64Key_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos =
      generate_custom_agg_target_infos(key_column_widths,
                                       {kSUM, kSUM, kCOUNT, kMAX, kMIN},
                                       {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT},
                                       {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_64Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kAVG, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kDOUBLE, kBIGINT, kINT, kDOUBLE, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kINT, kBIGINT, kINT, kSMALLINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_64Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_32Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kAVG, kAVG, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kDOUBLE, kDOUBLE, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kTINYINT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_32Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_16Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{2};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kMAX, kAVG, kMAX},
      {kDOUBLE, kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kDOUBLE, kTINYINT},
      {kDOUBLE, kINT, kFLOAT, kBIGINT, kINT, kSMALLINT, kBIGINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_16Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{2};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_8Key_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{1};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kAVG, kMAX, kAVG, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kINT, kSMALLINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_8Key_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{1};
  const int8_t suggested_agg_width = 1;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT},
      {kDOUBLE, kFLOAT, kBIGINT, kINT, kSMALLINT, kTINYINT});
  auto query_mem_desc =
      perfect_hash_one_col_desc(target_infos, suggested_agg_width, 0, 118);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {3, 7, 16, 37, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

// Multi-column perfect hash:
TEST(PerfectHashRowWise, TwoCol_64_64_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kMIN, kMAX, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, TwoCol_64_64_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kAVG, kMAX, kAVG, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kDOUBLE, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, TwoCol_64_64_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE},
      {kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashRowWise, TwoCol_64_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kAVG, kMAX, kMAX, kAVG, kMAX},
      {kTINYINT, kSMALLINT, kINT, kDOUBLE, kBIGINT, kFLOAT, kDOUBLE, kDOUBLE},
      {kTINYINT, kSMALLINT, kINT, kSMALLINT, kBIGINT, kFLOAT, kINT, kDOUBLE});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, OneCol_TwoCol_64_64_64Agg_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kMIN, kMAX, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kBIGINT});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, TwoCol_64_64_64Agg_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kSUM, kSUM, kCOUNT, kAVG, kMAX, kAVG, kMIN},
      {kBIGINT, kBIGINT, kBIGINT, kDOUBLE, kBIGINT, kDOUBLE, kBIGINT},
      {kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT, kBIGINT});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {1, 2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, TwoCol_64_64_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE},
      {kTINYINT, kSMALLINT, kINT, kBIGINT, kFLOAT, kDOUBLE});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(PerfectHashColumnar, TwoCol_64_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kAVG, kMAX, kMAX, kAVG, kMAX},
      {kTINYINT, kSMALLINT, kINT, kDOUBLE, kBIGINT, kFLOAT, kDOUBLE, kDOUBLE},
      {kTINYINT, kSMALLINT, kINT, kSMALLINT, kBIGINT, kFLOAT, kINT, kDOUBLE});
  auto query_mem_desc = perfect_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 13, 67, 127}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

// Baseline Hash:
TEST(BaselineHashRowWise, TwoCol_64_64_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_64_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kFLOAT, kDOUBLE, kBIGINT, kTINYINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE},
      {kFLOAT, kTINYINT, kBIGINT, kTINYINT, kINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_64_32_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_32_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kFLOAT, kDOUBLE, kBIGINT, kTINYINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE},
      {kFLOAT, kTINYINT, kBIGINT, kTINYINT, kINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_32_32_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashRowWise, TwoCol_32_32_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kFLOAT, kDOUBLE, kBIGINT, kTINYINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE},
      {kFLOAT, kTINYINT, kBIGINT, kTINYINT, kINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_64_64_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kFLOAT, kDOUBLE, kBIGINT, kTINYINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE},
      {kFLOAT, kTINYINT, kBIGINT, kTINYINT, kINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_64_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{8, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kFLOAT, kDOUBLE, kBIGINT, kTINYINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE},
      {kFLOAT, kTINYINT, kBIGINT, kTINYINT, kINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_64_32_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{8, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_32_64_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4, 8};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kFLOAT, kDOUBLE, kBIGINT, kTINYINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE},
      {kFLOAT, kTINYINT, kBIGINT, kTINYINT, kINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_32_32_MixedAggs_wo_avg) {
  std::vector<int8_t> key_column_widths{4, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kMAX, kMAX, kMAX, kMAX, kMAX},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE},
      {kFLOAT, kBIGINT, kTINYINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

TEST(BaselineHashColumnar, TwoCol_32_32_MixedAggs_w_avg) {
  std::vector<int8_t> key_column_widths{4, 4};
  const int8_t suggested_agg_width = 8;
  std::vector<TargetInfo> target_infos = generate_custom_agg_target_infos(
      key_column_widths,
      {kMAX, kAVG, kMAX, kMAX, kMAX, kAVG, kMAX, kMAX},
      {kFLOAT, kDOUBLE, kBIGINT, kTINYINT, kINT, kDOUBLE, kSMALLINT, kDOUBLE},
      {kFLOAT, kTINYINT, kBIGINT, kTINYINT, kINT, kINT, kSMALLINT, kDOUBLE});
  auto query_mem_desc = baseline_hash_two_col_desc(target_infos, suggested_agg_width);
  query_mem_desc.setAllTargetGroupbyIndices({0, 1, -1, -1, -1, -1, -1, -1, -1, -1});
  query_mem_desc.setOutputColumnar(true);
  for (auto is_parallel : {false, true}) {
    for (auto step_size : {2, 3, 5, 13, 67}) {
      test_columnar_conversion(target_infos, query_mem_desc, step_size, is_parallel);
    }
  }
}

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  TestHelpers::init_sys_catalog();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
