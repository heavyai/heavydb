/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TestHelpers.h"

#include <array>
#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>

#include "../QueryEngine/Execute.h"
#include "../QueryEngine/InputMetadata.h"
#include "../QueryEngine/RelAlgExecutor.h"
#include "../QueryRunner/QueryRunner.h"
#include "Shared/scope.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_is_test_env;
extern bool g_enable_watchdog;
extern size_t g_big_group_threshold;
extern size_t kMaxBufferSize;
extern size_t kMaxNumElemsForBucketizedRange;
extern size_t kWatchdogBaselineNumMaxGroups;
extern bool g_trust_unenforced_table_constraints;
extern bool g_enable_result_reduction_pipeline;
extern bool g_enable_cpu_sub_tasks;
extern size_t g_cpu_sub_task_size;

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

inline void run_ddl_statement(const std::string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
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
class HighCardinalityStringEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS high_cardinality_str;");
    run_ddl_statement(
        "CREATE TABLE high_cardinality_str (x INT, str TEXT ENCODING DICT (32));");
    QR::get()->runSQL("INSERT INTO high_cardinality_str VALUES (1, 'hi');",
                      ExecutorDeviceType::CPU);
    QR::get()->runSQL("INSERT INTO high_cardinality_str VALUES (2, 'bye');",
                      ExecutorDeviceType::CPU);
  }

  void TearDown() override {
    run_ddl_statement("DROP TABLE IF EXISTS high_cardinality_str;");
  }
};

TEST_F(HighCardinalityStringEnv, PerfectHashNoFallback) {
  // make our own executor with a custom col ranges cache
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, "", "", SystemParameters());
  auto cat = QR::get()->getCatalog().get();
  CHECK(cat);

  auto td = cat->getMetadataForTable("high_cardinality_str");
  CHECK(td);
  auto cd = cat->getMetadataForColumn(td->tableId, "str");
  CHECK(cd);
  auto filter_cd = cat->getMetadataForColumn(td->tableId, "x");
  CHECK(filter_cd);

  auto db_id = cat->getDatabaseId();
  PhysicalInput group_phys_input{cd->columnId, td->tableId, db_id};
  PhysicalInput filter_phys_input{filter_cd->columnId, td->tableId, db_id};

  std::unordered_set<PhysicalInput> phys_inputs{group_phys_input, filter_phys_input};
  std::unordered_set<shared::TableKey> phys_table_ids;
  phys_table_ids.insert({cat->getDatabaseId(), group_phys_input.table_id});
  executor->setupCaching(phys_inputs, phys_table_ids);

  auto input_descs = std::vector<InputDescriptor>{InputDescriptor(db_id, td->tableId, 0)};
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(cd->columnId, td->tableId, db_id, 0));
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(filter_cd->columnId, td->tableId, db_id, 0));

  std::vector<InputTableInfo> table_infos = get_table_infos(input_descs, executor.get());

  auto count_expr = makeExpr<Analyzer::AggExpr>(
      SQLTypeInfo(kBIGINT, false), kCOUNT, nullptr, false, nullptr);
  auto group_expr = makeExpr<Analyzer::ColumnVar>(
      cd->columnType, shared::ColumnKey{db_id, td->tableId, cd->columnId}, 0);
  auto filter_col_expr = makeExpr<Analyzer::ColumnVar>(
      filter_cd->columnType,
      shared::ColumnKey{db_id, td->tableId, filter_cd->columnId},
      0);
  Datum d{int64_t(1)};
  auto filter_val_expr = makeExpr<Analyzer::Constant>(SQLTypeInfo(kINT, false), false, d);
  auto simple_filter_expr = makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                                        false,
                                                        SQLOps::kEQ,
                                                        SQLQualifier::kONE,
                                                        filter_col_expr,
                                                        filter_val_expr);
  RelAlgExecutionUnit ra_exe_unit{input_descs,
                                  input_col_descs,
                                  {simple_filter_expr},
                                  {},
                                  {},
                                  {group_expr},
                                  {count_expr.get()},
                                  nullptr,
                                  SortInfo(),
                                  0};
  executor->mockDeviceIdSelectionLogicToOnlyUseSingleDevice();

  ColumnCacheMap column_cache;
  size_t max_groups_buffer_entry_guess = 1;

  auto result =
      executor->executeWorkUnit(max_groups_buffer_entry_guess,
                                /*is_agg=*/true,
                                table_infos,
                                ra_exe_unit,
                                CompilationOptions::defaults(ExecutorDeviceType::CPU),
                                ExecutionOptions::defaults(),
                                nullptr,
                                /*has_cardinality_estimation=*/false,
                                column_cache);
  EXPECT_TRUE(result);
  EXPECT_EQ(result->rowCount(), size_t(1));
  auto row = result->getNextRow(false, false);
  EXPECT_EQ(row.size(), size_t(1));
  EXPECT_EQ(v<int64_t>(row[0]), 1);
}

std::unordered_set<PhysicalInput> setup_str_col_caching(PhysicalInput& group_phys_input,
                                                        const int64_t min,
                                                        const int64_t max,
                                                        PhysicalInput& filter_phys_input,
                                                        Executor* executor) {
  std::unordered_set<PhysicalInput> phys_inputs{group_phys_input, filter_phys_input};
  std::unordered_set<shared::TableKey> phys_table_ids;
  auto db_id = QR::get()->getCatalog()->getDatabaseId();
  phys_table_ids.insert({db_id, group_phys_input.table_id});
  executor->setupCaching(phys_inputs, phys_table_ids);
  auto filter_col_range = executor->getColRange(filter_phys_input);
  // reset the col range to trigger the optimization
  AggregatedColRange col_range_cache;
  col_range_cache.setColRange(group_phys_input,
                              ExpressionRange::makeIntRange(min, max, 0, false));
  col_range_cache.setColRange(filter_phys_input, filter_col_range);
  executor->setColRangeCache(col_range_cache);
  return phys_inputs;
}

TEST_F(HighCardinalityStringEnv, BaselineFallbackTest) {
  // make our own executor with a custom col ranges cache
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, "", "", SystemParameters());
  auto cat = QR::get()->getCatalog().get();
  CHECK(cat);

  auto td = cat->getMetadataForTable("high_cardinality_str");
  CHECK(td);
  auto cd = cat->getMetadataForColumn(td->tableId, "str");
  CHECK(cd);
  auto filter_cd = cat->getMetadataForColumn(td->tableId, "x");
  CHECK(filter_cd);

  auto db_id = cat->getDatabaseId();
  PhysicalInput group_phys_input{cd->columnId, td->tableId, db_id};
  PhysicalInput filter_phys_input{filter_cd->columnId, td->tableId, db_id};

  // 134217728 is 1 additional value over the max buffer size
  auto phys_inputs = setup_str_col_caching(
      group_phys_input, /*min=*/0, /*max=*/134217728, filter_phys_input, executor.get());

  auto input_descs = std::vector<InputDescriptor>{InputDescriptor(db_id, td->tableId, 0)};
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(cd->columnId, td->tableId, db_id, 0));
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(filter_cd->columnId, td->tableId, db_id, 0));

  std::vector<InputTableInfo> table_infos = get_table_infos(input_descs, executor.get());

  auto count_expr = makeExpr<Analyzer::AggExpr>(
      SQLTypeInfo(kBIGINT, false), kCOUNT, nullptr, false, nullptr);
  auto group_expr = makeExpr<Analyzer::ColumnVar>(
      cd->columnType, shared::ColumnKey{db_id, td->tableId, cd->columnId}, 0);
  auto filter_col_expr = makeExpr<Analyzer::ColumnVar>(
      filter_cd->columnType,
      shared::ColumnKey{db_id, td->tableId, filter_cd->columnId},
      0);
  Datum d{int64_t(1)};
  auto filter_val_expr = makeExpr<Analyzer::Constant>(SQLTypeInfo(kINT, false), false, d);
  auto simple_filter_expr = makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                                        false,
                                                        SQLOps::kEQ,
                                                        SQLQualifier::kONE,
                                                        filter_col_expr,
                                                        filter_val_expr);
  RelAlgExecutionUnit ra_exe_unit{input_descs,
                                  input_col_descs,
                                  {simple_filter_expr},
                                  {},
                                  {},
                                  {group_expr},
                                  {count_expr.get()},
                                  nullptr,
                                  SortInfo(),
                                  0};
  executor->mockDeviceIdSelectionLogicToOnlyUseSingleDevice();

  ColumnCacheMap column_cache;
  size_t max_groups_buffer_entry_guess = 1;
  // expect throw w/out cardinality estimation
  EXPECT_THROW(
      executor->executeWorkUnit(max_groups_buffer_entry_guess,
                                /*is_agg=*/true,
                                table_infos,
                                ra_exe_unit,
                                CompilationOptions::defaults(ExecutorDeviceType::CPU),
                                ExecutionOptions::defaults(),
                                nullptr,
                                /*has_cardinality_estimation=*/false,
                                column_cache),
      CardinalityEstimationRequired);

  auto result =
      executor->executeWorkUnit(max_groups_buffer_entry_guess,
                                /*is_agg=*/true,
                                table_infos,
                                ra_exe_unit,
                                CompilationOptions::defaults(ExecutorDeviceType::CPU),
                                ExecutionOptions::defaults(),
                                nullptr,
                                /*has_cardinality_estimation=*/true,
                                column_cache);
  EXPECT_TRUE(result);
  EXPECT_EQ(result->rowCount(), size_t(1));
  auto row = result->getNextRow(false, false);
  EXPECT_EQ(row.size(), size_t(1));
  EXPECT_EQ(v<int64_t>(row[0]), 1);
}

TEST_F(HighCardinalityStringEnv, BaselineNoFilters) {
  // Exercise the same one-entry-over-limit boundary without making sanitizers scan
  // the production-sized perfect-hash buffer. The limit is restored before this test
  // returns, including when execution throws.
  const auto original_max_buffer_size = kMaxBufferSize;
  ScopeGuard reset_max_buffer_size = [original_max_buffer_size] {
    kMaxBufferSize = original_max_buffer_size;
  };
  kMaxBufferSize = 1 << 20;

  // make our own executor with a custom col ranges cache
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, "", "", SystemParameters());
  auto cat = QR::get()->getCatalog().get();
  CHECK(cat);

  auto td = cat->getMetadataForTable("high_cardinality_str");
  CHECK(td);
  auto cd = cat->getMetadataForColumn(td->tableId, "str");
  CHECK(cd);
  auto filter_cd = cat->getMetadataForColumn(td->tableId, "x");
  CHECK(filter_cd);

  auto db_id = cat->getDatabaseId();
  PhysicalInput group_phys_input{cd->columnId, td->tableId, db_id};
  PhysicalInput filter_phys_input{filter_cd->columnId, td->tableId, db_id};

  // The inclusive range contains one more entry than the configured perfect-hash limit.
  const auto max_perfect_hash_entry_count =
      static_cast<int64_t>(kMaxBufferSize / (2 * sizeof(int64_t)));
  auto phys_inputs = setup_str_col_caching(group_phys_input,
                                           /*min=*/0,
                                           /*max=*/max_perfect_hash_entry_count,
                                           filter_phys_input,
                                           executor.get());

  auto input_descs = std::vector<InputDescriptor>{InputDescriptor(db_id, td->tableId, 0)};
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(cd->columnId, td->tableId, db_id, 0));
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(filter_cd->columnId, td->tableId, db_id, 0));

  std::vector<InputTableInfo> table_infos = get_table_infos(input_descs, executor.get());

  auto count_expr = makeExpr<Analyzer::AggExpr>(
      SQLTypeInfo(kBIGINT, false), kCOUNT, nullptr, false, nullptr);
  auto group_expr = makeExpr<Analyzer::ColumnVar>(
      cd->columnType, shared::ColumnKey{db_id, td->tableId, cd->columnId}, 0);

  RelAlgExecutionUnit ra_exe_unit{input_descs,
                                  input_col_descs,
                                  {},
                                  {},
                                  {},
                                  {group_expr},
                                  {count_expr.get()},
                                  nullptr,
                                  SortInfo(),
                                  0};
  executor->mockDeviceIdSelectionLogicToOnlyUseSingleDevice();

  ColumnCacheMap column_cache;
  size_t max_groups_buffer_entry_guess = 1;
  auto execution_options = ExecutionOptions::defaults();
  // This test covers the unfiltered baseline-hash cardinality path. Disable the
  // independent watchdog admission limit so it cannot preempt that invariant.
  execution_options.with_watchdog = false;
  // no filters, so expect no throw w/out cardinality estimation
  auto result =
      executor->executeWorkUnit(max_groups_buffer_entry_guess,
                                /*is_agg=*/true,
                                table_infos,
                                ra_exe_unit,
                                CompilationOptions::defaults(ExecutorDeviceType::CPU),
                                execution_options,
                                nullptr,
                                /*has_cardinality_estimation=*/false,
                                column_cache);
  EXPECT_TRUE(result);
  EXPECT_EQ(result->rowCount(), size_t(2));
  {
    auto row = result->getNextRow(false, false);
    EXPECT_EQ(row.size(), size_t(1));
    EXPECT_EQ(v<int64_t>(row[0]), 1);
  }
  {
    auto row = result->getNextRow(false, false);
    EXPECT_EQ(row.size(), size_t(1));
    EXPECT_EQ(v<int64_t>(row[0]), 1);
  }
}

class LowCardinalityThresholdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS low_cardinality;");
    run_ddl_statement("CREATE TABLE low_cardinality (fl text,ar text, dep text);");

    // write some data to a file
    boost::filesystem::path filename =
        boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();

    filename.replace_extension(boost::filesystem::path{".csv"});

    std::fstream f(filename.native(), std::ios::binary | std::ios::out | std::ios::trunc);

    CHECK(f.is_open());
    for (size_t i = 0; i < g_big_group_threshold; i++) {
      f << i << ", " << i + 1 << ", " << i + 2 << std::endl;
    }
    f.close();

    // Note the path::string() method  returns the appropriate type on win and linux
    run_ddl_statement("COPY low_cardinality FROM '" + filename.string() +
                      "' WITH (header='false');");
  }

  void TearDown() override { run_ddl_statement("DROP TABLE IF EXISTS low_cardinality;"); }
};

TEST_F(LowCardinalityThresholdTest, GroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto result = QR::get()->runSQL(
        R"(select fl,ar,dep from low_cardinality group by fl,ar,dep;)", dt);
    EXPECT_EQ(result->rowCount(), g_big_group_threshold);
  }
}

TEST_F(LowCardinalityThresholdTest, ResourceAwareWatchdogAdmitsCpuGroupBy) {
  const auto original_watchdog_state = g_enable_watchdog;
  const auto original_pipeline_state = g_enable_result_reduction_pipeline;
  const auto original_group_limit = kWatchdogBaselineNumMaxGroups;
  ScopeGuard restore_globals = [=] {
    g_enable_watchdog = original_watchdog_state;
    g_enable_result_reduction_pipeline = original_pipeline_state;
    kWatchdogBaselineNumMaxGroups = original_group_limit;
  };

  g_enable_watchdog = true;
  g_enable_result_reduction_pipeline = true;
  kWatchdogBaselineNumMaxGroups = 0;

  auto result = QR::get()->runSQL(
      R"(SELECT fl, ar, dep FROM low_cardinality GROUP BY fl, ar, dep;)",
      ExecutorDeviceType::CPU);
  EXPECT_EQ(result->rowCount(), g_big_group_threshold);
}

TEST(CpuSubtaskGroupBy, ReusesArenaSlotContextsAcrossQueries) {
  const auto original_pipeline_state = g_enable_result_reduction_pipeline;
  const auto original_subtask_state = g_enable_cpu_sub_tasks;
  const auto original_subtask_size = g_cpu_sub_task_size;
  ScopeGuard restore_globals = [=] {
    g_enable_result_reduction_pipeline = original_pipeline_state;
    g_enable_cpu_sub_tasks = original_subtask_state;
    g_cpu_sub_task_size = original_subtask_size;
  };

  constexpr auto table_name = "cpu_subtask_groupby_contexts";
  run_ddl_statement("DROP TABLE IF EXISTS cpu_subtask_groupby_contexts;");
  ScopeGuard drop_table = [] {
    run_ddl_statement("DROP TABLE IF EXISTS cpu_subtask_groupby_contexts;");
  };
  run_ddl_statement(
      "CREATE TABLE cpu_subtask_groupby_contexts "
      "(slot_key INTEGER, bucket INTEGER, metric_value BIGINT) "
      "WITH (FRAGMENT_SIZE=64);");

  std::ostringstream insert;
  insert << "INSERT INTO " << table_name << " VALUES ";
  constexpr size_t row_count = 256;
  for (size_t row_idx = 0; row_idx < row_count; ++row_idx) {
    if (row_idx) {
      insert << ',';
    }
    const auto slot_key = row_idx + 1 == row_count ? size_t(500000) : row_idx;
    insert << '(' << slot_key << ',' << row_idx % 4 << ',' << row_idx + 1 << ')';
  }
  insert << ';';
  QR::get()->runSQL(insert.str(), ExecutorDeviceType::CPU);

  g_enable_result_reduction_pipeline = true;
  g_enable_cpu_sub_tasks = true;
  g_cpu_sub_task_size = 1;

  const auto wide_result = QR::get()->runSQL(
      "SELECT slot_key, SUM(metric_value) FROM cpu_subtask_groupby_contexts "
      "GROUP BY slot_key;",
      ExecutorDeviceType::CPU);
  ASSERT_NE(wide_result, nullptr);
  ASSERT_EQ(row_count, wide_result->rowCount());

  const std::array<int64_t, 4> expected_sums{8128, 8192, 8256, 8320};
  for (size_t iteration = 0; iteration < 3; ++iteration) {
    const auto result = QR::get()->runSQL(
        "SELECT bucket, SUM(metric_value), COUNT(*) "
        "FROM cpu_subtask_groupby_contexts "
        "GROUP BY bucket ORDER BY bucket;",
        ExecutorDeviceType::CPU);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(expected_sums.size(), result->rowCount());
    for (size_t bucket = 0; bucket < expected_sums.size(); ++bucket) {
      const auto row = result->getNextRow(false, false);
      ASSERT_EQ(size_t(3), row.size());
      EXPECT_EQ(static_cast<int64_t>(bucket), v<int64_t>(row[0]));
      EXPECT_EQ(expected_sums[bucket], v<int64_t>(row[1]));
      EXPECT_EQ(int64_t(64), v<int64_t>(row[2]));
    }
  }
}

class BigCardinalityThresholdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_enable_watchdog = true;
    auto const max_num_elem_for_bucketized_range = g_big_group_threshold + 1;

    run_ddl_statement("DROP TABLE IF EXISTS big_cardinality;");
    run_ddl_statement("CREATE TABLE big_cardinality (fl text,ar text, dep text);");

    // write some data to a file
    boost::filesystem::path filename =
        boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();

    filename.replace_extension(boost::filesystem::path{".csv"});

    std::fstream f(filename.native(), std::ios::binary | std::ios::out | std::ios::trunc);
    CHECK(f.is_open()) << filename.string();

    // add enough groups to trigger the watchdog exception if we use a poor estimate
    for (size_t i = 0; i < max_num_elem_for_bucketized_range; i++) {
      f << i << ", " << i + 1 << ", " << i + 2 << std::endl;
    }
    f.close();

    run_ddl_statement("COPY big_cardinality FROM '" + filename.string() +
                      "' WITH (header='false');");
  }

  void TearDown() override {
    g_enable_watchdog = false;
    run_ddl_statement("DROP TABLE IF EXISTS big_cardinality;");
  }

  size_t initial_g_watchdog_baseline_max_groups{0};
};

TEST_F(BigCardinalityThresholdTest, EmptyFilters) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto result = QR::get()->runSQL(
        R"(SELECT fl,ar,dep FROM big_cardinality WHERE fl = 'a' GROUP BY fl,ar,dep;)",
        dt);
    EXPECT_EQ(result->rowCount(), size_t(0));
  }
}

namespace {

RelAlgExecutionUnit makeSyntheticGroupByUnit(const int32_t db_id,
                                             const int32_t table_id) {
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  groupby_exprs.emplace_back();
  groupby_exprs.emplace_back();
  return RelAlgExecutionUnit{{InputDescriptor(db_id, table_id, 0)},
                             {},
                             {},
                             {},
                             {},
                             groupby_exprs,
                             {},
                             nullptr,
                             SortInfo(),
                             0};
}

RelAlgExecutionUnit makeSyntheticJoinGroupByUnit(const int32_t outer_db_id,
                                                 const int32_t outer_table_id,
                                                 const int32_t inner_db_id,
                                                 const int32_t inner_table_id,
                                                 const bool group_by_outer) {
  const auto group_db_id = group_by_outer ? outer_db_id : inner_db_id;
  const auto group_table_id = group_by_outer ? outer_table_id : inner_table_id;
  const auto rte_idx = group_by_outer ? 0 : 1;
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  groupby_exprs.emplace_back(
      makeExpr<Analyzer::ColumnVar>(SQLTypeInfo(kBIGINT, false),
                                    shared::ColumnKey{group_db_id, group_table_id, 1},
                                    rte_idx));
  return RelAlgExecutionUnit{{InputDescriptor(outer_db_id, outer_table_id, 0),
                              InputDescriptor(inner_db_id, inner_table_id, 1)},
                             {},
                             {},
                             {},
                             {},
                             groupby_exprs,
                             {},
                             nullptr,
                             SortInfo(),
                             0};
}

TEST(SlabLimitedSpeculativeGroupByBound, CapsOnlyOversizedProvenBounds) {
  auto executor = QR::get()->getExecutor().get();
  ASSERT_NE(executor, nullptr);
  if (executor->maxGpuSlabSize() == 0) {
    GTEST_SKIP() << "GPU slab sizing is unavailable";
  }

  const auto exe_unit = makeSyntheticGroupByUnit(/*db_id=*/1, /*table_id=*/1);
  EXPECT_EQ(std::nullopt,
            slab_limited_speculative_groupby_entry_guess_for_test(
                exe_unit, executor, /*proven_upper_bound=*/1));

  const auto oversized_upper_bound = executor->maxGpuSlabSize();
  const auto speculative_bound = slab_limited_speculative_groupby_entry_guess_for_test(
      exe_unit, executor, oversized_upper_bound);
  ASSERT_TRUE(speculative_bound.has_value());
  EXPECT_GT(*speculative_bound, size_t(0));
  EXPECT_LT(*speculative_bound, oversized_upper_bound);
  EXPECT_EQ(std::nullopt,
            slab_limited_speculative_groupby_entry_guess_for_test(
                exe_unit, executor, *speculative_bound));
}

RelAlgExecutionUnit makeSyntheticNonAmplifyingJoinGroupByUnit(
    const shared::TableKey& outer_table_key,
    const int32_t outer_join_column_id,
    const shared::TableKey& inner_table_key,
    const int32_t inner_join_column_id,
    const int32_t inner_to_temporary_join_column_id,
    const int32_t inner_group_column_id,
    const shared::TableKey& temporary_table_key,
    const RelAlgNode* temporary_source) {
  const auto bigint_type = SQLTypeInfo(kBIGINT, false);
  auto outer_join_column = makeExpr<Analyzer::ColumnVar>(
      bigint_type,
      shared::ColumnKey{
          outer_table_key.db_id, outer_table_key.table_id, outer_join_column_id},
      0);
  auto inner_join_column = makeExpr<Analyzer::ColumnVar>(
      bigint_type,
      shared::ColumnKey{
          inner_table_key.db_id, inner_table_key.table_id, inner_join_column_id},
      1);
  auto temporary_join_column = makeExpr<Analyzer::ColumnVar>(
      bigint_type,
      shared::ColumnKey{temporary_table_key.db_id, temporary_table_key.table_id, 0},
      2);
  auto inner_to_temporary_join_column =
      makeExpr<Analyzer::ColumnVar>(bigint_type,
                                    shared::ColumnKey{inner_table_key.db_id,
                                                      inner_table_key.table_id,
                                                      inner_to_temporary_join_column_id},
                                    1);
  JoinQualsPerNestingLevel join_quals{
      JoinCondition{{makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                                 false,
                                                 kEQ,
                                                 kONE,
                                                 outer_join_column,
                                                 inner_join_column)},
                    JoinType::INNER},
      JoinCondition{{makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                                 false,
                                                 kEQ,
                                                 kONE,
                                                 inner_to_temporary_join_column,
                                                 temporary_join_column)},
                    JoinType::INNER}};

  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  groupby_exprs.emplace_back(outer_join_column);
  groupby_exprs.emplace_back(makeExpr<Analyzer::ColumnVar>(
      bigint_type,
      shared::ColumnKey{
          inner_table_key.db_id, inner_table_key.table_id, inner_group_column_id},
      1));
  RelAlgExecutionUnit exe_unit{
      {InputDescriptor(outer_table_key.db_id, outer_table_key.table_id, 0),
       InputDescriptor(inner_table_key.db_id, inner_table_key.table_id, 1),
       InputDescriptor(temporary_table_key.db_id, temporary_table_key.table_id, 2)},
      {},
      {},
      {},
      join_quals,
      groupby_exprs,
      {},
      nullptr,
      SortInfo(),
      0};
  exe_unit.table_id_to_node_map[temporary_table_key] = temporary_source;
  return exe_unit;
}

RelAlgExecutionUnit makeSyntheticUniqueTemporaryJoinGroupByUnit(
    const shared::TableKey& outer_table_key,
    const int32_t outer_join_column_id,
    const shared::TableKey& temporary_table_key,
    const RelAlgNode* temporary_source) {
  const auto bigint_type = SQLTypeInfo(kBIGINT, false);
  auto outer_join_column = makeExpr<Analyzer::ColumnVar>(
      bigint_type,
      shared::ColumnKey{
          outer_table_key.db_id, outer_table_key.table_id, outer_join_column_id},
      0);
  auto temporary_join_column = makeExpr<Analyzer::ColumnVar>(
      bigint_type,
      shared::ColumnKey{temporary_table_key.db_id, temporary_table_key.table_id, 0},
      1);
  JoinQualsPerNestingLevel join_quals{
      JoinCondition{{makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                                 false,
                                                 kEQ,
                                                 kONE,
                                                 outer_join_column,
                                                 temporary_join_column)},
                    JoinType::INNER}};
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs{outer_join_column};
  RelAlgExecutionUnit exe_unit{
      {InputDescriptor(outer_table_key.db_id, outer_table_key.table_id, 0),
       InputDescriptor(temporary_table_key.db_id, temporary_table_key.table_id, 1)},
      {},
      {},
      {},
      join_quals,
      groupby_exprs,
      {},
      nullptr,
      SortInfo(),
      0};
  exe_unit.table_id_to_node_map[temporary_table_key] = temporary_source;
  return exe_unit;
}

RelAlgExecutionUnit makeSyntheticCompositeTemporaryJoinGroupByUnit(
    const shared::TableKey& outer_table_key,
    const std::vector<int32_t>& outer_group_column_ids,
    const shared::TableKey& temporary_table_key,
    const std::vector<int32_t>& temporary_join_column_ids,
    const RelAlgNode* temporary_source,
    const bool reverse_join_operands = false) {
  CHECK(!outer_group_column_ids.empty());
  CHECK(!temporary_join_column_ids.empty());
  CHECK_LE(temporary_join_column_ids.size(), outer_group_column_ids.size());
  const auto bigint_type = SQLTypeInfo(kBIGINT, false);
  std::vector<std::shared_ptr<Analyzer::Expr>> outer_columns;
  outer_columns.reserve(outer_group_column_ids.size());
  std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
  for (const auto column_id : outer_group_column_ids) {
    auto column = makeExpr<Analyzer::ColumnVar>(
        bigint_type,
        shared::ColumnKey{outer_table_key.db_id, outer_table_key.table_id, column_id},
        0);
    outer_columns.push_back(column);
    groupby_exprs.emplace_back(std::move(column));
  }

  std::vector<std::shared_ptr<Analyzer::Expr>> temporary_columns;
  temporary_columns.reserve(temporary_join_column_ids.size());
  for (const auto column_id : temporary_join_column_ids) {
    temporary_columns.emplace_back(makeExpr<Analyzer::ColumnVar>(
        bigint_type,
        shared::ColumnKey{
            temporary_table_key.db_id, temporary_table_key.table_id, column_id},
        1));
  }
  outer_columns.resize(temporary_join_column_ids.size());
  std::shared_ptr<Analyzer::Expr> outer_join_operand =
      outer_columns.size() == size_t(1)
          ? outer_columns.front()
          : makeExpr<Analyzer::ExpressionTuple>(outer_columns);
  std::shared_ptr<Analyzer::Expr> temporary_join_operand =
      temporary_columns.size() == size_t(1)
          ? temporary_columns.front()
          : makeExpr<Analyzer::ExpressionTuple>(temporary_columns);
  if (reverse_join_operands) {
    std::swap(outer_join_operand, temporary_join_operand);
  }
  JoinQualsPerNestingLevel join_quals{
      JoinCondition{{makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                                 false,
                                                 kEQ,
                                                 kONE,
                                                 outer_join_operand,
                                                 temporary_join_operand)},
                    JoinType::INNER}};
  RelAlgExecutionUnit exe_unit{
      {InputDescriptor(outer_table_key.db_id, outer_table_key.table_id, 0),
       InputDescriptor(temporary_table_key.db_id, temporary_table_key.table_id, 1)},
      {},
      {},
      {},
      join_quals,
      groupby_exprs,
      {},
      nullptr,
      SortInfo(),
      0};
  exe_unit.table_id_to_node_map[temporary_table_key] = temporary_source;
  return exe_unit;
}

RelAlgExecutionUnit makeSyntheticProjectionUnit(const int32_t db_id,
                                                const int32_t table_id) {
  return RelAlgExecutionUnit{{InputDescriptor(db_id, table_id, 0)},
                             {},
                             {},
                             {},
                             {},
                             {},
                             {},
                             nullptr,
                             SortInfo(),
                             0};
}

InputTableInfo makeFragmentedTableInfo(const int32_t db_id,
                                       const int32_t table_id,
                                       const size_t fragment_count,
                                       const size_t row_count) {
  CHECK_GT(fragment_count, size_t(0));
  Fragmenter_Namespace::TableInfo table_info;
  table_info.setPhysicalNumTuples(row_count);
  const auto rows_per_fragment = row_count / fragment_count;
  const auto remainder = row_count % fragment_count;
  for (size_t fragment_idx = 0; fragment_idx < fragment_count; ++fragment_idx) {
    Fragmenter_Namespace::FragmentInfo fragment_info;
    fragment_info.fragmentId = static_cast<int>(fragment_idx);
    const auto fragment_rows = rows_per_fragment + (fragment_idx < remainder ? 1 : 0);
    fragment_info.setPhysicalNumTuples(fragment_rows);
    table_info.fragments.push_back(fragment_info);
  }
  return {{db_id, table_id}, table_info};
}

InputTableInfo makeGpuDistributedFragmentedTableInfo(
    const int32_t db_id,
    const int32_t table_id,
    const std::vector<std::pair<int, size_t>>& device_rows) {
  Fragmenter_Namespace::TableInfo table_info;
  size_t total_rows = 0;
  for (size_t fragment_idx = 0; fragment_idx < device_rows.size(); ++fragment_idx) {
    const auto [device_id, row_count] = device_rows[fragment_idx];
    Fragmenter_Namespace::FragmentInfo fragment_info;
    fragment_info.fragmentId = static_cast<int>(fragment_idx);
    fragment_info.deviceIds = {/*disk=*/0, /*cpu=*/0, /*gpu=*/device_id};
    fragment_info.setPhysicalNumTuples(row_count);
    table_info.fragments.push_back(fragment_info);
    total_rows += row_count;
  }
  table_info.setPhysicalNumTuples(total_rows);
  return {{db_id, table_id}, table_info};
}

}  // namespace

TEST(KernelPerFragmentGroupByDispatch, KeepsSmallFragmentedTempGroupByMultifrag) {
  if (skip_tests(ExecutorDeviceType::GPU)) {
    GTEST_SKIP() << "GPU not available";
  }
  auto executor = QR::get()->getExecutor().get();
  ASSERT_NE(executor, nullptr);
  ASSERT_GT(executor->maxGpuSlabSize(), size_t(0));
  // Earlier tests in this binary install a one-device selection mock on the shared
  // unitary executor. This heuristic check should use the default available GPU set.
  executor->clearDevicesToUse();

  const auto q18_temp_unit = makeSyntheticGroupByUnit(/*db_id=*/0, /*table_id=*/-17);
  const std::vector<InputTableInfo> q18_temp_table_infos{
      makeFragmentedTableInfo(/*db_id=*/0,
                              /*table_id=*/-17,
                              /*fragment_count=*/376,
                              /*row_count=*/63707)};

  // Q18 regressed when this small fragmented temporary result was split into hundreds
  // of per-fragment kernels. It should stay as one multifragment GPU group-by.
  EXPECT_FALSE(should_prefer_kernel_per_fragment_groupby_for_test(
      q18_temp_unit,
      q18_temp_table_infos,
      /*max_groups_buffer_entry_guess=*/63707,
      executor));

  const auto large_table_unit = makeSyntheticGroupByUnit(/*db_id=*/1, /*table_id=*/1);
  constexpr size_t row_width = 2 * sizeof(int64_t);
  ASSERT_GE(executor->maxGpuSlabSize(), row_width * size_t(16));
  // Model a global result buffer at the slab limit while keeping each of the 16
  // fragment buffers small enough to fit. This remains a KPF candidate on a single
  // GPU, where the per-device batch upper bound otherwise covers the whole table.
  const auto large_group_count = executor->maxGpuSlabSize() / row_width;
  const std::vector<InputTableInfo> large_table_infos{
      makeFragmentedTableInfo(/*db_id=*/1,
                              /*table_id=*/1,
                              /*fragment_count=*/16,
                              /*row_count=*/large_group_count)};

  // The Q18 guard must not disable the existing large-table KPF path.
  EXPECT_TRUE(should_prefer_kernel_per_fragment_groupby_for_test(
      large_table_unit, large_table_infos, large_group_count, executor));

  executor->clearDevicesToUse();
  executor->mockDeviceIdSelectionLogicToOnlyUseSingleDevice();

  const auto outer_key_join_unit = makeSyntheticJoinGroupByUnit(
      /*outer_db_id=*/2,
      /*outer_table_id=*/2,
      /*inner_db_id=*/2,
      /*inner_table_id=*/-2,
      /*group_by_outer=*/true);
  const std::vector<InputTableInfo> join_table_infos{
      makeFragmentedTableInfo(/*db_id=*/2,
                              /*table_id=*/2,
                              /*fragment_count=*/16,
                              /*row_count=*/large_group_count),
      makeFragmentedTableInfo(/*db_id=*/2,
                              /*table_id=*/-2,
                              /*fragment_count=*/1,
                              /*row_count=*/1024)};

  // A grouped join whose keys depend only on the fragmented outer input can bound each
  // kernel's group buffer by the outer fragment. With one selected GPU, this avoids one
  // oversized multifragment kernel while preserving the normal multifrag path elsewhere.
  EXPECT_TRUE(should_prefer_kernel_per_fragment_groupby_for_test(
      outer_key_join_unit, join_table_infos, large_group_count, executor));
  EXPECT_TRUE(should_prefer_kernel_per_fragment_groupby_for_test(
      outer_key_join_unit,
      join_table_infos,
      /*max_groups_buffer_entry_guess=*/1,
      executor));

  AggregatedColRange dense_range_cache;
  dense_range_cache.setColRange(
      PhysicalInput{/*col_id=*/1, /*table_id=*/2, /*db_id=*/2},
      ExpressionRange::makeIntRange(/*min=*/0, /*max=*/1023, /*bucket=*/0, false));
  executor->setColRangeCache(dense_range_cache);
  ScopeGuard clear_dense_range_cache = [executor] {
    executor->setColRangeCache(AggregatedColRange{});
  };

  // A bounded dense key range that fits one global GPU buffer should remain a
  // multifragment launch even when the aggregate also consumes a temporary join input.
  EXPECT_FALSE(should_prefer_kernel_per_fragment_groupby_for_test(
      outer_key_join_unit, join_table_infos, large_group_count, executor));

  const auto inner_key_join_unit = makeSyntheticJoinGroupByUnit(
      /*outer_db_id=*/2,
      /*outer_table_id=*/2,
      /*inner_db_id=*/2,
      /*inner_table_id=*/-2,
      /*group_by_outer=*/false);
  EXPECT_FALSE(should_prefer_kernel_per_fragment_groupby_for_test(
      inner_key_join_unit, join_table_infos, large_group_count, executor));

  executor->clearDevicesToUse();
}

class NonAmplifyingJoinGroupByDispatch : public ::testing::Test {
 protected:
  void SetUp() override {
    dropTables();
    run_ddl_statement(
        "CREATE TABLE kpf_outer(k BIGINT NOT NULL, customer_key BIGINT NOT NULL) "
        "WITH (fragment_size=2);");
    run_ddl_statement(
        "CREATE TABLE kpf_orders("
        "order_key BIGINT NOT NULL, customer_key BIGINT NOT NULL, attr BIGINT, "
        "CONSTRAINT kpf_orders_pk PRIMARY KEY(order_key));");
    run_ddl_statement(
        "CREATE TABLE kpf_customer("
        "customer_key BIGINT NOT NULL, "
        "CONSTRAINT kpf_customer_pk PRIMARY KEY(customer_key));");
    QR::get()->runSQL(
        "INSERT INTO kpf_outer VALUES "
        "(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60);",
        ExecutorDeviceType::CPU);
    QR::get()->runSQL(
        "INSERT INTO kpf_orders VALUES "
        "(1, 10, 101), (2, 20, 102), (3, 30, 103), "
        "(4, 40, 104), (5, 50, 105), (6, 60, 106);",
        ExecutorDeviceType::CPU);
    QR::get()->runSQL(
        "INSERT INTO kpf_customer VALUES "
        "(10), (20), (30), (40), (50), (60);",
        ExecutorDeviceType::CPU);
  }

  void TearDown() override { dropTables(); }

  static void dropTables() {
    run_ddl_statement("DROP TABLE IF EXISTS kpf_outer;");
    run_ddl_statement("DROP TABLE IF EXISTS kpf_orders;");
    run_ddl_statement("DROP TABLE IF EXISTS kpf_customer;");
  }
};

TEST_F(NonAmplifyingJoinGroupByDispatch, UsesOuterFragmentBoundForUniqueInnerInputs) {
  if (skip_tests(ExecutorDeviceType::GPU)) {
    GTEST_SKIP() << "GPU not available";
  }
  auto executor = QR::get()->getExecutor().get();
  auto catalog = QR::get()->getCatalog();
  ASSERT_NE(executor, nullptr);
  ASSERT_NE(catalog, nullptr);
  executor->clearDevicesToUse();

  const auto outer_td = catalog->getMetadataForTable("kpf_outer");
  const auto orders_td = catalog->getMetadataForTable("kpf_orders");
  const auto customer_td = catalog->getMetadataForTable("kpf_customer");
  ASSERT_NE(outer_td, nullptr);
  ASSERT_NE(orders_td, nullptr);
  ASSERT_NE(customer_td, nullptr);
  const auto outer_key_cd = catalog->getMetadataForColumn(outer_td->tableId, "k");
  const auto orders_key_cd =
      catalog->getMetadataForColumn(orders_td->tableId, "order_key");
  const auto orders_customer_cd =
      catalog->getMetadataForColumn(orders_td->tableId, "customer_key");
  const auto orders_attr_cd = catalog->getMetadataForColumn(orders_td->tableId, "attr");
  ASSERT_NE(outer_key_cd, nullptr);
  ASSERT_NE(orders_key_cd, nullptr);
  ASSERT_NE(orders_customer_cd, nullptr);
  ASSERT_NE(orders_attr_cd, nullptr);

  const auto customer_scan = std::make_shared<RelScan>(
      customer_td, std::vector<std::string>{"customer_key"}, *catalog);
  std::vector<std::unique_ptr<const RexScalar>> customer_projection_exprs;
  customer_projection_exprs.emplace_back(
      std::make_unique<RexInput>(customer_scan.get(), 0));
  const auto customer_project = std::make_shared<RelProject>(
      customer_projection_exprs, std::vector<std::string>{"customer_key"}, customer_scan);

  const auto db_id = catalog->getDatabaseId();
  const shared::TableKey outer_table_key{db_id, outer_td->tableId};
  const shared::TableKey orders_table_key{db_id, orders_td->tableId};
  const shared::TableKey temporary_customer_key{0, -17001};
  const auto exe_unit =
      makeSyntheticNonAmplifyingJoinGroupByUnit(outer_table_key,
                                                outer_key_cd->columnId,
                                                orders_table_key,
                                                orders_key_cd->columnId,
                                                orders_customer_cd->columnId,
                                                orders_attr_cd->columnId,
                                                temporary_customer_key,
                                                customer_project.get());

  constexpr size_t fragment_count = 16;
  constexpr size_t outer_row_count = 16 * 1024 * 1024;
  const std::vector<InputTableInfo> table_infos{
      makeFragmentedTableInfo(db_id, outer_td->tableId, fragment_count, outer_row_count),
      makeFragmentedTableInfo(db_id, orders_td->tableId, 1, 1024 * 1024),
      makeFragmentedTableInfo(0, temporary_customer_key.table_id, 1, 1024)};

  const auto previous_constraint_trust = g_trust_unenforced_table_constraints;
  ScopeGuard restore_constraint_trust = [&] {
    g_trust_unenforced_table_constraints = previous_constraint_trust;
  };
  g_trust_unenforced_table_constraints = false;
  EXPECT_EQ(std::nullopt,
            non_amplifying_join_groupby_initial_guess_for_test(exe_unit, table_infos));

  g_trust_unenforced_table_constraints = true;
  EXPECT_EQ(std::optional<size_t>(outer_row_count / fragment_count),
            non_amplifying_join_groupby_initial_guess_for_test(exe_unit, table_infos));
}

TEST_F(NonAmplifyingJoinGroupByDispatch,
       UsesProjectedTrustedPrimaryKeyForTemporaryJoinBound) {
  auto catalog = QR::get()->getCatalog();
  ASSERT_NE(catalog, nullptr);
  const auto outer_td = catalog->getMetadataForTable("kpf_outer");
  const auto customer_td = catalog->getMetadataForTable("kpf_customer");
  ASSERT_NE(outer_td, nullptr);
  ASSERT_NE(customer_td, nullptr);
  const auto outer_key_cd = catalog->getMetadataForColumn(outer_td->tableId, "k");
  ASSERT_NE(outer_key_cd, nullptr);

  const auto customer_scan = std::make_shared<RelScan>(
      customer_td, std::vector<std::string>{"customer_key"}, *catalog);
  std::vector<std::unique_ptr<const RexScalar>> projection_exprs;
  projection_exprs.emplace_back(std::make_unique<RexInput>(customer_scan.get(), 0));
  const auto customer_project = std::make_shared<RelProject>(
      projection_exprs, std::vector<std::string>{"customer_key"}, customer_scan);

  const auto db_id = catalog->getDatabaseId();
  const shared::TableKey outer_table_key{db_id, outer_td->tableId};
  const shared::TableKey temporary_table_key{0, -17002};
  const auto exe_unit =
      makeSyntheticUniqueTemporaryJoinGroupByUnit(outer_table_key,
                                                  outer_key_cd->columnId,
                                                  temporary_table_key,
                                                  customer_project.get());
  constexpr size_t temporary_row_count = 23;
  const std::vector<InputTableInfo> table_infos{
      makeFragmentedTableInfo(db_id, outer_td->tableId, 1, 1024),
      makeFragmentedTableInfo(temporary_table_key.db_id,
                              temporary_table_key.table_id,
                              1,
                              temporary_row_count)};

  const auto previous_constraint_trust = g_trust_unenforced_table_constraints;
  ScopeGuard restore_constraint_trust = [&] {
    g_trust_unenforced_table_constraints = previous_constraint_trust;
  };
  g_trust_unenforced_table_constraints = false;
  EXPECT_EQ(
      std::nullopt,
      unique_temp_join_group_cardinality_upper_bound_for_test(exe_unit, table_infos));

  g_trust_unenforced_table_constraints = true;
  EXPECT_EQ(
      std::optional<size_t>(temporary_row_count),
      unique_temp_join_group_cardinality_upper_bound_for_test(exe_unit, table_infos));
}

TEST_F(NonAmplifyingJoinGroupByDispatch, RejectsProjectedNonUniqueTemporaryJoinColumn) {
  auto catalog = QR::get()->getCatalog();
  ASSERT_NE(catalog, nullptr);
  const auto outer_td = catalog->getMetadataForTable("kpf_outer");
  const auto orders_td = catalog->getMetadataForTable("kpf_orders");
  ASSERT_NE(outer_td, nullptr);
  ASSERT_NE(orders_td, nullptr);
  const auto outer_key_cd = catalog->getMetadataForColumn(outer_td->tableId, "k");
  ASSERT_NE(outer_key_cd, nullptr);

  const auto orders_scan = std::make_shared<RelScan>(
      orders_td, std::vector<std::string>{"order_key", "customer_key", "attr"}, *catalog);
  std::vector<std::unique_ptr<const RexScalar>> projection_exprs;
  projection_exprs.emplace_back(std::make_unique<RexInput>(orders_scan.get(), 1));
  const auto orders_project = std::make_shared<RelProject>(
      projection_exprs, std::vector<std::string>{"customer_key"}, orders_scan);

  const auto db_id = catalog->getDatabaseId();
  const shared::TableKey outer_table_key{db_id, outer_td->tableId};
  const shared::TableKey temporary_table_key{0, -17003};
  const auto exe_unit = makeSyntheticUniqueTemporaryJoinGroupByUnit(
      outer_table_key, outer_key_cd->columnId, temporary_table_key, orders_project.get());
  const std::vector<InputTableInfo> table_infos{
      makeFragmentedTableInfo(db_id, outer_td->tableId, 1, 1024),
      makeFragmentedTableInfo(
          temporary_table_key.db_id, temporary_table_key.table_id, 1, 23)};

  const auto previous_constraint_trust = g_trust_unenforced_table_constraints;
  ScopeGuard restore_constraint_trust = [&] {
    g_trust_unenforced_table_constraints = previous_constraint_trust;
  };
  g_trust_unenforced_table_constraints = true;
  EXPECT_EQ(
      std::nullopt,
      unique_temp_join_group_cardinality_upper_bound_for_test(exe_unit, table_infos));
}

TEST_F(NonAmplifyingJoinGroupByDispatch,
       UsesCompositeTemporaryAggregateKeyForTupleJoinBound) {
  auto catalog = QR::get()->getCatalog();
  ASSERT_NE(catalog, nullptr);
  const auto outer_td = catalog->getMetadataForTable("kpf_outer");
  const auto orders_td = catalog->getMetadataForTable("kpf_orders");
  ASSERT_NE(outer_td, nullptr);
  ASSERT_NE(orders_td, nullptr);
  const auto outer_key_cd = catalog->getMetadataForColumn(outer_td->tableId, "k");
  const auto outer_customer_cd =
      catalog->getMetadataForColumn(outer_td->tableId, "customer_key");
  ASSERT_NE(outer_key_cd, nullptr);
  ASSERT_NE(outer_customer_cd, nullptr);

  const auto orders_scan = std::make_shared<RelScan>(
      orders_td, std::vector<std::string>{"order_key", "customer_key", "attr"}, *catalog);
  std::vector<std::unique_ptr<const RexAgg>> aggregate_exprs;
  const auto keyset_aggregate = std::make_shared<RelAggregate>(
      2,
      aggregate_exprs,
      std::vector<std::string>{"order_key", "customer_key"},
      orders_scan);

  const auto db_id = catalog->getDatabaseId();
  const shared::TableKey outer_table_key{db_id, outer_td->tableId};
  const shared::TableKey temporary_table_key{0, -17004};
  constexpr size_t temporary_row_count = 29;
  const std::vector<InputTableInfo> table_infos{
      makeFragmentedTableInfo(db_id, outer_td->tableId, 1, 1024),
      makeFragmentedTableInfo(temporary_table_key.db_id,
                              temporary_table_key.table_id,
                              1,
                              temporary_row_count)};
  const std::vector<int32_t> outer_group_column_ids{outer_key_cd->columnId,
                                                    outer_customer_cd->columnId};
  const std::vector<int32_t> temporary_join_column_ids{0, 1};

  for (const bool reverse_join_operands : {false, true}) {
    const auto exe_unit =
        makeSyntheticCompositeTemporaryJoinGroupByUnit(outer_table_key,
                                                       outer_group_column_ids,
                                                       temporary_table_key,
                                                       temporary_join_column_ids,
                                                       keyset_aggregate.get(),
                                                       reverse_join_operands);
    EXPECT_EQ(
        std::optional<size_t>(temporary_row_count),
        unique_temp_join_group_cardinality_upper_bound_for_test(exe_unit, table_infos));
  }
}

TEST_F(NonAmplifyingJoinGroupByDispatch, RejectsIncompleteCompositeTemporaryJoinMapping) {
  auto catalog = QR::get()->getCatalog();
  ASSERT_NE(catalog, nullptr);
  const auto outer_td = catalog->getMetadataForTable("kpf_outer");
  const auto orders_td = catalog->getMetadataForTable("kpf_orders");
  ASSERT_NE(outer_td, nullptr);
  ASSERT_NE(orders_td, nullptr);
  const auto outer_key_cd = catalog->getMetadataForColumn(outer_td->tableId, "k");
  const auto outer_customer_cd =
      catalog->getMetadataForColumn(outer_td->tableId, "customer_key");
  ASSERT_NE(outer_key_cd, nullptr);
  ASSERT_NE(outer_customer_cd, nullptr);

  const auto orders_scan = std::make_shared<RelScan>(
      orders_td, std::vector<std::string>{"order_key", "customer_key", "attr"}, *catalog);
  std::vector<std::unique_ptr<const RexAgg>> aggregate_exprs;
  const auto keyset_aggregate = std::make_shared<RelAggregate>(
      2,
      aggregate_exprs,
      std::vector<std::string>{"order_key", "customer_key"},
      orders_scan);

  const auto db_id = catalog->getDatabaseId();
  const shared::TableKey outer_table_key{db_id, outer_td->tableId};
  const shared::TableKey temporary_table_key{0, -17005};
  const auto exe_unit = makeSyntheticCompositeTemporaryJoinGroupByUnit(
      outer_table_key,
      {outer_key_cd->columnId, outer_customer_cd->columnId},
      temporary_table_key,
      {0},
      keyset_aggregate.get());
  const std::vector<InputTableInfo> table_infos{
      makeFragmentedTableInfo(db_id, outer_td->tableId, 1, 1024),
      makeFragmentedTableInfo(
          temporary_table_key.db_id, temporary_table_key.table_id, 1, 29)};

  EXPECT_EQ(
      std::nullopt,
      unique_temp_join_group_cardinality_upper_bound_for_test(exe_unit, table_infos));
}

TEST_F(NonAmplifyingJoinGroupByDispatch, RejectsNonUniqueCompositeTemporaryJoinColumns) {
  auto catalog = QR::get()->getCatalog();
  ASSERT_NE(catalog, nullptr);
  const auto outer_td = catalog->getMetadataForTable("kpf_outer");
  const auto orders_td = catalog->getMetadataForTable("kpf_orders");
  ASSERT_NE(outer_td, nullptr);
  ASSERT_NE(orders_td, nullptr);
  const auto outer_key_cd = catalog->getMetadataForColumn(outer_td->tableId, "k");
  const auto outer_customer_cd =
      catalog->getMetadataForColumn(outer_td->tableId, "customer_key");
  ASSERT_NE(outer_key_cd, nullptr);
  ASSERT_NE(outer_customer_cd, nullptr);

  const auto orders_scan = std::make_shared<RelScan>(
      orders_td, std::vector<std::string>{"order_key", "customer_key", "attr"}, *catalog);
  std::vector<std::unique_ptr<const RexScalar>> projection_exprs;
  projection_exprs.emplace_back(std::make_unique<RexInput>(orders_scan.get(), 1));
  projection_exprs.emplace_back(std::make_unique<RexInput>(orders_scan.get(), 2));
  const auto orders_project = std::make_shared<RelProject>(
      projection_exprs, std::vector<std::string>{"customer_key", "attr"}, orders_scan);

  const auto db_id = catalog->getDatabaseId();
  const shared::TableKey outer_table_key{db_id, outer_td->tableId};
  const shared::TableKey temporary_table_key{0, -17006};
  const auto exe_unit = makeSyntheticCompositeTemporaryJoinGroupByUnit(
      outer_table_key,
      {outer_key_cd->columnId, outer_customer_cd->columnId},
      temporary_table_key,
      {0, 1},
      orders_project.get());
  const std::vector<InputTableInfo> table_infos{
      makeFragmentedTableInfo(db_id, outer_td->tableId, 1, 1024),
      makeFragmentedTableInfo(
          temporary_table_key.db_id, temporary_table_key.table_id, 1, 29)};

  const auto previous_constraint_trust = g_trust_unenforced_table_constraints;
  ScopeGuard restore_constraint_trust = [&] {
    g_trust_unenforced_table_constraints = previous_constraint_trust;
  };
  g_trust_unenforced_table_constraints = true;
  EXPECT_EQ(
      std::nullopt,
      unique_temp_join_group_cardinality_upper_bound_for_test(exe_unit, table_infos));
}

TEST_F(NonAmplifyingJoinGroupByDispatch, RetriesWhenGlobalGroupsExceedInitialGuess) {
  if (skip_tests(ExecutorDeviceType::GPU)) {
    GTEST_SKIP() << "GPU not available";
  }
  const auto previous_constraint_trust = g_trust_unenforced_table_constraints;
  const auto previous_reduction_pipeline = g_enable_result_reduction_pipeline;
  ScopeGuard restore_flags = [&] {
    g_trust_unenforced_table_constraints = previous_constraint_trust;
    g_enable_result_reduction_pipeline = previous_reduction_pipeline;
  };
  g_trust_unenforced_table_constraints = true;
  g_enable_result_reduction_pipeline = true;
  Executor::clearCardinalityCache();

  const auto result = QR::get()->runSQL(
      "SELECT o.k, i.attr, COUNT(*) "
      "FROM kpf_outer o JOIN kpf_orders i ON o.k = i.order_key "
      "GROUP BY o.k, i.attr ORDER BY o.k;",
      ExecutorDeviceType::GPU);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(size_t(6), result->rowCount());
  for (int64_t expected_key = 1; expected_key <= 6; ++expected_key) {
    const auto row = result->getNextRow(false, false);
    ASSERT_EQ(size_t(3), row.size());
    EXPECT_EQ(expected_key, v<int64_t>(row[0]));
    EXPECT_EQ(expected_key + 100, v<int64_t>(row[1]));
    EXPECT_EQ(int64_t(1), v<int64_t>(row[2]));
  }
}

TEST(KernelPerFragmentProjectionDispatch, BalancesFragmentAlignedGpuBatches) {
  constexpr size_t fragment_rows = 16;
  std::vector<std::pair<int, size_t>> device_rows;
  for (size_t fragment_idx = 0; fragment_idx < 12; ++fragment_idx) {
    device_rows.emplace_back(/*device_id=*/0, fragment_rows);
    device_rows.emplace_back(/*device_id=*/1,
                             fragment_idx == 11 ? size_t(12) : fragment_rows);
  }
  const std::vector<InputTableInfo> table_infos{makeGpuDistributedFragmentedTableInfo(
      /*db_id=*/1, /*table_id=*/1, device_rows)};

  // A 125-row allocation cap permits two kernels per device. Balance six fragments
  // into each kernel instead of filling one 112-row kernel and sizing its short tail
  // to the same capacity.
  EXPECT_EQ(std::optional<size_t>(96),
            projection_batched_fragment_output_upper_bound_for_test(
                table_infos, /*max_rows_per_batch=*/125));

  // When one per-device allocation fits, account for indivisible fragment placement
  // rather than using the lower global-row average.
  EXPECT_EQ(std::optional<size_t>(192),
            projection_batched_fragment_output_upper_bound_for_test(
                table_infos, /*max_rows_per_batch=*/200));
}

TEST(ProjectionOutputSizing, UsesOnlyCompleteDeviceAlignedPreflightCounts) {
  const std::vector<InputTableInfo> table_infos{makeGpuDistributedFragmentedTableInfo(
      /*db_id=*/1,
      /*table_id=*/1,
      {{0, 16}, {1, 16}, {0, 16}, {1, 16}})};
  auto exe_unit = makeSyntheticProjectionUnit(/*db_id=*/1, /*table_id=*/1);
  exe_unit.per_device_cardinality = {{{0, 2}, 7}, {{1, 3}, 11}};
  EXPECT_EQ(std::optional<size_t>(11),
            exact_per_device_projection_output_bound_for_test(exe_unit, table_infos));

  exe_unit.per_device_cardinality = {{{0, 2}, 7}, {{1}, 11}};
  EXPECT_EQ(std::nullopt,
            exact_per_device_projection_output_bound_for_test(exe_unit, table_infos));

  exe_unit.per_device_cardinality = {{{0, 1}, 7}, {{2, 3}, 11}};
  EXPECT_EQ(std::nullopt,
            exact_per_device_projection_output_bound_for_test(exe_unit, table_infos));

  exe_unit.per_device_cardinality = {{{0, 2}, 7}, {{1, 2, 3}, 11}};
  EXPECT_EQ(std::nullopt,
            exact_per_device_projection_output_bound_for_test(exe_unit, table_infos));
}

TEST(ProjectionOutputSizing, CanonicalizesCachedPerDevicePreflightCounts) {
  const Executor::FilteredCountCacheValue first{42, {{{3, 1}, 7}, {{2, 0}, 11}}};
  const Executor::FilteredCountCacheValue differently_completed{
      42, {{{0, 2}, 11}, {{1, 3}, 7}}};
  const Executor::PerDeviceCardinality expected{{{0, 2}, 11}, {{1, 3}, 7}};

  EXPECT_EQ(first, differently_completed);
  EXPECT_EQ(expected, first.per_device_cardinality);
}

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  QR::init(BASE_PATH);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
