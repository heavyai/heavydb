/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "TestHelpers.h"

#include <boost/filesystem.hpp>
#include <fstream>

#include "../QueryEngine/Execute.h"
#include "../QueryEngine/InputMetadata.h"
#include "../QueryRunner/QueryRunner.h"
#include "Shared/scope.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_is_test_env;
extern bool g_enable_watchdog;
extern size_t g_big_group_threshold;
extern size_t kMaxNumElemsForBucketizedRange;

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
class SerialKernelCapacityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Executor::nukeCacheOfExecutors();
    run_ddl_statement("DROP TABLE IF EXISTS serial_kernel_capacity;");
    run_ddl_statement(
        "CREATE TABLE serial_kernel_capacity (x INT) "
        "WITH (FRAGMENT_SIZE=2);");
    QR::get()->runSQL(
        "INSERT INTO serial_kernel_capacity VALUES (1), (2), (3), "
        "(4), (5), (6);",
        ExecutorDeviceType::CPU);
  }

  void TearDown() override {
    run_ddl_statement("DROP TABLE serial_kernel_capacity;");
    // Direct kernel calls bypass the normal query-end metadata-cache cleanup.
    Executor::nukeCacheOfExecutors();
  }

  void checkSerialCapacity(const bool is_agg) {
    auto executor = QR::get()->getExecutor();
    auto cat = QR::get()->getCatalog();
    const auto td = cat->getMetadataForTable("serial_kernel_capacity");
    const auto cd = cat->getMetadataForColumn(td->tableId, "x");
    const auto db_id = cat->getDatabaseId();
    executor->setupCaching({PhysicalInput{cd->columnId, td->tableId, db_id}},
                           {shared::TableKey{db_id, td->tableId}});
    const auto input =
        std::make_shared<const InputColDescriptor>(cd->columnId, td->tableId, db_id, 0);
    std::shared_ptr<Analyzer::Expr> target = makeExpr<Analyzer::ColumnVar>(
        cd->columnType, shared::ColumnKey{db_id, td->tableId, cd->columnId}, 0);
    std::list<std::shared_ptr<Analyzer::Expr>> groupby_exprs;
    if (is_agg) {
      target = makeExpr<Analyzer::AggExpr>(
          SQLTypeInfo(kBIGINT, false), kCOUNT, target, false, nullptr);
    } else {
      groupby_exprs.push_back(nullptr);
    }
    const RelAlgExecutionUnit unit{{input->getScanDesc()},
                                   {input},
                                   {},
                                   {},
                                   {},
                                   groupby_exprs,
                                   {target.get()},
                                   nullptr,
                                   SortInfo(),
                                   0};
    const auto table_infos = get_table_infos(unit, executor.get());
    ASSERT_EQ(table_infos.size(), size_t(1));
    ASSERT_EQ(table_infos.front().info.fragments.size(), size_t(3));
    const auto co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
    const auto eo = ExecutionOptions::defaults();
    const auto resource_mgr = Executor::executor_resource_mgr_;
    ASSERT_TRUE(resource_mgr);
    using ExecutorResourceMgr_Namespace::ResourceType;
    const auto [allocated_slots, total_slots] =
        resource_mgr->get_resource_info(ResourceType::CPU_SLOTS);
    ASSERT_EQ(allocated_slots, size_t(0));
    ScopeGuard restore_resources = [&] {
      Executor::executor_resource_mgr_ = resource_mgr;
      resource_mgr->set_resource(ResourceType::CPU_SLOTS, total_slots);
    };

    // Two distinct budgets detect the stale cpu_threads() default on any host.
    // Zero selects the no-resource-manager fallback, not a zero-sized pool.
    for (const size_t idle_slots : {1, 2, 0}) {
      SCOPED_TRACE(idle_slots);
      if (idle_slots) {
        resource_mgr->set_resource(ResourceType::CPU_SLOTS, idle_slots);
      } else {
        Executor::executor_resource_mgr_ = nullptr;
      }
      const size_t expected_threads = idle_slots ? 1 + idle_slots : cpu_threads();
      size_t fragments_checked = 0;
      const auto check_result = [&](const ResultSetPtr& result) {
        ASSERT_TRUE(result);
        EXPECT_EQ(result->getQueryMemDesc().getAvailableCpuThreads(), expected_threads);
        EXPECT_EQ(result->rowCount(), is_agg ? size_t(1) : size_t(2));
        if (is_agg) {
          const auto row = result->getNextRow(false, false);
          ASSERT_EQ(row.size(), size_t(1));
          EXPECT_EQ(v<int64_t>(row[0]), int64_t(2));
        }
        ++fragments_checked;
      };
      executor->executeUpdate(
          unit,
          table_infos,
          td,
          co,
          eo,
          *cat,
          executor->getRowSetMemoryOwner(),
          [&](const UpdateLogForFragment& log, TableUpdateMetadata&) {
            check_result(log.getResultSet());
          },
          is_agg);
      EXPECT_EQ(fragments_checked, size_t(3));
    }
  }
};

TEST_F(SerialKernelCapacityTest, ProjectionUsesResourceCapacity) {
  checkSerialCapacity(/*is_agg=*/false);
}

TEST_F(SerialKernelCapacityTest, AggregateUsesResourceCapacity) {
  checkSerialCapacity(/*is_agg=*/true);
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
  // no filters, so expect no throw w/out cardinality estimation
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
