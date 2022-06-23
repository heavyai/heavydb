/*
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

#include "Calcite/CalciteJNI.h"
#include "DataMgr/DataMgrBufferProvider.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "SchemaMgr/SimpleSchemaProvider.h"
#include "Shared/Globals.h"
#include "Shared/scope.h"

#include "ArrowTestHelpers.h"
#include "TestDataProvider.h"
#include "TestHelpers.h"
#include "TestRelAlgDagBuilder.h"

#include <gtest/gtest.h>
#include <boost/program_options.hpp>

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;
constexpr int TEST1_TABLE_ID = 1;
constexpr int TEST2_TABLE_ID = 2;
constexpr int TEST_AGG_TABLE_ID = 3;
constexpr int TEST_STREAMING_TABLE_ID = 4;

static bool use_groupby_buffer_desc = false;

using ArrowTestHelpers::compare_res_data;

class TestSchemaProvider : public SimpleSchemaProvider {
 public:
  TestSchemaProvider() : SimpleSchemaProvider(TEST_SCHEMA_ID, "test") {
    // Table test1
    addTableInfo(TEST_DB_ID,
                 TEST1_TABLE_ID,
                 "test1",
                 false,
                 Data_Namespace::MemoryLevel::CPU_LEVEL,
                 1);
    addColumnInfo(
        TEST_DB_ID, TEST1_TABLE_ID, 1, "col_bi", SQLTypeInfo(SQLTypes::kBIGINT), false);
    addColumnInfo(
        TEST_DB_ID, TEST1_TABLE_ID, 2, "col_i", SQLTypeInfo(SQLTypes::kINT), false);
    addColumnInfo(
        TEST_DB_ID, TEST1_TABLE_ID, 3, "col_f", SQLTypeInfo(SQLTypes::kFLOAT), false);
    addColumnInfo(
        TEST_DB_ID, TEST1_TABLE_ID, 4, "col_d", SQLTypeInfo(SQLTypes::kDOUBLE), false);
    addRowidColumn(TEST_DB_ID, TEST1_TABLE_ID);

    // Table test2
    addTableInfo(TEST_DB_ID,
                 TEST2_TABLE_ID,
                 "test2",
                 false,
                 Data_Namespace::MemoryLevel::CPU_LEVEL,
                 1);
    addColumnInfo(
        TEST_DB_ID, TEST2_TABLE_ID, 1, "col_bi", SQLTypeInfo(SQLTypes::kBIGINT), false);
    addColumnInfo(
        TEST_DB_ID, TEST2_TABLE_ID, 2, "col_i", SQLTypeInfo(SQLTypes::kINT), false);
    addColumnInfo(
        TEST_DB_ID, TEST2_TABLE_ID, 3, "col_f", SQLTypeInfo(SQLTypes::kFLOAT), false);
    addColumnInfo(
        TEST_DB_ID, TEST2_TABLE_ID, 4, "col_d", SQLTypeInfo(SQLTypes::kDOUBLE), false);
    addRowidColumn(TEST_DB_ID, TEST2_TABLE_ID);

    // Table test_agg
    addTableInfo(TEST_DB_ID,
                 TEST_AGG_TABLE_ID,
                 "test_agg",
                 false,
                 Data_Namespace::MemoryLevel::CPU_LEVEL,
                 1);
    addColumnInfo(
        TEST_DB_ID, TEST_AGG_TABLE_ID, 1, "id", SQLTypeInfo(SQLTypes::kINT), false);
    addColumnInfo(
        TEST_DB_ID, TEST_AGG_TABLE_ID, 2, "val", SQLTypeInfo(SQLTypes::kINT), false);
    addRowidColumn(TEST_DB_ID, TEST_AGG_TABLE_ID);

    // Table test_streaming
    addTableInfo(TEST_DB_ID,
                 TEST_STREAMING_TABLE_ID,
                 "test_streaming",
                 false,
                 Data_Namespace::MemoryLevel::CPU_LEVEL,
                 1,
                 true);
    addColumnInfo(
        TEST_DB_ID, TEST_STREAMING_TABLE_ID, 1, "id", SQLTypeInfo(SQLTypes::kINT), false);
    addColumnInfo(TEST_DB_ID,
                  TEST_STREAMING_TABLE_ID,
                  2,
                  "val",
                  SQLTypeInfo(SQLTypes::kINT),
                  false);
    addRowidColumn(TEST_DB_ID, TEST_STREAMING_TABLE_ID);
  }

  ~TestSchemaProvider() override = default;
};

class TestDataProvider : public TestHelpers::TestDataProvider {
 public:
  TestDataProvider(SchemaProviderPtr schema_provider)
      : TestHelpers::TestDataProvider(TEST_DB_ID, schema_provider) {
    TestHelpers::TestTableData test1(TEST_DB_ID, TEST1_TABLE_ID, 4, schema_provider_);
    test1.addColFragment<int64_t>(1, {1, 2, 3, 4, 5});
    test1.addColFragment<int32_t>(2, {10, 20, 30, 40, 50});
    test1.addColFragment<float>(3, {1.1, 2.2, 3.3, 4.4, 5.5});
    test1.addColFragment<double>(4, {10.1, 20.2, 30.3, 40.4, 50.5});
    tables_.emplace(std::make_pair(TEST1_TABLE_ID, test1));

    TestHelpers::TestTableData test2(TEST_DB_ID, TEST2_TABLE_ID, 4, schema_provider_);
    test2.addColFragment<int64_t>(1, {1, 2, 3});
    test2.addColFragment<int64_t>(1, {4, 5, 6});
    test2.addColFragment<int64_t>(1, {7, 8, 9});
    test2.addColFragment<int32_t>(2, {110, 120, 130});
    test2.addColFragment<int32_t>(2, {140, 150, 160});
    test2.addColFragment<int32_t>(2, {170, 180, 190});
    test2.addColFragment<float>(3, {101.1, 102.2, 103.3});
    test2.addColFragment<float>(3, {104.4, 105.5, 106.6});
    test2.addColFragment<float>(3, {107.7, 108.8, 109.9});
    test2.addColFragment<double>(4, {110.1, 120.2, 130.3});
    test2.addColFragment<double>(4, {140.4, 150.5, 160.6});
    test2.addColFragment<double>(4, {170.7, 180.8, 190.9});
    tables_.emplace(std::make_pair(TEST2_TABLE_ID, test2));

    TestHelpers::TestTableData test_agg(
        TEST_DB_ID, TEST_AGG_TABLE_ID, 2, schema_provider_);
    test_agg.addColFragment<int32_t>(1, {1, 2, 1, 2, 1});
    test_agg.addColFragment<int32_t>(1, {2, 1, 3, 1, 3});
    test_agg.addColFragment<int32_t>(2, {10, 20, 30, 40, 50});
    test_agg.addColFragment<int32_t>(
        2, {inline_null_value<int32_t>(), 70, inline_null_value<int32_t>(), 90, 100});
    tables_.emplace(std::make_pair(TEST_AGG_TABLE_ID, test_agg));

    TestHelpers::TestTableData test_streaming(
        TEST_DB_ID, TEST_STREAMING_TABLE_ID, 2, schema_provider_);
    tables_.emplace(std::make_pair(TEST_STREAMING_TABLE_ID, test_streaming));
  }

  ~TestDataProvider() override = default;
};

class NoCatalogSqlTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    config_ = std::make_shared<Config>();

    schema_provider_ = std::make_shared<TestSchemaProvider>();

    SystemParameters system_parameters;
    data_mgr_ = std::make_shared<DataMgr>(
        "", system_parameters, std::map<GpuMgrName, std::unique_ptr<GpuMgr>>(), false);

    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID,
                                 std::make_shared<TestDataProvider>(schema_provider_));

    executor_ = Executor::getExecutor(0,
                                      data_mgr_.get(),
                                      data_mgr_->getBufferProvider(),
                                      config_,
                                      "",
                                      "",
                                      system_parameters);

    init_calcite("");
  }

  static void init_calcite(const std::string& udf_filename) {
    calcite_ = std::make_shared<CalciteJNI>(schema_provider_, config_, udf_filename);
  }

  static void TearDownTestSuite() {
    data_mgr_.reset();
    schema_provider_.reset();
    executor_.reset();
    calcite_.reset();
  }

  ExecutionResult runRAQuery(const std::string& query_ra) {
    auto dag = std::make_unique<RelAlgDagBuilder>(query_ra, TEST_DB_ID, schema_provider_);
    auto ra_executor = RelAlgExecutor(
        executor_.get(), schema_provider_, data_mgr_->getDataProvider(), std::move(dag));

    auto co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
    co.use_groupby_buffer_desc = use_groupby_buffer_desc;
    return ra_executor.executeRelAlgQuery(co, ExecutionOptions(), false);
  }

  ExecutionResult runSqlQuery(const std::string& sql) {
    const auto query_ra = calcite_->process("test_db", pg_shim(sql));
    return runRAQuery(query_ra);
  }

  RelAlgExecutor getExecutor(const std::string& sql) {
    const auto query_ra = calcite_->process("test_db", pg_shim(sql));
    auto dag = std::make_unique<RelAlgDagBuilder>(query_ra, TEST_DB_ID, schema_provider_);
    return RelAlgExecutor(
        executor_.get(), schema_provider_, data_mgr_->getDataProvider(), std::move(dag));
  }

  TestDataProvider& getDataProvider() {
    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    auto data_provider_ptr = ps_mgr->getDataProvider(TEST_SCHEMA_ID);
    return dynamic_cast<TestDataProvider&>(*data_provider_ptr);
  }

 protected:
  static ConfigPtr config_;
  static std::shared_ptr<DataMgr> data_mgr_;
  static SchemaProviderPtr schema_provider_;
  static std::shared_ptr<Executor> executor_;
  static std::shared_ptr<CalciteJNI> calcite_;
};

ConfigPtr NoCatalogSqlTest::config_;
std::shared_ptr<DataMgr> NoCatalogSqlTest::data_mgr_;
SchemaProviderPtr NoCatalogSqlTest::schema_provider_;
std::shared_ptr<Executor> NoCatalogSqlTest::executor_;
std::shared_ptr<CalciteJNI> NoCatalogSqlTest::calcite_;

TEST_F(NoCatalogSqlTest, SelectSingleColumn) {
  auto res = runSqlQuery("SELECT col_i FROM test1;");
  compare_res_data(res, std::vector<int>({10, 20, 30, 40, 50}));
}

TEST_F(NoCatalogSqlTest, SelectAllColumns) {
  auto res = runSqlQuery("SELECT * FROM test1;");
  compare_res_data(res,
                   std::vector<int64_t>({1, 2, 3, 4, 5}),
                   std::vector<int>({10, 20, 30, 40, 50}),
                   std::vector<float>({1.1, 2.2, 3.3, 4.4, 5.5}),
                   std::vector<double>({10.1, 20.2, 30.3, 40.4, 50.5}));
}

TEST_F(NoCatalogSqlTest, SelectAllColumnsMultiFrag) {
  auto res = runSqlQuery("SELECT * FROM test2;");
  compare_res_data(
      res,
      std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}),
      std::vector<int>({110, 120, 130, 140, 150, 160, 170, 180, 190}),
      std::vector<float>({101.1, 102.2, 103.3, 104.4, 105.5, 106.6, 107.7, 108.8, 109.9}),
      std::vector<double>(
          {110.1, 120.2, 130.3, 140.4, 150.5, 160.6, 170.7, 180.8, 190.9}));
}

TEST_F(NoCatalogSqlTest, GroupBySingleColumn) {
  auto res = runSqlQuery(
      "SELECT id, COUNT(*), COUNT(val), SUM(val), AVG(val) FROM test_agg GROUP BY id "
      "ORDER BY id;");
  compare_res_data(res,
                   std::vector<int32_t>({1, 2, 3}),
                   std::vector<int32_t>({5, 3, 2}),
                   std::vector<int32_t>({5, 2, 1}),
                   std::vector<int64_t>({250, 60, 100}),
                   std::vector<double>({50, 30, 100}));
}

TEST_F(NoCatalogSqlTest, StreamingAggregate) {
  auto ra_executor = getExecutor("SELECT SUM(val) FROM test_streaming;");
  ra_executor.prepareStreamingExecution(CompilationOptions(), ExecutionOptions());
  TestDataProvider& data_provider = getDataProvider();

  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 1, {1, 2, 3});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 1, {2, 1, 2});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 2, {3, 3, 3});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 2, {3, 1, 4});

  (void)ra_executor.runOnBatch({TEST_DB_ID, TEST_STREAMING_TABLE_ID, {0, 1}});

  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 1, {4, 5, 6});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 2, {7, 8, 9});

  (void)ra_executor.runOnBatch({TEST_DB_ID, TEST_STREAMING_TABLE_ID, {2}});

  auto rs = ra_executor.finishStreamingExecution();

  std::vector<std::string> col_names;
  col_names.push_back("sum");
  auto converter = std::make_unique<ArrowResultSetConverter>(rs, col_names, -1);
  auto at = converter->convertToArrowTable();

  ArrowTestHelpers::compare_arrow_table(at, std::vector<int64_t>{41});
}

TEST_F(NoCatalogSqlTest, StreamingFilter) {
  GTEST_SKIP();
  auto ra_executor = getExecutor("SELECT val FROM test_streaming WHERE val > 20;");
  ra_executor.prepareStreamingExecution(CompilationOptions(), ExecutionOptions());

  std::vector<std::string> col_names;
  col_names.push_back("val");

  TestDataProvider& data_provider = getDataProvider();

  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 1, {10, 20, 30});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 1, {2, 1, 2});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 2, {3, 30, 3});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 2, {30, 1, 40});

  ASSERT_EQ(ra_executor.runOnBatch({TEST_DB_ID, TEST_STREAMING_TABLE_ID, {0, 1}}),
            nullptr);

  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 1, {40, 50, 60});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 2, {70, 8, 90});

  ASSERT_EQ(ra_executor.runOnBatch({TEST_DB_ID, TEST_STREAMING_TABLE_ID, {2}}), nullptr);

  auto rs = ra_executor.finishStreamingExecution();

  auto converter = std::make_unique<ArrowResultSetConverter>(rs, col_names, -1);
  auto at = converter->convertToArrowTable();
  ArrowTestHelpers::compare_arrow_table(at, std::vector<int32_t>{30, 30, 40, 70, 90});
}

TEST_F(NoCatalogSqlTest, MultipleCalciteMultipleThreads) {
  constexpr int TEST_NTHREADS = 100;
  std::vector<ExecutionResult> res;
  std::vector<std::future<void>> threads;
  res.resize(TEST_NTHREADS);
  threads.resize(TEST_NTHREADS);
  for (int i = 0; i < TEST_NTHREADS; ++i) {
    threads[i] = std::async(std::launch::async, [this, i, &res]() {
      auto calcite = std::make_unique<CalciteJNI>(schema_provider_, config_);
      auto query_ra = calcite->process(
          "test_db", "SELECT col_bi + " + std::to_string(i) + " FROM test1;");
      res[i] = runRAQuery(query_ra);
    });
  }
  for (int i = 0; i < TEST_NTHREADS; ++i) {
    threads[i].wait();
  }
  for (int i = 0; i < TEST_NTHREADS; ++i) {
    compare_res_data(res[i], std::vector<int64_t>({1 + i, 2 + i, 3 + i, 4 + i, 5 + i}));
  }
}

TEST(CalciteReinitTest, SingleThread) {
  auto schema_provider = std::make_shared<TestSchemaProvider>();
  auto config = std::make_shared<Config>();
  for (int i = 0; i < 10; ++i) {
    auto calcite = std::make_shared<CalciteJNI>(schema_provider, config);
    auto query_ra = calcite->process("test_db", "SELECT 1;");
    CHECK(query_ra.find("LogicalValues") != std::string::npos);
    CHECK(query_ra.find("LogicalProject") != std::string::npos);
  }
}

TEST(CalciteReinitTest, MultipleThreads) {
  auto schema_provider = std::make_shared<TestSchemaProvider>();
  auto config = std::make_shared<Config>();
  for (int i = 0; i < 10; ++i) {
    auto f = std::async(std::launch::async, [schema_provider, config]() {
      auto calcite = std::make_shared<CalciteJNI>(schema_provider, config);
      auto query_ra = calcite->process("test_db", "SELECT 1;");
      CHECK(query_ra.find("LogicalValues") != std::string::npos);
      CHECK(query_ra.find("LogicalProject") != std::string::npos);
    });
    f.wait();
  }
}

void parse_cli_args_to_globals(int argc, char* argv[]) {
  namespace po = boost::program_options;

  po::options_description desc("Options");

  desc.add_options()("help,h", "Print help messages ");

  desc.add_options()("use-groupby-buffer-desc",
                     po::bool_switch()->default_value(false),
                     "Use GroupBy Buffer Descriptor for hash tables.");

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

    if (vm.count("help")) {
      std::cout << desc;
      std::exit(EXIT_SUCCESS);
    }
    po::notify(vm);
    use_groupby_buffer_desc = vm["use-groupby-buffer-desc"].as<bool>();

  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    std::cout << desc;
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  parse_cli_args_to_globals(argc, argv);

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
