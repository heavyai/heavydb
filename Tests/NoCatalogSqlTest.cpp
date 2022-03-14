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

#include "Calcite/Calcite.h"
#include "DataMgr/DataMgrBufferProvider.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "SchemaMgr/SimpleSchemaProvider.h"

#include "gen-cpp/CalciteServer.h"

#include "ArrowTestHelpers.h"
#include "SchemaJson.h"
#include "TestDataProvider.h"
#include "TestHelpers.h"
#include "TestRelAlgDagBuilder.h"

#include <gtest/gtest.h>

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;
constexpr int TEST1_TABLE_ID = 1;
constexpr int TEST2_TABLE_ID = 2;
constexpr int TEST_AGG_TABLE_ID = 3;
constexpr int TEST_STREAMING_TABLE_ID = 4;

constexpr int CALCITE_PORT = 3278;

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
    schema_provider_ = std::make_shared<TestSchemaProvider>();

    SystemParameters system_parameters;
    data_mgr_ = std::make_shared<DataMgr>("", system_parameters, nullptr, false);
    data_provider_ = std::make_shared<DataMgrDataProvider>(data_mgr_.get());

    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID,
                                 std::make_shared<TestDataProvider>(schema_provider_));

    executor_ = std::make_shared<Executor>(0,
                                           data_mgr_.get(),
                                           data_mgr_->getBufferProvider(),
                                           system_parameters.cuda_block_size,
                                           system_parameters.cuda_grid_size,
                                           system_parameters.max_gpu_slab_size,
                                           "",
                                           "");

    init_calcite("");
  }

  static void init_calcite(const std::string& udf_filename) {
    calcite_ =
        std::make_shared<Calcite>(-1, CALCITE_PORT, "", 1024, 5000, true, udf_filename);
  }

  static void TearDownTestSuite() {
    data_mgr_.reset();
    schema_provider_.reset();
    executor_.reset();
    calcite_.reset();
  }

  ExecutionResult runSqlQuery(const std::string& sql) {
    auto schema_json = schema_to_json(schema_provider_);
    const auto query_ra =
        calcite_->process("admin", "test_db", pg_shim(sql), schema_json).plan_result;
    auto dag = std::make_unique<RelAlgDagBuilder>(
        query_ra, TEST_DB_ID, schema_provider_, nullptr);
    auto ra_executor = RelAlgExecutor(
        executor_.get(), TEST_DB_ID, schema_provider_, data_provider_, std::move(dag));
    return ra_executor.executeRelAlgQuery(
        CompilationOptions(), ExecutionOptions(), false, nullptr);
  }

  RelAlgExecutor getExecutor(const std::string& sql) {
    auto schema_json = schema_to_json(schema_provider_);
    const auto query_ra =
        calcite_->process("admin", "test_db", pg_shim(sql), schema_json).plan_result;
    auto dag = std::make_unique<RelAlgDagBuilder>(
        query_ra, TEST_DB_ID, schema_provider_, nullptr);
    return RelAlgExecutor(
        executor_.get(), TEST_DB_ID, schema_provider_, data_provider_, std::move(dag));
  }

  TestDataProvider& getDataProvider() {
    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    auto data_provider_ptr = ps_mgr->getDataProvider(TEST_SCHEMA_ID);
    return dynamic_cast<TestDataProvider&>(*data_provider_ptr);
  }

 protected:
  static std::shared_ptr<DataMgr> data_mgr_;
  static SchemaProviderPtr schema_provider_;
  static DataProviderPtr data_provider_;
  static std::shared_ptr<Executor> executor_;
  static std::shared_ptr<Calcite> calcite_;
};

std::shared_ptr<DataMgr> NoCatalogSqlTest::data_mgr_;
SchemaProviderPtr NoCatalogSqlTest::schema_provider_;
DataProviderPtr NoCatalogSqlTest::data_provider_;
std::shared_ptr<Executor> NoCatalogSqlTest::executor_;
std::shared_ptr<Calcite> NoCatalogSqlTest::calcite_;

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

  (void)ra_executor.runOnBatch({TEST_STREAMING_TABLE_ID, {0, 1}});

  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 1, {4, 5, 6});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 2, {7, 8, 9});

  (void)ra_executor.runOnBatch({TEST_STREAMING_TABLE_ID, {2}});

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

  ASSERT_EQ(ra_executor.runOnBatch({TEST_STREAMING_TABLE_ID, {0, 1}}), nullptr);

  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 1, {40, 50, 60});
  data_provider.addTableColumn<int32_t>(TEST_STREAMING_TABLE_ID, 2, {70, 8, 90});

  ASSERT_EQ(ra_executor.runOnBatch({TEST_STREAMING_TABLE_ID, {2}}), nullptr);

  auto rs = ra_executor.finishStreamingExecution();

  auto converter = std::make_unique<ArrowResultSetConverter>(rs, col_names, -1);
  auto at = converter->convertToArrowTable();
  ArrowTestHelpers::compare_arrow_table(at, std::vector<int32_t>{30, 30, 40, 70, 90});
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
