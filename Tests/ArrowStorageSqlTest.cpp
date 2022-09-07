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

#include "ArrowStorage/ArrowStorage.h"
#include "Calcite/CalciteJNI.h"
#include "DataMgr/DataMgrBufferProvider.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/RelAlgExecutor.h"

#include "ArrowSQLRunner/ArrowSQLRunner.h"

#include "ArrowTestHelpers.h"
#include "TestHelpers.h"

#include <gtest/gtest.h>

using namespace std::string_literals;
using ArrowTestHelpers::compare_res_data;
using namespace TestHelpers::ArrowSQLRunner;

namespace {

std::string getFilePath(const std::string& file_name) {
  return std::string("../../Tests/ArrowStorageDataFiles/") + file_name;
}

ExecutionResult runSqlQuery(const std::string& sql) {
  return TestHelpers::ArrowSQLRunner::runSqlQuery(
      sql, CompilationOptions(), ExecutionOptions());
}

}  // anonymous namespace

class ArrowStorageSqlTest : public ::testing::Test,
                            public testing::WithParamInterface<std::string> {
 protected:
  static void SetUpTestSuite() {
    ArrowStorage::TableOptions small_frag_opts;
    small_frag_opts.fragment_size = 3;
    getStorage()->importCsvFile(getFilePath("mixed_data.csv"),
                                "mixed_data",
                                {{"col1", ctx().int32()},
                                 {"col2", ctx().fp32()},
                                 {"col3", ctx().extDict(ctx().text(), 0)},
                                 {"col4", ctx().text()}});
    getStorage()->importCsvFile(getFilePath("mixed_data.csv"),
                                "mixed_data_multifrag",
                                {{"col1", ctx().int32()},
                                 {"col2", ctx().fp32()},
                                 {"col3", ctx().extDict(ctx().text(), 0)},
                                 {"col4", ctx().text()}},
                                small_frag_opts);
  }
};

TEST_P(ArrowStorageSqlTest, SimpleSelect) {
  auto res = runSqlQuery("SELECT col1, col2, col3 FROM "s + GetParam() + ";");
  compare_res_data(
      res,
      std::vector<int32_t>({10, 20, 10, 20, 10, 20, 10, 20, 10, 20}),
      std::vector<float>({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}),
      std::vector<std::string>(
          {"s0"s, "s1"s, "s0"s, "s2"s, "s0"s, "s1"s, "s0"s, "s2"s, "s3"s, "s3"s}));
}

TEST_P(ArrowStorageSqlTest, SelectWithFilter) {
  auto res =
      runSqlQuery("SELECT col1, col2, col3 FROM "s + GetParam() + " WHERE col1 = 10;");
  compare_res_data(res,
                   std::vector<int32_t>({10, 10, 10, 10, 10}),
                   std::vector<float>({0.0f, 2.0f, 4.0f, 6.0f, 8.0f}),
                   std::vector<std::string>({"s0"s, "s0"s, "s0"s, "s0"s, "s3"s}));
}

TEST_P(ArrowStorageSqlTest, SelectWithStringFilter) {
  auto res =
      runSqlQuery("SELECT col1, col2, col3 FROM "s + GetParam() + " WHERE col4 = 'dd2';");
  compare_res_data(res,
                   std::vector<int32_t>({10}),
                   std::vector<float>({2.0f}),
                   std::vector<std::string>({"s0"s}));
}

TEST_P(ArrowStorageSqlTest, SelectWithDictStringFilter) {
  auto res =
      runSqlQuery("SELECT col1, col2, col3 FROM "s + GetParam() + " WHERE col3 = 's0';");
  compare_res_data(res,
                   std::vector<int32_t>({10, 10, 10, 10}),
                   std::vector<float>({0.0f, 2.0f, 4.0f, 6.0f}),
                   std::vector<std::string>({"s0"s, "s0"s, "s0"s, "s0"s}));
}

TEST_P(ArrowStorageSqlTest, GroupBy) {
  auto res = runSqlQuery("SELECT col1, SUM(col2) FROM "s + GetParam() +
                         " GROUP BY col1 ORDER BY col1;");
  compare_res_data(
      res, std::vector<int32_t>({10, 20}), std::vector<float>({20.0f, 25.0f}));
}

TEST_P(ArrowStorageSqlTest, GroupByDict) {
  auto res = runSqlQuery("SELECT SUM(col1), SUM(col2), col3 FROM "s + GetParam() +
                         " GROUP BY col3 ORDER BY col3;");
  compare_res_data(res,
                   std::vector<int64_t>({40, 40, 40, 30}),
                   std::vector<float>({12.0f, 6.0f, 10.0f, 17.0f}),
                   std::vector<std::string>({"s0"s, "s1"s, "s2"s, "s3"s}));
}

INSTANTIATE_TEST_SUITE_P(ArrowStorageSqlTest,
                         ArrowStorageSqlTest,
                         testing::Values("mixed_data"s, "mixed_data_multifrag"s));

class ArrowStorageTaxiTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    ArrowStorage::TableOptions table_options;
    ArrowStorage::CsvParseOptions parse_options;
    parse_options.header = false;
    getStorage()->importCsvFile(
        getFilePath("taxi_sample.csv"),
        "trips",
        {{"trip_id", ctx().int32()},
         {"vendor_id", ctx().extDict(ctx().text(), 0)},
         {"pickup_datetime", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
         {"dropoff_datetime", ctx().timestamp(hdk::ir::TimeUnit::kSecond)},
         {"store_and_fwd_flag", ctx().extDict(ctx().text(), 0)},
         {"rate_code_id", ctx().int16()},
         {"pickup_longitude", ctx().fp64()},
         {"pickup_latitude", ctx().fp64()},
         {"dropoff_longitude", ctx().fp64()},
         {"dropoff_latitude", ctx().fp64()},
         {"passenger_count", ctx().int16()},
         {"trip_distance", ctx().decimal64(14, 2)},
         {"fare_amount", ctx().decimal64(14, 2)},
         {"extra", ctx().decimal64(14, 2)},
         {"mta_tax", ctx().decimal64(14, 2)},
         {"tip_amount", ctx().decimal64(14, 2)},
         {"tolls_amount", ctx().decimal64(14, 2)},
         {"ehail_fee", ctx().decimal64(14, 2)},
         {"improvement_surcharge", ctx().decimal64(14, 2)},
         {"total_amount", ctx().decimal64(14, 2)},
         {"payment_type", ctx().extDict(ctx().text(), 0)},
         {"trip_type", ctx().int16()},
         {"pickup", ctx().extDict(ctx().text(), 0)},
         {"dropoff", ctx().extDict(ctx().text(), 0)},
         {"cab_type", ctx().extDict(ctx().text(), 0)},
         {"precipitation", ctx().decimal64(14, 2)},
         {"snow_depth", ctx().int16()},
         {"snowfall", ctx().decimal64(14, 2)},
         {"max_temperature", ctx().int16()},
         {"min_temperature", ctx().int16()},
         {"average_wind_speed", ctx().decimal64(14, 2)},
         {"pickup_nyct2010_gid", ctx().int16()},
         {"pickup_ctlabel", ctx().extDict(ctx().text(), 0)},
         {"pickup_borocode", ctx().int16()},
         {"pickup_boroname", ctx().extDict(ctx().text(), 0)},
         {"pickup_ct2010", ctx().extDict(ctx().text(), 0)},
         {"pickup_boroct2010", ctx().extDict(ctx().text(), 0)},
         {"pickup_cdeligibil", ctx().extDict(ctx().text(), 0)},
         {"pickup_ntacode", ctx().extDict(ctx().text(), 0)},
         {"pickup_ntaname", ctx().extDict(ctx().text(), 0)},
         {"pickup_puma", ctx().extDict(ctx().text(), 0)},
         {"dropoff_nyct2010_gid", ctx().int16()},
         {"dropoff_ctlabel", ctx().extDict(ctx().text(), 0)},
         {"dropoff_borocode", ctx().int16()},
         {"dropoff_boroname", ctx().extDict(ctx().text(), 0)},
         {"dropoff_ct2010", ctx().extDict(ctx().text(), 0)},
         {"dropoff_boroct2010", ctx().extDict(ctx().text(), 0)},
         {"dropoff_cdeligibil", ctx().extDict(ctx().text(), 0)},
         {"dropoff_ntacode", ctx().extDict(ctx().text(), 0)},
         {"dropoff_ntaname", ctx().extDict(ctx().text(), 0)},
         {"dropoff_puma", ctx().extDict(ctx().text(), 0)}},
        table_options,
        parse_options);
  }
};

TEST_F(ArrowStorageTaxiTest, TaxiQuery1) {
  auto res = runSqlQuery("SELECT cab_type, count(*) FROM trips GROUP BY cab_type;");
  compare_res_data(res, std::vector<std::string>({"green"s}), std::vector<int32_t>({20}));
}

TEST_F(ArrowStorageTaxiTest, TaxiQuery2) {
  auto res = runSqlQuery(
      "SELECT passenger_count, AVG(total_amount) FROM trips GROUP BY passenger_count "
      "ORDER BY passenger_count;");
  compare_res_data(res,
                   std::vector<int16_t>({1, 2, 5}),
                   std::vector<double>({98.19f / 16, 75.0f, 13.58f / 3}));
}

TEST_F(ArrowStorageTaxiTest, TaxiQuery3) {
  auto res = runSqlQuery(
      "SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, "
      "count(*) FROM trips GROUP BY passenger_count, pickup_year ORDER BY "
      "passenger_count;");
  compare_res_data(res,
                   std::vector<int16_t>({1, 2, 5}),
                   std::vector<int64_t>({2013, 2013, 2013}),
                   std::vector<int32_t>({16, 1, 3}));
}

TEST_F(ArrowStorageTaxiTest, TaxiQuery4) {
  auto res = runSqlQuery(
      "SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, "
      "cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips GROUP BY "
      "passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;");
  compare_res_data(res,
                   std::vector<int16_t>({1, 5, 2}),
                   std::vector<int64_t>({2013, 2013, 2013}),
                   std::vector<int32_t>({0, 0, 0}),
                   std::vector<int32_t>({16, 3, 1}));
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  init();

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  reset();

  return err;
}
