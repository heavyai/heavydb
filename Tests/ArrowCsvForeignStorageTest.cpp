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

#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <limits>
#include <string>

#include "Archive/PosixFileArchive.h"
#include "Catalog/Catalog.h"
#include "DataMgr/ForeignStorage/ArrowCsvForeignStorage.h"
#include "Import/Importer.h"
#include "Parser/parser.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/geo_types.h"
#include "Shared/scope.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std;
using namespace TestHelpers;
using QR = QueryRunner::QueryRunner;

extern bool g_use_date_in_days_default_encoding;

namespace {

inline void run_ddl_statement(const string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

TargetValue run_simple_agg(const string& query_str) {
  auto rows = QR::get()->runSQL(query_str, ExecutorDeviceType::CPU, false);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

template <typename T>
void check_query(const std::string& query, const std::vector<T>& expects) {
  auto rows = QR::get()->runSQL(query, ExecutorDeviceType::CPU, false);
  CHECK_EQ(expects.size(), rows->rowCount());
  for (auto exp : expects) {
    auto crt_row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(1), crt_row.size());
    CHECK_EQ(exp, v<T>(crt_row[0]));
  }
}

template <typename T>
void check_table(const std::string& query, const std::vector<std::vector<T>>& expects) {
  auto rows = QR::get()->runSQL(query, ExecutorDeviceType::CPU, false);
  CHECK_EQ(expects.size(), rows->rowCount());
  CHECK_EQ(expects[0].size(), rows->colCount());
  for (auto exp_row : expects) {
    auto crt_row = rows->getNextRow(true, true);
    for (size_t val_idx = 0; val_idx < exp_row.size(); ++val_idx) {
      if (typeid(T) == typeid(double)) {
        EXPECT_DOUBLE_EQ(exp_row[val_idx], v<T>(crt_row[val_idx]));
      } else if (typeid(T) == typeid(float)) {
        EXPECT_FLOAT_EQ(exp_row[val_idx], v<T>(crt_row[val_idx]));
      } else {
        CHECK_EQ(exp_row[val_idx], v<T>(crt_row[val_idx]));
      }
    }
  }
}

const char* trips_table_ddl = R"( 
CREATE TEMPORARY TABLE trips (
trip_id BIGINT,
vendor_id TEXT ENCODING NONE,
pickup_datetime TIMESTAMP,
dropoff_datetime TIMESTAMP,
store_and_fwd_flag TEXT ENCODING DICT,
rate_code_id BIGINT,
pickup_longitude DOUBLE,
pickup_latitude DOUBLE,
dropoff_longitude DOUBLE,
dropoff_latitude DOUBLE,
passenger_count BIGINT,
trip_distance DOUBLE,
fare_amount DOUBLE,
extra DOUBLE,
mta_tax DOUBLE,
tip_amount DOUBLE,
tolls_amount DOUBLE,
ehail_fee DOUBLE,
improvement_surcharge DOUBLE,
total_amount DOUBLE,
payment_type TEXT ENCODING DICT,
trip_type BIGINT,
pickup TEXT ENCODING DICT,
dropoff TEXT ENCODING NONE,
cab_type TEXT ENCODING DICT,
precipitation DOUBLE,
snow_depth BIGINT,
snowfall DOUBLE,
max_temperature BIGINT,
min_temperature BIGINT,
average_wind_speed DOUBLE,
pickup_nyct2010_gid BIGINT,
pickup_ctlabel DOUBLE,
pickup_borocode BIGINT,
pickup_boroname TEXT ENCODING NONE,
pickup_ct2010 BIGINT,
pickup_boroct2010 BIGINT,
pickup_cdeligibil TEXT ENCODING DICT,
pickup_ntacode TEXT ENCODING DICT,
pickup_ntaname TEXT ENCODING DICT,
pickup_puma BIGINT,
dropoff_nyct2010_gid BIGINT,
dropoff_ctlabel DOUBLE,
dropoff_borocode BIGINT,
dropoff_boroname TEXT ENCODING NONE,
dropoff_ct2010 BIGINT,
dropoff_boroct2010 BIGINT,
dropoff_cdeligibil TEXT ENCODING NONE,
dropoff_ntacode TEXT ENCODING NONE,
dropoff_ntaname TEXT ENCODING NONE,
dropoff_puma BIGINT) WITH (storage_type='CSV:../../Tests/Import/datafiles/trips_with_headers_top1000.csv', fragment_size=100);
)";

class NycTaxiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl_statement(trips_table_ddl));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
  }
};

TEST_F(NycTaxiTest, RunSimpleQuery) {
  // TODO: expect +1 rows when move to arrow 0.15 as current arrow doesn't support
  // headerless csv
  ASSERT_EQ(999,
            v<int64_t>(run_simple_agg(
                "SELECT count(vendor_id) FROM trips where vendor_id < '5'")));
}

TEST_F(NycTaxiTest, GroupByColumnWithNulls) {
  // TODO: expect +1 rows when move to arrow 0.15 as current arrow doesn't support
  // headerless csv
  ASSERT_EQ(
      619,
      v<int64_t>(run_simple_agg(
          " select count(*) from (select pickup, count(*) from trips group by pickup)")));
}

TEST_F(NycTaxiTest, CheckGroupBy) {
  check_query<NullableString>(
      "select pickup_ntaname from trips where pickup_ntaname IS NOT NULL group by "
      "pickup_ntaname order by pickup_ntaname limit 5;",
      {"Astoria",
       "Bedford Park-Fordham North",
       "Belmont",
       "Briarwood-Jamaica Hills",
       "Central Harlem North-Polo Grounds"});

  check_query<double>(
      "select tip_amount from trips group by tip_amount order by tip_amount limit 5;",
      {0, 0.01, 0.02, 0.03, 0.05});

  check_query<NullableString>(
      "select store_and_fwd_flag from trips group by store_and_fwd_flag order by "
      "store_and_fwd_flag limit 5;",
      {"N", "Y"});
}

TEST_F(NycTaxiTest, RunSelects) {
  check_query<int64_t>(
      "select rate_code_id from trips group by rate_code_id order by rate_code_id limit "
      "5;",
      {1, 2, 3, 4, 5});
}

TEST_F(NycTaxiTest, RunSelectsEncodingNoneNotNull) {
  check_query<NullableString>(
      "select dropoff_ntaname from trips where dropoff_ntaname is not NULL order by "
      "dropoff_ntaname limit 50;",
      {"Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Allerton-Pelham Gardens",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Battery Park City-Lower Manhattan",
       "Battery Park City-Lower Manhattan",
       "Battery Park City-Lower Manhattan",
       "Battery Park City-Lower Manhattan",
       "Battery Park City-Lower Manhattan",
       "Bedford Park-Fordham North",
       "Bedford Park-Fordham North",
       "Bellerose",
       "Belmont",
       "Belmont",
       "Belmont",
       "Belmont",
       "Belmont",
       "Borough Park",
       "Briarwood-Jamaica Hills",
       "Brooklyn Heights-Cobble Hill",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds"});
}

TEST_F(NycTaxiTest, RunSelectsEncodingNoneWhereGreater) {
  check_query<NullableString>(
      "select dropoff_ntaname from trips where dropoff_ntaname > "
      "'Queensbridge-Ravenswood-Long Island City' order by dropoff_ntaname limit 10;",
      {"Richmond Hill",
       "Richmond Hill",
       "Richmond Hill",
       "Richmond Hill",
       "Richmond Hill",
       "Richmond Hill",
       "Rosedale",
       "SoHo-TriBeCa-Civic Center-Little Italy",
       "SoHo-TriBeCa-Civic Center-Little Italy",
       "Soundview-Bruckner"});
}

TEST_F(NycTaxiTest, RunSelectsEncodingDictWhereGreater) {
  check_query<NullableString>(
      "select pickup_ntaname from trips where pickup_ntaname is not NULL and "
      "pickup_ntaname > 'Queensbridge-Ravenswood-Long Island City' order by "
      "pickup_ntaname limit 3;",
      {"Rego Park", "Richmond Hill", "Richmond Hill"});
}

TEST(Unsupported, Syntax) {
  run_ddl_statement("DROP TABLE IF EXISTS fsi_unsupported;");
  EXPECT_ANY_THROW(
      run_ddl_statement("CREATE TABLE fsi_unsupported (x INT, y DOUBLE) WITH "
                        "(storage_type='CSV:../../Tests/Import/datafiles/"
                        "trips_with_headers_top1000.csv');"));
}

TEST(DecimalDataTest, DifferentSizesOfDecimal) {
  run_ddl_statement(
      "CREATE DATAFRAME fsi_decimal (decimal2 DECIMAL(4,1), decimal4 NUMERIC(9,2), "
      "decimal8 DECIMAL(18,5)) from "
      "'CSV:../../Tests/Import/datafiles/decimal_data.csv';");
  check_table<double>("SELECT * FROM fsi_decimal",
                      {{4, 0, 1.1},
                       {213.4, 2389341.23, 4857364039384.75638},
                       {999.9, 9384612.78, 2947583746581.92748}});
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()("test-help",
                     "Print all ArrowCsvForeighStorageTest specific options (for gtest "
                     "options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: ArrowCsvForeighStorageTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  logger::init(log_options);

  QR::init(BASE_PATH);
  registerArrowCsvForeignStorage();
  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}