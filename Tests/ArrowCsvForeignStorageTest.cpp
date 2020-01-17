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

#include "TestHelpers.h"

#include <algorithm>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include "../Archive/PosixFileArchive.h"
#include "../Catalog/Catalog.h"
#include "../DataMgr/ForeignStorage/ArrowCsvForeignStorage.h"
#include "../Import/Importer.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ResultSet.h"
#include "../QueryRunner/QueryRunner.h"
#include "../Shared/geo_types.h"
#include "../Shared/scope.h"
#include "boost/filesystem.hpp"

#include <arrow/util/task-group.h>
#include <arrow/util/thread-pool.h>

#include <atomic>

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

const char* trips_table_ddl = R"( 
trips (
trip_id BIGINT,
vendor_id BIGINT,
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
payment_type BIGINT,
trip_type BIGINT,
pickup TEXT ENCODING DICT,
dropoff TEXT ENCODING DICT,
cab_type TEXT,
precipitation DOUBLE,
snow_depth BIGINT,
snowfall DOUBLE,
max_temperature BIGINT,
min_temperature BIGINT,
average_wind_speed DOUBLE,
pickup_nyct2010_gid BIGINT,
pickup_ctlabel DOUBLE,
pickup_borocode BIGINT,
pickup_boroname TEXT  ,
pickup_ct2010 BIGINT,
pickup_boroct2010 BIGINT,
pickup_cdeligibil TEXT ENCODING DICT,
pickup_ntacode TEXT ENCODING DICT,
pickup_ntaname TEXT ENCODING DICT,
pickup_puma BIGINT,
dropoff_nyct2010_gid BIGINT,
dropoff_ctlabel DOUBLE,
dropoff_borocode BIGINT,
dropoff_boroname TEXT ENCODING DICT,
dropoff_ct2010 BIGINT,
dropoff_boroct2010 BIGINT,
dropoff_cdeligibil TEXT ENCODING DICT,
dropoff_ntacode TEXT ENCODING DICT,
dropoff_ntaname TEXT ENCODING DICT,
dropoff_puma BIGINT) WITH (storage_type='CSV:/localdisk/artemale/scripts/trips_xaa.csv');
  )";

class NycTaxiTemporaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(
        run_ddl_statement("CREATE TEMPORARY TABLE " + std::string(trips_table_ddl)));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
  }
};

TEST_F(NycTaxiTemporaryTest, RunSimpleQuery) {}
TEST_F(NycTaxiTemporaryTest, RunSimpleQuery2) {}
TEST_F(NycTaxiTemporaryTest, RunSimpleQuery3) {}
TEST_F(NycTaxiTemporaryTest, RunSimpleQuery4) {}
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
