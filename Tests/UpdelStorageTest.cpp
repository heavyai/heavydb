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

#include <csignal>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include "boost/filesystem.hpp"
#include <boost/iterator/counting_iterator.hpp>

#include "Catalog/Catalog.h"
#include "Parser/parser.h"
#include "QueryEngine/ResultSet.h"
#include "Import/Importer.h"
#include "Shared/UpdelRoll.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 9093

using namespace std;
using namespace Catalog_Namespace;

namespace {

std::unique_ptr<SessionInfo> gsession;
bool g_hoist_literals{true};

std::shared_ptr<Calcite> g_calcite = nullptr;

void calcite_shutdown_handler() {
  if (g_calcite) {
    g_calcite->close_calcite_server();
  }
}

void mapd_signal_handler(int signal_number) {
  LOG(ERROR) << "Interrupt signal (" << signal_number << ") received.";
  calcite_shutdown_handler();
  // shut down logging force a flush
  google::ShutdownGoogleLogging();
  // terminate program
  if (signal_number == SIGTERM) {
    std::exit(EXIT_SUCCESS);
  } else {
    std::exit(signal_number);
  }
}

void register_signal_handler() {
  std::signal(SIGTERM, mapd_signal_handler);
  std::signal(SIGSEGV, mapd_signal_handler);
  std::signal(SIGABRT, mapd_signal_handler);
}

inline void run_ddl_statement(const string& input_str) {
  QueryRunner::run_ddl_statement(input_str, gsession);
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

std::shared_ptr<ResultSet> run_query(const string& query_str) {
  return QueryRunner::run_multiple_agg(query_str, gsession, ExecutorDeviceType::CPU, g_hoist_literals, true);
}

bool compare_agg(const string& table, const string& column, const int64_t cnt, const double avg) {
  std::string query_str = "SELECT COUNT(*), AVG(" + column + ") FROM " + table + ";";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  auto r_avg = v<double>(crt_row[1]);
  // std::cout << "r_avg: " << std::to_string(r_avg) << ", avg: " << std::to_string(avg);
  return r_cnt == cnt && fabs(r_avg - avg) < 1E-6;
}

template <typename T>
void update_prepare_offsets_values(const int64_t cnt,
                                   const int step,
                                   const T val,
                                   std::vector<uint64_t>& fragOffsets,
                                   std::vector<ScalarTargetValue>& rhsValues) {
  for (int64_t i = 0; i < cnt; i += step) {
    fragOffsets.push_back(i);
  }
  rhsValues.push_back(ScalarTargetValue(val));
}

template <typename T>
void update_common(const string& table,
                   const string& column,
                   const int64_t cnt,
                   const int step,
                   const T& val,
                   const SQLTypeInfo& rhsType,
                   const bool commit = true) {
  UpdelRoll updelRoll;
  std::vector<uint64_t> fragOffsets;
  std::vector<ScalarTargetValue> rhsValues;
  update_prepare_offsets_values<T>(cnt, step, val, fragOffsets, rhsValues);
  Fragmenter_Namespace::InsertOrderFragmenter::updateColumn(&gsession->get_catalog(),
                                                            table,
                                                            column,
                                                            0,  // 1st frag since we have only 100 rows
                                                            fragOffsets,
                                                            rhsValues,
                                                            rhsType,
                                                            Data_Namespace::MemoryLevel::CPU_LEVEL,
                                                            updelRoll);
  if (commit)
    updelRoll.commitUpdate();
}

bool update_a_numeric_column(const string& table,
                             const string& column,
                             const int64_t cnt,
                             const int step,
                             const double val,
                             const double avg,
                             const bool commit = true,
                             const bool by_string = false) {
  if (by_string)
    update_common<std::string>(table, column, cnt, step, std::to_string(val), SQLTypeInfo(), commit);
  else
    update_common<double>(table, column, cnt, step, val, SQLTypeInfo(), commit);
  return compare_agg(table, column, cnt, avg);
}

template <typename T>
bool nullize_a_fixed_encoded_column(const string& table, const string& column, const int64_t cnt) {
  update_common<int64_t>(table, column, cnt, 1, inline_int_null_value<T>(), SQLTypeInfo());
  std::string query_str = "SELECT count() FROM " + table + " WHERE " + column + " IS NULL;";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == cnt;
}

bool update_a_encoded_string_column(const string& table,
                                    const string& column,
                                    const int64_t cnt,
                                    const int step,
                                    const string& val,
                                    const bool commit = true) {
  update_common<const std::string>(table, column, cnt, step, val, SQLTypeInfo(), commit);
  // count updated string
  std::string query_str = "SELECT count() FROM " + table + " WHERE " + column + " = '" + val + "';";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == (commit ? cnt / step : 0);
}

#define update_a_datetime_column update_a_encoded_string_column

bool update_a_boolean_column(const string& table,
                             const string& column,
                             const int64_t cnt,
                             const int step,
                             const bool val,
                             const bool commit = true) {
  update_common<const std::string>(table, column, cnt, step, val ? "T" : "F", SQLTypeInfo(), commit);
  // count updated bools
  std::string query_str = "SELECT count() FROM " + table + " WHERE " + (val ? "" : " NOT ") + column + ";";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == (commit ? cnt / step : 0);
}

template <typename T>
bool update_column_from_decimal(const string& table,
                                const string& column,
                                const int64_t cnt,
                                const int64_t rhsDecimal,
                                const SQLTypeInfo& rhsType,
                                const SQLTypeInfo& lhsType,
                                const double max_loss,
                                const bool commit = true) {
  update_common<int64_t>(table, column, cnt, 1, rhsDecimal, rhsType, commit);
  std::string query_str = "SELECT DISTINCT " + column + " FROM " + table + ";";
  auto rows = run_query(query_str);
  CHECK_EQ(size_t(1), rows->rowCount());
  auto crt_row = rows->getNextRow(true, false);  // no decimal_to_double convert
  auto l_decimal = v<T>(crt_row[0]);
  int64_t r_decimal = rhsDecimal;

  if (lhsType.is_decimal())
    l_decimal = convert_decimal_value_to_scale(l_decimal, lhsType, rhsType);
  else
    l_decimal *= pow(10, rhsType.get_scale());

  auto r_loss = fabs(l_decimal - r_decimal) / r_decimal;
  if (!(r_loss <= max_loss))
    std::cout << "l_decimal: " << l_decimal << ", r_decimal: " << r_decimal << ", r_loss: " << r_loss
              << ", max_loss: " << max_loss << std::endl;
  return r_loss <= max_loss;
}

void import_table_file(const string& table, const string& file) {
  std::string query_str =
      string("COPY trips FROM '") + "../../Tests/Import/datafiles/" + file + "' WITH (header='true');";

  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed))
    throw std::runtime_error("Failed to parse: " + query_str);
  CHECK_EQ(parse_trees.size(), size_t(1));

  const auto& stmt = parse_trees.front();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
  if (!ddl)
    throw std::runtime_error("Not a DDLStmt: " + query_str);
  ddl->execute(*gsession);
}

class SQLTestEnv : public ::testing::Environment {
 public:
  virtual void SetUp() {
    boost::filesystem::path base_path{BASE_PATH};
    CHECK(boost::filesystem::exists(base_path));
    auto system_db_file = base_path / "mapd_catalogs" / MAPD_SYSTEM_DB;
    auto data_dir = base_path / "mapd_data";
    UserMetadata user;
    DBMetadata db;

    register_signal_handler();
    google::InstallFailureFunction(&calcite_shutdown_handler);

    g_calcite = std::make_shared<Calcite>(-1, CALCITEPORT, base_path.string(), 1024);
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
    // if no catalog create one
    auto& sys_cat = SysCatalog::instance();
    sys_cat.init(base_path.string(), dataMgr, {}, g_calcite, !boost::filesystem::exists(system_db_file), false);
    CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));
    // if no user create one
    if (!sys_cat.getMetadataForUser("gtest", user)) {
      sys_cat.createUser("gtest", "test!test!", false);
      CHECK(sys_cat.getMetadataForUser("gtest", user));
    }
    // if no db create one
    if (!sys_cat.getMetadataForDB("gtest_db", db)) {
      sys_cat.createDatabase("gtest_db", user.userId);
      CHECK(sys_cat.getMetadataForDB("gtest_db", db));
    }
    gsession.reset(new SessionInfo(
        std::make_shared<Catalog>(base_path.string(), db, dataMgr, std::vector<LeafHostInfo>{}, g_calcite),
        user,
        ExecutorDeviceType::GPU,
        ""));
  }
};

// don't use R"()" format; somehow it causes many blank lines
// to be output on console. how come?
const char* create_table_trips =
    "	CREATE TABLE trips ("
    "			medallion               TEXT ENCODING DICT,"
    "			hack_license            TEXT ENCODING DICT,"
    "			vendor_id               TEXT ENCODING DICT,"
    "			rate_code_id            SMALLINT ENCODING FIXED(8),"
    "			store_and_fwd_flag      TEXT ENCODING DICT,"
    "			pickup_datetime         TIMESTAMP,"
    "			dropoff_datetime        TIMESTAMP,"
    "			passenger_count         INTEGER ENCODING FIXED(16),"
    "			trip_time_in_secs       INTEGER,"
    "			trip_distance           FLOAT,"
    "			pickup_longitude        DECIMAL(14,7),"
    "			pickup_latitude         DECIMAL(14,7),"
    "			dropoff_longitude       DOUBLE,"
    "			dropoff_latitude        DECIMAL(19,5),"
    "			deleted                 BOOLEAN"
    "			) WITH (FRAGMENT_SIZE=75000000);";

void init_table_data(const string& table = "trips",
                     const string& create_table_cmd = create_table_trips,
                     const string& file = "trip_data_b.txt") {
  run_ddl_statement("drop table if exists " + table + ";");
  run_ddl_statement(create_table_cmd);
  if (file.size())
    import_table_file(table, file);
}

// It is currently not possible to do select query on temp table w/o MapDHandler. Perhaps in
// the future a thrift rpc like `push_table_details` can be added to enable such queries here...
#if 0
const char* create_temp_table = "CREATE TEMPORARY TABLE temp(i int) WITH (vacuum='delayed');";

class UpdateStorageTest_Temp : public ::testing::Test {
 protected:
  virtual void SetUp() {
    ASSERT_NO_THROW(init_table_data("temp", create_temp_table, ""););
    EXPECT_NO_THROW(run_query("insert into temp values(1)"););
    EXPECT_NO_THROW(run_query("insert into temp values(1)"););
  }

  virtual void TearDown() { ASSERT_NO_THROW(run_ddl_statement("drop table temp;");); }
};

TEST_F(UpdateStorageTest_Temp, Update_temp) {
  EXPECT_TRUE(update_a_numeric_column("temp", "i", 2, 1, 99, 99));
}

TEST_F(UpdateStorageTest_Temp, Update_temp_rollback) {
  EXPECT_TRUE(update_a_numeric_column("temp", "i", 2, 1, 99, 1, true));
}
#endif

class UpdateStorageTest : public ::testing::Test {
 protected:
  virtual void SetUp() { ASSERT_NO_THROW(init_table_data();); }

  virtual void TearDown() { ASSERT_NO_THROW(run_ddl_statement("drop table trips;");); }
};

TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_null_8) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int8_t>("trips", "rate_code_id", 100));
}
TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_null_16) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int16_t>("trips", "rate_code_id", 100));
}
TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_null_32_throw) {
  EXPECT_THROW(nullize_a_fixed_encoded_column<int32_t>("trips", "rate_code_id", 100), std::runtime_error);
}
TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_null_64_throw) {
  EXPECT_THROW(nullize_a_fixed_encoded_column<int64_t>("trips", "rate_code_id", 100), std::runtime_error);
}

TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_null_16) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int16_t>("trips", "passenger_count", 100));
}
TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_null_32) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int32_t>("trips", "passenger_count", 100));
}
TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_null_64_throw) {
  EXPECT_THROW(nullize_a_fixed_encoded_column<int64_t>("trips", "passenger_count", 100), std::runtime_error);
}

TEST_F(UpdateStorageTest, All_integer_trip_time_in_secs_null_32) {
  EXPECT_TRUE(nullize_a_fixed_encoded_column<int32_t>("trips", "trip_time_in_secs", 100));
}
TEST_F(UpdateStorageTest, All_integer_trip_time_in_secs_null_64_throw) {
  EXPECT_THROW(nullize_a_fixed_encoded_column<int64_t>("trips", "trip_time_in_secs", 100), std::runtime_error);
}

TEST_F(UpdateStorageTest, All_fixed_encoded_smallint_rate_code_id_throw) {
  EXPECT_TRUE(update_a_numeric_column("trips", "rate_code_id", 100, 1, 44 * 2, 44 * 2.0));
  EXPECT_THROW(update_a_numeric_column("trips", "rate_code_id", 100, 1, +257, +257), std::runtime_error);
  EXPECT_THROW(update_a_numeric_column("trips", "rate_code_id", 100, 1, -256, -256), std::runtime_error);
}

#define SQLTypeInfo_dropoff_latitude SQLTypeInfo(kDECIMAL, 19, 5, false)
TEST_F(UpdateStorageTest, All_RHS_decimal_10_0_LHS_decimal_19_5) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "dropoff_latitude",
                                                  100,
                                                  1234506789,
                                                  SQLTypeInfo(kDECIMAL, 10, 0, false),
                                                  SQLTypeInfo_dropoff_latitude,
                                                  0));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_10_2_LHS_decimal_19_5) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "dropoff_latitude",
                                                  100,
                                                  1234506789,
                                                  SQLTypeInfo(kDECIMAL, 10, 2, false),
                                                  SQLTypeInfo_dropoff_latitude,
                                                  0));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_15_2_LHS_decimal_19_5) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "dropoff_latitude",
                                                  100,
                                                  123456789012345,
                                                  SQLTypeInfo(kDECIMAL, 15, 2, false),
                                                  SQLTypeInfo_dropoff_latitude,
                                                  0));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_17_2_LHS_decimal_19_5_throw) {
  EXPECT_THROW(update_column_from_decimal<int64_t>("trips",
                                                   "dropoff_latitude",
                                                   100,
                                                   12345678901234567,
                                                   SQLTypeInfo(kDECIMAL, 17, 2, false),
                                                   SQLTypeInfo_dropoff_latitude,
                                                   0),
               std::runtime_error);
}

#define SQLTypeInfo_dropoff_longitude SQLTypeInfo(kDOUBLE, false)
TEST_F(UpdateStorageTest, All_RHS_decimal_10_0_LHS_double) {
  EXPECT_TRUE(update_column_from_decimal<double>("trips",
                                                 "dropoff_longitude",
                                                 100,
                                                 1234506789,
                                                 SQLTypeInfo(kDECIMAL, 10, 0, false),
                                                 SQLTypeInfo_dropoff_longitude,
                                                 1E-15));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_10_2_LHS_double) {
  EXPECT_TRUE(update_column_from_decimal<double>("trips",
                                                 "dropoff_longitude",
                                                 100,
                                                 1234506789,
                                                 SQLTypeInfo(kDECIMAL, 10, 2, false),
                                                 SQLTypeInfo_dropoff_longitude,
                                                 1E-15));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_15_2_LHS_double) {
  EXPECT_TRUE(update_column_from_decimal<double>("trips",
                                                 "dropoff_longitude",
                                                 100,
                                                 123456789012345,
                                                 SQLTypeInfo(kDECIMAL, 15, 2, false),
                                                 SQLTypeInfo_dropoff_longitude,
                                                 1E-15));
}

#define SQLTypeInfo_trip_time_in_secs SQLTypeInfo(kINT, false)
TEST_F(UpdateStorageTest, All_RHS_decimal_10_0_LHS_integer) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "trip_time_in_secs",
                                                  100,
                                                  1234506789,
                                                  SQLTypeInfo(kDECIMAL, 10, 0, false),
                                                  SQLTypeInfo_trip_time_in_secs,
                                                  1));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_10_2_LHS_integer) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "trip_time_in_secs",
                                                  100,
                                                  1234506789,
                                                  SQLTypeInfo(kDECIMAL, 10, 2, false),
                                                  SQLTypeInfo_trip_time_in_secs,
                                                  1));
}
TEST_F(UpdateStorageTest, All_RHS_decimal_15_5_LHS_integer) {
  EXPECT_TRUE(update_column_from_decimal<int64_t>("trips",
                                                  "trip_time_in_secs",
                                                  100,
                                                  123456789012345,
                                                  SQLTypeInfo(kDECIMAL, 15, 5, false),
                                                  SQLTypeInfo_trip_time_in_secs,
                                                  1));
}

TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_x1_throw) {
  EXPECT_TRUE(update_a_numeric_column("trips", "passenger_count", 100, 1, 4 * 2, 4 * 2.0));
  EXPECT_THROW(update_a_numeric_column("trips", "passenger_count", 100, 1, +65537, +65537), std::runtime_error);
  EXPECT_THROW(update_a_numeric_column("trips", "passenger_count", 100, 1, -65536, -65536), std::runtime_error);
}
TEST_F(UpdateStorageTest, All_fixed_encoded_integer_passenger_count_x1_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips", "passenger_count", 100, 1, 4 * 2, 4 * 1.0, false));
}

TEST_F(UpdateStorageTest, Half_fixed_encoded_integer_passenger_count_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips", "passenger_count", 100, 2, 4 * 2, 4. * 1.5));
}
TEST_F(UpdateStorageTest, Half_fixed_encoded_integer_passenger_count_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips", "passenger_count", 100, 2, 4 * 2, 4. * 1.0, false));
}

TEST_F(UpdateStorageTest, All_int_trip_time_in_secs_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips", "trip_time_in_secs", 100, 1, 382 * 2, 382 * 2.0));
}
TEST_F(UpdateStorageTest, All_int_trip_time_in_secs_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips", "trip_time_in_secs", 100, 1, 382 * 2, 382 * 1.0, false));
}

TEST_F(UpdateStorageTest, Half_int_trip_time_in_secs_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips", "trip_time_in_secs", 100, 2, 382 * 2, 382. * 1.5));
}
TEST_F(UpdateStorageTest, Half_int_trip_time_in_secs_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips", "trip_time_in_secs", 100, 2, 382 * 2, 382. * 1.0, false));
}

TEST_F(UpdateStorageTest, All_float_trip_distance_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips", "trip_distance", 100, 1, 1 * 2, 1 * 2.0));
}
TEST_F(UpdateStorageTest, All_float_trip_distance_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips", "trip_distance", 100, 1, 1 * 2, 1 * 1.0, false));
}

TEST_F(UpdateStorageTest, Half_float_trip_distance_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips", "trip_distance", 100, 2, 1 * 2, 1. * 1.5));
}
TEST_F(UpdateStorageTest, Half_float_trip_distance_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips", "trip_distance", 100, 2, 1 * 2, 1. * 1.0, false));
}

TEST_F(UpdateStorageTest, All_decimal_pickup_longitude_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips", "pickup_longitude", 100, 1, -73.978165 * 2, -73.978165 * 2.0));
}
TEST_F(UpdateStorageTest, All_decimal_pickup_longitude_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips", "pickup_longitude", 100, 1, -73.978165 * 2, -73.978165 * 1.0, false));
}

TEST_F(UpdateStorageTest, Half_decimal_pickup_longitude_x2) {
  EXPECT_TRUE(update_a_numeric_column("trips", "pickup_longitude", 100, 2, -73.978165 * 2, -73.978165 * 1.5));
}
TEST_F(UpdateStorageTest, Half_decimal_pickup_longitude_x2_rollback) {
  EXPECT_TRUE(update_a_numeric_column("trips", "pickup_longitude", 100, 2, -73.978165 * 2, -73.978165 * 1.0, false));
}
TEST_F(UpdateStorageTest, Half_decimal_pickup_longitude_x2_by_string) {
  EXPECT_TRUE(
      update_a_numeric_column("trips", "pickup_longitude", 100, 2, -73.978165 * 2, -73.978165 * 1.5, true, true));
}
TEST_F(UpdateStorageTest, Half_decimal_pickup_longitude_x2_rollback_by_string) {
  EXPECT_TRUE(
      update_a_numeric_column("trips", "pickup_longitude", 100, 2, -73.978165 * 2, -73.978165 * 1.0, false, true));
}

TEST_F(UpdateStorageTest, All_string_vendor_id) {
  EXPECT_TRUE(update_a_encoded_string_column("trips", "vendor_id", 100, 1, "abcxyz"));
}
TEST_F(UpdateStorageTest, All_string_vendor_id_rollback) {
  EXPECT_TRUE(update_a_encoded_string_column("trips", "vendor_id", 100, 1, "abcxyz", false));
}

TEST_F(UpdateStorageTest, Half_string_vendor_id) {
  EXPECT_TRUE(update_a_encoded_string_column("trips", "vendor_id", 100, 2, "xyzabc"));
}
TEST_F(UpdateStorageTest, Half_string_vendor_id_rollback) {
  EXPECT_TRUE(update_a_encoded_string_column("trips", "vendor_id", 100, 2, "xyzabc", false));
}

TEST_F(UpdateStorageTest, Half_datetime_pickup_datetime_id) {
  EXPECT_TRUE(update_a_datetime_column("trips", "pickup_datetime", 100, 2, "2018-04-06 18:27:59"));
}

TEST_F(UpdateStorageTest, All_boolean_deleted) {
  EXPECT_TRUE(update_a_boolean_column("trips", "deleted", 100, 1, true));
}
TEST_F(UpdateStorageTest, All_boolean_deleted_rollback) {
  EXPECT_TRUE(update_a_boolean_column("trips", "deleted", 100, 1, true, false));
}

TEST_F(UpdateStorageTest, Half_boolean_deleted) {
  EXPECT_TRUE(update_a_boolean_column("trips", "deleted", 100, 2, true));
}
TEST_F(UpdateStorageTest, Half_boolean_deleted_rollback) {
  EXPECT_TRUE(update_a_boolean_column("trips", "deleted", 100, 2, true, false));
}

const char* create_table_times =
    "	CREATE TABLE times ("
    "			t_date      DATE,"
    "			t_time      TIME,"
    "			t_datetime  TIMESTAMP,"
    "			t_idx       INTEGER"
    "			);";

TEST(UpdateStorageTest_Times, Update_times) {
  EXPECT_NO_THROW(init_table_data("times", create_table_times, ""););
  EXPECT_NO_THROW(run_query("insert into times values('2000-1-1', '01:01:01', '2000-1-1 01:01:01', 1)"););
  EXPECT_TRUE(update_a_datetime_column("times", "t_date", 1, 1, "2018-1-1"));
  EXPECT_TRUE(update_a_datetime_column("times", "t_time", 1, 1, "18:01:01"));
  EXPECT_TRUE(update_a_datetime_column("times", "t_datetime", 1, 1, "2018-1-1 18:01:01"));
  EXPECT_NO_THROW(run_ddl_statement("drop table times;"););
}
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SQLTestEnv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
