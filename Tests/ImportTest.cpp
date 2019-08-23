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

#include "../Import/Importer.h"

#include <algorithm>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include "../Catalog/Catalog.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ResultSet.h"
#include "../QueryRunner/QueryRunner.h"
#include "../Shared/geo_types.h"
#include "boost/filesystem.hpp"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std;
using namespace TestHelpers;

extern bool g_use_date_in_days_default_encoding;

extern size_t g_leaf_count;

namespace {

bool g_aggregator{false};
size_t g_num_leafs{1};

#define SKIP_ALL_ON_AGGREGATOR()                         \
  if (g_aggregator) {                                    \
    LOG(ERROR) << "Tests not valid in distributed mode"; \
    return;                                              \
  }

bool g_hoist_literals{true};

using QR = QueryRunner::QueryRunner;

inline void run_ddl_statement(const string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

std::shared_ptr<ResultSet> run_query(const string& query_str) {
  return QR::get()->runSQL(query_str, ExecutorDeviceType::CPU, g_hoist_literals);
}

bool compare_agg(const int64_t cnt, const double avg) {
  std::string query_str = "SELECT COUNT(*), AVG(trip_distance) FROM trips;";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  auto r_avg = v<double>(crt_row[1]);
  if (!(r_cnt == cnt && fabs(r_avg - avg) < 1E-9)) {
    LOG(ERROR) << "error: " << r_cnt << ":" << cnt << ", " << r_avg << ":" << avg;
  }
  return r_cnt == cnt && fabs(r_avg - avg) < 1E-9;
}

#ifdef ENABLE_IMPORT_PARQUET
bool import_test_parquet_with_null(const int64_t cnt) {
  std::string query_str = "select count(*) from trips where rate_code_id is null;";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == cnt;
}
#endif

bool import_test_common(const string& query_str, const int64_t cnt, const double avg) {
  run_ddl_statement(query_str);
  return compare_agg(cnt, avg);
}

bool import_test_common_geo(const string& query_str,
                            const std::string& table,
                            const int64_t cnt,
                            const double avg) {
  // TODO(adb): Return ddl from QueryRunner::run_ddl_statement and use that
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed)) {
    return false;
  }
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::CopyTableStmt* ddl = dynamic_cast<Parser::CopyTableStmt*>(stmt.get());
  if (!ddl) {
    return false;
  }
  ddl->execute(*QR::get()->getSession());

  // was it a geo copy from?
  bool was_geo_copy_from = ddl->was_geo_copy_from();
  if (!was_geo_copy_from) {
    return false;
  }

  // get the rest of the payload
  std::string geo_copy_from_table, geo_copy_from_file_name, geo_copy_from_partitions;
  Importer_NS::CopyParams geo_copy_from_copy_params;
  ddl->get_geo_copy_from_payload(geo_copy_from_table,
                                 geo_copy_from_file_name,
                                 geo_copy_from_copy_params,
                                 geo_copy_from_partitions);

  // was it the right table?
  if (geo_copy_from_table != "geo") {
    return false;
  }

  // @TODO simon.eves
  // test other stuff
  // filename
  // CopyParams contents

  // success
  return true;
}

void import_test_geofile_importer(const std::string& file_str,
                                  const std::string& table_name,
                                  const bool compression,
                                  const bool create_table = true) {
  Importer_NS::ImportDriver import_driver(
      QR::get()->getCatalog(),
      QueryRunner::get_user_metadata(QR::get()->getSession()),
      ExecutorDeviceType::CPU);

  auto file_path = boost::filesystem::path("../../Tests/Import/datafiles/" + file_str);

  ASSERT_TRUE(boost::filesystem::exists(file_path));

  ASSERT_NO_THROW(import_driver.importGeoTable(
      file_path.string(), table_name, compression, create_table));
}

bool import_test_local(const string& filename, const int64_t cnt, const double avg) {
  return import_test_common(
      string("COPY trips FROM '") + "../../Tests/Import/datafiles/" + filename +
          "' WITH (header='true'" +
          (filename.find(".parquet") != std::string::npos ? ",parquet='true'" : "") +
          ");",
      cnt,
      avg);
}

bool import_test_line_endings_in_quotes_local(const string& filename, const int64_t cnt) {
  string query_str =
      "COPY random_strings_with_line_endings FROM '../../Tests/Import/datafiles/" +
      filename +
      "' WITH (header='false', quoted='true', max_reject=1, buffer_size=1048576);";
  run_ddl_statement(query_str);
  std::string select_query_str = "SELECT COUNT(*) FROM random_strings_with_line_endings;";
  auto rows = run_query(select_query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == cnt;
}

bool import_test_local_geo(const string& filename,
                           const string& other_options,
                           const int64_t cnt,
                           const double avg) {
  return import_test_common_geo(string("COPY geo FROM '") +
                                    "../../Tests/Import/datafiles/" + filename +
                                    "' WITH (geo='true'" + other_options + ");",
                                "geo",
                                cnt,
                                avg);
}

#ifdef HAVE_AWS_S3
bool import_test_s3(const string& prefix,
                    const string& filename,
                    const int64_t cnt,
                    const double avg) {
  // unlikely we will expose any credentials in clear text here.
  // likely credentials will be passed as the "tester"'s env.
  // though s3 sdk should by default access the env, if any,
  // we still read them out to test coverage of the code
  // that passes credentials on per user basis.
  char* env;
  std::string s3_region, s3_access_key, s3_secret_key;
  if (0 != (env = getenv("AWS_REGION"))) {
    s3_region = env;
  }
  if (0 != (env = getenv("AWS_ACCESS_KEY_ID"))) {
    s3_access_key = env;
  }
  if (0 != (env = getenv("AWS_SECRET_ACCESS_KEY"))) {
    s3_secret_key = env;
  }

  return import_test_common(
      string("COPY trips FROM '") + "s3://mapd-parquet-testdata/" + prefix + "/" +
          filename + "' WITH (header='true'" +
          (s3_access_key.size() ? ",s3_access_key='" + s3_access_key + "'" : "") +
          (s3_secret_key.size() ? ",s3_secret_key='" + s3_secret_key + "'" : "") +
          (s3_region.size() ? ",s3_region='" + s3_region + "'" : "") +
          (prefix.find(".parquet") != std::string::npos ||
                   filename.find(".parquet") != std::string::npos
               ? ",parquet='true'"
               : "") +
          ");",
      cnt,
      avg);
}

bool import_test_s3_compressed(const string& filename,
                               const int64_t cnt,
                               const double avg) {
  return import_test_s3("trip.compressed", filename, cnt, avg);
}
#endif  // HAVE_AWS_S3

#ifdef ENABLE_IMPORT_PARQUET
bool import_test_local_parquet(const string& prefix,
                               const string& filename,
                               const int64_t cnt,
                               const double avg) {
  return import_test_local(prefix + "/" + filename, cnt, avg);
}
#ifdef HAVE_AWS_S3
bool import_test_s3_parquet(const string& prefix,
                            const string& filename,
                            const int64_t cnt,
                            const double avg) {
  return import_test_s3(prefix, filename, cnt, avg);
}
#endif  // HAVE_AWS_S3
#endif  // ENABLE_IMPORT_PARQUET

#ifdef ENABLE_IMPORT_PARQUET
bool import_test_local_parquet_with_geo_point(const string& prefix,
                                              const string& filename,
                                              const int64_t cnt,
                                              const double avg) {
  run_ddl_statement("alter table trips add column pt_dropoff point;");
  EXPECT_TRUE(import_test_local_parquet(prefix, filename, cnt, avg));
  std::string query_str =
      "select count(*) from trips where abs(dropoff_longitude-st_x(pt_dropoff))<0.01 and "
      "abs(dropoff_latitude-st_y(pt_dropoff))<0.01;";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == cnt;
}
#endif  // ENABLE_IMPORT_PARQUET

std::string TypeToString(SQLTypes type) {
  return SQLTypeInfo(type, false).get_type_name();
}

void d(const SQLTypes expected_type, const std::string& str) {
  auto detected_type = Importer_NS::Detector::detect_sqltype(str);
  EXPECT_EQ(TypeToString(expected_type), TypeToString(detected_type))
      << "String: " << str;
}

TEST(Detect, DateTime) {
  d(kDATE, "2016-01-02");
  d(kDATE, "02/01/2016");
  d(kDATE, "01-Feb-16");
  d(kDATE, "01/Feb/2016");
  d(kDATE, "01/Feb/16");
  d(kTIMESTAMP, "2016-01-02T03:04");
  d(kTIMESTAMP, "2016-01-02T030405");
  d(kTIMESTAMP, "2016-01-02T03:04:05");
  d(kTIMESTAMP, "1776-01-02T03:04:05");
  d(kTIMESTAMP, "9999-01-02T03:04:05");
  d(kTIME, "03:04");
  d(kTIME, "03:04:05");
  d(kTEXT, "33:04");
}

TEST(Detect, Numeric) {
  d(kSMALLINT, "1");
  d(kSMALLINT, "12345");
  d(kINT, "123456");
  d(kINT, "1234567890");
  d(kBIGINT, "12345678901");
  d(kFLOAT, "1.");
  d(kFLOAT, "1.2345678");
  // d(kDOUBLE, "1.2345678901");
  // d(kDOUBLE, "1.23456789012345678901234567890");
  d(kTEXT, "1.22.22");
}

const char* create_table_mini_sort = R"(
  CREATE TABLE sortab(
    i int,
    f float,
    ia int[2],
    pt point,
    sa text[],
    s2 text encoding dict(16),
    dt date,
    d2 date ENCODING FIXED(16),
    tm timestamp,
    t4 timestamp ENCODING FIXED(32),
    va int[])
  )";

class ImportTestMiniSort : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists sortab;"));
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists sortctas;"));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists sortab;"));
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists sortctas;"));
  }
};

void create_minisort_table_on_column(const std::string& column_name) {
  ASSERT_NO_THROW(run_ddl_statement(
      std::string(create_table_mini_sort) +
      (column_name.size() ? " with (sort_column='" + column_name + "');" : ";")));
  EXPECT_NO_THROW(
      run_ddl_statement("copy sortab from '../../Tests/Import/datafiles/mini_sort.txt' "
                        "with (header='false');"));
}

void check_minisort_on_expects(const std::string& table_name,
                               const std::vector<int>& expects) {
  auto rows = run_query("SELECT i FROM " + table_name + ";");
  CHECK_EQ(expects.size(), rows->rowCount());
  for (auto exp : expects) {
    auto crt_row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(1), crt_row.size());
    CHECK_EQ(int64_t(exp), v<int64_t>(crt_row[0]));
  }
}

void test_minisort_on_column(const std::string& column_name,
                             const std::vector<int> expects) {
  create_minisort_table_on_column(column_name);
  check_minisort_on_expects("sortab", expects);
}

void create_minisort_table_on_column_with_ctas(const std::string& column_name) {
  EXPECT_NO_THROW(
      run_ddl_statement(
          "create table sortctas as select * from sortab" +
          (column_name.size() ? " with (sort_column='" + column_name + "');" : ";")););
}

void test_minisort_on_column_with_ctas(const std::string& column_name,
                                       const std::vector<int> expects) {
  create_minisort_table_on_column("");
  create_minisort_table_on_column_with_ctas(column_name);
  check_minisort_on_expects("sortctas", expects);
}

TEST_F(ImportTestMiniSort, on_none) {
  test_minisort_on_column("", {5, 3, 1, 2, 4});
}

TEST_F(ImportTestMiniSort, on_int) {
  test_minisort_on_column("i", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, on_float) {
  test_minisort_on_column("f", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, on_int_array) {
  test_minisort_on_column("ia", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, on_string_array) {
  test_minisort_on_column("sa", {5, 3, 1, 2, 4});
}

TEST_F(ImportTestMiniSort, on_string_2b) {
  test_minisort_on_column("s2", {5, 3, 1, 2, 4});
}

TEST_F(ImportTestMiniSort, on_date) {
  test_minisort_on_column("dt", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, on_date_2b) {
  test_minisort_on_column("d2", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, on_time) {
  test_minisort_on_column("tm", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, on_time_4b) {
  test_minisort_on_column("t4", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, on_varlen_array) {
  test_minisort_on_column("va", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, on_geo_point) {
  test_minisort_on_column("pt", {2, 3, 4, 5, 1});
}

TEST_F(ImportTestMiniSort, ctas_on_none) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("", {5, 3, 1, 2, 4});
}

TEST_F(ImportTestMiniSort, ctas_on_int) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("i", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, ctas_on_float) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("f", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, ctas_on_int_array) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("ia", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, ctas_on_string_array) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("sa", {5, 3, 1, 2, 4});
}

TEST_F(ImportTestMiniSort, ctas_on_string_2b) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("s2", {5, 3, 1, 2, 4});
}

TEST_F(ImportTestMiniSort, ctas_on_date) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("dt", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, ctas_on_date_2b) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("d2", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, ctas_on_time) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("tm", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, ctas_on_time_4b) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("t4", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, ctas_on_varlen_array) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("va", {1, 2, 3, 4, 5});
}

TEST_F(ImportTestMiniSort, ctas_on_geo_point) {
  SKIP_ALL_ON_AGGREGATOR();
  test_minisort_on_column_with_ctas("pt", {2, 3, 4, 5, 1});
}

const char* create_table_mixed_varlen = R"(
    CREATE TABLE import_test_mixed_varlen(
      pt GEOMETRY(POINT),
      ls GEOMETRY(LINESTRING),
      faii INTEGER[2],
      fadc DECIMAL(5,2)[2],
      fatx TEXT[] ENCODING DICT(32),
      fatx2 TEXT[2] ENCODING DICT(32)
    );
  )";

class ImportTestMixedVarlen : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_mixed_varlen;"));
    ASSERT_NO_THROW(run_ddl_statement(create_table_mixed_varlen););
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_mixed_varlen;"));
  }
};

TEST_F(ImportTestMixedVarlen, Fix_failed_import_arrays_after_geos) {
  EXPECT_NO_THROW(
      run_ddl_statement("copy import_test_mixed_varlen from "
                        "'../../Tests/Import/datafiles/mixed_varlen.txt' with "
                        "(header='false');"));
  std::string query_str = "SELECT COUNT(*) FROM import_test_mixed_varlen;";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  CHECK_EQ(int64_t(1), v<int64_t>(crt_row[0]));
}

const char* create_table_date = R"(
    CREATE TABLE import_test_date(
      date_text TEXT ENCODING DICT(32),
      date_date DATE,
      date_date_not_null DATE NOT NULL,
      date_i32 DATE ENCODING FIXED(32),
      date_i16 DATE ENCODING FIXED(16)
    );
)";

class ImportTestDate : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_date;"));
    ASSERT_NO_THROW(run_ddl_statement(create_table_date));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_date;"));
  }
};

std::string convert_date_to_string(int64_t d) {
  if (d == std::numeric_limits<int64_t>::min()) {
    return std::string("NULL");
  }
  const auto date = static_cast<time_t>(d);
  std::tm tm_struct;
  gmtime_r(&date, &tm_struct);
  char buf[11];
  strftime(buf, 11, "%F", &tm_struct);
  return std::string(buf);
}

inline void run_mixed_dates_test() {
  ASSERT_NO_THROW(run_ddl_statement(
      "COPY import_test_date FROM '../../Tests/Import/datafiles/mixed_dates.txt';"));

  auto rows = run_query("SELECT * FROM import_test_date;");
  ASSERT_EQ(size_t(11), rows->entryCount());
  for (size_t i = 0; i < 10; i++) {
    const auto crt_row = rows->getNextRow(true, true);
    ASSERT_EQ(size_t(5), crt_row.size());
    const auto date_truth_str_nullable = v<NullableString>(crt_row[0]);
    const auto date_truth_str = boost::get<std::string>(&date_truth_str_nullable);
    CHECK(date_truth_str);
    for (size_t j = 1; j < crt_row.size(); j++) {
      const auto date = v<int64_t>(crt_row[j]);
      const auto date_str = convert_date_to_string(static_cast<int64_t>(date));
      ASSERT_EQ(*date_truth_str, date_str);
    }
  }

  // Last row is NULL (except for column 2 which is NOT NULL)
  const auto crt_row = rows->getNextRow(true, true);
  ASSERT_EQ(size_t(5), crt_row.size());
  for (size_t j = 1; j < crt_row.size(); j++) {
    if (j == 2) {
      continue;
    }
    const auto date_null = v<int64_t>(crt_row[j]);
    ASSERT_EQ(date_null, std::numeric_limits<int64_t>::min());
  }
}

TEST_F(ImportTestDate, ImportMixedDates) {
  SKIP_ALL_ON_AGGREGATOR();  // global variable not available on leaf nodes
  run_mixed_dates_test();
}

class ImportTestLegacyDate : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_date;"));
    g_use_date_in_days_default_encoding = false;
    ASSERT_NO_THROW(run_ddl_statement(create_table_date));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_date;"));
    g_use_date_in_days_default_encoding = true;
  }
};

TEST_F(ImportTestLegacyDate, ImportMixedDates) {
  SKIP_ALL_ON_AGGREGATOR();  // global variable not available on leaf nodes
  run_mixed_dates_test();
}

const char* create_table_date_arr = R"(
    CREATE TABLE import_test_date_arr(
      date_text TEXT[],
      date_date DATE[],
      date_date_fixed DATE[2],
      date_date_not_null DATE[] NOT NULL
    );
)";

class ImportTestDateArray : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_date_arr;"));
    ASSERT_NO_THROW(run_ddl_statement(create_table_date_arr));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_date_arr;"));
  }
};

void decode_str_array(const TargetValue& r, std::vector<std::string>& arr) {
  const auto atv = boost::get<ArrayTargetValue>(&r);
  CHECK(atv);
  if (!atv->is_initialized()) {
    return;
  }
  const auto& vec = atv->get();
  for (const auto& stv : vec) {
    const auto ns = v<NullableString>(stv);
    const auto str = boost::get<std::string>(&ns);
    CHECK(str);
    arr.push_back(*str);
  }
  CHECK_EQ(arr.size(), vec.size());
}

TEST_F(ImportTestDateArray, ImportMixedDateArrays) {
  EXPECT_NO_THROW(
      run_ddl_statement("COPY import_test_date_arr FROM "
                        "'../../Tests/Import/datafiles/mixed_date_arrays.txt';"));

  auto rows = run_query("SELECT * FROM import_test_date_arr;");
  ASSERT_EQ(size_t(10), rows->entryCount());
  for (size_t i = 0; i < 3; i++) {
    const auto crt_row = rows->getNextRow(true, true);
    ASSERT_EQ(size_t(4), crt_row.size());
    std::vector<std::string> truth_arr;
    decode_str_array(crt_row[0], truth_arr);
    for (size_t j = 1; j < crt_row.size(); j++) {
      const auto date_arr = boost::get<ArrayTargetValue>(&crt_row[j]);
      CHECK(date_arr && date_arr->is_initialized());
      const auto& vec = date_arr->get();
      for (size_t k = 0; k < vec.size(); k++) {
        const auto date = v<int64_t>(vec[k]);
        const auto date_str = convert_date_to_string(static_cast<int64_t>(date));
        ASSERT_EQ(truth_arr[k], date_str);
      }
    }
  }
  // Date arrays with NULL dates
  for (size_t i = 3; i < 6; i++) {
    const auto crt_row = rows->getNextRow(true, true);
    ASSERT_EQ(size_t(4), crt_row.size());
    std::vector<std::string> truth_arr;
    decode_str_array(crt_row[0], truth_arr);
    for (size_t j = 1; j < crt_row.size() - 1; j++) {
      const auto date_arr = boost::get<ArrayTargetValue>(&crt_row[j]);
      CHECK(date_arr && date_arr->is_initialized());
      const auto& vec = date_arr->get();
      for (size_t k = 0; k < vec.size(); k++) {
        const auto date = v<int64_t>(vec[k]);
        const auto date_str = convert_date_to_string(static_cast<int64_t>(date));
        ASSERT_EQ(truth_arr[k], date_str);
      }
    }
  }
  // NULL date arrays, empty date arrays, NULL fixed date arrays
  for (size_t i = 6; i < rows->entryCount(); i++) {
    const auto crt_row = rows->getNextRow(true, true);
    ASSERT_EQ(size_t(4), crt_row.size());
    const auto date_arr1 = boost::get<ArrayTargetValue>(&crt_row[1]);
    CHECK(date_arr1);
    if (i == 9) {
      // Empty date array
      CHECK(date_arr1->is_initialized());
      const auto& vec = date_arr1->get();
      ASSERT_EQ(size_t(0), vec.size());
    } else {
      // NULL array
      CHECK(!date_arr1->is_initialized());
    }
    const auto date_arr2 = boost::get<ArrayTargetValue>(&crt_row[2]);
    CHECK(date_arr2);
    if (i == 9) {
      // Fixlen array - not NULL, filled with NULLs
      CHECK(date_arr2->is_initialized());
      const auto& vec = date_arr2->get();
      for (size_t k = 0; k < vec.size(); k++) {
        const auto date = v<int64_t>(vec[k]);
        const auto date_str = convert_date_to_string(static_cast<int64_t>(date));
        ASSERT_EQ("NULL", date_str);
      }
    } else {
      // NULL fixlen array
      CHECK(!date_arr2->is_initialized());
    }
  }
}

const char* create_table_timestamps = R"(
    CREATE TABLE import_test_timestamps(
      ts0_text TEXT ENCODING DICT(32),
      ts3_text TEXT ENCODING DICT(32),
      ts6_text TEXT ENCODING DICT(32),
      ts9_text TEXT ENCODING DICT(32),
      ts_0 TIMESTAMP(0),
      ts_0_i32 TIMESTAMP ENCODING FIXED(32),
      ts_0_not_null TIMESTAMP NOT NULL,
      ts_3 TIMESTAMP(3),
      ts_3_not_null TIMESTAMP(3) NOT NULL,
      ts_6 TIMESTAMP(6),
      ts_6_not_null TIMESTAMP(6) NOT NULL,
      ts_9 TIMESTAMP(9),
      ts_9_not_null TIMESTAMP(9) NOT NULL
    );
)";

class ImportTestTimestamps : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_timestamps;"));
    ASSERT_NO_THROW(run_ddl_statement(create_table_timestamps));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists import_test_date;"));
  }
};

std::string convert_timestamp_to_string(const time_t timeval, const int dimen) {
  std::tm tm_struct;
  if (dimen > 0) {
    auto scale = static_cast<int64_t>(std::pow(10, dimen));
    auto dv = std::div(static_cast<int64_t>(timeval), scale);
    auto modulus = (dv.rem + scale) % scale;
    time_t sec = dv.quot - (dv.quot < 0 && modulus > 0);
    gmtime_r(&sec, &tm_struct);
    char buf[21];
    strftime(buf, 21, "%F %T.", &tm_struct);
    auto subsecond = std::to_string(modulus);
    return std::string(buf) + std::string(dimen - subsecond.length(), '0') + subsecond;
  } else {
    time_t sec = timeval;
    gmtime_r(&sec, &tm_struct);
    char buf[20];
    strftime(buf, 20, "%F %T", &tm_struct);
    return std::string(buf);
  }
}

inline void run_mixed_timestamps_test() {
  EXPECT_NO_THROW(
      run_ddl_statement("COPY import_test_timestamps FROM "
                        "'../../Tests/Import/datafiles/mixed_timestamps.txt';"));

  auto rows = run_query("SELECT * FROM import_test_timestamps");
  ASSERT_EQ(size_t(11), rows->entryCount());
  for (size_t i = 0; i < rows->entryCount() - 1; i++) {
    const auto crt_row = rows->getNextRow(true, true);
    ASSERT_EQ(size_t(13), crt_row.size());
    const auto ts0_str_nullable = v<NullableString>(crt_row[0]);
    const auto ts0_str = boost::get<std::string>(&ts0_str_nullable);
    const auto ts3_str_nullable = v<NullableString>(crt_row[1]);
    const auto ts3_str = boost::get<std::string>(&ts3_str_nullable);
    const auto ts6_str_nullable = v<NullableString>(crt_row[2]);
    const auto ts6_str = boost::get<std::string>(&ts6_str_nullable);
    const auto ts9_str_nullable = v<NullableString>(crt_row[3]);
    const auto ts9_str = boost::get<std::string>(&ts9_str_nullable);
    CHECK(ts0_str && ts3_str && ts6_str && ts9_str);
    for (size_t j = 4; j < crt_row.size(); j++) {
      const auto timeval = v<int64_t>(crt_row[j]);
      const auto ti = rows->getColType(j);
      CHECK(ti.is_timestamp());
      const auto ts_str = convert_timestamp_to_string(timeval, ti.get_dimension());
      switch (ti.get_dimension()) {
        case 0:
          ASSERT_EQ(*ts0_str, ts_str);
          break;
        case 3:
          ASSERT_EQ(*ts3_str, ts_str);
          break;
        case 6:
          ASSERT_EQ(*ts6_str, ts_str);
          break;
        case 9:
          ASSERT_EQ(*ts9_str, ts_str);
          break;
        default:
          CHECK(false);
      }
    }
  }

  const auto crt_row = rows->getNextRow(true, true);
  ASSERT_EQ(size_t(13), crt_row.size());
  for (size_t j = 4; j < crt_row.size(); j++) {
    if (j == 6 || j == 8 || j == 10 || j == 12) {
      continue;
    }
    const auto ts_null = v<int64_t>(crt_row[j]);
    ASSERT_EQ(ts_null, std::numeric_limits<int64_t>::min());
  }
}

TEST_F(ImportTestTimestamps, ImportMixedTimestamps) {
  run_mixed_timestamps_test();
}

const char* create_table_trips = R"(
    CREATE TABLE trips (
      medallion               TEXT ENCODING DICT,
      hack_license            TEXT ENCODING DICT,
      vendor_id               TEXT ENCODING DICT,
      rate_code_id            SMALLINT,
      store_and_fwd_flag      TEXT ENCODING DICT,
      pickup_datetime         TIMESTAMP,
      dropoff_datetime        TIMESTAMP,
      passenger_count         SMALLINT,
      trip_time_in_secs       INTEGER,
      trip_distance           DECIMAL(14,2),
      pickup_longitude        DECIMAL(14,2),
      pickup_latitude         DECIMAL(14,2),
      dropoff_longitude       DECIMAL(14,2),
      dropoff_latitude        DECIMAL(14,2)
    ) WITH (FRAGMENT_SIZE=75000000);
  )";

const char* create_table_random_strings_with_line_endings = R"(
    CREATE TABLE random_strings_with_line_endings (
      random_string TEXT
    ) WITH (FRAGMENT_SIZE=75000000);
  )";

class ImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_trips););
    ASSERT_NO_THROW(
        run_ddl_statement("drop table if exists random_strings_with_line_endings;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_random_strings_with_line_endings););
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table trips;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table random_strings_with_line_endings;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geo;"););
  }
};

#ifdef ENABLE_IMPORT_PARQUET
// parquet test cases
TEST_F(ImportTest, One_parquet_file_1k_rows_in_10_groups) {
  EXPECT_TRUE(
      import_test_local_parquet(".", "trip_data_1k_rows_in_10_grps.parquet", 1000, 1.0));
}
TEST_F(ImportTest, One_parquet_file) {
  EXPECT_TRUE(import_test_local_parquet(
      "trip.parquet",
      "part-00000-027865e6-e4d9-40b9-97ff-83c5c5531154-c000.snappy.parquet",
      100,
      1.0));
  EXPECT_TRUE(import_test_parquet_with_null(100));
}
TEST_F(ImportTest, One_parquet_file_drop) {
  EXPECT_TRUE(import_test_local_parquet(
      "trip+1.parquet",
      "part-00000-00496d78-a271-4067-b637-cf955cc1cece-c000.snappy.parquet",
      100,
      1.0));
}
TEST_F(ImportTest, All_parquet_file) {
  EXPECT_TRUE(import_test_local_parquet("trip.parquet", "*.parquet", 1200, 1.0));
  EXPECT_TRUE(import_test_parquet_with_null(1200));
}
TEST_F(ImportTest, All_parquet_file_drop) {
  EXPECT_TRUE(import_test_local_parquet("trip+1.parquet", "*.parquet", 1200, 1.0));
}
TEST_F(ImportTest, One_parquet_file_with_geo_point) {
  EXPECT_TRUE(import_test_local_parquet_with_geo_point(
      "trip_data_with_point.parquet",
      "part-00000-6dbefb0c-abbd-4c39-93e7-0026e36b7b7c-c000.snappy.parquet",
      100,
      1.0));
}
#ifdef HAVE_AWS_S3
// s3 parquet test cases
TEST_F(ImportTest, S3_One_parquet_file) {
  EXPECT_TRUE(import_test_s3_parquet(
      "trip.parquet",
      "part-00000-0284f745-1595-4743-b5c4-3aa0262e4de3-c000.snappy.parquet",
      100,
      1.0));
}
TEST_F(ImportTest, S3_One_parquet_file_drop) {
  EXPECT_TRUE(import_test_s3_parquet(
      "trip+1.parquet",
      "part-00000-00496d78-a271-4067-b637-cf955cc1cece-c000.snappy.parquet",
      100,
      1.0));
}
TEST_F(ImportTest, S3_All_parquet_file) {
  EXPECT_TRUE(import_test_s3_parquet("trip.parquet", "", 1200, 1.0));
}
TEST_F(ImportTest, S3_All_parquet_file_drop) {
  EXPECT_TRUE(import_test_s3_parquet("trip+1.parquet", "", 1200, 1.0));
}
TEST_F(ImportTest, S3_Null_Prefix) {
  EXPECT_THROW(run_ddl_statement("copy trips from 's3://omnisci_ficticiousbucket/';"),
               std::runtime_error);
}
TEST_F(ImportTest, S3_Wildcard_Prefix) {
  EXPECT_THROW(run_ddl_statement("copy trips from 's3://omnisci_ficticiousbucket/*';"),
               std::runtime_error);
}
#endif  // HAVE_AWS_S3
#endif  // ENABLE_IMPORT_PARQUET

TEST_F(ImportTest, One_csv_file) {
  EXPECT_TRUE(import_test_local("trip_data_9.csv", 100, 1.0));
}

TEST_F(ImportTest, random_strings_with_line_endings) {
  EXPECT_TRUE(import_test_line_endings_in_quotes_local(
      "random_strings_with_line_endings.7z", 19261));
}

TEST_F(ImportTest, One_csv_file_no_newline) {
  EXPECT_TRUE(import_test_local("trip_data_no_newline_1.csv", 100, 1.0));
}

TEST_F(ImportTest, Many_csv_file) {
  EXPECT_TRUE(import_test_local("trip_data_*.csv", 1200, 1.0));
}

TEST_F(ImportTest, Many_csv_file_no_newline) {
  EXPECT_TRUE(import_test_local("trip_data_no_newline_*.csv", 200, 1.0));
}

TEST_F(ImportTest, One_gz_file) {
  EXPECT_TRUE(import_test_local("trip_data_9.gz", 100, 1.0));
}

TEST_F(ImportTest, One_bz2_file) {
  EXPECT_TRUE(import_test_local("trip_data_9.bz2", 100, 1.0));
}

TEST_F(ImportTest, One_tar_with_many_csv_files) {
  EXPECT_TRUE(import_test_local("trip_data.tar", 1000, 1.0));
}

TEST_F(ImportTest, One_tgz_with_many_csv_files) {
  EXPECT_TRUE(import_test_local("trip_data.tgz", 100000, 1.0));
}

TEST_F(ImportTest, One_rar_with_many_csv_files) {
  EXPECT_TRUE(import_test_local("trip_data.rar", 1000, 1.0));
}

TEST_F(ImportTest, One_zip_with_many_csv_files) {
  EXPECT_TRUE(import_test_local("trip_data.zip", 1000, 1.0));
}

TEST_F(ImportTest, One_7z_with_many_csv_files) {
  EXPECT_TRUE(import_test_local("trip_data.7z", 1000, 1.0));
}

TEST_F(ImportTest, One_tgz_with_many_csv_files_no_newline) {
  EXPECT_TRUE(import_test_local("trip_data_some_with_no_newline.tgz", 500, 1.0));
}

// Sharding tests
const char* create_table_trips_sharded = R"(
    CREATE TABLE trips (
      id                      INTEGER,
      medallion               TEXT ENCODING DICT,
      hack_license            TEXT ENCODING DICT,
      vendor_id               TEXT ENCODING DICT,
      rate_code_id            SMALLINT,
      store_and_fwd_flag      TEXT ENCODING DICT,
      pickup_date             DATE,
      drop_date               DATE ENCODING FIXED(16),
      pickup_datetime         TIMESTAMP,
      dropoff_datetime        TIMESTAMP,
      passenger_count         SMALLINT,
      trip_time_in_secs       INTEGER,
      trip_distance           DECIMAL(14,2),
      pickup_longitude        DECIMAL(14,2),
      pickup_latitude         DECIMAL(14,2),
      dropoff_longitude       DECIMAL(14,2),
      dropoff_latitude        DECIMAL(14,2),
      shard key (id)
    ) WITH (FRAGMENT_SIZE=75000000, SHARD_COUNT=2);
  )";
class ImportTestSharded : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_trips_sharded););
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table trips;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geo;"););
  }
};

TEST_F(ImportTestSharded, One_csv_file) {
  EXPECT_TRUE(import_test_local("sharded_trip_data_9.csv", 100, 1.0));
}

const char* create_table_trips_dict_sharded_text = R"(
    CREATE TABLE trips (
      id                      INTEGER,
      medallion               TEXT ENCODING DICT,
      hack_license            TEXT ENCODING DICT,
      vendor_id               TEXT ENCODING DICT,
      rate_code_id            SMALLINT,
      store_and_fwd_flag      TEXT ENCODING DICT,
      pickup_date             DATE,
      drop_date               DATE ENCODING FIXED(16),
      pickup_datetime         TIMESTAMP,
      dropoff_datetime        TIMESTAMP,
      passenger_count         SMALLINT,
      trip_time_in_secs       INTEGER,
      trip_distance           DECIMAL(14,2),
      pickup_longitude        DECIMAL(14,2),
      pickup_latitude         DECIMAL(14,2),
      dropoff_longitude       DECIMAL(14,2),
      dropoff_latitude        DECIMAL(14,2),
      shard key (medallion)
    ) WITH (FRAGMENT_SIZE=75000000, SHARD_COUNT=2);
  )";
class ImportTestShardedText : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_trips_dict_sharded_text););
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table trips;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geo;"););
  }
};

TEST_F(ImportTestShardedText, One_csv_file) {
  EXPECT_TRUE(import_test_local("sharded_trip_data_9.csv", 100, 1.0));
}

const char* create_table_trips_dict_sharded_text_8bit = R"(
    CREATE TABLE trips (
      id                      INTEGER,
      medallion               TEXT ENCODING DICT (8),
      hack_license            TEXT ENCODING DICT,
      vendor_id               TEXT ENCODING DICT,
      rate_code_id            SMALLINT,
      store_and_fwd_flag      TEXT ENCODING DICT,
      pickup_date             DATE,
      drop_date               DATE ENCODING FIXED(16),
      pickup_datetime         TIMESTAMP,
      dropoff_datetime        TIMESTAMP,
      passenger_count         SMALLINT,
      trip_time_in_secs       INTEGER,
      trip_distance           DECIMAL(14,2),
      pickup_longitude        DECIMAL(14,2),
      pickup_latitude         DECIMAL(14,2),
      dropoff_longitude       DECIMAL(14,2),
      dropoff_latitude        DECIMAL(14,2),
      shard key (medallion)
    ) WITH (FRAGMENT_SIZE=75000000, SHARD_COUNT=2);
  )";
class ImportTestShardedText8 : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_trips_dict_sharded_text_8bit););
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table trips;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geo;"););
  }
};

TEST_F(ImportTestShardedText8, One_csv_file) {
  EXPECT_TRUE(import_test_local("sharded_trip_data_9.csv", 100, 1.0));
}

namespace {
const char* create_table_geo = R"(
    CREATE TABLE geospatial (
      p1 POINT,
      l LINESTRING,
      poly POLYGON,
      mpoly MULTIPOLYGON,
      p2 POINT,
      p3 POINT,
      p4 POINT,
      trip_distance DOUBLE
    ) WITH (FRAGMENT_SIZE=65000000);
  )";

void check_geo_import() {
  auto rows = run_query(R"(
      SELECT p1, l, poly, mpoly, p2, p3, p4, trip_distance
        FROM geospatial
        WHERE trip_distance = 1.0;
    )");
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(8), crt_row.size());
  const auto p1 = boost::get<std::string>(v<NullableString>(crt_row[0]));
  ASSERT_TRUE(Geo_namespace::GeoPoint("POINT (1 1)") == Geo_namespace::GeoPoint(p1));
  const auto linestring = boost::get<std::string>(v<NullableString>(crt_row[1]));
  ASSERT_TRUE(Geo_namespace::GeoLineString("LINESTRING (1 0,2 2,3 3)") ==
              Geo_namespace::GeoLineString(linestring));
  const auto poly = boost::get<std::string>(v<NullableString>(crt_row[2]));
  ASSERT_TRUE(Geo_namespace::GeoPolygon("POLYGON ((0 0,2 0,0 2,0 0))") ==
              Geo_namespace::GeoPolygon(poly));
  const auto mpoly = boost::get<std::string>(v<NullableString>(crt_row[3]));
  ASSERT_TRUE(Geo_namespace::GeoMultiPolygon("MULTIPOLYGON (((0 0,2 0,0 2,0 0)))") ==
              Geo_namespace::GeoMultiPolygon(mpoly));
  const auto p2 = boost::get<std::string>(v<NullableString>(crt_row[4]));
  ASSERT_TRUE(Geo_namespace::GeoPoint("POINT (1 1)") == Geo_namespace::GeoPoint(p2));
  const auto p3 = boost::get<std::string>(v<NullableString>(crt_row[5]));
  ASSERT_TRUE(Geo_namespace::GeoPoint("POINT (1 1)") == Geo_namespace::GeoPoint(p3));
  const auto p4 = boost::get<std::string>(v<NullableString>(crt_row[6]));
  ASSERT_TRUE(Geo_namespace::GeoPoint("POINT (1 1)") == Geo_namespace::GeoPoint(p4));
  const auto trip_distance = v<double>(crt_row[7]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

void check_geo_gdal_point_import() {
  auto rows = run_query("SELECT omnisci_geo, trip FROM geospatial WHERE trip = 1.0");
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  const auto point = boost::get<std::string>(v<NullableString>(crt_row[0]));
  ASSERT_TRUE(Geo_namespace::GeoPoint("POINT (1 1)") == Geo_namespace::GeoPoint(point));
  const auto trip_distance = v<double>(crt_row[1]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

void check_geo_gdal_mpoly_import() {
  auto rows = run_query("SELECT omnisci_geo, trip FROM geospatial WHERE trip = 1.0");
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  const auto mpoly = boost::get<std::string>(v<NullableString>(crt_row[0]));
  ASSERT_TRUE(Geo_namespace::GeoMultiPolygon("MULTIPOLYGON (((0 0,2 0,0 2,0 0)))") ==
              Geo_namespace::GeoMultiPolygon(mpoly));
  const auto trip_distance = v<double>(crt_row[1]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

void check_geo_num_rows(const std::string& project_columns,
                        const size_t num_expected_rows) {
  auto rows = run_query("SELECT " + project_columns + " FROM geospatial");
  ASSERT_TRUE(rows->entryCount() == num_expected_rows);
}

void check_geo_gdal_point_tv_import() {
  auto rows = run_query("SELECT omnisci_geo, trip FROM geospatial WHERE trip = 1.0");
  rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  auto crt_row = rows->getNextRow(true, true);
  compare_geo_target(crt_row[0], GeoPointTargetValue({1.0, 1.0}), 1e-7);
  const auto trip_distance = v<double>(crt_row[1]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

void check_geo_gdal_mpoly_tv_import() {
  auto rows = run_query("SELECT omnisci_geo, trip FROM geospatial WHERE trip = 1.0");
  rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  auto crt_row = rows->getNextRow(true, true);
  compare_geo_target(crt_row[0],
                     GeoMultiPolyTargetValue({0.0, 0.0, 2.0, 0.0, 0.0, 2.0}, {3}, {1}),
                     1e-7);
  const auto trip_distance = v<double>(crt_row[1]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

}  // namespace

class GeoImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geospatial;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_geo););
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table geospatial;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geospatial;"););
  }
};

TEST_F(GeoImportTest, CSV_Import) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial.csv");
  run_ddl_statement("COPY geospatial FROM '" + file_path.string() + "';");
  check_geo_import();
  check_geo_num_rows("p1, l, poly, mpoly, p2, p3, p4, trip_distance", 10);
}

TEST_F(GeoImportTest, CSV_Import_Empties) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial_empties.csv");
  run_ddl_statement("COPY geospatial FROM '" + file_path.string() + "';");
  check_geo_import();
  check_geo_num_rows("p1, l, poly, mpoly, p2, p3, p4, trip_distance",
                     6);  // we expect it to drop the 4 rows containing 'EMPTY'
}

TEST_F(GeoImportTest, CSV_Import_Degenerate) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial_degenerate.csv");
  run_ddl_statement("COPY geospatial FROM '" + file_path.string() + "';");
  check_geo_import();
  check_geo_num_rows("p1, l, poly, mpoly, p2, p3, p4, trip_distance",
                     6);  // we expect it to drop the 4 rows containing degenerate polys
}

// the remaining tests in this group are incomplete but leave them as placeholders

TEST_F(GeoImportTest, Geo_CSV_Local_Type_Geometry) {
  EXPECT_TRUE(
      import_test_local_geo("geospatial.csv", ", geo_coords_type='geometry'", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_Type_Geography) {
  EXPECT_THROW(
      import_test_local_geo("geospatial.csv", ", geo_coords_type='geography'", 10, 4.5),
      std::runtime_error);
}

TEST_F(GeoImportTest, Geo_CSV_Local_Type_Other) {
  EXPECT_THROW(
      import_test_local_geo("geospatial.csv", ", geo_coords_type='other'", 10, 4.5),
      std::runtime_error);
}

TEST_F(GeoImportTest, Geo_CSV_Local_Encoding_NONE) {
  EXPECT_TRUE(
      import_test_local_geo("geospatial.csv", ", geo_coords_encoding='none'", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_Encoding_GEOINT32) {
  EXPECT_TRUE(import_test_local_geo(
      "geospatial.csv", ", geo_coords_encoding='compressed(32)'", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_Encoding_Other) {
  EXPECT_THROW(
      import_test_local_geo("geospatial.csv", ", geo_coords_encoding='other'", 10, 4.5),
      std::runtime_error);
}

TEST_F(GeoImportTest, Geo_CSV_Local_SRID_LonLat) {
  EXPECT_TRUE(import_test_local_geo("geospatial.csv", ", geo_coords_srid=4326", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_SRID_Mercator) {
  EXPECT_TRUE(
      import_test_local_geo("geospatial.csv", ", geo_coords_srid=900913", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_SRID_Other) {
  EXPECT_THROW(
      import_test_local_geo("geospatial.csv", ", geo_coords_srid=12345", 10, 4.5),
      std::runtime_error);
}

class GeoGDALImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geospatial;"););
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geospatial;"););
  }
};

TEST_F(GeoGDALImportTest, Geojson_Point_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_point/geospatial_point.geojson");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_point_import();
}

TEST_F(GeoGDALImportTest, Geojson_MultiPolygon_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.geojson");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_mpoly_import();
  check_geo_num_rows("omnisci_geo, trip", 10);
}

TEST_F(GeoGDALImportTest, Geojson_MultiPolygon_Import_Empties) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly_empties.geojson");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_mpoly_import();
  check_geo_num_rows("omnisci_geo, trip", 8);  // we expect it to drop 2 of the 10 rows
}

TEST_F(GeoGDALImportTest, Geojson_MultiPolygon_Import_Degenerate) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly_degenerate.geojson");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_mpoly_import();
  check_geo_num_rows("omnisci_geo, trip", 8);  // we expect it to drop 2 of the 10 rows
}

TEST_F(GeoGDALImportTest, Shapefile_Point_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_point_import();
}

TEST_F(GeoGDALImportTest, Shapefile_MultiPolygon_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_mpoly_import();
}

TEST_F(GeoGDALImportTest, Shapefile_Point_Import_Compressed) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", true);
  check_geo_gdal_point_tv_import();
}

TEST_F(GeoGDALImportTest, Shapefile_MultiPolygon_Import_Compressed) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", true);
  check_geo_gdal_mpoly_tv_import();
}

TEST_F(GeoGDALImportTest, Shapefile_Point_Import_3857) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_point/geospatial_point_3857.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_point_tv_import();
}

TEST_F(GeoGDALImportTest, Shapefile_MultiPolygon_Import_3857) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly_3857.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_mpoly_tv_import();
}

TEST_F(GeoGDALImportTest, Geojson_MultiPolygon_Append) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.geojson");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_num_rows("omnisci_geo, trip", 10);
  ASSERT_NO_THROW(
      import_test_geofile_importer(file_path.string(), "geospatial", false, false));
  check_geo_num_rows("omnisci_geo, trip", 20);
}

TEST_F(GeoGDALImportTest, Geodatabase_Simple) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geodatabase/S_USA.Experimental_Area_Locations.gdb.zip");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_num_rows("omnisci_geo, ESTABLISHED", 87);
}

TEST_F(GeoGDALImportTest, KML_Simple) {
  SKIP_ALL_ON_AGGREGATOR();
  if (!Importer_NS::Importer::hasGDALLibKML()) {
    LOG(ERROR) << "Test requires LibKML support in GDAL";
  } else {
    const auto file_path = boost::filesystem::path("KML/test.kml");
    import_test_geofile_importer(file_path.string(), "geospatial", false);
    check_geo_num_rows("omnisci_geo, FID", 10);
  }
}

#ifdef HAVE_AWS_S3
// s3 compressed (non-parquet) test cases
TEST_F(ImportTest, S3_One_csv_file) {
  EXPECT_TRUE(import_test_s3_compressed("trip_data_9.csv", 100, 1.0));
}

TEST_F(ImportTest, S3_One_gz_file) {
  EXPECT_TRUE(import_test_s3_compressed("trip_data_9.gz", 100, 1.0));
}

TEST_F(ImportTest, S3_One_bz2_file) {
  EXPECT_TRUE(import_test_s3_compressed("trip_data_9.bz2", 100, 1.0));
}

TEST_F(ImportTest, S3_One_tar_with_many_csv_files) {
  EXPECT_TRUE(import_test_s3_compressed("trip_data.tar", 1000, 1.0));
}

TEST_F(ImportTest, S3_One_tgz_with_many_csv_files) {
  EXPECT_TRUE(import_test_s3_compressed("trip_data.tgz", 100000, 1.0));
}

TEST_F(ImportTest, S3_One_rar_with_many_csv_files) {
  EXPECT_TRUE(import_test_s3_compressed("trip_data.rar", 1000, 1.0));
}

TEST_F(ImportTest, S3_One_zip_with_many_csv_files) {
  EXPECT_TRUE(import_test_s3_compressed("trip_data.zip", 1000, 1.0));
}

TEST_F(ImportTest, S3_One_7z_with_many_csv_files) {
  EXPECT_TRUE(import_test_s3_compressed("trip_data.7z", 1000, 1.0));
}

TEST_F(ImportTest, S3_All_files) {
  EXPECT_TRUE(import_test_s3_compressed("", 105200, 1.0));
}

TEST_F(ImportTest, S3_GCS_One_gz_file) {
  EXPECT_TRUE(import_test_common(
      std::string(
          "COPY trips FROM 's3://omnisci-importtest-data/trip-data/trip_data_9.gz' "
          "WITH (header='true', s3_endpoint='storage.googleapis.com');"),
      100,
      1.0));
}

TEST_F(ImportTest, S3_GCS_One_geo_file) {
  EXPECT_TRUE(
      import_test_common_geo("COPY geo FROM "
                             "'s3://omnisci-importtest-data/geo-data/"
                             "S_USA.Experimental_Area_Locations.gdb.zip' "
                             "WITH (geo='true', s3_endpoint='storage.googleapis.com');",
                             "geo",
                             87,
                             1.0));
}
#endif  // HAVE_AWS_S3
}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()(
      "test-help",
      "Print all ImportTest specific options (for gtest options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: ImportTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  logger::init(log_options);

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
