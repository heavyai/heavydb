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

#include <Tests/TestHelpers.h>

#include <algorithm>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/process.hpp>
#include <boost/program_options.hpp>
#include <boost/range/combine.hpp>

#include "Archive/PosixFileArchive.h"
#include "Catalog/Catalog.h"
#ifdef HAVE_AWS_S3
#include "AwsHelpers.h"
#include "DataMgr/OmniSciAwsSdk.h"
#endif  // HAVE_AWS_S3
#include "Geospatial/GDAL.h"
#include "Geospatial/Types.h"
#include "ImportExport/DelimitedParserUtils.h"
#include "ImportExport/Importer.h"
#include "Parser/parser.h"
#include "QueryEngine/ResultSet.h"
#include "Shared/file_path_util.h"
#include "Shared/import_helpers.h"
#include "Shared/misc.h"
#include "Shared/scope.h"

#include "DBHandlerTestHelpers.h"
#include "ThriftHandler/DBHandler.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std;
using namespace TestHelpers;

extern bool g_use_date_in_days_default_encoding;
extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;
extern bool g_enable_parquet_import_fsi;

namespace {

bool g_regenerate_export_test_reference_files = false;

#define SKIP_ALL_ON_AGGREGATOR()                         \
  if (isDistributedMode()) {                             \
    LOG(ERROR) << "Tests not valid in distributed mode"; \
    return;                                              \
  }

std::string options_to_string(const std::map<std::string, std::string>& options,
                              bool seperate = true) {
  std::string options_str;
  for (auto const& [key, val] : options) {
    options_str += (seperate ? "," : "") + key + "='" + val + "'";
    seperate = true;
  }
  return options_str;
}

std::string TypeToString(SQLTypes type) {
  return SQLTypeInfo(type, false).get_type_name();
}

void d(const SQLTypes expected_type, const std::string& str) {
  auto detected_type = import_export::Detector::detect_sqltype(str);
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
  d(kTIME, "1.22.22");
}

class ImportExportTestBase : public DBHandlerTestFixture {
 protected:
  void SetUp() override { DBHandlerTestFixture::SetUp(); }

  void TearDown() override { DBHandlerTestFixture::TearDown(); }

  bool compareAgg(const int64_t cnt, const double avg) {
    std::string query_str = "SELECT COUNT(*), AVG(trip_distance) FROM trips;";
    sqlAndCompareResult(query_str, {{cnt, avg}});
    return true;
  }

  bool importTestCommon(const string& query_str, const int64_t cnt, const double avg) {
    sql(query_str);
    return compareAgg(cnt, avg);
  }

  bool importTestLocal(const string& filename,
                       const int64_t cnt,
                       const double avg,
                       const std::map<std::string, std::string>& options = {}) {
    return importTestCommon(
        string("COPY trips FROM '") + "../../Tests/Import/datafiles/" + filename +
            "' WITH (header='true'" +
            (filename.find(".parquet") != std::string::npos ? ",parquet='true'" : "") +
            options_to_string(options) + ");",
        cnt,
        avg);
  }

#ifdef HAVE_AWS_S3
  bool importTestS3(const string& prefix,
                    const string& filename,
                    const int64_t cnt,
                    const double avg,
                    const std::map<std::string, std::string>& options = {}) {
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

    return importTestCommon(
        string("COPY trips FROM '") + "s3://mapd-parquet-testdata/" + prefix + "/" +
            filename + "' WITH (header='true'" +
            (s3_access_key.size() ? ",s3_access_key='" + s3_access_key + "'" : "") +
            (s3_secret_key.size() ? ",s3_secret_key='" + s3_secret_key + "'" : "") +
            (s3_region.size() ? ",s3_region='" + s3_region + "'" : "") +
            (prefix.find(".parquet") != std::string::npos ||
                     filename.find(".parquet") != std::string::npos
                 ? ",parquet='true'"
                 : "") +
            options_to_string(options) + ");",
        cnt,
        avg);
  }

  bool importTestS3Compressed(const string& filename,
                              const int64_t cnt,
                              const double avg,
                              const std::map<std::string, std::string>& options = {}) {
    return importTestS3("trip.compressed", filename, cnt, avg, options);
  }

#endif
};

const char* create_table_trips_to_skip_header = R"(
    CREATE TABLE trips (
      trip_distance DECIMAL(14,2),
      random_string TEXT
    );
  )";

class ImportTestSkipHeader : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists trips;");
    sql(create_table_trips_to_skip_header);
  }

  void TearDown() override {
    sql("drop table trips;");
    ImportExportTestBase::TearDown();
  }
};

TEST_F(ImportTestSkipHeader, Skip_Header) {
  // save existing size and restore it after test so that changing it to a tiny size
  // of 10 below for this test won't affect performance of other tests.
  const auto archive_read_buf_size_state = g_archive_read_buf_size;
  // 10 makes sure that the first block returned by PosixFileArchive::read_data_block
  // does not contain the first line delimiter
  g_archive_read_buf_size = 10;
  ScopeGuard reset_archive_read_buf_size = [&archive_read_buf_size_state] {
    g_archive_read_buf_size = archive_read_buf_size_state;
  };
  EXPECT_TRUE(importTestLocal("skip_header.txt", 1, 1.0));
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

class ImportTestMixedVarlen : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists import_test_mixed_varlen;");
    sql(create_table_mixed_varlen);
  }

  void TearDown() override {
    sql("drop table if exists import_test_mixed_varlen;");
    ImportExportTestBase::TearDown();
  }
};

TEST_F(ImportTestMixedVarlen, Fix_failed_import_arrays_after_geos) {
  sql("copy import_test_mixed_varlen from "
      "'../../Tests/Import/datafiles/mixed_varlen.txt' with "
      "(header='false');");
  std::string query_str = "SELECT COUNT(*) FROM import_test_mixed_varlen;";
  sqlAndCompareResult(query_str, {{1L}});
}

const char* create_table_date = R"(
    CREATE TABLE import_test_date(
      id INT,
      date_text TEXT ENCODING DICT(32),
      date_date DATE,
      date_date_not_null DATE NOT NULL,
      date_i32 DATE ENCODING FIXED(32),
      date_i16 DATE ENCODING FIXED(16)
    );
)";

std::string convert_date_to_string(int64_t d) {
  if (d == std::numeric_limits<int64_t>::min()) {
    return std::string("NULL");
  }
  char buf[16];
  size_t const len = shared::formatDate(buf, 16, d);
  CHECK_LE(10u, len) << d;
  return std::string(buf);
}

class ImportTestDate : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists import_test_date;");
    sql(create_table_date);
  }

  void TearDown() override {
    sql("drop table if exists import_test_date;");
    ImportExportTestBase::TearDown();
  }

  void runMixedDatesTest() {
    ASSERT_NO_THROW(
        sql("COPY import_test_date FROM "
            "'../../Tests/Import/datafiles/mixed_dates.txt';"));

    TQueryResult result;
    sql(result, "SELECT * FROM import_test_date ORDER BY id;");
    // clang-format off
    assertResultSetEqual(
      {
        {1L, "2018-12-21", "12/21/2018", "12/21/2018", "12/21/2018", "12/21/2018"},
        {2L, "2018-12-21", "12/21/2018", "12/21/2018", "12/21/2018", "12/21/2018"},
        {3L, "2018-08-15", "08/15/2018", "08/15/2018", "08/15/2018", "08/15/2018"},
        {4L, "2018-08-15", "08/15/2018", "08/15/2018", "08/15/2018", "08/15/2018"},
        {5L, "1950-02-14", "02/14/1950", "02/14/1950", "02/14/1950", "02/14/1950"},
        {6L, "1960-12-31", "12/31/1960", "12/31/1960", "12/31/1960", "12/31/1960"},
        {7L, "1940-05-05", "05/05/1940", "05/05/1940", "05/05/1940", "05/05/1940"},
        {8L, "2040-05-05", "05/05/2040", "05/05/2040", "05/05/2040", "05/05/2040"},
        {9L, "2000-01-01", "01/01/2000", "01/01/2000", "01/01/2000", "01/01/2000"},
        {10L, "2000-12-31", "12/31/2000", "12/31/2000", "12/31/2000", "12/31/2000"},
        {11L, Null, Null, "01/01/2000", Null, Null}
      }, result);
    // clang-format on
  }
};

TEST_F(ImportTestDate, ImportMixedDates) {
  SKIP_ALL_ON_AGGREGATOR();  // global variable not available on leaf nodes
  runMixedDatesTest();
}

class ImportTestInt : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    const char* create_table_date = R"(
    CREATE TABLE inttable(
      b bigint,
      b32 bigint encoding fixed(32),
      b16 bigint encoding fixed(16),
      b8 bigint encoding fixed(8),
      bnn bigint not null,
      bnn32 bigint not null encoding fixed(32),
      bnn16 bigint not null encoding fixed(16),
      bnn8 bigint not null encoding fixed(8),
      i int,
      i16 int encoding fixed(16),
      i8 int encoding fixed(8),
      inn int not null,
      inn16 int not null encoding fixed(16),
      inn8 int not null encoding fixed(8),
      s smallint,
      s8 smallint encoding fixed(8),
      snn smallint not null,
      snn8 smallint not null encoding fixed(8),
      t tinyint,
      tnn tinyint not null
    );
)";
    sql("drop table if exists inttable;");
    sql(create_table_date);
  }

  void TearDown() override {
    sql("drop table if exists inttable;");
    ImportExportTestBase::TearDown();
  }
};

TEST_F(ImportTestInt, ImportBadInt) {
  SKIP_ALL_ON_AGGREGATOR();  // global variable not available on leaf nodes
  // this dataset tests that rows outside the allowed valus are rejected
  // no rows should be added
  ASSERT_NO_THROW(
      sql("COPY inttable FROM "
          "'../../Tests/Import/datafiles/int_bad_test.txt';"));

  sqlAndCompareResult("SELECT * FROM inttable;", {});
};

TEST_F(ImportTestInt, ImportGoodInt) {
  SKIP_ALL_ON_AGGREGATOR();  // global variable not available on leaf nodes
  // this dataset tests that rows inside the allowed values are accepted
  // all rows should be added
  ASSERT_NO_THROW(
      sql("COPY inttable FROM "
          "'../../Tests/Import/datafiles/int_good_test.txt';"));

  constexpr long long_min = std::numeric_limits<int64_t>::min();
  // clang-format off
  sqlAndCompareResult("SELECT * FROM inttable ORDER BY b;",
    {
      {-9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {-9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, Null, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -128L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, long_min, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, long_min, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, long_min, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, long_min, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, -127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, -32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, -2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -128L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -127L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -2147483648L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -32768L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -32768L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -127L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -32767L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -2147483648L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -2147483648L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, long_min, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, long_min, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, long_min, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, long_min, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, -127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, -32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, -2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 2147483647L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -128L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -127L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -32768L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -32768L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -127L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -32767L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -2147483648L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 127L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -2147483648L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 32767L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -2147483648L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {9223372036854775807L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L},
      {Null, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, 1L, -128L}
    });
  // clang-format on
}

class ImportTestLegacyDate : public ImportTestDate {
 protected:
  void SetUp() override {
    ImportTestDate::SetUp();
    sql("drop table if exists import_test_date;");
    g_use_date_in_days_default_encoding = false;
    sql(create_table_date);
  }

  void TearDown() override {
    sql("drop table if exists import_test_date;");
    g_use_date_in_days_default_encoding = true;
    ImportTestDate::TearDown();
  }
};

TEST_F(ImportTestLegacyDate, ImportMixedDates) {
  SKIP_ALL_ON_AGGREGATOR();  // global variable not available on leaf nodes
  runMixedDatesTest();
}

const char* create_table_date_arr = R"(
    CREATE TABLE import_test_date_arr(
      id INT,
      date_date DATE[],
      date_date_fixed DATE[2],
      date_date_not_null DATE[] NOT NULL
    );
)";

class ImportTestDateArray : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists import_test_date_arr;");
    sql(create_table_date_arr);
  }

  void TearDown() override {
    sql("drop table if exists import_test_date_arr;");
    ImportExportTestBase::TearDown();
  }

  void compareDateAndTruthArray(
      const std::vector<std::optional<int64_t>>& date_array,
      const std::vector<std::optional<std::string>>& truth_array) const {
    const auto array_size =
        std::min(date_array.size(),
                 truth_array.size());  // compare only elements that should match
    for (size_t i = 0; i < array_size; ++i) {
      ASSERT_TRUE((date_array[i].has_value() && truth_array[i].has_value()) ||
                  (!date_array[i].has_value() && !truth_array[i].has_value()))
          << " mismatch between expected and observed arrays: one is null and the other "
             "is not";
      if (date_array[i] && truth_array[i]) {
        const auto date_str = convert_date_to_string(*date_array[i]);
        ASSERT_EQ(date_str, truth_array[i]);
      }
    }
  }
};

TEST_F(ImportTestDateArray, ImportMixedDateArrays) {
  ASSERT_NO_THROW(
      sql("COPY import_test_date_arr FROM "
          "'../../Tests/Import/datafiles/mixed_date_arrays.txt';"));

  // clang-format off
  sqlAndCompareResult("SELECT * FROM import_test_date_arr ORDER BY id;",
    {
      {1L, array({"2018-12-21","2018-12-21"}), array({"12/21/2018","12/21/2018"}), array({"12/21/2018","12/21/2018"})},
      {2L, array({"2018-12-21","2018-08-15"}), array({"12/21/2018","08/15/2018"}), array({"12/21/2018","08/15/2018"})},
      {3L, array({"2018-12-21","1960-12-31","2018-12-21"}), array({"12/21/2018","12/31/1960"}), array({"12/21/2018","12/31/1960","12/21/2018"})},
      {4L, array({"2018-12-21",NULL_BIGINT}), array({"12/21/2018",NULL_BIGINT}), array({"12/21/2018","12/21/2018"})},
      {5L, array({NULL_BIGINT,"2018-12-21"}), array({NULL_BIGINT,"12/21/2018"}), array({"12/21/2018","12/21/2018"})},
      {6L, array({"2018-12-21",NULL_BIGINT}), array({"12/21/2018",NULL_BIGINT}), array({"12/21/2018","12/21/2018"})},
      {8L, Null, Null, array({"12/21/2018","12/21/2018"})},
      {9L, Null, Null, array({"12/21/2018","12/21/2018"})},
      {10L, Null, Null, array({"12/21/2018","12/21/2018"})},
      {11L, {}, array({NULL_BIGINT,NULL_BIGINT}), array({"12/21/2018","12/21/2018"})}
    });
  // clang-format on
}

class ParquetImportErrorHandling : public ImportExportTestBase {
 protected:
  void SetUp() override {
    g_enable_fsi = true;
    g_enable_s3_fsi = true;
    g_enable_parquet_import_fsi = true;
    ImportExportTestBase::SetUp();
    ASSERT_NO_THROW(sql("DROP TABLE IF EXISTS test_table;"));
  }

  void TearDown() override {
    ASSERT_NO_THROW(sql("DROP TABLE IF EXISTS test_table;"));
    ImportExportTestBase::TearDown();
    g_enable_fsi = false;
    g_enable_s3_fsi = false;
    g_enable_parquet_import_fsi = false;
  }

  const std::string fsi_file_base_dir = "../../Tests/FsiDataFiles/";
};

TEST_F(ParquetImportErrorHandling, MismatchNumberOfColumns) {
  sql("CREATE TABLE test_table (i INT);");
  queryAndAssertException(
      "COPY test_table FROM '" + fsi_file_base_dir +
          "/two_col_1_2.parquet' WITH (parquet='True');",
      "Mismatched number of logical columns: (expected 1 columns, has 2): in file "
      "'../../Tests/FsiDataFiles/two_col_1_2.parquet'");
}

TEST_F(ParquetImportErrorHandling, MismatchColumnType) {
  sql("CREATE TABLE test_table (i INT, t TEXT);");
  queryAndAssertException(
      "COPY test_table FROM '" + fsi_file_base_dir +
          "/two_col_1_2.parquet' WITH (parquet='True');",
      "Conversion from Parquet type \"INT64\" to OmniSci type \"TEXT\" is not allowed. "
      "Please use an appropriate column type. Parquet column: col2, OmniSci column: t, "
      "Parquet file: ../../Tests/FsiDataFiles/two_col_1_2.parquet.");
}

TEST_F(ParquetImportErrorHandling, FixedEncodingIntType) {
  sql("CREATE TABLE test_table (i1 BIGINT ENCODING FIXED (32), i2 BIGINT);");
  queryAndAssertException(
      "COPY test_table FROM '" + fsi_file_base_dir +
          "/two_col_1_2.parquet' WITH (parquet='True');",
      "Using fixed length encoding with Parquet import is not supported; please use the "
      "coercible non fixed-length encoded column type instead. Parquet column: col1, "
      "OmniSci column: i1, Parquet file: ../../Tests/FsiDataFiles/two_col_1_2.parquet.");
}

TEST_F(ParquetImportErrorHandling, OneInvalidInteger) {
  sql("CREATE TABLE test_table (id INT, i INT, p POINT, a INT[], t TEXT);");
  sql("COPY test_table FROM '" + fsi_file_base_dir +
      "/invalid_parquet/one_invalid_row_int.parquet' WITH (parquet='True');");
  TQueryResult query;
  sql(query, "SELECT * FROM test_table ORDER BY id;");
  assertResultSetEqual({{1L, 100L, "POINT (0 0)", array({1L, 2L}), "a"},
                        {2L, 200L, "POINT (1 0)", array({3L, 4L}), "b"},
                        {4L, 400L, "POINT (2 2)", array({8L, 9L}), "d"}},
                       query);
}

TEST_F(ParquetImportErrorHandling, OneInvalidGeoType) {
  sql("CREATE TABLE test_table (id INT, i INT, p POINT, a INT[], t TEXT);");
  sql("COPY test_table FROM '" + fsi_file_base_dir +
      "/invalid_parquet/one_invalid_row_geo.parquet' WITH (parquet='True');");
  TQueryResult query;
  sql(query, "SELECT * FROM test_table ORDER BY id;");
  assertResultSetEqual({{1L, 100L, "POINT (0 0)", array({1L, 2L}), "a"},
                        {2L, 200L, "POINT (1 0)", array({3L, 4L}), "b"},
                        {4L, 400L, "POINT (2 2)", array({8L, 9L}), "d"}},
                       query);
}

TEST_F(ParquetImportErrorHandling, OneInvalidArrayType) {
  sql("CREATE TABLE test_table (id INT, i INT, p POINT, a INT[], t TEXT);");
  sql("COPY test_table FROM '" + fsi_file_base_dir +
      "/invalid_parquet/one_invalid_row_array.parquet' WITH (parquet='True');");
  TQueryResult query;
  sql(query, "SELECT * FROM test_table ORDER BY id;");
  assertResultSetEqual({{1L, 100L, "POINT (0 0)", array({1L, 2L}), "a"},
                        {2L, 200L, "POINT (1 0)", array({3L, 4L}), "b"},
                        {4L, 400L, "POINT (2 2)", array({8L, 9L}), "d"}},
                       query);
}

TEST_F(ParquetImportErrorHandling, OneInvalidString) {
  sql("CREATE TABLE test_table (id INT, i INT, p POINT, a INT[], t TEXT);");
  sql("COPY test_table FROM '" + fsi_file_base_dir +
      "/invalid_parquet/one_invalid_row_text.parquet' WITH (parquet='True');");
  TQueryResult query;
  sql(query, "SELECT * FROM test_table ORDER BY id;");
  assertResultSetEqual({{1L, 100L, "POINT (0 0)", array({1L, 2L}), "a"},
                        {2L, 200L, "POINT (1 0)", array({3L, 4L}), "b"},
                        {4L, 400L, "POINT (2 2)", array({8L, 9L}), "d"}},
                       query);
}

TEST_F(ParquetImportErrorHandling, MaxReject) {
  sql("CREATE TABLE test_table (id INT, i INT, p POINT, a INT[], t TEXT);");
  sql("COPY test_table FROM '" + fsi_file_base_dir +
      "/invalid_parquet/' WITH (parquet='True', max_reject=3);");
  TQueryResult query;
  sql(query, "SELECT count(*) FROM test_table;");
  assertResultSetEqual({{0L}}, query);  // confirm no data was loaded into table
}

using ImportAndSelectTestParameters = std::tuple</*file_type=*/std::string,
                                                 /*data_source_type=*/std::string>;

class ImportAndSelectTest
    : public ImportExportTestBase,
      public ::testing::WithParamInterface<ImportAndSelectTestParameters> {
 protected:
  static void SetUpTestSuite() {
#ifdef HAVE_AWS_S3
    omnisci_aws_sdk::init_sdk();
    g_allow_s3_server_privileges = true;
#endif
  }

  static void TearDownTestSuite() {
#ifdef HAVE_AWS_S3
    omnisci_aws_sdk::shutdown_sdk();
    g_allow_s3_server_privileges = false;
#endif
  }

  void SetUp() override {
    g_enable_fsi = true;
    g_enable_s3_fsi = true;
    g_enable_parquet_import_fsi = true;
    ImportExportTestBase::SetUp();
    ASSERT_NO_THROW(sql("DROP TABLE IF EXISTS import_test_new;"));
  }

  void TearDown() override {
    ASSERT_NO_THROW(sql("DROP TABLE IF EXISTS import_test_new;"));
    ImportExportTestBase::TearDown();
    g_enable_fsi = false;
    g_enable_s3_fsi = false;
    g_enable_parquet_import_fsi = false;
  }

#ifdef HAVE_AWS_S3
  // Have necessary credentials to access private buckets
  bool insufficientPrivateCredentials() const {
    return !is_valid_aws_key(get_aws_keys_from_env());
  }
#endif

  bool testShouldBeSkipped() {
    std::string file_type, data_source_type;
    std::tie(file_type, data_source_type) = GetParam();
    if (data_source_type == "s3_private") {
#ifdef HAVE_AWS_S3
      return insufficientPrivateCredentials();
#else
      return true;
#endif
    }
    return false;
  }

  TQueryResult createTableCopyFromAndSelect(const std::string& schema,
                                            const std::string& file_name_base,
                                            const std::string& select_query,
                                            const std::string& table_options = {},
                                            const bool is_dir = false) {
    std::string file_type, data_source_type;
    std::tie(file_type, data_source_type) = GetParam();

    std::string query = "CREATE TABLE import_test_new (" + schema + ")";
    if (!table_options.empty()) {
      query += " WITH (" + table_options + ")";
    }
    query += ";";

    std::string base_name = file_name_base + "." + file_type;
    if (is_dir) {
      base_name = file_name_base + "_" + file_type + "_dir";
    }
    std::string file_path;
    if (data_source_type == "local") {
      file_path = "'../../Tests/FsiDataFiles/" + base_name + "'";
    } else if (data_source_type == "s3_private") {
      file_path = "'s3://omnisci-fsi-test/FsiDataFiles/" + base_name + "'";
    } else if (data_source_type == "s3_public") {
      file_path = "'s3://omnisci-fsi-test-public/FsiDataFiles/" + base_name + "'";
    }
    EXPECT_NO_THROW(sql(query));
    EXPECT_NO_THROW(sql("COPY import_test_new FROM " + file_path +
                        getCopyFromOptions(file_type, data_source_type) + ";"));
    TQueryResult result;
    sql(result, select_query);
    return result;
  }

  std::string getCopyFromOptions(const std::string& file_type,
                                 const std::string data_source_type) {
    std::vector<std::string> options;
    if (file_type == "parquet") {
      options.push_back("parquet='True'");
    }
    if (data_source_type == "s3_public" || data_source_type == "s3_private") {
      options.push_back("s3_region='us-west-1'");
    }
    if (options.empty()) {
      return {};
    }
    std::string options_string = join(options, ", ");
    return "WITH (" + options_string + ")";
  }
};

TEST_P(ImportAndSelectTest, GeoTypes) {
  if (testShouldBeSkipped()) {
    GTEST_SKIP();
  }
  auto query = createTableCopyFromAndSelect(
      "index int, p POINT, l LINESTRING, poly POLYGON, multipoly MULTIPOLYGON",
      "geo_types",
      "SELECT * FROM import_test_new ORDER BY index;");
  // clang-format off
    assertResultSetEqual({
    {
      i(1), "POINT (0 0)", "LINESTRING (0 0,0 0)", "POLYGON ((0 0,1 0,0 1,1 1,0 0))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)))"
    },
    {
      i(2), Null, Null, Null, Null
    },
    {
      i(3), "POINT (1 1)", "LINESTRING (1 1,2 2,3 3)", "POLYGON ((5 4,7 4,6 5,5 4))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"
    },
    {
      i(4), "POINT (2 2)", "LINESTRING (2 2,3 3)", "POLYGON ((1 1,3 1,2 3,1 1))",
      "MULTIPOLYGON (((0 0,3 0,0 3,0 0)),((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"
    },
    {
      i(5), Null, Null, Null, Null
    }},
    query);
  // clang-format on
}

TEST_P(ImportAndSelectTest, ArrayTypes) {
  if (testShouldBeSkipped()) {
    GTEST_SKIP();
  }
  auto query = createTableCopyFromAndSelect(
      "index INT, b BOOLEAN[], t TINYINT[], s SMALLINT[], i INTEGER[], bi BIGINT[], f "
      "FLOAT[], tm TIME[], tp TIMESTAMP[], d DATE[], txt TEXT[], fixedpoint "
      "DECIMAL(10,5)[]",
      "array_types",
      "SELECT * FROM import_test_new ORDER BY index;");
  // clang-format off
  assertResultSetEqual({
    {
      1L, array({True}), array({50L, 100L}), array({30000L, 20000L}), array({2000000000L}),
      array({9000000000000000000L}), array({10.1f, 11.1f}), array({"00:00:10"}),
      array({"1/1/2000 00:00:59", "1/1/2010 00:00:59"}), array({"1/1/2000", "2/2/2000"}),
      array({"text_1"}),array({1.23,2.34})
    },
    {
      2L, array({False, True}), array({110L}), array({30500L}), array({2000500000L}),
      array({9000000050000000000L}), array({100.12f}), array({"00:10:00", "00:20:00"}),
      array({"6/15/2020 00:59:59"}), array({"6/15/2020"}),
      array({"text_2", "text_3"}),array({3.456,4.5,5.6})
    },
    {
      3L, array({True}), array({120L}), array({31000L}), array({2100000000L, 200000000L}),
      array({9100000000000000000L, 9200000000000000000L}), array({1000.123f}), array({"10:00:00"}),
      array({"12/31/2500 23:59:59"}), array({"12/31/2500"}),
      array({"text_4"}),array({6.78})
    }},
    query);
  // clang-format on
}

TEST_P(ImportAndSelectTest, FixedLengthArrayTypes) {
  if (testShouldBeSkipped()) {
    GTEST_SKIP();
  }
  auto query = createTableCopyFromAndSelect(
      "index INT, b BOOLEAN[2], t TINYINT[2], s SMALLINT[2], i INTEGER[2], bi BIGINT[2], "
      "f FLOAT[2], tm TIME[2], tp TIMESTAMP[2], d DATE[2], txt TEXT[2], fixedpoint "
      "DECIMAL(10,5)[2]",
      "array_fixed_len_types",
      "SELECT * FROM import_test_new ORDER BY index;");
  // clang-format off
  assertResultSetEqual({
    {
      1L, array({True,False}), array({50L, 100L}), array({30000L, 20000L}), array({2000000000L,-100000L}),
      array({9000000000000000000L,-9000000000000000000L}), array({10.1f, 11.1f}), array({"00:00:10","01:00:10"}),
      array({"1/1/2000 00:00:59", "1/1/2010 00:00:59"}), array({"1/1/2000", "2/2/2000"}),
      array({"text_1","text_2"}),array({1.23,2.34})
    },
    {
      2L, array({False, True}), array({110L,101L}), array({30500L,10001L}), array({2000500000L,-23233L}),
      array({9000000050000000000L,-9200000000000000000L}), array({100.12f,2.22f}), array({"00:10:00", "00:20:00"}),
      array({"6/15/2020 00:59:59","8/22/2020 00:00:59"}), array({"6/15/2020","8/22/2020"}),
      array({"text_3", "text_4"}),array({3.456,4.5})
    },
    {
      3L, array({True,True}), array({120L,44L}), array({31000L,8123L}), array({2100000000L, 200000000L}),
      array({9100000000000000000L, 9200000000000000000L}), array({1000.123f,1392.22f}), array({"10:00:00","20:00:00"}),
      array({"12/31/2500 23:59:59","1/1/2500 23:59:59"}), array({"12/31/2500","1/1/2500"}),
      array({"text_5","text_6"}),array({6.78,5.6})
    }},
    query);
  // clang-format on
}

TEST_P(ImportAndSelectTest, ScalarTypes) {
  if (testShouldBeSkipped()) {
    GTEST_SKIP();
  }
  auto query = createTableCopyFromAndSelect(
      "b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
      "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
      "txt_2 TEXT ENCODING NONE",
      "scalar_types",
      "SELECT * FROM import_test_new ORDER BY s;");

  // clang-format off
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {True, 100L, 30000L, 2000000000L, 9000000000000000000L, 10.1f, 100.1234,
        "00:00:10", "1/1/2000 00:00:59", "1/1/2000", "text_1", "quoted text"},
      {False, 110L, 30500L, 2000500000L, 9000000050000000000L, 100.12f, 2.1234,
        "00:10:00", "6/15/2020 00:59:59", "6/15/2020", "text_2", "quoted text 2"},
      {True, 120L, 31000L, 2100000000L, 9100000000000000000L, 1000.123f, 100.1,
        "10:00:00", "12/31/2500 23:59:59", "12/31/2500", "text_3", "quoted text 3"},
  };
  // clang-format on

  std::string data_source_type = std::get<1>(GetParam());
  if (data_source_type == "local") {
    expected_values.push_back(
        {Null, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null});
  }
  assertResultSetEqual(expected_values, query);
}

TEST_P(ImportAndSelectTest, Sharded) {
  if (testShouldBeSkipped()) {
    GTEST_SKIP();
  }
  auto query = createTableCopyFromAndSelect(
      "b BOOLEAN, t TINYINT, s SMALLINT, i INTEGER, bi BIGINT, f FLOAT, "
      "dc DECIMAL(10, 5), tm TIME, tp TIMESTAMP, d DATE, txt TEXT, "
      "txt_2 TEXT ENCODING NONE, shard key(t)",
      "scalar_types",
      "SELECT * FROM import_test_new ORDER BY s;",
      "SHARD_COUNT=2");

  // clang-format off
  auto expected_values = std::vector<std::vector<NullableTargetValue>>{
      {True, 100L, 30000L, 2000000000L, 9000000000000000000L, 10.1f, 100.1234,
        "00:00:10", "1/1/2000 00:00:59", "1/1/2000", "text_1", "quoted text"},
      {False, 110L, 30500L, 2000500000L, 9000000050000000000L, 100.12f, 2.1234,
        "00:10:00", "6/15/2020 00:59:59", "6/15/2020", "text_2", "quoted text 2"},
      {True, 120L, 31000L, 2100000000L, 9100000000000000000L, 1000.123f, 100.1,
        "10:00:00", "12/31/2500 23:59:59", "12/31/2500", "text_3", "quoted text 3"},
  };
  // clang-format on

  std::string data_source_type = std::get<1>(GetParam());
  if (data_source_type == "local") {
    expected_values.push_back(
        {Null, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null, Null});
  }
  // clang-format off
    assertResultSetEqual(expected_values,
    query);
  // clang-format on
}

TEST_P(ImportAndSelectTest, Multifile) {
  if (testShouldBeSkipped()) {
    GTEST_SKIP();
  }
  auto query = createTableCopyFromAndSelect("t TEXT, i INT, f FLOAT",
                                            "example_2",
                                            "SELECT * FROM import_test_new ORDER BY i,t;",
                                            {},
                                            true);

  assertResultSetEqual(
      {
          {"a", 1L, 1.1f},
          {"aa", 1L, 1.1f},
          {"aaa", 1L, 1.1f},
          {"aa", 2L, 2.2f},
          {"aaa", 2L, 2.2f},
          {"aaa", 3L, 3.3f},
      },
      query);
}

namespace {
auto print_import_and_select_test_param = [](const auto& param_info) {
  std::string file_type, data_source_type;
  std::tie(file_type, data_source_type) = param_info.param;
  return file_type + "_" + data_source_type;
};
}

INSTANTIATE_TEST_SUITE_P(FileAndDataSourceTypes,
                         ImportAndSelectTest,
                         ::testing::Combine(::testing::Values("csv"),
                                            ::testing::Values("local"
#ifdef HAVE_AWS_S3
                                                              ,
                                                              "s3_private",
                                                              "s3_public"
#endif
                                                              )),
                         print_import_and_select_test_param);

const char* create_table_timestamps = R"(
    CREATE TABLE import_test_timestamps(
      id INT,
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

class ImportTestTimestamps : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists import_test_timestamps;");
    sql(create_table_timestamps);
  }

  void TearDown() override {
    sql("drop table if exists import_test_date;");
    ImportExportTestBase::TearDown();
  }
};

TEST_F(ImportTestTimestamps, ImportMixedTimestamps) {
  ASSERT_NO_THROW(
      sql("COPY import_test_timestamps FROM "
          "'../../Tests/Import/datafiles/mixed_timestamps.txt';"));

  // clang-format off
  sqlAndCompareResult("SELECT * FROM import_test_timestamps ORDER BY id;",
                      {{1L,
                        "01/07/2019 12:07:31",
                        "01/07/2019 12:07:31",
                        "01/07/2019 12:07:31",
                        "01/07/2019 12:07:31.000",
                        "01/07/2019 12:07:31.000",
                        "01/07/2019 12:07:31.000000",
                        "01/07/2019 12:07:31.000000",
                        "01/07/2019 12:07:31.000000000",
                        "01/07/2019 12:07:31.000000000"},
                       {2L,
                        "01/07/2019 12:07:31",
                        "01/07/2019 12:07:31",
                        "01/07/2019 12:07:31",
                        "01/07/2019 12:07:31.123",
                        "01/07/2019 12:07:31.123",
                        "01/07/2019 12:07:31.123456",
                        "01/07/2019 12:07:31.123456",
                        "01/07/2019 12:07:31.123456789",
                        "01/07/2019 12:07:31.123456789"},
                       {3L,
                        "08/15/1947 00:00:00",
                        "08/15/1947 00:00:00",
                        "08/15/1947 00:00:00",
                        "08/15/1947 00:00:00.000",
                        "08/15/1947 00:00:00.000",
                        "08/15/1947 00:00:00.000000",
                        "08/15/1947 00:00:00.000000",
                        "08/15/1947 00:00:00.000000000",
                        "08/15/1947 00:00:00.000000000"},
                       {4L,
                        "08/15/1947 00:00:00",
                        "08/15/1947 00:00:00",
                        "08/15/1947 00:00:00",
                        "08/15/1947 00:00:00.123",
                        "08/15/1947 00:00:00.123",
                        "08/15/1947 00:00:00.123456",
                        "08/15/1947 00:00:00.123456",
                        "08/15/1947 00:00:00.123456000",
                        "08/15/1947 00:00:00.123456000"},
                       {5L,
                        "11/30/2037 23:22:12",
                        "11/30/2037 23:22:12",
                        "11/30/2037 23:22:12",
                        "11/30/2037 23:22:12.123",
                        "11/30/2037 23:22:12.123",
                        "11/30/2037 23:22:12.123000",
                        "11/30/2037 23:22:12.123000",
                        "11/30/2037 23:22:12.123000000",
                        "11/30/2037 23:22:12.123000000"},
                       {6L,
                        "11/30/2037 23:22:12",
                        "11/30/2037 23:22:12",
                        "11/30/2037 23:22:12",
                        "11/30/2037 23:22:12.100",
                        "11/30/2037 23:22:12.100",
                        "11/30/2037 23:22:12.100000",
                        "11/30/2037 23:22:12.100000",
                        "11/30/2037 23:22:12.100000000",
                        "11/30/2037 23:22:12.100000000"},
                       {7L,
                        "02/03/1937 01:02:00",
                        "02/03/1937 01:02:00",
                        "02/03/1937 01:02:00",
                        "02/03/1937 01:02:00.000",
                        "02/03/1937 01:02:00.000",
                        "02/03/1937 01:02:00.000000",
                        "02/03/1937 01:02:00.000000",
                        "02/03/1937 01:02:00.000000000",
                        "02/03/1937 01:02:00.000000000"},
                       {8L,
                        "02/03/1937 01:02:04",
                        "02/03/1937 01:02:04",
                        "02/03/1937 01:02:04",
                        "02/03/1937 01:02:04.300",
                        "02/03/1937 01:02:04.300",
                        "02/03/1937 01:02:04.300000",
                        "02/03/1937 01:02:04.300000",
                        "02/03/1937 01:02:04.300000000",
                        "02/03/1937 01:02:04.300000000"},
                       {9L,
                        "02/03/1937 01:02:04",
                        "02/03/1937 01:02:04",
                        "02/03/1937 01:02:04",
                        "02/03/1937 01:02:04.300",
                        "02/03/1937 01:02:04.300",
                        "02/03/1937 01:02:04.300000",
                        "02/03/1937 01:02:04.300000",
                        "02/03/1937 01:02:04.300000000",
                        "02/03/1937 01:02:04.300000000"},
                       {10L,
                        "02/03/1937 13:02:04",
                        "02/03/1937 13:02:04",
                        "02/03/1937 13:02:04",
                        "02/03/1937 13:02:04.300",
                        "02/03/1937 13:02:04.300",
                        "02/03/1937 13:02:04.300000",
                        "02/03/1937 13:02:04.300000",
                        "02/03/1937 13:02:04.300000000",
                        "02/03/1937 13:02:04.300000000"},
                       {11L,
                        Null,
                        Null,
                        "05/23/2010 13:34:23",
                        Null,
                        "05/23/2010 13:34:23.000",
                        Null,
                        "05/23/2010 13:34:23.000000",
                        Null,
                        "05/23/2010 13:34:23.000000000"}});
  // clang-format on
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
      trip_distance           DECIMAL(5,2),
      pickup_longitude        DECIMAL(14,2),
      pickup_latitude         DECIMAL(14,2),
      dropoff_longitude       DECIMAL(14,2),
      dropoff_latitude        DECIMAL(14,2)
    ) WITH (FRAGMENT_SIZE=75000000);
  )";

const char* create_table_with_array_including_quoted_fields = R"(
  CREATE TABLE array_including_quoted_fields (
    i1            INTEGER,
    t1            TEXT,
    t2            TEXT,
    stringArray   TEXT[]
  ) WITH (FRAGMENT_SIZE=75000000);
)";

const char* create_table_random_strings_with_line_endings = R"(
    CREATE TABLE random_strings_with_line_endings (
      random_string TEXT
    ) WITH (FRAGMENT_SIZE=75000000);
  )";

const char* create_table_with_quoted_fields = R"(
    CREATE TABLE with_quoted_fields (
      id        INTEGER,
      dt1       DATE ENCODING DAYS(32),
      str1      TEXT,
      bool1     BOOLEAN,
      smallint1 SMALLINT,
      ts0       TIMESTAMP
    ) WITH (FRAGMENT_SIZE=75000000);
  )";

class ImportTest : public ImportExportTestBase {
 protected:
#ifdef HAVE_AWS_S3
  static void SetUpTestSuite() { omnisci_aws_sdk::init_sdk(); }

  static void TearDownTestSuite() { omnisci_aws_sdk::shutdown_sdk(); }
#endif

  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists trips;");
    sql(create_table_trips);
    sql("drop table if exists random_strings_with_line_endings;");
    sql(create_table_random_strings_with_line_endings);
    sql("drop table if exists with_quoted_fields;");
    sql(create_table_with_quoted_fields);
    sql("drop table if exists array_including_quoted_fields;");
    sql(create_table_with_array_including_quoted_fields);
  }

  void TearDown() override {
    sql("drop table trips;");
    sql("drop table random_strings_with_line_endings;");
    sql("drop table with_quoted_fields;");
    sql("drop table if exists geo;");
    sql("drop table if exists array_including_quoted_fields;");
    sql("drop table if exists unique_rowgroups;");
    ImportExportTestBase::TearDown();
  }

#ifdef ENABLE_IMPORT_PARQUET
  bool importTestLocalParquet(const string& prefix,
                              const string& filename,
                              const int64_t cnt,
                              const double avg,
                              const std::map<std::string, std::string>& options = {}) {
    return importTestLocal(prefix + "/" + filename, cnt, avg, options);
  }

  bool importTestParquetWithNull(const int64_t cnt) {
    sqlAndCompareResult("select count(*) from trips where rate_code_id is null;",
                        {{ cnt }});
    return true;
  }

  bool importTestLocalParquetWithGeoPoint(const string& prefix,
                                          const string& filename,
                                          const int64_t cnt,
                                          const double avg) {
    sql("alter table trips add column pt_dropoff point;");
    EXPECT_TRUE(importTestLocalParquet(prefix, filename, cnt, avg));
    std::string query_str =
        "select count(*) from trips where abs(dropoff_longitude-st_x(pt_dropoff))<0.01 "
        "and "
        "abs(dropoff_latitude-st_y(pt_dropoff))<0.01;";
    sqlAndCompareResult(query_str, {{ cnt }});
    return true;
  }

#endif

  void importTestWithQuotedFields(const std::string& filename,
                                  const std::string& quoted) {
    string query_str = "COPY with_quoted_fields FROM '../../Tests/Import/datafiles/" +
                       filename + "' WITH (header='true', quoted='" + quoted + "');";
    sql(query_str);
  }

  bool importTestLineEndingsInQuotesLocal(const string& filename, const int64_t cnt) {
    string query_str =
        "COPY random_strings_with_line_endings FROM '../../Tests/Import/datafiles/" +
        filename +
        "' WITH (header='false', quoted='true', max_reject=1, buffer_size=1048576);";
    sql(query_str);
    std::string select_query_str =
        "SELECT COUNT(*) FROM random_strings_with_line_endings;";
    sqlAndCompareResult(select_query_str, {{cnt}});
    return true;
  }

  bool importTestArrayIncludingQuotedFieldsLocal(const string& filename,
                                                 const int64_t row_count,
                                                 const string& other_options) {
    string query_str =
        "COPY array_including_quoted_fields FROM '../../Tests/Import/datafiles/" +
        filename + "' WITH (header='false', quoted='true', " + other_options + ");";
    sql(query_str);

    std::string select_query_str = "SELECT * FROM array_including_quoted_fields;";
    sqlAndCompareResult(select_query_str,
                        {{1L,
                          "field1",
                          "field2_part1,field2_part2",
                          array({"field1", "field2_part1,field2_part2"})},
                         {2L,
                          "\"field1\"",
                          "\"field2_part1,field2_part2\"",
                          array({"\"field1\"", "\"field2_part1,field2_part2\""})}});

    return true;
  }
};

#ifdef ENABLE_IMPORT_PARQUET

// parquet test cases
TEST_F(ImportTest, One_parquet_file_1k_rows_in_10_groups) {
  EXPECT_TRUE(importTestLocalParquet(
      ".", "trip_data_dir/trip_data_1k_rows_in_10_grps.parquet", 1000, 1.0));
}
TEST_F(ImportTest, One_parquet_file) {
  EXPECT_TRUE(importTestLocalParquet(
      "trip.parquet",
      "part-00000-027865e6-e4d9-40b9-97ff-83c5c5531154-c000.snappy.parquet",
      100,
      1.0));
  EXPECT_TRUE(importTestParquetWithNull(100));
}
TEST_F(ImportTest, One_parquet_file_gzip) {
  EXPECT_TRUE(importTestLocalParquet(
      "trip_gzip.parquet",
      "part-00000-10535b0e-9ae5-4d8d-9045-3c70593cc34b-c000.gz.parquet",
      100,
      1.0));
  EXPECT_TRUE(importTestParquetWithNull(100));
}
TEST_F(ImportTest, One_parquet_file_drop) {
  EXPECT_TRUE(importTestLocalParquet(
      "trip+1.parquet",
      "part-00000-00496d78-a271-4067-b637-cf955cc1cece-c000.snappy.parquet",
      100,
      1.0));
}
TEST_F(ImportTest, All_parquet_file) {
  EXPECT_TRUE(importTestLocalParquet("trip.parquet", "*.parquet", 1200, 1.0));
  EXPECT_TRUE(importTestParquetWithNull(1200));
}
TEST_F(ImportTest, All_parquet_file_gzip) {
  EXPECT_TRUE(importTestLocalParquet("trip_gzip.parquet", "*.parquet", 1200, 1.0));
}
TEST_F(ImportTest, All_parquet_file_drop) {
  EXPECT_TRUE(importTestLocalParquet("trip+1.parquet", "*.parquet", 1200, 1.0));
}
TEST_F(ImportTest, One_parquet_file_with_geo_point) {
  sql("alter table trips add column pt_dropoff point;");
  EXPECT_TRUE(importTestLocalParquet(
      "trip_data_with_point.parquet",
      "part-00000-6dbefb0c-abbd-4c39-93e7-0026e36b7b7c-c000.snappy.parquet",
      100,
      1.0));
  std::string query_str =
      "select count(*) from trips where abs(dropoff_longitude-st_x(pt_dropoff))<0.01 and "
      "abs(dropoff_latitude-st_y(pt_dropoff))<0.01;";

  sqlAndCompareResult(query_str, {{100L}});
}
TEST_F(ImportTest, OneParquetFileWithUniqueRowGroups) {
  sql("DROP TABLE IF EXISTS unique_rowgroups;");
  sql("CREATE TABLE unique_rowgroups (a float, b float, c float, d float);");
  sql("COPY unique_rowgroups FROM "
      "'../../Tests/Import/datafiles/unique_rowgroups.parquet' "
      "WITH (parquet='true');");
  std::string select_query_str = "SELECT * FROM unique_rowgroups ORDER BY a;";
  sqlAndCompareResult(select_query_str,
                      {{1.f, 3.f, 6.f, 7.1f},
                       {2.f, 4.f, 7.f, 5.91e-4f},
                       {3.f, 5.f, 8.f, 1.1f},
                       {4.f, 6.f, 9.f, 2.2123e-2f},
                       {5.f, 7.f, 10.f, -1.f},
                       {6.f, 8.f, 1.f, -100.f}});
  sql("DROP TABLE unique_rowgroups;");
}
#ifdef HAVE_AWS_S3
// s3 parquet test cases
TEST_F(ImportTest, S3_One_parquet_file) {
  EXPECT_TRUE(
      importTestS3("trip.parquet",
                   "part-00000-0284f745-1595-4743-b5c4-3aa0262e4de3-c000.snappy.parquet",
                   100,
                   1.0));
}
TEST_F(ImportTest, S3_One_parquet_file_drop) {
  EXPECT_TRUE(
      importTestS3("trip+1.parquet",
                   "part-00000-00496d78-a271-4067-b637-cf955cc1cece-c000.snappy.parquet",
                   100,
                   1.0));
}
TEST_F(ImportTest, S3_All_parquet_file) {
  EXPECT_TRUE(importTestS3("trip.parquet", "", 1200, 1.0));
}
TEST_F(ImportTest, S3_All_parquet_file_drop) {
  EXPECT_TRUE(importTestS3("trip+1.parquet", "", 1200, 1.0));
}
TEST_F(ImportTest, S3_Regex_path_filter_parquet_match) {
  EXPECT_TRUE(importTestS3(
      "trip.parquet",
      "",
      100,
      1.0,
      {{"REGEX_PATH_FILTER",
        ".*part-00000-9109acad-a559-4a00-b05c-878aeb8bca24-c000.snappy.parquet.*"}}));
}
TEST_F(ImportTest, S3_Regex_path_filter_parquet_no_match) {
  EXPECT_THROW(
      importTestS3(
          "trip.parquet", "", -1, -1.0, {{"REGEX_PATH_FILTER", "very?obscure?pattern"}}),
      TOmniSciException);
}
TEST_F(ImportTest, S3_Null_Prefix) {
  EXPECT_THROW(sql("copy trips from 's3://omnisci_ficticiousbucket/';"),
               TOmniSciException);
}
TEST_F(ImportTest, S3_Wildcard_Prefix) {
  EXPECT_THROW(sql("copy trips from 's3://omnisci_ficticiousbucket/*';"),
               TOmniSciException);
}
#endif  // HAVE_AWS_S3
#endif  // ENABLE_IMPORT_PARQUET

TEST_F(ImportTest, One_csv_file) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/csv/trip_data_9.csv", 100, 1.0));
}

TEST_F(ImportTest, tsv_file) {
  // Test the delimeter option
  EXPECT_TRUE(importTestCommon(
      string("COPY trips FROM ") +
          "'../../Tests/Import/datafiles/trip_data_dir/csv/trip_data.tsv'" +
          " WITH (header='true', delimiter = '\t');",
      100,
      1.0));

  // Test: Via omnisql the delimiter can get flattened/escaped
  EXPECT_TRUE(importTestCommon(
      string("COPY trips FROM ") +
          "'../../Tests/Import/datafiles/trip_data_dir/csv/trip_data.tsv'" +
          " WITH (header='true', delimiter = '\\t');",
      200,
      1.0));
}

TEST_F(ImportTest, array_including_quoted_fields) {
  EXPECT_TRUE(importTestArrayIncludingQuotedFieldsLocal(
      "array_including_quoted_fields.csv", 2, "array_delimiter=','"));
}

TEST_F(ImportTest, array_including_quoted_fields_different_delimiter) {
  sql("drop table if exists array_including_quoted_fields;");
  sql(create_table_with_array_including_quoted_fields);
  EXPECT_TRUE(importTestArrayIncludingQuotedFieldsLocal(
      "array_including_quoted_fields_different_delimiter.csv", 2, "array_delimiter='|'"));
}

TEST_F(ImportTest, random_strings_with_line_endings) {
  EXPECT_TRUE(
      importTestLineEndingsInQuotesLocal("random_strings_with_line_endings.7z", 19261));
}

// TODO: expose and validate rows imported/rejected count
TEST_F(ImportTest, with_quoted_fields) {
  for (auto quoted : {"false", "true"}) {
    EXPECT_NO_THROW(
        importTestWithQuotedFields("with_quoted_fields_doublequotes.csv", quoted));
    EXPECT_NO_THROW(
        importTestWithQuotedFields("with_quoted_fields_noquotes.csv", quoted));
  }
}

TEST_F(ImportTest, One_csv_file_no_newline) {
  EXPECT_TRUE(importTestLocal(
      "trip_data_dir/csv/no_newline/trip_data_no_newline_1.csv", 100, 1.0));
}

TEST_F(ImportTest, Many_csv_file) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/csv/trip_data_*.csv", 1000, 1.0));
}

TEST_F(ImportTest, Many_csv_file_no_newline) {
  EXPECT_TRUE(importTestLocal(
      "trip_data_dir/csv/no_newline/trip_data_no_newline_*.csv", 200, 1.0));
}

TEST_F(ImportTest, One_gz_file) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/compressed/trip_data_9.gz", 100, 1.0));
}

TEST_F(ImportTest, One_bz2_file) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/compressed/trip_data_9.bz2", 100, 1.0));
}

TEST_F(ImportTest, One_tar_with_many_csv_files) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/compressed/trip_data.tar", 1000, 1.0));
}

TEST_F(ImportTest, One_tgz_with_many_csv_files) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/compressed/trip_data.tgz", 100000, 1.0));
}

TEST_F(ImportTest, One_rar_with_many_csv_files) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/compressed/trip_data.rar", 1000, 1.0));
}

TEST_F(ImportTest, One_zip_with_many_csv_files) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/compressed/trip_data.zip", 1000, 1.0));
}

TEST_F(ImportTest, One_7z_with_many_csv_files) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/compressed/trip_data.7z", 1000, 1.0));
}

TEST_F(ImportTest, One_tgz_with_many_csv_files_no_newline) {
  EXPECT_TRUE(importTestLocal(
      "trip_data_dir/compressed/trip_data_some_with_no_newline.tgz", 500, 1.0));
}

TEST_F(ImportTest, No_match_wildcard) {
  std::string expected_error_message{
      "File or directory \"../../Tests/Import/datafiles/no_match*\" "
      "does not exist."};
  queryAndAssertException("COPY trips FROM '../../Tests/Import/datafiles/no_match*';",
                          expected_error_message);
}

TEST_F(ImportTest, Many_files_directory) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/csv", 1200, 1.0));
}

TEST_F(ImportTest, Regex_path_filter_match) {
  EXPECT_TRUE(importTestLocal(
      "trip_data_dir/csv", 300, 1.0, {{"REGEX_PATH_FILTER", ".*trip_data_[5-7]\\.csv"}}));
}

TEST_F(ImportTest, Regex_path_filter_no_match) {
  EXPECT_THROW(
      importTestLocal(
          "trip_data_dir/csv", -1, -1.0, {{"REGEX_PATH_FILTER", "very?obscure?path"}}),
      TOmniSciException);
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
class ImportTestSharded : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists trips;");
    sql(create_table_trips_sharded);
  }

  void TearDown() override {
    sql("drop table trips;");
    sql("drop table if exists geo;");
    ImportExportTestBase::TearDown();
  }
};

TEST_F(ImportTestSharded, One_csv_file) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/sharded_trip_data_9.csv", 100, 1.0));
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
class ImportTestShardedText : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists trips;");
    sql(create_table_trips_dict_sharded_text);
  }

  void TearDown() override {
    sql("drop table trips;");
    sql("drop table if exists geo;");
    ImportExportTestBase::TearDown();
  }
};

TEST_F(ImportTestShardedText, One_csv_file) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/sharded_trip_data_9.csv", 100, 1.0));
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
class ImportTestShardedText8 : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists trips;");
    sql(create_table_trips_dict_sharded_text_8bit);
  }

  void TearDown() override {
    sql("drop table trips;");
    sql("drop table if exists geo;");
    ImportExportTestBase::TearDown();
  }
};

TEST_F(ImportTestShardedText8, One_csv_file) {
  EXPECT_TRUE(importTestLocal("trip_data_dir/sharded_trip_data_9.csv", 100, 1.0));
}

namespace {
const char* create_table_geo = R"(
    CREATE TABLE geospatial (
      p1 POINT,
      l LINESTRING,
      poly POLYGON NOT NULL,
      mpoly MULTIPOLYGON,
      p2 GEOMETRY(POINT, 4326) ENCODING NONE,
      p3 GEOMETRY(POINT, 4326) NOT NULL ENCODING NONE,
      p4 GEOMETRY(POINT) NOT NULL,
      trip_distance DOUBLE
    ) WITH (FRAGMENT_SIZE=65000000);
  )";

const char* create_table_geo_transform = R"(
    CREATE TABLE geospatial_transform (
      pt0 GEOMETRY(POINT, 4326),
      pt1 GEOMETRY(POINT)
    ) WITH (FRAGMENT_SIZE=65000000);
  )";

}  // namespace

class ImportTestGeo : public ImportExportTestBase {
 protected:
  void SetUp() override {
    ImportExportTestBase::SetUp();
    import_export::delimited_parser::set_max_buffer_resize(max_buffer_resize_);
    sql("drop table if exists geospatial;");
    sql(create_table_geo);
    sql("drop table if exists geospatial_transform;");
    sql(create_table_geo_transform);
  }

  void TearDown() override {
    sql("drop table if exists geospatial;");
    sql("drop table if exists geospatial;");
    sql("drop table if exists geospatial_transform;");
    ImportExportTestBase::TearDown();
  }

  inline static size_t max_buffer_resize_ =
      import_export::delimited_parser::get_max_buffer_resize();

  bool importTestCommonGeo(const string& query_str) {
    sql(query_str);
    return true;
  }

  bool importTestLocalGeo(const string& filename, const string& other_options) {
    bool is_csv = boost::iends_with(filename, ".csv");
    return importTestCommonGeo(
        string("COPY geospatial FROM '") +
        boost::filesystem::canonical("../../Tests/Import/datafiles/" + filename)
            .string() +
        "' WITH (geo=" + (is_csv ? "'false'" : "'true'") + " " + other_options + ");");
  }

  void checkGeoNumRows(const std::string& project_columns,
                       const size_t num_expected_rows) {
    TQueryResult result;
    sql(result, "SELECT " + project_columns + " FROM geospatial");
    ASSERT_EQ(getRowCount(result), num_expected_rows);
  }

  void checkGeoImport(bool expect_nulls = false) {
    const std::string select_query_str = R"(
      SELECT p1, l, poly, mpoly, p2, p3, p4, trip_distance
        FROM geospatial
        WHERE trip_distance = 1.0;
    )";

    if (!expect_nulls) {
      sqlAndCompareResult(select_query_str,
                          {{"POINT (1 1)",
                            "LINESTRING (1 0,2 2,3 3)",
                            "POLYGON ((0 0,2 0,0 2,0 0))",
                            "MULTIPOLYGON (((0 0,2 0,0 2,0 0)))",
                            "POINT (1 1)",
                            "POINT (1 1)",
                            "POINT (1 1)",
                            1.0f}});
    } else {
      sqlAndCompareResult(select_query_str,
                          {{Null,
                            Null,
                            "POLYGON ((0 0,2 0,0 2,0 0))",
                            Null,
                            Null,
                            "POINT (1 1)",
                            "POINT (1 1)",
                            1.0f}});
    }
  }
};

TEST_F(ImportTestGeo, CSV_Import) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial.csv");
  sql("COPY geospatial FROM '" + file_path.string() + "';");
  checkGeoImport();
  checkGeoNumRows("p1, l, poly, mpoly, p2, p3, p4, trip_distance", 10);
}

TEST_F(ImportTestGeo, CSV_Import_Buffer_Size_Less_Than_Row_Size) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial.csv");
  sql("COPY geospatial FROM '" + file_path.string() + "' WITH (buffer_size = 80);");
  checkGeoImport();
  checkGeoNumRows("p1, l, poly, mpoly, p2, p3, p4, trip_distance", 10);
}

TEST_F(ImportTestGeo, CSV_Import_Max_Buffer_Resize_Less_Than_Row_Size) {
  import_export::delimited_parser::set_max_buffer_resize(170);
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial.csv");

  std::string expected_error_message{
      "Unable to find an end of line character after reading 170 characters. "
      "Please ensure that the correct \"line_delimiter\" option is specified "
      "or update the \"buffer_size\" option appropriately. Row number: 10. "
      "First few characters in row: "
      "\"POINT(9 9)\", \"LINESTRING(9 0, 18 18, 19 19)\", \"PO"};
  queryAndAssertException(
      "COPY geospatial FROM '" + file_path.string() + "' WITH (buffer_size = 80);",
      expected_error_message);
}

TEST_F(ImportTestGeo, CSV_Import_Empties) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial_empties.csv");
  sql("COPY geospatial FROM '" + file_path.string() + "';");
  checkGeoImport();
  checkGeoNumRows("p1, l, poly, mpoly, p2, p3, p4, trip_distance",
                  6);  // we expect it to drop the 4 rows containing 'EMPTY'
}

TEST_F(ImportTestGeo, CSV_Import_Nulls) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial_nulls.csv");
  sql("COPY geospatial FROM '" + file_path.string() + "';");
  checkGeoImport(true);
  checkGeoNumRows("p1, l, poly, mpoly, p2, p3, p4, trip_distance",
                  7);  // drop 3 rows containing NULL geo for NOT NULL columns
}

TEST_F(ImportTestGeo, CSV_Import_Degenerate) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial_degenerate.csv");
  sql("COPY geospatial FROM '" + file_path.string() + "';");
  checkGeoImport();
  checkGeoNumRows("p1, l, poly, mpoly, p2, p3, p4, trip_distance",
                  6);  // we expect it to drop the 4 rows containing degenerate polys
}

TEST_F(ImportTestGeo, CSV_Import_Transform_Point_2263) {
  const auto file_path = boost::filesystem::path(
      "../../Tests/Import/datafiles/geospatial_transform/point_2263.csv");
  sql("COPY geospatial_transform FROM '" + file_path.string() +
      "' WITH (source_srid=2263);");
  sqlAndCompareResult(R"(
      SELECT count(*) FROM geospatial_transform
        WHERE ST_Distance(pt0, ST_SetSRID(pt1,4326))<0.00000000001;
    )",
                      {{7L}});
}

TEST_F(ImportTestGeo, CSV_Import_Transform_Point_Coords_2263) {
  const auto file_path = boost::filesystem::path(
      "../../Tests/Import/datafiles/geospatial_transform/point_coords_2263.csv");
  sql("COPY geospatial_transform FROM '" + file_path.string() +
      "' WITH (source_srid=2263);");
  sqlAndCompareResult(R"(
      SELECT count(*) FROM geospatial_transform
        WHERE ST_Distance(pt0, ST_SetSRID(pt1,4326))<0.00000000001;
    )",
                      {{7L}});
}

// the remaining tests in this group are incomplete but leave them as placeholders

TEST_F(ImportTestGeo, Geo_CSV_Local_Type_Geometry) {
  EXPECT_TRUE(importTestLocalGeo("geospatial.csv", ", geo_coords_type='geometry'"));
}

TEST_F(ImportTestGeo, Geo_CSV_Local_Type_Geography) {
  EXPECT_THROW(importTestLocalGeo("geospatial.csv", ", geo_coords_type='geography'"),
               TOmniSciException);
}

TEST_F(ImportTestGeo, Geo_CSV_Local_Type_Other) {
  EXPECT_THROW(importTestLocalGeo("geospatial.csv", ", geo_coords_type='other'"),
               TOmniSciException);
}

TEST_F(ImportTestGeo, Geo_CSV_Local_Encoding_NONE) {
  EXPECT_TRUE(importTestLocalGeo("geospatial.csv", ", geo_coords_encoding='none'"));
}

TEST_F(ImportTestGeo, Geo_CSV_Local_Encoding_GEOINT32) {
  EXPECT_TRUE(
      importTestLocalGeo("geospatial.csv", ", geo_coords_encoding='compressed(32)'"));
}

TEST_F(ImportTestGeo, Geo_CSV_Local_Encoding_Other) {
  EXPECT_THROW(importTestLocalGeo("geospatial.csv", ", geo_coords_encoding='other'"),
               TOmniSciException);
}

TEST_F(ImportTestGeo, Geo_CSV_Local_SRID_LonLat) {
  EXPECT_TRUE(importTestLocalGeo("geospatial.csv", ", geo_coords_srid=4326"));
}

TEST_F(ImportTestGeo, Geo_CSV_Local_SRID_Mercator) {
  EXPECT_TRUE(importTestLocalGeo("geospatial.csv", ", geo_coords_srid=900913"));
}

TEST_F(ImportTestGeo, Geo_CSV_Local_SRID_Other) {
  EXPECT_THROW(importTestLocalGeo("geospatial.csv", ", geo_coords_srid=12345"),
               TOmniSciException);
}

TEST_F(ImportTestGeo, Geo_CSV_WKB) {
  const auto file_path =
      boost::filesystem::path("../../Tests/Import/datafiles/geospatial_wkb.csv");
  sql("COPY geospatial FROM '" + file_path.string() + "';");
  checkGeoImport();
  checkGeoNumRows("p1, l, poly, mpoly, p2, p3, p4, trip_distance", 1);
}

class ImportTestGDAL : public ImportTestGeo {
 protected:
  void SetUp() override {
    ImportTestGeo::SetUp();
    sql("drop table if exists geospatial;");
  }

  void TearDown() override {
    sql("drop table if exists geospatial;");
    ImportTestGeo::TearDown();
  }

  void importGeoTable(const std::string& file_path,
                      const std::string& table_name,
                      const bool compression,
                      const bool create_table,
                      const bool explode_collections) {
    std::string options = "";
    if (compression) {
      options += ", geo_coords_encoding = 'COMPRESSED(32)'";
    } else {
      options += ", geo_coords_encoding = 'none'";
    }
    if (explode_collections) {
      options += ", geo_explode_collections = 'true'";
    } else {
      options += ", geo_explode_collections = 'false'";
    }
    std::string copy_query = "COPY " + table_name + " FROM '" + file_path +
                             "' WITH (geo='true' " + options + ");";

    auto& cat = getCatalog();
    const std::string geo_column_name(OMNISCI_GEO_PREFIX);

    import_export::CopyParams copy_params;
    if (compression) {
      copy_params.geo_coords_encoding = EncodingType::kENCODING_GEOINT;
      copy_params.geo_coords_comp_param = 32;
    } else {
      copy_params.geo_coords_encoding = EncodingType::kENCODING_NONE;
      copy_params.geo_coords_comp_param = 0;
    }
    copy_params.geo_assign_render_groups = true;
    copy_params.geo_explode_collections = explode_collections;

    // direct Importer calls assume that any GDAL VSI prefixes have
    // already been added, so we must check for the file types we
    // are going to encounter, and add the appropriate prefixes here
    auto vsi_file_path{file_path};
    if (boost::filesystem::path(vsi_file_path).extension() == ".gz") {
      vsi_file_path = "/vsigzip/" + vsi_file_path;
    }

    std::map<std::string, std::string> colname_to_src;
    auto cds = import_export::Importer::gdalToColumnDescriptors(
        vsi_file_path, geo_column_name, copy_params);

    for (auto& cd : cds) {
      const auto col_name_sanitized = ImportHelpers::sanitize_name(cd.columnName);
      const auto ret =
          colname_to_src.insert(std::make_pair(col_name_sanitized, cd.columnName));
      CHECK(ret.second);
      cd.columnName = col_name_sanitized;
    }

    if (create_table) {
      const auto td = cat.getMetadataForTable(table_name);
      if (td != nullptr) {
        throw std::runtime_error(
            "Error: Table " + table_name +
            " already exists. Possible failure to correctly re-create "
            "mapd_data directory.");
      }
      if (table_name != ImportHelpers::sanitize_name(table_name)) {
        throw std::runtime_error("Invalid characters in table name: " + table_name);
      }

      std::string stmt{"CREATE TABLE " + table_name};
      std::vector<std::string> col_stmts;

      for (auto& cd : cds) {
        if (cd.columnType.get_type() == SQLTypes::kINTERVAL_DAY_TIME ||
            cd.columnType.get_type() == SQLTypes::kINTERVAL_YEAR_MONTH) {
          throw std::runtime_error(
              "Unsupported type: INTERVAL_DAY_TIME or INTERVAL_YEAR_MONTH for col " +
              cd.columnName + " (table: " + table_name + ")");
        }

        if (cd.columnType.get_type() == SQLTypes::kDECIMAL) {
          if (cd.columnType.get_precision() == 0 && cd.columnType.get_scale() == 0) {
            cd.columnType.set_precision(14);
            cd.columnType.set_scale(7);
          }
        }

        std::string col_stmt;
        col_stmt.append(cd.columnName + " " + cd.columnType.get_type_name() + " ");

        if (cd.columnType.get_compression() != EncodingType::kENCODING_NONE) {
          col_stmt.append("ENCODING " + cd.columnType.get_compression_name() + " ");
        } else {
          if (cd.columnType.is_string()) {
            col_stmt.append("ENCODING NONE");
          } else if (cd.columnType.is_geometry()) {
            if (cd.columnType.get_output_srid() == 4326) {
              col_stmt.append("ENCODING NONE");
            }
          }
        }
        col_stmts.push_back(col_stmt);
      }

      stmt.append(" (" + boost::algorithm::join(col_stmts, ",") + ");");
      sql(stmt);

      LOG(INFO) << "Created table: " << table_name;
    } else {
      LOG(INFO) << "Not creating table: " << table_name;
    }

    const auto td = cat.getMetadataForTable(table_name);
    if (td == nullptr) {
      throw std::runtime_error("Error: Failed to create table " + table_name);
    }

    sql(copy_query);
  }

  void importTestGeofileImporter(const std::string& file_str,
                                 const std::string& table_name,
                                 const bool compression,
                                 const bool create_table,
                                 const bool explode_collections) {
    auto file_path =
        boost::filesystem::canonical("../../Tests/Import/datafiles/" + file_str);

    ASSERT_TRUE(boost::filesystem::exists(file_path));

    ASSERT_NO_THROW(importGeoTable(
        file_path.string(), table_name, compression, create_table, explode_collections));
  }

  void checkGeoGdalPointImport() { checkGeoGdalAgainstWktString("POINT (1 1)"); }

  void checkGeoGdalPolyOrMpolyImport(const bool mpoly, const bool exploded) {
    TQueryResult result;
    sql(result, "SELECT omnisci_geo, trip FROM geospatial WHERE trip = 1.0");

    if (mpoly && exploded) {
      // mpoly explodes to poly (not promoted)
      assertResultSetEqual(
          {{"POLYGON ((0 0,2 0,0 2,0 0))", 1.0f}, {"POLYGON ((0 0,2 0,0 2,0 0))", 1.0f}},
          result);
    } else if (mpoly) {
      // mpoly imports as mpoly
      assertResultSetEqual(
          {{"MULTIPOLYGON (((0 0,2 0,0 2,0 0)),((0 0,2 0,0 2,0 0)))", 1.0f}}, result);
    } else {
      // poly imports as mpoly (promoted)
      assertResultSetEqual({{"MULTIPOLYGON (((0 0,2 0,0 2,0 0)))", 1.0f}}, result);
    }
  }

  void checkGeoGdalAgainstWktString(const std::string wkt_string) {
    sqlAndCompareResult("SELECT omnisci_geo, trip FROM geospatial WHERE trip = 1.0",
                        {{wkt_string, 1.0f}});
  }

  void checkGeoGdalPointImport(const std::string wkt_string) {
    checkGeoGdalAgainstWktString(wkt_string);
  }

  void checkGeoGdalMpolyImport(const std::string wkt_string) {
    checkGeoGdalAgainstWktString(wkt_string);
  }
};

TEST_F(ImportTestGDAL, Geojson_Point_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_point/geospatial_point.geojson");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPointImport();
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Geojson_Poly_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_poly.geojson");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPolyOrMpolyImport(false, false);  // poly, not exploded
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Geojson_MultiPolygon_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.geojson");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPolyOrMpolyImport(true, false);  // mpoly, not exploded
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Geojson_MultiPolygon_Explode_MPoly_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.geojson");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, true);
  checkGeoGdalPolyOrMpolyImport(true, true);  // mpoly, exploded
  checkGeoNumRows("omnisci_geo, trip", 20);   // 10M -> 20P
}

TEST_F(ImportTestGDAL, Geojson_MultiPolygon_Explode_Mixed_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mixed.geojson");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, true);
  checkGeoGdalPolyOrMpolyImport(true, true);  // mpoly, exploded
  checkGeoNumRows("omnisci_geo, trip", 15);   // 5M + 5P -> 15P
}

TEST_F(ImportTestGDAL, Geojson_MultiPolygon_Import_Empties) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly_empties.geojson");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPolyOrMpolyImport(true, false);  // mpoly, not exploded
  checkGeoNumRows("omnisci_geo, trip", 8);     // we expect it to drop 2 of the 10 rows
}

TEST_F(ImportTestGDAL, Geojson_MultiPolygon_Import_Degenerate) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly_degenerate.geojson");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPolyOrMpolyImport(true, false);  // mpoly, not exploded
  checkGeoNumRows("omnisci_geo, trip", 8);     // we expect it to drop 2 of the 10 rows
}

TEST_F(ImportTestGDAL, Shapefile_Point_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point.shp");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPointImport();
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Shapefile_MultiPolygon_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.shp");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPolyOrMpolyImport(false, false);  // poly, not exploded
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Shapefile_Point_Import_Compressed) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point.shp");
  importTestGeofileImporter(file_path.string(), "geospatial", true, true, false);
  checkGeoGdalPointImport("POINT (0.999999940861017 0.999999982770532)");
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Shapefile_MultiPolygon_Import_Compressed) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.shp");
  importTestGeofileImporter(file_path.string(), "geospatial", true, true, false);
  checkGeoGdalMpolyImport(
      "MULTIPOLYGON (((0 0,1.99999996554106 0.0,0.0 1.99999996554106,0 0)))");
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Shapefile_Point_Import_3857) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_point/geospatial_point_3857.shp");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPointImport("POINT (1.0 1.0)");
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Shapefile_MultiPolygon_Import_3857) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly_3857.shp");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalMpolyImport("MULTIPOLYGON (((0 0,2.0 0.0,0.0 2.0,0 0)))");
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, Geojson_MultiPolygon_Append) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.geojson");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoNumRows("omnisci_geo, trip", 10);
  ASSERT_NO_THROW(
      importTestGeofileImporter(file_path.string(), "geospatial", false, false, false));
  checkGeoNumRows("omnisci_geo, trip", 20);
}

TEST_F(ImportTestGDAL, Geodatabase_Simple) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path =
      boost::filesystem::path("geodatabase/S_USA.Experimental_Area_Locations.gdb.zip");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoNumRows("omnisci_geo, ESTABLISHED", 87);
}

TEST_F(ImportTestGDAL, KML_Simple) {
  SKIP_ALL_ON_AGGREGATOR();
  if (!Geospatial::GDAL::supportsDriver("libkml")) {
    LOG(ERROR) << "Test requires LibKML support in GDAL";
  } else {
    const auto file_path = boost::filesystem::path("KML/test.kml");
    importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
    checkGeoNumRows("omnisci_geo, FID", 10);
  }
}

TEST_F(ImportTestGDAL, FlatGeobuf_Point_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point.fgb");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPointImport();
  checkGeoNumRows("omnisci_geo, trip", 10);
}

TEST_F(ImportTestGDAL, FlatGeobuf_MultiPolygon_Import) {
  SKIP_ALL_ON_AGGREGATOR();
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.fgb");
  importTestGeofileImporter(file_path.string(), "geospatial", false, true, false);
  checkGeoGdalPolyOrMpolyImport(false, false);  // poly, not exploded
  checkGeoNumRows("omnisci_geo, trip", 10);
}

#ifdef HAVE_AWS_S3
// s3 compressed (non-parquet) test cases
TEST_F(ImportTest, S3_One_csv_file) {
  EXPECT_TRUE(importTestS3Compressed("trip_data_9.csv", 100, 1.0));
}

TEST_F(ImportTest, S3_One_gz_file) {
  EXPECT_TRUE(importTestS3Compressed("trip_data_9.gz", 100, 1.0));
}

TEST_F(ImportTest, S3_One_bz2_file) {
  EXPECT_TRUE(importTestS3Compressed("trip_data_9.bz2", 100, 1.0));
}

TEST_F(ImportTest, S3_One_tar_with_many_csv_files) {
  EXPECT_TRUE(importTestS3Compressed("trip_data.tar", 1000, 1.0));
}

TEST_F(ImportTest, S3_One_tgz_with_many_csv_files) {
  EXPECT_TRUE(importTestS3Compressed("trip_data.tgz", 100000, 1.0));
}

TEST_F(ImportTest, S3_One_rar_with_many_csv_files) {
  EXPECT_TRUE(importTestS3Compressed("trip_data.rar", 1000, 1.0));
}

TEST_F(ImportTest, S3_One_zip_with_many_csv_files) {
  EXPECT_TRUE(importTestS3Compressed("trip_data.zip", 1000, 1.0));
}

TEST_F(ImportTest, S3_One_7z_with_many_csv_files) {
  EXPECT_TRUE(importTestS3Compressed("trip_data.7z", 1000, 1.0));
}

TEST_F(ImportTest, S3_All_files) {
  EXPECT_TRUE(importTestS3Compressed("", 105200, 1.0));
}

TEST_F(ImportTest, S3_Regex_path_filter_match) {
  EXPECT_TRUE(importTestS3Compressed(
      "", 300, 1.0, {{"REGEX_PATH_FILTER", ".*trip_data_[5-7]\\.csv"}}));
}

TEST_F(ImportTest, S3_Regex_path_filter_no_match) {
  EXPECT_THROW(importTestS3Compressed(
                   "", -1, -1.0, {{"REGEX_PATH_FILTER", "very?obscure?pattern"}}),
               TOmniSciException);
}

TEST_F(ImportTest, S3_GCS_One_gz_file) {
  EXPECT_TRUE(importTestCommon(
      std::string(
          "COPY trips FROM 's3://omnisci-importtest-data/trip-data/trip_data_9.gz' "
          "WITH (header='true', s3_endpoint='storage.googleapis.com');"),
      100,
      1.0));
}

TEST_F(ImportTestGeo, S3_GCS_One_geo_file) {
  EXPECT_TRUE(
      importTestCommonGeo("COPY geopatial FROM "
                          "'s3://omnisci-importtest-data/geo-data/"
                          "S_USA.Experimental_Area_Locations.gdb.zip' "
                          "WITH (geo='true', s3_endpoint='storage.googleapis.com');"));
}

class ImportServerPrivilegeTest : public ImportExportTestBase {
 protected:
  inline const static std::string AWS_DUMMY_CREDENTIALS_DIR =
      to_string(BASE_PATH) + "/aws";
  inline static std::map<std::string, std::string> aws_environment_;

  static void SetUpTestSuite() {
    omnisci_aws_sdk::init_sdk();
    g_allow_s3_server_privileges = true;
    aws_environment_ = unset_aws_env();
    create_stub_aws_profile(AWS_DUMMY_CREDENTIALS_DIR);
  }

  static void TearDownTestSuite() {
    omnisci_aws_sdk::shutdown_sdk();
    g_allow_s3_server_privileges = false;
    restore_aws_env(aws_environment_);
    boost::filesystem::remove_all(AWS_DUMMY_CREDENTIALS_DIR);
  }

  void SetUp() override {
    ImportExportTestBase::SetUp();
    sql("drop table if exists test_table_1;");
    sql("create table test_table_1(C1 Int, C2 Text "
        "Encoding None, C3 Text Encoding None)");
  }

  void TearDown() override {
    sql("drop table test_table_1;");
    ImportExportTestBase::TearDown();
  }

  void importPublicBucket() {
    std::string query_stmt =
        "copy test_table_1 from 's3://omnisci-fsi-test-public/FsiDataFiles/0_255.csv';";
    sql(query_stmt);
  }

  void importPrivateBucket(std::string s3_access_key = "",
                           std::string s3_secret_key = "",
                           std::string s3_session_token = "",
                           std::string s3_region = "us-west-1") {
    std::string query_stmt =
        "copy test_table_1 from 's3://omnisci-fsi-test/FsiDataFiles/0_255.csv' WITH(";
    if (s3_access_key.size()) {
      query_stmt += "s3_access_key='" + s3_access_key + "', ";
    }
    if (s3_secret_key.size()) {
      query_stmt += "s3_secret_key='" + s3_secret_key + "', ";
    }
    if (s3_session_token.size()) {
      query_stmt += "s3_session_token='" + s3_session_token + "', ";
    }
    if (s3_region.size()) {
      query_stmt += "s3_region='" + s3_region + "'";
    }
    query_stmt += ");";
    sql(query_stmt);
  }
};

TEST_F(ImportServerPrivilegeTest, S3_Public_without_credentials) {
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_NO_THROW(importPublicBucket());
}

TEST_F(ImportServerPrivilegeTest, S3_Private_without_credentials) {
  if (is_valid_aws_role()) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_THROW(importPrivateBucket(), TOmniSciException);
}

TEST_F(ImportServerPrivilegeTest, S3_Private_with_invalid_specified_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_THROW(importPrivateBucket("invalid_key", "invalid_secret"), TOmniSciException);
}

TEST_F(ImportServerPrivilegeTest, S3_Private_with_valid_specified_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  const auto aws_access_key_id = aws_environment_.find("AWS_ACCESS_KEY_ID")->second;
  const auto aws_secret_access_key =
      aws_environment_.find("AWS_SECRET_ACCESS_KEY")->second;
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_NO_THROW(importPrivateBucket(aws_access_key_id, aws_secret_access_key));
}

TEST_F(ImportServerPrivilegeTest, S3_Private_with_env_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  restore_aws_keys(aws_environment_);
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_NO_THROW(importPrivateBucket());
  unset_aws_keys();
}

TEST_F(ImportServerPrivilegeTest, S3_Private_with_profile_credentials) {
  if (!is_valid_aws_key(aws_environment_)) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, true, aws_environment_);
  EXPECT_NO_THROW(importPrivateBucket());
}

TEST_F(ImportServerPrivilegeTest, S3_Private_with_role_credentials) {
  if (!is_valid_aws_role()) {
    GTEST_SKIP();
  }
  set_aws_profile(AWS_DUMMY_CREDENTIALS_DIR, false);
  EXPECT_NO_THROW(importPrivateBucket());
}
#endif  // HAVE_AWS_S3

class SortedImportTest
    : public ImportExportTestBase,
      public testing::WithParamInterface<std::tuple<std::string, std::string>> {
 public:
#ifdef HAVE_AWS_S3
  static void SetUpTestSuite() { omnisci_aws_sdk::init_sdk(); }

  static void TearDownTestSuite() { omnisci_aws_sdk::shutdown_sdk(); }
#endif

  void SetUp() override {
    ImportExportTestBase::SetUp();
    locality_ = get<0>(GetParam());
    file_type_ = get<1>(GetParam());
    ASSERT_NO_THROW(sql("drop table if exists test_table_1;"));
    ASSERT_NO_THROW(sql("create table test_table_1(C1 Int);"));
  }

  void TearDown() override {
    ASSERT_NO_THROW(sql("drop table test_table_1;"));
    ImportExportTestBase::TearDown();
  }

 protected:
  string locality_;
  string file_type_;

  std::string getSourceDir() {
    if (locality_ == "local") {
      return "../../Tests/FsiDataFiles/sorted_dir/" + file_type_ + "/";
    }
    CHECK(locality_ == "s3");
    return "s3://omnisci-fsi-test-public/FsiDataFiles/sorted_dir/" + file_type_ + "/";
  }

  std::string createCopyFromQuery(map<std::string, std::string> options = {},
                                  std::string source_dir = "") {
    source_dir = source_dir.empty() ? getSourceDir() : source_dir;
    if (file_type_ == "csv") {
      options.emplace("parquet", "false");
    } else {
      CHECK(file_type_ == "parquet");
      options.emplace("parquet", "true");
    }
    options.emplace("HEADER", "true");
    return "COPY test_table_1 FROM '" + source_dir + "' WITH (" +
           options_to_string(options, false) + ");";
  }
};

INSTANTIATE_TEST_SUITE_P(SortedImportTest,
                         SortedImportTest,
                         testing::Combine(testing::Values("local"),
                                          testing::Values("csv", "parquet")));

#ifdef HAVE_AWS_S3
INSTANTIATE_TEST_SUITE_P(S3SortedImportTest,
                         SortedImportTest,
                         testing::Combine(testing::Values("s3"),
                                          testing::Values("csv", "parquet")));
#endif  // HAVE_AWS_S3

TEST_P(SortedImportTest, SortedOnPathname) {
  sql(createCopyFromQuery({{"FILE_SORT_ORDER_BY", "pathNAME"}}));
  sqlAndCompareResult("SELECT * FROM test_table_1;", {{i(2)}, {i(1)}, {i(0)}, {i(9)}});
}

TEST_P(SortedImportTest, SortedOnDateModified) {
  if (locality_ == "local") {
    auto source_dir =
        boost::filesystem::absolute("../../Tests/FsiDataFiles/sorted_dir/" + file_type_);
    auto temp_dir = boost::filesystem::absolute("temp_sorted_on_date_modified");
    boost::filesystem::remove_all(temp_dir);
    boost::filesystem::copy(source_dir, temp_dir);

    // some platforms won't copy directory contents on a directory copy
    for (auto& file : boost::filesystem::recursive_directory_iterator(source_dir)) {
      auto source_file = file.path();
      auto dest_file = temp_dir / file.path().filename();
      if (!boost::filesystem::exists(dest_file)) {
        boost::filesystem::copy(file.path(), temp_dir / file.path().filename());
      }
    }
    auto reference_time = boost::filesystem::last_write_time(temp_dir);
    boost::filesystem::last_write_time(temp_dir / ("zzz." + file_type_),
                                       reference_time - 2);
    boost::filesystem::last_write_time(temp_dir / ("a_21_2021-01-01." + file_type_),
                                       reference_time - 1);
    boost::filesystem::last_write_time(temp_dir / ("c_00_2021-02-15." + file_type_),
                                       reference_time);
    boost::filesystem::last_write_time(temp_dir / ("b_15_2021-12-31." + file_type_),
                                       reference_time + 1);

    sql(createCopyFromQuery({{"FILE_SORT_ORDER_BY", "DATE_MODIFIED"}},
                            temp_dir.string()));
    sqlAndCompareResult("SELECT * FROM test_table_1;", {{i(9)}, {i(2)}, {i(0)}, {i(1)}});

    boost::filesystem::remove_all(temp_dir);
  } else {
    sql(createCopyFromQuery({{"FILE_SORT_ORDER_BY", "DATE_MODIFIED"}}));
    sqlAndCompareResult("SELECT * FROM test_table_1;", {{i(9)}, {i(2)}, {i(0)}, {i(1)}});
  }
}

TEST_P(SortedImportTest, SortedOnRegex) {
  sql(createCopyFromQuery(
      {{"FILE_SORT_ORDER_BY", "REGEX"}, {"FILE_SORT_REGEX", ".*[a-z]_[0-9]([0-9])_.*"}}));
  sqlAndCompareResult("SELECT * FROM test_table_1;", {{i(9)}, {i(0)}, {i(2)}, {i(1)}});
}

TEST_P(SortedImportTest, SortedOnRegexDate) {
  sql(createCopyFromQuery({{"FILE_SORT_ORDER_BY", "REGEX_DATE"},
                           {"FILE_SORT_REGEX", ".*[a-z]_[0-9][0-9]_(.*)\\..*"}}));
  sqlAndCompareResult("SELECT * FROM test_table_1;", {{i(9)}, {i(2)}, {i(0)}, {i(1)}});
}

TEST_P(SortedImportTest, SortedOnRegexNumberAndMultiCaptureGroup) {
  sql(createCopyFromQuery({{"FILE_SORT_ORDER_BY", "REGEX_NUMBER"},
                           {"FILE_SORT_REGEX", ".*[a-z]_(.*)_(.*)-(.*)-(.*)\\..*"}}));
  sqlAndCompareResult("SELECT * FROM test_table_1;", {{i(9)}, {i(0)}, {i(1)}, {i(2)}});
}

TEST_P(SortedImportTest, SortedOnNonRegexWithSortRegex) {
  queryAndAssertException(
      createCopyFromQuery({{"FILE_SORT_REGEX", "xxx"}}),
      "Option \"FILE_SORT_REGEX\" must not be set for selected option "
      "\"FILE_SORT_ORDER_BY='PATHNAME'\".");
}

TEST_P(SortedImportTest, SortedOnRegexWithoutSortRegex) {
  queryAndAssertException(createCopyFromQuery({{"FILE_SORT_ORDER_BY", "REGEX"}}),
                          "Option \"FILE_SORT_REGEX\" must be set for selected option "
                          "\"FILE_SORT_ORDER_BY='REGEX'\".");
}

class ExportTest : public ImportTestGDAL {
 protected:
  void SetUp() override {
    DBHandlerTestFixture::SetUp();
    sql("drop table if exists query_export_test;");
    sql("drop table if exists query_export_test_reimport;");
    ASSERT_NO_THROW(removeAllFilesFromExport());
  }

  void TearDown() override {
    sql("drop table if exists query_export_test;");
    sql("drop table if exists query_export_test_reimport;");
    ASSERT_NO_THROW(removeAllFilesFromExport());
    DBHandlerTestFixture::TearDown();
  }

  void removeAllFilesFromExport() {
    boost::filesystem::path path_to_remove(BASE_PATH "/mapd_export/");
    if (boost::filesystem::exists(path_to_remove)) {
      for (boost::filesystem::directory_iterator end_dir_it, it(path_to_remove);
           it != end_dir_it;
           ++it) {
        boost::filesystem::remove_all(it->path());
      }
    }
  }

  // clang-format off
  constexpr static const char* NON_GEO_COLUMN_NAMES_AND_TYPES =
    "col_big BIGINT,"
    "col_big_var_array BIGINT[],"
    "col_boolean BOOLEAN,"
    "col_boolean_var_array BOOLEAN[],"
    "col_date DATE ENCODING DAYS(32),"
    "col_date_var_array DATE[],"
    "col_decimal DECIMAL(8,2) ENCODING FIXED(32),"
    "col_decimal_var_array DECIMAL(8,2)[],"
    "col_dict_none1 TEXT ENCODING NONE,"
    "col_dict_text1 TEXT ENCODING DICT(32),"
    "col_dict_var_array TEXT[] ENCODING DICT(32),"
    "col_double DOUBLE,"
    "col_double_var_array DOUBLE[],"
    "col_float FLOAT,"
    "col_float_var_array FLOAT[],"
    "col_integer INTEGER,"
    "col_integer_var_array INTEGER[],"
    "col_numeric DECIMAL(8,2) ENCODING FIXED(32),"
    "col_numeric_var_array DECIMAL(8,2)[],"
    "col_small SMALLINT,"
    "col_small_var_array SMALLINT[],"
    "col_time TIME,"
    "col_time_var_array TIME[],"
    "col_tiny TINYINT,"
    "col_tiny_var_array TINYINT[],"
    "col_ts0 TIMESTAMP(0),"
    "col_ts0_var_array TIMESTAMP[],"
    "col_ts3 TIMESTAMP(3),"
    "col_ts6 TIMESTAMP(6),"
    "col_ts9 TIMESTAMP(9)";
  constexpr static const char* GEO_COLUMN_NAMES_AND_TYPES =
    "col_point GEOMETRY(POINT, 4326),"
    "col_linestring GEOMETRY(LINESTRING, 4326),"
    "col_polygon GEOMETRY(POLYGON, 4326),"
    "col_multipolygon GEOMETRY(MULTIPOLYGON, 4326)";
  constexpr static const char* NON_GEO_COLUMN_NAMES =
    "col_big,"
    "col_big_var_array,"
    "col_boolean,"
    "col_boolean_var_array,"
    "col_date,"
    "col_date_var_array,"
    "col_decimal,"
    "col_decimal_var_array,"
    "col_dict_none1,"
    "col_dict_text1,"
    "col_dict_var_array,"
    "col_double,"
    "col_double_var_array,"
    "col_float,"
    "col_float_var_array,"
    "col_integer,"
    "col_integer_var_array,"
    "col_numeric,"
    "col_numeric_var_array,"
    "col_small,"
    "col_small_var_array,"
    "col_time,"
    "col_time_var_array,"
    "col_tiny,"
    "col_tiny_var_array,"
    "col_ts0,"
    "col_ts0_var_array,"
    "col_ts3,"
    "col_ts6,"
    "col_ts9";
  constexpr static const char* NON_GEO_COLUMN_NAMES_NO_ARRAYS =
    "col_big,"
    "col_boolean,"
    "col_date,"
    "col_decimal,"
    "col_dict_none1,"
    "col_dict_text1,"
    "col_double,"
    "col_float,"
    "col_integer,"
    "col_numeric,"
    "col_small,"
    "col_time,"
    "col_tiny,"
    "col_ts0,"
    "col_ts3,"
    "col_ts6,"
    "col_ts9";
  // clang-format on

  void doCreateAndImport() {
    ASSERT_NO_THROW(sql(std::string("CREATE TABLE query_export_test (") +
                        NON_GEO_COLUMN_NAMES_AND_TYPES + ", " +
                        GEO_COLUMN_NAMES_AND_TYPES + ");"));
    ASSERT_NO_THROW(sql(
        "COPY query_export_test FROM "
        "'../../Tests/Export/QueryExport/datafiles/query_export_test_source.csv' WITH "
        "(header='true', array_delimiter='|');"));
  }

  void doExport(const std::string& file_path,
                const std::string& file_type,
                const std::string& file_compression,
                const std::string& geo_type,
                const bool with_array_columns,
                const bool force_invalid_srid) {
    std::string ddl = "COPY (SELECT ";
    ddl += (with_array_columns ? NON_GEO_COLUMN_NAMES : NON_GEO_COLUMN_NAMES_NO_ARRAYS);
    ddl += ", ";
    if (force_invalid_srid) {
      ddl += "ST_SetSRID(col_" + geo_type + ", 0)";
    } else {
      ddl += "col_" + geo_type;
    }
    ddl += " FROM query_export_test) TO '" + file_path + "'";

    auto f = (file_type.size() > 0);
    auto c = (file_compression.size() > 0);
    if (f || c) {
      ddl += " WITH (";
      if (f) {
        ddl += "file_type='" + file_type + "'";
        if (file_type == "CSV") {
          ddl += ", header='true'";
        }
      }
      if (f && c) {
        ddl += ", ";
      }
      if (c) {
        ddl += "file_compression='" + file_compression + "'";
      }
      ddl += ")";
    }

    ddl += ";";

    sql(ddl);
  }

  void doImportAgainAndCompare(const std::string& file,
                               const std::string& file_type,
                               const std::string& geo_type,
                               const bool with_array_columns) {
    // re-import exported file(s) to new table
    auto actual_file =
        BASE_PATH "/mapd_export/" + getDbHandlerAndSessionId().second + "/" + file;
    actual_file = boost::filesystem::canonical(actual_file).string();
    if (file_type == "" || file_type == "CSV") {
      // create table
      std::string ddl = "CREATE TABLE query_export_test_reimport (";
      ddl += NON_GEO_COLUMN_NAMES_AND_TYPES;
      if (geo_type == "point") {
        ddl += ", col_point GEOMETRY(POINT, 4326));";
      } else if (geo_type == "linestring") {
        ddl += ", col_linestring GEOMETRY(LINESTRING, 4326));";
      } else if (geo_type == "polygon") {
        ddl += ", col_polygon GEOMETRY(POLYGON, 4326));";
      } else if (geo_type == "multipolygon") {
        ddl += ", col_multipolygon GEOMETRY(MULTIPOLYGON, 4326));";
      } else {
        CHECK(false);
      }
      ASSERT_NO_THROW(sql(ddl));

      // import to that table
      auto import_options = std::string("array_delimiter='|', header=") +
                            (file_type == "CSV" ? "'true'" : "'false'");
      ASSERT_NO_THROW(sql("COPY query_export_test_reimport FROM '" + actual_file +
                          "' WITH (" + import_options + ");"));
    } else {
      ASSERT_NO_THROW(
          importGeoTable(actual_file, "query_export_test_reimport", false, true, false));
    }

    // select a comparable value from the first row
    // tolerate any re-ordering due to export query non-determinism
    // scope this block so that the ResultSet is destroyed before the table is dropped
    {
      sqlAndCompareResult(
          "SELECT col_big FROM query_export_test_reimport ORDER BY col_big",
          {{20395569495L},
           {31334726270L},
           {31851544292L},
           {53000912292L},
           {84212876526L}});
    }

    // drop the table
    ASSERT_NO_THROW(sql("drop table query_export_test_reimport;"));
  }

  void doCompareBinary(const std::string& file, const bool gzipped) {
    if (!g_regenerate_export_test_reference_files) {
      auto actual_exported_file =
          BASE_PATH "/mapd_export/" + getDbHandlerAndSessionId().second + "/" + file;
      auto actual_reference_file = "../../Tests/Export/QueryExport/datafiles/" + file;
      auto exported_file_contents = readBinaryFile(actual_exported_file, gzipped);
      auto reference_file_contents = readBinaryFile(actual_reference_file, gzipped);
      ASSERT_EQ(exported_file_contents, reference_file_contents);
    }
  }

  void doCompareText(const std::string& file, const bool gzipped) {
    if (!g_regenerate_export_test_reference_files) {
      auto actual_exported_file =
          BASE_PATH "/mapd_export/" + getDbHandlerAndSessionId().second + "/" + file;
      auto actual_reference_file = "../../Tests/Export/QueryExport/datafiles/" + file;
      auto exported_lines = readTextFile(actual_exported_file, gzipped);
      auto reference_lines = readTextFile(actual_reference_file, gzipped);
      // sort lines to account for query output order non-determinism
      std::sort(exported_lines.begin(), exported_lines.end());
      std::sort(reference_lines.begin(), reference_lines.end());
      // compare, ignoring any comma moved by the sort
      compareLines(exported_lines, reference_lines, COMPARE_IGNORING_COMMA_DIFF);
    }
  }

  void doCompareWithOGRInfo(const std::string& file,
                            const std::string& layer_name,
                            const bool ignore_trailing_comma_diff) {
    if (!g_regenerate_export_test_reference_files) {
      auto actual_exported_file =
          BASE_PATH "/mapd_export/" + getDbHandlerAndSessionId().second + "/" + file;
      auto actual_reference_file = "../../Tests/Export/QueryExport/datafiles/" + file;
      auto exported_lines = readFileWithOGRInfo(actual_exported_file, layer_name);
      auto reference_lines = readFileWithOGRInfo(actual_reference_file, layer_name);
      // sort lines to account for query output order non-determinism
      std::sort(exported_lines.begin(), exported_lines.end());
      std::sort(reference_lines.begin(), reference_lines.end());
      compareLines(exported_lines, reference_lines, ignore_trailing_comma_diff);
    }
  }

  void removeExportedFile(const std::string& file) {
    auto exported_file =
        BASE_PATH "/mapd_export/" + getDbHandlerAndSessionId().second + "/" + file;
    if (g_regenerate_export_test_reference_files) {
      auto reference_file = "../../Tests/Export/QueryExport/datafiles/" + file;
      ASSERT_NO_THROW(boost::filesystem::copy_file(
          exported_file,
          reference_file,
          boost::filesystem::copy_option::overwrite_if_exists));
    }
    ASSERT_NO_THROW(boost::filesystem::remove(exported_file));
  }

  void doTestArrayNullHandling(const std::string& file,
                               const std::string& other_options) {
    std::string exp_file =
        BASE_PATH "/mapd_export/" + getDbHandlerAndSessionId().second + "/" + file;
    ASSERT_NO_THROW(
        sql("CREATE TABLE query_export_test (col_int INTEGER, "
            "col_int_var_array INTEGER[], col_point GEOMETRY(POINT, 4326));"));
    ASSERT_NO_THROW(
        sql("COPY query_export_test FROM "
            "'../../Tests/Export/QueryExport/datafiles/"
            "query_export_test_array_null_handling.csv' WITH "
            "(header='true', array_delimiter='|');"));
    // this may or may not throw
    sql("COPY (SELECT * FROM query_export_test) TO '" + exp_file +
        "' WITH (file_type='GeoJSON'" + other_options + ");");
    ASSERT_NO_THROW(doCompareText(file, PLAIN_TEXT));
    ASSERT_NO_THROW(removeExportedFile(file));
  }

  void doTestNulls(const std::string& file,
                   const std::string& file_type,
                   const std::string& select) {
    std::string exp_file =
        BASE_PATH "/mapd_export/" + getDbHandlerAndSessionId().second + "/" + file;
    ASSERT_NO_THROW(
        sql("CREATE TABLE query_export_test (a GEOMETRY(POINT, 4326), b "
            "GEOMETRY(LINESTRING, 4326), c GEOMETRY(POLYGON, 4326), d "
            "GEOMETRY(MULTIPOLYGON, 4326));"));
    ASSERT_NO_THROW(
        sql("COPY query_export_test FROM "
            "'../../Tests/Export/QueryExport/datafiles/"
            "query_export_test_nulls.csv' WITH (header='true');"));
    ASSERT_NO_THROW(sql("COPY (SELECT " + select + " FROM query_export_test) TO '" +
                        exp_file + "' WITH (file_type='" + file_type + "');"));
    ASSERT_NO_THROW(doCompareText(file, PLAIN_TEXT));
    ASSERT_NO_THROW(removeExportedFile(file));
    ASSERT_NO_THROW(sql("DROP TABLE query_export_test;"));
  }

  constexpr static bool WITH_ARRAYS = true;
  constexpr static bool NO_ARRAYS = false;
  constexpr static bool INVALID_SRID = true;
  constexpr static bool DEFAULT_SRID = false;
  constexpr static bool GZIPPED = true;
  constexpr static bool PLAIN_TEXT = false;
  constexpr static bool COMPARE_IGNORING_COMMA_DIFF = true;
  constexpr static bool COMPARE_EXPLICIT = false;

  constexpr static std::array<const char*, 4> GEO_TYPES = {"point",
                                                           "linestring",
                                                           "polygon",
                                                           "multipolygon"};

 private:
  std::vector<std::string> readTextFile(
      const std::string& file,
      const bool gzipped,
      const std::vector<std::string>& skip_lines_containing_any_of = {}) {
    std::vector<std::string> lines;
    if (gzipped) {
      std::ifstream in_stream(file, std::ios_base::in | std::ios_base::binary);
      boost::iostreams::filtering_streambuf<boost::iostreams::input> in_buf;
      in_buf.push(boost::iostreams::gzip_decompressor());
      in_buf.push(in_stream);
      std::istream in_stream_gunzip(&in_buf);
      std::string line;
      while (std::getline(in_stream_gunzip, line)) {
        if (!lineContainsAnyOf(line, skip_lines_containing_any_of)) {
          lines.push_back(line);
        }
      }
      in_stream.close();
    } else {
      std::ifstream in_stream(file, std::ios_base::in);
      std::string line;
      while (std::getline(in_stream, line)) {
        if (!lineContainsAnyOf(line, skip_lines_containing_any_of)) {
          lines.push_back(line);
        }
      }
      in_stream.close();
    }
    return lines;
  }

  std::string readBinaryFile(const std::string& file, const bool gzipped) {
    std::stringstream buffer;
    if (gzipped) {
      std::ifstream in_stream(file, std::ios_base::in | std::ios_base::binary);
      boost::iostreams::filtering_streambuf<boost::iostreams::input> in_buf;
      in_buf.push(boost::iostreams::gzip_decompressor());
      in_buf.push(in_stream);
      std::istream in_stream_gunzip(&in_buf);
      buffer << in_stream_gunzip.rdbuf();
      in_stream.close();
    } else {
      std::ifstream in_stream(file, std::ios_base::in);
      buffer << in_stream.rdbuf();
      in_stream.close();
    }
    return buffer.str();
  }

  std::vector<std::string> readFileWithOGRInfo(const std::string& file,
                                               const std::string& layer_name) {
    std::string temp_file = BASE_PATH "/mapd_export/" + std::to_string(getpid()) + ".tmp";
    std::string ogrinfo_cmd = "ogrinfo " + file + " " + layer_name;
    boost::process::system(ogrinfo_cmd, boost::process::std_out > temp_file);
    auto lines =
        readTextFile(temp_file, false, {"DBF_DATE_LAST_UPDATE", "INFO: Open of"});
    boost::filesystem::remove(temp_file);
    return lines;
  }

  void compareLines(const std::vector<std::string>& exported_lines,
                    const std::vector<std::string>& reference_lines,
                    const bool ignore_trailing_comma_diff) {
    ASSERT_NE(exported_lines.size(), 0U);
    ASSERT_NE(reference_lines.size(), 0U);
    ASSERT_EQ(exported_lines.size(), reference_lines.size());
    for (uint32_t i = 0; i < exported_lines.size(); i++) {
      auto const& exported_line = exported_lines[i];
      auto const& reference_line = reference_lines[i];
      // lines from a GeoJSON may differ by trailing comma if the non-deterministic
      // query export row order was different from that of the reference file, as
      // the last data line in the export will not have a trailing comma, so that
      // comma will move after sort even though there are no other differences
      if (ignore_trailing_comma_diff &&
          exported_line.size() == reference_line.size() + 1) {
        ASSERT_EQ(exported_line.substr(0, reference_line.size()), reference_line);
        ASSERT_EQ(exported_line[exported_line.size() - 1], ',');
      } else if (ignore_trailing_comma_diff &&
                 reference_line.size() == exported_line.size() + 1) {
        ASSERT_EQ(exported_line, reference_line.substr(0, exported_line.size()));
        ASSERT_EQ(reference_line[reference_line.size() - 1], ',');
      } else {
        ASSERT_EQ(exported_line, reference_line);
      }
    }
  }

  bool lineContainsAnyOf(const std::string& line,
                         const std::vector<std::string>& tokens) {
    for (auto const& token : tokens) {
      if (line.find(token) != std::string::npos) {
        return true;
      }
    }
    return false;
  }
};

#define RUN_TEST_ON_ALL_GEO_TYPES()        \
  for (const char* geo_type : GEO_TYPES) { \
    run_test(std::string(geo_type));       \
  }

TEST_F(ExportTest, Default) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_csv_no_header_" + geo_type + ".csv";
    ASSERT_NO_THROW(doExport(exp_file, "", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(doCompareText(exp_file, PLAIN_TEXT));
    doImportAgainAndCompare(exp_file, "", geo_type, WITH_ARRAYS);
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, InvalidFileType) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  std::string geo_type = "point";
  std::string exp_file = "query_export_test_csv_" + geo_type + ".csv";
  EXPECT_THROW(doExport(exp_file, "Fred", "", geo_type, WITH_ARRAYS, DEFAULT_SRID),
               TOmniSciException);
}

TEST_F(ExportTest, InvalidCompressionType) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  std::string geo_type = "point";
  std::string exp_file = "query_export_test_csv_" + geo_type + ".csv";
  EXPECT_THROW(doExport(exp_file, "", "Fred", geo_type, WITH_ARRAYS, DEFAULT_SRID),
               TOmniSciException);
}

TEST_F(ExportTest, CSV) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_csv_" + geo_type + ".csv";
    ASSERT_NO_THROW(doExport(exp_file, "CSV", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(doCompareText(exp_file, PLAIN_TEXT));
    doImportAgainAndCompare(exp_file, "CSV", geo_type, WITH_ARRAYS);
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, CSV_Overwrite) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_csv_" + geo_type + ".csv";
    ASSERT_NO_THROW(doExport(exp_file, "CSV", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(doExport(exp_file, "CSV", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, CSV_InvalidName) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  std::string geo_type = "point";
  std::string exp_file = "query_export_test_csv_" + geo_type + ".jpg";
  EXPECT_THROW(doExport(exp_file, "CSV", "", geo_type, WITH_ARRAYS, DEFAULT_SRID),
               TOmniSciException);
}

TEST_F(ExportTest, CSV_Zip_Unimplemented) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_csv_" + geo_type + ".csv";
    EXPECT_THROW(doExport(exp_file, "CSV", "Zip", geo_type, WITH_ARRAYS, DEFAULT_SRID),
                 TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, CSV_GZip_Unimplemented) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojson_" + geo_type + ".geojson";
    EXPECT_THROW(doExport(exp_file, "CSV", "GZip", geo_type, WITH_ARRAYS, DEFAULT_SRID),
                 TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, CSV_Nulls) {
  SKIP_ALL_ON_AGGREGATOR();
  ASSERT_NO_THROW(doTestNulls("query_export_test_csv_nulls.csv", "CSV", "*"));
}

TEST_F(ExportTest, GeoJSON) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojson_" + geo_type + ".geojson";
    ASSERT_NO_THROW(
        doExport(exp_file, "GeoJSON", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(doCompareText(exp_file, PLAIN_TEXT));
    doImportAgainAndCompare(exp_file, "GeoJSON", geo_type, WITH_ARRAYS);
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSON_Overwrite) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojson_" + geo_type + ".geojson";
    ASSERT_NO_THROW(
        doExport(exp_file, "GeoJSON", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(
        doExport(exp_file, "GeoJSON", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSON_InvalidName) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  std::string geo_type = "point";
  std::string exp_file = "query_export_test_geojson_" + geo_type + ".jpg";
  EXPECT_THROW(doExport(exp_file, "GeoJSON", "", geo_type, WITH_ARRAYS, DEFAULT_SRID),
               TOmniSciException);
}

TEST_F(ExportTest, GeoJSON_Invalid_SRID) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojson_" + geo_type + ".geojson";
    EXPECT_THROW(doExport(exp_file, "GeoJSON", "", geo_type, WITH_ARRAYS, INVALID_SRID),
                 TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSON_GZip) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string req_file = "query_export_test_geojson_" + geo_type + ".geojson";
    std::string exp_file = req_file + ".gz";
    ASSERT_NO_THROW(
        doExport(req_file, "GeoJSON", "GZip", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(doCompareText(exp_file, GZIPPED));
    doImportAgainAndCompare(exp_file, "GeoJSON", geo_type, WITH_ARRAYS);
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSON_Zip_Unimplemented) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojson_" + geo_type + ".geojson";
    EXPECT_THROW(
        doExport(exp_file, "GeoJSON", "Zip", geo_type, WITH_ARRAYS, DEFAULT_SRID),
        TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSON_Nulls) {
  SKIP_ALL_ON_AGGREGATOR();
  ASSERT_NO_THROW(
      doTestNulls("query_export_test_geojson_nulls_point.geojson", "GeoJSON", "a"));
  ASSERT_NO_THROW(
      doTestNulls("query_export_test_geojson_nulls_linestring.geojson", "GeoJSON", "b"));
  ASSERT_NO_THROW(
      doTestNulls("query_export_test_geojson_nulls_polygon.geojson", "GeoJSON", "c"));
  ASSERT_NO_THROW(doTestNulls(
      "query_export_test_geojson_nulls_multipolygon.geojson", "GeoJSON", "d"));
}

TEST_F(ExportTest, GeoJSONL_GeoJSON) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojsonl_" + geo_type + ".geojson";
    ASSERT_NO_THROW(
        doExport(exp_file, "GeoJSONL", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(doCompareText(exp_file, PLAIN_TEXT));
    doImportAgainAndCompare(exp_file, "GeoJSONL", geo_type, WITH_ARRAYS);
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSONL_Json) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojsonl_" + geo_type + ".json";
    ASSERT_NO_THROW(
        doExport(exp_file, "GeoJSONL", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(doCompareText(exp_file, PLAIN_TEXT));
    doImportAgainAndCompare(exp_file, "GeoJSONL", geo_type, WITH_ARRAYS);
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSONL_Overwrite) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojsonl_" + geo_type + ".geojson";
    ASSERT_NO_THROW(
        doExport(exp_file, "GeoJSONL", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(
        doExport(exp_file, "GeoJSONL", "", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSONL_InvalidName) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  std::string geo_type = "point";
  std::string exp_file = "query_export_test_geojsonl_" + geo_type + ".jpg";
  EXPECT_THROW(doExport(exp_file, "GeoJSONL", "", geo_type, WITH_ARRAYS, DEFAULT_SRID),
               TOmniSciException);
}

TEST_F(ExportTest, GeoJSONL_Invalid_SRID) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojsonl_" + geo_type + ".geojson";
    EXPECT_THROW(doExport(exp_file, "GeoJSONL", "", geo_type, WITH_ARRAYS, INVALID_SRID),
                 TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSONL_GZip) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string req_file = "query_export_test_geojsonl_" + geo_type + ".geojson";
    std::string exp_file = req_file + ".gz";
    ASSERT_NO_THROW(
        doExport(req_file, "GeoJSONL", "GZip", geo_type, WITH_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(doCompareText(exp_file, GZIPPED));
    doImportAgainAndCompare(exp_file, "GeoJSONL", geo_type, WITH_ARRAYS);
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSONL_Zip_Unimplemented) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojsonl_" + geo_type + ".geojson";
    EXPECT_THROW(
        doExport(exp_file, "GeoJSONL", "Zip", geo_type, WITH_ARRAYS, DEFAULT_SRID),
        TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, GeoJSONL_Nulls) {
  SKIP_ALL_ON_AGGREGATOR();
  ASSERT_NO_THROW(
      doTestNulls("query_export_test_geojsonl_nulls_point.geojson", "GeoJSONL", "a"));
  ASSERT_NO_THROW(doTestNulls(
      "query_export_test_geojsonl_nulls_linestring.geojson", "GeoJSONL", "b"));
  ASSERT_NO_THROW(
      doTestNulls("query_export_test_geojsonl_nulls_polygon.geojson", "GeoJSONL", "c"));
  ASSERT_NO_THROW(doTestNulls(
      "query_export_test_geojsonl_nulls_multipolygon.geojson", "GeoJSONL", "d"));
}

TEST_F(ExportTest, Shapefile) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string shp_file = "query_export_test_shapefile_" + geo_type + ".shp";
    std::string shx_file = "query_export_test_shapefile_" + geo_type + ".shx";
    std::string prj_file = "query_export_test_shapefile_" + geo_type + ".prj";
    std::string dbf_file = "query_export_test_shapefile_" + geo_type + ".dbf";
    ASSERT_NO_THROW(
        doExport(shp_file, "Shapefile", "", geo_type, NO_ARRAYS, DEFAULT_SRID));
    std::string layer_name = "query_export_test_shapefile_" + geo_type;
    ASSERT_NO_THROW(doCompareWithOGRInfo(shp_file, layer_name, COMPARE_EXPLICIT));
    doImportAgainAndCompare(shp_file, "Shapefile", geo_type, NO_ARRAYS);
    removeExportedFile(shp_file);
    removeExportedFile(shx_file);
    removeExportedFile(prj_file);
    removeExportedFile(dbf_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, Shapefile_Overwrite) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string shp_file = "query_export_test_shapefile_" + geo_type + ".shp";
    std::string shx_file = "query_export_test_shapefile_" + geo_type + ".shx";
    std::string prj_file = "query_export_test_shapefile_" + geo_type + ".prj";
    std::string dbf_file = "query_export_test_shapefile_" + geo_type + ".dbf";
    ASSERT_NO_THROW(
        doExport(shp_file, "Shapefile", "", geo_type, NO_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(
        doExport(shp_file, "Shapefile", "", geo_type, NO_ARRAYS, DEFAULT_SRID));
    removeExportedFile(shp_file);
    removeExportedFile(shx_file);
    removeExportedFile(prj_file);
    removeExportedFile(dbf_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, Shapefile_InvalidName) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  std::string geo_type = "point";
  std::string shp_file = "query_export_test_shapefile_" + geo_type + ".jpg";
  EXPECT_THROW(doExport(shp_file, "Shapefile", "", geo_type, NO_ARRAYS, DEFAULT_SRID),
               TOmniSciException);
}

TEST_F(ExportTest, Shapefile_Invalid_SRID) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string shp_file = "query_export_test_shapefile_" + geo_type + ".shp";
    EXPECT_THROW(doExport(shp_file, "Shapefile", "", geo_type, NO_ARRAYS, INVALID_SRID),
                 TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, Shapefile_RejectArrayColumns) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  std::string geo_type = "point";
  std::string shp_file = "query_export_test_shapefile_" + geo_type + ".shp";
  EXPECT_THROW(doExport(shp_file, "Shapefile", "", geo_type, WITH_ARRAYS, DEFAULT_SRID),
               TOmniSciException);
}

TEST_F(ExportTest, Shapefile_GZip_Unimplemented) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string shp_file = "query_export_test_shapefile_" + geo_type + ".shp";
    EXPECT_THROW(
        doExport(shp_file, "Shapefile", "GZip", geo_type, NO_ARRAYS, DEFAULT_SRID),
        TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, Shapefile_Zip_Unimplemented) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string shp_file = "query_export_test_shapefile_" + geo_type + ".shp";
    EXPECT_THROW(
        doExport(shp_file, "Shapefile", "Zip", geo_type, NO_ARRAYS, DEFAULT_SRID),
        TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, FlatGeobuf) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_fgb_" + geo_type + ".fgb";
    ASSERT_NO_THROW(
        doExport(exp_file, "FlatGeobuf", "", geo_type, NO_ARRAYS, DEFAULT_SRID));
    std::string layer_name = "query_export_test_fgb_" + geo_type;
    ASSERT_NO_THROW(doCompareWithOGRInfo(exp_file, layer_name, COMPARE_EXPLICIT));
    doImportAgainAndCompare(exp_file, "FlatGeobuf", geo_type, NO_ARRAYS);
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, FlatGeobuf_Overwrite) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_fgb_" + geo_type + ".fgb";
    ASSERT_NO_THROW(
        doExport(exp_file, "FlatGeobuf", "", geo_type, NO_ARRAYS, DEFAULT_SRID));
    ASSERT_NO_THROW(
        doExport(exp_file, "FlatGeobuf", "", geo_type, NO_ARRAYS, DEFAULT_SRID));
    removeExportedFile(exp_file);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, FlatGeobuf_InvalidName) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  std::string geo_type = "point";
  std::string exp_file = "query_export_test_fgb_" + geo_type + ".jpg";
  EXPECT_THROW(doExport(exp_file, "FlatGeobuf", "", geo_type, NO_ARRAYS, DEFAULT_SRID),
               TOmniSciException);
}

TEST_F(ExportTest, FlatGeobuf_Invalid_SRID) {
  SKIP_ALL_ON_AGGREGATOR();
  doCreateAndImport();
  auto run_test = [&](const std::string& geo_type) {
    std::string exp_file = "query_export_test_geojsonl_" + geo_type + ".fgb";
    EXPECT_THROW(doExport(exp_file, "FlatGeobuf", "", geo_type, NO_ARRAYS, INVALID_SRID),
                 TOmniSciException);
  };
  RUN_TEST_ON_ALL_GEO_TYPES();
}

TEST_F(ExportTest, Array_Null_Handling_Default) {
  SKIP_ALL_ON_AGGREGATOR();
  EXPECT_THROW(doTestArrayNullHandling(
                   "query_export_test_array_null_handling_default.geojson", ""),
               TOmniSciException);
}

TEST_F(ExportTest, Array_Null_Handling_Raw) {
  SKIP_ALL_ON_AGGREGATOR();
  ASSERT_NO_THROW(
      doTestArrayNullHandling("query_export_test_array_null_handling_raw.geojson",
                              ", array_null_handling='raw'"));
}

TEST_F(ExportTest, Array_Null_Handling_Zero) {
  SKIP_ALL_ON_AGGREGATOR();
  ASSERT_NO_THROW(
      doTestArrayNullHandling("query_export_test_array_null_handling_zero.geojson",
                              ", array_null_handling='zero'"));
}

TEST_F(ExportTest, Array_Null_Handling_NullField) {
  SKIP_ALL_ON_AGGREGATOR();
  ASSERT_NO_THROW(
      doTestArrayNullHandling("query_export_test_array_null_handling_nullfield.geojson",
                              ", array_null_handling='nullfield'"));
}

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
      "Print all ImportExportTest specific options (for gtest options use `--help`).");

  desc.add_options()(
      "regenerate-export-test-reference-files",
      po::bool_switch(&g_regenerate_export_test_reference_files)
          ->default_value(g_regenerate_export_test_reference_files)
          ->implicit_value(true),
      "Regenerate Export Test Reference Files (writes to source tree, use with care!)");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: ImportExportTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  if (g_regenerate_export_test_reference_files) {
    // first check we're running in the right directory
    auto write_path =
        boost::filesystem::canonical("../../Tests/Export/QueryExport/datafiles");
    if (!boost::filesystem::is_directory(write_path)) {
      std::cerr << "Failed to locate Export Test Reference Files directory!" << std::endl;
      std::cerr << "Ensure you are running ImportExportTest from $BUILD/Tests!"
                << std::endl;
      return 1;
    }

    // are you sure?
    std::cout << "Will overwrite Export Test Reference Files in:" << std::endl;
    std::cout << "  " << write_path.string() << std::endl;
    std::cout << "Please enter the response 'yes' to confirm:" << std::endl << "> ";
    std::string response;
    std::getline(std::cin, response);
    if (response != "yes") {
      return 0;
    }
    std::cout << std::endl;
  }

  logger::init(log_options);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  g_enable_fsi = false;
  return err;
}
