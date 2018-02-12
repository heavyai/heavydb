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

#include "../Import/Importer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include "boost/filesystem.hpp"
#include "../Catalog/Catalog.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ResultSet.h"

#include "QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 9093

using namespace std;
using namespace Catalog_Namespace;

namespace {

std::unique_ptr<SessionInfo> gsession;
bool g_hoist_literals{true};

void run_ddl(const string& input_str) {
  SQLParser parser;
  list<std::unique_ptr<Parser::Stmt>> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(input_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
  CHECK(ddl != nullptr);
  ddl->execute(*gsession);
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
  return run_multiple_agg(query_str, gsession, ExecutorDeviceType::CPU, g_hoist_literals, true);
}

bool compare_agg(const int64_t cnt, const double avg) {
  std::string query_str = "SELECT COUNT(*), AVG(trip_distance) FROM trips;";
  auto rows = run_query(query_str);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  auto r_avg = v<double>(crt_row[1]);
  return r_cnt == cnt && fabs(r_avg - avg) < 1E-9;
}

bool import_test_common(const string& query_str, const int64_t cnt, const double avg) {
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed))
    return false;
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
  if (!ddl)
    return false;
  ddl->execute(*gsession);

  return compare_agg(cnt, avg);
}

bool import_test_local(const string& filename, const int64_t cnt, const double avg) {
  return import_test_common(
      string("COPY trips FROM '") + "../../Tests/Import/datafiles/" + filename + "' WITH (header='true');", cnt, avg);
}

#ifdef HAVE_AWS_S3
bool import_test_s3(const string& prefix, const string& filename, const int64_t cnt, const double avg) {
  // unlikely we will expose any credentials in clear text here.
  // likely credentials will be passed as the "tester"'s env.
  // though s3 sdk should by default access the env, if any,
  // we still read them out to test coverage of the code
  // that passes credentials on per user basis.
  char* env;
  std::string s3_region, s3_access_key, s3_secret_key;
  if (0 != (env = getenv("AWS_REGION")))
    s3_region = env;
  if (0 != (env = getenv("AWS_ACCESS_KEY_ID")))
    s3_access_key = env;
  if (0 != (env = getenv("AWS_SECRET_ACCESS_KEY")))
    s3_secret_key = env;

  return import_test_common(string("COPY trips FROM '") + "s3://mapd-parquet-testdata/" + prefix + "/" + filename +
                                "' WITH (header='true'" +
                                (s3_access_key.size() ? ",s3_access_key='" + s3_access_key + "'" : "") +
                                (s3_secret_key.size() ? ",s3_secret_key='" + s3_secret_key + "'" : "") +
                                (s3_region.size() ? ",s3_region='" + s3_region + "'" : "") + ");",
                            cnt,
                            avg);
}

bool import_test_s3_compressed(const string& filename, const int64_t cnt, const double avg) {
  return import_test_s3("trip.compressed", filename, cnt, avg);
}
#if 0
bool import_test_s3_parquet(const string& filename, const int64_t cnt, const double avg) {
  return import_test_s3("trip.parquet", filename, cnt, avg);
}
#endif
#endif  // HAVE_AWS_S3
class SQLTestEnv : public ::testing::Environment {
 public:
  virtual void SetUp() {
    boost::filesystem::path base_path{BASE_PATH};
    CHECK(boost::filesystem::exists(base_path));
    auto system_db_file = base_path / "mapd_catalogs" / MAPD_SYSTEM_DB;
    auto data_dir = base_path / "mapd_data";
    UserMetadata user;
    DBMetadata db;
    auto calcite = std::make_shared<Calcite>(-1, CALCITEPORT, base_path.string(), 1024);
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
    // if no catalog create one
    auto& sys_cat = SysCatalog::instance();
    if (!boost::filesystem::exists(system_db_file)) {
      sys_cat.init(base_path.string(), dataMgr, {}, calcite, true, false);
      sys_cat.initDB();
    } else {
      sys_cat.init(base_path.string(), dataMgr, {}, calcite, false, false);
    }
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
        std::make_shared<Catalog>(base_path.string(), db, dataMgr, std::vector<LeafHostInfo>{}, calcite),
        user,
        ExecutorDeviceType::GPU,
        ""));
  }
};

std::string TypeToString(SQLTypes type) {
  return SQLTypeInfo(type, false).get_type_name();
}

void d(const SQLTypes expected_type, const std::string& str) {
  auto detected_type = Importer_NS::Detector::detect_sqltype(str);
  EXPECT_EQ(TypeToString(expected_type), TypeToString(detected_type)) << "String: " << str;
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

// don't use R"()" format; somehow it causes many blank lines
// to be output on console. how come?
const char* create_table_trips =
    "	CREATE TABLE trips (									"
    "			medallion               TEXT ENCODING DICT,	"
    "			hack_license            TEXT ENCODING DICT,	"
    "			vendor_id               TEXT ENCODING DICT,	"
    "			rate_code_id            SMALLINT,			"
    "			store_and_fwd_flag      TEXT ENCODING DICT,	"
    "			pickup_datetime         TIMESTAMP,			"
    "			dropoff_datetime        TIMESTAMP,			"
    "			passenger_count         SMALLINT,			"
    "			trip_time_in_secs       INTEGER,				"
    "			trip_distance           DECIMAL(14,2),		"
    "			pickup_longitude        DECIMAL(14,2),		"
    "			pickup_latitude         DECIMAL(14,2),		"
    "			dropoff_longitude       DECIMAL(14,2),		"
    "			dropoff_latitude        DECIMAL(14,2)		"
    "			) WITH (FRAGMENT_SIZE=75000000);				";

class ImportTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    ASSERT_NO_THROW(run_ddl("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl(create_table_trips););
  }

  virtual void TearDown() { ASSERT_NO_THROW(run_ddl("drop table trips;");); }
};

TEST_F(ImportTest, One_csv_file) {
  EXPECT_TRUE(import_test_local("trip_data_9.csv", 100, 1.0));
}

TEST_F(ImportTest, One_gz_file) {
  EXPECT_TRUE(import_test_local("trip_data_9.gz", 100, 1.0));
}

TEST_F(ImportTest, One_bz2_file) {
  EXPECT_TRUE(import_test_local("trip_data_9.bz2", 100, 1.0));
}

TEST_F(ImportTest, Many_csv_file) {
  EXPECT_TRUE(import_test_local("trip_data_*.csv", 1000, 1.0));
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
#endif  // HAVE_AWS_S3
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
