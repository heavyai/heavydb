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
#include <string>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include "boost/filesystem.hpp"
#include "../Catalog/Catalog.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ResultSet.h"
#include "../QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 39093

using namespace std;
using namespace TestHelpers;

namespace {

std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;
bool g_hoist_literals{true};

inline void run_ddl_statement(const string& input_str) {
  QueryRunner::run_ddl_statement(input_str, g_session);
}

std::shared_ptr<ResultSet> run_query(const string& query_str) {
  return QueryRunner::run_multiple_agg(query_str, g_session, ExecutorDeviceType::CPU, g_hoist_literals, true);
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
  run_ddl_statement(query_str);
  return compare_agg(cnt, avg);
}

bool import_test_common_geo(const string& query_str, const std::string& table, const int64_t cnt, const double avg) {
  // TODO(adb): Return ddl from QueryRunner::run_ddl_statement and use that
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed))
    return false;
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::CopyTableStmt* ddl = dynamic_cast<Parser::CopyTableStmt*>(stmt.get());
  if (!ddl)
    return false;
  ddl->execute(*g_session);

  // was it a geo copy from?
  bool was_geo_copy_from = ddl->was_geo_copy_from();
  if (!was_geo_copy_from)
    return false;

  // get the rest of the payload
  std::string geo_copy_from_table, geo_copy_from_file_name;
  Importer_NS::CopyParams geo_copy_from_copy_params;
  ddl->get_geo_copy_from_payload(geo_copy_from_table, geo_copy_from_file_name, geo_copy_from_copy_params);

  // was it the right table?
  if (geo_copy_from_table != "geo")
    return false;

  // @TODO simon.eves
  // test other stuff
  // filename
  // CopyParams contents

  // success
  return true;
}

void import_test_geofile_importer(const std::string& file_str,
                                  const std::string& table_name,
                                  const bool compression = false) {
  Importer_NS::ImportDriver import_driver(QueryRunner::get_catalog(g_session.get()),
                                          QueryRunner::get_user_metadata(g_session.get()),
                                          ExecutorDeviceType::CPU);

  auto file_path = boost::filesystem::path("../../Tests/Import/datafiles/" + file_str);

  ASSERT_TRUE(boost::filesystem::exists(file_path));

  ASSERT_NO_THROW(import_driver.import_geo_table(file_path.string(), table_name, compression));
}

bool import_test_local(const string& filename, const int64_t cnt, const double avg) {
  return import_test_common(
      string("COPY trips FROM '") + "../../Tests/Import/datafiles/" + filename + "' WITH (header='true');", cnt, avg);
}

bool import_test_local_geo(const string& filename, const string& other_options, const int64_t cnt, const double avg) {
  return import_test_common_geo(string("COPY geo FROM '") + "../../Tests/Import/datafiles/" + filename +
                                    "' WITH (geo='true'" + other_options + ");",
                                "geo",
                                cnt,
                                avg);
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
    g_session.reset(QueryRunner::get_session(BASE_PATH,
                                             "gtest",
                                             "test!test!",
                                             "gtest_db",
                                             std::vector<LeafHostInfo>{},
                                             std::vector<LeafHostInfo>{},
                                             false,
                                             true,
                                             true));
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
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_trips););
  }

  virtual void TearDown() {
    ASSERT_NO_THROW(run_ddl_statement("drop table trips;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geo;"););
  }
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

// Sharding tests
const char* create_table_trips_sharded =
    "	CREATE TABLE trips (									"
    "     id                      INTEGER,            "
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
    "			dropoff_latitude        DECIMAL(14,2),		"
    "     shard key (id)                           "
    "			) WITH (FRAGMENT_SIZE=75000000, SHARD_COUNT=2);";
class ImportTestSharded : public ::testing::Test {
 protected:
  virtual void SetUp() {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_trips_sharded););
  }

  virtual void TearDown() {
    ASSERT_NO_THROW(run_ddl_statement("drop table trips;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geo;"););
  }
};

TEST_F(ImportTestSharded, One_csv_file) {
  EXPECT_TRUE(import_test_local("sharded_trip_data_9.csv", 100, 1.0));
}

namespace {
const char* create_table_geo =
    "  CREATE TABLE geospatial ("
    "   p1 POINT,"
    "   l LINESTRING,"
    "   poly POLYGON,"
    "   mpoly MULTIPOLYGON,"
    "   p2 POINT,"
    "   p3 POINT,"
    "   p4 POINT,"
    "   trip_distance DOUBLE"
    " ) WITH (FRAGMENT_SIZE=65000000);";

void check_geo_import() {
  auto rows =
      run_query("SELECT p1, l, poly, mpoly, p2, p3, p4, trip_distance FROM geospatial WHERE trip_distance = 1.0");
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(8), crt_row.size());
  const auto p1 = boost::get<std::string>(v<NullableString>(crt_row[0]));
  ASSERT_EQ("POINT (1 1)", p1);
  const auto linestring = boost::get<std::string>(v<NullableString>(crt_row[1]));
  ASSERT_EQ("LINESTRING (1 0,2 2,3 3)", linestring);
  const auto poly = boost::get<std::string>(v<NullableString>(crt_row[2]));
  ASSERT_EQ("POLYGON ((0 0,2 0,0 2,0 0))", poly);
  const auto mpoly = boost::get<std::string>(v<NullableString>(crt_row[3]));
  ASSERT_EQ("MULTIPOLYGON (((0 0,2 0,0 2,0 0)))", mpoly);
  const auto p2 = boost::get<std::string>(v<NullableString>(crt_row[4]));
  ASSERT_EQ("POINT (1 1)", p2);
  const auto p3 = boost::get<std::string>(v<NullableString>(crt_row[5]));
  ASSERT_EQ("POINT (1 1)", p3);
  const auto p4 = boost::get<std::string>(v<NullableString>(crt_row[6]));
  ASSERT_EQ("POINT (1 1)", p4);
  const auto trip_distance = v<double>(crt_row[7]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

void check_geo_gdal_point_import() {
  auto rows = run_query(
      "SELECT mapd_geo, trip FROM geospatial WHERE "
      "trip = 1.0");
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  const auto point = boost::get<std::string>(v<NullableString>(crt_row[0]));
  ASSERT_EQ("POINT (1 1)", point);
  const auto trip_distance = v<double>(crt_row[1]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

void check_geo_gdal_mpoly_import() {
  auto rows = run_query(
      "SELECT mapd_geo, trip FROM geospatial WHERE "
      "trip = 1.0");
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(2), crt_row.size());
  const auto mpoly = boost::get<std::string>(v<NullableString>(crt_row[0]));
  ASSERT_EQ("MULTIPOLYGON (((0 0,2 0,0 2,0 0)))", mpoly);
  const auto trip_distance = v<double>(crt_row[1]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

void check_geo_gdal_point_coords_import() {
  auto rows = run_query(
      "SELECT mapd_geo, trip FROM geospatial WHERE "
      "trip = 1.0");
  rows->setGeoReturnDouble();
  auto crt_row = rows->getNextRow(true, true);
  compare_array(crt_row[0], std::vector<double>{1.0, 1.0}, 1e-7);
  const auto trip_distance = v<double>(crt_row[1]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}

void check_geo_gdal_mpoly_coords_import() {
  auto rows = run_query(
      "SELECT mapd_geo, trip FROM geospatial WHERE "
      "trip = 1.0");
  rows->setGeoReturnDouble();
  auto crt_row = rows->getNextRow(true, true);
  compare_array(crt_row[0], std::vector<double>{0.0, 0.0, 2.0, 0.0, 0.0, 2.0}, 1e-7);
  const auto trip_distance = v<double>(crt_row[1]);
  ASSERT_NEAR(1.0, trip_distance, 1e-7);
}
}

class GeoImportTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geospatial;"););
    ASSERT_NO_THROW(run_ddl_statement(create_table_geo););
  }

  virtual void TearDown() {
    ASSERT_NO_THROW(run_ddl_statement("drop table geospatial;"););
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists geospatial;"););
  }
};

TEST_F(GeoImportTest, Geo_CSV_Local_Default) {
  EXPECT_TRUE(import_test_local_geo("geospatial.csv", "", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_Type_Geometry) {
  EXPECT_TRUE(import_test_local_geo("geospatial.csv", ", geo_coords_type='geometry'", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_Type_Geography) {
  EXPECT_THROW(import_test_local_geo("geospatial.csv", ", geo_coords_type='geography'", 10, 4.5), std::runtime_error);
}

TEST_F(GeoImportTest, Geo_CSV_Local_Type_Other) {
  EXPECT_THROW(import_test_local_geo("geospatial.csv", ", geo_coords_type='other'", 10, 4.5), std::runtime_error);
}

TEST_F(GeoImportTest, Geo_CSV_Local_Encoding_NONE) {
  EXPECT_TRUE(import_test_local_geo("geospatial.csv", ", geo_coords_encoding='none'", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_Encoding_GEOINT32) {
  EXPECT_TRUE(import_test_local_geo("geospatial.csv", ", geo_coords_encoding='compressed(32)'", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_Encoding_Other) {
  EXPECT_THROW(import_test_local_geo("geospatial.csv", ", geo_coords_encoding='other'", 10, 4.5), std::runtime_error);
}

TEST_F(GeoImportTest, Geo_CSV_Local_SRID_LonLat) {
  EXPECT_TRUE(import_test_local_geo("geospatial.csv", ", geo_coords_srid=4326", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_SRID_Mercator) {
  EXPECT_TRUE(import_test_local_geo("geospatial.csv", ", geo_coords_srid=900913", 10, 4.5));
}

TEST_F(GeoImportTest, Geo_CSV_Local_SRID_Other) {
  EXPECT_THROW(import_test_local_geo("geospatial.csv", ", geo_coords_srid=12345", 10, 4.5), std::runtime_error);
}

TEST_F(GeoImportTest, CSV_Import) {
  const auto file_path = boost::filesystem::path("../../Tests/Import/datafiles/geospatial.csv");
  run_ddl_statement("COPY geospatial FROM '" + file_path.string() + "';");
  check_geo_import();
}

class GeoGDALImportTest : public ::testing::Test {
 protected:
  virtual void SetUp() { ASSERT_NO_THROW(run_ddl_statement("drop table if exists geospatial;");); }

  virtual void TearDown() { ASSERT_NO_THROW(run_ddl_statement("drop table if exists geospatial;");); }
};

TEST_F(GeoGDALImportTest, Geojson_Point_Import) {
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point.geojson");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_point_import();
}

TEST_F(GeoGDALImportTest, Geojson_MultiPolygon_Import) {
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.geojson");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_mpoly_import();
}

TEST_F(GeoGDALImportTest, Shapefile_Point_Import) {
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_point_import();
}

TEST_F(GeoGDALImportTest, Shapefile_MultiPolygon_Import) {
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_mpoly_import();
}

TEST_F(GeoGDALImportTest, Shapefile_Point_Import_Compressed) {
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", true);
  check_geo_gdal_point_coords_import();
}

TEST_F(GeoGDALImportTest, Shapefile_MultiPolygon_Import_Compressed) {
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", true);
  check_geo_gdal_mpoly_coords_import();
}

TEST_F(GeoGDALImportTest, Shapefile_Point_Import_3857) {
  const auto file_path = boost::filesystem::path("geospatial_point/geospatial_point_3857.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_point_coords_import();
}

TEST_F(GeoGDALImportTest, Shapefile_MultiPolygon_Import_3857) {
  const auto file_path = boost::filesystem::path("geospatial_mpoly/geospatial_mpoly_3857.shp");
  import_test_geofile_importer(file_path.string(), "geospatial", false);
  check_geo_gdal_mpoly_coords_import();
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
