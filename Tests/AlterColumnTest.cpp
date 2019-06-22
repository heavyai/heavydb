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

#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include "boost/filesystem.hpp"

#include "Catalog/Catalog.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Import/Importer.h"
#include "Parser/parser.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/UpdelRoll.h"
#include "Shared/geo_types.h"
#include "Tests/TestHelpers.h"

#include <tuple>
#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace Catalog_Namespace;
using namespace TestHelpers;

using QR = QueryRunner::QueryRunner;

namespace {

bool g_hoist_literals{true};

inline void run_ddl_statement(const std::string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

std::shared_ptr<ResultSet> run_query(const std::string& query_str) {
  return QR::get()->runSQL(query_str, ExecutorDeviceType::CPU, g_hoist_literals, true);
}

template <typename E = std::runtime_error>
bool alter_common(const std::string& table,
                  const std::string& column,
                  const std::string& type,
                  const std::string& comp,
                  const std::string& val,
                  const std::string& val2,
                  const bool expect_throw = false) {
  std::string alter_query = "alter table " + table + " add column " + column + " " + type;
  if (val != "") {
    alter_query += " default " + val;
  }
  if (comp != "") {
    alter_query += " encoding " + comp;
  }

  if (expect_throw) {
    EXPECT_THROW(run_ddl_statement(alter_query + ";"), E);
    return true;
  } else {
    EXPECT_NO_THROW(run_ddl_statement(alter_query + ";"););
  }

  if (val2 != "") {
    std::string query_str = "SELECT " + column + " FROM " + table;
    auto rows = run_query(query_str + ";");
    int r_cnt = 0;
    while (true) {
      auto crt_row = rows->getNextRow(true, true);
      if (0 == crt_row.size()) {
        break;
      }
      auto geo = boost::get<std::string>(v<NullableString>(crt_row[0]));
#if 1
      if (geo == val2) {
        ++r_cnt;
      }
#else
      // somehow these do not work as advertised ...
      using namespace Geo_namespace;
      if (boost::iequals(type, "POINT") && GeoPoint(geo) == GeoPoint(val2))
        ++r_cnt;
      else if (boost::iequals(type, "LINESTRING") &&
               GeoLineString(geo) == GeoLineString(val2))
        ++r_cnt;
      else if (boost::iequals(type, "POLYGON") && GeoPolygon(geo) == GeoPolygon(val2))
        ++r_cnt;
      else if (boost::iequals(type, "MULTIPOLYGON") &&
               GeoMultiPolygon(geo) == GeoMultiPolygon(val2))
        ++r_cnt;
#endif
    }
    return r_cnt == 100;
  } else {
    std::string query_str =
        "SELECT count() FROM " + table + " WHERE " + column +
        ("" == val || boost::iequals("NULL", val) ? " IS NULL" : (" = " + val));
    auto rows = run_query(query_str + ";");
    auto crt_row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(1), crt_row.size());
    auto r_cnt = v<int64_t>(crt_row[0]);
    return r_cnt == 100;
  }
}

void import_table_file(const std::string& table, const std::string& file) {
  const auto query_str = std::string("COPY trips FROM '") +
                         "../../Tests/Import/datafiles/" + file +
                         "' WITH (header='true');";

  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed)) {
    throw std::runtime_error("Failed to parse: " + query_str);
  }
  CHECK_EQ(parse_trees.size(), size_t(1));

  const auto& stmt = parse_trees.front();
  auto copy_stmt = dynamic_cast<Parser::CopyTableStmt*>(stmt.get());
  if (!copy_stmt) {
    throw std::runtime_error("Expected a CopyTableStatment: " + query_str);
  }
  QR::get()->runImport(copy_stmt);
}

// don't use R"()" format; somehow it causes many blank lines
// to be output on console. how come?
const char* create_table_trips =
    "    CREATE TABLE trips ("
    "            medallion               TEXT ENCODING DICT,"
    "            hack_license            TEXT ENCODING DICT,"
    "            vendor_id               TEXT ENCODING DICT,"
    "            rate_code_id            SMALLINT,"
    "            store_and_fwd_flag      TEXT ENCODING DICT,"
    "            pickup_datetime         TIMESTAMP,"
    "            dropoff_datetime        TIMESTAMP,"
    "            passenger_count         SMALLINT,"
    "            trip_time_in_secs       INTEGER,"
    "            trip_distance           FLOAT,"
    "            pickup_longitude        DECIMAL(14,7),"
    "            pickup_latitude         DECIMAL(14,7),"
    "            dropoff_longitude       DOUBLE,"
    "            dropoff_latitude        DECIMAL(18,5),"
    "            deleted                 BOOLEAN"
    "            ) WITH (FRAGMENT_SIZE=50);";  // so 2 fragments here

void init_table_data(const std::string& table = "trips",
                     const std::string& create_table_cmd = create_table_trips,
                     const std::string& file = "trip_data_b.txt") {
  run_ddl_statement("drop table if exists " + table + ";");
  run_ddl_statement(create_table_cmd);
  if (file.size()) {
    import_table_file(table, file);
  }
}

class AlterColumnTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_NO_THROW(init_table_data();); }
  void TearDown() override { ASSERT_NO_THROW(run_ddl_statement("drop table trips;");); }
};

#define MT std::make_tuple
std::vector<std::tuple<std::string, std::string, std::string, std::string>> type_vals = {
    MT("text", "none", "'abc'", ""),
    MT("text", "dict(8)", "'ijk'", ""),
    MT("text", "dict(32)", "'xyz'", ""),
    MT("float", "", "1.25", ""),
    MT("double", "", "1.25", ""),
    MT("smallint", "", "123", ""),
    MT("integer", "", "123", ""),
    MT("bigint", "", "123", ""),
    MT("bigint encoding fixed(8)", "", "", ""),
    MT("bigint encoding fixed(16)", "", "", ""),
    MT("bigint encoding fixed(32)", "", "", ""),
    MT("decimal(8)", "", "123", ""),
    MT("decimal(8,2)", "", "1.23", ""),
    MT("date", "", "'2011-10-23'", ""),
    MT("time", "", "'10:23:45'", ""),
    MT("timestamp", "", "'2011-10-23 10:23:45'", ""),
    MT("POINT", "", "'POINT (1 2)'", "POINT (1 2)"),
    MT("LINESTRING", "", "'LINESTRING (1 1,2 2,3 3)'", "LINESTRING (1 1,2 2,3 3)"),
    MT("POLYGON",
       "",
       "'POLYGON((0 0,0 9,9 9,9 0),(1 1,2 2,3 3))'",
       "POLYGON ((9 0,9 9,0 9,0 0,9 0),(3 3,2 2,1 1,3 3))"),
    MT("MULTIPOLYGON",
       "",
       "'MULTIPOLYGON(((0 0,0 9,9 9,9 0),(1 1,2 2,3 3)))'",
       "MULTIPOLYGON (((9 0,9 9,0 9,0 0,9 0),(3 3,2 2,1 1,3 3)))"),
};
#undef MT

TEST_F(AlterColumnTest, Add_column_with_default) {
  int cid = 0;
  for (const auto& tv : type_vals) {
    EXPECT_TRUE(alter_common("trips",
                             "x" + std::to_string(++cid),
                             std::get<0>(tv),
                             std::get<1>(tv),
                             std::get<2>(tv),
                             std::get<3>(tv),
                             false));
  }
}

TEST_F(AlterColumnTest, Add_column_with_null) {
  int cid = 0;
  for (const auto& tv : type_vals) {
    if (std::get<3>(tv) == "") {
      EXPECT_TRUE(alter_common("trips",
                               "x" + std::to_string(++cid),
                               std::get<0>(tv),
                               std::get<1>(tv),
                               "",
                               "",
                               false));
    } else {
      EXPECT_TRUE(alter_common("trips",
                               "x" + std::to_string(++cid),
                               std::get<0>(tv),
                               std::get<1>(tv),
                               "",
                               std::get<3>(tv),
                               true));
    }
  }
}

TEST(AlterColumnTest2, Drop_after_fail_to_add) {
  EXPECT_NO_THROW(run_ddl_statement("drop table if exists t;"););
  EXPECT_NO_THROW(run_ddl_statement("create table t(c1 int);"););
  EXPECT_NO_THROW(run_query("insert into t values (10);"););
  EXPECT_THROW(
      run_ddl_statement("alter table t add column c2 TEXT NOT NULL ENCODING DICT;"),
      std::runtime_error);
  EXPECT_NO_THROW(run_ddl_statement("drop table t;"););
}

}  // namespace

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

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
