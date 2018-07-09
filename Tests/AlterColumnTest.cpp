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
#include "Tests/TestHelpers.h"

#include <tuple>
#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 9093

using namespace std;
using namespace Catalog_Namespace;
using namespace TestHelpers;

namespace {

std::unique_ptr<SessionInfo> gsession;
bool g_hoist_literals{true};

inline void run_ddl_statement(const string& input_str) {
  QueryRunner::run_ddl_statement(input_str, gsession);
}

std::shared_ptr<ResultSet> run_query(const string& query_str) {
  return QueryRunner::run_multiple_agg(query_str, gsession, ExecutorDeviceType::CPU, g_hoist_literals, true);
}

bool alter_common(const string& table,
                  const string& column,
                  const string& type,
                  const string& comp,
                  const string& val) {
  std::string alter_query = "alter table " + table + " add column " + column + " " + type;
  if (val != "")
    alter_query += " default " + val;
  if (comp != "")
    alter_query += " encoding " + comp;
  EXPECT_NO_THROW(run_ddl_statement(alter_query + ";"););

  std::string query_str = "SELECT count() FROM " + table + " WHERE " + column +
                          ("" == val || boost::iequals("NULL", val) ? " IS NULL" : (" = " + val));
  auto rows = run_query(query_str + ";");
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  auto r_cnt = v<int64_t>(crt_row[0]);
  return r_cnt == 100;
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
    gsession.reset(QueryRunner::get_session(BASE_PATH,
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
    "            dropoff_latitude        DECIMAL(19,5),"
    "            deleted                 BOOLEAN"
    "            ) WITH (FRAGMENT_SIZE=50);";  // so 2 fragments here

void init_table_data(const string& table = "trips",
                     const string& create_table_cmd = create_table_trips,
                     const string& file = "trip_data_b.txt") {
  run_ddl_statement("drop table if exists " + table + ";");
  run_ddl_statement(create_table_cmd);
  if (file.size())
    import_table_file(table, file);
}

class AlterColumnTest : public ::testing::Test {
 protected:
  virtual void SetUp() { ASSERT_NO_THROW(init_table_data();); }
  virtual void TearDown() { ASSERT_NO_THROW(run_ddl_statement("drop table trips;");); }
};

TEST_F(AlterColumnTest, Add_column) {
#define MT std::make_tuple
  std::vector<std::tuple<std::string, std::string, std::string>> type_vals = {
      MT("text", "none", "'abc'"),
      MT("text", "dict(8)", "'ijk'"),
      MT("text", "dict(32)", "'xyz'"),
      MT("float", "", "1.25"),
      MT("double", "", "1.25"),
      MT("smallint", "", "123"),
      MT("integer", "", "123"),
      MT("bigint", "", "123"),
      MT("decimal(8)", "", "123"),
      MT("decimal(8,2)", "", "1.23"),
      MT("date", "", "'2011-10-23'"),
      MT("time", "", "'10:23:45'"),
      MT("timestamp", "", "'2011-10-23 10:23:45'"),
  };
  int cid = 0;
  for (const auto& tv : type_vals)
    EXPECT_TRUE(alter_common("trips", "x" + std::to_string(++cid), std::get<0>(tv), std::get<1>(tv), std::get<2>(tv)));
  for (const auto& tv : type_vals)
    EXPECT_TRUE(alter_common("trips", "x" + std::to_string(++cid), std::get<0>(tv), std::get<1>(tv), ""));
}
}  // namespace

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
