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

#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <memory>
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include <boost/functional/hash.hpp>
#include "../Catalog/Catalog.h"
#include "../Parser/parser.h"
#include "../Analyzer/Analyzer.h"
#include "../Parser/ParserNode.h"
#include "../DataMgr/DataMgr.h"
#include "../Fragmenter/Fragmenter.h"
#include "PopulateTableRandom.h"
#include "ScanTable.h"
#include "gtest/gtest.h"
#include "glog/logging.h"
#include <thread>
#include <future>

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Fragmenter_Namespace;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

// doesnt need real calcite server
#define CALCITEPORT 39093

namespace {
std::unique_ptr<SessionInfo> gsession;
;

void run_ddl(const string& input_str) {
  SQLParser parser;
  list<std::unique_ptr<Parser::Stmt>> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(input_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front().get();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl != nullptr);
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
    auto calcite = std::make_shared<Calcite>(-1, CALCITEPORT, data_dir.string(), 1024);
    {
      auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
      auto& sys_cat = SysCatalog::instance();
      if (!boost::filesystem::exists(system_db_file)) {
        sys_cat.init(base_path.string(), dataMgr, {}, calcite, true, false);
        sys_cat.initDB();
      } else {
        sys_cat.init(base_path.string(), dataMgr, {}, calcite, false, false);
      }
      CHECK(sys_cat.getMetadataForUser(MAPD_ROOT_USER, user));
      if (!sys_cat.getMetadataForUser("gtest", user)) {
        sys_cat.createUser("gtest", "test!test!", false);
        CHECK(sys_cat.getMetadataForUser("gtest", user));
      }
      if (!sys_cat.getMetadataForDB("gtest_db", db)) {
        sys_cat.createDatabase("gtest_db", user.userId);
        CHECK(sys_cat.getMetadataForDB("gtest_db", db));
      }
    }
    auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(), 0, false, 0);
    gsession.reset(new SessionInfo(
        std::make_shared<Catalog>(base_path.string(), db, dataMgr, std::vector<LeafHostInfo>{}, calcite),
        user,
        ExecutorDeviceType::GPU,
        ""));
  }
};

bool load_data_test(string table_name, size_t num_rows) {
  vector<size_t> insert_col_hashs = populate_table_random(table_name, num_rows, gsession->get_catalog());
  return true;
}

#define SMALL 10000000   // - 10M
#define LARGE 100000000  // - 100M

static size_t load_data_for_thread_test_2(int num_rows, string table_name) {
  int initial_num_rows, num_rows_step;
  initial_num_rows = num_rows_step = SMALL / 2;  // insert 5M rows per iteration
  vector<size_t> insert_col_hashs;

  if (num_rows < initial_num_rows) {  // to handle special case when only few rows should be added
    insert_col_hashs = populate_table_random(table_name, num_rows, gsession->get_catalog());
  } else {
    for (int cur_num_rows = initial_num_rows; cur_num_rows <= num_rows; cur_num_rows += num_rows_step) {
      if (cur_num_rows == num_rows) {
        insert_col_hashs = populate_table_random(table_name, num_rows_step, gsession->get_catalog());
      } else {
        populate_table_random(table_name, num_rows_step, gsession->get_catalog());
      }
    }
  }
  return insert_col_hashs.size();
}

}  // namespace

TEST(DataLoad, Numbers) {
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers;"););
  ASSERT_NO_THROW(run_ddl("create table numbers (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  EXPECT_TRUE(load_data_test("numbers", LARGE));
  ASSERT_NO_THROW(run_ddl("drop table numbers;"););
}

TEST(DataLoad, Strings) {
  ASSERT_NO_THROW(run_ddl("drop table if exists strings;"););
  ASSERT_NO_THROW(run_ddl("create table strings (x varchar(10), y text);"););
  EXPECT_TRUE(load_data_test("strings", SMALL));
  ASSERT_NO_THROW(run_ddl("drop table strings;"););
}

TEST(StorageSmall, AllTypes) {
  ASSERT_NO_THROW(run_ddl("drop table if exists alltypes;"););
  ASSERT_NO_THROW(run_ddl("create table alltypes (a smallint, b int, c bigint, d numeric(7,3), e double, f float, "
                          "g timestamp(0), h time(0), i date, x varchar(10), y text);"););
  EXPECT_TRUE(load_data_test("alltypes", SMALL));
  ASSERT_NO_THROW(run_ddl("drop table alltypes;"););
}

TEST(DataLoad, Numbers_Parallel_Load) {
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_1;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_2;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_3;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_4;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_5;"););

  /* create tables in single thread */
  ASSERT_NO_THROW(run_ddl("create table numbers_1 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_2 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_3 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_4 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_5 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););

  /* load data into tables using parallel threads */
  int numThreads = 5;
  vector<string> db_table;
  std::vector<std::future<size_t>> threads;
  string table_name("numbers_");

  int num_rows = SMALL;
  for (int i = 1; i <= numThreads; i++) {
    int num_table_rows = num_rows * (numThreads - i + 1);
    db_table.push_back(table_name + to_string(i));
    threads.push_back(std::async(std::launch::async, load_data_for_thread_test_2, num_table_rows, db_table[i - 1]));
  }

  for (auto& p : threads) {
    int num_columns_inserted = (int)p.get();
    ASSERT_EQ(num_columns_inserted, 6);  // each table was created with 6 columns
  }

  /* delete tables in single thread */
  ASSERT_NO_THROW(run_ddl("drop table numbers_1;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_2;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_3;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_4;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_5;"););
}

TEST(DataLoad, NumbersTable_Parallel_CreateDropTable) {
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_1;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_2;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_3;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_4;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_5;"););

  /* create tables in single thread */
  ASSERT_NO_THROW(run_ddl("create table numbers_1 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_2 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_3 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_4 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_5 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););

  /* Load table numbers_4 with data in the main thread, so it will be available for sure when drop_table on it will be
   * executed later. Don't use new thread for loading table numbers_4 (see commented out), as one can't be sure that
   * this action in the new/independent thread will be completed before executing drop_table in the main thread.
   * It's enough to load just 1 row of data in the table numbers_4 to make sure it exists in the storage layer.
   *
   * threads.push_back(std::async(std::launch::async, load_data_for_thread_test_4, 1, table_name_temp));
   */
  string table_name("numbers_");
  string table_name_temp(table_name + to_string(4));
  EXPECT_TRUE(load_data_test(table_name_temp, 1));

  /* load data into tables numbers_1/2/3/5 using parallel threads */
  int numThreads = 5;
  vector<string> db_table;
  std::vector<std::future<size_t>> threads;
  int num_rows = SMALL;
  for (int i = 1; i <= numThreads; i++) {
    int num_table_rows = num_rows * (numThreads - i + 1);
    db_table.push_back(table_name + to_string(i));
    if (i == 4)
      continue;  // table numbers_4 has been loaded already
    threads.push_back(std::async(std::launch::async, load_data_for_thread_test_2, num_table_rows, db_table[i - 1]));
  }

  /* drop table numbers_4  while loading other tables in independent threads */
  ASSERT_NO_THROW(run_ddl("drop table numbers_4;"););

  /* create table numbers_6 and load it with data */
  ASSERT_NO_THROW(run_ddl("create table numbers_6 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  int num_table_rows = SMALL;
  db_table.push_back(table_name + to_string(6));
  threads.push_back(std::async(std::launch::async, load_data_for_thread_test_2, num_table_rows, db_table[5]));

  for (auto& p : threads) {
    int num_columns_inserted = (int)p.get();
    ASSERT_EQ(num_columns_inserted, 6);  // each table was created with 6 columns
  }

  /* delete tables in single thread */
  ASSERT_NO_THROW(run_ddl("drop table numbers_1;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_2;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_3;"););
  // ASSERT_NO_THROW(run_ddl("drop table numbers_4;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_5;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_6;"););
}

TEST(DataLoad, NumbersTable_Parallel_CreateDropCreateTable_InsertRows) {
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_1;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_2;"););
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers_3;"););

  /* create tables in single thread */
  ASSERT_NO_THROW(run_ddl("create table numbers_1 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_2 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_3 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_4 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  ASSERT_NO_THROW(run_ddl("create table numbers_5 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););

  /* Load table numbers_2 with data in the main thread, so it will be available for sure when drop_table on it will be
   * executed later. Don't use new thread for loading table numbers_2 (see commented out), as one can't be sure that
   * this action in the new/independent thread will be completed before executing drop_table in the main thread.
   * It's enough to load just 1 row of data in the table numbers_2 to make sure it exists in the storage layer.
   *
   * threads.push_back(std::async(std::launch::async, load_data_for_thread_test_2, 1, table_name_temp));
   */
  string table_name("numbers_");
  string table_name_temp(table_name + to_string(2));
  EXPECT_TRUE(load_data_test(table_name_temp, 1));

  /* load data into tables numbers_1/3/4/5 using parallel threads */
  int numThreads = 5;
  vector<string> db_table;
  std::vector<std::future<size_t>> threads;

  int num_rows = SMALL;
  for (int i = 1; i <= numThreads; i++) {
    int num_table_rows = num_rows * (numThreads - i + 1);
    db_table.push_back(table_name + to_string(i));
    if (i == 2)
      continue;  // table numbers_2  has been loaded already
    threads.push_back(std::async(std::launch::async, load_data_for_thread_test_2, num_table_rows, db_table[i - 1]));
  }

  /* drop table numbers_2 while loading other tables in independent threads */
  ASSERT_NO_THROW(run_ddl("drop table numbers_2;"););

  /* create table numbers_6 and load it with data */
  ASSERT_NO_THROW(run_ddl("create table numbers_6 (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  int num_table_rows = SMALL;
  db_table.push_back(table_name + to_string(6));
  threads.push_back(std::async(std::launch::async, load_data_for_thread_test_2, num_table_rows, db_table[5]));

  /* recreate table numbers_2, this table will have new tb_id which will be different from
   * the tb_id of dropped table numbers_2;
   * this is true when new table's schema is same and/or is different than the one for the dropped table.
   */
  ASSERT_NO_THROW(run_ddl("create table numbers_2 (e "
                          "double, f double, g double, h double, i double, j double);"););
  /* insert rows in table numbers_2, this table have been dropped and recreated, so data can be loaded */
  int num_rows_for_dropped_table = SMALL * 2;
  threads.push_back(
      std::async(std::launch::async, load_data_for_thread_test_2, num_rows_for_dropped_table, table_name_temp));

  for (auto& p : threads) {
    int num_columns_inserted = (int)p.get();
    ASSERT_EQ(num_columns_inserted, 6);  // each table was created with 6 columns
  }

  /* delete tables in single thread */
  ASSERT_NO_THROW(run_ddl("drop table numbers_1;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_2;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_3;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_4;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_5;"););
  ASSERT_NO_THROW(run_ddl("drop table numbers_6;"););
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SQLTestEnv);
  return RUN_ALL_TESTS();
}
