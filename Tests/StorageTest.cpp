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

#include <thread>

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

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Fragmenter_Namespace;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#define CALCITEPORT 0

namespace {
std::unique_ptr<SessionInfo> gsession;

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

bool storage_test(const string& table_name, size_t num_rows) {
  vector<size_t> insert_col_hashs = populate_table_random(table_name, num_rows, gsession->get_catalog());
  vector<size_t> scan_col_hashs = scan_table_return_hash(table_name, gsession->get_catalog());
  vector<size_t> scan_col_hashs2 = scan_table_return_hash_non_iter(table_name, gsession->get_catalog());
  return insert_col_hashs == scan_col_hashs && insert_col_hashs == scan_col_hashs2;
}

void simple_thread_wrapper(const string& table_name, size_t num_rows, size_t thread_id) {
  populate_table_random(table_name, num_rows, gsession->get_catalog());
}

bool storage_test_parallel(const string& table_name, size_t num_rows, size_t thread_count) {
  // Constructs a number of threads and have them push records to the table in parallel
  vector<std::thread> myThreads;
  for (size_t i = 0; i < thread_count; i++) {
    myThreads.push_back(std::thread(simple_thread_wrapper, table_name, num_rows / thread_count, i));
  }
  for (auto& t : myThreads) {
    t.join();
  }
  vector<size_t> scan_col_hashs = scan_table_return_hash(table_name, gsession->get_catalog());
  vector<size_t> scan_col_hashs2 = scan_table_return_hash_non_iter(table_name, gsession->get_catalog());
  return scan_col_hashs == scan_col_hashs2;
}
}  // namespace

#define SMALL 100000
#define LARGE 1000000

TEST(StorageLarge, Numbers) {
  ASSERT_NO_THROW(run_ddl("drop table if exists numbers;"););
  ASSERT_NO_THROW(run_ddl("create table numbers (a smallint, b int, c bigint, d numeric(7,3), e "
                          "double, f float);"););
  EXPECT_TRUE(storage_test("numbers", LARGE));
  ASSERT_NO_THROW(run_ddl("drop table numbers;"););
}

TEST(StorageSmall, Strings) {
  ASSERT_NO_THROW(run_ddl("drop table if exists strings;"););
  ASSERT_NO_THROW(run_ddl("create table strings (x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test("strings", SMALL));
  ASSERT_NO_THROW(run_ddl("drop table strings;"););
}

TEST(StorageSmall, AllTypes) {
  ASSERT_NO_THROW(run_ddl("drop table if exists alltypes;"););
  ASSERT_NO_THROW(run_ddl("create table alltypes (a smallint, b int, c bigint, d numeric(7,3), e double, f float, "
                          "g timestamp(0), h time(0), i date, x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test("alltypes", SMALL));
  ASSERT_NO_THROW(run_ddl("drop table alltypes;"););
}

TEST(StorageRename, AllTypes) {
  ASSERT_NO_THROW(run_ddl("drop table if exists original_table;"););
  ASSERT_NO_THROW(
      run_ddl("create table original_table (a smallint, b int, c bigint, d numeric(7,3), e double, f float, "
              "g timestamp(0), h time(0), i date, x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test("original_table", SMALL));

  ASSERT_NO_THROW(run_ddl("drop table if exists new_table;"););
  ASSERT_NO_THROW(run_ddl("create table new_table (a smallint, b int, c bigint, d numeric(7,3), e double, f float, "
                          "g timestamp(0), h time(0), i date, x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test("new_table", SMALL));

  ASSERT_NO_THROW(run_ddl("alter table original_table rename to old_table;"););

  ASSERT_NO_THROW(run_ddl("alter table new_table rename to original_table;"););

  ASSERT_NO_THROW(run_ddl("drop table old_table;"););

  ASSERT_NO_THROW(run_ddl("create table new_table (a smallint, b int, c bigint, d numeric(7,3), e double, f float, "
                          "g timestamp(0), h time(0), i date, x varchar(10) encoding none, y text encoding none);"););

  ASSERT_NO_THROW(run_ddl("drop table original_table;"););
  ASSERT_NO_THROW(run_ddl("drop table new_table;"););
}

TEST(StorageSmallParallel, AllTypes) {
  ASSERT_NO_THROW(run_ddl("drop table if exists alltypes;"););
  ASSERT_NO_THROW(run_ddl("create table alltypes (a smallint, b int, c bigint, d numeric(7,3), e double, f float, "
                          "g timestamp(0), h time(0), i date, x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test_parallel("alltypes", SMALL, std::thread::hardware_concurrency()));
  ASSERT_NO_THROW(run_ddl("drop table alltypes;"););
}

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SQLTestEnv);
  return RUN_ALL_TESTS();
}
