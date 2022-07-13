/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include <cstring>
#include <iostream>
#include <string>

#include <cstdlib>
#include <exception>
#include <memory>

#include <thread>

#include <boost/functional/hash.hpp>
#include "../Analyzer/Analyzer.h"
#include "../Catalog/Catalog.h"
#include "../DataMgr/DataMgr.h"
#include "../Fragmenter/Fragmenter.h"
#include "../QueryRunner/QueryRunner.h"
#include "PopulateTableRandom.h"
#include "ScanTable.h"
#include "TestHelpers.h"
#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace Catalog_Namespace;
using namespace Analyzer;
using namespace Fragmenter_Namespace;

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using QR = QueryRunner::QueryRunner;
namespace {

inline void run_ddl_statement(const string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

bool storage_test(const string& table_name, size_t num_rows) {
  vector<size_t> insert_col_hashs =
      populate_table_random(table_name, num_rows, *QR::get()->getCatalog());
  vector<size_t> scan_col_hashs =
      scan_table_return_hash(table_name, *QR::get()->getCatalog());
  vector<size_t> scan_col_hashs2 =
      scan_table_return_hash_non_iter(table_name, *QR::get()->getCatalog());
  return insert_col_hashs == scan_col_hashs && insert_col_hashs == scan_col_hashs2;
}

void simple_thread_wrapper(const string& table_name, size_t num_rows, size_t thread_id) {
  populate_table_random(table_name, num_rows, *QR::get()->getCatalog());
}

bool storage_test_parallel(const string& table_name,
                           size_t num_rows,
                           size_t thread_count) {
  // Constructs a number of threads and have them push records to the table in parallel
  vector<std::thread> myThreads;
  for (size_t i = 0; i < thread_count; i++) {
    myThreads.emplace_back(simple_thread_wrapper, table_name, num_rows / thread_count, i);
  }
  for (auto& t : myThreads) {
    t.join();
  }
  vector<size_t> scan_col_hashs =
      scan_table_return_hash(table_name, *QR::get()->getCatalog());
  vector<size_t> scan_col_hashs2 =
      scan_table_return_hash_non_iter(table_name, *QR::get()->getCatalog());
  return scan_col_hashs == scan_col_hashs2;
}
}  // namespace

#define SMALL 100000
#define LARGE 1000000

TEST(StorageLarge, Numbers) {
  ASSERT_NO_THROW(run_ddl_statement("drop table if exists numbers;"););
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table numbers (a smallint, b int, c bigint, d numeric(17,3), e "
          "double, f float);"););
  EXPECT_TRUE(storage_test("numbers", LARGE));
  ASSERT_NO_THROW(run_ddl_statement("drop table numbers;"););
}

TEST(StorageSmall, Strings) {
  ASSERT_NO_THROW(run_ddl_statement("drop table if exists strings;"););
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table strings (x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test("strings", SMALL));
  ASSERT_NO_THROW(run_ddl_statement("drop table strings;"););
}

TEST(StorageSmall, AllTypes) {
  ASSERT_NO_THROW(run_ddl_statement("drop table if exists alltypes;"););
  ASSERT_NO_THROW(
      run_ddl_statement("create table alltypes (a smallint, b int, c bigint, d "
                        "numeric(17,3), e double, f float, "
                        "g timestamp(0), g_3 timestamp(3), g_6 timestamp(6), g_9 "
                        "timestamp(9), h time(0), i date, "
                        "x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test("alltypes", SMALL));
  ASSERT_NO_THROW(run_ddl_statement("drop table alltypes;"););
}

TEST(StorageRename, AllTypes) {
  ASSERT_NO_THROW(run_ddl_statement("drop table if exists original_table;"););
  ASSERT_NO_THROW(
      run_ddl_statement("create table original_table (a smallint, b int, c bigint, d "
                        "numeric(17,3), e double, f float, "
                        "g timestamp(0), g_3 timestamp(3), g_6 timestamp(6), g_9 "
                        "timestamp(9), h time(0), i date, "
                        "x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test("original_table", SMALL));

  ASSERT_NO_THROW(run_ddl_statement("drop table if exists new_table;"););
  ASSERT_NO_THROW(
      run_ddl_statement("create table new_table (a smallint, b int, c bigint, d "
                        "numeric(17,3), e double, f float, "
                        "g timestamp(0), g_3 timestamp(3), g_6 timestamp(6), g_9 "
                        "timestamp(9), h time(0), i date, "
                        "x varchar(10) encoding none, y text encoding none);"););
  EXPECT_TRUE(storage_test("new_table", SMALL));

  ASSERT_NO_THROW(run_ddl_statement("alter table original_table rename to old_table;"););

  ASSERT_NO_THROW(run_ddl_statement("alter table new_table rename to original_table;"););

  ASSERT_NO_THROW(run_ddl_statement("drop table old_table;"););

  ASSERT_NO_THROW(
      run_ddl_statement("create table new_table (a smallint, b int, c bigint, d "
                        "numeric(17,3), e double, f float, "
                        "g timestamp(0), g_3 timestamp(3), g_6 timestamp(6), g_9 "
                        "timestamp(9), h time(0), i date, "
                        "x varchar(10) encoding none, y text encoding none);"););

  ASSERT_NO_THROW(run_ddl_statement("drop table original_table;"););
  ASSERT_NO_THROW(run_ddl_statement("drop table new_table;"););
}

TEST(StorageSmallParallel, AllTypes) {
  ASSERT_NO_THROW(run_ddl_statement("drop table if exists alltypes;"););
  ASSERT_NO_THROW(
      run_ddl_statement(
          "create table alltypes (a smallint, b int, c bigint, d numeric(17,3), e "
          "double, f float, g timestamp(0), g_3 timestamp(3), g_6 timestamp(6), g_9 "
          "timestamp(9), h time(0), i date, x varchar(10) encoding none, y text encoding "
          "none);"););
  EXPECT_TRUE(
      storage_test_parallel("alltypes", SMALL, std::thread::hardware_concurrency()));
  ASSERT_NO_THROW(run_ddl_statement("drop table alltypes;"););
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

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
