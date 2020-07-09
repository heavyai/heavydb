/*
 * Copyright 2020 OmniSci, Inc.
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
#include "Embedded/DBEngine.h"
#include <boost/program_options.hpp>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#ifndef CALCITE_PORT
#define CALCITE_PORT 5555
#endif

using namespace std;
using namespace EmbeddedDatabase;

DBEngine* engine = nullptr; 

int select_int(const string& query_str) {
  auto cursor = engine->executeDML(query_str);
  auto row = cursor->getNextRow();
std::cout << row.getInt(0) << std::endl;
  return row.getInt(0);
}

void run_ddl(const string& query_str) {
  engine->executeDDL(query_str);
}

std::shared_ptr<arrow::RecordBatch> run_dml(const string& query_str) {
  auto cursor = engine->executeDML(query_str);
  return cursor ? cursor->getArrowRecordBatch(): nullptr;
}

TEST(Insert, Dict) {
  EXPECT_NO_THROW(run_ddl("DROP TABLE IF EXISTS dist_test;"));
  EXPECT_NO_THROW(run_ddl(
      "CREATE TABLE dist_test (col1 TEXT ENCODING DICT) WITH (partitions='replicated');"));
  EXPECT_NO_THROW(run_dml("INSERT INTO dist_test VALUES('t1');"));
  ASSERT_EQ(1, select_int("SELECT count(*) FROM dist_test;"));

  EXPECT_NO_THROW(run_dml("INSERT INTO dist_test VALUES('t2');"));
  ASSERT_EQ(2, select_int("SELECT count(*) FROM dist_test;"));

  EXPECT_NO_THROW(run_dml("INSERT INTO dist_test VALUES('t3');"));
  ASSERT_EQ(3, select_int("SELECT count(*) FROM dist_test;"));

  run_ddl("DROP TABLE IF EXISTS dist_test;");
}

TEST(Insert, Decimal) {
  EXPECT_NO_THROW(run_ddl("DROP TABLE IF EXISTS dec_test;"));
  EXPECT_NO_THROW(run_ddl("CREATE TABLE dec_test (big_dec DECIMAL(17, 2), "
      "med_dec DECIMAL(9, 2), small_dec DECIMAL(4, 2)) WITH (fragment_size=2);"));

  EXPECT_NO_THROW(run_dml("INSERT INTO dec_test VALUES("
      "999999999999999.99, 9999999.99, 99.99);"));
  ASSERT_EQ(1, select_int("SELECT count(*) FROM dec_test;"));

  EXPECT_NO_THROW(run_dml("INSERT INTO dec_test VALUES("
      "-999999999999999.99, -9999999.99, -99.99);"));
  ASSERT_EQ(2, select_int("SELECT count(*) FROM dec_test;"));

  EXPECT_NO_THROW(run_dml("INSERT INTO dec_test VALUES("
      "12.2382, 12.2382 , 12.2382);"));
  ASSERT_EQ(3, select_int("SELECT count(*) FROM dec_test;"));

  EXPECT_NO_THROW(run_ddl("DROP TABLE IF EXISTS dec_test;"));
}

TEST(Insert, ShardedTableWithGeo) {
  run_ddl("DROP TABLE IF EXISTS table_with_geo_and_shard_key;");
  EXPECT_NO_THROW(
    run_ddl("CREATE TABLE table_with_geo_and_shard_key (x Int, poly "
            "POLYGON, b SMALLINT, SHARD KEY(b)) WITH (shard_count = 4);"));

  EXPECT_NO_THROW(
    run_dml("INSERT INTO table_with_geo_and_shard_key VALUES (1, "
            "'POLYGON((0 0, 1 1, 2 2, 3 3))', 0);"));
  EXPECT_NO_THROW(
    run_dml("INSERT INTO table_with_geo_and_shard_key (x, poly, b) VALUES (1, "
            "'POLYGON((0 0, 1 1, 2 2, 3 3))', 1);"));
  EXPECT_NO_THROW(
    run_dml("INSERT INTO table_with_geo_and_shard_key (b, poly, x) VALUES (2, "
            "'POLYGON((0 0, 1 1, 2 2, 3 3))', 1);"));
  EXPECT_NO_THROW(
    run_dml("INSERT INTO table_with_geo_and_shard_key (x, b, poly) VALUES (1, 3, "
            "'POLYGON((0 0, 1 1, 2 2, 3 3))');"));
  EXPECT_NO_THROW(
    run_dml("INSERT INTO table_with_geo_and_shard_key (poly, x, b) VALUES ("
            "'POLYGON((0 0, 1 1, 2 2, 3 3))', 1, 4);"));

  ASSERT_EQ(5, select_int("SELECT count(*) FROM table_with_geo_and_shard_key;"));
  run_ddl("DROP TABLE IF EXISTS table_with_geo_and_shard_key;");
}

TEST(Update, Text) {
  run_ddl("DROP TABLE IF EXISTS text_default;");
  run_ddl("CREATE TABLE text_default (t text);");

  run_dml("insert into text_default values ('do');");
  run_dml("insert into text_default values ('you');");
  run_dml("insert into text_default values ('know');");
  run_dml("insert into text_default values ('the');");
  run_dml("insert into text_default values ('muffin');");
  run_dml("insert into text_default values ('man');");

  EXPECT_NO_THROW(
    run_dml("update text_default set t='pizza' where char_length(t) <= 3;"));
  ASSERT_EQ(4,
    select_int("select count(t) from text_default where t='pizza';"));
  run_ddl("DROP TABLE IF EXISTS text_default;");
}

TEST(Select, FilterAndSimpleAggregation) {
  const ssize_t num_rows{10};
  run_ddl("DROP TABLE IF EXISTS test;");

  EXPECT_NO_THROW(
    run_ddl("CREATE TABLE test(x int not null, w tinyint, y int, z smallint, t bigint, "
            "b boolean, f float, ff float, fn float, d double, dn double, "
            "str varchar(10), null_str text, fixed_str text, fixed_null_str text, "
            "real_str text, shared_dict text, m timestamp(0), m_3 timestamp(3), "
            "m_6 timestamp(6), m_9 timestamp(9), n time(0), o date, o1 date, o2 date, "
            "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not null, ss text, "
            "u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null, "
            "smallint_nulls smallint, bn boolean not null);"));

  for (ssize_t i = 0; i < num_rows; ++i) {
    EXPECT_NO_THROW(run_dml(
        "INSERT INTO test VALUES(7, -8, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, "
        "'foo', null, 'foo', null, "
        "'real_foo', 'foo',"
        "'2014-12-13 22:23:15', '2014-12-13 22:23:15.323', '1999-07-11 "
        "14:02:53.874533', "
        "'2006-04-26 "
        "03:49:04.607435125', "
        "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, "
        "'fish', "
        "null, "
        "2147483647, -2147483648, null, -1, 32767, 't');"));
    EXPECT_NO_THROW(run_dml(
        "INSERT INTO test VALUES(8, -7, 43, -78, 1002, 'f', 1.2, 101.2, -101.2, 2.4, "
        "-2002.4, 'bar', null, 'bar', null, "
        "'real_bar', NULL, '2014-12-13 22:23:15', '2014-12-13 22:23:15.323', "
        "'2014-12-13 "
        "22:23:15.874533', "
        "'2014-12-13 22:23:15.607435763', '15:13:14', NULL, NULL, NULL, NULL, 222.2, "
        "222.2, "
        "null, null, null, "
        "-2147483647, "
        "9223372036854775807, -9223372036854775808, null, 'f');"));
    EXPECT_NO_THROW(run_dml(
        "INSERT INTO test VALUES(7, -7, 43, 102, 1002, null, 1.3, 1000.3, -1000.3, 2.6, "
        "-220.6, 'baz', null, null, null, "
        "'real_baz', 'baz', '2014-12-14 22:23:15', '2014-12-14 22:23:15.750', "
        "'2014-12-14 22:23:15.437321', "
        "'2014-12-14 22:23:15.934567401', '15:13:14', '1999-09-09', '1999-09-09', "
        "'1999-09-09', 11, "
        "333.3, 333.3, "
        "'boat', null, 1, "
        "-1, 1, -9223372036854775808, 1, 't');"));
  }

  ASSERT_EQ(30, 
    select_int("SELECT COUNT(*) FROM test;"));
  ASSERT_EQ(30, 
    select_int("SELECT COUNT(f) FROM test;"));
  ASSERT_EQ(7, 
    select_int("SELECT MIN(x) FROM test;"));
  ASSERT_EQ(8, 
    select_int("SELECT MAX(x) FROM test;"));
  ASSERT_EQ(-78, 
    select_int("SELECT MIN(z) FROM test;"));
  ASSERT_EQ(102, 
    select_int("SELECT MAX(z) FROM test;"));
  ASSERT_EQ(1001, 
    select_int("SELECT MIN(t) FROM test;"));
  ASSERT_EQ(1002, 
    select_int("SELECT MAX(t) FROM test;"));

  run_ddl("DROP TABLE IF EXISTS test;");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()("test-help",
                     "Print all EmbeddedDbTest specific options (for gtest "
                     "options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: EmbeddedDbTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  logger::init(log_options);

  engine = DBEngine::create(BASE_PATH, CALCITE_PORT);

  int err{0};

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  engine->reset();
  return err;
}
