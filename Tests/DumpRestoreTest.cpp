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

#include <algorithm>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/process.hpp>
#include <boost/process/search_path.hpp>
#include <boost/program_options.hpp>
#include <boost/variant.hpp>
#include <boost/variant/get.hpp>

#ifdef HAVE_AWS_S3
#include "AwsHelpers.h"
#include "DataMgr/HeavyDbAwsSdk.h"
#endif
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace TestHelpers;

extern size_t g_leaf_count;
extern bool g_test_rollback_dump_restore;
extern bool g_allow_s3_server_privileges;

const static std::string tar_ball_path = "/tmp/_Orz__";

namespace {
bool g_hoist_literals{true};

using QR = QueryRunner::QueryRunner;

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str) {
  return QR::get()->runSQL(query_str, ExecutorDeviceType::CPU, g_hoist_literals);
}

void clear() {
  EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS s;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS t;"));
  EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS x;"));
  CHECK_EQ(0, boost::process::system(("rm -rf " + tar_ball_path).c_str()));
}

static int nshard;
static int nrow;

void reset() {
  clear();
  // preformat shard key phrases
  std::string phrase_shard_key = nshard > 1 ? ", SHARD KEY (i)" : "";
  std::string phrase_shard_count =
      nshard > 1 ? ", SHARD_COUNT = " + std::to_string(nshard) : "";
  // create table s which has a encoded text column to be referenced by table t
  EXPECT_NO_THROW(run_ddl_statement("CREATE TABLE s(i int, j int, s text" +
                                    phrase_shard_key + ") WITH (FRAGMENT_SIZE=2" +
                                    phrase_shard_count + ");"));
  // create table t which has 3 encoded text columns:
  //	 column s: to be domestically referenced by column t.t
  //   column t: domestically references column t.s
  //   column f: foreignly references column s.s
  EXPECT_NO_THROW(run_ddl_statement(
      "CREATE TABLE t(i int, j int, s text, d text, f text" + phrase_shard_key +
      ", SHARED DICTIONARY (d) REFERENCES t(s)"    // domestic ref
      + ", SHARED DICTIONARY (f) REFERENCES s(s)"  // foreign ref
      + ") WITH (FRAGMENT_SIZE=2" + phrase_shard_count + ");"));
  // insert nrow rows to tables s and t
  TestHelpers::ValuesGenerator gen_s("s");
  TestHelpers::ValuesGenerator gen_t("t");
  // make dicts of s.s and t.s have different layouts
  for (int i = 1 * nrow; i < 2 * nrow; ++i) {
    const auto s = std::to_string(i);
    run_multiple_agg(gen_s(i, i, s));
  }
  for (int i = 0 * nrow; i < 1 * nrow; ++i) {
    const auto s = std::to_string(i);
    run_multiple_agg(gen_t(i, i, s, s, s));
  }
}
}  // namespace

#define NROWS 20
template <int NSHARDS, int NR = NROWS>
class DumpRestoreTest : public ::testing::Test {
  void SetUp() override {
    nshard = NSHARDS;
    nrow = NR;
  }

  void TearDown() override { clear(); }
};

void check_table(const std::string& table, const bool alter, const int delta) {
  // check columns table.j/s/d/f are still equal semantically, b/c if sxth went
  // wrong with dump or restore of data or dict files, the columns should mess up.
  auto rows = run_multiple_agg("SELECT j, s, d, f" + std::string(alter ? ",y" : "") +
                               " FROM " + table + ";");
  for (int r = 0; r < NROWS; ++r) {
    auto row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(alter ? 5 : 4), row.size());
    auto j = std::to_string(v<int64_t>(row[0]) - delta);
    auto nullable_s = v<NullableString>(row[1]);
    auto nullable_d = v<NullableString>(row[2]);
    auto nullable_f = v<NullableString>(row[3]);
    auto s = boost::get<std::string>(&nullable_s);
    auto d = boost::get<std::string>(&nullable_d);
    auto f = boost::get<std::string>(&nullable_f);
    CHECK(s);
    CHECK(d);
    CHECK(f);
    EXPECT_EQ(j, *s);
    EXPECT_EQ(j, *d);
    EXPECT_EQ(j, *f);
    if (alter) {
      auto y = v<int64_t>(row[4]);
      EXPECT_EQ(y, 77);
    }
  }
}

void dump_restore(const bool migrate,
                  const bool alter,
                  const bool rollback,
                  const std::vector<std::string>& with_options) {
  reset();
  // if set, alter pivot table t to make it have "been altered"
  if (alter) {
    EXPECT_NO_THROW(run_ddl_statement("ALTER TABLE t ADD COLUMN y int DEFAULT 77;"));
  }
  const std::string with_options_clause =
      with_options.size() ? (" WITH (" + boost::algorithm::join(with_options, ",") + ")")
                          : "";
  // dump pivot table t
  EXPECT_NO_THROW(run_ddl_statement("DUMP TABLE t TO '" + tar_ball_path + "'" +
                                    with_options_clause + ";"));
  // restore is to table t while migrate is to table x
  const std::string table = migrate ? "x" : "t";
  // increment column table.j by a delta if testing rollback restore
  int delta = 0;
  if (rollback) {
    delta = NROWS;
    EXPECT_NO_THROW(
        run_multiple_agg("UPDATE t SET j = j + " + std::to_string(delta) + ";"));
  }
  // rollback table restore/migrate?
  const auto run_restore = "RESTORE TABLE " + table + " FROM '" + tar_ball_path + "'" +
                           with_options_clause + ";";
  // TODO: v1.0 simply throws to avoid accidentally overwrite target table.
  // Will add a REPLACE TABLE to explicitly replace target table.
  // After that, remove the first following if-block to pass test!!!
  if (!migrate) {
    EXPECT_THROW(run_ddl_statement(run_restore), std::runtime_error);
  } else if (true == (g_test_rollback_dump_restore = rollback)) {
    EXPECT_THROW(run_ddl_statement(run_restore), std::runtime_error);
  } else {
    EXPECT_NO_THROW(run_ddl_statement(run_restore));
  }
  if (migrate && rollback) {
    EXPECT_THROW(run_ddl_statement("DROP TABLE x;"), std::runtime_error);
  } else {
    EXPECT_NO_THROW(check_table(table, alter, delta));
  }
}

void dump_restore(const bool migrate, const bool alter, const bool rollback) {
  // test two compression modes only so as not to hold cit back too much
  if (boost::process::search_path("lz4").string().empty()) {
    dump_restore(migrate, alter, rollback, {});  // gzip
    dump_restore(migrate, alter, rollback, {"compression='none'"});
  } else {
    dump_restore(migrate, alter, rollback, {});  // lz4
    dump_restore(migrate, alter, rollback, {"compression='gzip'"});
  }
}

using DumpRestoreTest_Unsharded = DumpRestoreTest<1>;
using DumpRestoreTest_Sharded = DumpRestoreTest<2>;

#define BODY_F(test_class, test_name) test_class##_##test_name##_body()
#define TEST_F1(test_class, test_name, sharded_or_not) \
  TEST_F(test_class##_##sharded_or_not, test_name) { BODY_F(test_class, test_name); }
#define TEST_UNSHARDED_AND_SHARDED(test_class, test_name) \
  TEST_F1(test_class, test_name, Unsharded)               \
  TEST_F1(test_class, test_name, Sharded)

void BODY_F(DumpRestoreTest, DumpRestore) {
  dump_restore(false, false, false);
}
void BODY_F(DumpRestoreTest, DumpRestore_Rollback) {
  dump_restore(false, false, true);
}
void BODY_F(DumpRestoreTest, DumpRestore_Altered) {
  dump_restore(false, true, false);
}
void BODY_F(DumpRestoreTest, DumpRestore_Altered_Rollback) {
  dump_restore(false, true, true);
}
void BODY_F(DumpRestoreTest, DumpMigrate) {
  dump_restore(true, false, false);
}
void BODY_F(DumpRestoreTest, DumpMigrate_Rollback) {
  dump_restore(true, false, true);
}
void BODY_F(DumpRestoreTest, DumpMigrate_Altered) {
  dump_restore(true, true, false);
}
void BODY_F(DumpRestoreTest, DumpMigrate_Altered_Rollback) {
  dump_restore(true, true, true);
}

// restore table tests
TEST_UNSHARDED_AND_SHARDED(DumpRestoreTest, DumpRestore)
TEST_UNSHARDED_AND_SHARDED(DumpRestoreTest, DumpRestore_Rollback)
TEST_UNSHARDED_AND_SHARDED(DumpRestoreTest, DumpRestore_Altered)
TEST_UNSHARDED_AND_SHARDED(DumpRestoreTest, DumpRestore_Altered_Rollback)
// migrate table tests
TEST_UNSHARDED_AND_SHARDED(DumpRestoreTest, DumpMigrate)
TEST_UNSHARDED_AND_SHARDED(DumpRestoreTest, DumpMigrate_Rollback)
TEST_UNSHARDED_AND_SHARDED(DumpRestoreTest, DumpMigrate_Altered)
TEST_UNSHARDED_AND_SHARDED(DumpRestoreTest, DumpMigrate_Altered_Rollback)

class DumpAndRestoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    boost::filesystem::remove_all(tar_ball_path);
    run_ddl_statement("DROP TABLE IF EXISTS test_table;");
    run_ddl_statement("DROP TABLE IF EXISTS test_table_2;");
    run_ddl_statement("DROP TABLE IF EXISTS render_groups;");
    g_test_rollback_dump_restore = false;
  }

  void TearDown() override {
    boost::filesystem::remove_all(tar_ball_path);
    run_ddl_statement("DROP TABLE IF EXISTS test_table;");
    run_ddl_statement("DROP TABLE IF EXISTS test_table_2;");
    run_ddl_statement("DROP TABLE IF EXISTS render_groups;");
  }

  void queryAndAssertException(const std::string& query,
                               const std::string& error_message) {
    try {
      run_ddl_statement(query);
      FAIL() << "An exception should have been thrown for this test case";
    } catch (const std::exception& e) {
      ASSERT_STREQ(error_message.c_str(), e.what());
    }
  }

  void sqlAndCompareResult(const std::string& sql,
                           const std::vector<std::string>& expected_result) {
    auto result = run_multiple_agg(sql);
    ASSERT_EQ(expected_result.size(), result->colCount());
    auto row = result->getNextRow(true, true);
    for (size_t i = 0; i < expected_result.size(); i++) {
      auto& target_value = boost::get<ScalarTargetValue>(row[i]);
      auto& nullable_str = boost::get<NullableString>(target_value);
      auto& str = boost::get<std::string>(nullable_str);
      EXPECT_EQ(expected_result[i], str);
    }
  }

  void sqlAndCompareArrayResult(
      const std::string& sql,
      const std::vector<std::vector<std::string>>& expected_result) {
    auto result = run_multiple_agg(sql);
    ASSERT_EQ(expected_result.size(), result->colCount());
    auto row = result->getNextRow(true, true);
    for (size_t i = 0; i < expected_result.size(); i++) {
      auto& expected_array = expected_result[i];
      auto& target_value_vector = boost::get<ArrayTargetValue>(row[i]).get();
      ASSERT_EQ(expected_array.size(), target_value_vector.size());
      size_t j = 0;
      for (const auto& target_value : target_value_vector) {
        auto& nullable_str = boost::get<NullableString>(target_value);
        auto& str = boost::get<std::string>(nullable_str);
        EXPECT_EQ(expected_array[j], str);
        j++;
      }
    }
  }

  void sqlAndCompareResult(const std::string& sql,
                           const std::vector<int64_t>& expected_values) {
    auto result = run_multiple_agg(sql);
    ASSERT_EQ(expected_values.size(), result->rowCount());
    for (auto expected_value : expected_values) {
      auto row = result->getNextRow(true, true);
      auto& target_value = boost::get<ScalarTargetValue>(row[0]);
      EXPECT_EQ(expected_value, boost::get<int64_t>(target_value));
    }
  }
};

TEST_F(DumpAndRestoreTest, ForeignTableDump) {
  run_ddl_statement(
      "CREATE FOREIGN TABLE test_foreign_table (a INT) SERVER default_local_delimited "
      "WITH (file_path='../../Tests/FsiDataFiles/1.csv');");
  queryAndAssertException("DUMP TABLE test_foreign_table TO '" + tar_ball_path + "';",
                          "Dumping a foreign table is not supported.");
}

TEST_F(DumpAndRestoreTest, Geo_DifferentEncodings) {
  run_ddl_statement(
      "CREATE TABLE test_table ("
      "p1 GEOMETRY(POINT, 4326) ENCODING COMPRESSED(32), p2 GEOMETRY(POINT, 4326) "
      "ENCODING NONE, "
      "l1 GEOMETRY(LINESTRING, 4326) ENCODING COMPRESSED(32), l2 GEOMETRY(LINESTRING, "
      "4326) ENCODING NONE, "
      "poly1 GEOMETRY(POLYGON, 4326) ENCODING COMPRESSED(32), poly2 GEOMETRY(POLYGON, "
      "4326) ENCODING NONE, "
      "mpoly1 GEOMETRY(MULTIPOLYGON, 4326) ENCODING COMPRESSED(32), "
      "mpoly2 GEOMETRY(MULTIPOLYGON, 4326) ENCODING NONE);");

  run_multiple_agg(
      "INSERT INTO test_table VALUES("
      "'POINT(1 0)', 'POINT(1 0)', 'LINESTRING(1 0,0 1)', 'LINESTRING(1 0,0 1)', "
      "'POLYGON((0 0,1 0,0 1,0 0))', 'POLYGON((0 0,1 0,0 1,0 0)))', "
      "'MULTIPOLYGON(((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))', "
      "'MULTIPOLYGON(((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))');");

  std::vector<std::string> expected_result{
      "POINT (0.999999940861017 0.0)",
      "POINT (1 0)",
      "LINESTRING (0.999999940861017 0.0,0.0 0.999999982770532)",
      "LINESTRING (1 0,0 1)",
      "POLYGON ((0 0,0.999999940861017 0.0,0.0 0.999999982770532,0 0))",
      "POLYGON ((0 0,1 0,0 1,0 0))",
      "MULTIPOLYGON (((0 0,0.999999940861017 0.0,0.0 0.999999982770532,0 0)),((0 "
      "0,1.99999996554106 0.0,0.0 1.99999996554106,0 0)))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"};

  sqlAndCompareResult("SELECT * FROM test_table;", expected_result);
  run_ddl_statement("DUMP TABLE test_table TO '" + tar_ball_path + "';");
  run_ddl_statement("RESTORE TABLE test_table_2 FROM '" + tar_ball_path + "';");
  sqlAndCompareResult("SELECT * FROM test_table_2;", expected_result);
}

TEST_F(DumpAndRestoreTest, Geo_DifferentSRIDs) {
  run_ddl_statement(
      "CREATE TABLE test_table ("
      "p1 POINT, p2 GEOMETRY(POINT, 4326), p3 GEOMETRY(POINT, 900913), "
      "l1 LINESTRING, l2 GEOMETRY(LINESTRING, 4326), l3 GEOMETRY(LINESTRING, 900913), "
      "poly1 POLYGON, poly2 GEOMETRY(POLYGON, 4326), poly3 GEOMETRY(POLYGON, 900913), "
      "mpoly1 MULTIPOLYGON, mpoly2 GEOMETRY(MULTIPOLYGON, 4326), mpoly3 "
      "GEOMETRY(MULTIPOLYGON, 900913));");

  run_multiple_agg(
      "INSERT INTO test_table VALUES("
      "'POINT(1 0)', 'POINT(1 0)', 'POINT(1 0)', "
      "'LINESTRING(1 0,0 1)', 'LINESTRING(1 0,0 1)', 'LINESTRING(1 0,0 1)', "
      "'POLYGON((0 0,1 0,0 1,0 0))', 'POLYGON((0 0,1 0,0 1,0 0))', 'POLYGON((0 0,1 0,0 "
      "1,0 0)))', "
      "'MULTIPOLYGON(((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))',"
      "'MULTIPOLYGON(((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))', "
      "'MULTIPOLYGON(((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))');");

  std::vector<std::string> expected_result{
      "POINT (1 0)",
      "POINT (0.999999940861017 0.0)",
      "POINT (1 0)",
      "LINESTRING (1 0,0 1)",
      "LINESTRING (0.999999940861017 0.0,0.0 0.999999982770532)",
      "LINESTRING (1 0,0 1)",
      "POLYGON ((0 0,1 0,0 1,0 0))",
      "POLYGON ((0 0,0.999999940861017 0.0,0.0 0.999999982770532,0 0))",
      "POLYGON ((0 0,1 0,0 1,0 0))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))",
      "MULTIPOLYGON (((0 0,0.999999940861017 0.0,0.0 0.999999982770532,0 0)),((0 "
      "0,1.99999996554106 0.0,0.0 1.99999996554106,0 0)))",
      "MULTIPOLYGON (((0 0,1 0,0 1,0 0)),((0 0,2 0,0 2,0 0)))"};

  sqlAndCompareResult("SELECT * FROM test_table;", expected_result);
  run_ddl_statement("DUMP TABLE test_table TO '" + tar_ball_path + "';");
  run_ddl_statement("RESTORE TABLE test_table_2 FROM '" + tar_ball_path + "';");
  sqlAndCompareResult("SELECT * FROM test_table_2;", expected_result);
}

TEST_F(DumpAndRestoreTest, TextArray) {
  run_ddl_statement("CREATE TABLE test_table (t1 TEXT[], t2 TEXT[5]);");
  run_multiple_agg(
      "INSERT INTO test_table VALUES({'a', 'b'}, {'a', 'b', 'c', 'd', 'e'});");

  std::vector<std::vector<std::string>> expected_result{{"a", "b"},
                                                        {"a", "b", "c", "d", "e"}};
  sqlAndCompareArrayResult("SELECT * FROM test_table;", expected_result);
  run_ddl_statement("DUMP TABLE test_table TO '" + tar_ball_path + "';");
  run_ddl_statement("RESTORE TABLE test_table_2 FROM '" + tar_ball_path + "';");
  sqlAndCompareArrayResult("SELECT * FROM test_table_2;", expected_result);
}

TEST_F(DumpAndRestoreTest, TableWithUncappedEpoch) {
  run_ddl_statement("CREATE TABLE test_table (i INTEGER);");
  QR::get()->getCatalog()->setUncappedTableEpoch("test_table");
  run_multiple_agg("INSERT INTO test_table VALUES(1);");
  run_multiple_agg("INSERT INTO test_table VALUES(2);");
  run_multiple_agg("INSERT INTO test_table VALUES(3);");

  sqlAndCompareResult("SELECT * FROM test_table;", {1, 2, 3});
  run_ddl_statement("DUMP TABLE test_table TO '" + tar_ball_path + "';");
  run_ddl_statement("RESTORE TABLE test_table_2 FROM '" + tar_ball_path + "';");
  sqlAndCompareResult("SELECT * FROM test_table_2;", {1, 2, 3});
  auto td = QR::get()->getCatalog()->getMetadataForTable("test_table_2", false);
  ASSERT_TRUE(td != nullptr);
  ASSERT_EQ(DEFAULT_MAX_ROLLBACK_EPOCHS, td->maxRollbackEpochs);
}

TEST_F(DumpAndRestoreTest, TableWithMaxRollbackEpochs) {
  run_ddl_statement(
      "CREATE TABLE test_table (i INTEGER) WITH (max_rollback_epochs = 10);");
  run_multiple_agg("INSERT INTO test_table VALUES(1);");
  run_multiple_agg("INSERT INTO test_table VALUES(2);");
  run_multiple_agg("INSERT INTO test_table VALUES(3);");

  sqlAndCompareResult("SELECT * FROM test_table;", {1, 2, 3});
  run_ddl_statement("DUMP TABLE test_table TO '" + tar_ball_path + "';");
  run_ddl_statement("RESTORE TABLE test_table_2 FROM '" + tar_ball_path + "';");
  sqlAndCompareResult("SELECT * FROM test_table_2;", {1, 2, 3});
  auto td = QR::get()->getCatalog()->getMetadataForTable("test_table_2", false);
  ASSERT_TRUE(td != nullptr);
  ASSERT_EQ(10, td->maxRollbackEpochs);
}

TEST_F(DumpAndRestoreTest, TableWithDefaultColumnValues) {
  run_ddl_statement(
      "CREATE TABLE test_table (idx INTEGER NOT NULL, i INTEGER DEFAULT 14,"
      "big_i BIGINT DEFAULT 314958734, null_i INTEGER, int_a INTEGER[] "
      "DEFAULT ARRAY[1, 2, 3], text_a TEXT[] DEFAULT ARRAY['a', 'b'] ENCODING DICT(32),"
      "dt TEXT DEFAULT 'World' ENCODING DICT(32), ls GEOMETRY(LINESTRING) "
      "DEFAULT 'LINESTRING (1 1,2 2,3 3)' ENCODING NONE, p GEOMETRY(POINT) DEFAULT "
      "'POINT (1 2)' ENCODING NONE,  d DATE DEFAULT '2011-10-23' ENCODING DAYS(32), "
      "ta TIMESTAMP[] DEFAULT ARRAY['2011-10-23 07:15:01', '2012-09-17 11:59:11'], "
      "f FLOAT DEFAULT 1.15, n DECIMAL(3,2) DEFAULT 1.25 ENCODING FIXED(16));");
  run_ddl_statement("DUMP TABLE test_table TO '" + tar_ball_path + "';");
  run_ddl_statement("RESTORE TABLE test_table_2 FROM '" + tar_ball_path + "';");
  auto td = QR::get()->getCatalog()->getMetadataForTable("test_table_2", false);
  std::string schema = QR::get()->getCatalog()->dumpCreateTable(td, false);
  std::string expected_schema =
      "CREATE TABLE test_table_2 (idx INTEGER NOT NULL, i INTEGER "
      "DEFAULT 14, big_i "
      "BIGINT DEFAULT 314958734, null_i INTEGER, int_a INTEGER[] DEFAULT "
      "ARRAY[1, 2, 3], text_a TEXT[] DEFAULT ARRAY['a', 'b'] ENCODING DICT(32), "
      "dt TEXT DEFAULT 'World' ENCODING DICT(32), ls GEOMETRY(LINESTRING) DEFAULT "
      "'LINESTRING (1 1,2 2,3 3)' ENCODING NONE, p GEOMETRY(POINT) DEFAULT 'POINT "
      "(1 2)' ENCODING NONE, d DATE DEFAULT '2011-10-23' ENCODING DAYS(32), ta "
      "TIMESTAMP(0)[] DEFAULT ARRAY['2011-10-23 07:15:01', '2012-09-17 11:59:11'], f "
      "FLOAT DEFAULT 1.15, n DECIMAL(3,2) DEFAULT 1.25 ENCODING FIXED(16));";
  ASSERT_EQ(schema, expected_schema);
}

void compression_test_create_table(std::string table, int nrows) {
  run_ddl_statement("CREATE TABLE " + table + " (i INTEGER, s TEXT);");
  for (int i = 0; i < nrows; i++) {
    run_multiple_agg("INSERT INTO " + table + " VALUES(" + std::to_string(i) + ", '_" +
                     std::to_string(i) + "_');");
  }
}

TargetValue compression_test_query(const std::string& query_str,
                                   const ExecutorDeviceType device_type) {
  auto rows = QR::get()->runSQL(query_str, device_type, true);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

void compression_test_clean(std::string file_path) {
  // files
  boost::filesystem::remove(file_path + ".df");
  boost::filesystem::remove(file_path + ".gz");
  boost::filesystem::remove(file_path + ".lz");
  boost::filesystem::remove(file_path + ".nn");

  // tables
  run_ddl_statement("DROP TABLE IF EXISTS test_src;");
  run_ddl_statement("DROP TABLE IF EXISTS test_dst_df;");
  run_ddl_statement("DROP TABLE IF EXISTS test_dst_gz;");
  run_ddl_statement("DROP TABLE IF EXISTS test_dst_lz;");
  run_ddl_statement("DROP TABLE IF EXISTS test_dst_nn;");
}

TEST_F(DumpAndRestoreTest, CompressionOptions) {
  std::string test_file_path = "/tmp/compression_test";
  compression_test_clean(test_file_path);

  // create a table
  int64_t nRows = 173;
  compression_test_create_table("test_src", nRows);

  // DUMP/RESTORE table: default compression
  run_ddl_statement("DUMP TABLE test_src TO '" + test_file_path + ".df';");
  run_ddl_statement("RESTORE TABLE test_dst_df FROM '" + test_file_path + ".df';");
  auto query = "SELECT count(*) FROM test_dst_df;";
  EXPECT_EQ(nRows, v<int64_t>(compression_test_query(query, ExecutorDeviceType::CPU)));

  // DUMP/RESTORE table: none compression
  run_ddl_statement("DUMP TABLE test_src TO '" + test_file_path + ".nn" +
                    "' WITH (compression='none');");
  run_ddl_statement("RESTORE TABLE test_dst_nn FROM '" + test_file_path +
                    ".nn' WITH (compression='none');");
  query = "SELECT count(*) FROM test_dst_nn;";
  EXPECT_EQ(nRows, v<int64_t>(compression_test_query(query, ExecutorDeviceType::CPU)));

  // DUMP/RESTORE table: GZIP compression
  bool hasGZIP = !boost::process::search_path("gzip").string().empty();
  if (hasGZIP) {
    run_ddl_statement("DUMP TABLE test_src TO '" + test_file_path + ".gz" +
                      "' WITH (compression='gzip');");
    run_ddl_statement("RESTORE TABLE test_dst_gz FROM '" + test_file_path +
                      ".gz' WITH (compression='gzip');");
    auto query = "SELECT count(*) FROM test_dst_gz;";
    EXPECT_EQ(nRows, v<int64_t>(compression_test_query(query, ExecutorDeviceType::CPU)));
  }

  // DUMP/RESTORE table: LZ4 compression
  bool hasLZ4 = !boost::process::search_path("lz4").string().empty();
  if (hasLZ4) {
    run_ddl_statement("DUMP TABLE test_src TO '" + test_file_path + ".lz" +
                      "' WITH (compression='lz4');");
    run_ddl_statement("RESTORE TABLE test_dst_lz FROM '" + test_file_path +
                      ".lz' WITH (compression='lz4');");
    query = "SELECT count(*) FROM test_dst_lz;";
    EXPECT_EQ(nRows, v<int64_t>(compression_test_query(query, ExecutorDeviceType::CPU)));
  }

  // DUMP/RESTORE table: bogus compression type should throw
  EXPECT_THROW(run_ddl_statement("DUMP TABLE test_src TO '" + test_file_path + ".bg" +
                                 "' WITH (compression='bogus');"),
               std::runtime_error);

  // check file sizes on disk
  if (hasGZIP && hasLZ4) {
    uint64_t sz_df = boost::filesystem::file_size(test_file_path + ".df");
    uint64_t sz_gz = boost::filesystem::file_size(test_file_path + ".gz");
    uint64_t sz_lz = boost::filesystem::file_size(test_file_path + ".lz");
    uint64_t sz_nn = boost::filesystem::file_size(test_file_path + ".nn");

    // size :  uncompressed(none) > lz4 > gzip  ... default is gzip
    ASSERT_GT(sz_nn, sz_lz);
    ASSERT_GT(sz_lz, sz_gz);

    // sz_gz - sz_df should be equal, but might differ by a few bytes
    //   use 100 as a 'close-enough' tolerance
    ASSERT_TRUE(abs((int)(sz_gz - sz_df)) < 100);
  }

  // cleanup
  compression_test_clean(test_file_path);
}

TEST_F(DumpAndRestoreTest, PreRebrandTableDump) {
  auto file_path =
      boost::filesystem::canonical("../../Tests/Export/TableDump/pre_rebrand.gz")
          .string();
  run_ddl_statement("RESTORE TABLE test_table FROM '" + file_path + "';");
  sqlAndCompareResult("SELECT * FROM test_table;", {1, 2, 3});
}

// Make sure that files are adjusted for altered tables even if using pre-rebrand files.
TEST_F(DumpAndRestoreTest, PreRebrandAlteredTable) {
  auto file_path =
      boost::filesystem::canonical("../../Tests/Export/TableDump/altered_table.gz")
          .string();
  run_ddl_statement("RESTORE TABLE test_table FROM '" + file_path +
                    "' WITH (compression='gzip');");
  sqlAndCompareResult("SELECT * FROM test_table;", std::vector<int64_t>{1});
}

// Check for error when restoring a table that contains pages from a deleted column.
TEST_F(DumpAndRestoreTest, DumpAlteredTable) {
  run_ddl_statement("CREATE TABLE test_table (c1 int, c2 int);");
  run_multiple_agg("INSERT INTO test_table VALUES(1, 2);");
  run_ddl_statement("ALTER TABLE test_table DROP COLUMN c2;");
  run_ddl_statement("DUMP TABLE test_table TO '" + tar_ball_path + "';");
  run_ddl_statement("DROP TABLE test_table;");
  run_ddl_statement("RESTORE TABLE test_table FROM '" + tar_ball_path + "';");
  sqlAndCompareResult("SELECT * FROM test_table;", std::vector<int64_t>{1});
}

TEST_F(DumpAndRestoreTest, DropRenderGroupColumns) {
  static constexpr int kNullInt = std::numeric_limits<int>::min();
  auto file_path = boost::filesystem::canonical(
                       "../../Tests/Export/TableDump/dump_with_render_groups_good.gz")
                       .string();
  EXPECT_NO_THROW(run_ddl_statement("RESTORE TABLE render_groups FROM '" + file_path +
                                    "' WITH (compression='gzip');"));
  sqlAndCompareResult("SELECT id FROM render_groups;",
                      std::vector<int64_t>{1, 2, 3, 4, 5});
  sqlAndCompareResult("SELECT ST_NPOINTS(poly) FROM render_groups;",
                      std::vector<int64_t>{4, kNullInt, 3, 3, 0});
  sqlAndCompareResult("SELECT ST_NPOINTS(multipoly) FROM render_groups;",
                      std::vector<int64_t>{6, kNullInt, 9, 9, 0});
}

TEST_F(DumpAndRestoreTest, DropRenderGroupColumnsBad) {
  auto file_path = boost::filesystem::canonical(
                       "../../Tests/Export/TableDump/dump_with_render_groups_bad.gz")
                       .string();
  EXPECT_THROW(run_ddl_statement("RESTORE TABLE render_groups FROM '" + file_path +
                                 "' WITH (compression='gzip');"),
               std::runtime_error);
}

#ifdef HAVE_AWS_S3
class S3RestoreTest : public DumpAndRestoreTest {
 protected:
  static void SetUpTestSuite() { heavydb_aws_sdk::init_sdk(); }

  static void TearDownTestSuite() { heavydb_aws_sdk::shutdown_sdk(); }

  void SetUp() override {
    g_allow_s3_server_privileges = false;
    run_ddl_statement("DROP TABLE IF EXISTS test_table;");
  }

  void TearDown() override {
    g_allow_s3_server_privileges = false;
    run_ddl_statement("DROP TABLE IF EXISTS test_table;");
  }

  bool insufficientPrivateCredentials() const {
    return !is_valid_aws_key(get_aws_keys_from_env());
  }

  void queryAndAssertPartialException(const std::string& query,
                                      const std::string& error_message) {
    try {
      run_ddl_statement(query);
      FAIL() << "An exception should have been thrown for this test case";
    } catch (const std::exception& e) {
      std::string exception_message{e.what()};
      ASSERT_TRUE(exception_message.find(error_message) != std::string::npos)
          << "Exception message: " << exception_message
          << ", expected partial error message: " << error_message;
    }
  }

  std::string getRestoreQuery(const std::string& object_key,
                              const std::string& region = aws_region_,
                              const std::string& access_key = {},
                              const std::string& secret_key = {},
                              const std::string& session_token = {},
                              const std::string& endpoint = {}) {
    std::string query{"RESTORE TABLE test_table FROM 's3://" + object_key +
                      "' "
                      "WITH (compression = 'gzip'"};
    if (!region.empty()) {
      query += ", s3_region = '" + region + "'";
    }
    if (!access_key.empty()) {
      query += ", s3_access_key = '" + access_key + "'";
    }
    if (!secret_key.empty()) {
      query += ", s3_secret_key = '" + secret_key + "'";
    }
    if (!session_token.empty()) {
      query += ", s3_session_token = '" + session_token + "'";
    }
    if (!endpoint.empty()) {
      query += ", s3_endpoint = '" + endpoint + "'";
    }
    query += ");";
    return query;
  }

  static inline const std::string aws_region_{"us-west-1"};
  static inline const std::string dump_file_name_{"test_table_dump.gz"};
  static inline const std::string public_bucket_{"heavydb-dump-restore-test-public"};
  static inline const std::string public_object_key_{
      "heavydb-dump-restore-test-public/test_table_dump.gz"};
  static inline const std::string private_object_key_{
      "heavydb-dump-restore-test/test_table_dump.gz"};
  static inline const std::string gcs_object_key_{
      "omnisci-importtest-data/dump-restore-test-data/test_table_dump.gz"};
};

TEST_F(S3RestoreTest, PublicBucket) {
  run_ddl_statement(getRestoreQuery(public_object_key_));
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY i;", {1, 2, 3});
}

TEST_F(S3RestoreTest, PrivateBucket) {
  if (insufficientPrivateCredentials()) {
    GTEST_SKIP() << "Insufficient private credentials to run test";
  }
  const auto [access_key, secret_key] = get_aws_keys_from_env();
  run_ddl_statement(
      getRestoreQuery(private_object_key_, aws_region_, access_key, secret_key));
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY i;", {1, 2, 3});
}

TEST_F(S3RestoreTest, PrivateBucketUsingSessionToken) {
  if (insufficientPrivateCredentials()) {
    GTEST_SKIP() << "Insufficient private credentials to run test";
  }
  const auto aws_keys = get_aws_keys_from_env();
  Aws::Client::ClientConfiguration client_config;
  client_config.region = aws_region_;
  const auto sts_credentials = generate_sts_credentials(aws_keys, client_config);

  run_ddl_statement(getRestoreQuery(private_object_key_,
                                    aws_region_,
                                    sts_credentials.GetAccessKeyId(),
                                    sts_credentials.GetSecretAccessKey(),
                                    sts_credentials.GetSessionToken()));
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY i;", {1, 2, 3});
}

TEST_F(S3RestoreTest, PrivateBucketWithServerPrivileges) {
  if (insufficientPrivateCredentials()) {
    GTEST_SKIP() << "Insufficient private credentials to run test";
  }
  g_allow_s3_server_privileges = true;
  run_ddl_statement(getRestoreQuery(private_object_key_));
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY i;", {1, 2, 3});
}

TEST_F(S3RestoreTest, CustomS3Endpoint) {
  run_ddl_statement(getRestoreQuery(
      gcs_object_key_, aws_region_, {}, {}, {}, "storage.googleapis.com"));
  sqlAndCompareResult("SELECT * FROM test_table ORDER BY i;", {1, 2, 3});
}

TEST_F(S3RestoreTest, PrivateBucketNoAccess) {
  queryAndAssertPartialException(
      getRestoreQuery(
          private_object_key_, aws_region_, "invalid_access_key", "invalid_secret_key"),
      "failed to get object '" + dump_file_name_ + "' of s3 url 's3://" +
          private_object_key_ + "': InvalidAccessKeyId");
}

TEST_F(S3RestoreTest, WrongOptionType) {
  queryAndAssertException("RESTORE TABLE test_table FROM 's3://" + public_object_key_ +
                              "' "
                              "WITH (compression = 'gzip', s3_region = 10);",
                          "\"s3_region\" option must be a string.");
}

TEST_F(S3RestoreTest, MissingRegion) {
  queryAndAssertException(getRestoreQuery(public_object_key_, {}),
                          "Required parameter \"s3_region\" not set. Please specify the "
                          "\"s3_region\" configuration parameter.");
}

TEST_F(S3RestoreTest, S3BucketWithMultipleFiles) {
  queryAndAssertException(
      getRestoreQuery(public_bucket_),
      "S3 URI references multiple files. Only one file can be restored at a time.");
}
#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;
  po::options_description desc("Options");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

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
