/*
 * Copyright 2019 OmniSci, Inc.
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
#include <boost/program_options.hpp>
#include <boost/variant.hpp>
#include <boost/variant/get.hpp>

#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace TestHelpers;

extern size_t g_leaf_count;
extern bool g_test_rollback_dump_restore;

const static std::string tar_ball_path = "/tmp/_Orz_.tgz";

namespace {
bool g_hoist_literals{true};

using QR = QueryRunner::QueryRunner;

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QR::get()->runDDLStatement(create_table_stmt);
}

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str) {
  return QR::get()->runSQL(query_str, ExecutorDeviceType::CPU, g_hoist_literals);
}

}  // namespace

#define NROWS 100
template <int NSHARDS, int NR = NROWS>
class DumpRestoreTest : public ::testing::Test {
  void SetUp() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS s;"));
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS t;"));
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS x;"));
    // preformat shard key phrases
    std::string phrase_shard_key = NSHARDS > 1 ? ", SHARD KEY (i)" : "";
    std::string phrase_shard_count =
        NSHARDS > 1 ? ", SHARD_COUNT = " + std::to_string(NSHARDS) : "";
    // create table s which has a encoded text column to be referenced by table t
    EXPECT_NO_THROW(run_ddl_statement("CREATE TABLE s(i int, j int, s text" +
                                      phrase_shard_key + ") WITH (FRAGMENT_SIZE=10" +
                                      phrase_shard_count + ");"));
    // create table t which has 3 encoded text columns:
    //	 column s: to be domestically referenced by column t.t
    //   column t: domestically references column t.s
    //   column f: foreignly references column s.s
    EXPECT_NO_THROW(run_ddl_statement(
        "CREATE TABLE t(i int, j int, s text, d text, f text" + phrase_shard_key +
        ", SHARED DICTIONARY (d) REFERENCES t(s)"    // domestic ref
        + ", SHARED DICTIONARY (f) REFERENCES s(s)"  // foreign ref
        + ") WITH (FRAGMENT_SIZE=10" + phrase_shard_count + ");"));
    // insert NR rows to tables s and t
    TestHelpers::ValuesGenerator gen_s("s");
    TestHelpers::ValuesGenerator gen_t("t");
    // make dicts of s.s and t.s have different layouts
    for (int i = 1 * NR; i < 2 * NR; ++i) {
      const auto s = std::to_string(i);
      run_multiple_agg(gen_s(i, i, s));
    }
    for (int i = 0 * NR; i < 1 * NR; ++i) {
      const auto s = std::to_string(i);
      run_multiple_agg(gen_t(i, i, s, s, s));
    }
  }

  void TearDown() override {
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS s;"));
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS t;"));
    EXPECT_NO_THROW(run_ddl_statement("DROP TABLE IF EXISTS x;"));
    system(("rm -rf " + tar_ball_path).c_str());
  }
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

void dump_restore(const bool migrate, const bool alter, const bool rollback) {
  // if set, alter pivot table t to make it have "been altered"
  if (alter) {
    EXPECT_NO_THROW(run_ddl_statement("ALTER TABLE t ADD COLUMN y int DEFAULT 77;"));
  }
  // dump pivot table t
  EXPECT_NO_THROW(run_ddl_statement("DUMP TABLE t TO '" + tar_ball_path + "';"));
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
  const auto run_restore = "RESTORE TABLE " + table + " FROM '" + tar_ball_path + "';";
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
