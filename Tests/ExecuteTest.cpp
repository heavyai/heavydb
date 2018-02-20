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

#include "QueryRunner.h"

#include "../Import/Importer.h"
#include "../Parser/parser.h"
#include "../QueryEngine/ArrowResultSet.h"
#include "../SqliteConnector/SqliteConnector.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <sstream>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std;

namespace {

std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;
bool g_hoist_literals{true};
size_t g_shard_count{0};

struct ShardInfo {
  const std::string shard_col;
  const size_t shard_count;
};

struct SharedDictionaryInfo {
  const std::string col;
  const std::string ref_table;
  const std::string ref_col;
};

std::string build_create_table_statement(const std::string& columns_definition,
                                         const std::string& table_name,
                                         const ShardInfo& shard_info,
                                         const std::vector<SharedDictionaryInfo>& shared_dict_info,
                                         const size_t fragment_size) {
  const std::string shard_key_def{shard_info.shard_col.empty() ? "" : ", SHARD KEY (" + shard_info.shard_col + ")"};

  std::vector<std::string> shared_dict_def;
  if (shared_dict_info.size() > 0) {
    for (size_t idx = 0; idx < shared_dict_info.size(); ++idx) {
      shared_dict_def.push_back(", SHARED DICTIONARY (" + shared_dict_info[idx].col + ") REFERENCES " +
                                shared_dict_info[idx].ref_table + "(" + shared_dict_info[idx].ref_col + ")");
    }
  }
  const std::string fragment_size_def{shard_info.shard_col.empty() ? "fragment_size=" + std::to_string(fragment_size)
                                                                   : ""};

  const std::string shard_count_def{
      shard_info.shard_col.empty() ? "" : "shard_count=" + std::to_string(shard_info.shard_count)};

  return "CREATE TABLE " + table_name + "(" + columns_definition + shard_key_def +
         boost::algorithm::join(shared_dict_def, "") + ") WITH (" + fragment_size_def + shard_count_def + ");";
}

std::shared_ptr<ResultSet> run_multiple_agg(const string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool allow_loop_joins) {
  return run_multiple_agg(query_str, g_session, device_type, g_hoist_literals, allow_loop_joins);
}

std::shared_ptr<ResultSet> run_multiple_agg(const string& query_str, const ExecutorDeviceType device_type) {
  return run_multiple_agg(query_str, device_type, true);
}

TargetValue run_simple_agg(const string& query_str, const ExecutorDeviceType device_type) {
  auto rows = run_multiple_agg(query_str, device_type);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  return crt_row[0];
}

template <class T>
T v(const TargetValue& r) {
  auto scalar_r = boost::get<ScalarTargetValue>(&r);
  CHECK(scalar_r);
  auto p = boost::get<T>(scalar_r);
  CHECK(p);
  return *p;
}

void run_ddl_statement(const string& create_table_stmt) {
  SQLParser parser;
  list<std::unique_ptr<Parser::Stmt>> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(create_table_stmt, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front().get();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl);
  if (ddl != nullptr)
    ddl->execute(*g_session);
}

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !g_session->get_catalog().get_dataMgr().gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

bool approx_eq(const double v, const double target, const double eps = 0.01) {
  const auto v_u64 = *reinterpret_cast<const uint64_t*>(may_alias_ptr(&v));
  const auto target_u64 = *reinterpret_cast<const uint64_t*>(may_alias_ptr(&target));
  return v_u64 == target_u64 || (target - eps < v && v < target + eps);
}

class SQLiteComparator {
 public:
  SQLiteComparator() : connector_("sqliteTestDB", "") {}

  void query(const std::string& query_string) { connector_.query(query_string); }

  void compare(const std::string& query_string, const ExecutorDeviceType device_type) {
    const auto mapd_results = run_multiple_agg(query_string, device_type);
    compare_impl(mapd_results.get(), query_string, device_type, false);
  }

  void compare_arrow_output(const std::string& query_string,
                            const std::string& sqlite_query_string,
                            const ExecutorDeviceType device_type) {
    const auto results = run_select_query(query_string, g_session, device_type, g_hoist_literals, true);
    const auto arrow_mapd_results = result_set_arrow_loopback(results);
    compare_impl(arrow_mapd_results.get(), sqlite_query_string, device_type, false);
  }

  void compare(const std::string& query_string,
               const std::string& sqlite_query_string,
               const ExecutorDeviceType device_type) {
    const auto mapd_results = run_multiple_agg(query_string, device_type);
    compare_impl(mapd_results.get(), sqlite_query_string, device_type, false);
  }

  // added to deal with time shift for now testing
  void compare_timstamp_approx(const std::string& query_string, const ExecutorDeviceType device_type) {
    const auto mapd_results = run_multiple_agg(query_string, device_type);
    compare_impl(mapd_results.get(), query_string, device_type, true);
  }

 private:
  template <class MapDResults>
  void compare_impl(const MapDResults* mapd_results,
                    const std::string& sqlite_query_string,
                    const ExecutorDeviceType device_type,
                    bool timestamp_approx) {
    connector_.query(sqlite_query_string);
    ASSERT_EQ(connector_.getNumRows(), mapd_results->rowCount());
    const int num_rows{static_cast<int>(connector_.getNumRows())};
    if (mapd_results->definitelyHasNoRows()) {
      ASSERT_EQ(0, num_rows);
      return;
    }
    if (!num_rows) {
      return;
    }
    CHECK_EQ(connector_.getNumCols(), mapd_results->colCount());
    const int num_cols{static_cast<int>(connector_.getNumCols())};
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const auto crt_row = mapd_results->getNextRow(true, true);
      CHECK(!crt_row.empty());
      CHECK_EQ(static_cast<size_t>(num_cols), crt_row.size());
      for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        const auto ref_col_type = connector_.columnTypes[col_idx];
        const auto mapd_variant = crt_row[col_idx];
        const auto scalar_mapd_variant = boost::get<ScalarTargetValue>(&mapd_variant);
        CHECK(scalar_mapd_variant);
        const auto mapd_ti = mapd_results->getColType(col_idx);
        const auto mapd_type = mapd_ti.get_type();
        checkTypeConsistency(ref_col_type, mapd_ti);
        const bool ref_is_null = connector_.isNull(row_idx, col_idx);
        switch (mapd_type) {
          case kSMALLINT:
          case kINT:
          case kBIGINT: {
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              const auto ref_val = connector_.getData<int64_t>(row_idx, col_idx);
              ASSERT_EQ(ref_val, mapd_val);
            }
            break;
          }
          case kTEXT:
          case kCHAR:
          case kVARCHAR: {
            const auto mapd_as_str_p = boost::get<NullableString>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_str_p);
            const auto mapd_str_notnull = boost::get<std::string>(mapd_as_str_p);
            if (ref_is_null) {
              // CHECK(!mapd_str_notnull); // JUST TO DEBUG SOMETHING TO BE UNCOMENTED
            } else {
              CHECK(mapd_str_notnull);
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              const auto mapd_val = *mapd_str_notnull;
              ASSERT_EQ(ref_val, mapd_val);
            }
            break;
          }
          case kNUMERIC:
          case kDECIMAL:
          case kDOUBLE: {
            const auto mapd_as_double_p = boost::get<double>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_double_p);
            const auto mapd_val = *mapd_as_double_p;
            if (ref_is_null) {
              ASSERT_EQ(inline_fp_null_val(SQLTypeInfo(kDOUBLE, false)), mapd_val);
            } else {
              const auto ref_val = connector_.getData<double>(row_idx, col_idx);
              ASSERT_TRUE(approx_eq(ref_val, mapd_val));
            }
            break;
          }
          case kFLOAT: {
            const auto mapd_as_float_p = boost::get<float>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_float_p);
            const auto mapd_val = *mapd_as_float_p;
            if (ref_is_null) {
              if (inline_fp_null_val(SQLTypeInfo(kFLOAT, false)) != mapd_val) {
                CHECK(false);
              }
            } else {
              const auto ref_val = connector_.getData<float>(row_idx, col_idx);
              if (!approx_eq(ref_val, mapd_val)) {
                CHECK(false);
              }
            }
            break;
          }
          case kTIMESTAMP:
          case kDATE: {
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            CHECK(mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              struct tm tm_struct {
                0
              };
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              const auto end_str =
                  strptime(ref_val.c_str(), mapd_type == kTIMESTAMP ? "%Y-%m-%d %H:%M:%S" : "%Y-%m-%d", &tm_struct);
              if (end_str != nullptr) {
                ASSERT_EQ(0, *end_str);
                ASSERT_EQ(ref_val.size(), static_cast<size_t>(end_str - ref_val.c_str()));
              }
              if (timestamp_approx) {
                // approximate result give 10 second lee way
                ASSERT_NEAR(*mapd_as_int_p, timegm(&tm_struct), 10);
              } else {
                ASSERT_EQ(*mapd_as_int_p, timegm(&tm_struct));
              }
            }
            break;
          }
          case kBOOLEAN: {
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            CHECK(mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              if (ref_val == "t") {
                ASSERT_EQ(1, *mapd_as_int_p);
              } else {
                CHECK_EQ("f", ref_val);
                ASSERT_EQ(0, *mapd_as_int_p);
              }
            }
            break;
          }
          case kTIME: {
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            CHECK(mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              std::vector<std::string> time_tokens;
              boost::split(time_tokens, ref_val, boost::is_any_of(":"));
              ASSERT_EQ(size_t(3), time_tokens.size());
              ASSERT_EQ(boost::lexical_cast<int64_t>(time_tokens[0]) * 3600 +
                            boost::lexical_cast<int64_t>(time_tokens[1]) * 60 +
                            boost::lexical_cast<int64_t>(time_tokens[2]),
                        *mapd_as_int_p);
            }
            break;
          }
          default:
            CHECK(false);
        }
      }
    }
  }

 private:
  static void checkTypeConsistency(const int ref_col_type, const SQLTypeInfo& mapd_ti) {
    if (ref_col_type == SQLITE_NULL) {
      // TODO(alex): re-enable the check that mapd_ti is nullable,
      //             got invalidated because of outer joins
      return;
    }
    if (mapd_ti.is_integer()) {
      CHECK_EQ(SQLITE_INTEGER, ref_col_type);
    } else if (mapd_ti.is_fp() || mapd_ti.is_decimal()) {
      CHECK_EQ(SQLITE_FLOAT, ref_col_type);
    } else {
      CHECK_EQ(SQLITE_TEXT, ref_col_type);
    }
  }

  SqliteConnector connector_;
};

const ssize_t g_num_rows{10};
SQLiteComparator g_sqlite_comparator;

void c(const std::string& query_string, const ExecutorDeviceType device_type) {
  g_sqlite_comparator.compare(query_string, device_type);
}

void c(const std::string& query_string, const std::string& sqlite_query_string, const ExecutorDeviceType device_type) {
  g_sqlite_comparator.compare(query_string, sqlite_query_string, device_type);
}

/* timestamp approximate checking for NOW() */
void cta(const std::string& query_string, const ExecutorDeviceType device_type) {
  g_sqlite_comparator.compare_timstamp_approx(query_string, device_type);
}

void c_arrow(const std::string& query_string, const ExecutorDeviceType device_type) {
  g_sqlite_comparator.compare_arrow_output(query_string, query_string, device_type);
}

}  // namespace

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

bool validate_statement_syntax(const std::string& stmt) {
  SQLParser parser;
  list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  return parser.parse(stmt, parse_trees, last_parsed) == 0;
}

TEST(Create, PageSize) {
  std::vector<std::string> page_sizes = {"2097152", "4194304", "10485760"};
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (const auto& page_size : page_sizes) {
      run_ddl_statement("DROP TABLE IF EXISTS test1;");
      EXPECT_NO_THROW(run_ddl_statement("CREATE TABLE test1 (t1 TEXT) WITH (page_size=" + page_size + ");"));
      EXPECT_NO_THROW(run_multiple_agg("INSERT INTO test1 VALUES('hello, MapD');", dt));
      EXPECT_NO_THROW(run_multiple_agg("SELECT * FROM test1;", dt));
    }
  }
}
// Code is commented out while we resolve the leak in parser
// TEST(Create, PageSize_NegativeCase) {
//  run_ddl_statement("DROP TABLE IF EXISTS test1;");
//  ASSERT_EQ(validate_statement_syntax("CREATE TABLE test1 (t1 TEXT) WITH (page_size=null);"), false);
//  ASSERT_EQ(validate_statement_syntax("CREATE TABLE test1 (t1 TEXT) WITH (page_size=);"), false);
//  EXPECT_THROW(run_ddl_statement("CREATE TABLE test1 (t1 TEXT) WITH (page_size=-1);"), std::runtime_error);
//  EXPECT_THROW(run_ddl_statement("CREATE TABLE test1 (t1 TEXT) WITH (page_size=0);"), std::runtime_error);
//  EXPECT_THROW(run_ddl_statement("CREATE TABLE test1 (t1 TEXT) WITH (page_size=2147483648);"), std::runtime_error);
//}

TEST(Select, FilterAndSimpleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test;", dt);
    c("SELECT COUNT(f) FROM test;", dt);
    c("SELECT MIN(x) FROM test;", dt);
    c("SELECT MAX(x) FROM test;", dt);
    c("SELECT MIN(z) FROM test;", dt);
    c("SELECT MAX(z) FROM test;", dt);
    c("SELECT MIN(t) FROM test;", dt);
    c("SELECT MAX(t) FROM test;", dt);
    c("SELECT MIN(ff) FROM test;", dt);
    c("SELECT MIN(fn) FROM test;", dt);
    c("SELECT SUM(ff) FROM test;", dt);
    c("SELECT SUM(fn) FROM test;", dt);
    c("SELECT SUM(x + y) FROM test;", dt);
    c("SELECT SUM(x + y + z) FROM test;", dt);
    c("SELECT SUM(x + y + z + t) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > 1000 AND t < 1002;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 102) OR (t > 1000 AND t < 1003);", dt);
    c("SELECT COUNT(*) FROM test WHERE x <> 7;", dt);
    c("SELECT COUNT(*) FROM test WHERE z <> 102;", dt);
    c("SELECT COUNT(*) FROM test WHERE t <> 1002;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y = 49;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y + z = 150;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + y + z + t = 1151;", dt);
    c("SELECT SUM(x + y) FROM test WHERE x + y = 49;", dt);
    c("SELECT SUM(x + y + z) FROM test WHERE x + y = 49;", dt);
    c("SELECT SUM(x + y + z + t) FROM test WHERE x + y = 49;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y = -35;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y + z = 66;", dt);
    c("SELECT COUNT(*) FROM test WHERE x - y + z + t = 1067;", dt);
    c("SELECT COUNT(*) FROM test WHERE y - x = 35;", dt);
    c("SELECT SUM(2 * x) FROM test WHERE x = 7;", dt);
    c("SELECT SUM(2 * x + z) FROM test WHERE x = 7;", dt);
    c("SELECT SUM(x + y) FROM test WHERE x - y = -35;", dt);
    c("SELECT SUM(x + y) FROM test WHERE y - x = 35;", dt);
    c("SELECT SUM(x + y - z) FROM test WHERE y - x = 35;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT SUM(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT MIN(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + 1 = 151;", dt);
    c("SELECT MAX(x * y + 15) FROM test WHERE x + y + z + t + 1 = 1152;", dt);
    c("SELECT MIN(x) FROM test WHERE x = 7;", dt);
    c("SELECT MIN(z) FROM test WHERE z = 101;", dt);
    c("SELECT MIN(t) FROM test WHERE t = 1001;", dt);
    c("SELECT AVG(x + y) FROM test;", dt);
    c("SELECT AVG(x + y + z) FROM test;", dt);
    c("SELECT AVG(x + y + z + t) FROM test;", dt);
    c("SELECT AVG(y) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT AVG(y) FROM test WHERE z > 100 AND z < 102;", dt);
    c("SELECT AVG(y) FROM test WHERE t > 1000 AND t < 1002;", dt);
    c("SELECT MIN(dd) FROM test;", dt);
    c("SELECT MAX(dd) FROM test;", dt);
    c("SELECT SUM(dd) FROM test;", dt);
    c("SELECT AVG(dd) FROM test;", dt);
    c("SELECT AVG(dd) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 100;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 200;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 300;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 111.0;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 111.1;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > 222.2;", dt);
    c("SELECT MAX(x + dd) FROM test;", dt);
    c("SELECT MAX(x + 2 * dd), MIN(x + 2 * dd) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(111.0 AS decimal(10, 2));", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(222.0 AS decimal(10, 2));", dt);
    c("SELECT COUNT(*) FROM test WHERE dd > CAST(333.0 AS decimal(10, 2));", dt);
    c("SELECT MIN(dd * dd) FROM test;", dt);
    c("SELECT MAX(dd * dd) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE u IS NOT NULL;", dt);
    c("SELECT AVG(u * f) FROM test;", dt);
    c("SELECT AVG(u * d) FROM test;", dt);
    c("SELECT SUM(-y) FROM test;", dt);
    c("SELECT SUM(-z) FROM test;", dt);
    c("SELECT SUM(-t) FROM test;", dt);
    c("SELECT SUM(-dd) FROM test;", dt);
    c("SELECT SUM(-f) FROM test;", dt);
    c("SELECT SUM(-d) FROM test;", dt);
    c("SELECT SUM(dd * 0.99) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE 1<>2;", dt);
    c("SELECT COUNT(*) FROM test WHERE 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE 22 > 33;", dt);
    c("SELECT COUNT(*) FROM test WHERE ff < 23.0/4.0 AND 22 < 33;", dt);
    c("SELECT COUNT(*) FROM test WHERE x + 3*8/2 < 35 + y - 20/5;", dt);
    c("SELECT x + 2 * 10/4 + 3 AS expr FROM test WHERE x + 3*8/2 < 35 + y - 20/5 ORDER BY expr ASC;", dt);
    c("SELECT COUNT(*) FROM test WHERE ff + 3.0*8 < 20.0/5;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y AND 0=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y AND 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y OR 1<1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y OR 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < 35 AND x < y AND 1=1 AND 0=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE 1>2 AND x < 35 AND x < y AND y < 10;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y GROUP BY x HAVING 0=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y GROUP BY x HAVING 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE ofq >= 0 OR ofq IS NULL;", dt);
    c("SELECT COUNT(*) AS val FROM test WHERE (test.dd = 0.5 OR test.dd = 3);", dt);
    c("SELECT MAX(dd_notnull * 1) FROM test;", dt);
    c("SELECT x, COUNT(*) AS n FROM test GROUP BY x, ufd ORDER BY x, n;", dt);
    c("SELECT MIN(x), MAX(x) FROM test WHERE real_str LIKE '%nope%';", dt);
    c("SELECT COUNT(*) FROM test WHERE (x > 7 AND y / (x - 7) < 44);", dt);
    c("SELECT x, AVG(ff) AS val FROM test GROUP BY x ORDER BY val;", dt);
    c("SELECT x, MAX(fn) as val FROM test WHERE fn IS NOT NULL GROUP BY x ORDER BY val;", dt);
    c("SELECT MAX(dn) FROM test WHERE dn IS NOT NULL;", dt);
    c("SELECT x, MAX(dn) as val FROM test WHERE dn IS NOT NULL GROUP BY x ORDER BY val;", dt);
    c("SELECT COUNT(*) as val FROM test GROUP BY x, y, ufd ORDER BY val;", dt);
    ASSERT_NEAR(static_cast<double>(-1000.3),
                v<double>(run_simple_agg("SELECT AVG(fn) AS val FROM test GROUP BY rowid ORDER BY val LIMIT 1;", dt)),
                static_cast<double>(0.2));
    c("SELECT COUNT(*) FROM test WHERE d = 2.2", dt);
    c("SELECT COUNT(*) FROM test WHERE fx + 1 IS NULL;", dt);
    c("SELECT COUNT(ss) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE null IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE null_str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE null IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 > '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 <= '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 = '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 <> '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o >= CAST('1999-09-09' AS DATE);", dt);
    ASSERT_EQ(19, v<int64_t>(run_simple_agg("SELECT rowid FROM test WHERE rowid = 19;", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT MAX(rowid) - MIN(rowid) + 1 FROM test;", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) = 0;", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) = 7;", dt)));
    ASSERT_EQ(5, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) <> 0;", dt)));
    ASSERT_EQ(20, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) <> 7;", dt)));
    c("SELECT MIN(x) FROM test WHERE x <> 7 AND x <> 8;", dt);
    c("SELECT MIN(x) FROM test WHERE z <> 101 AND z <> 102;", dt);
    c("SELECT MIN(x) FROM test WHERE t <> 1001 AND t <> 1002;", dt);
    ASSERT_NEAR(static_cast<double>(0.5),
                v<double>(run_simple_agg("SELECT STDDEV_POP(x) FROM test;", dt)),
                static_cast<double>(0.2));
    ASSERT_NEAR(static_cast<double>(0.5),
                v<double>(run_simple_agg("SELECT STDDEV_SAMP(x) FROM test;", dt)),
                static_cast<double>(0.2));
    ASSERT_NEAR(static_cast<double>(0.2),
                v<double>(run_simple_agg("SELECT VAR_POP(x) FROM test;", dt)),
                static_cast<double>(0.1));
    ASSERT_NEAR(static_cast<double>(0.2),
                v<double>(run_simple_agg("SELECT VAR_SAMP(x) FROM test;", dt)),
                static_cast<double>(0.1));
    ASSERT_NEAR(static_cast<double>(92.0),
                v<double>(run_simple_agg("SELECT STDDEV_POP(dd) FROM test;", dt)),
                static_cast<double>(2.0));
    ASSERT_NEAR(static_cast<double>(94.5),
                v<double>(run_simple_agg("SELECT STDDEV_SAMP(dd) FROM test;", dt)),
                static_cast<double>(1.0));
    ASSERT_NEAR(
        static_cast<double>(94.5),
        v<double>(run_simple_agg(
            "SELECT POWER(((SUM(dd * dd) - SUM(dd) * SUM(dd) / COUNT(dd)) / (COUNT(dd) - 1)), 0.5) FROM test;", dt)),
        static_cast<double>(1.0));
    ASSERT_NEAR(static_cast<double>(8485.0),
                v<double>(run_simple_agg("SELECT VAR_POP(dd) FROM test;", dt)),
                static_cast<double>(10.0));
    ASSERT_NEAR(static_cast<double>(8932.0),
                v<double>(run_simple_agg("SELECT VAR_SAMP(dd) FROM test;", dt)),
                static_cast<double>(10.0));
    ASSERT_EQ(20, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test HAVING STDDEV_POP(x) < 1.0;", dt)));
    ASSERT_EQ(20, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test HAVING STDDEV_POP(x) * 5 < 3.0;", dt)));
    ASSERT_NEAR(static_cast<double>(0.65),
                v<double>(run_simple_agg("SELECT stddev(x) + VARIANCE(x) FROM test;", dt)),
                static_cast<double>(0.10));
    ASSERT_NEAR(static_cast<float>(0.5),
                v<float>(run_simple_agg("SELECT STDDEV_POP_FLOAT(x) FROM test;", dt)),
                static_cast<float>(0.2));
    ASSERT_NEAR(static_cast<float>(0.5),
                v<float>(run_simple_agg("SELECT STDDEV_SAMP_FLOAT(x) FROM test;", dt)),
                static_cast<float>(0.2));
    ASSERT_NEAR(static_cast<float>(0.2),
                v<float>(run_simple_agg("SELECT VAR_POP_FLOAT(x) FROM test;", dt)),
                static_cast<float>(0.1));
    ASSERT_NEAR(static_cast<float>(0.2),
                v<float>(run_simple_agg("SELECT VAR_SAMP_FLOAT(x) FROM test;", dt)),
                static_cast<float>(0.1));
    ASSERT_NEAR(static_cast<float>(92.0),
                v<float>(run_simple_agg("SELECT STDDEV_POP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(2.0));
    ASSERT_NEAR(static_cast<float>(94.5),
                v<float>(run_simple_agg("SELECT STDDEV_SAMP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(1.0));
    ASSERT_NEAR(
        static_cast<double>(94.5),
        v<double>(run_simple_agg(
            "SELECT POWER(((SUM(dd * dd) - SUM(dd) * SUM(dd) / COUNT(dd)) / (COUNT(dd) - 1)), 0.5) FROM test;", dt)),
        static_cast<double>(1.0));
    ASSERT_NEAR(static_cast<float>(8485.0),
                v<float>(run_simple_agg("SELECT VAR_POP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(10.0));
    ASSERT_NEAR(static_cast<float>(8932.0),
                v<float>(run_simple_agg("SELECT VAR_SAMP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(10.0));
    ASSERT_EQ(20, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test HAVING STDDEV_POP_FLOAT(x) < 1.0;", dt)));
    ASSERT_EQ(20, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test HAVING STDDEV_POP_FLOAT(x) * 5 < 3.0;", dt)));
    ASSERT_NEAR(static_cast<float>(0.65),
                v<float>(run_simple_agg("SELECT stddev_FLOAT(x) + VARIANCE_float(x) FROM test;", dt)),
                static_cast<float>(0.10));
    ASSERT_NEAR(static_cast<double>(0.125),
                v<double>(run_simple_agg("SELECT COVAR_POP(x, y) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<float>(0.125),
                v<float>(run_simple_agg("SELECT COVAR_POP_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.001));
    ASSERT_NEAR(static_cast<double>(0.125),  // covar_pop expansion
                v<double>(run_simple_agg("SELECT avg(x * y) - avg(x) * avg(y) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<double>(0.131),
                v<double>(run_simple_agg("SELECT COVAR_SAMP(x, y) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<float>(0.131),
                v<float>(run_simple_agg("SELECT COVAR_SAMP_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.001));
    ASSERT_NEAR(static_cast<double>(0.131),  // covar_samp expansion
                v<double>(run_simple_agg("SELECT ((sum(x * y) - sum(x) * avg(y)) / (count(x) - 1)) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<double>(0.58),
                v<double>(run_simple_agg("SELECT CORRELATION(x, y) FROM test;", dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<float>(0.58),
                v<float>(run_simple_agg("SELECT CORRELATION_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.01));
    ASSERT_NEAR(static_cast<double>(0.58),
                v<double>(run_simple_agg("SELECT CORR(x, y) FROM test;", dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<float>(0.58),
                v<float>(run_simple_agg("SELECT CORR_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.01));
    ASSERT_NEAR(static_cast<double>(0.33),
                v<double>(run_simple_agg("SELECT POWER(CORR(x, y), 2) FROM test;", dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(0.58),  // corr expansion
                v<double>(run_simple_agg("SELECT (avg(x * y) - avg(x) * avg(y)) /"
                                         "(stddev_pop(x) * stddev_pop(y)) FROM test;",
                                         dt)),
                static_cast<double>(0.01));
  }
}

TEST(Select, LimitAndOffset) {
  CHECK(g_num_rows >= 4);
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 5;", dt);
      ASSERT_EQ(size_t(5), rows->rowCount());
    }
    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 5 OFFSET 3;", dt);
      ASSERT_EQ(size_t(5), rows->rowCount());
    }
    {
      const auto rows = run_multiple_agg("SELECT * FROM test WHERE x <> 8 LIMIT 3 OFFSET 1;", dt);
      ASSERT_EQ(size_t(3), rows->rowCount());
    }
    EXPECT_THROW(run_multiple_agg("SELECT * FROM test LIMIT 0;", dt), std::runtime_error);
  }
}

TEST(Select, FloatAndDoubleTests) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(f) FROM test;", dt);
    c("SELECT MAX(f) FROM test;", dt);
    c("SELECT AVG(f) FROM test;", dt);
    c("SELECT MIN(d) FROM test;", dt);
    c("SELECT MAX(d) FROM test;", dt);
    c("SELECT AVG(d) FROM test;", dt);
    c("SELECT SUM(f) FROM test;", dt);
    c("SELECT SUM(d) FROM test;", dt);
    c("SELECT SUM(f + d) FROM test;", dt);
    c("SELECT AVG(x * f) FROM test;", dt);
    c("SELECT AVG(z - 200) FROM test;", dt);
    c("SELECT SUM(CAST(x AS FLOAT)) FROM test;", dt);
    c("SELECT SUM(CAST(x AS FLOAT)) FROM test GROUP BY z;", dt);
    c("SELECT AVG(CAST(x AS FLOAT)) FROM test;", dt);
    c("SELECT AVG(CAST(x AS FLOAT)) FROM test GROUP BY y;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.101 AND f < 1.299;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.201 AND f < 1.4;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2 AND d > 2.0 AND d < 2.4;", dt);
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2 OR (d > 2.0 AND d < 3.0);", dt);
    c("SELECT SUM(x + y) FROM test WHERE f > 1.0 AND f < 1.2;", dt);
    c("SELECT SUM(x + y) FROM test WHERE d + f > 3.0 AND d + f < 4.0;", dt);
    c("SELECT SUM(f + d) FROM test WHERE x - y = -35;", dt);
    c("SELECT SUM(f + d) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT SUM(f * d + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), AVG(x * f + 15), COUNT(*) FROM test WHERE x + y > 47 AND x + y < 51;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(y) > 42.0 ORDER BY n;", dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) > 1.09 ORDER BY n;", dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) > 1.09 AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(d) > 2.2 AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(d) > 2.2 AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f) + AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) + AVG(d) > 3.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f) + AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) + AVG(d) > 3.5 ORDER BY n;",
      dt);
    c("SELECT f + d AS s, x * y FROM test ORDER by s DESC;", dt);
    c("SELECT COUNT(*) AS n FROM test GROUP BY f ORDER BY n;", dt);
    c("SELECT f, COUNT(*) FROM test GROUP BY f HAVING f > 1.25;", dt);
    c("SELECT COUNT(*) AS n FROM test GROUP BY d ORDER BY n;", dt);
    c("SELECT MIN(x + y) AS n FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY f + 1, f + d ORDER BY n;", dt);
    c("SELECT f + d AS s FROM test GROUP BY s ORDER BY s DESC;", dt);
    c("SELECT f + 1 AS s, AVG(u * f) FROM test GROUP BY s ORDER BY s DESC;", dt);
    c("SELECT (CAST(dd AS float) * 0.5) AS key FROM test GROUP BY key ORDER BY key DESC;", dt);
    c("SELECT (CAST(dd AS double) * 0.5) AS key FROM test GROUP BY key ORDER BY key DESC;", dt);
    c("SELECT fn FROM test ORDER BY fn ASC NULLS FIRST;", "SELECT fn FROM test ORDER BY fn ASC;", dt);
    c("SELECT fn FROM test WHERE fn < 0 OR fn IS NULL ORDER BY fn ASC NULLS FIRST;",
      "SELECT fn FROM test WHERE fn < 0 OR fn IS NULL ORDER BY fn ASC;",
      dt);
    ASSERT_NEAR(
        static_cast<double>(1.3),
        v<double>(run_simple_agg("SELECT AVG(f) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(y) + STDDEV(y) "
                                 "> 42.0 ORDER BY n + VARIANCE(y);",
                                 dt)),
        static_cast<double>(0.1));
    ASSERT_NEAR(static_cast<double>(92.0),
                v<double>(run_simple_agg("SELECT STDDEV_POP(dd) AS n FROM test ORDER BY n;", dt)),
                static_cast<double>(1.0));
  }
}

TEST(Select, FilterShortCircuit) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > 1000 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > 1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND t > 1004 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > 1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND t > 1004 AND t > 1005 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > 1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND UNLIKELY(t < 111) AND (str LIKE 'f__%%');",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND UNLIKELY(z < 200) AND z > 100 AND z < 102 AND t > 1000 AND "
      "t > 1000 AND t > 1001  AND UNLIKELY(t < 1111 AND t > 1100) AND (str LIKE 'f__%%') AND t > 1002 AND t > 1003;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE UNLIKELY(x IN (7, 8, 9, 10)) AND y > 42;", dt);
  }
}

TEST(Select, FilterAndMultipleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT AVG(x), AVG(y) FROM test;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), COUNT(*) FROM test WHERE x + y > 47 AND x + y < 51;", dt);
    c("SELECT str, AVG(x), COUNT(*) as xx, COUNT(*) as countval FROM test GROUP BY str ORDER BY str;", dt);
  }
}

TEST(Select, FilterAndGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", dt);
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x + 1, x + y;", dt);
    c("SELECT x, y, COUNT(*) FROM test GROUP BY x, y;", dt);
    c("SELECT x, dd, COUNT(*) FROM test GROUP BY x, dd ORDER BY x, dd;", dt);
    c("SELECT dd AS key1, COUNT(*) AS value1 FROM test GROUP BY key1 HAVING key1 IS NOT NULL ORDER BY key1, value1 "
      "DESC "
      "LIMIT 12;",
      dt);
    c("SELECT 'literal_string' AS key0 FROM test GROUP BY key0;", dt);
    c("SELECT str, MIN(y) FROM test WHERE y IS NOT NULL GROUP BY str ORDER BY str DESC;", dt);
    c("SELECT x, MAX(z) FROM test WHERE z IS NOT NULL GROUP BY x HAVING x > 7;", dt);
    c("SELECT CAST((dd - 0.5) * 2.0 AS int) AS key0, COUNT(*) AS val FROM test WHERE (dd >= 100.0 AND dd < 400.0) "
      "GROUP "
      "BY key0 HAVING key0 >= 0 AND key0 < 400 ORDER BY val DESC LIMIT 50 OFFSET 0;",
      dt);
    c("SELECT y, AVG(CASE WHEN x BETWEEN 6 AND 7 THEN x END) FROM test GROUP BY y ORDER BY y;", dt);
    c("SELECT x, AVG(u), COUNT(*) AS n FROM test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT f, ss FROM test GROUP BY f, ss ORDER BY f DESC;", dt);
    c("SELECT fx, COUNT(*) FROM test GROUP BY fx HAVING COUNT(*) > 5;", dt);
    c("SELECT CASE WHEN x > 8 THEN 100000000 ELSE 42 END AS c, COUNT(*) FROM test GROUP BY c;", dt);
    c("SELECT COUNT(*) FROM test WHERE CAST((CAST(x AS FLOAT) - 0) * 0.2 AS INT) = 1;", dt);
    c("SELECT CAST(CAST(d AS FLOAT) AS INTEGER) AS key, COUNT(*) FROM test GROUP BY key;", dt);
    c("SELECT x * 2 AS x2, COUNT(DISTINCT y) AS n FROM test GROUP BY x2 ORDER BY n DESC;", dt);
    c("SELECT x, COUNT(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt);
    c("SELECT str, SUM(y - y) FROM test GROUP BY str ORDER BY str ASC;", dt);
    c("SELECT str, SUM(y - y) FROM test WHERE y - y IS NOT NULL GROUP BY str ORDER BY str ASC;", dt);
    c("select shared_dict,m from test where (m >= CAST('2014-12-13 22:23:15' AS TIMESTAMP(0)) and m <= "
      "CAST('2014-12-14 22:23:15' AS TIMESTAMP(0)))  and CAST(m AS TIMESTAMP(0)) BETWEEN '2014-12-14 22:23:15' AND "
      "'2014-12-13 22:23:15' group by shared_dict,m;",
      dt);
    EXPECT_THROW(run_multiple_agg("SELECT x, MIN(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT x, MAX(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT MIN(str) FROM test GROUP BY x;", dt), std::runtime_error);
  }
}

TEST(Select, FilterAndGroupByMultipleAgg) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", dt);
    c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x + 1, x + y;", dt);
  }
}

TEST(Select, Having) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING MAX(x) > 5 ORDER BY n;", dt);
    c("SELECT MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING MAX(x) > 5 ORDER BY n LIMIT 1;", dt);
    c("SELECT MAX(y) AS n FROM test WHERE x > 7 GROUP BY z HAVING MAX(x) < 100 ORDER BY n;", dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 ORDER BY n;", dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 AND COUNT(*) > 5 ORDER BY n;", dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 AND COUNT(*) > 9 ORDER BY n;", dt);
    c("SELECT str, COUNT(*) AS n FROM test GROUP BY str HAVING str IN ('bar', 'baz') ORDER BY str;", dt);
    c("SELECT str, COUNT(*) AS n FROM test GROUP BY str HAVING str LIKE 'ba_' ORDER BY str;", dt);
    c("SELECT ss, COUNT(*) AS n FROM test GROUP BY ss HAVING ss LIKE 'bo_' ORDER BY ss;", dt);
    c("SELECT x, COUNT(*) FROM test WHERE x > 9 GROUP BY x HAVING x > 15;", dt);
    c("SELECT x, AVG(y), AVG(y) FROM test GROUP BY x HAVING x >= 0 ORDER BY x;", dt);
    c("SELECT AVG(y), x, AVG(y) FROM test GROUP BY x HAVING x >= 0 ORDER BY x;", dt);
    c("SELECT x, y, COUNT(*) FROM test WHERE real_str LIKE 'nope%' GROUP BY x, y HAVING x >= 0 AND x < 12 AND y >= 0 "
      "AND y < 12 ORDER BY x, y;",
      dt);
  }
}

TEST(Select, CountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(distinct x) FROM test;", dt);
    c("SELECT COUNT(distinct b) FROM test;", dt);
    c("SELECT COUNT(distinct f) FROM test;", dt);
    c("SELECT COUNT(distinct d) FROM test;", dt);
    c("SELECT COUNT(distinct str) FROM test;", dt);
    c("SELECT COUNT(distinct ss) FROM test;", dt);
    c("SELECT COUNT(distinct x + 1) FROM test;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test GROUP BY y ORDER BY n;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(distinct dd) AS n FROM test GROUP BY y ORDER BY n;", dt);
    c("SELECT z, str, AVG(z), COUNT(distinct z) FROM test GROUP BY z, str ORDER BY z, str;", dt);
    c("SELECT AVG(z), COUNT(distinct x) AS dx FROM test GROUP BY y HAVING dx > 1;", dt);
    c("SELECT z, str, COUNT(distinct f) FROM test GROUP BY z, str ORDER BY str DESC;", dt);
    c("SELECT COUNT(distinct x * (50000 - 1)) FROM test;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct real_str) FROM test;", dt), std::runtime_error);
  }
}

TEST(Select, ApproxCountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT APPROX_COUNT_DISTINCT(x) FROM test;", "SELECT COUNT(distinct x) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(b) FROM test;", "SELECT COUNT(distinct b) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(f) FROM test;", "SELECT COUNT(distinct f) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(d) FROM test;", "SELECT COUNT(distinct d) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(str) FROM test;", "SELECT COUNT(distinct str) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(null_str) FROM test;", "SELECT COUNT(distinct null_str) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(ss) FROM test WHERE ss IS NOT NULL;", "SELECT COUNT(distinct ss) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(x + 1) FROM test;", "SELECT COUNT(distinct x + 1) FROM test;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x) FROM test GROUP BY y ORDER "
      "BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x + 1) FROM test GROUP BY y "
      "ORDER BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(dd) AS n FROM test GROUP BY y ORDER BY n;",
      "SELECT COUNT(distinct dd) AS n FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT z, str, AVG(z), APPROX_COUNT_DISTINCT(z) FROM test GROUP BY z, str ORDER BY z;",
      "SELECT z, str, AVG(z), COUNT(distinct z) FROM test GROUP BY z, str ORDER BY z;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(null_str) AS n FROM test GROUP BY x ORDER BY n;",
      "SELECT COUNT(distinct null_str) AS n FROM test GROUP BY x ORDER BY n;",
      dt);
    c("SELECT z, APPROX_COUNT_DISTINCT(null_str) AS n FROM test GROUP BY z ORDER BY z, n;",
      "SELECT z, COUNT(distinct null_str) AS n FROM test GROUP BY z ORDER BY z, n;",
      dt);
    c("SELECT AVG(z), APPROX_COUNT_DISTINCT(x) AS dx FROM test GROUP BY y HAVING dx > 1;",
      "SELECT AVG(z), COUNT(distinct x) AS dx FROM test GROUP BY y HAVING dx > 1;",
      dt);
    c("SELECT approx_value, exact_value FROM (SELECT APPROX_COUNT_DISTINCT(x) AS approx_value FROM test), (SELECT "
      "COUNT(distinct x) AS exact_value FROM test);",
      "SELECT approx_value, exact_value FROM (SELECT COUNT(distinct x) AS approx_value FROM test), (SELECT "
      "COUNT(distinct x) AS exact_value FROM test);",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(x, 1) FROM test;", "SELECT COUNT(distinct x) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(b, 10) FROM test;", "SELECT COUNT(distinct b) FROM test;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(f, 20) FROM test;", "SELECT COUNT(distinct f) FROM test;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x, 1) FROM test GROUP BY y ORDER "
      "BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x + 1, 1) FROM test GROUP BY y "
      "ORDER BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM test GROUP BY y ORDER BY n;",
      dt);
    EXPECT_THROW(run_multiple_agg("SELECT APPROX_COUNT_DISTINCT(real_str) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT APPROX_COUNT_DISTINCT(x, 0) FROM test;", dt), std::runtime_error);
  }
}

TEST(Select, ScanNoAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT * FROM test ORDER BY x ASC, y ASC;", dt);
    c("SELECT t.* FROM test t ORDER BY x ASC, y ASC;", dt);
    c("SELECT x, z, t FROM test ORDER BY x ASC, y ASC;", dt);
    c("SELECT x, y, x + 1 FROM test ORDER BY x ASC, y ASC;", dt);
    c("SELECT x + z, t FROM test WHERE x <> 7 AND y > 42;", dt);
    c("SELECT * FROM test WHERE x > 8;", dt);
  }
}

TEST(Select, OrderBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    const auto rows = run_multiple_agg("SELECT x, y, z + t, x * y AS m FROM test ORDER BY 3 desc LIMIT 5;", dt);
    CHECK_EQ(rows->rowCount(), std::min(size_t(5), static_cast<size_t>(g_num_rows)));
    CHECK_EQ(rows->colCount(), size_t(4));
    for (size_t row_idx = 0; row_idx < rows->rowCount(); ++row_idx) {
      ASSERT_TRUE(v<int64_t>(rows->getRowAt(row_idx, 0, true)) == 8 ||
                  v<int64_t>(rows->getRowAt(row_idx, 0, true)) == 7);
      ASSERT_EQ(v<int64_t>(rows->getRowAt(row_idx, 1, true)), 43);
      ASSERT_EQ(v<int64_t>(rows->getRowAt(row_idx, 2, true)), 1104);
      ASSERT_TRUE(v<int64_t>(rows->getRowAt(row_idx, 3, true)) == 344 ||
                  v<int64_t>(rows->getRowAt(row_idx, 3, true)) == 301);
    }
    c("SELECT x, COUNT(distinct y) AS n FROM test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT x x1, x, COUNT(*) AS val FROM test GROUP BY x HAVING val > 5 ORDER BY val DESC LIMIT 5;", dt);
    c("SELECT ufd, COUNT(*) n FROM test GROUP BY ufd, str ORDER BY ufd, n;", dt);
    c("SELECT -x, COUNT(*) FROM test GROUP BY x ORDER BY x DESC;", dt);
    c("SELECT real_str FROM test WHERE real_str LIKE '%real%' ORDER BY real_str ASC;", dt);
    c("SELECT ss FROM test GROUP by ss ORDER BY ss ASC NULLS FIRST;",
      "SELECT ss FROM test GROUP by ss ORDER BY ss ASC;",
      dt);
    c("SELECT str, COUNT(*) n FROM test WHERE x < 0 GROUP BY str ORDER BY n DESC LIMIT 5;", dt);
    c("SELECT x FROM test ORDER BY x LIMIT 50;", dt);
    c("SELECT x FROM test ORDER BY x LIMIT 5;", dt);
    c("SELECT x FROM test ORDER BY x ASC LIMIT 20;", dt);
    c("SELECT dd FROM test ORDER BY dd ASC LIMIT 20;", dt);
    c("SELECT f FROM test ORDER BY f ASC LIMIT 5;", dt);
    c("SELECT f FROM test ORDER BY f ASC LIMIT 20;", dt);
    c("SELECT fn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT fn as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT fn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT fn as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT dn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT dn as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT dn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT dn as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT ff as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT ff as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT ff as k FROM test ORDER BY k ASC NULLS FIRST  LIMIT 20;",
      "SELECT ff as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT d as k FROM test ORDER BY k ASC LIMIT 5;", dt);
    c("SELECT d as k FROM test ORDER BY k ASC LIMIT 20;", dt);
    c("SELECT dn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT dn as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT dn as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT dn as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT ofq AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT ofq as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT ofq AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT ofq as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT ufq as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT ufq as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT ufq as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT ufq as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT CAST(ofd AS FLOAT) as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT CAST(ofd AS FLOAT) as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT CAST(ofd AS FLOAT) as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT CAST(ofd AS FLOAT) as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT CAST(ufd AS FLOAT) as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 5;",
      "SELECT CAST(ufd AS FLOAT) as k FROM test ORDER BY k ASC LIMIT 5;",
      dt);
    c("SELECT CAST(ufd AS FLOAT) as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT CAST(ufd AS FLOAT) as k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT m AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT m AS k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT n AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT n AS k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
    c("SELECT o AS k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
      "SELECT o AS k FROM test ORDER BY k ASC LIMIT 20;",
      dt);
  }
}

TEST(Select, ComplexQueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) * MAX(y) - SUM(z) FROM test;", dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 AND 200 GROUP BY x, y;", dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x);",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x) + 35;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x) + 36;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test "
      "WHERE z BETWEEN 100 AND 200 GROUP BY a, y;",
      dt);
    c("SELECT x, y FROM (SELECT a.str AS str, b.x AS x, a.y AS y FROM test a, join_test b WHERE a.x = b.x) WHERE str = "
      "'foo' ORDER BY x LIMIT 1;",
      dt);
    const auto rows = run_multiple_agg(
        "SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test "
        "WHERE z BETWEEN 100 AND 200 GROUP BY x, y ORDER BY a DESC LIMIT 2;",
        dt);
    ASSERT_EQ(rows->rowCount(), size_t(2));
    {
      auto crt_row = rows->getNextRow(true, true);
      CHECK_EQ(size_t(2), crt_row.size());
      ASSERT_EQ(v<int64_t>(crt_row[0]), 50);
      ASSERT_EQ(v<int64_t>(crt_row[1]), -295);
    }
    {
      auto crt_row = rows->getNextRow(true, true);
      CHECK_EQ(size_t(2), crt_row.size());
      ASSERT_EQ(v<int64_t>(crt_row[0]), 49);
      ASSERT_EQ(v<int64_t>(crt_row[1]), -590);
    }
    auto empty_row = rows->getNextRow(true, true);
    CHECK(empty_row.empty());
  }
}

TEST(Select, GroupByExprNoFilterNoAggregate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x + y AS a FROM test GROUP BY a ORDER BY a;", dt);
    ASSERT_EQ(8,
              v<int64_t>(run_simple_agg(
                  "SELECT TRUNCATE(x, 0) AS foo FROM test GROUP BY TRUNCATE(x, 0) ORDER BY foo DESC LIMIT 1;", dt)));
  }
}

TEST(Select, DistinctProjection) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT DISTINCT str FROM test ORDER BY str;", dt);
    c("SELECT DISTINCT(str), SUM(x) FROM test WHERE x > 7 GROUP BY str LIMIT 2;", dt);
  }
}

TEST(Select, Case) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) FROM test;", dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 END) FROM test;", dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) "
      "FROM test WHERE CASE WHEN y BETWEEN 42 AND 43 THEN 5 ELSE 4 END > 4;",
      dt);
    ASSERT_EQ(std::numeric_limits<int64_t>::min(),
              v<int64_t>(run_simple_agg(
                  "SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) FROM test "
                  "WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5 ELSE 4 END > 4;",
                  dt)));
    c("SELECT CASE WHEN x + y > 50 THEN 77 ELSE 88 END AS foo, COUNT(*) FROM test GROUP BY foo ORDER BY foo;", dt);
    ASSERT_EQ(std::numeric_limits<double>::min(),
              v<double>(run_simple_agg(
                  "SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1.1 WHEN x BETWEEN 8 AND 9 THEN 2.2 ELSE 3.3 END) FROM "
                  "test WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5.1 ELSE 3.9 END > 4;",
                  dt)));
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN 'oops 2' ELSE real_str END c FROM "
      "test ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN 'oops 2' ELSE str END c FROM test "
      "ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' END c FROM "
      "test ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN real_str ELSE 'ooops' END AS g "
      "FROM test ORDER BY g ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN str ELSE 'ooops' END c FROM test "
      "ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' END c FROM "
      "test ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN str WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' END AS g, "
      "COUNT(*) FROM test GROUP BY g ORDER BY g;",
      dt);
    c("SELECT y AS key0, SUM(CASE WHEN x > 7 THEN x / (x - 7) ELSE 99 END) FROM test GROUP BY key0 ORDER BY key0;", dt);
    c("SELECT CASE WHEN str IN ('str1', 'str3', 'str8') THEN 'foo' WHEN str IN ('str2', 'str4', 'str9') THEN 'bar' "
      "ELSE 'baz' END AS bucketed_str, COUNT(*) AS n FROM query_rewrite_test GROUP BY bucketed_str ORDER BY n DESC;",
      dt);
    c("SELECT CASE WHEN y > 40 THEN x END c, x FROM test ORDER BY c ASC;", dt);
    c("SELECT COUNT(CASE WHEN str = 'foo' THEN 1 END) FROM test;", dt);
    c("SELECT COUNT(CASE WHEN str = 'foo' THEN 1 ELSE NULL END) FROM test;", dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN y ELSE y END AS foobar FROM test ORDER BY foobar DESC;", dt);
    c("SELECT x, AVG(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x ORDER BY x;", dt);
    c("SELECT x, SUM(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x ORDER BY x;", dt);
    c("SELECT x, COUNT(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x ORDER BY x;", dt);
    c("SELECT CASE WHEN x > 8 THEN 'oops' ELSE 'ok' END FROM test LIMIT 1;", dt);
    c("SELECT CASE WHEN x < 9 THEN 'ok' ELSE 'oops' END FROM test LIMIT 1;", dt);
    c("SELECT CASE WHEN str IN ('foo', 'bar') THEN str END key1, COUNT(*) FROM test GROUP BY str HAVING key1 IS NOT "
      "NULL ORDER BY key1;",
      dt);
    c("SELECT CASE WHEN str IN ('foo') THEN 'FOO' WHEN str IN ('bar') THEN 'BAR' ELSE 'BAZ' END AS g, COUNT(*) "
      "FROM test GROUP BY g ORDER BY g DESC;",
      dt);
    c("SELECT x, COUNT(case when y = 42 then 1 else 0 end) AS n1, COUNT(*) AS n2 FROM test GROUP BY x ORDER BY n2 "
      "DESC;",
      dt);
    c("SELECT CASE WHEN test.str = 'foo' THEN 'foo' ELSE test.str END AS g FROM test GROUP BY g ORDER BY g ASC;", dt);
    ASSERT_EQ(int64_t(1418428800),
              v<int64_t>(run_simple_agg("SELECT CASE WHEN 1 > 0 THEN DATE_TRUNC(day, m) ELSE DATE_TRUNC(year, m) END "
                                        "AS date_bin FROM test GROUP BY date_bin;",
                                        dt)));
    ASSERT_EQ(int64_t(1388534400),
              v<int64_t>(run_simple_agg("SELECT CASE WHEN 1 < 0 THEN DATE_TRUNC(day, m) ELSE DATE_TRUNC(year, m) END "
                                        "AS date_bin FROM test GROUP BY date_bin;",
                                        dt)));
    c("SELECT COUNT(CASE WHEN str IN ('foo', 'bar') THEN 'foo_bar' END) from test;", dt);
    ASSERT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg("SELECT MIN(CASE WHEN x BETWEEN 7 AND 8 THEN true ELSE false END) FROM test;", dt)));
    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg("SELECT MIN(CASE WHEN x BETWEEN 6 AND 7 THEN true ELSE false END) FROM test;", dt)));
  }
}

TEST(Select, Strings) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, COUNT(*) FROM test GROUP BY str HAVING COUNT(*) > 5 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 4 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 5 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test where str IS NOT NULL GROUP BY str ORDER BY str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE 'ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%eal_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%ba%';", dt);
    c("SELECT * FROM test WHERE str LIKE '%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE str LIKE 'f%%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE str LIKE 'f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE ss LIKE 'f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE str LIKE '@f%%' ESCAPE '@' ORDER BY x ASC, y ASC;", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE 'ba_' or str LIKE 'fo_';", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str > 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str > 'fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE str >= 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' < str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'fo' < str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <= str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' = str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str <> 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <> str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str = 'foo' OR str = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str <> str;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss <> str;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss = str;", dt);
    c("SELECT COUNT(*) FROM test WHERE LENGTH(str) = 3;", dt);
    c("SELECT fixed_str, COUNT(*) FROM test GROUP BY fixed_str HAVING COUNT(*) > 5 ORDER BY fixed_str;", dt);
    c("SELECT fixed_str, COUNT(*) FROM test WHERE fixed_str = 'bar' GROUP BY fixed_str HAVING COUNT(*) > 4 ORDER BY "
      "fixed_str;",
      dt);
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(str) = 3;", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str ILIKE 'f%%';", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE (str ILIKE 'f%%');", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE ( str ILIKE 'f%%' );", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str ILIKE 'McDonald''s';", dt)));
    ASSERT_EQ("foo",
              boost::get<std::string>(
                  v<NullableString>(run_simple_agg("SELECT str FROM test WHERE REGEXP_LIKE(str, '^f.?.+');", dt))));
    ASSERT_EQ("bar",
              boost::get<std::string>(
                  v<NullableString>(run_simple_agg("SELECT str FROM test WHERE REGEXP_LIKE(str, '^[a-z]+r$');", dt))));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP '.*';", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP '...';", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP '.+.+.+';", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP '.?.?.?';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or str REGEXP 'fo.';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str, 'ba.') or str REGEXP 'fo.?';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or REGEXP_LIKE(str, 'fo.+');", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.+';", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str, '.?ba.*');", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str,'ba.') or REGEXP_LIKE(str, 'fo.+');", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or REGEXP_LIKE(str, 'fo.+');", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str, 'ba.') or str REGEXP 'fo.?';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or str REGEXP 'fo.';", dt)));
  }
}

TEST(Select, SharedDictionary) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT shared_dict, COUNT(*) FROM test GROUP BY shared_dict HAVING COUNT(*) > 5 ORDER BY shared_dict;", dt);
    c("SELECT shared_dict, COUNT(*) FROM test WHERE shared_dict = 'bar' GROUP BY shared_dict HAVING COUNT(*) > 4 ORDER "
      "BY shared_dict;",
      dt);
    c("SELECT shared_dict, COUNT(*) FROM test WHERE shared_dict = 'bar' GROUP BY shared_dict HAVING COUNT(*) > 5 ORDER "
      "BY shared_dict;",
      dt);
    c("SELECT shared_dict, COUNT(*) FROM test where shared_dict IS NOT NULL GROUP BY shared_dict ORDER BY shared_dict;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE '%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE 'ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE '%eal_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE '%ba%';", dt);
    c("SELECT * FROM test WHERE shared_dict LIKE '%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE shared_dict LIKE 'f%%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE shared_dict LIKE 'f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE ss LIKE 'f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE shared_dict LIKE '@f%%' ESCAPE '@' ORDER BY x ASC, y ASC;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE 'ba_' or shared_dict LIKE 'fo_';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' = shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict <> 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <> shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict = 'foo' OR shared_dict = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict <> shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict >= 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' < shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'fo' < shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <= shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE LENGTH(shared_dict) = 3;", dt);

    EXPECT_THROW(run_ddl_statement("CREATE TABLE t1(a text, b text, SHARED DICTIONARY (b) REFERENCES t1(a), SHARED "
                                   "DICTIONARY (a) REFERENCES t1(b));"),
                 std::runtime_error);

    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(shared_dict) = 3;", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict ILIKE 'f%%';", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE (shared_dict ILIKE 'f%%');", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE ( shared_dict ILIKE 'f%%' );", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict ILIKE 'McDonald''s';", dt)));

    ASSERT_EQ("foo",
              boost::get<std::string>(v<NullableString>(
                  run_simple_agg("SELECT shared_dict FROM test WHERE REGEXP_LIKE(shared_dict, '^f.?.+');", dt))));
    ASSERT_EQ("baz",
              boost::get<std::string>(v<NullableString>(
                  run_simple_agg("SELECT shared_dict FROM test WHERE REGEXP_LIKE(shared_dict, '^[a-z]+z$');", dt))));

    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.*';", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '...';", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.+.+.+';", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.?.?.?';", dt)));

    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP 'ba.' or shared_dict REGEXP 'fo.';", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, 'ba.') or shared_dict REGEXP 'fo.?';", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP 'ba.' or REGEXP_LIKE(shared_dict, 'fo.+');", dt)));
    ASSERT_EQ(5, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP 'ba.+';", dt)));
    ASSERT_EQ(5, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, '.?ba.*');", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict,'ba.') or REGEXP_LIKE(shared_dict, 'fo.+');",
                  dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP 'ba.' or REGEXP_LIKE(shared_dict, 'fo.+');", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, 'ba.') or shared_dict REGEXP 'fo.?';", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP 'ba.' or shared_dict REGEXP 'fo.';", dt)));
  }
}

TEST(Select, StringCompare) {
  if (g_fast_strcmp) {
    for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
      SKIP_NO_GPU();
      c("SELECT COUNT(*) FROM test WHERE str = 'ba';", dt);
      c("SELECT COUNT(*) FROM test WHERE str <> 'ba';", dt);

      c("SELECT COUNT(*) FROM test WHERE shared_dict < 'ba';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict < 'bar';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict < 'baf';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict < 'baz';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict < 'bbz';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict < 'foo';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict < 'foon';", dt);

      c("SELECT COUNT(*) FROM test WHERE shared_dict > 'ba';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict > 'bar';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict > 'baf';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict > 'baz';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict > 'bbz';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict > 'foo';", dt);
      c("SELECT COUNT(*) FROM test WHERE shared_dict > 'foon';", dt);

      c("SELECT COUNT(*) FROM test WHERE real_str <= 'ba';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str <= 'bar';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str <= 'baf';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str <= 'baz';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str <= 'bbz';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str <= 'foo';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str <= 'foon';", dt);

      c("SELECT COUNT(*) FROM test WHERE real_str >= 'ba';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str >= 'bar';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str >= 'baf';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str >= 'baz';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str >= 'bbz';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str >= 'foo';", dt);
      c("SELECT COUNT(*) FROM test WHERE real_str >= 'foon';", dt);

      c("SELECT COUNT(*) FROM test WHERE real_str <= 'äâ';", dt);

      c("SELECT COUNT(*) FROM test WHERE 'ba' < shared_dict;", dt);
      c("SELECT COUNT(*) FROM test WHERE 'bar' < shared_dict;", dt);
      c("SELECT COUNT(*) FROM test WHERE 'ba' > shared_dict;", dt);
      c("SELECT COUNT(*) FROM test WHERE 'bar' > shared_dict;", dt);

      EXPECT_THROW(
          run_multiple_agg("SELECT COUNT(*) FROM test, test_inner WHERE test.shared_dict < test_inner.str", dt),
          std::runtime_error);
    }
  }
}

TEST(Select, StringsNoneEncoding) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE '%eal_bar';", dt);
    c("SELECT * FROM test WHERE real_str LIKE '%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_@f%%' ESCAPE '@' ORDER BY x ASC, y ASC;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba_' or real_str LIKE 'real_fo_';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str > 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str > 'real_fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str >= 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' < real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_fo' < real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' <= real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <> 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'real_bar' <> real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = 'real_foo' OR real_str = 'real_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <> real_str;", dt);
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str ILIKE 'rEaL_f%%';", dt)));
    c("SELECT COUNT(*) FROM test WHERE LENGTH(real_str) = 8;", dt);
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(real_str) = 8;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(real_str,'real_.*.*.*');", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_ba.*';", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str REGEXP '.*';", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_f.*.*';", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_f.+\%';", dt)));
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE real_str LIKE str;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(real_str, str);", dt),
                 std::runtime_error);
  }
}

namespace {

void check_date_trunc_groups(const ResultSet& rows) {
  {
    const auto crt_row = rows.getNextRow(true, true);
    CHECK(!crt_row.empty());
    CHECK_EQ(size_t(3), crt_row.size());
    const auto sv0 = v<int64_t>(crt_row[0]);
    ASSERT_EQ(int64_t(936144000), sv0);
    const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
    ASSERT_EQ("foo", sv1);
    const auto sv2 = v<int64_t>(crt_row[2]);
    ASSERT_EQ(g_num_rows, sv2);
  }
  {
    const auto crt_row = rows.getNextRow(true, true);
    CHECK(!crt_row.empty());
    CHECK_EQ(size_t(3), crt_row.size());
    const auto sv0 = v<int64_t>(crt_row[0]);
    ASSERT_EQ(inline_int_null_val(rows.getColType(0)), sv0);
    const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
    ASSERT_EQ("bar", sv1);
    const auto sv2 = v<int64_t>(crt_row[2]);
    ASSERT_EQ(g_num_rows / 2, sv2);
  }
  {
    const auto crt_row = rows.getNextRow(true, true);
    CHECK(!crt_row.empty());
    CHECK_EQ(size_t(3), crt_row.size());
    const auto sv0 = v<int64_t>(crt_row[0]);
    ASSERT_EQ(int64_t(936144000), sv0);
    const auto sv1 = boost::get<std::string>(v<NullableString>(crt_row[1]));
    ASSERT_EQ("baz", sv1);
    const auto sv2 = v<int64_t>(crt_row[2]);
    ASSERT_EQ(g_num_rows / 2, sv2);
  }
  const auto crt_row = rows.getNextRow(true, true);
  CHECK(crt_row.empty());
}

void check_one_date_trunc_group(const ResultSet& rows, const int64_t ref_ts) {
  const auto crt_row = rows.getNextRow(true, true);
  ASSERT_EQ(size_t(1), crt_row.size());
  const auto actual_ts = v<int64_t>(crt_row[0]);
  ASSERT_EQ(ref_ts, actual_ts);
  const auto empty_row = rows.getNextRow(true, true);
  ASSERT_TRUE(empty_row.empty());
}

}  // namespace

TEST(Select, Time) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // check DATE Formats
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('10/09/1999' AS DATE) > o;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('10-Sep-99' AS DATE) > o;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('31/Oct/2013' AS DATE) > o;", dt)));
    // check TIME FORMATS
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('151315' AS TIME) > n;", dt)));

    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) <= o;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) <= n;", dt)));
    cta("SELECT DATETIME('NOW') FROM test limit 1;", dt);
    // these next tests work because all dates are before now 2015-12-8 17:00:00
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m < NOW();", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m > timestamp(0) '2014-12-13T000000';", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST(o AS TIMESTAMP) > timestamp(0) '1999-09-08T160000';", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST(o AS TIMESTAMP) > timestamp(0) '1999-09-10T160000';", dt)));
    ASSERT_EQ(14185957950L, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(20140, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(120, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(140, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(22, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM m)) FROM test;", dt)));
    ASSERT_EQ(23, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM m)) FROM test;", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM m)) FROM test;", dt)));
    ASSERT_EQ(6, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOW FROM m)) FROM test;", dt)));
    ASSERT_EQ(348, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOY FROM m)) FROM test;", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM n)) FROM test;", dt)));
    ASSERT_EQ(13, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM n)) FROM test;", dt)));
    ASSERT_EQ(14, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM n)) FROM test;", dt)));
    ASSERT_EQ(1999, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM o)) FROM test;", dt)));
    ASSERT_EQ(9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM o)) FROM test;", dt)));
    ASSERT_EQ(9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM o)) FROM test;", dt)));
    ASSERT_EQ(4, v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM o) FROM test WHERE o IS NOT NULL;", dt)));
    ASSERT_EQ(252, v<int64_t>(run_simple_agg("SELECT EXTRACT(DOY FROM o) FROM test WHERE o IS NOT NULL;", dt)));
    ASSERT_EQ(936835200L, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM o)) FROM test;", dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(WEEK FROM CAST('2012-01-01 20:15:12' AS TIMESTAMP))) FROM test limit 1;", dt)));
    ASSERT_EQ(10L,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(WEEK FROM CAST('2008-03-03 20:15:12' AS TIMESTAMP))) FROM test limit 1;", dt)));
    // Monday
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(DOW FROM CAST('2008-03-03 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    // Monday
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(ISODOW FROM CAST('2008-03-03 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    // Sunday
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(DOW FROM CAST('2008-03-02 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    // Sunday
    ASSERT_EQ(7L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(ISODOW FROM CAST('2008-03-02 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));

    // do some DATE_TRUNC tests
    /*
     * year
     * month
     * day
     * hour
     * minute
     * second
     *
     * millennium
     * century
     * decade
     * milliseconds
     * microseconds
     * week
     */
    ASSERT_EQ(1325376000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(year, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1335830400L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(month, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336435200L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(day, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336507200L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(hour, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(second, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(978307200L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millennium, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(978307200L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(century, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1293840000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millisecond, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(microsecond, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(1336262400L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(week, CAST('2012-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));

    ASSERT_EQ(-2114380800L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(year, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2104012800L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(month, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103408000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(day, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103336000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(hour, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(second, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-30578688000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millennium, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2177452800L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(century, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2177452800L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millisecond, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(microsecond, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2103840000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(week, CAST('1903-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));

    ASSERT_EQ(31536000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, CAST('1972-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    ASSERT_EQ(662688000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, CAST('2000-05-08 20:15:12' AS TIMESTAMP)) FROM test limit 1;", dt)));
    // test QUARTER
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT(quarter FROM CAST('2008-11-27 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT(quarter FROM CAST('2008-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1199145600L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('2008-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1230768000L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('2009-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1254355200L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('2009-11-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(946684800L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('2000-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(-2208988800L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC(quarter, CAST('1900-03-21 12:12:12' AS timestamp)) FROM test limit 1;", dt)));
    // test DATE format processing
    ASSERT_EQ(1434844800L, v<int64_t>(run_simple_agg("select CAST('2015-06-21' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE o < CAST('06/21/2015' AS DATE);", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE o < CAST('21-Jun-15' AS DATE);", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE o < CAST('21/Jun/2015' AS DATE);", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE o < CAST('1434844800' AS DATE);", dt)));

    // test different input formats
    // added new format for customer
    ASSERT_EQ(1434896116L,
              v<int64_t>(run_simple_agg("select CAST('2015-06-21 14:15:16' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= CAST('2015-06-21:141516' AS TIMESTAMP);", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('21-JUN-15 2.15.16.12345 PM' AS TIMESTAMP);", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('21-JUN-15 2.15.16.12345 AM' AS TIMESTAMP);", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('21-JUN-15 2:15:16 AM' AS TIMESTAMP);", dt)));

    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('06/21/2015 14:15:16' AS TIMESTAMP);", dt)));

    // Support ISO date offset format
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('21/Aug/2015:12:13:14 -0600' AS TIMESTAMP);", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('2015-08-21T12:13:14 -0600' AS TIMESTAMP);", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('21-Aug-15 12:13:14 -0600' AS TIMESTAMP);", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('21/Aug/2015:13:13:14 -0500' AS TIMESTAMP);", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m <= CAST('2015-08-21T18:13:14' AS TIMESTAMP);", dt)));
    // add test for quarterday behaviour
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T04:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T00:00:00' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T08:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T14:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T23:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1440115200L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T04:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1440136800L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T08:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1440158400L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T13:23:11' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(1440180000L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T23:59:59' AS timestamp)) FROM test limit 1;", dt)));
    ASSERT_EQ(
        2007,
        v<int64_t>(run_simple_agg("SELECT DATEPART('year', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        2007,
        v<int64_t>(run_simple_agg("SELECT DATEPART('yyyy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        2007,
        v<int64_t>(run_simple_agg("SELECT DATEPART('yy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('quarter', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg("SELECT DATEPART('qq', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        4, v<int64_t>(run_simple_agg("SELECT DATEPART('q', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(10,
              v<int64_t>(
                  run_simple_agg("SELECT DATEPART('month', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg("SELECT DATEPART('mm', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg("SELECT DATEPART('m', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(303,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('dayofyear', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        303,
        v<int64_t>(run_simple_agg("SELECT DATEPART('dy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        303,
        v<int64_t>(run_simple_agg("SELECT DATEPART('y', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg("SELECT DATEPART('day', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg("SELECT DATEPART('dd', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg("SELECT DATEPART('d', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg("SELECT DATEPART('hour', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg("SELECT DATEPART('hh', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('minute', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT DATEPART('mi', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT DATEPART('n', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(32,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('second', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg("SELECT DATEPART('ss', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg("SELECT DATEPART('s', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;", dt)));
    ASSERT_EQ(32, v<int64_t>(run_simple_agg("SELECT DATEPART('s', TIMESTAMP '2007-10-30 12:15:32') FROM test;", dt)));
    ASSERT_EQ(3,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', CAST('2006-01-07 00:00:00' as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(36,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', CAST('2006-01-07 00:00:00' as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1096,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', CAST('2006-01-07 00:00:00' as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(12,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('quarter', CAST('2006-01-07 00:00:00' as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', DATE '2009-2-28', DATE '2009-03-01') FROM TEST LIMIT 1;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', DATE '2008-2-28', DATE '2008-03-01') FROM TEST LIMIT 1;", dt)));
    ASSERT_EQ(-425,
              v<int64_t>(run_simple_agg(
                  "select DATEDIFF('day', DATE '1971-03-02', DATE '1970-01-01') from test limit 1;", dt)));
    ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT DATEDIFF('day', o, o + INTERVAL '1' DAY) FROM TEST LIMIT 1;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(
                  run_simple_agg("SELECT count(*) from test where DATEDIFF('day', CAST (m AS DATE), o) < -5570;", dt)));
    // DATEADD tests
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('day', 1, CAST('2017-05-31' AS DATE)) = TIMESTAMP '2017-06-01 0:00:00' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('day', 2, DATE '2017-05-31') = TIMESTAMP '2017-06-02 0:00:00' from test limit 1;", dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('day', -1, CAST('2017-05-31' AS DATE)) = TIMESTAMP '2017-05-30 0:00:00' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('day', -2, DATE '2017-05-31') = TIMESTAMP '2017-05-29 0:00:00' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('hour', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 2:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('hour', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 11:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('hour', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 0:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('hour', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-30 15:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('minute', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 1:12:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('minute', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 1:21:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('minute', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 1:10:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('minute', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 1:01:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('second', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 1:11:12' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('second', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 1:11:21' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('second', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 1:11:10' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('second', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                                        "'2017-05-31 1:11:01' from test limit 1;",
                                        dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 1, DATE '2017-01-10') = TIMESTAMP "
                                        "'2017-02-10 0:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 10, DATE '2017-01-10') = TIMESTAMP "
                                        "'2017-11-10 0:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 1, DATE '2009-01-30') = TIMESTAMP "
                                        "'2009-02-28 0:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 1, DATE '2008-01-30') = TIMESTAMP "
                                        "'2008-02-29 0:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 1, TIMESTAMP '2009-01-30 1:11:11') = TIMESTAMP "
                                        "'2009-02-28 1:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', -1, TIMESTAMP '2009-03-30 1:11:11') = TIMESTAMP "
                                        "'2009-02-28 1:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', -4, TIMESTAMP '2009-03-30 1:11:11') = TIMESTAMP "
                                        "'2008-11-30 1:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 5, TIMESTAMP '2009-01-31 1:11:11') = TIMESTAMP "
                                        "'2009-6-30 1:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', 1, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
                                        "'2009-02-28 1:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(YEAR, 1, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
                                        "'2009-02-28 1:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(YEAR, -8, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
                                        "'2000-02-29 1:11:11' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(YEAR, -8, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
                                        "'2000-02-29 1:11:11' from test limit 1;",
                                        dt)));

    ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT m = TIMESTAMP '2014-12-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('day', 1, m) = TIMESTAMP '2014-12-14 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('day', -1, m) = TIMESTAMP '2014-12-12 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('day', 1, m) = TIMESTAMP '2014-12-14 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('day', -1, m) = TIMESTAMP '2014-12-12 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT o = DATE '1999-09-09' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('day', 1, o) = TIMESTAMP '1999-09-10 0:00:00' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('day', -3, o) = TIMESTAMP '1999-09-06 0:00:00' from test limit 1;", dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, 1, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-03 1:23:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, -1, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-01 1:23:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, 15, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-17 1:23:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, -15, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-02-15 1:23:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 1, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-02 2:23:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, -1, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-02 0:23:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 15, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-02 16:23:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, -15, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-01 10:23:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, 15, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-02 1:38:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, -15, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-02 1:08:45' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, 15, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-02 1:24:00' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, -15, TIMESTAMP '2009-03-02 1:23:45') = TIMESTAMP '2009-03-02 1:23:30' "
                  "FROM TEST LIMIT 1;",
                  dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, 1, m) = TIMESTAMP '2014-12-14 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, -1, m) = TIMESTAMP '2014-12-12 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, 15, m) = TIMESTAMP '2014-12-28 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, -15, m) = TIMESTAMP '2014-11-28 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, 1, m) = TIMESTAMP '2014-12-13 23:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, -1, m) = TIMESTAMP '2014-12-13 21:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, 15, m) = TIMESTAMP '2014-12-14 13:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, -15, m) = TIMESTAMP '2014-12-13 7:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(MINUTE, 15, m) = TIMESTAMP '2014-12-13 22:38:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(MINUTE, -15, m) = TIMESTAMP '2014-12-13 22:08:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SECOND, 15, m) = TIMESTAMP '2014-12-13 22:23:30' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SECOND, -15, m) = TIMESTAMP '2014-12-13 22:23:00' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(MONTH, 1, m) = TIMESTAMP '2015-01-13 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(MONTH, -1, m) = TIMESTAMP '2014-11-13 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(MONTH, 5, m) = TIMESTAMP '2015-05-13 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, -5, m) = TIMESTAMP '2014-12-08 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(YEAR, 1, m) = TIMESTAMP '2015-12-13 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(YEAR, -1, m) = TIMESTAMP '2013-12-13 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(YEAR, 5, m) = TIMESTAMP '2019-12-13 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(YEAR, -5, m) = TIMESTAMP '2009-12-13 22:23:15' "
                                        "FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "select count(*) from test where TIMESTAMPADD(YEAR, 15, CAST(o AS TIMESTAMP)) > m;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "select count(*) from test where TIMESTAMPADD(YEAR, 16, CAST(o AS TIMESTAMP)) > m;", dt)));

    ASSERT_EQ(128885,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(minute, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                                        "'2003-05-01 12:05:55') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(2148,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(hour, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                                        "'2003-05-01 12:05:55') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(89,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(day, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                                        "'2003-05-01 12:05:55') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(3,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(month, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                                        "'2003-05-01 12:05:55') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(-3,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(month, TIMESTAMP '2003-05-01 12:05:55', TIMESTAMP "
                                        "'2003-02-01 0:00:00') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(
        5, v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(month, m, m + INTERVAL '5' MONTH) FROM TEST LIMIT 1;", dt)));
    ASSERT_EQ(
        -5,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(month, m, m - INTERVAL '5' MONTH) FROM TEST LIMIT 1;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "select count(*) from test where TIMESTAMPDIFF(YEAR, m, CAST(o AS TIMESTAMP)) < 0;", dt)));

    ASSERT_EQ(1418428800L, v<int64_t>(run_simple_agg("SELECT CAST(m AS date) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(1336435200L,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(CAST('2012-05-08 20:15:12' AS TIMESTAMP) AS DATE) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test GROUP BY CAST(m AS date);", dt)));
    const auto rows = run_multiple_agg(
        "SELECT DATE_TRUNC(month, CAST(o AS TIMESTAMP(0))) AS key0, str AS key1, COUNT(*) AS val FROM test GROUP BY "
        "key0, key1 ORDER BY val DESC, key1;",
        dt);
    check_date_trunc_groups(*rows);
    const auto one_row = run_multiple_agg(
        "SELECT DATE_TRUNC(year, CASE WHEN str = 'foo' THEN m END) d FROM test GROUP BY d "
        "HAVING d IS NOT NULL;",
        dt);
    check_one_date_trunc_group(*one_row, 1388534400);
  }
}

TEST(Select, In) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE x IN (7, 8);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (9, 10);", dt);
    c("SELECT COUNT(*) FROM test WHERE z IN (101, 102);", dt);
    c("SELECT COUNT(*) FROM test WHERE z IN (201, 202);", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar');", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar', 'real_baz', 'foo');", dt);
    c("SELECT COUNT(*) FROM test WHERE str IN ('foo', 'bar', 'real_foo');", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);",
      dt);
  }
}

TEST(Select, DivByZero) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT x / 0 FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT 1 / 0 FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct x / 0) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT d / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / (f - f) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY y / (x - x);", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY z, y / (x - x);", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY MOD(y , (x - x));", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(x) / SUM(CASE WHEN str = 'none' THEN y ELSE 0 END) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(*) FROM test WHERE y / (x - x) = 0;", dt), std::runtime_error);
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x = x OR  y / (x - x) = y;", dt)));
  }
}

TEST(Select, ReturnNullFromDivByZero) {
  g_null_div_by_zero = true;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x / 0 FROM test;", dt);
    c("SELECT 1 / 0 FROM test;", dt);
    c("SELECT f / 0. FROM test;", dt);
    c("SELECT d / 0. FROM test;", dt);
    c("SELECT f / (f - f) FROM test;", dt);
    c("SELECT COUNT(*) FROM test GROUP BY y / (x - x);", dt);
    c("SELECT COUNT(*) FROM test GROUP BY z, y / (x - x);", dt);
    c("SELECT SUM(x) / SUM(CASE WHEN str = 'none' THEN y ELSE 0 END) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE y / (x - x) = 0;", dt);
    c("SELECT COUNT(*) FROM test WHERE x = x OR  y / (x - x) = y;", dt);
  }
}

TEST(Select, OverflowAndUnderFlow) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE z + 32600 > 0;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE z + 32666 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -32670 - z < 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE (z + 16333) * 2 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE x + 2147483640 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -x - 2147483642 < 0;", dt), std::runtime_error);
    c("SELECT COUNT(*) FROM test WHERE t + 9223372036854774000 > 0;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE t + 9223372036854775000 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -t - 9223372036854775000 < 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofd + x - 2 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ufd * 3 - ofd * 1024 < -2;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofd * 2 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofq + 1 > 0;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -ufq - 9223372036854775000 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -92233720368547758 - ofq <= 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT cast((z - -32666) *0.000190 as int) as key0, "
                                  "COUNT(*) AS val FROM test WHERE (z >= -32666 AND z < 31496) "
                                  "GROUP BY key0 HAVING key0 >= 0 AND key0 < 12 ORDER BY val "
                                  "DESC LIMIT 50 OFFSET 0;",
                                  dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT dd * 2000000000000000 FROM test LIMIT 5;", dt), std::runtime_error);
    c("SELECT dd * 200000000000000 FROM test ORDER BY dd ASC LIMIT 5;",
      dt);  // overflow avoided through decimal mul optimization
    c("SELECT COUNT(*) FROM test WHERE dd + 2.0000000000000009 > 110.0;", dt);  // no overflow in the cast
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE dd + 2.00000000000000099 > 110.0;", dt),
                 std::runtime_error);  // overflow in the cast due to higher precision
    c("SELECT dd / 2.00000009 FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // dividend still fits after cast and division upscaling
    EXPECT_THROW(run_multiple_agg("SELECT dd / 2.000000099 FROM test LIMIT 1;", dt),
                 std::runtime_error);  // dividend overflows after cast and division upscaling due to higher precision
    c("SELECT (dd - 40.6364668888) / 2 FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // decimal div by const optimization avoids overflow
    c("SELECT (dd - 40.6364668888) / x FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // decimal div by int cast optimization avoids overflow
    c("SELECT (dd - 40.63646688) / dd FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // dividend still fits after upscaling from cast and division
    EXPECT_THROW(run_multiple_agg("select (dd-40.6364668888)/dd from test limit 1;", dt),
                 std::runtime_error);  // dividend overflows on upscaling on a slightly higher precision, test detection
    c("SELECT cast((cast(z as int) - -32666) *0.000190 as int) as key0, "
      "COUNT(*) AS val FROM test WHERE (z >= -32666 AND z < 31496) "
      "GROUP BY key0 HAVING key0 >= 0 AND key0 < 12 ORDER BY val "
      "DESC LIMIT 50 OFFSET 0;",
      dt);
    c("select -1 * dd as expr from test order by expr asc;", dt);
    c("select dd * -1 as expr from test order by expr asc;", dt);
    c("select (dd - 1000000111.10) * dd as expr from test order by expr asc;", dt);
    c("select dd * (dd - 1000000111.10) as expr from test order by expr asc;", dt);
    // avoiding overflows in decimal compares against higher precision literals:
    // truncate literals based on the other side's precision, e.g. for d which is DECIMAL(14,2)
    c("select count(*) from big_decimal_range_test where (d >  4.955357142857142);", dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d >= 4.955357142857142);", dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d <  4.955357142857142);", dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d <= 4.955357142857142);", dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d >= 4.950357142857142);", dt);  // compare with 4.951
    c("select count(*) from big_decimal_range_test where (d <  4.950357142857142);", dt);  // compare with 4.951
    c("select count(*) from big_decimal_range_test where (d < 59016609.300000056);", dt);  // compare with 59016609.301
    c("select count(*) from test where (t*123456 > 9681668.33071388567);", dt);            // compare with 9681668.3
    c("select count(*) from test where (x*12345678 < 9681668.33071388567);", dt);          // compare with 9681668.3
    c("select count(*) from test where (z*12345678 < 9681668.33071388567);", dt);          // compare with 9681668.3
    c("select count(*) from test where dd <= 111.222;", dt);
    c("select count(*) from test where dd >= -15264923.533545015;", dt);
    // avoiding overflows with constant folding and pushed down casts
    c("select count(*) + (604*575) from test;", dt);
    c("select count(*) - (604*575) from test;", dt);
    c("select count(*) * (604*575) from test;", dt);
    c("select (604*575) / count(*) from test;", dt);
    c("select (400604+575) / count(*) from test;", dt);
    c("select cast(count(*) as DOUBLE) + (604*575) from test;", dt);
    c("select cast(count(*) as DOUBLE) - (604*575) from test;", dt);
    c("select cast(count(*) as DOUBLE) * (604*575) from test;", dt);
    c("select (604*575) / cast(count(*) as DOUBLE) from test;", dt);
    c("select (12345-123456789012345) / cast(count(*) as DOUBLE) from test;", dt);
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(CAST(EXTRACT(QUARTER FROM CAST(NULL AS TIMESTAMP)) AS BIGINT) - 1) FROM test;", dt)));
#ifdef ENABLE_COMPACTION
    c("SELECT SUM(ofd) FROM test GROUP BY x;", dt);
    c("SELECT SUM(ufd) FROM test GROUP BY x;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(ofq) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(ufq) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(ofq) FROM test GROUP BY x;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT SUM(ufq) FROM test GROUP BY x;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT AVG(ofq) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT AVG(ufq) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT AVG(ofq) FROM test GROUP BY y;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT AVG(ufq) FROM test GROUP BY y;", dt), std::runtime_error);
#endif  // ENABLE_COMPACTION
  }
}

TEST(Select, BooleanColumn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(g_num_rows + g_num_rows / 2, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE b;", dt)));
    ASSERT_EQ(g_num_rows / 2, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE NOT b;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x < 8 AND b;", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x < 8 AND NOT b;", dt)));
    ASSERT_EQ(5, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 7 OR false;", dt)));
    ASSERT_EQ(7, v<int64_t>(run_simple_agg("SELECT MAX(x) FROM test WHERE b = CAST('t' AS boolean);", dt)));
    ASSERT_EQ(3 * g_num_rows,
              v<int64_t>(run_simple_agg(" SELECT SUM(2 *(CASE when x = 7 then 1 else 0 END)) FROM test;", dt)));
    c("SELECT COUNT(*) AS n FROM test GROUP BY x = 7, b ORDER BY n;", dt);
  }
}

TEST(Select, UnsupportedCast) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT CAST(x AS VARCHAR) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(f AS VARCHAR) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(d AS VARCHAR) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(f AS DECIMAL) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(dd AS DECIMAL) FROM test;", dt), std::runtime_error);
  }
}

TEST(Select, CastFromLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(2.3 AS SMALLINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS INT) FROM test;", dt);
    c("SELECT CAST(2.3 AS BIGINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS FLOAT) FROM test;", dt);
    c("SELECT CAST(2.3 AS DOUBLE) FROM test;", dt);
    c("SELECT CAST(2.3 AS DECIMAL(2, 1)) FROM test;", dt);
    c("SELECT CAST(2.3 AS NUMERIC(2, 1)) FROM test;", dt);
    c("SELECT CAST(CAST(10 AS float) / CAST(3600 as float) AS float) FROM test LIMIT 1;", dt);
    c("SELECT CAST(CAST(10 AS double) / CAST(3600 as double) AS double) FROM test LIMIT 1;", dt);
    c("SELECT z from test where z = -78;", dt);
  }
}

TEST(Select, CastFromNull) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(NULL AS SMALLINT) FROM test;", dt);
    c("SELECT CAST(NULL AS INT) FROM test;", dt);
    c("SELECT CAST(NULL AS BIGINT) FROM test;", dt);
    c("SELECT CAST(NULL AS FLOAT) FROM test;", dt);
    c("SELECT CAST(NULL AS DOUBLE) FROM test;", dt);
    c("SELECT CAST(NULL AS DECIMAL) FROM test;", dt);
    c("SELECT CAST(NULL AS NUMERIC) FROM test;", dt);
  }
}

TEST(Select, TimeInterval) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(60 * 60 * 1000L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' HOUR FROM test LIMIT 1;", dt)));
    ASSERT_EQ(24 * 60 * 60 * 1000L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' DAY FROM test LIMIT 1;", dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE INTERVAL '1' MONTH < INTERVAL '2' MONTH;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE INTERVAL '1' DAY < INTERVAL '2' DAY;", dt)));
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test GROUP BY INTERVAL '1' DAY;", dt)));
    ASSERT_EQ(3 * 60 * 60 * 1000L, v<int64_t>(run_simple_agg("SELECT 3 * INTERVAL '1' HOUR FROM test LIMIT 1;", dt)));
    ASSERT_EQ(3 * 60 * 60 * 1000L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' HOUR * 3 FROM test LIMIT 1;", dt)));
    ASSERT_EQ(7L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' MONTH * x FROM test WHERE x <> 8 LIMIT 1;", dt)));
    ASSERT_EQ(7L, v<int64_t>(run_simple_agg("SELECT x * INTERVAL '1' MONTH FROM test WHERE x <> 8 LIMIT 1;", dt)));
    ASSERT_EQ(42L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' MONTH * y FROM test WHERE y <> 43 LIMIT 1;", dt)));
    ASSERT_EQ(42L, v<int64_t>(run_simple_agg("SELECT y * INTERVAL '1' MONTH FROM test WHERE y <> 43 LIMIT 1;", dt)));
    ASSERT_EQ(1002L,
              v<int64_t>(run_simple_agg("SELECT INTERVAL '1' MONTH * t FROM test WHERE t <> 1001 LIMIT 1;", dt)));
    ASSERT_EQ(1002L,
              v<int64_t>(run_simple_agg("SELECT t * INTERVAL '1' MONTH FROM test WHERE t <> 1001 LIMIT 1;", dt)));
    ASSERT_EQ(3L, v<int64_t>(run_simple_agg("SELECT INTERVAL '1' MONTH + INTERVAL '2' MONTH FROM test LIMIT 1;", dt)));
    ASSERT_EQ(1388534400L,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(m AS date) + CAST(TRUNCATE(-1 * (EXTRACT(DOY FROM m) - 1), 0) AS INTEGER) * INTERVAL "
                  "'1' DAY AS g FROM test GROUP BY g;",
                  dt)));
    ASSERT_EQ(1417392000L,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(m AS date) + CAST(TRUNCATE(-1 * (EXTRACT(DAY FROM m) - 1), 0) AS INTEGER) * INTERVAL "
                  "'1' DAY AS g FROM test GROUP BY g;",
                  dt)));
    ASSERT_EQ(1418508000L,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(m AS date) + EXTRACT(HOUR FROM m) * INTERVAL '1' HOUR AS g FROM test GROUP BY g;", dt)));
    ASSERT_EQ(1388534400L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SQL_TSI_DAY, CAST(TRUNCATE(-1 * (EXTRACT(DOY from m) - 1), 0) AS INTEGER), "
                  "CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                  dt)));
    ASSERT_EQ(1417392000L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SQL_TSI_DAY, CAST(TRUNCATE(-1 * (EXTRACT(DAY from m) - 1), 0) AS INTEGER), "
                  "CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                  dt)));
    ASSERT_EQ(1418508000L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SQL_TSI_HOUR, EXTRACT(HOUR from m), CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                  dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-1-31' + INTERVAL '1' YEAR) = DATE '2009-01-31' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-1-31' + INTERVAL '5' YEAR) = DATE '2013-01-31' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-1-31' - INTERVAL '1' YEAR) = DATE '2007-01-31' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-1-31' - INTERVAL '4' YEAR) = DATE '2004-01-31' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-1-31' + INTERVAL '1' MONTH) = DATE '2008-02-29' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-1-31' + INTERVAL '5' MONTH) = DATE '2008-06-30' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-1-31' - INTERVAL '1' MONTH) = DATE '2007-12-31' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-1-31' - INTERVAL '4' MONTH) = DATE '2007-09-30' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-2-28' + INTERVAL '1' DAY) = DATE '2008-02-29' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2009-2-28' + INTERVAL '1' DAY) = DATE '2009-03-01' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-2-28' + INTERVAL '4' DAY) = DATE '2008-03-03' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2009-2-28' + INTERVAL '4' DAY) = DATE '2009-03-04' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-03-01' - INTERVAL '1' DAY) = DATE '2008-02-29' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2009-03-01' - INTERVAL '1' DAY) = DATE '2009-02-28' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2008-03-03' - INTERVAL '4' DAY) = DATE '2008-02-28' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (DATE '2009-03-04' - INTERVAL '4' DAY) = DATE '2009-02-28' from test limit 1;", dt)));
    ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT m = TIMESTAMP '2014-12-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m + INTERVAL '1' SECOND) = TIMESTAMP '2014-12-13 22:23:16' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m + INTERVAL '1' MINUTE) = TIMESTAMP '2014-12-13 22:24:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m + INTERVAL '1' HOUR) = TIMESTAMP '2014-12-13 23:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m + INTERVAL '2' DAY) = TIMESTAMP '2014-12-15 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m + INTERVAL '1' MONTH) = TIMESTAMP '2015-01-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m + INTERVAL '1' YEAR) = TIMESTAMP '2015-12-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m - 5 * INTERVAL '1' SECOND) = TIMESTAMP '2014-12-13 22:23:10' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m - x * INTERVAL '1' MINUTE) = TIMESTAMP '2014-12-13 22:16:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m - 2 * x * INTERVAL '1' HOUR) = TIMESTAMP '2014-12-13 8:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m - x * INTERVAL '1' DAY) = TIMESTAMP '2014-12-06 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m - x * INTERVAL '1' MONTH) = TIMESTAMP '2014-05-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m - x * INTERVAL '1' YEAR) = TIMESTAMP '2007-12-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m - INTERVAL '5' DAY + INTERVAL '2' HOUR - x * INTERVAL '2' SECOND) +"
                                        "(x - 1) * INTERVAL '1' MONTH - x * INTERVAL '10' YEAR = "
                                        "TIMESTAMP '1945-06-09 00:23:01' from test limit 1;",
                                        dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("select count(*) from test where m < CAST (o AS TIMESTAMP) + INTERVAL '10' "
                                        "YEAR AND m > CAST(o AS TIMESTAMP) - INTERVAL '10' YEAR;",
                                        dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("select count(*) from test where m < CAST (o AS TIMESTAMP) + INTERVAL '16' "
                                        "YEAR AND m > CAST(o AS TIMESTAMP) - INTERVAL '16' YEAR;",
                                        dt)));

    ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT o = DATE '1999-09-09' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (o + INTERVAL '10' DAY) = DATE '1999-09-19' from test limit 1;", dt)));
  }
}

TEST(Select, UnsupportedNodes) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT 1 + 2;", dt), std::runtime_error);
    // MAT No longer throws a logicalValues gets a regular parse error'
    // EXPECT_THROW(run_multiple_agg("SELECT *;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT x, COUNT(*) FROM test GROUP BY ROLLUP(x);", dt), std::runtime_error);
  }
}

TEST(Select, UnsupportedMultipleArgAggregate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct x, y) FROM test;", dt), std::runtime_error);
  }
}

namespace Importer_NS {

ArrayDatum StringToArray(const std::string& s, const SQLTypeInfo& ti, const CopyParams& copy_params);
void parseStringArray(const std::string& s, const CopyParams& copy_params, std::vector<std::string>& string_vec);

}  // namespace Importer_NS

namespace {

const size_t g_array_test_row_count{20};

void import_array_test(const std::string& table_name) {
  CHECK_EQ(size_t(0), g_array_test_row_count % 4);
  auto& cat = g_session->get_catalog();
  const auto td = cat.getMetadataForTable(table_name);
  CHECK(td);
  Importer_NS::Loader loader(cat, td);
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  const auto col_descs = cat.getAllColumnMetadataForTable(td->tableId, false, false);
  for (const auto cd : col_descs) {
    import_buffers.emplace_back(new Importer_NS::TypedImportBuffer(
        cd,
        cd->columnType.get_compression() == kENCODING_DICT
            ? cat.getMetadataForDict(cd->columnType.get_comp_param())->stringDict.get()
            : nullptr));
  }
  Importer_NS::CopyParams copy_params;
  copy_params.array_begin = '{';
  copy_params.array_end = '}';
  for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
    for (const auto& import_buffer : import_buffers) {
      const auto& ti = import_buffer->getTypeInfo();
      switch (ti.get_type()) {
        case kINT:
          import_buffer->addInt(7 + row_idx);
          break;
        case kARRAY: {
          const auto& elem_ti = ti.get_elem_type();
          std::vector<std::string> array_elems;
          switch (elem_ti.get_type()) {
            case kBOOLEAN: {
              for (size_t i = 0; i < 3; ++i) {
                if (row_idx % 2) {
                  array_elems.push_back("T");
                  array_elems.push_back("F");
                } else {
                  array_elems.push_back("F");
                  array_elems.push_back("T");
                }
              }
              break;
            }
            case kSMALLINT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.push_back(std::to_string(row_idx + i + 1));
              }
              break;
            case kINT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.push_back(std::to_string((row_idx + i + 1) * 10));
              }
              break;
            case kBIGINT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.push_back(std::to_string((row_idx + i + 1) * 100));
              }
              break;
            case kTEXT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.emplace_back(2, 'a' + row_idx + i);
              }
              break;
            case kFLOAT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.emplace_back(std::to_string(row_idx + i + 1) + "." + std::to_string(row_idx + i + 1));
              }
              break;
            case kDOUBLE:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.emplace_back(std::to_string(11 * (row_idx + i + 1)) + "." +
                                         std::to_string(row_idx + i + 1));
              }
              break;
            default:
              CHECK(false);
          }
          if (elem_ti.is_string()) {
            import_buffer->addDictEncodedStringArray({array_elems});
          } else {
            auto arr_str = "{" + boost::algorithm::join(array_elems, ",") + "}";
            import_buffer->addArray(StringToArray(arr_str, ti, copy_params));
          }
          break;
        }
        case kTEXT:
          import_buffer->addString("real_str" + std::to_string(row_idx));
          break;
        default:
          CHECK(false);
      }
    }
  }
  loader.load(import_buffers, g_array_test_row_count);
}

void import_gpu_sort_test() {
  const std::string drop_old_gpu_sort_test{"DROP TABLE IF EXISTS gpu_sort_test;"};
  run_ddl_statement(drop_old_gpu_sort_test);
  g_sqlite_comparator.query(drop_old_gpu_sort_test);
  run_ddl_statement("CREATE TABLE gpu_sort_test(x int) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE gpu_sort_test(x int);");
  for (size_t i = 0; i < 4; ++i) {
    const std::string insert_query{"INSERT INTO gpu_sort_test VALUES(2);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (size_t i = 0; i < 6; ++i) {
    const std::string insert_query{"INSERT INTO gpu_sort_test VALUES(16000);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_query_rewrite_test() {
  const std::string drop_old_query_rewrite_test{"DROP TABLE IF EXISTS query_rewrite_test;"};
  run_ddl_statement(drop_old_query_rewrite_test);
  g_sqlite_comparator.query(drop_old_query_rewrite_test);
  run_ddl_statement("CREATE TABLE query_rewrite_test(x int, str text encoding dict) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE query_rewrite_test(x int, str text);");
  for (size_t i = 1; i <= 30; ++i) {
    for (size_t j = 1; j <= i % 2 + 1; ++j) {
      const std::string insert_query{"INSERT INTO query_rewrite_test VALUES(" + std::to_string(i) + ", 'str" +
                                     std::to_string(i) + "');"};
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
  }
}

void import_big_decimal_range_test() {
  const std::string drop_old_decimal_range_test("DROP TABLE IF EXISTS big_decimal_range_test;");
  run_ddl_statement(drop_old_decimal_range_test);
  g_sqlite_comparator.query(drop_old_decimal_range_test);
  run_ddl_statement("CREATE TABLE big_decimal_range_test(d DECIMAL(14, 2), d1 DECIMAL(17,11)) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE big_decimal_range_test(d DECIMAL(14, 2), d1 DECIMAL(17,11));");
  {
    const std::string insert_query{"INSERT INTO big_decimal_range_test VALUES(-40840124.400000, 1.3);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO big_decimal_range_test VALUES(59016609.300000, 1.3);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_subquery_test() {
  const std::string subquery_test("DROP TABLE IF EXISTS subquery_test;");
  run_ddl_statement(subquery_test);
  g_sqlite_comparator.query(subquery_test);
  run_ddl_statement("CREATE TABLE subquery_test(x int) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE subquery_test(x int);");
  CHECK_EQ(g_num_rows % 2, 0);
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    const std::string insert_query{"INSERT INTO subquery_test VALUES(7);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{"INSERT INTO subquery_test VALUES(8);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{"INSERT INTO subquery_test VALUES(9);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_text_group_by_test() {
  const std::string text_group_by_test("DROP TABLE IF EXISTS text_group_by_test;");
  run_ddl_statement(text_group_by_test);
  run_ddl_statement(
      "CREATE TABLE text_group_by_test(tdef TEXT, tdict TEXT ENCODING DICT, tnone TEXT ENCODING NONE ) WITH "
      "(fragment_size=200);");
  const std::string insert_query{"INSERT INTO text_group_by_test VALUES('hello','world',':-)');"};
  run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
}

void import_join_test() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS join_test;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  std::string columns_definition{"x int not null, y int, str text encoding dict, dup_str text encoding dict"};
  const auto create_test = build_create_table_statement(columns_definition,
                                                        "join_test",
                                                        {g_shard_count ? "dup_str" : "", g_shard_count},
                                                        {},
#ifdef ENABLE_MULTIFRAG_JOIN
                                                        2
#else
                                                        3
#endif
  );
  run_ddl_statement(create_test);
  g_sqlite_comparator.query("CREATE TABLE join_test(x int not null, y int, str text, dup_str text);");
  {
    const std::string insert_query{"INSERT INTO join_test VALUES(7, 43, 'foo', 'foo');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO join_test VALUES(8, null, 'bar', 'foo');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO join_test VALUES(9, null, 'baz', 'bar');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_hash_join_test() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS hash_join_test;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  const std::string create_test{
#ifdef ENABLE_MULTIFRAG_JOIN
      "CREATE TABLE hash_join_test(x int not null, str text encoding dict, t BIGINT) WITH (fragment_size=2);"
#else
      "CREATE TABLE hash_join_test(x int not null, str text encoding dict, t BIGINT) WITH (fragment_size=3);"
#endif
  };
  run_ddl_statement(create_test);
  g_sqlite_comparator.query("CREATE TABLE hash_join_test(x int not null, str text, t BIGINT);");
  {
    const std::string insert_query{"INSERT INTO hash_join_test VALUES(7, 'foo', 1001);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO hash_join_test VALUES(8, 'bar', 5000000000);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO hash_join_test VALUES(9, 'the', 1002);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_emp_table() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS emp;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  const std::string create_test{
      "CREATE TABLE emp(empno INT, ename TEXT ENCODING DICT, deptno INT) WITH (fragment_size=2);"};
  run_ddl_statement(create_test);
  g_sqlite_comparator.query("CREATE TABLE emp(empno INT, ename TEXT ENCODING DICT, deptno INT);");
  {
    const std::string insert_query{"INSERT INTO emp VALUES(1, 'Brock', 10);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO emp VALUES(2, 'Bill', 20);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO emp VALUES(3, 'Julia', 60);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO emp VALUES(4, 'David', 10);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_dept_table() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS dept;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  const std::string create_test{"CREATE TABLE dept(deptno INT, dname TEXT ENCODING DICT) WITH (fragment_size=2);"};
  run_ddl_statement(create_test);
  g_sqlite_comparator.query("CREATE TABLE dept(deptno INT, dname TEXT ENCODING DICT);");
  {
    const std::string insert_query{"INSERT INTO dept VALUES(10, 'Sales');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO dept VALUES(20, 'Dev');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO dept VALUES(30, 'Marketing');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO dept VALUES(40, 'HR');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO dept VALUES(50, 'QA');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

}  // namespace

TEST(Select, ArrayUnnest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    unsigned power10 = 1;
    for (const unsigned int_width : {16, 32, 64}) {
      auto result_rows = run_multiple_agg("SELECT COUNT(*), UNNEST(arr_i" + std::to_string(int_width) +
                                              ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                                          dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count + 2) * power10, v<int64_t>(result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(power10, v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 1, true)));
      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double"}) {
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr_" + float_type + ") AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));
    }
    {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_str) AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));
    }
    {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_bool) AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(size_t(2), result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count * 3), v<int64_t>(result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(int64_t(g_array_test_row_count * 3), v<int64_t>(result_rows->getRowAt(1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(0, v<int64_t>(result_rows->getRowAt(1, 1, true)));
    }
  }
}

TEST(Select, ArrayIndex) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
      ASSERT_EQ(
          1,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_i32[2] = " +
                                        std::to_string(10 * (row_idx + 2)) + " AND x = " + std::to_string(7 + row_idx) +
                                        " AND real_str LIKE 'real_str" + std::to_string(row_idx) + "';",
                                    dt)));
      ASSERT_EQ(
          0,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_i32[4] > 0 OR arr_i32[4] <= 0;", dt)));
      ASSERT_EQ(
          0,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_i32[0] > 0 OR arr_i32[0] <= 0;", dt)));
    }
    for (size_t i = 1; i <= 6; ++i) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count / 2),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_bool[" + std::to_string(i) + "];", dt)));
    }
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_bool[7];", dt)));
    ASSERT_EQ(0, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_bool[0];", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE NOT (arr_i16[7] > 0 AND arr_i16[7] <= 0);", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE NOT (arr_i16[2] > 0 AND arr_i16[2] <= 0);", dt)));
  }
}

TEST(Select, ArrayCountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (const unsigned int_width : {16, 32, 64}) {
      ASSERT_EQ(int64_t(g_array_test_row_count + 2),
                v<int64_t>(run_simple_agg(
                    "SELECT COUNT(distinct arr_i" + std::to_string(int_width) + ") FROM array_test;", dt)));
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(distinct arr_i" + std::to_string(int_width) + ") FROM array_test GROUP BY x;", dt);
      ASSERT_EQ(g_array_test_row_count, result_rows->rowCount());
      for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
        ASSERT_EQ(3, v<int64_t>(result_rows->getRowAt(row_idx, 0, true)));
      }
    }
    for (const std::string float_type : {"float", "double"}) {
      ASSERT_EQ(int64_t(g_array_test_row_count + 2),
                v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr_" + float_type + ") FROM array_test;", dt)));
    }
    ASSERT_EQ(int64_t(g_array_test_row_count + 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr_str) FROM array_test;", dt)));
    ASSERT_EQ(2, v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr_bool) FROM array_test;", dt)));
  }
}

TEST(Select, ArrayAnyAndAll) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    unsigned power10 = 1;
    for (const unsigned int_width : {16, 32, 64}) {
      ASSERT_EQ(2,
                v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE " + std::to_string(2 * power10) +
                                              " = ANY arr_i" + std::to_string(int_width) + ";",
                                          dt)));
      ASSERT_EQ(int64_t(g_array_test_row_count) - 2,
                v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE " + std::to_string(2 * power10) +
                                              " < ALL arr_i" + std::to_string(int_width) + ";",
                                          dt)));
      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double"}) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 1 < ANY arr_" + float_type + ";", dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 2 < ANY arr_" + float_type + ";", dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 0 < ALL arr_" + float_type + ";", dt)));
    }
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE x - 5 = ANY arr_i16;", dt)));
    ASSERT_EQ(1, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'aa' = ANY arr_str;", dt)));
    ASSERT_EQ(2, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' = ANY arr_str;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ANY arr_bool;", dt)));
    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ALL arr_bool;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' < ALL arr_str;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 1),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' <= ALL arr_str;", dt)));
    ASSERT_EQ(int64_t(1), v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' > ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(2), v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE 'bb' >= ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE  real_str = ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE  real_str <> ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 1),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE (NOT ('aa' = ANY arr_str));", dt)));
    // these two test just confirm that the regex does not mess with other similar patterns
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) as SMALL FROM array_test;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) as COMPANY FROM array_test;", dt)));
  }
}

TEST(Select, ArrayUnsupported) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT MIN(arr_i64) FROM array_test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT UNNEST(arr_str), COUNT(*) cc FROM array_test GROUP BY arr_str;", dt),
                 std::runtime_error);
  }
}

TEST(Select, OrRewrite) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE str = 'foo' OR str = 'bar' OR str = 'baz' OR str = 'foo' OR str = 'bar' OR str "
      "= 'baz' OR str = 'foo' OR str = 'bar' OR str = 'baz' OR str = 'baz' OR str = 'foo' OR str = 'bar' OR str = "
      "'baz';",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x = 7 OR x = 8 OR x = 7 OR x = 8 OR x = 7 OR x = 8 OR x = 7 OR x = 8 OR x = 7 "
      "OR x = 8 OR x = 7 OR x = 8;",
      dt);
  }
}

TEST(Select, GpuSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x, COUNT(*) AS n FROM gpu_sort_test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT x, COUNT(*), COUNT(*) AS n FROM gpu_sort_test GROUP BY x ORDER BY n DESC;", dt);
  }
}

TEST(Select, GroupByConstrainedByInQueryRewrite) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) AS n, x FROM query_rewrite_test WHERE x IN (2, 5) GROUP BY x HAVING n > 0 ORDER BY n DESC;", dt);
    c("SELECT COUNT(*) AS n, x FROM query_rewrite_test WHERE x IN (2, 99) GROUP BY x HAVING n > 0 ORDER BY n DESC;",
      dt);
    c("SELECT COUNT(*) AS n, str FROM query_rewrite_test WHERE str IN ('str2', 'str5') GROUP BY str HAVING n > 0 ORDER "
      "BY n DESC;",
      dt);
    c("SELECT COUNT(*) AS n, str FROM query_rewrite_test WHERE str IN ('str2', 'str99') GROUP BY str HAVING n > 0 "
      "ORDER BY n DESC;",
      dt);
  }
}

TEST(Select, BigDecimalRange) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(d AS INT) AS di, COUNT(*) FROM big_decimal_range_test GROUP BY di HAVING di > 0;", dt);
    c("select d1*2 from big_decimal_range_test;", dt);
    c("select 2*d1 from big_decimal_range_test;", dt);
    c("select d1 * (CAST(d1 as INT) + 1) from big_decimal_range_test;", dt);
    c("select (CAST(d1 as INT) + 1) * d1 from big_decimal_range_test;", dt);
  }
}

TEST(Drop, AfterDrop) {
  run_ddl_statement("create table droptest (i1 integer);");
  run_multiple_agg("insert into droptest values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into droptest values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3), v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM droptest;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table droptest;");
  run_ddl_statement("create table droptest (n1 integer);");
  run_multiple_agg("insert into droptest values(3);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into droptest values(4);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(7), v<int64_t>(run_simple_agg("SELECT SUM(n1) FROM droptest;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table droptest;");
}

TEST(Alter, AfterAlterTableName) {
  run_ddl_statement("create table alter_name_test (i1 integer);");
  run_multiple_agg("insert into alter_name_test values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_name_test values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3), v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_name_test;", ExecutorDeviceType::CPU)));
  run_ddl_statement("alter table alter_name_test rename to alter_name_test_after;");
  run_multiple_agg("insert into alter_name_test_after values(3);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_name_test_after values(4);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(10),
            v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_name_test_after;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table alter_name_test_after;");
}

TEST(Alter, AfterAlterColumnName) {
  run_ddl_statement("create table alter_column_test (i1 integer);");
  run_multiple_agg("insert into alter_column_test values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_column_test values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3), v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_column_test;", ExecutorDeviceType::CPU)));
  run_ddl_statement("alter table alter_column_test rename column i1 to n1;");
  run_multiple_agg("insert into alter_column_test values(3);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_column_test values(4);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(10), v<int64_t>(run_simple_agg("SELECT SUM(n1) FROM alter_column_test;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table alter_column_test;");
}

TEST(Select, Empty) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM emptytab;", dt);
    c("SELECT SUM(x) FROM emptytab;", dt);
    c("SELECT SUM(y) FROM emptytab;", dt);
    c("SELECT SUM(t) FROM emptytab;", dt);
    c("SELECT SUM(f) FROM emptytab;", dt);
    c("SELECT SUM(d) FROM emptytab;", dt);
    c("SELECT SUM(dd) FROM emptytab;", dt);
    c("SELECT MIN(x) FROM emptytab;", dt);
    c("SELECT MIN(y) FROM emptytab;", dt);
    c("SELECT MIN(t) FROM emptytab;", dt);
    c("SELECT MIN(f) FROM emptytab;", dt);
    c("SELECT MIN(d) FROM emptytab;", dt);
    c("SELECT MIN(dd) FROM emptytab;", dt);
    c("SELECT MAX(x) FROM emptytab;", dt);
    c("SELECT MAX(y) FROM emptytab;", dt);
    c("SELECT MAX(t) FROM emptytab;", dt);
    c("SELECT MAX(f) FROM emptytab;", dt);
    c("SELECT MAX(d) FROM emptytab;", dt);
    c("SELECT MAX(dd) FROM emptytab;", dt);
    c("SELECT AVG(x) FROM emptytab;", dt);
    c("SELECT AVG(y) FROM emptytab;", dt);
    c("SELECT AVG(t) FROM emptytab;", dt);
    c("SELECT AVG(f) FROM emptytab;", dt);
    c("SELECT AVG(d) FROM emptytab;", dt);
    c("SELECT AVG(dd) FROM emptytab;", dt);
    c("SELECT COUNT(*) FROM test, emptytab;", dt);
    c("SELECT MIN(ts), MAX(ts) FROM emptytab;", dt);
    c("SELECT SUM(test.x) FROM test, emptytab;", dt);
    c("SELECT SUM(test.y) FROM test, emptytab;", dt);
    c("SELECT SUM(emptytab.x) FROM test, emptytab;", dt);
    c("SELECT SUM(emptytab.y) FROM test, emptytab;", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(x) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(f) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(d) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(dd) FROM test WHERE x > 8;", dt);
    c("SELECT SUM(dd) FROM emptytab GROUP BY x, y;", dt);
    c("SELECT COUNT(DISTINCT x) FROM emptytab;", dt);
    c("SELECT APPROX_COUNT_DISTINCT(x * 1000000) FROM emptytab;",
      "SELECT COUNT(DISTINCT x * 1000000) FROM emptytab;",
      dt);
  }
}

TEST(Select, Subqueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, SUM(y) AS n FROM test WHERE x > (SELECT COUNT(*) FROM test) - 14 GROUP BY str ORDER BY str ASC;",
      dt);
    c("SELECT COUNT(*) FROM test, (SELECT x FROM test_inner) AS inner_x WHERE test.x = inner_x.x;", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test WHERE y > 42);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test GROUP BY x ORDER BY COUNT(*) DESC LIMIT 1);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test GROUP BY x);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM join_test);", dt);
    c("SELECT MIN(yy), MAX(yy) FROM (SELECT AVG(y) as yy FROM test GROUP BY x);", dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x NOT IN (SELECT x + 1 FROM subquery_test GROUP BY x);", dt);
    c("SELECT MAX(ct) FROM (SELECT COUNT(*) AS ct, str AS foo FROM test GROUP BY foo);", dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x IN (SELECT x AS foobar FROM subquery_test GROUP BY foobar);", dt);
    c("SELECT * FROM (SELECT x FROM test ORDER BY x) ORDER BY x;", dt);
    c("SELECT AVG(y) FROM (SELECT * FROM test ORDER BY z LIMIT 5);", dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x NOT IN (SELECT x + 1 FROM subquery_test GROUP BY x);", dt);
    ASSERT_EQ(int64_t(0), v<int64_t>(run_simple_agg("SELECT * FROM (SELECT rowid FROM test WHERE rowid = 0);", dt)));
    c("SELECT COUNT(*) FROM test WHERE x NOT IN (SELECT x FROM test GROUP BY x ORDER BY COUNT(*));", dt);
    c("SELECT COUNT(*) FROM test WHERE x NOT IN (SELECT x FROM test GROUP BY x);", dt);
    c("SELECT COUNT(*) FROM test WHERE f IN (SELECT DISTINCT f FROM test WHERE x > 7);", dt);
    c("SELECT emptytab. x, CASE WHEN emptytab. y IN (SELECT emptytab. y FROM emptytab GROUP BY emptytab. y) then "
      "emptytab. y END yy, sum(x) "
      "FROM emptytab GROUP BY emptytab. x, yy;",
      dt);
    c("WITH d1 AS (SELECT deptno, dname FROM dept LIMIT 10) SELECT ename, dname FROM emp, d1 WHERE emp.deptno = "
      "d1.deptno ORDER BY ename ASC LIMIT 10;",
      dt);
    c("SELECT x FROM (SELECT x, MAX(y), COUNT(*) AS n FROM test GROUP BY x HAVING MAX(y) > 42 ORDER BY n);", dt);
    c("SELECT CASE WHEN test.x IN (SELECT x FROM test_inner) THEN x ELSE NULL END AS c, COUNT(*) AS n FROM test WHERE "
      "y > 40 GROUP BY c ORDER BY n DESC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test WHERE x > (SELECT COUNT(*) FROM test WHERE x > 7) + 2 "
      "GROUP BY x);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE ofd IN (SELECT ofd FROM test GROUP BY ofd);", dt);
    c("SELECT COUNT(*) FROM test WHERE ofd NOT IN (SELECT ofd FROM test GROUP BY ofd);", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test WHERE ss NOT IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test WHERE str IN (SELECT str FROM test_in_bitmap GROUP BY str);", dt);
    c("SELECT COUNT(*) FROM test WHERE str NOT IN (SELECT str FROM test_in_bitmap GROUP BY str);", dt);
    c("SELECT COUNT(*) FROM test WHERE str IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test WHERE str NOT IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IN (SELECT str FROM test GROUP BY str);", dt);
    c("SELECT COUNT(*) FROM test WHERE ss NOT IN (SELECT str FROM test GROUP BY str);", dt);
    c("SELECT str, COUNT(*) FROM test WHERE x IN (SELECT x FROM test WHERE x > 8) GROUP BY str;", dt);
    c("SELECT COUNT(*) FROM test_in_bitmap WHERE str IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test_in_bitmap WHERE str NOT IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test_in_bitmap WHERE str IN (SELECT str FROM test GROUP BY str);", dt);
    c("SELECT COUNT(*) FROM test_in_bitmap WHERE str NOT IN (SELECT str FROM test GROUP BY str);", dt);
    c("SELECT COUNT(str) FROM (SELECT * FROM (SELECT * FROM test WHERE x = 7) WHERE y = 42) WHERE t > 1000;", dt);
    c("SELECT x_cap, y FROM (SELECT CASE WHEN x > 100 THEN 100 ELSE x END x_cap, y, t FROM emptytab) GROUP BY x_cap, "
      "y;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str IN (SELECT DISTINCT str FROM test);", dt);
#ifdef ENABLE_JOIN_EXEC
    c("SELECT SUM((x - (SELECT AVG(x) FROM test)) * (x - (SELECT AVG(x) FROM test)) / ((SELECT COUNT(x) FROM test) - "
      "1)) FROM test;",
      dt);
#endif
    EXPECT_THROW(run_multiple_agg("SELECT * FROM (SELECT * FROM test LIMIT 5);", dt), std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT AVG(SELECT x FROM test LIMIT 5) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE str < (SELECT str FROM test LIMIT 1);", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE str IN (SELECT x FROM test GROUP BY x);", dt),
                 std::runtime_error);
    ASSERT_NEAR(static_cast<double>(2.057),
                v<double>(run_simple_agg("SELECT AVG(dd) / (SELECT STDDEV(dd) FROM test) FROM test;", dt)),
                static_cast<double>(0.10));
  }
}

TEST(Select, Joins_Arrays) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test_inner WHERE test.x = ALL array_test_inner.arr_i16;", dt)));
    ASSERT_EQ(int64_t(60),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test_inner WHERE test.x = ANY array_test_inner.arr_i16;", dt)));
    ASSERT_EQ(int64_t(2 * g_array_test_row_count * g_num_rows - 60),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test_inner WHERE test.x <> ALL array_test_inner.arr_i16;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test_inner WHERE 7 = array_test_inner.arr_i16[1];", dt)));
    ASSERT_EQ(int64_t(2 * g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, array_test WHERE test.x = array_test.x AND 'bb' = ANY arr_str;", dt)));
    auto result_rows = run_multiple_agg(
        "SELECT UNNEST(array_test.arr_i16) AS a, test_inner.x, COUNT(*) FROM array_test, test_inner WHERE test_inner.x "
        "= array_test.arr_i16[1] GROUP BY a, test_inner.x;",
        dt);
    ASSERT_EQ(size_t(3), result_rows->rowCount());
    ASSERT_EQ(
        int64_t(g_array_test_row_count / 2 + g_array_test_row_count / 4),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test, test_inner WHERE EXTRACT(HOUR FROM test.m) = 22 AND test.x = test_inner.x;",
            dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test, test_inner WHERE array_test.arr_i32[array_test.x - 5] = 20 AND "
                  "array_test.x = "
                  "test_inner.x;",
                  dt)));
  }
}

TEST(Select, Joins_EmptyTable) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT test.x, emptytab.x FROM test, emptytab WHERE test.x = emptytab.x;", dt);
    c("SELECT COUNT(*) FROM test, emptytab GROUP BY test.x;", dt);
    c("SELECT COUNT(*) FROM test, emptytab, test_inner where test.x = emptytab.x;", dt);
    c("SELECT test.x, emptytab.x FROM test LEFT JOIN emptytab ON test.y = emptytab.y ORDER BY test.x ASC;", dt);
  }
}

TEST(Select, Joins_ImplicitJoins) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, hash_join_test WHERE test.t = hash_join_test.t;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x < test_inner.x + 1;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.real_str = test_inner.str;", dt);
    c("SELECT test_inner.x, COUNT(*) AS n FROM test, test_inner WHERE test.x = test_inner.x GROUP BY test_inner.x "
      "ORDER BY n;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str;", dt);
    c("SELECT test.str, COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str GROUP BY test.str;", dt);
    c("SELECT test_inner.str, COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str GROUP BY test_inner.str;",
      dt);
    c("SELECT test.str, COUNT(*) AS foobar FROM test, test_inner WHERE test.x = test_inner.x AND test.x > 6 GROUP BY "
      "test.str HAVING foobar > 5;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.real_str LIKE 'real_ba%' AND test.x = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE LENGTH(test.real_str) = 8 AND test.x = test_inner.x;", dt);
    c("SELECT a.x, b.str FROM test a, join_test b WHERE a.str = b.str GROUP BY a.x, b.str ORDER BY a.x, b.str;", dt);
    c("SELECT a.x, b.str FROM test a, join_test b WHERE a.str = b.str ORDER BY a.x, b.str;", dt);
    c("SELECT COUNT(1) FROM test a, join_test b, test_inner c WHERE a.str = b.str AND b.x = c.x", dt);
    c("SELECT COUNT(*) FROM test a, join_test b, test_inner c WHERE a.x = b.x AND a.y = b.x AND a.x = c.x AND c.str = "
      "'foo';",
      dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.x = b.x AND a.y = b.y;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.x = b.x AND a.str = b.str;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE (test.x = test_inner.x AND test.y = 42 AND test_inner.str = 'foo') "
      "OR (test.x = test_inner.x AND test.y = 43 AND test_inner.str = 'foo');",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.x OR test.x = test_inner.x;", dt);
    c("SELECT bar.str FROM test, bar WHERE test.str = bar.str;", dt);
    ASSERT_EQ(
        int64_t(3),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, join_test WHERE test.rowid = join_test.rowid;", dt)));
    ASSERT_EQ(7,
              v<int64_t>(run_simple_agg(
                  "SELECT test.x FROM test, test_inner WHERE test.x = test_inner.x AND test.rowid = 9;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.x AND test.rowid = 20;", dt)));
  }
}

TEST(Select, Joins_InnerJoin_TwoTables) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test a JOIN single_row_test b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) from test a JOIN single_row_test b ON a.ofd = b.x;", dt);
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.x = test_inner.x;", dt);
    c("SELECT a.y, z FROM test a JOIN test_inner b ON a.x = b.x order by a.y;", dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT COUNT(*) FROM test_inner_x a JOIN test_x b ON a.x = b.x;", dt);
    c("SELECT a.x FROM test a JOIN join_test b ON a.str = b.dup_str ORDER BY a.x;", dt);
    c("SELECT a.x FROM test_inner_x a JOIN test_x b ON a.x = b.x ORDER BY a.x;", dt);
    c("SELECT a.x FROM test a JOIN join_test b ON a.str = b.dup_str GROUP BY a.x ORDER BY a.x;", dt);
    c("SELECT a.x FROM test_inner_x a JOIN test_x b ON a.x = b.x GROUP BY a.x ORDER BY a.x;", dt);
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.x = test_inner.x AND test.rowid = test_inner.rowid;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.y = test_inner.y OR (test.y IS NULL AND test_inner.y IS NULL);",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.str = join_test.dup_str OR (test.str IS NULL AND "
      "join_test.dup_str IS NULL));",
      dt);
    c("SELECT t1.fixed_null_str FROM (SELECT fixed_null_str, SUM(x) n1 FROM test GROUP BY fixed_null_str) t1 INNER "
      "JOIN (SELECT fixed_null_str, SUM(y) n2 FROM test GROUP BY fixed_null_str) t2 ON ((t1.fixed_null_str = "
      "t2.fixed_null_str) OR (t1.fixed_null_str IS NULL AND t2.fixed_null_str IS NULL));",
      dt);
  }
}

TEST(Select, Joins_InnerJoin_AtLeastThreeTables) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
#ifdef ENABLE_JOIN_EXEC
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str;", dt);
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str JOIN "
      "join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT a.y, count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str "
      "GROUP BY a.y;",
      dt);
    c("SELECT a.x AS x, a.y, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = "
      "c.str "
      "ORDER BY a.y;",
      dt);
    c("SELECT a.x, b.x, b.str, c.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.x = c.x "
      "ORDER BY b.str;",
      dt);
    c("SELECT a.x, b.x, c.x FROM test a JOIN test_inner b ON a.x = b.x JOIN join_test c ON b.x = c.x;", dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str JOIN "
      "hash_join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str JOIN "
      "join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str JOIN "
      "hash_join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT a.x AS x, a.y, b.str FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str "
      "= c.str "
      "ORDER BY a.y;",
      dt);
    c("SELECT a.x, b.x, c.x FROM test a JOIN test_inner b ON a.x = b.x JOIN hash_join_test c ON b.x = c.x;", dt);
    c("SELECT a.x, b.x FROM test_inner a JOIN test_inner b ON a.x = b.x ORDER BY a.x;", dt);
    c("SELECT a.x, b.x FROM join_test a JOIN join_test b ON a.x = b.x ORDER BY a.x;", dt);
    c("SELECT COUNT(1) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON a.t = c.x;", dt);
    c("SELECT COUNT(*) FROM test a JOIN test_inner b ON a.str = b.str JOIN hash_join_test c ON a.x = c.x JOIN "
      "join_test d ON a.x > d.x;",
      dt);
    c("SELECT a.x, b.str, c.str, d.y FROM hash_join_test a JOIN test b ON a.x = b.x JOIN join_test c ON b.x = c.x JOIN "
      "test_inner d ON b.x = d.x ORDER BY a.x, b.str;",
      dt);
#endif
  }
}

TEST(Select, Joins_InnerJoin_Filters) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str WHERE a.y "
      "< 43;",
      dt);
    c("SELECT SUM(a.x), b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str "
      "WHERE a.y "
      "= 43 group by b.str;",
      dt);
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.str = test_inner.str AND test.x = 7;", dt);
    c("SELECT test.x, test_inner.str FROM test JOIN test_inner ON test.str = test_inner.str AND test.x <> 7;", dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = c.str "
      "WHERE a.y "
      "< 43;",
      dt);
    c("SELECT SUM(a.x), b.str FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = "
      "c.str "
      "WHERE a.y "
      "= 43 group by b.str;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.x = b.x JOIN test_inner c ON c.str = a.str WHERE c.str = "
      "'foo';",
      dt);
    c("SELECT COUNT(*) FROM test t1 JOIN test t2 ON t1.x = t2.x WHERE t1.y > t2.y;", dt);
    c("SELECT COUNT(*) FROM test t1 JOIN test t2 ON t1.x = t2.x WHERE t1.null_str = t2.null_str;", dt);
  }
}

TEST(Select, Joins_LeftOuterJoin) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x ORDER BY test.x ASC;",
      dt);
    c("SELECT test.x key1, CASE WHEN test_inner.x IS NULL THEN 99 ELSE test_inner.x END key2 FROM test LEFT OUTER JOIN "
      "test_inner ON test.x = test_inner.x GROUP BY key1, key2 ORDER BY key1;",
      dt);
    c("SELECT test_inner.x key1 FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x GROUP BY key1 HAVING "
      "key1 IS NOT NULL;",
      dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT JOIN test b ON a.x = b.x;", dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x ORDER BY a.x, b.str;", dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x ORDER BY a.x, b.str;", dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT OUTER JOIN test_x b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT COUNT(*) FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT a.x, b.str FROM test_inner a LEFT OUTER JOIN test_x b ON a.x = b.x ORDER BY a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str ORDER BY a.x, b.str IS NULL, "
      "b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str ORDER BY a.x, b.str IS NULL, "
      "b.str;",
      dt);
    c("SELECT COUNT(*) FROM test_inner_x a LEFT JOIN test_x b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) FROM test a LEFT JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT COUNT(*) FROM test a LEFT JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT a.x, b.str FROM test_inner_x a LEFT JOIN test_x b ON a.x = b.x ORDER BY a.x, b.str IS NULL, b.str;", dt);
    c("SELECT a.x, b.str FROM test a LEFT JOIN join_test b ON a.str = b.dup_str ORDER BY a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT JOIN join_test b ON a.str = b.dup_str ORDER BY a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x = test.x WHERE test_inner.str = test.str;", dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x < test.x WHERE test_inner.str = test.str;", dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x > test.x WHERE test_inner.str = test.str;", dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x >= test.x WHERE test_inner.str = test.str;", dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x <= test.x WHERE test_inner.str = test.str;", dt);
    c("SELECT test_inner.y, COUNT(*) n FROM test LEFT JOIN test_inner ON test_inner.x = test.x WHERE test_inner.str = "
      "'foo' GROUP BY test_inner.y ORDER BY n DESC;",
      dt);
    c("SELECT a.x, COUNT(b.y) FROM test a LEFT JOIN test_inner b ON b.x = a.x AND b.str NOT LIKE 'box' GROUP BY a.x "
      "ORDER BY a.x;",
      dt);
    c("SELECT a.x FROM test a LEFT OUTER JOIN test_inner b ON TRUE ORDER BY a.x ASC;",
      "SELECT a.x FROM test a LEFT OUTER JOIN test_inner b ON 1 ORDER BY a.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON test.x > test_inner.x INNER "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON test.x > test_inner.x INNER "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON test.x = test_inner.x INNER "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON test.x = test_inner.x INNER "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner ON test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner ON test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner ON test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner ON test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test.str = test_inner.str AND test.x = test_inner.x;", dt);
  }
}

TEST(Select, Joins_LeftJoin_Filters) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x WHERE test.y > 40 "
      "ORDER BY test.x ASC;",
      dt);
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x WHERE test.y > 42 "
      "ORDER BY test.x ASC;",
      dt);
    c("SELECT test.str AS foobar, test_inner.str FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x WHERE "
      "test.y > 42 ORDER BY foobar DESC LIMIT 8;",
      dt);
    c("SELECT test.x AS foobar, test_inner.x AS inner_foobar, test.f AS f_foobar FROM test LEFT OUTER JOIN test_inner "
      "ON test.str = test_inner.str WHERE test.y > 40 ORDER BY foobar DESC, f_foobar DESC;",
      dt);
    c("SELECT test.str AS foobar, test_inner.str FROM test LEFT OUTER JOIN test_inner ON test.x = test_inner.x WHERE "
      "test_inner.str IS NOT NULL ORDER BY foobar DESC;",
      dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT JOIN (SELECT * FROM test WHERE y > 40) b ON a.x = b.x;", dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN (SELECT * FROM test WHERE y > 40) b ON a.x = b.x ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT COUNT(*) FROM join_test a LEFT JOIN test b ON a.x = b.x AND a.x = 7;", dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x AND a.x = 7 ORDER BY a.x, b.str;", dt);
    c("SELECT COUNT(*) FROM join_test a LEFT JOIN test b ON a.x = b.x WHERE a.x = 7;", dt);
    c("SELECT a.x FROM join_test a LEFT JOIN test b ON a.x = b.x WHERE a.x = 7;", dt);
  }
}

TEST(Select, Joins_MultiCompositeColumns) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT a.x, b.str FROM test AS a JOIN join_test AS b ON a.str = b.str AND a.x = b.x ORDER BY a.x, b.str;", dt);
    c("SELECT a.x, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x AND a.str = b.str ORDER BY a.x, b.str;", dt);
    c("SELECT a.z, b.str FROM test a JOIN join_test b ON a.y = b.y AND a.x = b.x ORDER BY a.z, b.str;", dt);
    c("SELECT a.z, b.str FROM test a JOIN test_inner b ON a.y = b.y AND a.x = b.x ORDER BY a.z, b.str;", dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.x = b.x AND a.y = b.x JOIN test_inner c ON a.x = c.x WHERE "
      "c.str <> 'foo';",
      dt);
    c("SELECT a.x, b.x, d.str FROM test a JOIN test_inner b ON a.str = b.str JOIN hash_join_test c ON a.x = c.x JOIN "
      "join_test d ON a.x >= d.x AND a.x < d.x + 5 ORDER BY a.x, b.x;",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.x = join_test.x OR (test.x IS NULL AND join_test.x IS NULL)) "
      "AND (test.y = join_test.y OR (test.y IS NULL AND join_test.y IS NULL));",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.str = join_test.dup_str OR (test.str IS NULL AND "
      "join_test.dup_str IS NULL)) AND (test.x = join_test.x OR (test.x IS NULL AND join_test.x IS NULL));",
      dt);
  }
}

TEST(Select, Joins_BuildHashTable) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, join_test WHERE test.str = join_test.dup_str;", dt);
    // Intentionally duplicate previous string join to cover hash table building.
    c("SELECT COUNT(*) FROM test, join_test WHERE test.str = join_test.dup_str;", dt);
  }
}

TEST(Select, Joins_ComplexQueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test a JOIN (SELECT * FROM test WHERE y < 43) b ON a.x = b.x JOIN join_test c ON a.x = c.x "
      "WHERE a.fixed_str = 'foo';",
      dt);
    c("SELECT * FROM (SELECT a.y, b.str FROM test a JOIN join_test b ON a.x = b.x) ORDER BY y, str;", dt);
    c("SELECT x, dup_str FROM (SELECT * FROM test a JOIN join_test b ON a.x = b.x) WHERE y > 40 ORDER BY x, dup_str;",
      dt);
    c("SELECT a.x FROM (SELECT * FROM test WHERE x = 8) AS a JOIN (SELECT * FROM test_inner WHERE x = 7) AS b ON a.str "
      "= b.str WHERE a.y < 42;",
      dt);
    c("SELECT a.str as key0,a.fixed_str as key1,COUNT(*) AS color FROM test a JOIN (select str,count(*) "
      "from test group by str order by COUNT(*) desc limit 40) b on a.str=b.str JOIN (select "
      "fixed_str,count(*) from test group by fixed_str order by count(*) desc limit 40) c on "
      "c.fixed_str=a.fixed_str GROUP BY key0, key1 ORDER BY key0,key1;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN (SELECT str FROM test) b ON a.str = b.str OR false;",
      "SELECT COUNT(*) FROM test a JOIN (SELECT str FROM test) b ON a.str = b.str OR 0;",
      dt);
  }
}

TEST(Select, Joins_TimeAndDate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test a, test b WHERE a.m = b.m;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.n = b.n;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o;", dt);
  }
}

TEST(Select, Joins_OneOuterExpression) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x - 1 = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test_inner, test WHERE test.x - 1 = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 0 = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test_inner, test WHERE test.x + 0 = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 1 = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test_inner, test WHERE test.x + 1 = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o + INTERVAL '0' DAY = b.o;",
      "SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o;",
      dt);
    c("SELECT COUNT(*) FROM test b, test a WHERE a.o + INTERVAL '0' DAY = b.o;",
      "SELECT COUNT(*) FROM test b, test a WHERE a.o = b.o;",
      dt);
  }
}

TEST(Select, Joins_MultipleOuterExpressions) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x - 1 = test_inner.x AND test.str = test_inner.str;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 0 = test_inner.x AND test.str = test_inner.str;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str AND test.x + 0 = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 1 = test_inner.x AND test.str = test_inner.str;", dt);
    // The following query will fallback to loop join because we don't reorder the
    // expressions to be consistent with table order for composite equality yet.
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 0 = test_inner.x AND test_inner.str = test.str;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o + INTERVAL '0' DAY = b.o AND a.str = b.str;",
      "SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o AND a.str = b.str;",
      dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o + INTERVAL '0' DAY = b.o AND a.x = b.x;",
      "SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o AND a.x = b.x;",
      dt);
  }
}

TEST(Select, Joins_Unsupported) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
#ifndef ENABLE_JOIN_EXEC
    EXPECT_THROW(
        run_multiple_agg("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON "
                         "b.str = c.str;",
                         dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = "
                     "c.str JOIN join_test AS d ON c.x = d.x;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT a.x AS x, y, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c "
                     "ON b.str = c.str ORDER BY x;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON b.str = "
                     "c.str WHERE a.y "
                     "< 43;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT SUM(a.x), b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner AS c ON "
                     "b.str = c.str WHERE a.y "
                     "= 43 group by b.str;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT a.str as key0,a.fixed_str as key1,COUNT(*) AS color FROM test a JOIN (select str,count(*) "
                     "from test group by str order by COUNT(*) desc limit 40) b on a.str=b.str JOIN (select "
                     "fixed_str,count(*) from test group by fixed_str order by count(*) desc limit 40) c on "
                     "c.fixed_str=a.fixed_str GROUP BY key0, key1 ORDER BY key0,key1;",
                     dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT x, tnone FROM test LEFT JOIN text_group_by_test ON test.str = text_group_by_test.tdef;", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT * FROM test a JOIN test b on a.b = b.x;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT * FROM test a JOIN test b on a.f = b.f;", dt), std::runtime_error);
#endif
  }
}

TEST(Select, RuntimeFunctions) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT SUM(ABS(-x + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-y + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-z + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-t + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-dd + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-f + 1)) FROM test;", dt);
    c("SELECT SUM(ABS(-d + 1)) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE ABS(CAST(x AS float)) >= 0;", dt);
    c("SELECT MIN(ABS(-ofd + 2)) FROM test;", dt);
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-dd) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(x - 7) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(x - 7) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(x - 8) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(x - 8) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(y - 42) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(y - 42) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(y - 43) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(y - 43) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-f) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-d) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(ofd) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-ofd) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(ofd) IS NULL;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg("SELECT SUM(SIN(x) * SIN(x) + COS(x) * COS(x)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg("SELECT SUM(SIN(f) * SIN(f) + COS(f) * COS(f)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg("SELECT SUM(SIN(d) * SIN(d) + COS(d) * COS(d)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg("SELECT SUM(SIN(dd) * SIN(dd) + COS(dd) * COS(dd)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2),
                    v<double>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS double)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(2),
                    v<float>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS float)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS BIGINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS SMALLINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT FLOOR(CAST(2.3 AS INT)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2), v<double>(run_simple_agg("SELECT FLOOR(2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2), v<double>(run_simple_agg("SELECT FLOOR(2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(-3), v<double>(run_simple_agg("SELECT FLOOR(-2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(-2), v<double>(run_simple_agg("SELECT FLOOR(-2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(3),
                    v<double>(run_simple_agg("SELECT CEIL(CAST(2.3 AS double)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(3),
                    v<float>(run_simple_agg("SELECT CEIL(CAST(2.3 AS float)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT CEIL(CAST(2.3 AS BIGINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT CEIL(CAST(2.3 AS SMALLINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg("SELECT CEIL(CAST(2.3 AS INT)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(3), v<double>(run_simple_agg("SELECT CEIL(2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2), v<double>(run_simple_agg("SELECT CEIL(2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(-2), v<double>(run_simple_agg("SELECT CEIL(-2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(-2), v<double>(run_simple_agg("SELECT CEIL(-2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<float>(4129511.320307),
        v<double>(run_simple_agg(
            "SELECT DISTANCE_IN_METERS(-74.0059, 40.7217,-122.416667 , 37.783333) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<int64_t>(1000),
                    v<int64_t>(run_simple_agg("SELECT TRUNCATE(CAST(1171 AS SMALLINT),-3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(1000),
                    v<float>(run_simple_agg("SELECT TRUNCATE(CAST(1171.123 AS FLOAT),-3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(1000),
                    v<double>(run_simple_agg("SELECT TRUNCATE(CAST(1171.123 AS DOUBLE),-3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(1171.10),
                    v<double>(run_simple_agg("SELECT TRUNCATE(CAST(1171.123 AS DOUBLE),1) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(1171.11),
                    v<float>(run_simple_agg("SELECT TRUNCATE(CAST(1171.113 AS FLOAT),2) FROM test LIMIT 1;", dt)));
  }
}

TEST(Select, TextGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg(
                     " select count(*) from (SELECT tnone, count(*) cc from text_group_by_test group by tnone);", dt),
                 std::runtime_error);
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "select count(*) from (SELECT tdict, count(*) cc from text_group_by_test group by tdict)", dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "select count(*) from (SELECT tdef, count(*) cc from text_group_by_test group by tdef)", dt)));
  }
}

TEST(Select, UnsupportedExtensions) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT TRUNCATE(2016, 1) FROM test LIMIT 1;", dt), std::runtime_error);
  }
}

TEST(Select, UnsupportedSortOfIntermediateResult) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT 'foo' FROM test ORDER BY x;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT real_str FROM test ORDER BY x;", dt), std::runtime_error);
  }
}

TEST(Select, Views) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x, COUNT(*) FROM view_test WHERE y > 41 GROUP BY x;", dt);
    c("SELECT x FROM join_view_test WHERE x IS NULL;", dt);
  }
}

TEST(Select, CreateTableAsSelect) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT fixed_str, COUNT(*) FROM ctas_test GROUP BY fixed_str;", dt);
    c("SELECT x, COUNT(*) FROM ctas_test GROUP BY x;", dt);
    c("SELECT f, COUNT(*) FROM ctas_test GROUP BY f;", dt);
    c("SELECT d, COUNT(*) FROM ctas_test GROUP BY d;", dt);
    c("SELECT COUNT(*) FROM empty_ctas_test;", dt);
  }
}

TEST(Select, PgShim) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, SUM(x), COUNT(str) FROM test WHERE \"y\" = 42 AND str = 'Shim All The Things!' GROUP BY str;", dt);
  }
}

TEST(Select, CaseInsensitive) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT X, COUNT(*) AS N FROM test GROUP BY teSt.x ORDER BY n DESC;", dt);
  }
}

TEST(Select, Deserialization) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(CAST(x AS float) * 0.0000000000 AS INT) FROM test;", dt);
  }
}

TEST(Select, DesugarTransform) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT * FROM emptytab ORDER BY emptytab. x;", dt);
    c("SELECT COUNT(*) FROM TEST WHERE x IN (SELECT x + 1 AS foo FROM test GROUP BY foo ORDER BY COUNT(*) DESC LIMIT "
      "1);",
      dt);
  }
}

TEST(Select, ArrowOutput) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c_arrow("SELECT str, COUNT(*) FROM test GROUP BY str ORDER BY str ASC;", dt);
    c_arrow("SELECT x, y, z, t, f, d, str, ofd, ofq FROM test ORDER BY x ASC, y ASC;", dt);
    c_arrow("SELECT null_str, COUNT(*) FROM test GROUP BY null_str;", dt);
  }
}

TEST(Select, WatchdogTest) {
  g_enable_watchdog = true;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x, SUM(f) AS n FROM test GROUP BY x ORDER BY n DESC LIMIT 5;", dt);
    c("SELECT COUNT(*) FROM test WHERE str = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz';", dt);
  }
}

TEST(Truncate, Count) {
  run_ddl_statement("create table trunc_test (i1 integer, t1 text);");
  run_multiple_agg("insert into trunc_test values(1, '1');", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into trunc_test values(2, '2');", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3), v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM trunc_test;", ExecutorDeviceType::CPU)));
  run_ddl_statement("truncate table trunc_test;");
  run_multiple_agg("insert into trunc_test values(3, '3');", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into trunc_test values(4, '4');", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(7), v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM trunc_test;", ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table trunc_test;");
}

TEST(Select, Deleted) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test_inner_deleted;", dt);
    c("SELECT test.x, test_inner_deleted.x FROM test LEFT JOIN test_inner_deleted ON test.x <> test_inner_deleted.x "
      "ORDER BY test.x ASC;",
      dt);
    c("SELECT test.x, test_inner_deleted.x FROM test JOIN test_inner_deleted ON test.x <> test_inner_deleted.x ORDER "
      "BY test.x ASC;",
      dt);
    c("SELECT test.x, test_inner_deleted.x FROM test LEFT JOIN test_inner_deleted ON test.x = test_inner_deleted.x "
      "ORDER BY test.x ASC;",
      dt);
    c("SELECT test.x, test_inner_deleted.x FROM test JOIN test_inner_deleted ON test.x = test_inner_deleted.x ORDER BY "
      "test.x ASC;",
      dt);
  }
}

namespace {

int create_and_populate_tables() {
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_inner;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{"x int not null, y int, str text encoding dict"};
    const auto create_test_inner = build_create_table_statement(
        columns_definition, "test_inner", {g_shard_count ? "str" : "", g_shard_count}, {}, 2);
    run_ddl_statement(create_test_inner);
    g_sqlite_comparator.query("CREATE TABLE test_inner(x int not null, y int, str text);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_inner'";
    return -EEXIST;
  }
  {
    const std::string insert_query{"INSERT INTO test_inner VALUES(7, 43, 'foo');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_test_inner_deleted{"DROP TABLE IF EXISTS test_inner_deleted;"};
    run_ddl_statement(drop_old_test_inner_deleted);
    g_sqlite_comparator.query(drop_old_test_inner_deleted);
    std::string columns_definition{"x int not null, y int, str text encoding dict, deleted boolean"};
    const auto create_test_inner_deleted =
        build_create_table_statement(columns_definition, "test_inner_deleted", {"", 0}, {}, 2);
    run_ddl_statement(create_test_inner_deleted);
    auto& cat = g_session->get_catalog();
    const auto td = cat.getMetadataForTable("test_inner_deleted");
    CHECK(td);
    const auto cd = cat.getMetadataForColumn(td->tableId, "deleted");
    CHECK(cd);
    cat.setDeletedColumn(td, cd);
    g_sqlite_comparator.query("CREATE TABLE test_inner_deleted(x int not null, y int, str text);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_inner_deleted'";
    return -EEXIST;
  }
  {
    const std::string insert_query{"INSERT INTO test_inner_deleted VALUES(7, 43, 'foo', 't');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_inner_x;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{"x int not null, y int, str text encoding dict"};
    const auto create_test_inner = build_create_table_statement(
        columns_definition, "test_inner_x", {g_shard_count ? "x" : "", g_shard_count}, {}, 2);
    run_ddl_statement(create_test_inner);
    g_sqlite_comparator.query("CREATE TABLE test_inner_x(x int not null, y int, str text);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_inner_x'";
    return -EEXIST;
  }
  {
    const std::string insert_query{"INSERT INTO test_inner_x VALUES(7, 43, 'foo');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_bar{"DROP TABLE IF EXISTS bar;"};
    run_ddl_statement(drop_old_bar);
    g_sqlite_comparator.query(drop_old_bar);
    std::string columns_definition{"str text encoding dict"};
    const auto create_bar = build_create_table_statement(columns_definition, "bar", {"", 0}, {}, 2);
    run_ddl_statement(create_bar);
    g_sqlite_comparator.query("CREATE TABLE bar(str text);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'bar'";
    return -EEXIST;
  }
  {
    const std::string insert_query{"INSERT INTO bar VALUES('bar');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{
        "x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn float, d double, dn double, str "
        "varchar(10), null_str text encoding dict, fixed_str text encoding dict(16), fixed_null_str text encoding "
        "dict(16), real_str text encoding none, shared_dict text, m timestamp(0), n time(0), o date, o1 date encoding "
        "fixed(32), fx int "
        "encoding fixed(16), dd decimal(10, 2), dd_notnull decimal(10, 2) not null, ss text encoding dict, u int, ofd "
        "int, ufd int not null, ofq bigint, ufq bigint not null"};
    const std::string create_test =
        build_create_table_statement(columns_definition,
                                     "test",
                                     {g_shard_count ? "str" : "", g_shard_count},
                                     {{"str", "test_inner", "str"}, {"shared_dict", "test", "str"}},
                                     2);
    run_ddl_statement(create_test);
    g_sqlite_comparator.query(
        "CREATE TABLE test(x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), n time(0), o date, o1 date, fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test'";
    return -EEXIST;
  }
  CHECK_EQ(g_num_rows % 2, 0);
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, 'foo', null, 'foo', null, "
        "'real_foo', 'foo',"
        "'2014-12-13 "
        "22:23:15', "
        "'15:13:14', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, 'fish', null, 2147483647, -2147483648, null, -1);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(8, 43, -78, 1002, 'f', 1.2, 101.2, -101.2, 2.4, -2002.4, 'bar', null, 'bar', null, "
        "'real_bar', NULL, '2014-12-13 22:23:15', '15:13:14', NULL, NULL, NULL, 222.2, 222.2, null, null, null, "
        "-2147483647, "
        "9223372036854775807, -9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 43, 102, 1002, 't', 1.3, 1000.3, -1000.3, 2.6, -220.6, 'baz', null, null, null, "
        "'real_baz', 'baz', '2014-12-14 22:23:15', '15:13:14', '1999-09-09', '1999-09-09', 11, 333.3, 333.3, "
        "'boat', null, 1, "
        "-1, 1, -9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_x;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{
        "x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn float, d double, dn double, str "
        "text, null_str text encoding dict, fixed_str text encoding dict(16), real_str text encoding none, m "
        "timestamp(0), n time(0), o date, o1 date encoding fixed(32), fx int encoding fixed(16), dd decimal(10, 2), "
        "dd_notnull decimal(10, 2) not null, ss text encoding dict, u int, ofd int, ufd int not null, ofq bigint, ufq "
        "bigint not null"};
    const std::string create_test =
        build_create_table_statement(columns_definition, "test_x", {g_shard_count ? "x" : "", g_shard_count}, {}, 2);
    run_ddl_statement(create_test);
    g_sqlite_comparator.query(
        "CREATE TABLE test_x(x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn float, d "
        "double, dn double, str "
        "text, null_str text,"
        "fixed_str text, real_str text, m timestamp(0), n time(0), o date, o1 date, fx int, dd decimal(10, 2), "
        "dd_notnull decimal(10, 2) not null, ss text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not "
        "null);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_x'";
    return -EEXIST;
  }
  CHECK_EQ(g_num_rows % 2, 0);
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    const std::string insert_query{
        "INSERT INTO test_x VALUES(7, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, 'foo', null, 'foo', 'real_foo', "
        "'2014-12-13 "
        "22:23:15', "
        "'15:13:14', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, 'fish', null, 2147483647, -2147483648, null, -1);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test_x VALUES(8, 43, 102, 1002, 'f', 1.2, 101.2, -101.2, 2.4, -2002.4, 'bar', null, 'bar', "
        "'real_bar', "
        "'2014-12-13 "
        "22:23:15', "
        "'15:13:14', NULL, NULL, NULL, 222.2, 222.2, null, null, null, -2147483647, 9223372036854775807, "
        "-9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test_x VALUES(7, 43, 102, 1002, 't', 1.3, 1000.3, -1000.3, 2.6, -220.6, 'baz', null, 'baz', "
        "'real_baz', "
        "'2014-12-13 "
        "22:23:15', "
        "'15:13:14', '1999-09-09', '1999-09-09', 11, 333.3, 333.3, 'boat', null, 1, -1, 1, -9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_array_test{"DROP TABLE IF EXISTS array_test;"};
    run_ddl_statement(drop_old_array_test);
    const std::string create_array_test{
        "CREATE TABLE array_test(x int not null, arr_i16 smallint[], arr_i32 int[], arr_i64 bigint[], arr_str text[] "
        "encoding dict, arr_float float[], arr_double double[], arr_bool boolean[], real_str text encoding none);"};
    run_ddl_statement(create_array_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'array_test'";
    return -EEXIST;
  }
  import_array_test("array_test");
  try {
    const std::string drop_old_array_test{"DROP TABLE IF EXISTS array_test_inner;"};
    run_ddl_statement(drop_old_array_test);
    const std::string create_array_test{
        "CREATE TABLE array_test_inner(x int, arr_i16 smallint[], arr_i32 int[], arr_i64 bigint[], arr_str text[] "
        "encoding "
        "dict, "
        "arr_float float[], arr_double double[], arr_bool boolean[], real_str text encoding none) WITH "
        "(fragment_size=4000000, partitions='REPLICATED');"};
    run_ddl_statement(create_array_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'array_test_inner'";
    return -EEXIST;
  }
  import_array_test("array_test_inner");
  try {
    const std::string drop_old_single_row_test{"DROP TABLE IF EXISTS single_row_test;"};
    run_ddl_statement(drop_old_single_row_test);
    g_sqlite_comparator.query(drop_old_single_row_test);
    const std::string create_single_row_test{"CREATE TABLE single_row_test(x int);"};
    run_ddl_statement(create_single_row_test);
    g_sqlite_comparator.query(create_single_row_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'single_row_test'";
    return -EEXIST;
  }
  {
    const std::string insert_query{"INSERT INTO single_row_test VALUES(null);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    import_gpu_sort_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'gpu_sort_test'";
    return -EEXIST;
  }
  try {
    import_query_rewrite_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'query_rewrite_test'";
    return -EEXIST;
  }
  try {
    import_big_decimal_range_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'big_decimal_range_test'";
    return -EEXIST;
  }
  try {
    import_subquery_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'subquery_test'";
    return -EEXIST;
  }
  try {
    import_text_group_by_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'text_group_by_test'";
    return -EEXIST;
  }
  try {
    import_join_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'join_test'";
    return -EEXIST;
  }
  try {
    import_hash_join_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'hash_join_test'";
    return -EEXIST;
  }
  try {
    import_emp_table();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'emp'";
    return -EEXIST;
  }
  try {
    import_dept_table();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'dept'";
    return -EEXIST;
  }
  try {
    const std::string drop_old_empty{"DROP TABLE IF EXISTS emptytab;"};
    run_ddl_statement(drop_old_empty);
    g_sqlite_comparator.query(drop_old_empty);
    const std::string create_empty{
        "CREATE TABLE emptytab(x int not null, y int, t bigint not null, f float not null, d double not null, dd "
        "decimal(10, 2) not null, ts timestamp)"};
    run_ddl_statement(create_empty + " WITH (partitions='REPLICATED');");
    g_sqlite_comparator.query(create_empty + ";");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'emptytab'";
    return -EEXIST;
  }
  try {
    const std::string drop_old_test_in_bitmap{"DROP TABLE IF EXISTS test_in_bitmap;"};
    run_ddl_statement(drop_old_test_in_bitmap);
    g_sqlite_comparator.query(drop_old_test_in_bitmap);
    const std::string create_test_in_bitmap{"CREATE TABLE test_in_bitmap(str TEXT ENCODING DICT);"};
    run_ddl_statement(create_test_in_bitmap);
    g_sqlite_comparator.query("CREATE TABLE test_in_bitmap(str TEXT);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_in_bitmap'";
    return -EEXIST;
  }
  {
    std::string insert_query{"INSERT INTO test_in_bitmap VALUES('a');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    std::string insert_query{"INSERT INTO test_in_bitmap VALUES('b');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    std::string insert_query{"INSERT INTO test_in_bitmap VALUES('c');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    std::string insert_query{"INSERT INTO test_in_bitmap VALUES(NULL);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  return 0;
}

int create_views() {
  const std::string create_view_test{
      "CREATE VIEW view_test AS SELECT test.*, test_inner.* FROM test, test_inner WHERE test.str = test_inner.str;"};
  const std::string drop_old_view{"DROP VIEW IF EXISTS view_test;"};
  const std::string create_join_view_test{
      "CREATE VIEW join_view_test AS SELECT a.x AS x FROM test a JOIN test_inner b ON a.str = b.str;"};
  const std::string drop_old_join_view{"DROP VIEW IF EXISTS join_view_test;"};
  try {
    run_ddl_statement(drop_old_view);
    run_ddl_statement(create_view_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create view 'view_test' -- run_ddl_statement";
    return -EEXIST;
  }
  try {
    g_sqlite_comparator.query(drop_old_view);
    g_sqlite_comparator.query(create_view_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create view 'view_test' -- g_sqlite_comparator";
    return -EEXIST;
  }
  try {
    run_ddl_statement(drop_old_join_view);
    run_ddl_statement(create_join_view_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create view 'join_view_test' -- run_ddl_statement";
    return -EEXIST;
  }
  try {
    g_sqlite_comparator.query(drop_old_join_view);
    g_sqlite_comparator.query(create_join_view_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create view 'join_view_test' -- g_sqlite_comparator";
    return -EEXIST;
  }
  return 0;
}

int create_as_select() {
  try {
    const std::string drop_ctas_test{"DROP TABLE IF EXISTS ctas_test;"};
    run_ddl_statement(drop_ctas_test);
    g_sqlite_comparator.query(drop_ctas_test);
    const std::string create_ctas_test{
        "CREATE TABLE ctas_test AS SELECT x, f, d, str, fixed_str FROM test WHERE x > 7;"};
    run_ddl_statement(create_ctas_test);
    g_sqlite_comparator.query(create_ctas_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'ctas_test'";
    return -EEXIST;
  }
  return 0;
}

int create_as_select_empty() {
  try {
    const std::string drop_ctas_test{"DROP TABLE IF EXISTS empty_ctas_test;"};
    run_ddl_statement(drop_ctas_test);
    g_sqlite_comparator.query(drop_ctas_test);
    const std::string create_ctas_test{
        "CREATE TABLE empty_ctas_test AS SELECT x, f, d, str, fixed_str FROM test WHERE x > 8;"};
    run_ddl_statement(create_ctas_test);
    g_sqlite_comparator.query(create_ctas_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'empty_ctas_test'";
    return -EEXIST;
  }
  return 0;
}

void drop_tables() {
  const std::string drop_test_inner{"DROP TABLE test_inner;"};
  run_ddl_statement(drop_test_inner);
  g_sqlite_comparator.query(drop_test_inner);
  const std::string drop_test{"DROP TABLE test;"};
  run_ddl_statement(drop_test);
  g_sqlite_comparator.query(drop_test);
  const std::string drop_test_inner_x{"DROP TABLE test_inner_x;"};
  run_ddl_statement(drop_test_inner_x);
  g_sqlite_comparator.query(drop_test_inner_x);
  const std::string drop_test_inner_deleted{"DROP TABLE test_inner_deleted;"};
  run_ddl_statement(drop_test_inner_deleted);
  g_sqlite_comparator.query(drop_test_inner_deleted);
  const std::string drop_bar{"DROP TABLE bar;"};
  run_ddl_statement(drop_bar);
  g_sqlite_comparator.query(drop_bar);
  const std::string drop_test_x{"DROP TABLE test_x;"};
  run_ddl_statement(drop_test_x);
  g_sqlite_comparator.query(drop_test_x);
  const std::string drop_gpu_sort_test{"DROP TABLE gpu_sort_test;"};
  run_ddl_statement(drop_gpu_sort_test);
  g_sqlite_comparator.query(drop_gpu_sort_test);
  const std::string drop_query_rewrite_test{"DROP TABLE query_rewrite_test;"};
  run_ddl_statement(drop_query_rewrite_test);
  const std::string drop_big_decimal_range_test{"DROP TABLE big_decimal_range_test;"};
  run_ddl_statement(drop_big_decimal_range_test);
  g_sqlite_comparator.query(drop_query_rewrite_test);
  const std::string drop_array_test{"DROP TABLE array_test;"};
  run_ddl_statement(drop_array_test);
  const std::string drop_array_test_inner{"DROP TABLE array_test_inner;"};
  run_ddl_statement(drop_array_test_inner);
  const std::string drop_single_row_test{"DROP TABLE single_row_test;"};
  g_sqlite_comparator.query(drop_single_row_test);
  run_ddl_statement(drop_single_row_test);
  const std::string drop_subquery_test{"DROP TABLE subquery_test;"};
  run_ddl_statement(drop_subquery_test);
  g_sqlite_comparator.query(drop_subquery_test);
  const std::string drop_empty_test{"DROP TABLE emptytab;"};
  run_ddl_statement(drop_empty_test);
  g_sqlite_comparator.query(drop_empty_test);
  run_ddl_statement("DROP TABLE text_group_by_test;");
  const std::string drop_join_test{"DROP TABLE join_test;"};
  g_sqlite_comparator.query(drop_join_test);
  const std::string drop_hash_join_test{"DROP TABLE hash_join_test;"};
  run_ddl_statement(drop_hash_join_test);
  g_sqlite_comparator.query(drop_hash_join_test);
  run_ddl_statement(drop_join_test);
  const std::string drop_emp_table{"DROP TABLE emp;"};
  g_sqlite_comparator.query(drop_emp_table);
  run_ddl_statement(drop_emp_table);
  const std::string drop_dept_table{"DROP TABLE dept;"};
  g_sqlite_comparator.query(drop_dept_table);
  run_ddl_statement(drop_dept_table);
  const std::string drop_test_in_bitmap{"DROP TABLE test_in_bitmap;"};
  g_sqlite_comparator.query(drop_test_in_bitmap);
  run_ddl_statement(drop_test_in_bitmap);
  const std::string drop_ctas_test{"DROP TABLE ctas_test;"};
  g_sqlite_comparator.query(drop_ctas_test);
  run_ddl_statement(drop_ctas_test);
  const std::string drop_empty_ctas_test{"DROP TABLE empty_ctas_test;"};
  g_sqlite_comparator.query(drop_empty_ctas_test);
  run_ddl_statement(drop_empty_ctas_test);
}

void drop_views() {
  const std::string drop_view_test{"DROP VIEW view_test;"};
  run_ddl_statement(drop_view_test);
  g_sqlite_comparator.query(drop_view_test);
  const std::string drop_join_view_test{"DROP VIEW join_view_test;"};
  run_ddl_statement(drop_join_view_test);
  g_sqlite_comparator.query(drop_join_view_test);
}

size_t choose_shard_count() {
  CHECK(g_session);
  const auto cuda_mgr = g_session->get_catalog().get_dataMgr().cudaMgr_;
  const int device_count = cuda_mgr ? cuda_mgr->getDeviceCount() : 0;
  return device_count > 1 ? device_count : 0;
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << " after initialization";
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("disable-literal-hoisting", "Disable literal hoisting");
  desc.add_options()("with-sharding", "Create sharded tables");
  desc.add_options()("use-result-set",
                     po::value<bool>(&g_use_result_set)->default_value(g_use_result_set)->implicit_value(true),
                     "Use the new result set");
  desc.add_options()("left-deep-join-optimization",
                     po::value<bool>(&g_left_deep_join_optimization)
                         ->default_value(g_left_deep_join_optimization)
                         ->implicit_value(true),
                     "Enable left-deep join optimization");
  desc.add_options()(
      "from-table-reordering",
      po::value<bool>(&g_from_table_reordering)->default_value(g_from_table_reordering)->implicit_value(true),
      "Enable automatic table reordering in FROM clause");
  desc.add_options()("bigint-count",
                     po::value<bool>(&g_bigint_count)->default_value(g_bigint_count)->implicit_value(false),
                     "Use 64-bit count");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");
  desc.add_options()("use-existing-data",
                     "Don't create and drop tables and only run select tests (it implies --keep-data).");
  desc.add_options()("disable-fast-strcmp",
                     po::value<bool>(&g_fast_strcmp)->default_value(g_fast_strcmp)->implicit_value(false),
                     "Disable fast string comparison");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("disable-literal-hoisting"))
    g_hoist_literals = false;

  g_session.reset(get_session(BASE_PATH));

  if (vm.count("with-sharding"))
    g_shard_count = choose_shard_count();

  const bool use_existing_data = vm.count("use-existing-data");
  int err{0};
  if (use_existing_data) {
    testing::GTEST_FLAG(filter) = "Select*";
  } else {
    err = create_and_populate_tables();
    if (!err) {
      err = create_views();
    }
    if (!err) {
      err = create_as_select();
    }
    if (!err) {
      err = create_as_select_empty();
    }
  }
  if (err) {
    return err;
  }

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  Executor::nukeCacheOfExecutors();
  if (!use_existing_data && !vm.count("keep-data")) {
    drop_tables();
    drop_views();
  }
  g_session.reset(nullptr);
  return err;
}
