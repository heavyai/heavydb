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
#include "../Parser/parser.h"
#include "../QueryEngine/ArrowResultSet.h"
#include "../QueryEngine/Execute.h"
#include "../QueryEngine/RelAlgExecutionDescriptor.h"
#include "../QueryRunner/QueryRunner.h"
#include "../Shared/ConfigResolve.h"
#include "../Shared/TimeGM.h"
#include "../Shared/scope.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "DistributedLoader.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/any.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <sstream>

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std;
using namespace TestHelpers;

bool g_aggregator{false};

extern int g_test_against_columnId_gap;
extern bool g_enable_smem_group_by;
extern bool g_allow_cpu_retry;
extern bool g_enable_watchdog;

extern unsigned g_trivial_loop_join_threshold;
extern bool g_enable_overlaps_hashjoin;
extern double g_gpu_mem_limit_percent;

namespace {

std::unique_ptr<Catalog_Namespace::SessionInfo> g_session;
std::unique_ptr<QueryRunner::IROutputFile> g_ir_output_file;
bool g_hoist_literals{true};
size_t g_shard_count{0};
bool g_use_row_iterator{true};
size_t g_num_leafs{1};
bool g_keep_test_data{false};

size_t choose_shard_count() {
  CHECK(g_session);
  const auto cuda_mgr = g_session->getCatalog().getDataMgr().getCudaMgr();
  const int device_count = cuda_mgr ? cuda_mgr->getDeviceCount() : 0;
  return g_num_leafs * (device_count > 1 ? device_count : 0);
}

struct ShardInfo {
  const std::string shard_col;
  const size_t shard_count;
};

struct SharedDictionaryInfo {
  const std::string col;
  const std::string ref_table;
  const std::string ref_col;
};

std::string build_create_table_statement(
    const std::string& columns_definition,
    const std::string& table_name,
    const ShardInfo& shard_info,
    const std::vector<SharedDictionaryInfo>& shared_dict_info,
    const size_t fragment_size,
    const bool delete_support = true,
    const bool replicated = false) {
  const std::string shard_key_def{
      shard_info.shard_col.empty() ? "" : ", SHARD KEY (" + shard_info.shard_col + ")"};

  std::vector<std::string> shared_dict_def;
  if (shared_dict_info.size() > 0) {
    for (size_t idx = 0; idx < shared_dict_info.size(); ++idx) {
      shared_dict_def.push_back(", SHARED DICTIONARY (" + shared_dict_info[idx].col +
                                ") REFERENCES " + shared_dict_info[idx].ref_table + "(" +
                                shared_dict_info[idx].ref_col + ")");
    }
  }

  std::ostringstream with_statement_assembly;
  if (!shard_info.shard_col.empty()) {
    with_statement_assembly << "shard_count=" << shard_info.shard_count << ", ";
  }
  with_statement_assembly << "fragment_size=" << fragment_size;

  if (delete_support) {
    with_statement_assembly << ", vacuum='delayed'";
  }

  // const std::string fragment_size_def{shard_info.shard_col.empty() ? "fragment_size=" +
  // std::to_string(fragment_size)
  //                                                                 : ""};
  // const std::string shard_count_def{shard_info.shard_col.empty() ? "" : "shard_count="
  // +
  //                                                                          std::to_string(shard_info.shard_count)};
  const std::string replicated_def{!replicated ? "" : ", PARTITIONS='REPLICATED' "};

  return "CREATE TABLE " + table_name + "(" + columns_definition + shard_key_def +
         boost::algorithm::join(shared_dict_def, "") + ") WITH (" +
         with_statement_assembly.str() + replicated_def + ");";
}

std::shared_ptr<ResultSet> run_multiple_agg(const string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool allow_loop_joins) {
  return QueryRunner::run_multiple_agg(query_str,
                                       g_session,
                                       device_type,
                                       g_hoist_literals,
                                       allow_loop_joins,
                                       g_ir_output_file);
}

std::shared_ptr<ResultSet> run_multiple_agg(const string& query_str,
                                            const ExecutorDeviceType device_type) {
  return run_multiple_agg(query_str, device_type, true);
}

TargetValue run_simple_agg(const string& query_str,
                           const ExecutorDeviceType device_type,
                           const bool geo_return_geo_tv = true,
                           const bool allow_loop_joins = true) {
  auto rows = run_multiple_agg(query_str, device_type, allow_loop_joins);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size());
  return crt_row[0];
}

TargetValue get_first_target(const string& query_str,
                             const ExecutorDeviceType device_type,
                             const bool geo_return_geo_tv = true) {
  auto rows = run_multiple_agg(query_str, device_type);
  if (geo_return_geo_tv) {
    rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
  }
  auto crt_row = rows->getNextRow(true, true);
  CHECK_GE(crt_row.size(), size_t(1));
  return crt_row[0];
}

inline void run_ddl_statement(const std::string& create_table_stmt) {
  QueryRunner::run_ddl_statement(create_table_stmt, g_session);
}

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU &&
         !g_session->getCatalog().getDataMgr().gpusPresent();
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

bool approx_eq(const double v, const double target, const double eps = 0.01) {
  const auto v_u64 = *reinterpret_cast<const uint64_t*>(may_alias_ptr(&v));
  const auto target_u64 = *reinterpret_cast<const uint64_t*>(may_alias_ptr(&target));
  return v_u64 == target_u64 || (target - eps < v && v < target + eps);
}

int parse_fractional_seconds(uint sfrac, int ntotal, SQLTypeInfo& ti) {
  return TimeGM::instance().parse_fractional_seconds(sfrac, ntotal, ti);
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
    const auto results = QueryRunner::run_multiple_agg(
        query_string, g_session, device_type, g_hoist_literals, true);
    const auto arrow_mapd_results = result_set_arrow_loopback(nullptr, results);
    compare_impl(arrow_mapd_results.get(), sqlite_query_string, device_type, false);
  }

  void compare(const std::string& query_string,
               const std::string& sqlite_query_string,
               const ExecutorDeviceType device_type) {
    const auto mapd_results = run_multiple_agg(query_string, device_type);
    compare_impl(mapd_results.get(), sqlite_query_string, device_type, false);
  }

  // added to deal with time shift for now testing
  void compare_timstamp_approx(const std::string& query_string,
                               const ExecutorDeviceType device_type) {
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
    auto row_iterator = mapd_results->rowIterator(true, true);
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const auto crt_row =
          g_use_row_iterator ? *row_iterator++ : mapd_results->getNextRow(true, true);
      CHECK(!crt_row.empty());
      CHECK_EQ(static_cast<size_t>(num_cols), crt_row.size());
      for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        const auto ref_col_type = connector_.columnTypes[col_idx];
        const auto mapd_variant = crt_row[col_idx];
        const auto scalar_mapd_variant = boost::get<ScalarTargetValue>(&mapd_variant);
        CHECK(scalar_mapd_variant);
        auto mapd_ti = mapd_results->getColType(col_idx);
        const auto mapd_type = mapd_ti.get_type();
        checkTypeConsistency(ref_col_type, mapd_ti);
        const bool ref_is_null = connector_.isNull(row_idx, col_idx);
        switch (mapd_type) {
          case kTINYINT:
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
            time_t nsec = 0;
            const int dimen = mapd_ti.get_dimension();
            if (ref_is_null) {
              CHECK_EQ(inline_int_null_val(mapd_ti), mapd_val);
            } else {
              struct tm tm_struct {
                0
              };
              const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
              auto end_str =
                  strptime(ref_val.c_str(),
                           mapd_type == kTIMESTAMP ? "%Y-%m-%d %H:%M:%S" : "%Y-%m-%d",
                           &tm_struct);
              // handle fractional seconds
              if (end_str != nullptr && *end_str != '.') {
                if (end_str) {
                  ASSERT_EQ(0, *end_str);
                }
                ASSERT_EQ(ref_val.size(), static_cast<size_t>(end_str - ref_val.c_str()));
              }
              if (dimen > 0 && mapd_type == kTIMESTAMP) {
                int fs = 0;
                if (*end_str == '.') {
                  end_str++;
                  uint frac_num;
                  int ntotal;
                  sscanf(end_str, "%d%n", &frac_num, &ntotal);
                  fs = parse_fractional_seconds(frac_num, ntotal, mapd_ti);
                  nsec = timegm(&tm_struct) * pow(10, dimen);
                  nsec += fs;
                } else if (*end_str == '\0') {
                  nsec = timegm(&tm_struct) * pow(10, dimen);
                } else {
                  CHECK(false);
                }
              }
              if (timestamp_approx) {
                // approximate result give 10 second lee way
                ASSERT_NEAR(*mapd_as_int_p,
                            dimen > 0 ? nsec : timegm(&tm_struct),
                            dimen > 0 ? 10 * pow(10, dimen) : 10);
              } else {
                ASSERT_EQ(*mapd_as_int_p, dimen > 0 ? nsec : timegm(&tm_struct));
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

void c(const std::string& query_string,
       const std::string& sqlite_query_string,
       const ExecutorDeviceType device_type) {
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

#define SKIP_ALL_ON_AGGREGATOR()                           \
  if (g_aggregator) {                                      \
    LOG(WARNING) << "Tests not valid in distributed mode"; \
    return;                                                \
  }

#define SKIP_ON_AGGREGATOR(EXP) \
  if (!g_aggregator) {          \
    EXP;                        \
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
      EXPECT_NO_THROW(run_ddl_statement(
          "CREATE TABLE test1 (t1 TEXT) WITH (page_size=" + page_size + ");"));
      EXPECT_NO_THROW(run_multiple_agg("INSERT INTO test1 VALUES('hello, MapD');", dt));
      EXPECT_NO_THROW(run_multiple_agg("SELECT * FROM test1;", dt));
    }
  }
}
// Code is commented out while we resolve the leak in parser
// TEST(Create, PageSize_NegativeCase) {
//  run_ddl_statement("DROP TABLE IF EXISTS test1;");
//  ASSERT_EQ(validate_statement_syntax("CREATE TABLE test1 (t1 TEXT) WITH
//  (page_size=null);"), false); ASSERT_EQ(validate_statement_syntax("CREATE TABLE test1
//  (t1 TEXT) WITH (page_size=);"), false); EXPECT_THROW(run_ddl_statement("CREATE TABLE
//  test1 (t1 TEXT) WITH (page_size=-1);"), std::runtime_error);
//  EXPECT_THROW(run_ddl_statement("CREATE TABLE test1 (t1 TEXT) WITH (page_size=0);"),
//  std::runtime_error); EXPECT_THROW(run_ddl_statement("CREATE TABLE test1 (t1 TEXT) WITH
//  (page_size=2147483648);"), std::runtime_error);
//}

TEST(Insert, ArrayNulls) {
  const char* create_table_array_with_nulls =
      R"(create table table_array_with_nulls (i smallint, sia smallint[], fa2 float[2]);)";
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    run_ddl_statement("DROP TABLE IF EXISTS table_array_with_nulls;");
    EXPECT_NO_THROW(run_ddl_statement(create_table_array_with_nulls));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO table_array_with_nulls "
                         "VALUES(1, {1,1}, ARRAY[1.0,1.0]);",
                         dt));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO table_array_with_nulls "
                         "VALUES(2, {NULL,2}, {NULL,2.0});",
                         dt));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO table_array_with_nulls "
                         "VALUES(3, {3,NULL}, {3.0, NULL});",
                         dt));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO table_array_with_nulls "
                         "VALUES(4, {NULL,NULL}, {NULL,NULL});",
                         dt));
    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT MIN(sia[1]) FROM table_array_with_nulls;", dt)));
    ASSERT_EQ(3,
              v<int64_t>(
                  run_simple_agg("SELECT MAX(sia[1]) FROM table_array_with_nulls;", dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM table_array_with_nulls WHERE sia[2] IS NULL;", dt)));
    ASSERT_EQ(
        3.0,
        v<float>(run_simple_agg("SELECT MAX(fa2[1]) FROM table_array_with_nulls;", dt)));
    ASSERT_EQ(
        2.0,
        v<float>(run_simple_agg("SELECT MAX(fa2[2]) FROM table_array_with_nulls;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM table_array_with_nulls WHERE fa2[1] IS NOT NULL;",
                  dt)));
  }
}

TEST(Insert, DictBoundary) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    run_ddl_statement("DROP TABLE IF EXISTS table_with_small_dict;");
    EXPECT_NO_THROW(run_ddl_statement(
        "CREATE TABLE table_with_small_dict (i INT, t TEXT ENCODING DICT(8));"));

    for (int cVal = 0; cVal < 280; cVal++) {
      string insString = "INSERT INTO table_with_small_dict VALUES (" +
                         std::to_string(cVal) + ", '" + std::to_string(cVal) + "');";
      EXPECT_NO_THROW(run_multiple_agg(insString, dt));
    }

    ASSERT_EQ(
        280,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM table_with_small_dict;", dt)));
    ASSERT_EQ(255,
              v<int64_t>(run_simple_agg(
                  "SELECT count(distinct t) FROM table_with_small_dict;", dt)));
    ASSERT_EQ(25,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM table_with_small_dict WHERE t IS NULL;", dt)));
  }
}

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
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND t < 1002;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 103);", dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 OR (z > 100 AND z < 102) OR (t > "
      "1000 AND t < 1003);",
      dt);
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
    c("SELECT x + 2 * 10/4 + 3 AS expr FROM test WHERE x + 3*8/2 < 35 + y - 20/5 ORDER "
      "BY expr ASC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE ff + 3.0*8 < 20.0/5;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y AND 0=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y AND 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y OR 1<1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < y OR 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE x < 35 AND x < y AND 1=1 AND 0=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE 1>2 AND x < 35 AND x < y AND y < 10;", dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT COUNT(*) FROM test WHERE x < y GROUP BY x HAVING 0=1;", dt));
    c("SELECT COUNT(*) FROM test WHERE x < y GROUP BY x HAVING 1=1;", dt);
    c("SELECT COUNT(*) FROM test WHERE ofq >= 0 OR ofq IS NULL;", dt);
    c("SELECT COUNT(*) AS val FROM test WHERE (test.dd = 0.5 OR test.dd = 3);", dt);
    c("SELECT MAX(dd_notnull * 1) FROM test;", dt);
    c("SELECT x, COUNT(*) AS n FROM test GROUP BY x, ufd ORDER BY x, n;", dt);
    c("SELECT MIN(x), MAX(x) FROM test WHERE real_str LIKE '%nope%';", dt);
    c("SELECT COUNT(*) FROM test WHERE (x > 7 AND y / (x - 7) < 44);", dt);
    c("SELECT x, AVG(ff) AS val FROM test GROUP BY x ORDER BY val;", dt);
    c("SELECT x, MAX(fn) as val FROM test WHERE fn IS NOT NULL GROUP BY x ORDER BY val;",
      dt);
    c("SELECT MAX(dn) FROM test WHERE dn IS NOT NULL;", dt);
    c("SELECT x, MAX(dn) as val FROM test WHERE dn IS NOT NULL GROUP BY x ORDER BY val;",
      dt);
    c("SELECT COUNT(*) as val FROM test GROUP BY x, y, ufd ORDER BY val;", dt);
    ASSERT_NEAR(
        static_cast<double>(-1000.3),
        v<double>(run_simple_agg(
            "SELECT AVG(fn) AS val FROM test GROUP BY rowid ORDER BY val LIMIT 1;", dt)),
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
    c("SELECT COUNT(*) FROM test WHERE o2 > '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o2 <= '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o2 = '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o2 <> '1999-09-08';", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 = o2;", dt);
    c("SELECT COUNT(*) FROM test WHERE o1 <> o2;", dt);
    ASSERT_EQ(19,
              v<int64_t>(run_simple_agg("SELECT rowid FROM test WHERE rowid = 19;", dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT MAX(rowid) - MIN(rowid) + 1 FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) = 0;", dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) = 7;", dt)));
    ASSERT_EQ(5,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) <> 0;", dt)));
    ASSERT_EQ(20,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE MOD(x, 7) <> 7;", dt)));
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
        v<double>(run_simple_agg("SELECT POWER(((SUM(dd * dd) - SUM(dd) * SUM(dd) / "
                                 "COUNT(dd)) / (COUNT(dd) - 1)), 0.5) FROM test;",
                                 dt)),
        static_cast<double>(1.0));
    ASSERT_NEAR(static_cast<double>(8485.0),
                v<double>(run_simple_agg("SELECT VAR_POP(dd) FROM test;", dt)),
                static_cast<double>(10.0));
    ASSERT_NEAR(static_cast<double>(8932.0),
                v<double>(run_simple_agg("SELECT VAR_SAMP(dd) FROM test;", dt)),
                static_cast<double>(10.0));
    ASSERT_EQ(20,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test HAVING STDDEV_POP(x) < 1.0;", dt)));
    ASSERT_EQ(20,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test HAVING STDDEV_POP(x) * 5 < 3.0;", dt)));
    ASSERT_NEAR(
        static_cast<double>(0.65),
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
        v<double>(run_simple_agg("SELECT POWER(((SUM(dd * dd) - SUM(dd) * SUM(dd) / "
                                 "COUNT(dd)) / (COUNT(dd) - 1)), 0.5) FROM test;",
                                 dt)),
        static_cast<double>(1.0));
    ASSERT_NEAR(static_cast<float>(8485.0),
                v<float>(run_simple_agg("SELECT VAR_POP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(10.0));
    ASSERT_NEAR(static_cast<float>(8932.0),
                v<float>(run_simple_agg("SELECT VAR_SAMP_FLOAT(dd) FROM test;", dt)),
                static_cast<float>(10.0));
    ASSERT_EQ(20,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test HAVING STDDEV_POP_FLOAT(x) < 1.0;", dt)));
    ASSERT_EQ(
        20,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test HAVING STDDEV_POP_FLOAT(x) * 5 < 3.0;", dt)));
    ASSERT_NEAR(static_cast<float>(0.65),
                v<float>(run_simple_agg(
                    "SELECT stddev_FLOAT(x) + VARIANCE_float(x) FROM test;", dt)),
                static_cast<float>(0.10));
    ASSERT_NEAR(static_cast<double>(0.125),
                v<double>(run_simple_agg("SELECT COVAR_POP(x, y) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<float>(0.125),
                v<float>(run_simple_agg("SELECT COVAR_POP_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.001));
    ASSERT_NEAR(
        static_cast<double>(0.125),  // covar_pop expansion
        v<double>(run_simple_agg("SELECT avg(x * y) - avg(x) * avg(y) FROM test;", dt)),
        static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<double>(0.131),
                v<double>(run_simple_agg("SELECT COVAR_SAMP(x, y) FROM test;", dt)),
                static_cast<double>(0.001));
    ASSERT_NEAR(static_cast<float>(0.131),
                v<float>(run_simple_agg("SELECT COVAR_SAMP_FLOAT(x, y) FROM test;", dt)),
                static_cast<float>(0.001));
    ASSERT_NEAR(
        static_cast<double>(0.131),  // covar_samp expansion
        v<double>(run_simple_agg(
            "SELECT ((sum(x * y) - sum(x) * avg(y)) / (count(x) - 1)) FROM test;", dt)),
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

    // == Tests related to GPU shared-memory support
    if ((dt == ExecutorDeviceType::GPU) && g_enable_smem_group_by) {
      c("SELECT COUNT(*) FROM test GROUP BY x ORDER BY x DESC;", dt);
      c("SELECT y, COUNT(*) FROM test GROUP BY y ORDER BY y DESC;", dt);
      c("SELECT str, COUNT(*) FROM test GROUP BY str ORDER BY str DESC;", dt);
      c("SELECT COUNT(*), z FROM test where x = 7 GROUP BY z ORDER BY z DESC;", dt);
      c("SELECT z as z0, z as z1, COUNT(*) FROM test GROUP BY z0, z1 ORDER BY z0 DESC;",
        dt);
      ;
    }
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
      const auto rows =
          run_multiple_agg("SELECT * FROM test WHERE x <> 8 LIMIT 3 OFFSET 1;", dt);
      ASSERT_EQ(size_t(3), rows->rowCount());
    }
    c("SELECT str FROM (SELECT str, SUM(y) as total_y FROM test GROUP BY str ORDER BY "
      "total_y DESC, "
      "str LIMIT 1);",
      dt);
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
    c("SELECT COUNT(*) FROM test WHERE f > 1.0 AND f < 1.2 OR (d > 2.0 AND d < 3.0);",
      dt);
    c("SELECT SUM(x + y) FROM test WHERE f > 1.0 AND f < 1.2;", dt);
    c("SELECT SUM(x + y) FROM test WHERE d + f > 3.0 AND d + f < 4.0;", dt);
    c("SELECT SUM(f + d) FROM test WHERE x - y = -35;", dt);
    c("SELECT SUM(f + d) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT SUM(f * d + 15) FROM test WHERE x + y + 1 = 50;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), AVG(x * f + 15), COUNT(*) FROM test WHERE "
      "x + y > 47 AND x + y < 51;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(y) > 42.0 "
      "ORDER BY n;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) > 1.09 "
      "ORDER BY n;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(f) > 1.09 "
      "AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(d) > 2.2 "
      "AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING AVG(d) > 2.2 "
      "AND AVG(y) > 42.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f) + AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING "
      "AVG(f) + AVG(d) > 3.0 ORDER BY n;",
      dt);
    c("SELECT AVG(f) + AVG(d), MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING "
      "AVG(f) + AVG(d) > 3.5 ORDER BY n;",
      dt);
    c("SELECT f + d AS s, x * y FROM test ORDER by s DESC;", dt);
    c("SELECT COUNT(*) AS n FROM test GROUP BY f ORDER BY n;", dt);
    c("SELECT f, COUNT(*) FROM test GROUP BY f HAVING f > 1.25;", dt);
    c("SELECT COUNT(*) AS n FROM test GROUP BY d ORDER BY n;", dt);
    c("SELECT MIN(x + y) AS n FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY f + 1, "
      "f + d ORDER BY n;",
      dt);
    c("SELECT f + d AS s FROM test GROUP BY s ORDER BY s DESC;", dt);
    c("SELECT f + 1 AS s, AVG(u * f) FROM test GROUP BY s ORDER BY s DESC;", dt);
    c("SELECT (CAST(dd AS float) * 0.5) AS key FROM test GROUP BY key ORDER BY key DESC;",
      dt);
    c("SELECT (CAST(dd AS double) * 0.5) AS key FROM test GROUP BY key ORDER BY key "
      "DESC;",
      dt);

    c("SELECT fn FROM test ORDER BY fn ASC NULLS FIRST;",
      "SELECT fn FROM test ORDER BY fn ASC;",
      dt);
    c("SELECT fn FROM test WHERE fn < 0 OR fn IS NULL ORDER BY fn ASC NULLS FIRST;",
      "SELECT fn FROM test WHERE fn < 0 OR fn IS NULL ORDER BY fn ASC;",
      dt);
    ASSERT_NEAR(static_cast<double>(1.3),
                v<double>(run_simple_agg("SELECT AVG(f) AS n FROM test WHERE x = 7 GROUP "
                                         "BY z HAVING AVG(y) + STDDEV(y) "
                                         "> 42.0 ORDER BY n + VARIANCE(y);",
                                         dt)),
                static_cast<double>(0.1));
    ASSERT_NEAR(
        static_cast<double>(92.0),
        v<double>(run_simple_agg("SELECT STDDEV_POP(dd) AS n FROM test ORDER BY n;", dt)),
        static_cast<double>(1.0));
  }
}

TEST(Select, FilterShortCircuit) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND t > 1004 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND t > 1004 AND t > 1005 AND UNLIKELY(t < 1002);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND z > 100 AND z < 102 AND t > "
      "1000 AND t > 1000 AND t > 1001 "
      "AND t > 1002 AND t > 1003 AND UNLIKELY(t < 111) AND (str LIKE 'f__%%');",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x > 6 AND x < 8 AND UNLIKELY(z < 200) AND z > 100 "
      "AND z < 102 AND t > 1000 AND "
      "t > 1000 AND t > 1001  AND UNLIKELY(t < 1111 AND t > 1100) AND (str LIKE 'f__%%') "
      "AND t > 1002 AND t > 1003;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE UNLIKELY(x IN (7, 8, 9, 10)) AND y > 42;", dt);
  }
}

TEST(Select, FilterAndMultipleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT AVG(x), AVG(y) FROM test;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), COUNT(*) FROM test WHERE x + y > 47 AND x "
      "+ y < 51;",
      dt);
    c("SELECT str, AVG(x), COUNT(*) as xx, COUNT(*) as countval FROM test GROUP BY str "
      "ORDER BY str;",
      dt);
  }
}

TEST(Select, FilterAndGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", dt);
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x + 1, x + "
      "y;",
      dt);
    c("SELECT x, y, COUNT(*) FROM test GROUP BY x, y;", dt);
    c("SELECT x, dd, COUNT(*) FROM test GROUP BY x, dd ORDER BY x, dd;", dt);
    c("SELECT dd AS key1, COUNT(*) AS value1 FROM test GROUP BY key1 HAVING key1 IS NOT "
      "NULL ORDER BY key1, value1 "
      "DESC "
      "LIMIT 12;",
      dt);
    c("SELECT 'literal_string' AS key0 FROM test GROUP BY key0;", dt);
    c("SELECT str, MIN(y) FROM test WHERE y IS NOT NULL GROUP BY str ORDER BY str DESC;",
      dt);
    c("SELECT x, MAX(z) FROM test WHERE z IS NOT NULL GROUP BY x HAVING x > 7;", dt);
    c("SELECT CAST((dd - 0.5) * 2.0 AS int) AS key0, COUNT(*) AS val FROM test WHERE (dd "
      ">= 100.0 AND dd < 400.0) "
      "GROUP "
      "BY key0 HAVING key0 >= 0 AND key0 < 400 ORDER BY val DESC LIMIT 50 OFFSET 0;",
      dt);
    c("SELECT y, AVG(CASE WHEN x BETWEEN 6 AND 7 THEN x END) FROM test GROUP BY y ORDER "
      "BY y;",
      dt);
    c("SELECT x, AVG(u), COUNT(*) AS n FROM test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT f, ss FROM test GROUP BY f, ss ORDER BY f DESC;", dt);
    c("SELECT fx, COUNT(*) FROM test GROUP BY fx HAVING COUNT(*) > 5;", dt);
    c("SELECT fx, COUNT(*) n FROM test GROUP BY fx ORDER BY n DESC, fx IS NULL DESC;",
      dt);
    c("SELECT CASE WHEN x > 8 THEN 100000000 ELSE 42 END AS c, COUNT(*) FROM test GROUP "
      "BY c;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE CAST((CAST(x AS FLOAT) - 0) * 0.2 AS INT) = 1;",
      dt);
    c("SELECT CAST(CAST(d AS FLOAT) AS INTEGER) AS key, COUNT(*) FROM test GROUP BY key;",
      dt);
    c("SELECT x * 2 AS x2, COUNT(DISTINCT y) AS n FROM test GROUP BY x2 ORDER BY n DESC;",
      dt);
    c("SELECT x, COUNT(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt);
    c("SELECT str, SUM(y - y) FROM test GROUP BY str ORDER BY str ASC;", dt);
    c("SELECT str, SUM(y - y) FROM test WHERE y - y IS NOT NULL GROUP BY str ORDER BY "
      "str ASC;",
      dt);
    c("select shared_dict,m from test where (m >= CAST('2014-12-13 22:23:15' AS "
      "TIMESTAMP(0)) and m <= "
      "CAST('2014-12-14 22:23:15' AS TIMESTAMP(0)))  and CAST(m AS TIMESTAMP(0)) BETWEEN "
      "'2014-12-14 22:23:15' AND "
      "'2014-12-13 22:23:15' group by shared_dict,m;",
      dt);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT x, MIN(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT x, MAX(real_str) FROM test GROUP BY x ORDER BY x DESC;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT MIN(str) FROM test GROUP BY x;", dt),
                 std::runtime_error);
  }
}

TEST(Select, Arrays) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // Simple lazy projection
    compare_array(run_simple_agg("SELECT arr_i16 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({2, 3, 4}));
    compare_array(run_simple_agg("SELECT arr_i32 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({20, 30, 40}));
    compare_array(run_simple_agg("SELECT arr_i64 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({200, 300, 400}));
    compare_array(run_simple_agg("SELECT arr_str FROM array_test WHERE x = 8;", dt),
                  std::vector<std::string>({"bb", "cc", "dd"}));
    compare_array(run_simple_agg("SELECT arr_float FROM array_test WHERE x = 8;", dt),
                  std::vector<float>({2.2, 3.3, 4.4}));
    compare_array(run_simple_agg("SELECT arr_double FROM array_test WHERE x = 8;", dt),
                  std::vector<double>({22.2, 33.3, 44.4}));
    compare_array(run_simple_agg("SELECT arr_bool FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({1, 0, 1, 0, 1, 0}));

    compare_array(run_simple_agg("SELECT arr3_i8 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({2, 3, 4}));
    compare_array(run_simple_agg("SELECT arr3_i16 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({2, 3, 4}));
    compare_array(run_simple_agg("SELECT arr3_i32 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({20, 30, 40}));
    compare_array(run_simple_agg("SELECT arr3_i64 FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({200, 300, 400}));
    compare_array(run_simple_agg("SELECT arr3_float FROM array_test WHERE x = 8;", dt),
                  std::vector<float>({2.2, 3.3, 4.4}));
    compare_array(run_simple_agg("SELECT arr3_double FROM array_test WHERE x = 8;", dt),
                  std::vector<double>({22.2, 33.3, 44.4}));
    compare_array(run_simple_agg("SELECT arr6_bool FROM array_test WHERE x = 8;", dt),
                  std::vector<int64_t>({1, 0, 1, 0, 1, 0}));

    // Simple non-lazy projection
    compare_array(
        run_simple_agg("SELECT arr_i16 FROM array_test WHERE arr_i16[1] = 2;", dt),
        std::vector<int64_t>({2, 3, 4}));
    compare_array(
        run_simple_agg("SELECT arr_i32 FROM array_test WHERE arr_i32[1] = 20;", dt),
        std::vector<int64_t>({20, 30, 40}));
    compare_array(
        run_simple_agg("SELECT arr_i64 FROM array_test WHERE arr_i64[1] = 200;", dt),
        std::vector<int64_t>({200, 300, 400}));
    compare_array(
        run_simple_agg("SELECT arr_str FROM array_test WHERE arr_str[1] = 'bb';", dt),
        std::vector<std::string>({"bb", "cc", "dd"}));
    // TODO(adb): Calcite is casting the column value to DOUBLE to do the comparison,
    // which results in the comparison failing. Is this desired behavior or a bug? Adding
    // the CAST below for now to test projection.
    compare_array(
        run_simple_agg(
            "SELECT arr_float FROM array_test WHERE arr_float[1] = CAST(2.2 as FLOAT);",
            dt),
        std::vector<float>({2.2, 3.3, 4.4}));
    compare_array(
        run_simple_agg("SELECT arr_double FROM array_test WHERE arr_double[1] = 22.2;",
                       dt),
        std::vector<double>({22.2, 33.3, 44.4}));
    compare_array(run_simple_agg(
                      "SELECT arr_bool FROM array_test WHERE x < 9 AND arr_bool[1];", dt),
                  std::vector<int64_t>({1, 0, 1, 0, 1, 0}));

    compare_array(
        run_simple_agg("SELECT arr3_i8 FROM array_test WHERE arr3_i8[1] = 2;", dt),
        std::vector<int64_t>({2, 3, 4}));
    compare_array(
        run_simple_agg("SELECT arr3_i16 FROM array_test WHERE arr3_i16[1] = 2;", dt),
        std::vector<int64_t>({2, 3, 4}));
    compare_array(
        run_simple_agg("SELECT arr3_i32 FROM array_test WHERE arr3_i32[1] = 20;", dt),
        std::vector<int64_t>({20, 30, 40}));
    compare_array(
        run_simple_agg("SELECT arr3_i64 FROM array_test WHERE arr3_i64[1] = 200;", dt),
        std::vector<int64_t>({200, 300, 400}));
    compare_array(
        run_simple_agg(
            "SELECT arr3_float FROM array_test WHERE arr3_float[1] = CAST(2.2 AS FLOAT);",
            dt),
        std::vector<float>({2.2, 3.3, 4.4}));
    compare_array(
        run_simple_agg("SELECT arr3_double FROM array_test WHERE arr3_double[1] = 22.2;",
                       dt),
        std::vector<double>({22.2, 33.3, 44.4}));
    compare_array(
        run_simple_agg("SELECT arr6_bool FROM array_test WHERE x < 9 AND arr6_bool[1];",
                       dt),
        std::vector<int64_t>({1, 0, 1, 0, 1, 0}));
  }
}

TEST(Select, FilterCastToDecimal) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_EQ(static_cast<int64_t>(5),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 7.1;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(10),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE y > 42.5;", dt)));
    ASSERT_EQ(static_cast<int64_t>(10),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE ufd > -2147483648.0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(15),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE ofd > -2147483648;", dt)));
  }
}

TEST(Select, FilterAndGroupByMultipleAgg) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 "
      "GROUP BY x, y;",
      dt);
    c("SELECT MIN(x + y), COUNT(*), AVG(x + 1) FROM test WHERE x + y > 47 AND x + y < 53 "
      "GROUP BY x + 1, x + y;",
      dt);
  }
}

TEST(Select, GroupByKeylessAndNotKeyless) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT fixed_str FROM test WHERE fixed_str = 'fish' GROUP BY fixed_str;", dt);
    c("SELECT AVG(x), fixed_str FROM test WHERE fixed_str = 'fish' GROUP BY fixed_str;",
      dt);
    c("SELECT AVG(smallint_nulls), fixed_str FROM test WHERE fixed_str = 'foo' GROUP BY "
      "fixed_str;",
      dt);
    c("SELECT null_str, AVG(smallint_nulls) FROM test GROUP BY null_str;", dt);
  }
}

TEST(Select, Having) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING MAX(x) > 5 ORDER BY n;",
      dt);
    c("SELECT MAX(y) AS n FROM test WHERE x = 7 GROUP BY z HAVING MAX(x) > 5 ORDER BY n "
      "LIMIT 1;",
      dt);
    c("SELECT MAX(y) AS n FROM test WHERE x > 7 GROUP BY z HAVING MAX(x) < 100 ORDER BY "
      "n;",
      dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 ORDER "
      "BY n;",
      dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 AND "
      "COUNT(*) > 5 ORDER BY n;",
      dt);
    c("SELECT z, SUM(y) AS n FROM test WHERE x > 6 GROUP BY z HAVING MAX(x) < 100 AND "
      "COUNT(*) > 9 ORDER BY n;",
      dt);
    c("SELECT str, COUNT(*) AS n FROM test GROUP BY str HAVING str IN ('bar', 'baz') "
      "ORDER BY str;",
      dt);
    c("SELECT str, COUNT(*) AS n FROM test GROUP BY str HAVING str LIKE 'ba_' ORDER BY "
      "str;",
      dt);
    c("SELECT ss, COUNT(*) AS n FROM test GROUP BY ss HAVING ss LIKE 'bo_' ORDER BY ss;",
      dt);
    c("SELECT x, COUNT(*) FROM test WHERE x > 9 GROUP BY x HAVING x > 15;", dt);
    c("SELECT x, AVG(y), AVG(y) FROM test GROUP BY x HAVING x >= 0 ORDER BY x;", dt);
    c("SELECT AVG(y), x, AVG(y) FROM test GROUP BY x HAVING x >= 0 ORDER BY x;", dt);
    c("SELECT x, y, COUNT(*) FROM test WHERE real_str LIKE 'nope%' GROUP BY x, y HAVING "
      "x >= 0 AND x < 12 AND y >= 0 "
      "AND y < 12 ORDER BY x, y;",
      dt);
  }
}

TEST(Select, CountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(distinct x) FROM test;", dt);
    c("SELECT COUNT(distinct b) FROM test;", dt);
    SKIP_ON_AGGREGATOR(c("SELECT COUNT(distinct f) FROM test;",
                         dt));  // Exception: Cannot use a fast path for COUNT distinct
    SKIP_ON_AGGREGATOR(c("SELECT COUNT(distinct d) FROM test;",
                         dt));  // Exception: Cannot use a fast path for COUNT distinct
    c("SELECT COUNT(distinct str) FROM test;", dt);
    c("SELECT COUNT(distinct ss) FROM test;", dt);
    c("SELECT COUNT(distinct x + 1) FROM test;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test "
      "GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM "
      "test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(distinct dd) AS n FROM test GROUP BY y ORDER BY n;", dt);
    c("SELECT z, str, AVG(z), COUNT(distinct z) FROM test GROUP BY z, str ORDER BY z, "
      "str;",
      dt);
    c("SELECT AVG(z), COUNT(distinct x) AS dx FROM test GROUP BY y HAVING dx > 1;", dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT z, str, COUNT(distinct f) FROM test GROUP BY z, str ORDER BY str DESC;",
          dt));  // Exception: Cannot use a fast path for COUNT distinct
    c("SELECT COUNT(distinct x * (50000 - 1)) FROM test;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct real_str) FROM test;", dt),
                 std::runtime_error);  // Exception: Strings must be dictionary-encoded
                                       // for COUNT(DISTINCT).
  }
}

TEST(Select, ApproxCountDistinct) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT APPROX_COUNT_DISTINCT(x) FROM test;",
      "SELECT COUNT(distinct x) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(x) FROM test_empty;",
      "SELECT COUNT(distinct x) FROM test_empty;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(x) FROM test_one_row;",
      "SELECT COUNT(distinct x) FROM test_one_row;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(b) FROM test;",
      "SELECT COUNT(distinct b) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(f) FROM test;",
      "SELECT COUNT(distinct f) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(d) FROM test;",
      "SELECT COUNT(distinct d) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(str) FROM test;",
      "SELECT COUNT(distinct str) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(null_str) FROM test;",
      "SELECT COUNT(distinct null_str) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(ss) FROM test WHERE ss IS NOT NULL;",
      "SELECT COUNT(distinct ss) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(x + 1) FROM test;",
      "SELECT COUNT(distinct x + 1) FROM test;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x) "
      "FROM test GROUP BY y ORDER "
      "BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test "
      "GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x + "
      "1) FROM test GROUP BY y "
      "ORDER BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM "
      "test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(dd) AS n FROM test GROUP BY y ORDER BY n;",
      "SELECT COUNT(distinct dd) AS n FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT z, str, AVG(z), APPROX_COUNT_DISTINCT(z) FROM test GROUP BY z, str ORDER "
      "BY z;",
      "SELECT z, str, AVG(z), COUNT(distinct z) FROM test GROUP BY z, str ORDER BY z;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(null_str) AS n FROM test GROUP BY x ORDER BY n;",
      "SELECT COUNT(distinct null_str) AS n FROM test GROUP BY x ORDER BY n;",
      dt);
    c("SELECT z, APPROX_COUNT_DISTINCT(null_str) AS n FROM test GROUP BY z ORDER BY z, "
      "n;",
      "SELECT z, COUNT(distinct null_str) AS n FROM test GROUP BY z ORDER BY z, n;",
      dt);
    c("SELECT AVG(z), APPROX_COUNT_DISTINCT(x) AS dx FROM test GROUP BY y HAVING dx > 1;",
      "SELECT AVG(z), COUNT(distinct x) AS dx FROM test GROUP BY y HAVING dx > 1;",
      dt);
    SKIP_ON_AGGREGATOR(c(
        "SELECT approx_value, exact_value FROM (SELECT APPROX_COUNT_DISTINCT(x) AS "
        "approx_value FROM test), (SELECT "
        "COUNT(distinct x) AS exact_value FROM test);",
        "SELECT approx_value, exact_value FROM (SELECT COUNT(distinct x) AS approx_value "
        "FROM test), (SELECT "
        "COUNT(distinct x) AS exact_value FROM test);",
        dt));
    c("SELECT APPROX_COUNT_DISTINCT(x, 1) FROM test;",
      "SELECT COUNT(distinct x) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(b, 10) FROM test;",
      "SELECT COUNT(distinct b) FROM test;",
      dt);
    c("SELECT APPROX_COUNT_DISTINCT(f, 20) FROM test;",
      "SELECT COUNT(distinct f) FROM test;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x, 1) "
      "FROM test GROUP BY y ORDER "
      "BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test "
      "GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, APPROX_COUNT_DISTINCT(x + "
      "1, 1) FROM test GROUP BY y "
      "ORDER BY n;",
      "SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM "
      "test GROUP BY y ORDER BY n;",
      dt);
    EXPECT_NO_THROW(run_multiple_agg(
        "SELECT APPROX_COUNT_DISTINCT(x), SAMPLE(real_str) FROM test GROUP BY x;", dt));
    EXPECT_THROW(
        run_multiple_agg("SELECT APPROX_COUNT_DISTINCT(real_str) FROM test;", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT APPROX_COUNT_DISTINCT(x, 0) FROM test;", dt),
                 std::runtime_error);
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
    c("SELECT fx FROM test WHERE fx IS NULL;", dt);
    c("SELECT z,t,f,m,d,x,real_str,u,z,y FROM test WHERE z = -78 AND t = "
      "1002 AND x >= 8 AND y = 43 AND d > 1.0 AND f > 1.0 AND real_str = 'real_bar' "
      "ORDER BY f ASC;",
      dt);
    c("SELECT * FROM test WHERE d > 2.4 AND real_str IS NOT NULL AND fixed_str IS NULL "
      "AND z = 102 AND fn < 0 AND y = 43 AND t >= 0 AND x <> 8;",
      dt);
    c("SELECT * FROM test WHERE d > 2.4 AND real_str IS NOT NULL AND fixed_str IS NULL "
      "AND z = 102 AND fn < 0 AND y = 43 AND t >= 0 AND x = 8;",
      dt);
    c("SELECT real_str,f,fn,y,d,x,z,str,fixed_str,t,dn FROM test WHERE f IS NOT NULL AND "
      "y IS NOT NULL AND str = 'bar' AND x >= 7 AND t < 1003 AND z < 0;",
      dt);
    c("SELECT t,y,str,real_str,d,fixed_str,dn,fn,z,f,x FROM test WHERE f IS NOT NULL AND "
      "y IS NOT NULL AND str = 'baz' AND x >= 7 AND t < 1003 AND f > 1.2 LIMIT 1;",
      dt);
    c("SELECT fn,real_str,str,z,d,x,fixed_str,dn,y,t,f FROM test WHERE f < 1.4 AND "
      "real_str IS NOT NULL AND fixed_str IS NULL AND z = 102 AND dn < 0 AND y = 43;",
      dt);
    c("SELECT dn,str,y,z,fixed_str,fn,d,real_str,t,f,x FROM test WHERE z < 0 AND f < 2 "
      "AND d > 2.0 AND fn IS NOT NULL AND dn < 2000 AND str IS NOT NULL AND fixed_str = "
      "'bar' AND real_str = 'real_bar' AND t >= 1001 AND y >= 42 AND x > 7 ORDER BY z, "
      "x;",
      dt);
    c("SELECT z,f,d,str,real_str,x,dn,y,t,fn,fixed_str FROM test WHERE fn IS NULL AND dn "
      "IS NULL AND x >= 0 AND real_str = 'real_foo' ORDER BY y;",
      dt);
  }
}

TEST(Select, OrderBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    const auto rows = run_multiple_agg(
        "SELECT x, y, z + t, x * y AS m FROM test ORDER BY 3 desc LIMIT 5;", dt);
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
    c("SELECT x x1, x, COUNT(*) AS val FROM test GROUP BY x HAVING val > 5 ORDER BY val "
      "DESC LIMIT 5;",
      dt);
    c("SELECT ufd, COUNT(*) n FROM test GROUP BY ufd, str ORDER BY ufd, n;", dt);
    c("SELECT -x, COUNT(*) FROM test GROUP BY x ORDER BY x DESC;", dt);
    c("SELECT real_str FROM test WHERE real_str LIKE '%real%' ORDER BY real_str ASC;",
      dt);
    c("SELECT ss FROM test GROUP by ss ORDER BY ss ASC NULLS FIRST;",
      "SELECT ss FROM test GROUP by ss ORDER BY ss ASC;",
      dt);
    c("SELECT str, COUNT(*) n FROM test WHERE x < 0 GROUP BY str ORDER BY n DESC LIMIT "
      "5;",
      dt);
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
    c("SELECT ff as k FROM test ORDER BY k ASC NULLS FIRST LIMIT 20;",
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

TEST(Select, TopKHeap) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, x FROM proj_top ORDER BY x DESC LIMIT 1;", dt);
  }
}

TEST(Select, ComplexQueries) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) * MAX(y) - SUM(z) FROM test;", dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 "
      "AND 200 GROUP BY x, y;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 "
      "AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x);",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 "
      "AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x) + 35;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test WHERE z BETWEEN 100 "
      "AND 200 "
      "GROUP BY x, y HAVING y > 2 * x AND MIN(y) > MAX(x) + 36;",
      dt);
    c("SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test "
      "WHERE z BETWEEN 100 AND 200 GROUP BY a, y;",
      dt);
    c("SELECT x, y FROM (SELECT a.str AS str, b.x AS x, a.y AS y FROM test a, join_test "
      "b WHERE a.x = b.x) WHERE str = "
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

TEST(Select, GroupByPushDownFilterIntoExprRange) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    const auto rows = run_multiple_agg(
        "SELECT b, COUNT(*) AS n FROM test WHERE b GROUP BY b ORDER BY b", dt);
    ASSERT_EQ(
        size_t(1),
        rows->rowCount());  // Sqlite does not have a boolean type, so do this for now
    c("SELECT x, COUNT(*) AS n FROM test WHERE x > 7 GROUP BY x ORDER BY x", dt);
    c("SELECT y, COUNT(*) AS n FROM test WHERE y <= 43 GROUP BY y ORDER BY n DESC", dt);
    c("SELECT z, COUNT(*) AS n FROM test WHERE z <= 43 AND y > 10 GROUP BY z ORDER BY n "
      "DESC",
      dt);
    c("SELECT t, SUM(y) AS sum_y FROM test WHERE t < 2000 GROUP BY t ORDER BY t DESC",
      dt);
    c("SELECT t, SUM(y) AS sum_y FROM test WHERE t < 2000 GROUP BY t ORDER BY sum_y", dt);
    c("SELECT o, COUNT(*) as n FROM test WHERE o <= '1999-09-09' GROUP BY o ORDER BY n",
      dt);
    c("SELECT t + x, AVG(x) AS avg_x FROM test WHERE z <= 50 and t < 2000 GROUP BY t + x "
      "ORDER BY avg_x DESC",
      dt);
  }
}

TEST(Select, GroupByExprNoFilterNoAggregate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x + y AS a FROM test GROUP BY a ORDER BY a;", dt);
    ASSERT_EQ(8,
              v<int64_t>(run_simple_agg("SELECT TRUNCATE(x, 0) AS foo FROM test GROUP BY "
                                        "TRUNCATE(x, 0) ORDER BY foo DESC LIMIT 1;",
                                        dt)));
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
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE "
      "3 END) FROM test;",
      dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 END) FROM test;", dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE "
      "3 END) "
      "FROM test WHERE CASE WHEN y BETWEEN 42 AND 43 THEN 5 ELSE 4 END > 4;",
      dt);
    ASSERT_EQ(std::numeric_limits<int64_t>::min(),
              v<int64_t>(run_simple_agg(
                  "SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 "
                  "THEN 2 ELSE 3 END) FROM test "
                  "WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5 ELSE 4 END > 4;",
                  dt)));
    c("SELECT CASE WHEN x + y > 50 THEN 77 ELSE 88 END AS foo, COUNT(*) FROM test GROUP "
      "BY foo ORDER BY foo;",
      dt);
    ASSERT_EQ(std::numeric_limits<double>::min(),
              v<double>(run_simple_agg(
                  "SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1.1 WHEN x BETWEEN 8 AND "
                  "9 THEN 2.2 ELSE 3.3 END) FROM "
                  "test WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5.1 ELSE 3.9 END > 4;",
                  dt)));
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN "
      "'oops 2' ELSE real_str END c "
      "FROM "
      "test ORDER BY c ASC;",
      dt);

    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN "
      "'oops 2' ELSE str END c FROM "
      "test "
      "ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN "
      "'eight' ELSE 'ooops' END c FROM "
      "test ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN "
      "real_str ELSE 'ooops' END AS g "
      "FROM test ORDER BY g ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN str "
      "ELSE 'ooops' END c FROM test "
      "ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN "
      "'eight' ELSE 'ooops' END c FROM "
      "test ORDER BY c ASC;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN str WHEN x BETWEEN 7 AND 10 THEN 'eight' "
      "ELSE 'ooops' END AS g, "
      "COUNT(*) FROM test GROUP BY g ORDER BY g;",
      dt);
    c("SELECT y AS key0, SUM(CASE WHEN x > 7 THEN x / (x - 7) ELSE 99 END) FROM test "
      "GROUP BY key0 ORDER BY key0;",
      dt);
    c("SELECT CASE WHEN str IN ('str1', 'str3', 'str8') THEN 'foo' WHEN str IN ('str2', "
      "'str4', 'str9') THEN 'bar' "
      "ELSE 'baz' END AS bucketed_str, COUNT(*) AS n FROM query_rewrite_test GROUP BY "
      "bucketed_str ORDER BY n "
      "DESC;",
      dt);
    c("SELECT CASE WHEN y > 40 THEN x END c, x FROM test ORDER BY c ASC;", dt);
    c("SELECT COUNT(CASE WHEN str = 'foo' THEN 1 END) FROM test;", dt);
    c("SELECT COUNT(CASE WHEN str = 'foo' THEN 1 ELSE NULL END) FROM test;", dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN y ELSE y END AS foobar FROM test ORDER BY "
      "foobar DESC;",
      dt);
    c("SELECT x, AVG(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x "
      "ORDER BY x;",
      dt);
    c("SELECT x, SUM(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x "
      "ORDER BY x;",
      dt);
    c("SELECT x, COUNT(CASE WHEN y BETWEEN 41 AND 42 THEN y END) FROM test GROUP BY x "
      "ORDER BY x;",
      dt);
    c("SELECT CASE WHEN x > 8 THEN 'oops' ELSE 'ok' END FROM test LIMIT 1;", dt);
    c("SELECT CASE WHEN x < 9 THEN 'ok' ELSE 'oops' END FROM test LIMIT 1;", dt);
    c("SELECT CASE WHEN str IN ('foo', 'bar') THEN str END key1, COUNT(*) FROM test "
      "GROUP BY str HAVING key1 IS NOT "
      "NULL ORDER BY key1;",
      dt);

    c("SELECT CASE WHEN str IN ('foo') THEN 'FOO' WHEN str IN ('bar') THEN 'BAR' ELSE "
      "'BAZ' END AS g, COUNT(*) "
      "FROM test GROUP BY g ORDER BY g DESC;",
      dt);
    c("SELECT x, COUNT(case when y = 42 then 1 else 0 end) AS n1, COUNT(*) AS n2 FROM "
      "test GROUP BY x ORDER BY n2 "
      "DESC;",
      dt);
    c("SELECT CASE WHEN test.str = 'foo' THEN 'foo' ELSE test.str END AS g FROM test "
      "GROUP BY g ORDER BY g ASC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE CASE WHEN x > 8 THEN 'oops' END = 'oops' OR CASE "
      "WHEN x > 8 THEN 'oops' END = 'oops';",
      dt);
    ASSERT_EQ(
        int64_t(1418428800),
        v<int64_t>(run_simple_agg(
            "SELECT CASE WHEN 1 > 0 THEN DATE_TRUNC(day, m) ELSE DATE_TRUNC(year, m) END "
            "AS date_bin FROM test GROUP BY date_bin;",
            dt)));
    ASSERT_EQ(
        int64_t(1388534400),
        v<int64_t>(run_simple_agg(
            "SELECT CASE WHEN 1 < 0 THEN DATE_TRUNC(day, m) ELSE DATE_TRUNC(year, m) END "
            "AS date_bin FROM test GROUP BY date_bin;",
            dt)));
    c("SELECT COUNT(CASE WHEN str IN ('foo', 'bar') THEN 'foo_bar' END) from test;", dt);
    ASSERT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg(
            "SELECT MIN(CASE WHEN x BETWEEN 7 AND 8 THEN true ELSE false END) FROM test;",
            dt)));
    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg(
            "SELECT MIN(CASE WHEN x BETWEEN 6 AND 7 THEN true ELSE false END) FROM test;",
            dt)));
    c("SELECT CASE WHEN test.str in ('boo', 'simple', 'case', 'not', 'much', 'to', "
      "'see', 'foo_in_case', 'foo', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', "
      "'k', 'l') THEN 'foo_in_case' ELSE test.str END AS g FROM test GROUP BY g ORDER BY "
      "g ASC;",
      dt);
    c("SELECT CASE WHEN shared_dict is null THEN 'hello' ELSE 'world' END key0, count(*) "
      "val FROM test GROUP BY key0 ORDER BY val;",
      dt);

    const auto constrained_by_in_threshold_state = g_constrained_by_in_threshold;
    g_constrained_by_in_threshold = 0;
    ScopeGuard reset_constrained_by_in_threshold = [&constrained_by_in_threshold_state] {
      g_constrained_by_in_threshold = constrained_by_in_threshold_state;
    };
    c("SELECT fixed_str AS key0, str as key1, count(*) as val FROM test WHERE "
      "((fixed_str IN (SELECT fixed_str FROM test GROUP BY fixed_str))) GROUP BY key0, "
      "key1 ORDER BY val desc;",
      dt);
  }
}

TEST(Select, Strings) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT str, COUNT(*) FROM test GROUP BY str HAVING COUNT(*) > 5 ORDER BY str;",
      dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 4 "
      "ORDER BY str;",
      dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 5 "
      "ORDER BY str;",
      dt);
    c("SELECT str, COUNT(*) FROM test where str IS NOT NULL GROUP BY str ORDER BY str;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE ss IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE 'ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%eal_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%ba%';", dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT * FROM test WHERE str LIKE '%' ORDER BY x ASC, y ASC;", dt));
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
    SKIP_ON_AGGREGATOR(c("SELECT COUNT(*) FROM test WHERE str = real_str;", dt));
    c("SELECT COUNT(*) FROM test WHERE str <> str;", dt);
    SKIP_ON_AGGREGATOR(c("SELECT COUNT(*) FROM test WHERE ss <> str;", dt));
    SKIP_ON_AGGREGATOR(c("SELECT COUNT(*) FROM test WHERE ss = str;", dt));
    SKIP_ON_AGGREGATOR(c("SELECT COUNT(*) FROM test WHERE LENGTH(str) = 3;", dt));
    c("SELECT fixed_str, COUNT(*) FROM test GROUP BY fixed_str HAVING COUNT(*) > 5 ORDER "
      "BY fixed_str;",
      dt);
    c("SELECT fixed_str, COUNT(*) FROM test WHERE fixed_str = 'bar' GROUP BY fixed_str "
      "HAVING COUNT(*) > 4 ORDER BY "
      "fixed_str;",
      dt);
    c("SELECT COUNT(*) FROM emp WHERE ename LIKE 'D%%' OR ename = 'Julia';", dt);
    SKIP_ON_AGGREGATOR(
        ASSERT_EQ(2 * g_num_rows,
                  v<int64_t>(run_simple_agg(
                      "SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(str) = 3;", dt))));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str ILIKE 'f%%';", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE (str ILIKE 'f%%');", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE ( str ILIKE 'f%%' );", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str ILIKE 'McDonald''s';", dt)));
    ASSERT_EQ("foo",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT str FROM test WHERE REGEXP_LIKE(str, '^f.?.+');", dt))));
    ASSERT_EQ("bar",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT str FROM test WHERE REGEXP_LIKE(str, '^[a-z]+r$');", dt))));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP '.*';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP '...';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP '.+.+.+';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP '.?.?.?';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or str REGEXP 'fo.';",
                  dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE "
                                        "REGEXP_LIKE(str, 'ba.') or str REGEXP 'fo.?';",
                                        dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP "
                                        "'ba.' or REGEXP_LIKE(str, 'fo.+');",
                                        dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.+';", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(str, '.?ba.*');", dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE "
                                  "REGEXP_LIKE(str,'ba.') or REGEXP_LIKE(str, 'fo.+');",
                                  dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str REGEXP "
                                        "'ba.' or REGEXP_LIKE(str, 'fo.+');",
                                        dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE "
                                        "REGEXP_LIKE(str, 'ba.') or str REGEXP 'fo.?';",
                                        dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE str REGEXP 'ba.' or str REGEXP 'fo.';",
                  dt)));
  }
}

TEST(Select, SharedDictionary) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT shared_dict, COUNT(*) FROM test GROUP BY shared_dict HAVING COUNT(*) > 5 "
      "ORDER BY shared_dict;",
      dt);
    c("SELECT shared_dict, COUNT(*) FROM test WHERE shared_dict = 'bar' GROUP BY "
      "shared_dict HAVING COUNT(*) > 4 ORDER "
      "BY shared_dict;",
      dt);
    c("SELECT shared_dict, COUNT(*) FROM test WHERE shared_dict = 'bar' GROUP BY "
      "shared_dict HAVING COUNT(*) > 5 ORDER "
      "BY shared_dict;",
      dt);
    c("SELECT shared_dict, COUNT(*) FROM test where shared_dict IS NOT NULL GROUP BY "
      "shared_dict ORDER BY shared_dict;",
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
    c("SELECT * FROM test WHERE shared_dict LIKE '@f%%' ESCAPE '@' ORDER BY x ASC, y "
      "ASC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict LIKE 'ba_' or shared_dict LIKE 'fo_';",
      dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict = 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' = shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict <> 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <> shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict = 'foo' OR shared_dict = 'bar';", dt);
    SKIP_ON_AGGREGATOR(c("SELECT COUNT(*) FROM test WHERE shared_dict = real_str;", dt));
    c("SELECT COUNT(*) FROM test WHERE shared_dict <> shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict > 'fo';", dt);
    c("SELECT COUNT(*) FROM test WHERE shared_dict >= 'bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' < shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'fo' < shared_dict;", dt);
    c("SELECT COUNT(*) FROM test WHERE 'bar' <= shared_dict;", dt);
    SKIP_ON_AGGREGATOR(c("SELECT COUNT(*) FROM test WHERE LENGTH(shared_dict) = 3;", dt));

    EXPECT_THROW(run_ddl_statement("CREATE TABLE t1(a text, b text, SHARED DICTIONARY "
                                   "(b) REFERENCES t1(a), SHARED "
                                   "DICTIONARY (a) REFERENCES t1(b));"),
                 std::runtime_error);

    SKIP_ON_AGGREGATOR(ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(shared_dict) = 3;", dt))));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict ILIKE 'f%%';", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE (shared_dict ILIKE 'f%%');", dt)));
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE ( shared_dict ILIKE 'f%%' );", dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE shared_dict ILIKE 'McDonald''s';", dt)));

    ASSERT_EQ(
        "foo",
        boost::get<std::string>(v<NullableString>(run_simple_agg(
            "SELECT shared_dict FROM test WHERE REGEXP_LIKE(shared_dict, '^f.?.+');",
            dt))));
    ASSERT_EQ(
        "baz",
        boost::get<std::string>(v<NullableString>(run_simple_agg(
            "SELECT shared_dict FROM test WHERE REGEXP_LIKE(shared_dict, '^[a-z]+z$');",
            dt))));

    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.*';", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '...';", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.+.+.+';", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP '.?.?.?';", dt)));

    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict "
                                        "REGEXP 'ba.' or shared_dict REGEXP 'fo.';",
                                        dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, 'ba.') or "
                  "shared_dict REGEXP 'fo.?';",
                  dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP "
                                  "'ba.' or REGEXP_LIKE(shared_dict, 'fo.+');",
                                  dt)));
    ASSERT_EQ(5,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE shared_dict REGEXP 'ba.+';", dt)));
    ASSERT_EQ(
        5,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, '.?ba.*');", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict,'ba.') or "
                  "REGEXP_LIKE(shared_dict, 'fo.+');",
                  dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict REGEXP "
                                  "'ba.' or REGEXP_LIKE(shared_dict, 'fo.+');",
                                  dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(shared_dict, 'ba.') or "
                  "shared_dict REGEXP 'fo.?';",
                  dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE shared_dict "
                                        "REGEXP 'ba.' or shared_dict REGEXP 'fo.';",
                                        dt)));
  }
}

TEST(Select, StringCompare) {
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

    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test, test_inner WHERE "
                                  "test.shared_dict < test_inner.str",
                                  dt),
                 std::runtime_error);
  }
}

TEST(Select, StringsNoneEncoding) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE '%eal_bar';", dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT * FROM test WHERE real_str LIKE '%' ORDER BY x ASC, y ASC;", dt));
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%\%' ORDER BY x ASC, y ASC;", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_@f%%' ESCAPE '@' ORDER BY x ASC, y "
      "ASC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba_' or real_str LIKE "
      "'real_fo_';",
      dt);
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
    c("SELECT COUNT(*) FROM test WHERE real_str = 'real_foo' OR real_str = 'real_bar';",
      dt);
    c("SELECT COUNT(*) FROM test WHERE real_str = real_str;", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str <> real_str;", dt);
    ASSERT_EQ(g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE real_str ILIKE 'rEaL_f%%';", dt)));
    c("SELECT COUNT(*) FROM test WHERE LENGTH(real_str) = 8;", dt);
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(real_str) = 8;", dt)));
    SKIP_ON_AGGREGATOR(ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(real_str,'real_.*.*.*');",
            dt))));
    SKIP_ON_AGGREGATOR(ASSERT_EQ(
        g_num_rows,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_ba.*';", dt))));
    SKIP_ON_AGGREGATOR(
        ASSERT_EQ(2 * g_num_rows,
                  v<int64_t>(run_simple_agg(
                      "SELECT COUNT(*) FROM test WHERE real_str REGEXP '.*';", dt))));
    SKIP_ON_AGGREGATOR(ASSERT_EQ(
        g_num_rows,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_f.*.*';", dt))));
    SKIP_ON_AGGREGATOR(ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE real_str REGEXP 'real_f.+\%';", dt))));
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE real_str LIKE str;", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT COUNT(*) FROM test WHERE REGEXP_LIKE(real_str, str);", dt),
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

void check_one_date_trunc_group_with_agg(const ResultSet& rows,
                                         const int64_t ref_ts,
                                         const int64_t ref_agg) {
  const auto crt_row = rows.getNextRow(true, true);
  ASSERT_EQ(size_t(2), crt_row.size());
  const auto actual_ts = v<int64_t>(crt_row[0]);
  ASSERT_EQ(ref_ts, actual_ts);
  const auto actual_agg = v<int64_t>(crt_row[1]);
  ASSERT_EQ(ref_agg, actual_agg);
  const auto empty_row = rows.getNextRow(true, true);
  ASSERT_TRUE(empty_row.empty());
}

}  // namespace

TEST(Select, Time) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // check DATE Formats
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", dt)));
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('10/09/1999' AS DATE) > o;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('10-Sep-99' AS DATE) > o;", dt)));
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('31/Oct/2013' AS DATE) > o;", dt)));
    // check TIME FORMATS
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('151315' AS TIME) > n;", dt)));

    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) > o;", dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE CAST('1999-09-10' AS DATE) <= o;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) > n;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE CAST('15:13:15' AS TIME) <= n;", dt)));
    cta("SELECT DATETIME('NOW') FROM test limit 1;", dt);
    // these next tests work because all dates are before now 2015-12-8 17:00:00
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m < NOW();", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE m > timestamp(0) '2014-12-13T000000';",
                  dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST(o AS "
                                        "TIMESTAMP) > timestamp(0) '1999-09-08T160000';",
                                        dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CAST(o AS "
                                        "TIMESTAMP) > timestamp(0) '1999-09-10T160000';",
                                        dt)));
    ASSERT_EQ(14185957950L,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(EPOCH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(20140,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(YEAR FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(120,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(EXTRACT(MONTH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(140,
              v<int64_t>(
                  run_simple_agg("SELECT MAX(EXTRACT(DAY FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(
        22,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        23,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        6, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOW FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        348,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOY FROM m)) FROM test;", dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM n)) FROM test;", dt)));
    ASSERT_EQ(
        13,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM n)) FROM test;", dt)));
    ASSERT_EQ(
        14,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM n)) FROM test;", dt)));
    ASSERT_EQ(
        1999,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM o)) FROM test;", dt)));
    ASSERT_EQ(
        9,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM o)) FROM test;", dt)));
    ASSERT_EQ(
        9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM o)) FROM test;", dt)));
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(DOW FROM o) FROM test WHERE o IS NOT NULL;", dt)));
    ASSERT_EQ(252,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(DOY FROM o) FROM test WHERE o IS NOT NULL;", dt)));
    ASSERT_EQ(
        936835200L,
        v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM o)) FROM test;", dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK FROM CAST('2012-01-01 "
                                        "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(10L,
              v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(WEEK FROM CAST('2008-03-03 "
                                        "20:15:12' AS TIMESTAMP))) FROM test limit 1;",
                                        dt)));
    // Monday
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM CAST('2008-03-03 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    // Monday
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(ISODOW FROM CAST('2008-03-03 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    // Sunday
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM CAST('2008-03-02 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    // Sunday
    ASSERT_EQ(7L,
              v<int64_t>(run_simple_agg("SELECT EXTRACT(ISODOW FROM CAST('2008-03-02 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));

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
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(year, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1335830400L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(month, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1336435200L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(day, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1336507200L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(hour, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(second, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(978307200L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(millennium, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(978307200L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(century, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1293840000L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(decade, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(millisecond, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1336508112L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(microsecond, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1336262400L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(week, CAST('2012-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));

    ASSERT_EQ(-2114380800L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(year, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2104012800L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(month, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2103408000L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(day, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2103336000L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(hour, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(second, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-30578688000L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(millennium, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2177452800L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(century, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2177452800L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(decade, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(millisecond, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2103335088L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(microsecond, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2103840000L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(week, CAST('1903-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));

    ASSERT_EQ(31536000L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(decade, CAST('1972-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(662688000L,
              v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(decade, CAST('2000-05-08 "
                                        "20:15:12' AS TIMESTAMP)) FROM test limit 1;",
                                        dt)));
    // test QUARTER
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg("select EXTRACT(quarter FROM CAST('2008-11-27 "
                                        "12:12:12' AS timestamp)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("select EXTRACT(quarter FROM CAST('2008-03-21 "
                                        "12:12:12' AS timestamp)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1199145600L,
              v<int64_t>(run_simple_agg("select DATE_TRUNC(quarter, CAST('2008-03-21 "
                                        "12:12:12' AS timestamp)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1230768000L,
              v<int64_t>(run_simple_agg("select DATE_TRUNC(quarter, CAST('2009-03-21 "
                                        "12:12:12' AS timestamp)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1254355200L,
              v<int64_t>(run_simple_agg("select DATE_TRUNC(quarter, CAST('2009-11-21 "
                                        "12:12:12' AS timestamp)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(946684800L,
              v<int64_t>(run_simple_agg("select DATE_TRUNC(quarter, CAST('2000-03-21 "
                                        "12:12:12' AS timestamp)) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-2208988800L,
              v<int64_t>(run_simple_agg("select DATE_TRUNC(quarter, CAST('1900-03-21 "
                                        "12:12:12' AS timestamp)) FROM test limit 1;",
                                        dt)));
    // test DATE format processing
    ASSERT_EQ(1434844800L,
              v<int64_t>(run_simple_agg(
                  "select CAST('2015-06-21' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE o < CAST('06/21/2015' AS DATE);", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE o < CAST('21-Jun-15' AS DATE);", dt)));
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE o < CAST('21/Jun/2015' AS DATE);", dt)));
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE o < CAST('1434844800' AS DATE);", dt)));

    // test different input formats
    // added new format for customer
    ASSERT_EQ(
        1434896116L,
        v<int64_t>(run_simple_agg(
            "select CAST('2015-06-21 14:15:16' AS timestamp) FROM test limit 1;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('2015-06-21:141516' AS TIMESTAMP);",
                                        dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= CAST('21-JUN-15 "
                                  "2.15.16.12345 PM' AS TIMESTAMP);",
                                  dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= CAST('21-JUN-15 "
                                  "2.15.16.12345 AM' AS TIMESTAMP);",
                                  dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('21-JUN-15 2:15:16 AM' AS TIMESTAMP);",
                                        dt)));

    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('06/21/2015 14:15:16' AS TIMESTAMP);",
                                        dt)));

    // Support ISO date offset format
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                  "CAST('21/Aug/2015:12:13:14 -0600' AS TIMESTAMP);",
                                  dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('2015-08-21T12:13:14 -0600' AS TIMESTAMP);",
                                        dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('21-Aug-15 12:13:14 -0600' AS TIMESTAMP);",
                                        dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                  "CAST('21/Aug/2015:13:13:14 -0500' AS TIMESTAMP);",
                                  dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m <= "
                                        "CAST('2015-08-21T18:13:14' AS TIMESTAMP);",
                                        dt)));
    // add test for quarterday behaviour
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T04:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T00:00:00' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T08:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T14:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "select EXTRACT (QUARTERDAY FROM CAST('2015-08-21T23:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1440115200L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T04:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1440136800L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T08:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1440158400L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T13:23:11' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(1440180000L,
              v<int64_t>(run_simple_agg(
                  "select DATE_TRUNC (QUARTERDAY, CAST('2015-08-21T23:59:59' AS "
                  "timestamp)) FROM test limit 1;",
                  dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT DATEPART('year', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(2007,
              v<int64_t>(run_simple_agg("SELECT DATEPART('yyyy', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        2007,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('yy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(4,
              v<int64_t>(run_simple_agg("SELECT DATEPART('quarter', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('qq', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        4,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('q', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(10,
              v<int64_t>(run_simple_agg("SELECT DATEPART('month', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('mm', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('m', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(303,
              v<int64_t>(run_simple_agg("SELECT DATEPART('dayofyear', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        303,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('dy', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        303,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('y', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('day', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('dd', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        30,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('d', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(12,
              v<int64_t>(run_simple_agg("SELECT DATEPART('hour', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('hh', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT DATEPART('minute', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('mi', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('n', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(32,
              v<int64_t>(run_simple_agg("SELECT DATEPART('second', CAST('2007-10-30 "
                                        "12:15:32' AS TIMESTAMP)) FROM test;",
                                        dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('ss', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('s', CAST('2007-10-30 12:15:32' AS TIMESTAMP)) FROM test;",
            dt)));
    ASSERT_EQ(
        32,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('s', TIMESTAMP '2007-10-30 12:15:32') FROM test;", dt)));
    ASSERT_EQ(
        3,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('year', CAST('2006-01-07 00:00:00' as "
                                  "TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        36,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('month', CAST('2006-01-07 00:00:00' "
                                  "as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1096,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('day', CAST('2006-01-07 00:00:00' as "
                                  "TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg("SELECT DATEDIFF('quarter', CAST('2006-01-07 00:00:00' "
                                  "as TIMESTAMP), CAST('2009-01-07 00:00:00' AS "
                                  "TIMESTAMP)) FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('day', DATE '2009-2-28', DATE "
                                        "'2009-03-01') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('day', DATE '2008-2-28', DATE "
                                        "'2008-03-01') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(-425,
              v<int64_t>(run_simple_agg("select DATEDIFF('day', DATE '1971-03-02', DATE "
                                        "'1970-01-01') from test limit 1;",
                                        dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', o, o + INTERVAL '1' DAY) FROM TEST LIMIT 1;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT count(*) from test where DATEDIFF('day', "
                                        "CAST (m AS DATE), o) < -5570;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('second', m, TIMESTAMP(0) "
                                        "'2014-12-13 22:23:16') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('millisecond', m, TIMESTAMP(0) "
                                        "'2014-12-13 22:23:16') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(44000000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('microsecond', m, TIMESTAMP(0) "
                                        "'2014-12-13 22:23:59') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(34000000000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('nanosecond', m, TIMESTAMP(0) "
                                        "'2014-12-13 22:23:49') FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-1000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('millisecond', TIMESTAMP(0) "
                                        "'2014-12-13 22:23:16', m) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-44000000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('microsecond', TIMESTAMP(0) "
                                        "'2014-12-13 22:23:59', m) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(-34000000000,
              v<int64_t>(run_simple_agg("SELECT DATEDIFF('nanosecond', TIMESTAMP(0) "
                                        "'2014-12-13 22:23:49', m) FROM test limit 1;",
                                        dt)));
    // DATEADD tests
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, CAST('2017-05-31' AS DATE)) "
                                  "= TIMESTAMP '2017-06-01 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', 2, DATE '2017-05-31') = "
                                  "TIMESTAMP '2017-06-02 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', -1, CAST('2017-05-31' AS DATE)) "
                                  "= TIMESTAMP '2017-05-30 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('day', -2, DATE '2017-05-31') = "
                                  "TIMESTAMP '2017-05-29 0:00:00' from test limit 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('hour', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
                  "'2017-05-31 2:11:11' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('hour', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 11:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('hour', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 0:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('hour', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-30 15:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('minute', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:12:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('minute', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:21:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('minute', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:10:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('minute', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:01:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('second', 1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:11:12' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('second', 10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:11:21' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('second', -1, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:11:10' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('second', -10, TIMESTAMP '2017-05-31 1:11:11') = TIMESTAMP "
            "'2017-05-31 1:11:01' from test limit 1;",
            dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 1, DATE '2017-01-10') = TIMESTAMP "
                  "'2017-02-10 0:00:00' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 10, DATE '2017-01-10') = TIMESTAMP "
                  "'2017-11-10 0:00:00' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 1, DATE '2009-01-30') = TIMESTAMP "
                  "'2009-02-28 0:00:00' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 1, DATE '2008-01-30') = TIMESTAMP "
                  "'2008-02-29 0:00:00' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('month', 1, TIMESTAMP '2009-01-30 1:11:11') = TIMESTAMP "
            "'2009-02-28 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('month', -1, TIMESTAMP '2009-03-30 1:11:11') = TIMESTAMP "
            "'2009-02-28 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('month', -4, TIMESTAMP '2009-03-30 1:11:11') = TIMESTAMP "
            "'2008-11-30 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('month', 5, TIMESTAMP '2009-01-31 1:11:11') = TIMESTAMP "
            "'2009-6-30 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('year', 1, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
                  "'2009-02-28 1:11:11' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPADD(YEAR, 1, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
            "'2009-02-28 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPADD(YEAR, -8, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
            "'2000-02-29 1:11:11' from test limit 1;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPADD(YEAR, -8, TIMESTAMP '2008-02-29 1:11:11') = TIMESTAMP "
            "'2000-02-29 1:11:11' from test limit 1;",
            dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT m = TIMESTAMP '2014-12-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, m) = TIMESTAMP "
                                        "'2014-12-14 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -1, m) = TIMESTAMP "
                                        "'2014-12-12 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, m) = TIMESTAMP "
                                        "'2014-12-14 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -1, m) = TIMESTAMP "
                                        "'2014-12-12 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT o = DATE '1999-09-09' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 1, o) = TIMESTAMP "
                                        "'1999-09-10 0:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -3, o) = TIMESTAMP "
                                        "'1999-09-06 0:00:00' from test limit 1;",
                                        dt)));
    /* DATE ADD subseconds to default timestamp(0) */
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('millisecond', 1000, m) = TIMESTAMP "
                                  "'2014-12-13 22:23:16' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('microsecond', 1000000, m) = TIMESTAMP "
                                  "'2014-12-13 22:23:16' from test limit 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('nanosecond', 1000000000, m) = TIMESTAMP "
                  "'2014-12-13 22:23:16' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT DATEADD('millisecond', 5123, m) = TIMESTAMP "
                                  "'2014-12-13 22:23:20' from test limit 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('microsecond', 86400000000, m) = TIMESTAMP "
                  "'2014-12-14 22:23:15' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('nanosecond', 86400000000123, m) = TIMESTAMP "
                  "'2014-12-14 22:23:15' from test limit 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('weekday', -3, o) = TIMESTAMP "
                                        "'1999-09-06 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('decade', 3, o) = TIMESTAMP "
                                        "'2029-09-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('week', 1, o) = TIMESTAMP "
                                        "'1999-09-16 00:00:00' from test limit 1;",
                                        dt)));

    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, 1, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-03 1:23:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, -1, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-01 1:23:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, 15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-17 1:23:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(DAY, -15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-02-15 1:23:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, 1, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 2:23:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, -1, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 0:23:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, 15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 16:23:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(HOUR, -15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-01 10:23:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(MINUTE, 15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 1:38:45' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT TIMESTAMPADD(MINUTE, -15, TIMESTAMP '2009-03-02 "
                                 "1:23:45') = TIMESTAMP '2009-03-02 1:08:45' "
                                 "FROM TEST LIMIT 1;",
                                 dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SECOND, 15, TIMESTAMP '2009-03-02 "
                                  "1:23:45') = TIMESTAMP '2009-03-02 1:24:00' "
                                  "FROM TEST LIMIT 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT TIMESTAMPADD(SECOND, -15, TIMESTAMP '2009-03-02 "
                                 "1:23:45') = TIMESTAMP '2009-03-02 1:23:30' "
                                 "FROM TEST LIMIT 1;",
                                 dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, 1, m) = TIMESTAMP '2014-12-14 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, -1, m) = TIMESTAMP '2014-12-12 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, 15, m) = TIMESTAMP '2014-12-28 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, -15, m) = TIMESTAMP '2014-11-28 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 1, m) = TIMESTAMP '2014-12-13 23:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, -1, m) = TIMESTAMP '2014-12-13 21:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 15, m) = TIMESTAMP '2014-12-14 13:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, -15, m) = TIMESTAMP '2014-12-13 7:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, 15, m) = TIMESTAMP '2014-12-13 22:38:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, -15, m) = TIMESTAMP '2014-12-13 22:08:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, 15, m) = TIMESTAMP '2014-12-13 22:23:30' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, -15, m) = TIMESTAMP '2014-12-13 22:23:00' "
                  "FROM TEST LIMIT 1;",
                  dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, 1, m) = TIMESTAMP '2015-01-13 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, -1, m) = TIMESTAMP '2014-11-13 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, 5, m) = TIMESTAMP '2015-05-13 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY, -5, m) = TIMESTAMP '2014-12-08 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR, 1, m) = TIMESTAMP '2015-12-13 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR, -1, m) = TIMESTAMP '2013-12-13 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR, 5, m) = TIMESTAMP '2019-12-13 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR, -5, m) = TIMESTAMP '2009-12-13 22:23:15' "
                  "FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg("select count(*) from test where TIMESTAMPADD(YEAR, "
                                  "15, CAST(o AS TIMESTAMP)) > m;",
                                  dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("select count(*) from test where TIMESTAMPADD(YEAR, "
                                  "16, CAST(o AS TIMESTAMP)) > m;",
                                  dt)));

    ASSERT_EQ(
        128885,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(minute, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
            "'2003-05-01 12:05:55') FROM TEST LIMIT 1;",
            dt)));
    ASSERT_EQ(2148,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPDIFF(hour, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                  "'2003-05-01 12:05:55') FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(89,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPDIFF(day, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                  "'2003-05-01 12:05:55') FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(3,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPDIFF(month, TIMESTAMP '2003-02-01 0:00:00', TIMESTAMP "
                  "'2003-05-01 12:05:55') FROM TEST LIMIT 1;",
                  dt)));
    ASSERT_EQ(
        -3,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(month, TIMESTAMP '2003-05-01 12:05:55', TIMESTAMP "
            "'2003-02-01 0:00:00') FROM TEST LIMIT 1;",
            dt)));
    ASSERT_EQ(
        5,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(month, m, m + INTERVAL '5' MONTH) FROM TEST LIMIT 1;",
            dt)));
    ASSERT_EQ(
        -5,
        v<int64_t>(run_simple_agg(
            "SELECT TIMESTAMPDIFF(month, m, m - INTERVAL '5' MONTH) FROM TEST LIMIT 1;",
            dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("select count(*) from test where TIMESTAMPDIFF(YEAR, "
                                  "m, CAST(o AS TIMESTAMP)) < 0;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(year, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(14,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(month, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(426,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(day, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(60,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(week, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(613440,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(minute, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(10224,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(hour, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM TEST LIMIT 1;",
                                        dt)));
    ASSERT_EQ(36806400,
              v<int64_t>(run_simple_agg("SELECT TIMESTAMPDIFF(second, DATE '2018-01-02', "
                                        "DATE '2019-03-04') FROM TEST LIMIT 1;",
                                        dt)));

    ASSERT_EQ(
        1418428800L,
        v<int64_t>(run_simple_agg("SELECT CAST(m AS date) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(1336435200L,
              v<int64_t>(run_simple_agg("SELECT CAST(CAST('2012-05-08 20:15:12' AS "
                                        "TIMESTAMP) AS DATE) FROM test LIMIT 1;",
                                        dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test GROUP BY CAST(m AS date);", dt)));
    const auto rows = run_multiple_agg(
        "SELECT DATE_TRUNC(month, CAST(o AS TIMESTAMP(0))) AS key0, str AS key1, "
        "COUNT(*) AS val FROM test GROUP BY "
        "key0, key1 ORDER BY val DESC, key1;",
        dt);
    check_date_trunc_groups(*rows);
    const auto one_row = run_multiple_agg(
        "SELECT DATE_TRUNC(year, CASE WHEN str = 'foo' THEN m END) d FROM test GROUP BY "
        "d "
        "HAVING d IS NOT NULL;",
        dt);
    check_one_date_trunc_group(*one_row, 1388534400);
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test where "
                                        "DATE '2017-05-30' = DATE '2017-05-31' OR "
                                        "DATE '2017-05-31' = DATE '2017-05-30';",
                                        dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test where "
                                        "EXTRACT(DOW from TIMESTAMPADD(HOUR, -5, "
                                        "TIMESTAMP '2017-05-31 1:11:11')) = 1 OR "
                                        "EXTRACT(DOW from TIMESTAMPADD(HOUR, -5, "
                                        "TIMESTAMP '2017-05-31 1:11:11')) = 2;",
                                        dt)));
    std::vector<std::tuple<std::string, int64_t, int64_t>> date_trunc_queries{
        /*TIMESTAMP(0) */
        std::make_tuple("year, m", 1388534400L, 20),
        std::make_tuple("month, m", 1417392000L, 20),
        std::make_tuple("day, m", 1418428800L, 15),
        std::make_tuple("hour, m", 1418508000L, 15),
        std::make_tuple("minute, m", 1418509380L, 15),
        std::make_tuple("second, m", 1418509395L, 15),
        std::make_tuple("millennium, m", 978307200L, 20),
        std::make_tuple("century, m", 978307200L, 20),
        std::make_tuple("decade, m", 1293840000L, 20),
        std::make_tuple("week, m", 1417910400L, 15),
        std::make_tuple("nanosecond, m", 1418509395L, 15),
        std::make_tuple("microsecond, m", 1418509395L, 15),
        std::make_tuple("millisecond, m", 1418509395L, 15),
#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
        /* TIMESTAMP(3) */
        std::make_tuple("year, m_3", 1388534400000L, 20),
        std::make_tuple("month, m_3", 1417392000000L, 20),
        std::make_tuple("day, m_3", 1418428800000L, 15),
        std::make_tuple("hour, m_3", 1418508000000L, 15),
        std::make_tuple("minute, m_3", 1418509380000L, 15),
        std::make_tuple("second, m_3", 1418509395000L, 15),
        std::make_tuple("millennium, m_3", 978307200000L, 20),
        std::make_tuple("century, m_3", 978307200000L, 20),
        std::make_tuple("decade, m_3", 1293840000000L, 20),
        std::make_tuple("week, m_3", 1417910400000L, 15),
        std::make_tuple("nanosecond, m_3", 1418509395323L, 15),
        std::make_tuple("microsecond, m_3", 1418509395323L, 15),
        std::make_tuple("millisecond, m_3", 1418509395323L, 15),
        /* TIMESTAMP(6) */
        std::make_tuple("year, m_6", 915148800000000L, 10),
        std::make_tuple("month, m_6", 930787200000000L, 10),
        std::make_tuple("day, m_6", 931651200000000L, 10),
        std::make_tuple("hour, m_6", 931701600000000L, 10),
        /* std::make_tuple("minute, m_6", 931701720000000L, 10), // Exception with sort
           watchdog */
        std::make_tuple("second, m_6", 931701773000000L, 10),
        std::make_tuple("millennium, m_6", -30578688000000000L, 10),
        std::make_tuple("century, m_6", -2177452800000000L, 10),
        std::make_tuple("decade, m_6", 662688000000000L, 10),
        std::make_tuple("week, m_6", 931651200000000L, 10),
        std::make_tuple("nanosecond, m_6", 931701773874533L, 10),
        std::make_tuple("microsecond, m_6", 931701773874533L, 10),
        std::make_tuple("millisecond, m_6", 931701773874000L, 10),
        /* TIMESTAMP(9) */
        std::make_tuple("year, m_9", 1136073600000000000L, 10),
        std::make_tuple("month, m_9", 1143849600000000000L, 10),
        std::make_tuple("day, m_9", 1146009600000000000L, 10),
        std::make_tuple("hour, m_9", 1146020400000000000L, 10),
        /* std::make_tuple("minute, m_9", 1146023340000000000L, 10), // Exception with
           sort watchdog */
        std::make_tuple("second, m_9", 1146023344000000000L, 10),
        std::make_tuple("millennium, m_9", 978307200000000000L, 20),
        std::make_tuple("century, m_9", 978307200000000000L, 20),
        std::make_tuple("decade, m_9", 978307200000000000L, 10),
        std::make_tuple("week, m_9", 1145750400000000000L, 10),
        std::make_tuple("nanosecond, m_9", 1146023344607435125L, 10),
        std::make_tuple("microsecond, m_9", 1146023344607435000L, 10),
        std::make_tuple("millisecond, m_9", 1146023344607000000L, 10)
#endif
    };
    for (auto& query : date_trunc_queries) {
      const auto one_row = run_multiple_agg(
          "SELECT date_trunc(" + std::get<0>(query) +
              ") as key0,COUNT(*) AS val FROM test group by key0 order by key0 "
              "limit 1;",
          dt);
      check_one_date_trunc_group_with_agg(
          *one_row, std::get<1>(query), std::get<2>(query));
    }
    // Compressed DATE - limits test
    ASSERT_EQ(4708022400L,
              v<int64_t>(run_simple_agg(
                  "select CAST('2119-03-12' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(7998912000L,
              v<int64_t>(run_simple_agg("select CAST(CAST('2223-06-24 23:13:57' AS "
                                        "TIMESTAMP) AS DATE) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', 411, o) = TIMESTAMP "
                                        "'2410-09-12 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', -399, o) = TIMESTAMP "
                                        "'1600-08-31 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 6132, o) = TIMESTAMP "
                                        "'2510-09-13 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', -1100, o) = TIMESTAMP "
                                        "'1908-01-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 312456, o) = TIMESTAMP "
                                        "'2855-03-01 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -23674, o) = TIMESTAMP "
                                        "'1934-11-15 00:00:00' from test limit 1 ;",
                                        dt)));
    ASSERT_EQ(
        -303,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('year', DATE '2302-04-21', o) from test limit 1;", dt)));
    ASSERT_EQ(
        502,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('year', o, DATE '2501-04-21') from test limit 1;", dt)));
    ASSERT_EQ(
        -4896,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('month', DATE '2407-09-01', o) from test limit 1;", dt)));
    ASSERT_EQ(
        3818,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('month', o, DATE '2317-11-01') from test limit 1;", dt)));
    ASSERT_EQ(
        -86972,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', DATE '2237-10-23', o) from test limit 1;", dt)));
    ASSERT_EQ(
        86972,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', o, DATE '2237-10-23') from test limit 1;", dt)));
    ASSERT_EQ(
        2617,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('year', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('month', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        23,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('day', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('hour', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('minute', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('second', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        6,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('weekday', CAST ('2011-12-31' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(365,
              v<int64_t>(run_simple_agg("SELECT DATEPART('dayofyear', CAST ('2011-12-31' "
                                        "as DATE)) from test limit 1;",
                                        dt)));
    // Compressed DATE - limits test
    ASSERT_EQ(4708022400L,
              v<int64_t>(run_simple_agg(
                  "select CAST('2119-03-12' AS DATE) FROM test limit 1;", dt)));
    ASSERT_EQ(7998912000L,
              v<int64_t>(run_simple_agg("select CAST(CAST('2223-06-24 23:13:57' AS "
                                        "TIMESTAMP) AS DATE) FROM test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', 411, o) = TIMESTAMP "
                                        "'2410-09-12 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('year', -399, o) = TIMESTAMP "
                                        "'1600-08-31 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', 6132, o) = TIMESTAMP "
                                        "'2510-09-13 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('month', -1100, o) = TIMESTAMP "
                                        "'1908-01-09 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', 312456, o) = TIMESTAMP "
                                        "'2855-03-01 00:00:00' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT DATEADD('day', -23674, o) = TIMESTAMP "
                                        "'1934-11-15 00:00:00' from test limit 1 ;",
                                        dt)));
    ASSERT_EQ(
        -303,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('year', DATE '2302-04-21', o) from test limit 1;", dt)));
    ASSERT_EQ(
        502,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('year', o, DATE '2501-04-21') from test limit 1;", dt)));
    ASSERT_EQ(
        -4896,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('month', DATE '2407-09-01', o) from test limit 1;", dt)));
    ASSERT_EQ(
        3818,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('month', o, DATE '2317-11-01') from test limit 1;", dt)));
    ASSERT_EQ(
        -86972,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', DATE '2237-10-23', o) from test limit 1;", dt)));
    ASSERT_EQ(
        86972,
        v<int64_t>(run_simple_agg(
            "SELECT DATEDIFF('day', o, DATE '2237-10-23') from test limit 1;", dt)));
    ASSERT_EQ(
        2617,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('year', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        12,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('month', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        23,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('day', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('hour', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('minute', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT DATEPART('second', CAST ('2617-12-23' as DATE)) from test limit 1;",
            dt)));
    /* Compressed Date ColumnarResults fetch tests*/
    ASSERT_EQ(1999,
              v<int64_t>(run_simple_agg("select yr from (SELECT EXTRACT(year from o) as "
                                        "yr, o from test order by x) limit 1;",
                                        dt)));
    ASSERT_EQ(936835200,
              v<int64_t>(run_simple_agg("select dy from (SELECT DATE_TRUNC(day, o) as "
                                        "dy, o from test order by x) limit 1;",
                                        dt)));
    ASSERT_EQ(936921600,
              v<int64_t>(run_simple_agg("select dy from (SELECT DATEADD('day', 1, o) as "
                                        "dy, o from test order by x) limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "select dy from (SELECT DATEDIFF('day', o, DATE '1999-09-10') as dy, o "
                  "from test order by x) limit 1;",
                  dt)));
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
    c("SELECT COUNT(*) FROM test WHERE real_str IN ('real_foo', 'real_bar', 'real_baz', "
      "'foo');",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str IN ('foo', 'bar', 'real_foo');", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, "
      "14, 15, 16, 17, 18, 19, 20);",
      dt);
  }
}

TEST(Select, DivByZero) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT x / 0 FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT 1 / 0 FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct x / 0) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT d / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / (f - f) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY y / (x - x);", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY z, y / (x - x);", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY MOD(y , (x - x));", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT SUM(x) / SUM(CASE WHEN str = 'none' THEN y ELSE 0 END) FROM test;",
            dt),
        std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT COUNT(*) FROM test WHERE y / (x - x) = 0;", dt),
                 std::runtime_error);
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE x = x OR  y / (x - x) = y;", dt)));
  }
}

TEST(Select, ReturnNullFromDivByZero) {
  SKIP_ALL_ON_AGGREGATOR();

  g_null_div_by_zero = true;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x / 0 FROM test;", dt);
    c("SELECT 1 / 0 FROM test;", dt);
    c("SELECT f / 0. FROM test;", dt);
    c("SELECT d / 0. FROM test;", dt);
    c("SELECT f / (f - f) FROM test;", dt);
    c("SELECT COUNT(*) FROM test GROUP BY y / (x - x);", dt);
    c("SELECT COUNT(*) n FROM test GROUP BY z, y / (x - x) ORDER BY n ASC;", dt);
    c("SELECT SUM(x) / SUM(CASE WHEN str = 'none' THEN y ELSE 0 END) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE y / (x - x) = 0;", dt);
    c("SELECT COUNT(*) FROM test WHERE x = x OR  y / (x - x) = y;", dt);
  }
}

TEST(Select, ConstantFolding) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT 1 + 2 FROM test limit 1;", dt);
    c("SELECT 1 + 2.3 FROM test limit 1;", dt);
    c("SELECT 2.3 + 1 FROM test limit 1;", dt);
    c("SELECT 2 * 3 FROM test limit 1;", dt);
    c("SELECT 604 * 575 FROM test limit 1;", dt);
    c("SELECT 604 * (75 + 500) FROM test limit 1;", dt);
    c("SELECT 604 * (5 * 115) FROM test limit 1;", dt);
    c("SELECT 100000 + (1 - 604 * 575) FROM test limit 1;", dt);
    c("SELECT 1 + 604 * 575 FROM test limit 1;", dt);
    c("SELECT 2 + (1 - 604 * 575) FROM test limit 1;", dt);
    c("SELECT t + 604 * 575 FROM test limit 1;", dt);  // mul is folded in BIGINT
    EXPECT_THROW(run_simple_agg("SELECT z + 604 * 575 FROM test limit 1;", dt),
                 std::runtime_error);  // z is SMALLINT, mul overflows
    c("SELECT 9.1 + 2.9999999999 FROM test limit 1;", dt);
    c("SELECT -9.1 - 2.9999999999 FROM test limit 1;", dt);
    c("SELECT -(9.1 + 99.22) FROM test limit 1;", dt);
    c("SELECT 3/2 FROM test limit 1;", dt);
    c("SELECT 3/2.0 FROM test limit 1;", dt);
    c("SELECT 11.1 * 2.22 FROM test limit 1;", dt);
    c("SELECT 1.01 * 1.00001 FROM test limit 1;", dt);
    c("SELECT 11.1 * 2.222222222 FROM test limit 1;", dt);
    c("SELECT 9.99 * 9999.9 FROM test limit 1;", dt);
    c("SELECT 9.22337203685477 * 9.223 FROM test limit 1;", dt);
    c("SELECT 3.0+8 from test limit 1;", dt);
    c("SELECT 3.0*8 from test limit 1;", dt);
    c("SELECT 1.79769e+308 * 0.1 FROM test limit 1;", dt);
    c("SELECT COUNT(*) FROM test WHERE 3.0+8 < 30;", dt);
    c("SELECT COUNT(*) FROM test WHERE 3.0*8 > 30.01;", dt);
    c("SELECT COUNT(*) FROM test WHERE 3.0*8 > 30.0001;", dt);
    c("SELECT COUNT(*) FROM test WHERE ff + 3.0*8 < 60.0/2;", dt);
    c("SELECT COUNT(*) FROM test WHERE t > 0 AND t = t;", dt);
    c("SELECT COUNT(*) FROM test WHERE t > 0 AND t <> t;", dt);
    c("SELECT COUNT(*) FROM test WHERE t > 0 OR t = t;", dt);
    c("SELECT COUNT(*) FROM test WHERE t > 0 OR t <> t;", dt);
    c("SELECT COUNT(*) FROM test where (604=575) OR (33.0<>12 AND 2.0001e+4>20000.9) "
      "OR (NOT t>=t OR f<>f OR (x=x AND x-x=0));",
      dt);
  }
}

TEST(Select, OverflowAndUnderFlow) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE z + 32600 > 0;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE z + 32666 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE -32670 - z < 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE (z + 16333) * 2 > 0;", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE x + 2147483640 > 0;", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE -x - 2147483642 < 0;", dt),
        std::runtime_error);
    c("SELECT COUNT(*) FROM test WHERE t + 9223372036854774000 > 0;", dt);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT COUNT(*) FROM test WHERE t + 9223372036854775000 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT COUNT(*) FROM test WHERE -t - 9223372036854775000 < 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofd + x - 2 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg(
                     "SELECT COUNT(*) FROM test WHERE ufd * 3 - ofd * 1024 < -2;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofd * 2 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test WHERE ofq + 1 > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT COUNT(*) FROM test WHERE -ufq - 9223372036854775000 > 0;", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT COUNT(*) FROM test WHERE -92233720368547758 - ofq <= 0;",
                         dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT cast((z - -32666) *0.000190 as int) as key0, "
                         "COUNT(*) AS val FROM test WHERE (z >= -32666 AND z < 31496) "
                         "GROUP BY key0 HAVING key0 >= 0 AND key0 < 12 ORDER BY val "
                         "DESC LIMIT 50 OFFSET 0;",
                         dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT dd * 2000000000000000 FROM test LIMIT 5;", dt),
                 std::runtime_error);
    c("SELECT dd * 200000000000000 FROM test ORDER BY dd ASC LIMIT 5;",
      dt);  // overflow avoided through decimal mul optimization
    c("SELECT COUNT(*) FROM test WHERE dd + 2.0000000000000009 > 110.0;",
      dt);  // no overflow in the cast
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT COUNT(*) FROM test WHERE dd + 2.00000000000000099 > 110.0;", dt),
        std::runtime_error);  // overflow in the cast due to higher precision
    c("SELECT dd / 2.00000009 FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // dividend still fits after cast and division upscaling
    EXPECT_THROW(run_multiple_agg("SELECT dd / 2.000000099 FROM test LIMIT 1;", dt),
                 std::runtime_error);  // dividend overflows after cast and division
                                       // upscaling due to higher precision
    c("SELECT (dd - 40.6364668888) / 2 FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // decimal div by const optimization avoids overflow
    c("SELECT (dd - 40.6364668888) / x FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // decimal div by int cast optimization avoids overflow
    c("SELECT (dd - 40.63646688) / dd FROM test ORDER BY dd ASC LIMIT 1;",
      dt);  // dividend still fits after upscaling from cast and division
    EXPECT_THROW(run_multiple_agg("select (dd-40.6364668888)/dd from test limit 1;", dt),
                 std::runtime_error);  // dividend overflows on upscaling on a slightly
                                       // higher precision, test detection
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
    // truncate literals based on the other side's precision, e.g. for d which is
    // DECIMAL(14,2)
    c("select count(*) from big_decimal_range_test where (d >  4.955357142857142);",
      dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d >= 4.955357142857142);",
      dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d <  4.955357142857142);",
      dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d <= 4.955357142857142);",
      dt);  // compare with 4.955
    c("select count(*) from big_decimal_range_test where (d >= 4.950357142857142);",
      dt);  // compare with 4.951
    c("select count(*) from big_decimal_range_test where (d <  4.950357142857142);",
      dt);  // compare with 4.951
    c("select count(*) from big_decimal_range_test where (d < 59016609.300000056);",
      dt);  // compare with 59016609.301
    c("select count(*) from test where (t*123456 > 9681668.33071388567);",
      dt);  // compare with 9681668.3
    c("select count(*) from test where (x*12345678 < 9681668.33071388567);",
      dt);  // compare with 9681668.3
    c("select count(*) from test where (z*12345678 < 9681668.33071388567);",
      dt);  // compare with 9681668.3
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
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg("SELECT COUNT(CAST(EXTRACT(QUARTER FROM CAST(NULL AS "
                                  "TIMESTAMP)) AS BIGINT) - 1) FROM test;",
                                  dt)));
  }
}

TEST(Select, BooleanColumn) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE b;", dt)));
    ASSERT_EQ(g_num_rows / 2,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE NOT b;", dt)));
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE x < 8 AND b;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE x < 8 AND NOT b;", dt)));
    ASSERT_EQ(5,
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE x > 7 OR false;", dt)));
    ASSERT_EQ(7,
              v<int64_t>(run_simple_agg(
                  "SELECT MAX(x) FROM test WHERE b = CAST('t' AS boolean);", dt)));
    ASSERT_EQ(3 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  " SELECT SUM(2 *(CASE when x = 7 then 1 else 0 END)) FROM test;", dt)));
    c("SELECT COUNT(*) AS n FROM test GROUP BY x = 7, b ORDER BY n;", dt);
  }
}

TEST(Select, UnsupportedCast) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT CAST(x AS VARCHAR) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(f AS VARCHAR) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(d AS VARCHAR) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(f AS DECIMAL) FROM test;", dt),
                 std::runtime_error);
  }
}

TEST(Select, CastFromLiteral) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(2.3 AS TINYINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS SMALLINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS INT) FROM test;", dt);
    c("SELECT CAST(2.3 AS BIGINT) FROM test;", dt);
    c("SELECT CAST(2.3 AS FLOAT) FROM test;", dt);
    c("SELECT CAST(2.3 AS DOUBLE) FROM test;", dt);
    c("SELECT CAST(2.3 AS DECIMAL(2, 1)) FROM test;", dt);
    c("SELECT CAST(2.3 AS NUMERIC(2, 1)) FROM test;", dt);
    c("SELECT CAST(CAST(10 AS float) / CAST(3600 as float) AS float) FROM test LIMIT 1;",
      dt);
    c("SELECT CAST(CAST(10 AS double) / CAST(3600 as double) AS double) FROM test LIMIT "
      "1;",
      dt);
    c("SELECT z from test where z = -78;", dt);
  }
}

TEST(Select, CastFromNull) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(NULL AS TINYINT) FROM test;", dt);
    c("SELECT CAST(NULL AS SMALLINT) FROM test;", dt);
    c("SELECT CAST(NULL AS INT) FROM test;", dt);
    c("SELECT CAST(NULL AS BIGINT) FROM test;", dt);
    c("SELECT CAST(NULL AS FLOAT) FROM test;", dt);
    c("SELECT CAST(NULL AS DOUBLE) FROM test;", dt);
    c("SELECT CAST(NULL AS DECIMAL) FROM test;", dt);
    c("SELECT CAST(NULL AS NUMERIC) FROM test;", dt);
  }
}

TEST(Select, DropSecondaryDB) {
  run_ddl_statement("CREATE DATABASE SECONDARY_DB;");
  run_ddl_statement("DROP DATABASE SECONDARY_DB;");
}

TEST(Select, CastDecimalToDecimal) {
  run_ddl_statement("DROP TABLE IF EXISTS decimal_to_decimal_test;");
  run_ddl_statement("create table decimal_to_decimal_test (id INT, val DECIMAL(10,5));");
  run_multiple_agg("insert into decimal_to_decimal_test VALUES (1, 456.78956)",
                   ExecutorDeviceType::CPU);
  run_multiple_agg("insert into decimal_to_decimal_test VALUES (2, 456.12345)",
                   ExecutorDeviceType::CPU);

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_TRUE(
        approx_eq(456.78956,
                  v<double>(run_simple_agg(
                      "SELECT val FROM decimal_to_decimal_test WHERE id = 1;", dt))));
    ASSERT_TRUE(
        approx_eq(456.12345,
                  v<double>(run_simple_agg(
                      "SELECT val FROM decimal_to_decimal_test WHERE id = 2;", dt))));

    ASSERT_TRUE(
        approx_eq(456.7896,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,4)) FROM "
                                           "decimal_to_decimal_test WHERE id = 1;",
                                           dt))));
    ASSERT_TRUE(
        approx_eq(456.123,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,4)) FROM "
                                           "decimal_to_decimal_test WHERE id = 2;",
                                           dt))));

    ASSERT_TRUE(
        approx_eq(456.790,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,3)) FROM "
                                           "decimal_to_decimal_test WHERE id = 1;",
                                           dt))));
    ASSERT_TRUE(
        approx_eq(456.1234,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,3)) FROM "
                                           "decimal_to_decimal_test WHERE id = 2;",
                                           dt))));

    ASSERT_TRUE(
        approx_eq(456.79,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,2)) FROM "
                                           "decimal_to_decimal_test WHERE id = 1;",
                                           dt))));
    ASSERT_TRUE(
        approx_eq(456.12,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,2)) FROM "
                                           "decimal_to_decimal_test WHERE id = 2;",
                                           dt))));

    ASSERT_TRUE(
        approx_eq(456.8,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,1)) FROM "
                                           "decimal_to_decimal_test WHERE id = 1;",
                                           dt))));
    ASSERT_TRUE(
        approx_eq(456.1,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,1)) FROM "
                                           "decimal_to_decimal_test WHERE id = 2;",
                                           dt))));
    ASSERT_TRUE(
        approx_eq(457,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,0)) FROM "
                                           "decimal_to_decimal_test WHERE id = 1;",
                                           dt))));
    ASSERT_TRUE(
        approx_eq(456,
                  v<double>(run_simple_agg("SELECT CAST(val AS DECIMAL(10,0)) FROM "
                                           "decimal_to_decimal_test WHERE id = 2;",
                                           dt))));

    ASSERT_EQ(457,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(val AS BIGINT) FROM decimal_to_decimal_test WHERE id = 1;",
                  dt)));
    ASSERT_EQ(456,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(val AS BIGINT) FROM decimal_to_decimal_test WHERE id = 2;",
                  dt)));
  }
}

TEST(Select, ColumnWidths) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT DISTINCT x FROM test_inner ORDER BY x;", dt);
    c("SELECT DISTINCT y FROM test_inner ORDER BY y;", dt);
    c("SELECT DISTINCT xx FROM test_inner ORDER BY xx;", dt);
    c("SELECT x, xx, y FROM test_inner GROUP BY x, xx, y ORDER BY x, xx, y;", dt);
    c("SELECT x, xx, y FROM test_inner GROUP BY x, xx, y ORDER BY x, xx, y;", dt);
    c("SELECT DISTINCT str from test_inner ORDER BY str;", dt);
    c("SELECT DISTINCT t FROM test ORDER BY t;", dt);
    c("SELECT DISTINCT t, z FROM test GROUP BY t, z ORDER BY t, z;", dt);
    c("SELECT fn from test where fn < -100.7 ORDER BY fn;", dt);
    c("SELECT fixed_str, SUM(f)/SUM(t)  FROM test WHERE fixed_str IN ('foo','bar') GROUP "
      "BY fixed_str ORDER BY "
      "fixed_str;",
      dt);
  }
}

TEST(Select, TimeInterval) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(
        60 * 60 * 1000L,
        v<int64_t>(run_simple_agg("SELECT INTERVAL '1' HOUR FROM test LIMIT 1;", dt)));
    ASSERT_EQ(
        24 * 60 * 60 * 1000L,
        v<int64_t>(run_simple_agg("SELECT INTERVAL '1' DAY FROM test LIMIT 1;", dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE INTERVAL '1' MONTH < INTERVAL '2' MONTH;",
            dt)));
    ASSERT_EQ(
        2 * g_num_rows,
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test WHERE INTERVAL '1' DAY < INTERVAL '2' DAY;", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test GROUP BY INTERVAL '1' DAY;", dt)));
    ASSERT_EQ(3 * 60 * 60 * 1000L,
              v<int64_t>(
                  run_simple_agg("SELECT 3 * INTERVAL '1' HOUR FROM test LIMIT 1;", dt)));
    ASSERT_EQ(3 * 60 * 60 * 1000L,
              v<int64_t>(
                  run_simple_agg("SELECT INTERVAL '1' HOUR * 3 FROM test LIMIT 1;", dt)));
    ASSERT_EQ(7L,
              v<int64_t>(run_simple_agg(
                  "SELECT INTERVAL '1' MONTH * x FROM test WHERE x <> 8 LIMIT 1;", dt)));
    ASSERT_EQ(7L,
              v<int64_t>(run_simple_agg(
                  "SELECT x * INTERVAL '1' MONTH FROM test WHERE x <> 8 LIMIT 1;", dt)));
    ASSERT_EQ(42L,
              v<int64_t>(run_simple_agg(
                  "SELECT INTERVAL '1' MONTH * y FROM test WHERE y <> 43 LIMIT 1;", dt)));
    ASSERT_EQ(42L,
              v<int64_t>(run_simple_agg(
                  "SELECT y * INTERVAL '1' MONTH FROM test WHERE y <> 43 LIMIT 1;", dt)));
    ASSERT_EQ(
        1002L,
        v<int64_t>(run_simple_agg(
            "SELECT INTERVAL '1' MONTH * t FROM test WHERE t <> 1001 LIMIT 1;", dt)));
    ASSERT_EQ(
        1002L,
        v<int64_t>(run_simple_agg(
            "SELECT t * INTERVAL '1' MONTH FROM test WHERE t <> 1001 LIMIT 1;", dt)));
    ASSERT_EQ(
        3L,
        v<int64_t>(run_simple_agg(
            "SELECT INTERVAL '1' MONTH + INTERVAL '2' MONTH FROM test LIMIT 1;", dt)));
    ASSERT_EQ(
        1388534400L,
        v<int64_t>(run_simple_agg("SELECT CAST(m AS date) + CAST(TRUNCATE(-1 * "
                                  "(EXTRACT(DOY FROM m) - 1), 0) AS INTEGER) * INTERVAL "
                                  "'1' DAY AS g FROM test GROUP BY g;",
                                  dt)));
    ASSERT_EQ(
        1417392000L,
        v<int64_t>(run_simple_agg("SELECT CAST(m AS date) + CAST(TRUNCATE(-1 * "
                                  "(EXTRACT(DAY FROM m) - 1), 0) AS INTEGER) * INTERVAL "
                                  "'1' DAY AS g FROM test GROUP BY g;",
                                  dt)));
    ASSERT_EQ(1418508000L,
              v<int64_t>(run_simple_agg("SELECT CAST(m AS date) + EXTRACT(HOUR FROM m) * "
                                        "INTERVAL '1' HOUR AS g FROM test GROUP BY g;",
                                        dt)));
    ASSERT_EQ(
        1388534400L,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SQL_TSI_DAY, CAST(TRUNCATE(-1 * "
                                  "(EXTRACT(DOY from m) - 1), 0) AS INTEGER), "
                                  "CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                                  dt)));
    ASSERT_EQ(
        1417392000L,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SQL_TSI_DAY, CAST(TRUNCATE(-1 * "
                                  "(EXTRACT(DAY from m) - 1), 0) AS INTEGER), "
                                  "CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                                  dt)));
    ASSERT_EQ(
        1418508000L,
        v<int64_t>(run_simple_agg("SELECT TIMESTAMPADD(SQL_TSI_HOUR, EXTRACT(HOUR from "
                                  "m), CAST(m AS DATE)) AS g FROM test GROUP BY g;",
                                  dt)));

    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' + INTERVAL '1' YEAR) = "
                                        "DATE '2009-01-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' + INTERVAL '5' YEAR) = "
                                        "DATE '2013-01-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' - INTERVAL '1' YEAR) = "
                                        "DATE '2007-01-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' - INTERVAL '4' YEAR) = "
                                        "DATE '2004-01-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' + INTERVAL '1' MONTH) "
                                        "= DATE '2008-02-29' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' + INTERVAL '5' MONTH) "
                                        "= DATE '2008-06-30' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' - INTERVAL '1' MONTH) "
                                        "= DATE '2007-12-31' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-1-31' - INTERVAL '4' MONTH) "
                                        "= DATE '2007-09-30' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-2-28' + INTERVAL '1' DAY) = "
                                        "DATE '2008-02-29' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2009-2-28' + INTERVAL '1' DAY) = "
                                        "DATE '2009-03-01' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-2-28' + INTERVAL '4' DAY) = "
                                        "DATE '2008-03-03' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2009-2-28' + INTERVAL '4' DAY) = "
                                        "DATE '2009-03-04' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-03-01' - INTERVAL '1' DAY) = "
                                        "DATE '2008-02-29' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2009-03-01' - INTERVAL '1' DAY) = "
                                        "DATE '2009-02-28' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2008-03-03' - INTERVAL '4' DAY) = "
                                        "DATE '2008-02-28' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (DATE '2009-03-04' - INTERVAL '4' DAY) = "
                                        "DATE '2009-02-28' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT m = TIMESTAMP '2014-12-13 22:23:15' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' SECOND) = TIMESTAMP "
                                        "'2014-12-13 22:23:16' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' MINUTE) = TIMESTAMP "
                                        "'2014-12-13 22:24:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' HOUR) = TIMESTAMP "
                                        "'2014-12-13 23:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '2' DAY) = TIMESTAMP "
                                        "'2014-12-15 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' MONTH) = TIMESTAMP "
                                        "'2015-01-13 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m + INTERVAL '1' YEAR) = TIMESTAMP "
                                        "'2015-12-13 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT (m - 5 * INTERVAL '1' SECOND) = TIMESTAMP "
                                  "'2014-12-13 22:23:10' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT (m - x * INTERVAL '1' MINUTE) = TIMESTAMP "
                                  "'2014-12-13 22:16:15' from test limit 1;",
                                  dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg("SELECT (m - 2 * x * INTERVAL '1' HOUR) = TIMESTAMP "
                                  "'2014-12-13 8:23:15' from test limit 1;",
                                  dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m - x * INTERVAL '1' DAY) = TIMESTAMP "
                                        "'2014-12-06 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m - x * INTERVAL '1' MONTH) = TIMESTAMP "
                                        "'2014-05-13 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg("SELECT (m - x * INTERVAL '1' YEAR) = TIMESTAMP "
                                        "'2007-12-13 22:23:15' from test limit 1;",
                                        dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (m - INTERVAL '5' DAY + INTERVAL '2' HOUR - x * INTERVAL '2' "
                  "SECOND) +"
                  "(x - 1) * INTERVAL '1' MONTH - x * INTERVAL '10' YEAR = "
                  "TIMESTAMP '1945-06-09 00:23:01' from test limit 1;",
                  dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "select count(*) from test where m < CAST (o AS TIMESTAMP) + INTERVAL '10' "
            "YEAR AND m > CAST(o AS TIMESTAMP) - INTERVAL '10' YEAR;",
            dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg(
            "select count(*) from test where m < CAST (o AS TIMESTAMP) + INTERVAL '16' "
            "YEAR AND m > CAST(o AS TIMESTAMP) - INTERVAL '16' YEAR;",
            dt)));

    ASSERT_EQ(1,
              v<int64_t>(
                  run_simple_agg("SELECT o = DATE '1999-09-09' from test limit 1;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT (o + INTERVAL '10' DAY) = DATE '1999-09-19' from test limit 1;",
                  dt)));
  }
}

TEST(Select, UnsupportedNodes) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT 1 + 2;", dt), std::runtime_error);
    // MAT No longer throws a logicalValues gets a regular parse error'
    // EXPECT_THROW(run_multiple_agg("SELECT *;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT x, COUNT(*) FROM test GROUP BY ROLLUP(x);", dt),
                 std::runtime_error);
  }
}

TEST(Select, UnsupportedMultipleArgAggregate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct x, y) FROM test;", dt),
                 std::runtime_error);
  }
}

namespace Importer_NS {

ArrayDatum StringToArray(const std::string& s,
                         const SQLTypeInfo& ti,
                         const CopyParams& copy_params);
void parseStringArray(const std::string& s,
                      const CopyParams& copy_params,
                      std::vector<std::string>& string_vec);

}  // namespace Importer_NS

namespace {

const size_t g_array_test_row_count{20};

std::unique_ptr<Importer_NS::Loader> get_loader(const TableDescriptor* td) {
  auto& cat = g_session->getCatalog();
  auto loader = std::make_unique<Importer_NS::Loader>(cat, td);
  return loader;
}

void import_array_test(const std::string& table_name) {
  CHECK_EQ(size_t(0), g_array_test_row_count % 4);
  auto& cat = g_session->getCatalog();
  const auto td = cat.getMetadataForTable(table_name);
  CHECK(td);
  auto loader = get_loader(td);
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  const auto col_descs =
      cat.getAllColumnMetadataForTable(td->tableId, false, false, false);
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
                  array_elems.emplace_back("T");
                  array_elems.emplace_back("F");
                } else {
                  array_elems.emplace_back("F");
                  array_elems.emplace_back("T");
                }
              }
              break;
            }
            case kTINYINT:
              for (size_t i = 0; i < 3; ++i) {
                array_elems.push_back(std::to_string(row_idx + i + 1));
              }
              break;
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
                array_elems.emplace_back(std::to_string(row_idx + i + 1) + "." +
                                         std::to_string(row_idx + i + 1));
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
  loader->load(import_buffers, g_array_test_row_count);
}

void import_gpu_sort_test() {
  const std::string drop_old_gpu_sort_test{"DROP TABLE IF EXISTS gpu_sort_test;"};
  run_ddl_statement(drop_old_gpu_sort_test);
  g_sqlite_comparator.query(drop_old_gpu_sort_test);
  run_ddl_statement("CREATE TABLE gpu_sort_test(x int) WITH (fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE gpu_sort_test(x int);");
  TestHelpers::ValuesGenerator gen("gpu_sort_test");
  for (size_t i = 0; i < 4; ++i) {
    const auto insert_query = gen(2);
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (size_t i = 0; i < 6; ++i) {
    const auto insert_query = gen(16000);
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_query_rewrite_test() {
  const std::string drop_old_query_rewrite_test{
      "DROP TABLE IF EXISTS query_rewrite_test;"};
  run_ddl_statement(drop_old_query_rewrite_test);
  g_sqlite_comparator.query(drop_old_query_rewrite_test);
  run_ddl_statement(
      "CREATE TABLE query_rewrite_test(x int, str text encoding dict) WITH "
      "(fragment_size=2);");
  g_sqlite_comparator.query("CREATE TABLE query_rewrite_test(x int, str text);");
  for (size_t i = 1; i <= 30; ++i) {
    for (size_t j = 1; j <= i % 2 + 1; ++j) {
      const std::string insert_query{"INSERT INTO query_rewrite_test VALUES(" +
                                     std::to_string(i) + ", 'str" + std::to_string(i) +
                                     "');"};
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
  }
}

void import_big_decimal_range_test() {
  const std::string drop_old_decimal_range_test(
      "DROP TABLE IF EXISTS big_decimal_range_test;");
  run_ddl_statement(drop_old_decimal_range_test);
  g_sqlite_comparator.query(drop_old_decimal_range_test);
  run_ddl_statement(
      "CREATE TABLE big_decimal_range_test(d DECIMAL(14, 2), d1 DECIMAL(17,11)) WITH "
      "(fragment_size=2);");
  g_sqlite_comparator.query(
      "CREATE TABLE big_decimal_range_test(d DECIMAL(14, 2), d1 DECIMAL(17,11));");
  {
    const std::string insert_query{
        "INSERT INTO big_decimal_range_test VALUES(-40840124.400000, 1.3);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{
        "INSERT INTO big_decimal_range_test VALUES(59016609.300000, 1.3);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{
        "INSERT INTO big_decimal_range_test VALUES(-999999999999.99, 1.3);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_decimal_compression_test() {
  const std::string decimal_compression_test(
      "DROP TABLE IF EXISTS decimal_compression_test;");
  run_ddl_statement(decimal_compression_test);
  g_sqlite_comparator.query(decimal_compression_test);
  run_ddl_statement(
      "CREATE TABLE decimal_compression_test(big_dec DECIMAL(17, 2), med_dec DECIMAL(9, "
      "2), small_dec DECIMAL(4, 2)) WITH (fragment_size=2);");
  g_sqlite_comparator.query(
      "CREATE TABLE decimal_compression_test(big_dec DECIMAL(17, 2), med_dec DECIMAL(9, "
      "2), small_dec DECIMAL(4, 2));");
  {
    const std::string insert_query{
        "INSERT INTO decimal_compression_test VALUES(999999999999999.99, 9999999.99, "
        "99.99);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{
        "INSERT INTO decimal_compression_test VALUES(-999999999999999.99, -9999999.99, "
        "-99.99);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{
        "INSERT INTO decimal_compression_test VALUES(12.2382, 12.2382 , 12.2382);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    // sqlite does not do automatic rounding
    const std::string sqlite_insert_query{
        "INSERT INTO decimal_compression_test VALUES(12.24, 12.24 , 12.24);"};
    g_sqlite_comparator.query(sqlite_insert_query);
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
      "CREATE TABLE text_group_by_test(tdef TEXT, tdict TEXT ENCODING DICT, tnone TEXT "
      "ENCODING NONE ) WITH "
      "(fragment_size=200);");
  const std::string insert_query{
      "INSERT INTO text_group_by_test VALUES('hello','world',':-)');"};
  run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
}

void import_join_test(bool with_delete_support) {
  const std::string drop_old_test{"DROP TABLE IF EXISTS join_test;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  std::string columns_definition{
      "x int not null, y int, str text encoding dict, dup_str text encoding dict"};
  const auto create_test =
      build_create_table_statement(columns_definition,
                                   "join_test",
                                   {g_shard_count ? "dup_str" : "", g_shard_count},
                                   {},
                                   2,
                                   with_delete_support,
                                   g_aggregator);
  run_ddl_statement(create_test);
  g_sqlite_comparator.query(
      "CREATE TABLE join_test(x int not null, y int, str text, dup_str text);");
  {
    const std::string insert_query{"INSERT INTO join_test VALUES(7, 43, 'foo', 'foo');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{
        "INSERT INTO join_test VALUES(8, null, 'bar', 'foo');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{
        "INSERT INTO join_test VALUES(9, null, 'baz', 'bar');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_hash_join_test() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS hash_join_test;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);

  std::string replicated_dec{!g_aggregator ? "" : ", PARTITIONS='REPLICATED'"};

  const std::string create_test{
      "CREATE TABLE hash_join_test(x int not null, str text encoding dict, t BIGINT) "
      "WITH (fragment_size=2" +
      replicated_dec + ");"};
  run_ddl_statement(create_test);
  g_sqlite_comparator.query(
      "CREATE TABLE hash_join_test(x int not null, str text, t BIGINT);");
  {
    const std::string insert_query{"INSERT INTO hash_join_test VALUES(7, 'foo', 1001);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{
        "INSERT INTO hash_join_test VALUES(8, 'bar', 5000000000);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{"INSERT INTO hash_join_test VALUES(9, 'the', 1002);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
}

void import_coalesce_cols_join_test(const int id, bool with_delete_support) {
  const std::string table_name = "coalesce_cols_test_" + std::to_string(id);
  const std::string drop_old_test{"DROP TABLE IF EXISTS " + table_name + ";"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);

  std::string columns_definition{
      "x int not null, y int, str text encoding dict, dup_str text encoding dict, d "
      "date, t time, tz timestamp, dn decimal(5)"};
  const auto create_test = build_create_table_statement(columns_definition,
                                                        table_name,
                                                        {"", g_shard_count},
                                                        {},

                                                        id == 2 ? 2 : 20,
                                                        with_delete_support,
                                                        g_aggregator);
  run_ddl_statement(create_test);

  g_sqlite_comparator.query("CREATE TABLE " + table_name +
                            "(x int not null, y int, str text, dup_str text, d date, t "
                            "time, tz timestamp, dn decimal(5));");
  TestHelpers::ValuesGenerator gen(table_name);
  for (size_t i = 0; i < 5; i++) {
    const auto insert_query = gen(i,
                                  20 - i,
                                  "'test'",
                                  "'test'",
                                  "'2018-01-01'",
                                  "'12:34:56'",
                                  "'2018-01-01 12:34:56'",
                                  i * 1.1);
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (size_t i = 5; i < 10; i++) {
    const auto insert_query = gen(i,
                                  20 - i,
                                  "'test1'",
                                  "'test1'",
                                  "'2017-01-01'",
                                  "'12:34:00'",
                                  "'2017-01-01 12:34:56'",
                                  i * 1.1);
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  if (id > 0) {
    for (size_t i = 10; i < 15; i++) {
      const auto insert_query = gen(i,
                                    20 - i,
                                    "'test2'",
                                    "'test2'",
                                    "'2016-01-01'",
                                    "'12:00:56'",
                                    "'2016-01-01 12:34:56'",
                                    i * 1.1);
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
  }
  if (id > 1) {
    for (size_t i = 15; i < 20; i++) {
      const auto insert_query = gen(i,
                                    20 - i,
                                    "'test3'",
                                    "'test3'",
                                    "'2015-01-01'",
                                    "'10:34:56'",
                                    "'2015-01-01 12:34:56'",
                                    i * 1.1);
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
  }
}

void import_emp_table() {
  const std::string drop_old_test{"DROP TABLE IF EXISTS emp;"};
  run_ddl_statement(drop_old_test);
  g_sqlite_comparator.query(drop_old_test);
  const std::string create_test{
      "CREATE TABLE emp(empno INT, ename TEXT NOT NULL ENCODING DICT, deptno INT) WITH "
      "(fragment_size=2);"};
  run_ddl_statement(create_test);
  g_sqlite_comparator.query(
      "CREATE TABLE emp(empno INT, ename TEXT NOT NULL, deptno INT);");
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
  const std::string create_test{
      "CREATE TABLE dept(deptno INT, dname TEXT ENCODING DICT) WITH (fragment_size=2);"};
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

void import_geospatial_test() {
  const std::string geospatial_test("DROP TABLE IF EXISTS geospatial_test;");
  run_ddl_statement(geospatial_test);
  constexpr char create_ddl[] = R"(CREATE TABLE geospatial_test (
        id INT,
        p POINT,
        l LINESTRING,
        poly POLYGON,
        mpoly MULTIPOLYGON,
        gp GEOMETRY(POINT),
        gp4326 GEOMETRY(POINT,4326) ENCODING COMPRESSED(32),
        gp4326none GEOMETRY(POINT,4326) ENCODING NONE,
        gp900913 GEOMETRY(POINT,900913),
        gl4326none GEOMETRY(LINESTRING,4326) ENCODING NONE,
        gpoly4326 GEOMETRY(POLYGON,4326)
      ) WITH (fragment_size=2); 
  )";
  run_ddl_statement(create_ddl);
  TestHelpers::ValuesGenerator gen("geospatial_test");
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    const std::string point{"'POINT(" + std::to_string(i) + " " + std::to_string(i) +
                            ")'"};
    const std::string linestring{
        "'LINESTRING(" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        ")'"};
    const std::string poly{"'POLYGON((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                           std::to_string(i + 1) + ", 0 0))'"};
    const std::string mpoly{"'MULTIPOLYGON(((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                            std::to_string(i + 1) + ", 0 0)))'"};
    run_multiple_agg(gen(i,
                         point,
                         linestring,
                         poly,
                         mpoly,
                         point,
                         point,
                         point,
                         point,
                         linestring,
                         poly),
                     ExecutorDeviceType::CPU);
  }
}

void import_geospatial_join_test(const bool replicate_inner_table = false) {
  // Create a single fragment inner table that is half the size of the geospatial_test
  // (outer) table
  const std::string geospatial_test("DROP TABLE IF EXISTS geospatial_inner_join_test;");
  run_ddl_statement(geospatial_test);
  constexpr char create_ddl[] = R"(CREATE TABLE geospatial_inner_join_test (
        id INT,
        p POINT,
        l LINESTRING,
        poly POLYGON,
        mpoly MULTIPOLYGON
      ) WITH (fragment_size=20); 
  )";
  run_ddl_statement(create_ddl);
  TestHelpers::ValuesGenerator gen("geospatial_inner_join_test");
  for (ssize_t i = 0; i < g_num_rows; i += 2) {
    const std::string point{"'POINT(" + std::to_string(i) + " " + std::to_string(i) +
                            ")'"};
    const std::string linestring{
        "'LINESTRING(" + std::to_string(i) + " 0, " + std::to_string(2 * i) + " " +
        std::to_string(2 * i) +
        ((i % 2) ? (", " + std::to_string(2 * i + 1) + " " + std::to_string(2 * i + 1))
                 : "") +
        ")'"};
    const std::string poly{"'POLYGON((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                           std::to_string(i + 1) + ", 0 0))'"};
    const std::string mpoly{"'MULTIPOLYGON(((0 0, " + std::to_string(i + 1) + " 0, 0 " +
                            std::to_string(i + 1) + ", 0 0)))'"};
    run_multiple_agg(gen(i, point, linestring, poly, mpoly), ExecutorDeviceType::CPU);
  }
}

}  // namespace

TEST(Select, ArrayUnnest) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    unsigned power10 = 1;
    for (const unsigned int_width : {16, 32, 64}) {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_i" + std::to_string(int_width) +
                               ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                           dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count + 2) * power10,
                v<int64_t>(result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(1,
                v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(power10,
                v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 1, true)));

      auto fixed_result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr3_i" + std::to_string(int_width) +
                               ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                           dt);
      ASSERT_EQ(g_array_test_row_count + 2, fixed_result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count + 2) * power10,
                v<int64_t>(fixed_result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(
          1,
          v<int64_t>(fixed_result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(fixed_result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(
          power10,
          v<int64_t>(fixed_result_rows->getRowAt(g_array_test_row_count + 1, 1, true)));

      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double"}) {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_" + float_type +
                               ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                           dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(1,
                v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));

      auto fixed_result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr3_" + float_type +
                               ") AS a FROM array_test GROUP BY a ORDER BY a DESC;",
                           dt);
      ASSERT_EQ(g_array_test_row_count + 2, fixed_result_rows->rowCount());
      ASSERT_EQ(
          1,
          v<int64_t>(fixed_result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(fixed_result_rows->getRowAt(0, 0, true)));
    }
    {
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr_str) AS a FROM array_test GROUP BY a ORDER BY a "
          "DESC;",
          dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows->rowCount());
      ASSERT_EQ(1,
                v<int64_t>(result_rows->getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 0, true)));
    }
    {
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr_bool) AS a FROM array_test GROUP BY a ORDER BY a "
          "DESC;",
          dt);
      ASSERT_EQ(size_t(2), result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count * 3),
                v<int64_t>(result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(int64_t(g_array_test_row_count * 3),
                v<int64_t>(result_rows->getRowAt(1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(0, v<int64_t>(result_rows->getRowAt(1, 1, true)));

      auto fixed_result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr6_bool) AS a FROM array_test GROUP BY a ORDER BY a "
          "DESC;",
          dt);
      ASSERT_EQ(size_t(2), fixed_result_rows->rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count * 3),
                v<int64_t>(fixed_result_rows->getRowAt(0, 0, true)));
      ASSERT_EQ(int64_t(g_array_test_row_count * 3),
                v<int64_t>(fixed_result_rows->getRowAt(1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(fixed_result_rows->getRowAt(0, 1, true)));
      ASSERT_EQ(0, v<int64_t>(fixed_result_rows->getRowAt(1, 1, true)));
    }
  }
}

TEST(Select, ArrayIndex) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
      ASSERT_EQ(1,
                v<int64_t>(run_simple_agg(
                    "SELECT COUNT(*) FROM array_test WHERE arr_i32[2] = " +
                        std::to_string(10 * (row_idx + 2)) +
                        " AND x = " + std::to_string(7 + row_idx) +
                        " AND arr3_i32[2] = " + std::to_string(10 * (row_idx + 2)) +
                        " AND real_str LIKE 'real_str" + std::to_string(row_idx) + "';",
                    dt)));
      ASSERT_EQ(0,
                v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE "
                                          "arr_i32[4] > 0 OR arr_i32[4] <= 0 OR "
                                          "arr3_i32[4] > 0 OR arr3_i32[4] <= 0;",
                                          dt)));
      ASSERT_EQ(0,
                v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE "
                                          "arr_i32[0] > 0 OR arr_i32[0] <= 0 OR "
                                          "arr3_i32[0] > 0 OR arr3_i32[0] <= 0;",
                                          dt)));
    }
    for (size_t i = 1; i <= 6; ++i) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count / 2),
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE arr_bool[" +
                                        std::to_string(i) +
                                        "] AND "
                                        "arr6_bool[" +
                                        std::to_string(i) + "];",
                                    dt)));
    }
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE arr_bool[7];", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE arr6_bool[7];", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE arr_bool[0];", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE arr6_bool[0];", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE NOT "
                                        "(arr_i16[7] > 0 AND arr_i16[7] <= 0 AND "
                                        "arr3_i16[7] > 0 AND arr3_i16[7] <= 0);",
                                        dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE NOT "
                                        "(arr_i16[2] > 0 AND arr_i16[2] <= 0 AND "
                                        "arr3_i16[2] > 0 AND arr3_i16[2] <= 0);",
                                        dt)));
  }
}

TEST(Select, ArrayCountDistinct) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (const unsigned int_width : {16, 32, 64}) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count + 2),
          v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr_i" +
                                        std::to_string(int_width) + ") FROM array_test;",
                                    dt)));
      auto result_rows =
          run_multiple_agg("SELECT COUNT(distinct arr_i" + std::to_string(int_width) +
                               ") FROM array_test GROUP BY x;",
                           dt);
      ASSERT_EQ(g_array_test_row_count, result_rows->rowCount());
      for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
        ASSERT_EQ(3, v<int64_t>(result_rows->getRowAt(row_idx, 0, true)));
      }

      ASSERT_EQ(
          int64_t(g_array_test_row_count + 2),
          v<int64_t>(run_simple_agg("SELECT COUNT(distinct arr3_i" +
                                        std::to_string(int_width) + ") FROM array_test;",
                                    dt)));
      auto fixed_result_rows =
          run_multiple_agg("SELECT COUNT(distinct arr3_i" + std::to_string(int_width) +
                               ") FROM array_test GROUP BY x;",
                           dt);
      ASSERT_EQ(g_array_test_row_count, fixed_result_rows->rowCount());
      for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
        ASSERT_EQ(3, v<int64_t>(fixed_result_rows->getRowAt(row_idx, 0, true)));
      }
    }
    for (const std::string float_type : {"float", "double"}) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count + 2),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(distinct arr_" + float_type + ") FROM array_test;", dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count + 2),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(distinct arr3_" + float_type + ") FROM array_test;", dt)));
    }
    ASSERT_EQ(int64_t(g_array_test_row_count + 2),
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(distinct arr_str) FROM array_test;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(distinct arr_bool) FROM array_test;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(distinct arr6_bool) FROM array_test;", dt)));
  }
}

TEST(Select, ArrayAnyAndAll) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    unsigned power10 = 1;
    for (const unsigned int_width : {16, 32, 64}) {
      ASSERT_EQ(
          2,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE " +
                                        std::to_string(2 * power10) + " = ANY arr_i" +
                                        std::to_string(int_width) + " AND " +
                                        std::to_string(2 * power10) + " = ANY arr3_i" +
                                        std::to_string(int_width) + ";",
                                    dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count) - 2,
          v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test WHERE " +
                                        std::to_string(2 * power10) + " < ALL arr_i" +
                                        std::to_string(int_width) + " AND " +
                                        std::to_string(2 * power10) + " < ALL arr3_i" +
                                        std::to_string(int_width) + ";",
                                    dt)));
      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double"}) {
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(*) FROM array_test WHERE 1 < ANY arr_" + float_type + ";",
              dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(*) FROM array_test WHERE 2 < ANY arr_" + float_type + ";",
              dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(*) FROM array_test WHERE 0 < ALL arr_" + float_type + ";",
              dt)));
      ASSERT_EQ(
          int64_t(g_array_test_row_count),
          v<int64_t>(run_simple_agg(
              "SELECT COUNT(*) FROM array_test WHERE 0 < ALL arr3_" + float_type + ";",
              dt)));
    }
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE x - 5 = ANY arr_i16;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE x - 5 = ANY arr3_i16;", dt)));
    ASSERT_EQ(1,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'aa' = ANY arr_str;", dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' = ANY arr_str;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ANY arr_bool;",
            dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ANY arr6_bool;",
            dt)));
    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE CAST('t' AS boolean) = ALL arr_bool;",
            dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' < ALL arr_str;", dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count - 1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' <= ALL arr_str;", dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' > ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE 'bb' >= ANY arr_str;", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM array_test WHERE  real_str = ANY arr_str;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE  real_str <> ANY arr_str;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count - 1),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM array_test WHERE (NOT ('aa' = ANY arr_str));", dt)));
    // these two test just confirm that the regex does not mess with other similar
    // patterns
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) as SMALL FROM array_test;", dt)));
    ASSERT_EQ(
        int64_t(g_array_test_row_count),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) as COMPANY FROM array_test;", dt)));
  }
}

TEST(Select, ArrayUnsupported) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT MIN(arr_i64) FROM array_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT MIN(arr3_i64) FROM array_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT UNNEST(arr_str), COUNT(*) cc FROM array_test GROUP BY arr_str;", dt),
        std::runtime_error);
  }
}

TEST(Select, OrRewrite) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE str = 'foo' OR str = 'bar' OR str = 'baz' OR str "
      "= 'foo' OR str = 'bar' OR str "
      "= 'baz' OR str = 'foo' OR str = 'bar' OR str = 'baz' OR str = 'baz' OR str = "
      "'foo' OR str = 'bar' OR str = "
      "'baz';",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x = 7 OR x = 8 OR x = 7 OR x = 8 OR x = 7 OR x = "
      "8 OR x = 7 OR x = 8 OR x = 7 "
      "OR x = 8 OR x = 7 OR x = 8;",
      dt);
  }
}

TEST(Select, GpuSort) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x, COUNT(*) AS n FROM gpu_sort_test GROUP BY x ORDER BY n DESC;", dt);
    c("SELECT x, COUNT(*), COUNT(*) AS n FROM gpu_sort_test GROUP BY x ORDER BY n DESC;",
      dt);
  }
}

TEST(Select, GroupByConstrainedByInQueryRewrite) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) AS n, x FROM query_rewrite_test WHERE x IN (2, 5) GROUP BY x "
      "HAVING n > 0 ORDER BY n DESC;",
      dt);
    c("SELECT COUNT(*) AS n, x FROM query_rewrite_test WHERE x IN (2, 99) GROUP BY x "
      "HAVING n > 0 ORDER BY n DESC;",
      dt);

    c("SELECT COUNT(*) AS n, str FROM query_rewrite_test WHERE str IN ('str2', 'str5') "
      "GROUP BY str HAVING n > 0 "
      "ORDER "
      "BY n DESC;",
      dt);

    c("SELECT COUNT(*) AS n, str FROM query_rewrite_test WHERE str IN ('str2', 'str99') "
      "GROUP BY str HAVING n > 0 "
      "ORDER BY n DESC;",
      dt);
  }
}

TEST(Select, RedundantGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT DISTINCT(x) from test where y < 10 and z > 30 GROUP BY x;", dt);
  }
}

TEST(Select, BigDecimalRange) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT CAST(d AS BIGINT) AS di, COUNT(*) FROM big_decimal_range_test GROUP BY d "
      "HAVING di > 0 ORDER BY d;",
      dt);
    c("SELECT d1*2 FROM big_decimal_range_test ORDER BY d1;", dt);
    c("SELECT 2*d1 FROM big_decimal_range_test ORDER BY d1;", dt);
    c("SELECT d1 * (CAST(d1 as INT) + 1) FROM big_decimal_range_test ORDER BY d1;", dt);
    c("SELECT (CAST(d1 as INT) + 1) * d1 FROM big_decimal_range_test ORDER BY d1;", dt);
  }
}

TEST(Select, DecimalCompression) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    std::string mapd_sql = "";
    std::string sqlite_sql = "";

    mapd_sql =
        "SELECT AVG(big_dec), AVG(med_dec), AVG(small_dec) FROM "
        "decimal_compression_test;";
    sqlite_sql =
        "SELECT 1.0*AVG(big_dec), 1.0*AVG(med_dec), 1.0*AVG(small_dec) FROM "
        "decimal_compression_test;";
    c(mapd_sql, sqlite_sql, dt);
    c(sqlite_sql, sqlite_sql, dt);

    mapd_sql =
        "SELECT SUM(big_dec), SUM(med_dec), SUM(small_dec) FROM "
        "decimal_compression_test;";
    sqlite_sql =
        "SELECT 1.0*SUM(big_dec), 1.0*SUM(med_dec), 1.0*SUM(small_dec) FROM "
        "decimal_compression_test;";
    c(mapd_sql, sqlite_sql, dt);
    c(sqlite_sql, sqlite_sql, dt);

    mapd_sql =
        "SELECT MIN(big_dec), MIN(med_dec), MIN(small_dec) FROM "
        "decimal_compression_test;";
    sqlite_sql =
        "SELECT 1.0*MIN(big_dec), 1.0*MIN(med_dec), 1.0*MIN(small_dec) FROM "
        "decimal_compression_test;";
    c(mapd_sql, sqlite_sql, dt);
    c(sqlite_sql, sqlite_sql, dt);

    mapd_sql =
        "SELECT MAX(big_dec), MAX(med_dec), MAX(small_dec) FROM "
        "decimal_compression_test;";
    sqlite_sql =
        "SELECT 1.0*MAX(big_dec), 1.0*MAX(med_dec), 1.0*MAX(small_dec) FROM "
        "decimal_compression_test;";
    c(mapd_sql, sqlite_sql, dt);
    c(sqlite_sql, sqlite_sql, dt);

    mapd_sql =
        "SELECT big_dec, COUNT(*) as n, AVG(med_dec) as med_dec_avg, SUM(small_dec) as "
        "small_dec_sum FROM decimal_compression_test GROUP BY big_dec ORDER BY "
        "small_dec_sum;";
    sqlite_sql =
        "SELECT 1.0*big_dec, COUNT(*) as n, 1.0*AVG(med_dec) as med_dec_avg, "
        "1.0*SUM(small_dec) as small_dec_sum FROM decimal_compression_test GROUP BY "
        "big_dec ORDER BY small_dec_sum;";
    c(mapd_sql, sqlite_sql, dt);
    c(sqlite_sql, sqlite_sql, dt);
  }
}

TEST(Update, DecimalOverflow) {
  auto test = [](int precision, int scale) -> void {
    run_ddl_statement("DROP TABLE IF EXISTS decimal_overflow_test;");
    run_ddl_statement("CREATE TABLE decimal_overflow_test (d DECIMAL(" +
                      std::to_string(precision) + ", " + std::to_string(scale) + "));");
    run_multiple_agg("INSERT INTO decimal_overflow_test VALUES(null);",
                     ExecutorDeviceType::CPU);
    int64_t val = (int64_t)std::pow((double)10, precision - scale);
    run_multiple_agg(
        "INSERT INTO decimal_overflow_test VALUES(" + std::to_string(val - 1) + ");",
        ExecutorDeviceType::CPU);
    EXPECT_THROW(run_multiple_agg("INSERT INTO decimal_overflow_test VALUES(" +
                                      std::to_string(val) + ");",
                                  ExecutorDeviceType::CPU),
                 std::runtime_error);
    SKIP_ON_AGGREGATOR(
        run_multiple_agg("UPDATE decimal_overflow_test set d=d-1 WHERE d IS NOT NULL;",
                         ExecutorDeviceType::CPU));
    SKIP_ON_AGGREGATOR(
        run_multiple_agg("UPDATE decimal_overflow_test set d=d+1 WHERE d IS NOT NULL;",
                         ExecutorDeviceType::CPU));
    SKIP_ON_AGGREGATOR(EXPECT_THROW(
        run_multiple_agg("UPDATE decimal_overflow_test set d=d+1 WHERE d IS NOT NULL;",
                         ExecutorDeviceType::CPU),
        std::runtime_error));
  };

  test(1, 0);
  test(3, 2);
  test(4, 2);
  test(7, 2);
  test(7, 6);
  test(14, 2);
  test(17, 2);
  test(18, 9);
  EXPECT_THROW(test(18, 20), std::runtime_error);
  EXPECT_THROW(test(18, 18), std::runtime_error);
  EXPECT_THROW(test(19, 0), std::runtime_error);
}

TEST(Drop, AfterDrop) {
  run_ddl_statement("create table droptest (i1 integer);");
  run_multiple_agg("insert into droptest values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into droptest values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3),
            v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM droptest;",
                                      ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table droptest;");
  run_ddl_statement("create table droptest (n1 integer);");
  run_multiple_agg("insert into droptest values(3);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into droptest values(4);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(7),
            v<int64_t>(run_simple_agg("SELECT SUM(n1) FROM droptest;",
                                      ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table droptest;");
}

TEST(Alter, AfterAlterTableName) {
  run_ddl_statement("create table alter_name_test (i1 integer);");
  run_multiple_agg("insert into alter_name_test values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_name_test values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3),
            v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_name_test;",
                                      ExecutorDeviceType::CPU)));
  run_ddl_statement("alter table alter_name_test rename to alter_name_test_after;");
  run_multiple_agg("insert into alter_name_test_after values(3);",
                   ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_name_test_after values(4);",
                   ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(10),
            v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_name_test_after;",
                                      ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table alter_name_test_after;");
}

TEST(Alter, AfterAlterColumnName) {
  run_ddl_statement("create table alter_column_test (i1 integer);");
  run_multiple_agg("insert into alter_column_test values(1);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_column_test values(2);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3),
            v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM alter_column_test;",
                                      ExecutorDeviceType::CPU)));
  run_ddl_statement("alter table alter_column_test rename column i1 to n1;");
  run_multiple_agg("insert into alter_column_test values(3);", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into alter_column_test values(4);", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(10),
            v<int64_t>(run_simple_agg("SELECT SUM(n1) FROM alter_column_test;",
                                      ExecutorDeviceType::CPU)));
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
    c("SELECT str, SUM(y) AS n FROM test WHERE x > (SELECT COUNT(*) FROM test) - 14 "
      "GROUP BY str ORDER BY str ASC;",
      dt);
    c("SELECT COUNT(*) FROM test, (SELECT x FROM test_inner) AS inner_x WHERE test.x "
      "= inner_x.x;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test WHERE y > 42);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test GROUP BY x ORDER BY "
      "COUNT(*) DESC LIMIT 1);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test GROUP BY x);", dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM join_test);", dt);
    c("SELECT MIN(yy), MAX(yy) FROM (SELECT AVG(y) as yy FROM test GROUP BY x);", dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x NOT IN (SELECT x + 1 FROM "
      "subquery_test GROUP BY x);",
      dt);
    c("SELECT MAX(ct) FROM (SELECT COUNT(*) AS ct, str AS foo FROM test GROUP BY foo);",
      dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x IN (SELECT x AS foobar FROM "
      "subquery_test GROUP BY foobar);",
      dt);
    c("SELECT * FROM (SELECT x FROM test ORDER BY x) ORDER BY x;", dt);
    c("SELECT AVG(y) FROM (SELECT * FROM test ORDER BY z LIMIT 5);", dt);
    c("SELECT COUNT(*) FROM subquery_test WHERE x NOT IN (SELECT x + 1 FROM "
      "subquery_test GROUP BY x);",
      dt);
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg(
                  "SELECT * FROM (SELECT rowid FROM test WHERE rowid = 0);", dt)));
    c("SELECT COUNT(*) FROM test WHERE x NOT IN (SELECT x FROM test GROUP BY x ORDER BY "
      "COUNT(*));",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x NOT IN (SELECT x FROM test GROUP BY x);", dt);
    c("SELECT COUNT(*) FROM test WHERE f IN (SELECT DISTINCT f FROM test WHERE x > 7);",
      dt);
    c("SELECT emptytab. x, CASE WHEN emptytab. y IN (SELECT emptytab. y FROM emptytab "
      "GROUP BY emptytab. y) then "
      "emptytab. y END yy, sum(x) "
      "FROM emptytab GROUP BY emptytab. x, yy;",
      dt);
    // Throws join table must be replicated in distributed
    SKIP_ON_AGGREGATOR(
        c("WITH d1 AS (SELECT deptno, dname FROM dept LIMIT 10) SELECT ename, dname FROM "
          "emp, d1 WHERE emp.deptno = "
          "d1.deptno ORDER BY ename ASC LIMIT 10;",
          dt));
    c("SELECT x FROM (SELECT x, MAX(y), COUNT(*) AS n FROM test GROUP BY x HAVING MAX(y) "
      "> 42 ORDER BY n);",
      dt);
    c("SELECT CASE WHEN test.x IN (SELECT x FROM test_inner) THEN x ELSE NULL END AS c, "
      "COUNT(*) AS n FROM test WHERE "
      "y > 40 GROUP BY c ORDER BY n DESC;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE x IN (SELECT x FROM test WHERE x > (SELECT "
      "COUNT(*) FROM test WHERE x > 7) + 2 "
      "GROUP BY x);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE ofd IN (SELECT ofd FROM test GROUP BY ofd);", dt);
    c("SELECT COUNT(*) FROM test WHERE ofd NOT IN (SELECT ofd FROM test GROUP BY ofd);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE ss IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test WHERE ss NOT IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test WHERE str IN (SELECT str FROM test_in_bitmap GROUP BY "
      "str);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str NOT IN (SELECT str FROM test_in_bitmap GROUP "
      "BY str);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str IN (SELECT ss FROM test GROUP BY ss);", dt);
    c("SELECT COUNT(*) FROM test WHERE str NOT IN (SELECT ss FROM test GROUP BY ss);",
      dt);
    c("SELECT COUNT(*) FROM test WHERE ss IN (SELECT str FROM test GROUP BY str);", dt);
    c("SELECT COUNT(*) FROM test WHERE ss NOT IN (SELECT str FROM test GROUP BY str);",
      dt);
    c("SELECT str, COUNT(*) FROM test WHERE x IN (SELECT x FROM test WHERE x > 8) GROUP "
      "BY str;",
      dt);
    c("SELECT COUNT(*) FROM test_in_bitmap WHERE str IN (SELECT ss FROM test GROUP BY "
      "ss);",
      dt);
    c("SELECT COUNT(*) FROM test_in_bitmap WHERE str NOT IN (SELECT ss FROM test GROUP "
      "BY ss);",
      dt);
    c("SELECT COUNT(*) FROM test_in_bitmap WHERE str IN (SELECT str FROM test GROUP BY "
      "str);",
      dt);
    c("SELECT COUNT(*) FROM test_in_bitmap WHERE str NOT IN (SELECT str FROM test GROUP "
      "BY str);",
      dt);
    c("SELECT COUNT(str) FROM (SELECT * FROM (SELECT * FROM test WHERE x = 7) WHERE y = "
      "42) WHERE t > 1000;",
      dt);
    c("SELECT x_cap, y FROM (SELECT CASE WHEN x > 100 THEN 100 ELSE x END x_cap, y, t "
      "FROM emptytab) GROUP BY x_cap, "
      "y;",
      dt);
    c("SELECT COUNT(*) FROM test WHERE str IN (SELECT DISTINCT str FROM test);", dt);
    c("SELECT SUM((x - (SELECT AVG(x) FROM test)) * (x - (SELECT AVG(x) FROM test)) / "
      "((SELECT COUNT(x) FROM test) - "
      "1)) FROM test;",
      dt);
    EXPECT_THROW(run_multiple_agg("SELECT * FROM (SELECT * FROM test LIMIT 5);", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT AVG(SELECT x FROM test LIMIT 5) FROM test;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT COUNT(*) FROM test WHERE str < (SELECT str FROM test LIMIT 1);", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg(
            "SELECT COUNT(*) FROM test WHERE str IN (SELECT x FROM test GROUP BY x);",
            dt),
        std::runtime_error);
    ASSERT_NEAR(static_cast<double>(2.057),
                v<double>(run_simple_agg(
                    "SELECT AVG(dd) / (SELECT STDDEV(dd) FROM test) FROM test;", dt)),
                static_cast<double>(0.10));
    c("SELECT R.x, R.f, count(*) FROM (SELECT x,y,z,t,f,d FROM test WHERE x >= 7 AND z < "
      "0 AND t > 1001 AND d < 3) AS R WHERE R.y > 0 AND z < 0 AND t > 1001 AND d "
      "< 3 GROUP BY R.x, R.f ORDER BY R.x;",
      dt);
    c("SELECT R.y, R.d, count(*) FROM (SELECT x,y,z,t,f,d FROM test WHERE y > 42 AND f > "
      "1.0) AS R WHERE R.x > 0 AND t > 1001 AND f > 1.0 GROUP BY "
      "R.y, R.d ORDER BY R.d;",
      dt);
    c("SELECT R.x, R.f, count(*) FROM (SELECT x,y,z,t,f,d FROM test WHERE x >= 7 AND z < "
      "0 AND t > 1001 AND d < 3 LIMIT 3) AS R WHERE R.y > 0 AND z < 0 AND t > 1001 AND d "
      "< 3 GROUP BY R.x, R.f ORDER BY R.f;",
      dt);
    c("SELECT R.y, R.d, count(*) FROM (SELECT x,y,z,t,f,d FROM test WHERE y > 42 AND f > "
      "1.0 ORDER BY x DESC LIMIT 2) AS R WHERE R.x > 0 AND t > 1001 AND f > 1.0 GROUP BY "
      "R.y, R.d ORDER BY R.y;",
      dt);
  }
}

TEST(Select, Joins_Arrays) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, array_test_inner "
                                        "WHERE test.x = ALL array_test_inner.arr_i16;",
                                        dt)));
    ASSERT_EQ(int64_t(60),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, array_test_inner "
                                        "WHERE test.x = ANY array_test_inner.arr_i16;",
                                        dt)));
    ASSERT_EQ(int64_t(2 * g_array_test_row_count * g_num_rows - 60),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, array_test_inner "
                                        "WHERE test.x <> ALL array_test_inner.arr_i16;",
                                        dt)));
    ASSERT_EQ(int64_t(g_array_test_row_count),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, array_test_inner "
                                        "WHERE 7 = array_test_inner.arr_i16[1];",
                                        dt)));
    ASSERT_EQ(int64_t(2 * g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, array_test WHERE "
                                        "test.x = array_test.x AND 'bb' = ANY arr_str;",
                                        dt)));
    auto result_rows = run_multiple_agg(
        "SELECT UNNEST(array_test.arr_i16) AS a, test_inner.x, COUNT(*) FROM array_test, "
        "test_inner WHERE test_inner.x "
        "= array_test.arr_i16[1] GROUP BY a, test_inner.x;",
        dt);
    ASSERT_EQ(size_t(3), result_rows->rowCount());
    ASSERT_EQ(int64_t(g_array_test_row_count / 2 + g_array_test_row_count / 4),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test, test_inner WHERE EXTRACT(HOUR FROM test.m) "
                  "= 22 AND test.x = test_inner.x;",
                  dt)));
    ASSERT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM array_test, test_inner WHERE "
                                  "array_test.arr_i32[array_test.x - 5] = 20 AND "
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
    c("SELECT test.x, emptytab.x FROM test LEFT JOIN emptytab ON test.y = emptytab.y "
      "ORDER BY test.x ASC;",
      dt);
  }
}

TEST(Select, Joins_ImplicitJoins) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, hash_join_test WHERE test.t = hash_join_test.t;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x < test_inner.x + 1;", dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT COUNT(*) FROM test, test_inner WHERE test.real_str = test_inner.str;",
          dt));
    c("SELECT test_inner.x, COUNT(*) AS n FROM test, test_inner WHERE test.x = "
      "test_inner.x GROUP BY test_inner.x "
      "ORDER BY n;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str;", dt);
    c("SELECT test.str, COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str "
      "GROUP BY test.str;",
      dt);
    c("SELECT test_inner.str, COUNT(*) FROM test, test_inner WHERE test.str = "
      "test_inner.str GROUP BY test_inner.str;",
      dt);
    c("SELECT test.str, COUNT(*) AS foobar FROM test, test_inner WHERE test.x = "
      "test_inner.x AND test.x > 6 GROUP BY "
      "test.str HAVING foobar > 5;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.real_str LIKE 'real_ba%' AND "
      "test.x = test_inner.x;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE LENGTH(test.real_str) = 8 AND test.x "
      "= test_inner.x;",
      dt);
    c("SELECT a.x, b.str FROM test a, join_test b WHERE a.str = b.str GROUP BY a.x, "
      "b.str ORDER BY a.x, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a, join_test b WHERE a.str = b.str ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT COUNT(1) FROM test a, join_test b, test_inner c WHERE a.str = b.str AND "
      "b.x = c.x",
      dt);

    c("SELECT COUNT(*) FROM test a, join_test b, test_inner c WHERE a.x = b.x AND "
      "a.y = "
      "b.x AND a.x = c.x AND c.str = "
      "'foo';",
      dt);

    SKIP_ON_AGGREGATOR(
        c("SELECT COUNT(*) FROM test a, test b WHERE a.x = b.x AND a.y = b.y;", dt));
    SKIP_ON_AGGREGATOR(
        c("SELECT SUM(b.y) FROM test a, test b WHERE a.x = b.x AND a.y = b.y;", dt));
    SKIP_ON_AGGREGATOR(
        c("SELECT COUNT(*) FROM test a, test b WHERE a.x = b.x AND a.str = b.str;", dt));
    c("SELECT COUNT(*) FROM test, test_inner WHERE (test.x = test_inner.x AND test.y = "
      "42 AND test_inner.str = 'foo') "
      "OR (test.x = test_inner.x AND test.y = 43 AND test_inner.str = 'foo');",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.x OR test.x = "
      "test_inner.x;",
      dt);
    SKIP_ON_AGGREGATOR(c("SELECT bar.str FROM test, bar WHERE test.str = bar.str;", dt));

    ASSERT_EQ(int64_t(3),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, join_test "
                                        "WHERE test.rowid = join_test.rowid;",
                                        dt)));
    SKIP_ON_AGGREGATOR(
        ASSERT_EQ(7,
                  v<int64_t>(run_simple_agg("SELECT test.x FROM test, test_inner WHERE "
                                            "test.x = test_inner.x AND test.rowid = 9;",
                                            dt))));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, test_inner WHERE "
                                        "test.x = test_inner.x AND test.rowid = 20;",
                                        dt)));
  }
}

TEST(Select, Joins_DifferentIntegerTypes) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.xx;", dt);
    c("SELECT test_inner.xx, COUNT(*) AS n FROM test, test_inner WHERE test.x = "
      "test_inner.xx GROUP BY test_inner.xx ORDER BY n;",
      dt);
  }
}

TEST(Select, Joins_FilterPushDown) {
  auto default_flag = g_enable_filter_push_down;
  auto default_lower_frac = g_filter_push_down_low_frac;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    for (auto fpd : {std::make_pair(true, 1.0), std::make_pair(false, 0.0)}) {
      g_enable_filter_push_down = fpd.first;
      g_filter_push_down_low_frac = fpd.second;
      c("SELECT COUNT(*) FROM coalesce_cols_test_2 AS R, coalesce_cols_test_0 AS S "
        "WHERE R.y = S.y AND R.x > 2 AND (S.x > 1 OR S.y < 18);",
        dt);
      c("SELECT COUNT(*) FROM coalesce_cols_test_2 AS R, coalesce_cols_test_0 AS S "
        "WHERE R.x = S.x AND S.str = 'test1' AND ABS(S.dn - 2.2) < 0.001;",
        dt);
      c("SELECT S.y, COUNT(*) FROM coalesce_cols_test_2 AS R, coalesce_cols_test_0 AS S "
        "WHERE R.x = S.x AND S.t < time '12:40:23' AND S.d < date '2018-01-01' GROUP BY "
        "S.y ORDER BY S.y;",
        "SELECT R.y, COUNT(*) FROM coalesce_cols_test_2 AS R, coalesce_cols_test_0 AS S "
        "WHERE R.x = S.x AND S.t < time('12:40:23') AND S.d < date('2018-01-01') GROUP "
        "BY S.y "
        "ORDER BY S.y;",
        dt);
      c("SELECT R.y, COUNT(*) as cnt FROM coalesce_cols_test_2 AS R, "
        "coalesce_cols_test_1 AS S, coalesce_cols_test_0 AS T WHERE T.str = S.str AND "
        "S.x = R.x AND S.y < 10 GROUP "
        "BY R.y ORDER BY R.y;",
        dt);
      c("SELECT R.y, COUNT(*) as cnt FROM coalesce_cols_test_2 AS R, "
        "coalesce_cols_test_1 AS S, coalesce_cols_test_0 AS T WHERE T.y = S.y AND S.x = "
        "R.x AND T.x = 2 GROUP "
        "BY R.y ORDER BY R.y;",
        dt);
      c("SELECT R.y, COUNT(*) as cnt FROM coalesce_cols_test_2 AS R, "
        "coalesce_cols_test_1 AS S, coalesce_cols_test_0 AS T WHERE T.x = S.x AND S.y = "
        "R.y AND R.x < 20 AND S.y > 2 AND S.str <> 'foo' AND T.y < 18 AND T.x > 1 GROUP "
        "BY R.y ORDER BY R.y;",
        dt);
      c("SELECT T.x, COUNT(*) as cnt FROM coalesce_cols_test_2 AS R,"
        "coalesce_cols_test_1 AS S, "
        "coalesce_cols_test_0 AS T WHERE T.str = S.dup_str AND S.x = R.x AND T.y"
        "  = R.y AND R.x > 0 "
        "AND S.str ='test' AND S.y > 2 AND T.dup_str<> 'test4' GROUP BY T.x ORDER BY "
        "cnt;",
        dt);
      // self-join involved
      c("SELECT R.y, COUNT(*) as cnt FROM coalesce_cols_test_2 AS R, "
        "coalesce_cols_test_2 AS S, coalesce_cols_test_0 AS T WHERE T.x = S.x AND S.y = "
        "R.y AND R.x < 20 AND S.y > 2 AND S.str <> 'foo' AND T.y < 18 AND T.x > 1 GROUP "
        "BY R.y ORDER BY R.y;",
        dt);
    }
  }
  // reloading default values
  g_enable_filter_push_down = default_flag;
  g_filter_push_down_low_frac = default_lower_frac;
}

TEST(Select, Joins_InnerJoin_TwoTables) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test a JOIN single_row_test b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) from test a JOIN single_row_test b ON a.ofd = b.x;", dt);
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.x = test_inner.x;", dt);
    c("SELECT a.y, z FROM test a JOIN test_inner b ON a.x = b.x order by a.y;", dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.str = b.dup_str;", dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT COUNT(*) FROM test_inner_x a JOIN test_x b ON a.x = b.x;", dt));
    c("SELECT a.x FROM test a JOIN join_test b ON a.str = b.dup_str ORDER BY a.x;", dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT a.x FROM test_inner_x a JOIN test_x b ON a.x = b.x ORDER BY a.x;", dt));
    c("SELECT a.x FROM test a JOIN join_test b ON a.str = b.dup_str GROUP BY a.x ORDER "
      "BY a.x;",
      dt);
    SKIP_ON_AGGREGATOR(c(
        "SELECT a.x FROM test_inner_x a JOIN test_x b ON a.x = b.x GROUP BY a.x ORDER BY "
        "a.x;",
        dt));
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.x = test_inner.x AND test.rowid "
      "= test_inner.rowid;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.y = test_inner.y OR (test.y IS "
      "NULL AND test_inner.y IS NULL);",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.str = join_test.dup_str OR "
      "(test.str IS NULL AND "
      "join_test.dup_str IS NULL));",
      dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT t1.fixed_null_str FROM (SELECT fixed_null_str, SUM(x) n1 FROM test "
          "GROUP BY fixed_null_str) t1 INNER "
          "JOIN (SELECT fixed_null_str, SUM(y) n2 FROM test GROUP BY fixed_null_str) t2 "
          "ON ((t1.fixed_null_str = "
          "t2.fixed_null_str) OR (t1.fixed_null_str IS NULL AND t2.fixed_null_str IS "
          "NULL));",
          dt));
  }
}

TEST(Select, Joins_InnerJoin_Sharded) {
  SKIP_ALL_ON_AGGREGATOR();
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // Sharded Inner Joins
    c("SELECT st1.i, st2.i FROM st1 INNER JOIN st2 ON st1.i = st2.i ORDER BY st1.i;", dt);
    c("SELECT st1.j, st2.j FROM st1 INNER JOIN st2 ON st1.i = st2.i ORDER BY st1.i;", dt);
    c("SELECT st1.i, st2.i FROM st1 INNER JOIN st2 ON st1.i = st2.i WHERE st2.i > -1 "
      "ORDER BY st1.i;",
      dt);
    c("SELECT st1.i, st2.i FROM st1 INNER JOIN st2 ON st1.i = st2.i WHERE st2.i > 0 "
      "ORDER BY st1.i;",
      dt);
    c("SELECT st1.i, st1.s, st2.i, st2.s FROM st1 INNER JOIN st2 ON st1.i = st2.i WHERE "
      "st2.i > 0 ORDER BY st1.i;",
      dt);
    c("SELECT st1.i, st1.j, st1.s, st2.i, st2.s, st2.j FROM st1 INNER JOIN st2 ON st1.i "
      "= st2.i WHERE st2.i > 0 ORDER "
      "BY st1.i;",
      dt);
    c("SELECT st1.j, st1.s, st2.s, st2.j FROM st1 INNER JOIN st2 ON st1.i = st2.i WHERE "
      "st2.i > 0 ORDER BY st1.i;",
      dt);
    c("SELECT st1.j, st1.s, st2.s, st2.j FROM st1 INNER JOIN st2 ON st1.i = st2.i WHERE "
      "st2.i > 0 and st1.s <> 'foo' "
      "and st2.s <> 'foo' ORDER BY st1.i;",
      dt);

    // Non-sharded inner join (single frag)
    c("SELECT st1.i, st2.i FROM st1 INNER JOIN st2 ON st1.j = st2.j ORDER BY st1.i;", dt);
    c("SELECT st1.j, st2.j FROM st1 INNER JOIN st2 ON st1.j = st2.j ORDER BY st1.i;", dt);
    c("SELECT st1.j, st2.j FROM st1 INNER JOIN st2 ON st1.j = st2.j WHERE st2.j > -1 "
      "ORDER BY st1.i;",
      dt);
    c("SELECT st1.j, st2.j FROM st1 INNER JOIN st2 ON st1.j = st2.j WHERE st2.j > 0 "
      "ORDER BY st1.i;",
      dt);
    c("SELECT st1.j, st1.s, st2.j, st2.s FROM st1 INNER JOIN st2 ON st1.j = st2.j WHERE "
      "st2.j > 0 ORDER BY st1.i;",
      dt);
    c("SELECT st1.i, st1.j, st1.s, st2.i, st2.s, st2.j FROM st1 INNER JOIN st2 ON st1.j "
      "= st2.j WHERE st2.i > 0 ORDER "
      "BY st1.i;",
      dt);
    c("SELECT st1.i, st1.j, st1.s, st2.i, st2.s, st2.j FROM st1 INNER JOIN st2 ON st1.j "
      "= st2.j WHERE st2.j > 0 ORDER "
      "BY st1.i;",
      dt);
    c("SELECT st1.j, st1.s, st2.s, st2.j FROM st1 INNER JOIN st2 ON st1.j = st2.j WHERE "
      "st2.j > 0 ORDER BY st1.i;",
      dt);
    c("SELECT st1.j, st1.s, st2.s, st2.j FROM st1 INNER JOIN st2 ON st1.j = st2.j WHERE "
      "st2.j > 0 and st1.s <> 'foo' "
      "and st2.s <> 'foo' ORDER BY st1.i;",
      dt);
    // Non-sharded inner join (multi frag)
    c("SELECT st1.i, st3.i FROM st1 INNER JOIN st3 ON st1.j = st3.j ORDER BY st1.i;", dt);
    c("SELECT st1.j, st3.j FROM st1 INNER JOIN st3 ON st1.j = st3.j ORDER BY st1.i;", dt);
    c("SELECT st1.j, st3.j FROM st1 INNER JOIN st3 ON st1.j = st3.j WHERE st3.j > -1 "
      "ORDER BY st1.i;",
      dt);
    c("SELECT st1.j, st3.j FROM st1 INNER JOIN st3 ON st1.j = st3.j WHERE st3.j > 0 "
      "ORDER BY st1.i;",
      dt);
    c("SELECT st1.j, st1.s, st3.j, st3.s FROM st1 INNER JOIN st3 ON st1.j = st3.j WHERE "
      "st3.j > 0 ORDER BY st1.i;",
      dt);
    c("SELECT st1.i, st1.j, st1.s, st3.i, st3.s, st3.j FROM st1 INNER JOIN st3 ON st1.j "
      "= st3.j WHERE st3.i > 0 ORDER "
      "BY st1.i;",
      dt);
    c("SELECT st1.i, st1.j, st1.s, st3.i, st3.s, st3.j FROM st1 INNER JOIN st3 ON st1.j "
      "= st3.j WHERE st3.j > 0 ORDER "
      "BY st1.i;",
      dt);
    c("SELECT st1.j, st1.s, st3.s, st3.j FROM st1 INNER JOIN st3 ON st1.j = st3.j WHERE "
      "st3.j > 0 ORDER BY st1.i;",
      dt);
    c("SELECT st1.j, st1.s, st3.s, st3.j FROM st1 INNER JOIN st3 ON st1.j = st3.j WHERE "
      "st3.j > 0 and st1.s <> 'foo' "
      "and st3.s <> 'foo' ORDER BY st1.i;",
      dt);
  }
}

TEST(Select, Joins_Negative_ShardKey) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    size_t num_shards = 1;
    if (dt == ExecutorDeviceType::GPU && choose_shard_count() > 0) {
      num_shards = choose_shard_count();
    }

    std::string drop_ddl_1 = "DROP TABLE IF EXISTS shard_test_negative_1;";
    run_ddl_statement(drop_ddl_1);
    std::string drop_ddl_2 = "DROP TABLE IF EXISTS shard_test_negative_2;";
    run_ddl_statement(drop_ddl_2);

    std::string table_ddl_1 =
        "CREATE TABLE shard_test_negative_1 (i INTEGER, j TEXT ENCODING DICT(32), SHARD "
        "KEY(i)) WITH (shard_count = " +
        std::to_string(num_shards) + ");";
    run_ddl_statement(table_ddl_1);

    std::string table_ddl_2 =
        "CREATE TABLE shard_test_negative_2 (i INTEGER, j TEXT ENCODING DICT(32), SHARD "
        "KEY(i)) WITH (shard_count = " +
        std::to_string(num_shards) + ");";
    run_ddl_statement(table_ddl_2);

    for (size_t i = 0; i < 5; i++) {
      for (const auto table : {"shard_test_negative_1", "shard_test_negative_2"}) {
        const std::string insert_query{"INSERT INTO " + std::string(table) + " VALUES(" +
                                       std::to_string(i - 1) + ", " + std::to_string(i) +
                                       ");"};
        run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      }
    }

    ASSERT_EQ(static_cast<int64_t>(-1),
              v<int64_t>(run_simple_agg(
                  "SELECT i FROM shard_test_negative_1 WHERE i < 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(-1),
              v<int64_t>(run_simple_agg(
                  "SELECT i FROM shard_test_negative_2 WHERE i < 0;", dt)));

    ASSERT_EQ(static_cast<int64_t>(-1),
              v<int64_t>(run_simple_agg("SELECT t1.i FROM shard_test_negative_1 t1 INNER "
                                        "JOIN shard_test_negative_2 t2 "
                                        "ON t1.i = t2.i WHERE t2.i < 0;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(-1),
              v<int64_t>(run_simple_agg("SELECT t2.i FROM shard_test_negative_1 t1 INNER "
                                        "JOIN shard_test_negative_2 t2 "
                                        "ON t1.i = t2.i WHERE t1.i < 0;",
                                        dt)));

    ASSERT_EQ("0",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT t1.j FROM shard_test_negative_1 t1 INNER JOIN "
                  "shard_test_negative_2 t2 ON t1.i = t2.i WHERE t2.i < 0;",
                  dt))));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT t1.i FROM shard_test_negative_1 t1 INNER "
                                        "JOIN shard_test_negative_2 t2 "
                                        "ON t1.i = t2.i WHERE t2.i > 2;",
                                        dt)));
    ASSERT_EQ("4",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT t1.j FROM shard_test_negative_1 t1 INNER JOIN "
                  "shard_test_negative_2 t2 ON t1.i = t2.i WHERE t2.i > 2;",
                  dt))));
  }
}

TEST(Select, Joins_InnerJoin_AtLeastThreeTables) {
  SKIP_ALL_ON_AGGREGATOR();

  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON b.str = c.str;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON b.str = c.str JOIN "
      "join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT a.y, count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str "
      "GROUP BY a.y;",
      dt);
    c("SELECT a.x AS x, a.y, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = "
      "c.str "
      "ORDER BY a.y;",
      dt);
    c("SELECT a.x, b.x, b.str, c.str FROM test AS a JOIN join_test AS b ON a.x = b.x "
      "JOIN test_inner AS c ON b.x = c.x "
      "ORDER BY b.str;",
      dt);
    c("SELECT a.x, b.x, c.x FROM test a JOIN test_inner b ON a.x = b.x JOIN join_test c "
      "ON b.x = c.x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str JOIN "
      "hash_join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str JOIN "
      "join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON b.str = c.str JOIN "
      "hash_join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT a.x AS x, a.y, b.str FROM test AS a JOIN hash_join_test AS b ON a.x = b.x "
      "JOIN test_inner AS c ON b.str "
      "= c.str "
      "ORDER BY a.y;",
      dt);
    c("SELECT a.x, b.x, c.x FROM test a JOIN test_inner b ON a.x = b.x JOIN "
      "hash_join_test c ON b.x = c.x;",
      dt);
    c("SELECT a.x, b.x FROM test_inner a JOIN test_inner b ON a.x = b.x ORDER BY a.x;",
      dt);
    c("SELECT a.x, b.x FROM join_test a JOIN join_test b ON a.x = b.x ORDER BY a.x;", dt);
    c("SELECT COUNT(1) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON a.t = c.x;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN test_inner b ON a.str = b.str JOIN "
      "hash_join_test c ON a.x = c.x JOIN "
      "join_test d ON a.x > d.x;",
      dt);
    c("SELECT a.x, b.str, c.str, d.y FROM hash_join_test a JOIN test b ON a.x = b.x JOIN "
      "join_test c ON b.x = c.x JOIN "
      "test_inner d ON b.x = d.x ORDER BY a.x, b.str;",
      dt);
  }

  g_enable_watchdog = save_watchdog;
}

TEST(Select, Joins_InnerJoin_Filters) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON b.str = c.str WHERE a.y "
      "< 43;",
      dt);
    c("SELECT SUM(a.x), b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str "
      "WHERE a.y "
      "= 43 group by b.str;",
      dt);
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.str = test_inner.str AND test.x "
      "= 7;",
      dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT test.x, test_inner.str FROM test JOIN test_inner ON test.str = "
          "test_inner.str AND test.x <> 7;",
          dt));
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str "
      "WHERE a.y "
      "< 43;",
      dt);
    c("SELECT SUM(a.x), b.str FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = "
      "c.str "
      "WHERE a.y "
      "= 43 group by b.str;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.x = b.x JOIN test_inner c ON "
      "c.str = a.str WHERE c.str = "
      "'foo';",
      dt);
    c("SELECT COUNT(*) FROM test t1 JOIN test t2 ON t1.x = t2.x WHERE t1.y > t2.y;", dt);
    c("SELECT COUNT(*) FROM test t1 JOIN test t2 ON t1.x = t2.x WHERE t1.null_str = "
      "t2.null_str;",
      dt);
  }
}

TEST(Select, Joins_LeftOuterJoin) {
  SKIP_ALL_ON_AGGREGATOR();
  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = "
      "test_inner.x ORDER BY test.x ASC;",
      dt);
    c("SELECT test.x key1, CASE WHEN test_inner.x IS NULL THEN 99 ELSE test_inner.x END "
      "key2 FROM test LEFT OUTER JOIN "
      "test_inner ON test.x = test_inner.x GROUP BY key1, key2 ORDER BY key1;",
      dt);
    c("SELECT test_inner.x key1 FROM test LEFT OUTER JOIN test_inner ON test.x = "
      "test_inner.x GROUP BY key1 HAVING "
      "key1 IS NOT NULL;",
      dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT JOIN test b ON a.x = b.x;", dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT OUTER JOIN test_x b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str;",
      dt);
    c("SELECT COUNT(*) FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str;",
      dt);
    c("SELECT a.x, b.str FROM test_inner a LEFT OUTER JOIN test_x b ON a.x = b.x ORDER "
      "BY a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str "
      "ORDER BY a.x, b.str IS NULL, "
      "b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str "
      "ORDER BY a.x, b.str IS NULL, "
      "b.str;",
      dt);
    c("SELECT COUNT(*) FROM test_inner_x a LEFT JOIN test_x b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) FROM test a LEFT JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT COUNT(*) FROM test a LEFT JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT a.x, b.str FROM test_inner_x a LEFT JOIN test_x b ON a.x = b.x ORDER BY "
      "a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT JOIN join_test b ON a.str = b.dup_str ORDER BY "
      "a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT JOIN join_test b ON a.str = b.dup_str ORDER BY "
      "a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x = test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x < test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x > test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x >= test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x <= test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT test_inner.y, COUNT(*) n FROM test LEFT JOIN test_inner ON test_inner.x = "
      "test.x WHERE test_inner.str = "
      "'foo' GROUP BY test_inner.y ORDER BY n DESC;",
      dt);
    c("SELECT a.x, COUNT(b.y) FROM test a LEFT JOIN test_inner b ON b.x = a.x AND b.str "
      "NOT LIKE 'box' GROUP BY a.x "
      "ORDER BY a.x;",
      dt);
    c("SELECT a.x FROM test a LEFT OUTER JOIN test_inner b ON TRUE ORDER BY a.x ASC;",
      "SELECT a.x FROM test a LEFT OUTER JOIN test_inner b ON 1 ORDER BY a.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x > test_inner.x INNER "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x > test_inner.x INNER "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x = test_inner.x INNER "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x = test_inner.x INNER "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner "
      "ON test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner "
      "ON test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner "
      "ON test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner "
      "ON test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test.str = test_inner.str AND "
      "test.x = test_inner.x;",
      dt);
  }
  g_enable_watchdog = save_watchdog;
}

TEST(Select, Joins_LeftJoin_Filters) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = "
      "test_inner.x WHERE test.y > 40 "
      "ORDER BY test.x ASC;",
      dt);
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = "
      "test_inner.x WHERE test.y > 42 "
      "ORDER BY test.x ASC;",
      dt);
    c("SELECT test.str AS foobar, test_inner.str FROM test LEFT OUTER JOIN test_inner ON "
      "test.x = test_inner.x WHERE "
      "test.y > 42 ORDER BY foobar DESC LIMIT 8;",
      dt);
    c("SELECT test.x AS foobar, test_inner.x AS inner_foobar, test.f AS f_foobar FROM "
      "test LEFT OUTER JOIN test_inner "
      "ON test.str = test_inner.str WHERE test.y > 40 ORDER BY foobar DESC, f_foobar "
      "DESC;",
      dt);
    c("SELECT test.str AS foobar, test_inner.str FROM test LEFT OUTER JOIN test_inner ON "
      "test.x = test_inner.x WHERE "
      "test_inner.str IS NOT NULL ORDER BY foobar DESC;",
      dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT JOIN (SELECT * FROM test WHERE y > 40) b "
      "ON a.x = b.x;",
      dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN (SELECT * FROM test WHERE y > 40) b "
      "ON a.x = b.x ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT COUNT(*) FROM join_test a LEFT JOIN test b ON a.x = b.x AND a.x = 7;", dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x AND a.x = 7 "
      "ORDER BY a.x, b.str;",
      dt);
    c("SELECT COUNT(*) FROM join_test a LEFT JOIN test b ON a.x = b.x WHERE a.x = 7;",
      dt);
    c("SELECT a.x FROM join_test a LEFT JOIN test b ON a.x = b.x WHERE a.x = 7;", dt);
  }
}

TEST(Select, Joins_MultiCompositeColumns) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT a.x, b.str FROM test AS a JOIN join_test AS b ON a.str = b.str AND a.x = "
      "b.x ORDER BY a.x, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x AND a.str = "
      "b.str ORDER BY a.x, b.str;",
      dt);
    c("SELECT a.z, b.str FROM test a JOIN join_test b ON a.y = b.y AND a.x = b.x ORDER "
      "BY a.z, b.str;",
      dt);
    c("SELECT a.z, b.str FROM test a JOIN test_inner b ON a.y = b.y AND a.x = b.x ORDER "
      "BY a.z, b.str;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.x = b.x AND a.y = b.x JOIN "
      "test_inner c ON a.x = c.x WHERE "
      "c.str <> 'foo';",
      dt);
    c("SELECT a.x, b.x, d.str FROM test a JOIN test_inner b ON a.str = b.str JOIN "
      "hash_join_test c ON a.x = c.x JOIN "
      "join_test d ON a.x >= d.x AND a.x < d.x + 5 ORDER BY a.x, b.x;",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.x = join_test.x OR (test.x IS "
      "NULL AND join_test.x IS NULL)) "
      "AND (test.y = join_test.y OR (test.y IS NULL AND join_test.y IS NULL));",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.str = join_test.dup_str OR "
      "(test.str IS NULL AND "
      "join_test.dup_str IS NULL)) AND (test.x = join_test.x OR (test.x IS NULL AND "
      "join_test.x IS NULL));",
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

TEST(Select, Joins_CoalesceColumns) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.x = t1.x AND t0.y = t1.y;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.x = t1.x AND t0.str = t1.str;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.str = t1.str AND t0.dup_str = t1.dup_str;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.str = t1.str AND t0.dup_str = t1.dup_str AND t0.x = t1.x;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_1 t1 "
      "ON t0.x = t1.x AND t0.y = t1.y INNER JOIN coalesce_cols_test_2 t2 on t0.x = t2.x "
      "AND t1.y = t2.y;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.x = t1.x AND t0.d = t1.d;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.x = t1.x AND t0.d = t1.d AND t0.y = t1.y;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.d = t1.d AND t0.x = t1.x;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.d = t1.d AND t0.tz = t1.tz AND t0.x = t1.x;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.dn = t1.dn AND t0.tz = t1.tz AND t0.y = t1.y AND t0.x = t1.x;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_2 t1 "
      "ON t0.dn = t1.dn AND t0.y = t1.y AND t0.tz = t1.tz AND t0.x = t1.x;",
      dt);

    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_1 t1 "
      "ON t0.dn = t1.dn AND t0.y = t1.y AND t0.tz = t1.tz AND t0.x = t1.x INNER JOIN "
      "coalesce_cols_test_2 t2 ON t0.y = t2.y AND t0.tz = t1.tz AND t0.x = t1.x;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_1 t1 "
      "ON t0.dn = t1.dn AND t0.y = t1.y AND t0.tz = t1.tz AND t0.x = t1.x INNER JOIN "
      "coalesce_cols_test_2 t2 ON t0.d = t2.d AND t0.tz = t1.tz AND t0.x = t1.x;",
      dt);
    c("SELECT COUNT(*) FROM coalesce_cols_test_0 t0 INNER JOIN coalesce_cols_test_1 t1 "
      "ON t0.dn = t1.dn AND t0.str = t1.str AND t0.tz = t1.tz AND t0.x = t1.x INNER JOIN "
      "coalesce_cols_test_2 t2 ON t0.y = t2.y AND t0.tz = t1.tz AND t0.x = t1.x;",
      dt);
  }
}

TEST(Select, Joins_ComplexQueries) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test a JOIN (SELECT * FROM test WHERE y < 43) b ON a.x = b.x "
      "JOIN join_test c ON a.x = c.x "
      "WHERE a.fixed_str = 'foo';",
      dt);
    c("SELECT * FROM (SELECT a.y, b.str FROM test a JOIN join_test b ON a.x = b.x) ORDER "
      "BY y, str;",
      dt);
    c("SELECT x, dup_str FROM (SELECT * FROM test a JOIN join_test b ON a.x = b.x) WHERE "
      "y > 40 ORDER BY x, dup_str;",
      dt);
    c("SELECT a.x FROM (SELECT * FROM test WHERE x = 8) AS a JOIN (SELECT * FROM "
      "test_inner WHERE x = 7) AS b ON a.str "
      "= b.str WHERE a.y < 42;",
      dt);
    c("SELECT a.str as key0,a.fixed_str as key1,COUNT(*) AS color FROM test a JOIN "
      "(select str,count(*) "
      "from test group by str order by COUNT(*) desc limit 40) b on a.str=b.str JOIN "
      "(select "
      "fixed_str,count(*) from test group by fixed_str order by count(*) desc limit 40) "
      "c on "
      "c.fixed_str=a.fixed_str GROUP BY key0, key1 ORDER BY key0,key1;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN (SELECT str FROM test) b ON a.str = b.str OR "
      "false;",
      "SELECT COUNT(*) FROM test a JOIN (SELECT str FROM test) b ON a.str = b.str OR 0;",
      dt);
  }
}

TEST(Select, Joins_TimeAndDate) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // Inner joins
    c("SELECT COUNT(*) FROM test a, test b WHERE a.m = b.m;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.n = b.n;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o;", dt);
    // TODO(adb): The following tests use too much memory for TSAN. Commenting out for
    // now, will re-enable once we have bucketed perfect hash or overlaps hash for date
#if 0
    c("SELECT COUNT(*) FROM test a, test_inner b WHERE a.m = b.ts;", dt);
    c("SELECT COUNT(*) FROM test a, test_inner b WHERE a.o = b.dt;", dt);
    c("SELECT COUNT(*) FROM test a, test_inner b WHERE a.o = b.dt32;", dt);
    c("SELECT COUNT(*) FROM test a, test_inner b WHERE a.o = b.dt16;", dt);

    // Inner joins across types
    c("SELECT COUNT(*) FROM test_inner a, test_inner b WHERE a.dt = b.dt;", dt);
    c("SELECT COUNT(*) FROM test_inner a, test_inner b WHERE a.dt32 = b.dt;", dt);
    c("SELECT COUNT(*) FROM test_inner a, test_inner b WHERE a.dt16 = b.dt;", dt);
    c("SELECT COUNT(*) FROM test_inner a, test_inner b WHERE a.dt = b.dt32;", dt);
    c("SELECT COUNT(*) FROM test_inner a, test_inner b WHERE a.dt32 = b.dt32;", dt);
    c("SELECT COUNT(*) FROM test_inner a, test_inner b WHERE a.dt = b.dt16;", dt);
    c("SELECT COUNT(*) FROM test_inner a, test_inner b WHERE a.dt32 = b.dt16;", dt);
    c("SELECT COUNT(*) FROM test_inner a, test_inner b WHERE a.dt16 = b.dt16;", dt);
#endif

    // Outer joins
    c("SELECT a.x, a.o, b.dt FROM test a JOIN test_inner b ON a.o = b.dt;", dt);
  }
}

TEST(Select, Joins_OneOuterExpression) {
  SKIP_ALL_ON_AGGREGATOR();

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

class JoinTest : public ::testing::Test {
 protected:
  virtual ~JoinTest() {}

  virtual void SetUp() override {
    auto create_test_table = [](const std::string& table_name,
                                const size_t num_records,
                                const size_t start_index = 0) {
      run_ddl_statement("DROP TABLE IF EXISTS " + table_name);

      const std::string columns_definition{
          "x int not null, y int, str text encoding dict"};
      const auto table_create = build_create_table_statement(
          columns_definition, table_name, {"", 0}, {}, 50, true, false);
      run_ddl_statement(table_create);

      TestHelpers::ValuesGenerator gen(table_name);
      const std::unordered_map<int, std::string> str_map{
          {0, "'foo'"}, {1, "'bar'"}, {2, "'hello'"}, {3, "'world'"}};
      for (size_t i = start_index; i < start_index + num_records; i++) {
        const auto insert_query = gen(i, i, str_map.at(i % 4));
        run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      }
    };

    create_test_table("jointest_a", 20, 0);
    create_test_table("jointest_b", 0, 0);
    create_test_table("jointest_c", 20, 10);
  }

  virtual void TearDown() override {
    if (!g_keep_test_data) {
      auto execute_drop_table = [](const std::string& table_name) {
        const std::string ddl = "DROP TABLE " + table_name;
        run_ddl_statement(ddl);
      };

      execute_drop_table("jointest_a");
      execute_drop_table("jointest_b");
      execute_drop_table("jointest_c");
    }
  }
};

TEST_F(JoinTest, EmptyJoinTables) {
  const auto table_reordering_state = g_from_table_reordering;
  g_from_table_reordering = false;  // disable from table reordering

  SKIP_ALL_ON_AGGREGATOR();  // relevant for single node only

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM jointest_a a INNER JOIN "
                                        "jointest_b b ON a.x = b.x;",
                                        dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM jointest_b b INNER JOIN "
                                        "jointest_a a ON a.x = b.x;",
                                        dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM jointest_c c INNER JOIN "
                                        "(SELECT a.x FROM jointest_a a INNER JOIN "
                                        "jointest_b b ON a.x = b.x) as j ON j.x = c.x;",
                                        dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM jointest_a a INNER JOIN "
                                        "jointest_b b ON a.str = b.str;",
                                        dt)));

    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM jointest_b b INNER JOIN "
                                        "jointest_a a ON a.str = b.str;",
                                        dt)));
  }

  g_from_table_reordering = table_reordering_state;
}

TEST(Select, Joins_MultipleOuterExpressions) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x - 1 = test_inner.x AND "
      "test.str = test_inner.str;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 0 = test_inner.x AND "
      "test.str = test_inner.str;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str AND test.x "
      "+ 0 = test_inner.x;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 1 = test_inner.x AND "
      "test.str = test_inner.str;",
      dt);
    // The following query will fallback to loop join because we don't reorder the
    // expressions to be consistent with table order for composite equality yet.
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 0 = test_inner.x AND "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o + INTERVAL '0' DAY = b.o AND a.str "
      "= b.str;",
      "SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o AND a.str = b.str;",
      dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o + INTERVAL '0' DAY = b.o AND a.x = "
      "b.x;",
      "SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o AND a.x = b.x;",
      dt);
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
              v<int64_t>(
                  run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-dd) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(x - 7) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(x - 7) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(x - 8) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(x - 8) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(y - 42) = 0;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(y - 42) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(y - 43) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(y - 43) = 0;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-f) = -1;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2 * g_num_rows),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(-d) = -1;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows + g_num_rows / 2),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE SIGN(ofd) = 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows + g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(-ofd) = -1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows / 2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM test WHERE SIGN(ofd) IS NULL;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg(
                        "SELECT SUM(SIN(x) * SIN(x) + COS(x) * COS(x)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg(
                        "SELECT SUM(SIN(f) * SIN(f) + COS(f) * COS(f)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2 * g_num_rows),
                    v<double>(run_simple_agg(
                        "SELECT SUM(SIN(d) * SIN(d) + COS(d) * COS(d)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(2 * g_num_rows),
        v<double>(run_simple_agg(
            "SELECT SUM(SIN(dd) * SIN(dd) + COS(dd) * COS(dd)) FROM test;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2),
                    v<double>(run_simple_agg(
                        "SELECT FLOOR(CAST(2.3 AS double)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(2),
                    v<float>(run_simple_agg(
                        "SELECT FLOOR(CAST(2.3 AS float)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT FLOOR(CAST(2.3 AS BIGINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT FLOOR(CAST(2.3 AS SMALLINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT FLOOR(CAST(2.3 AS INT)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(2),
        v<double>(run_simple_agg("SELECT FLOOR(2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(2),
        v<double>(run_simple_agg("SELECT FLOOR(2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(-3),
        v<double>(run_simple_agg("SELECT FLOOR(-2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(-2),
        v<double>(run_simple_agg("SELECT FLOOR(-2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(3),
                    v<double>(run_simple_agg(
                        "SELECT CEIL(CAST(2.3 AS double)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(3),
                    v<float>(run_simple_agg(
                        "SELECT CEIL(CAST(2.3 AS float)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT CEIL(CAST(2.3 AS BIGINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT CEIL(CAST(2.3 AS SMALLINT)) FROM test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT CEIL(CAST(2.3 AS INT)) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(3),
                    v<double>(run_simple_agg("SELECT CEIL(2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(static_cast<double>(2),
                    v<double>(run_simple_agg("SELECT CEIL(2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(-2),
        v<double>(run_simple_agg("SELECT CEIL(-2.3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(-2),
        v<double>(run_simple_agg("SELECT CEIL(-2.0) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<float>(4129511.320307),
        v<double>(run_simple_agg("SELECT DISTANCE_IN_METERS(-74.0059, "
                                 "40.7217,-122.416667 , 37.783333) FROM test LIMIT 1;",
                                 dt)));
    ASSERT_FLOAT_EQ(
        static_cast<int64_t>(1000),
        v<int64_t>(run_simple_agg(
            "SELECT TRUNCATE(CAST(1171 AS SMALLINT),-3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<float>(1000),
        v<float>(run_simple_agg(
            "SELECT TRUNCATE(CAST(1171.123 AS FLOAT),-3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(1000),
        v<double>(run_simple_agg(
            "SELECT TRUNCATE(CAST(1171.123 AS DOUBLE),-3) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(1171.10),
        v<double>(run_simple_agg(
            "SELECT TRUNCATE(CAST(1171.123 AS DOUBLE),1) FROM test LIMIT 1;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<float>(1171.11),
        v<float>(run_simple_agg(
            "SELECT TRUNCATE(CAST(1171.113 AS FLOAT),2) FROM test LIMIT 1;", dt)));
  }
}

TEST(Select, TextGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg(" select count(*) from (SELECT tnone, count(*) cc from "
                                  "text_group_by_test group by tnone);",
                                  dt),
                 std::runtime_error);
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("select count(*) from (SELECT tdict, count(*) cc "
                                        "from text_group_by_test group by tdict)",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("select count(*) from (SELECT tdef, count(*) cc "
                                        "from text_group_by_test group by tdef)",
                                        dt)));
  }
}

TEST(Select, UnsupportedExtensions) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT TRUNCATE(2016, 1) FROM test LIMIT 1;", dt),
                 std::runtime_error);
  }
}

TEST(Select, UnsupportedSortOfIntermediateResult) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT real_str FROM test ORDER BY x;", dt),
                 std::runtime_error);
  }
}

TEST(Select, Views) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x, COUNT(*) FROM view_test WHERE y > 41 GROUP BY x;", dt);
    c("SELECT x FROM join_view_test WHERE x IS NULL;", dt);
  }
}

TEST(Select, Views_With_Subquery) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x, COUNT(*) FROM view_test WHERE y < (SELECT max(y) FROM test) GROUP BY x;",
      dt);
  }
}

TEST(Select, CreateTableAsSelect) {
  SKIP_ALL_ON_AGGREGATOR();

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
    c("SELECT str, SUM(x), COUNT(str) FROM test WHERE \"y\" = 42 AND str = 'Shim All The "
      "Things!' GROUP BY str;",
      dt);
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
    c("SELECT COUNT(*) FROM TEST WHERE x IN (SELECT x + 1 AS foo FROM test GROUP BY foo "
      "ORDER BY COUNT(*) DESC LIMIT "
      "1);",
      dt);
  }
}

TEST(Select, ArrowOutput) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c_arrow("SELECT str, COUNT(*) FROM test GROUP BY str ORDER BY str ASC;", dt);
    c_arrow("SELECT x, y, z, t, f, d, str, ofd, ofq FROM test ORDER BY x ASC, y ASC;",
            dt);
    c_arrow("SELECT null_str, COUNT(*) FROM test GROUP BY null_str;", dt);
  }
}

TEST(Select, WatchdogTest) {
  g_enable_watchdog = true;
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x, SUM(f) AS n FROM test GROUP BY x ORDER BY n DESC LIMIT 5;", dt);
    c("SELECT COUNT(*) FROM test WHERE str = "
      "'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz';",
      dt);
  }
}

TEST(Select, PuntToCPU) {
  SKIP_ALL_ON_AGGREGATOR();

  const auto cpu_retry_state = g_allow_cpu_retry;
  const auto watchdog_state = g_enable_watchdog;
  g_allow_cpu_retry = false;
  g_enable_watchdog = true;
  ScopeGuard reset_global_flag_state = [&cpu_retry_state, &watchdog_state] {
    g_allow_cpu_retry = cpu_retry_state;
    g_enable_watchdog = watchdog_state;
    g_gpu_mem_limit_percent = 0.9;  // Reset to 90%
  };

  const auto dt = ExecutorDeviceType::GPU;
  if (skip_tests(dt)) {
    return;
  }

  g_gpu_mem_limit_percent = 1e-10;
  EXPECT_THROW(run_multiple_agg("SELECT x, COUNT(*) FROM test GROUP BY x;", dt),
               std::runtime_error);
  EXPECT_THROW(run_multiple_agg("SELECT str, COUNT(*) FROM test GROUP BY str;", dt),
               std::runtime_error);
}

TEST(Select, TimestampMeridiesEncoding) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    run_ddl_statement("DROP TABLE IF EXISTS ts_meridies;");
    EXPECT_NO_THROW(run_ddl_statement("CREATE TABLE ts_meridies (ts TIMESTAMP(0));"));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies VALUES('2012-01-01 12:00:00 AM');", dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies VALUES('2012-01-01 12:00:00 a.m.');", dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies VALUES('2012-01-01 12:00:00 PM');", dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies VALUES('2012-01-01 12:00:00 p.m.');", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO ts_meridies VALUES('2012-01-01 3:00:00 AM');", dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies VALUES('2012-01-01 3:00:00 a.m.');", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO ts_meridies VALUES('2012-01-01 3:00:00 PM');", dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies VALUES('2012-01-01 3:00:00 p.m.');", dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies VALUES('2012-01-01 7:00:00.3456 AM');", dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies VALUES('2012-01-01 7:00:00.3456 p.m.');", dt));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM ts_meridies where extract(epoch from ts) = 1325376000;",
            dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM ts_meridies where extract(epoch from ts) = 1325419200;",
            dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM ts_meridies where extract(epoch from ts) = 1325386800;",
            dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM ts_meridies where extract(epoch from ts) = 1325430000;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM ts_meridies where extract(epoch from ts) = 1325401200;",
            dt)));
    ASSERT_EQ(
        1,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM ts_meridies where extract(epoch from ts) = 1325444400;",
            dt)));
  }
}

#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
TEST(Select, TimestampPrecisionMeridiesEncoding) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    run_ddl_statement("DROP TABLE IF EXISTS ts_meridies_precisions;");
    EXPECT_NO_THROW(
        run_ddl_statement("CREATE TABLE ts_meridies_precisions (ts3 TIMESTAMP(3), ts6 "
                          "TIMESTAMP(6), ts9 TIMESTAMP(9));"));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies_precisions VALUES('2012-01-01 12:00:00.123 AM', "
        "'2012-01-01 12:00:00.123456 AM', '2012-01-01 12:00:00.123456789 AM');",
        dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies_precisions VALUES('2012-01-01 12:00:00.123 a.m.', "
        "'2012-01-01 12:00:00.123456 a.m.', '2012-01-01 12:00:00.123456789 a.m.');",
        dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies_precisions VALUES('2012-01-01 12:00:00.123 PM', "
        "'2012-01-01 12:00:00.123456 PM', '2012-01-01 12:00:00.123456789 PM');",
        dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies_precisions VALUES('2012-01-01 12:00:00.123 p.m.', "
        "'2012-01-01 12:00:00.123456 p.m.', '2012-01-01 12:00:00.123456789 p.m.');",
        dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies_precisions VALUES('2012-01-01 3:00:00.123 AM', "
        "'2012-01-01 3:00:00.123456 AM', '2012-01-01 3:00:00.123456789 AM');",
        dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies_precisions VALUES('2012-01-01 3:00:00.123 a.m.', "
        "'2012-01-01 3:00:00.123456 a.m.', '2012-01-01 3:00:00.123456789 a.m.');",
        dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies_precisions VALUES('2012-01-01 3:00:00.123 PM', "
        "'2012-01-01 3:00:00.123456 PM', '2012-01-01 3:00:00.123456789 PM');",
        dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_meridies_precisions VALUES('2012-01-01 3:00:00.123 p.m.', "
        "'2012-01-01 3:00:00.123456 p.m.', '2012-01-01 3:00:00.123456789 p.m.');",
        dt));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions "
                                        "where extract(epoch from ts3) = 1325376000123;",
                                        dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions where "
                                  "extract(epoch from ts6) = 1325376000123456;",
                                  dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions where "
                                  "extract(epoch from ts9) = 1325376000123456789;",
                                  dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions "
                                        "where extract(epoch from ts3) = 1325419200123;",
                                        dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions where "
                                  "extract(epoch from ts6) = 1325419200123456;",
                                  dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions where "
                                  "extract(epoch from ts9) = 1325419200123456789;",
                                  dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions "
                                        "where extract(epoch from ts3) = 1325386800123;",
                                        dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions where "
                                  "extract(epoch from ts6) = 1325386800123456;",
                                  dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions where "
                                  "extract(epoch from ts9) = 1325386800123456789;",
                                  dt)));
    ASSERT_EQ(2,
              v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions "
                                        "where extract(epoch from ts3) = 1325430000123;",
                                        dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions where "
                                  "extract(epoch from ts6) = 1325430000123456;",
                                  dt)));
    ASSERT_EQ(
        2,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM ts_meridies_precisions where "
                                  "extract(epoch from ts9) = 1325430000123456789;",
                                  dt)));
  }
}
#endif

#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
TEST(Select, TimestampPrecision) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    /* ---DATE TRUNCATE--- */
    ASSERT_EQ(978307200000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millennium, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(978307200000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(century, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1293840000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1388534400000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(year, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1417392000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(month, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1417910400000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(week, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(
        1418428800000L,
        v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(day, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418508000000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(hour, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509380000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(minute, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(second, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millisecond, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(microsecond, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(nanosecond, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(-30578688000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millennium, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(-2177452800000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(century, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(662688000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(915148800000000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(year, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(930787200000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(month, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931651200000000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(week, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(
        931651200000000L,
        v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(day, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701600000000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(hour, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701720000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(minute, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(second, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millisecond, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(microsecond, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(nanosecond, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(978307200000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millennium, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(978307200000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(century, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(978307200000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(decade, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1136073600000000000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(year, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1143849600000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(month, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1145750400000000000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(week, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(
        1146009600000000000L,
        v<int64_t>(run_simple_agg("SELECT DATE_TRUNC(day, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146020400000000000L,
              v<int64_t>(
                  run_simple_agg("SELECT DATE_TRUNC(hour, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023340000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(minute, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(second, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344607000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(millisecond, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344607435000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(microsecond, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATE_TRUNC(nanosecond, m_9) FROM test limit 1;", dt)));
    /* ---Extract --- */
    ASSERT_EQ(1146023344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(epoch from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(nanosecond from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(607435L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(microsecond from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(607L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(millisecond from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(second from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(49L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(minute from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(hour from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(dow from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(isodow from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(17L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(week from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(26L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(day from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(116L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(doy from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(month from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(quarter from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(quarterday from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(2006L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(year from m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(epoch from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(nanosecond from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(microsecond from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(874L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(millisecond from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(53L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(second from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(minute from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(14L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(hour from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(0L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(dow from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(7L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(isodow from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(29L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(week from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(11L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(day from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(192L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(doy from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(7L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(month from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(quarter from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(quarterday from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1999L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(year from m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(epoch from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(nanosecond from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(microsecond from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(323L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(millisecond from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(15L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(second from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(23L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(minute from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(22L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(hour from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(6L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(dow from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(6L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(isodow from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(50L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(week from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(13L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(day from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(347L,
              v<int64_t>(
                  run_simple_agg("SELECT EXTRACT(doy from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(12L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(month from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(quarter from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(quarterday from m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(2014L,
              v<int64_t>(run_simple_agg(
                  "SELECT EXTRACT(year from m_3) FROM test limit 1;", dt)));
    /* ---INTERVAL --- */
    ASSERT_EQ(1177559344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 + INTERVAL '1' year) from test limit 1;", dt)));
    ASSERT_EQ(1148615344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 + INTERVAL '1' month) from test limit 1;", dt)));
    ASSERT_EQ(1146109744607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 + INTERVAL '1' day) from test limit 1;", dt)));
    ASSERT_EQ(1146026944607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 + INTERVAL '1' hour) from test limit 1;", dt)));
    ASSERT_EQ(1146023404607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 + INTERVAL '1' minute) from test limit 1;", dt)));
    ASSERT_EQ(1146023345607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 + INTERVAL '1' second) from test limit 1;", dt)));
    ASSERT_EQ(1114487344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 - INTERVAL '1' year) from test limit 1;", dt)));
    ASSERT_EQ(1143344944607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 - INTERVAL '1' month) from test limit 1;", dt)));
    ASSERT_EQ(1145936944607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 - INTERVAL '1' day) from test limit 1;", dt)));
    ASSERT_EQ(1146019744607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 - INTERVAL '1' hour) from test limit 1;", dt)));
    ASSERT_EQ(1146023284607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 - INTERVAL '1' minute) from test limit 1;", dt)));
    ASSERT_EQ(1146023343607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_9 - INTERVAL '1' second) from test limit 1;", dt)));
    ASSERT_EQ(963324173874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 + INTERVAL '1' year) from test limit 1;", dt)));
    ASSERT_EQ(934380173874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 + INTERVAL '1' month) from test limit 1;", dt)));
    ASSERT_EQ(931788173874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 + INTERVAL '1' day) from test limit 1;", dt)));
    ASSERT_EQ(931705373874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 + INTERVAL '1' hour) from test limit 1;", dt)));
    ASSERT_EQ(931701833874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 + INTERVAL '1' minute) from test limit 1;", dt)));
    ASSERT_EQ(931701774874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 + INTERVAL '1' second) from test limit 1;", dt)));
    ASSERT_EQ(900165773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 - INTERVAL '1' year) from test limit 1;", dt)));
    ASSERT_EQ(929109773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 - INTERVAL '1' month) from test limit 1;", dt)));
    ASSERT_EQ(931615373874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 - INTERVAL '1' day) from test limit 1;", dt)));
    ASSERT_EQ(931698173874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 - INTERVAL '1' hour) from test limit 1;", dt)));
    ASSERT_EQ(931701713874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 - INTERVAL '1' minute) from test limit 1;", dt)));
    ASSERT_EQ(931701772874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_6 - INTERVAL '1' second) from test limit 1;", dt)));
    ASSERT_EQ(1450045395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 + INTERVAL '1' year) from test limit 1;", dt)));
    ASSERT_EQ(1421187795323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 + INTERVAL '1' month) from test limit 1;", dt)));
    ASSERT_EQ(1418595795323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 + INTERVAL '1' day) from test limit 1;", dt)));
    ASSERT_EQ(1418512995323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 + INTERVAL '1' hour) from test limit 1;", dt)));
    ASSERT_EQ(1418509455323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 + INTERVAL '1' minute) from test limit 1;", dt)));
    ASSERT_EQ(1418509396323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 + INTERVAL '1' second) from test limit 1;", dt)));
    ASSERT_EQ(1386973395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 - INTERVAL '1' year) from test limit 1;", dt)));
    ASSERT_EQ(1415917395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 - INTERVAL '1' month) from test limit 1;", dt)));
    ASSERT_EQ(1418422995323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 - INTERVAL '1' day) from test limit 1;", dt)));
    ASSERT_EQ(1418505795323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 - INTERVAL '1' hour) from test limit 1;", dt)));
    ASSERT_EQ(1418509335323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 - INTERVAL '1' minute) from test limit 1;", dt)));
    ASSERT_EQ(1418509394323L,
              v<int64_t>(run_simple_agg(
                  "SELECT (m_3 - INTERVAL '1' second) from test limit 1;", dt)));
    /* ---DATE PART --- */
    ASSERT_EQ(2014L,
              v<int64_t>(
                  run_simple_agg("SELECT DATEPART('year', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('quarter', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(12L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('month', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(347L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('dayofyear', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(
        13L,
        v<int64_t>(run_simple_agg("SELECT DATEPART('day', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(22L,
              v<int64_t>(
                  run_simple_agg("SELECT DATEPART('hour', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(23L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('minute', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(15L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('second', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('millisecond', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('microsecond', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('nanosecond', m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1999L,
              v<int64_t>(
                  run_simple_agg("SELECT DATEPART('year', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('quarter', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(7L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('month', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(192L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('dayofyear', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(
        11L,
        v<int64_t>(run_simple_agg("SELECT DATEPART('day', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(14L,
              v<int64_t>(
                  run_simple_agg("SELECT DATEPART('hour', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('minute', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(53L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('second', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(874L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('millisecond', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('microsecond', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(0L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('nanosecond', m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(2006L,
              v<int64_t>(
                  run_simple_agg("SELECT DATEPART('year', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('quarter', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('month', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(116L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('dayofyear', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(
        26L,
        v<int64_t>(run_simple_agg("SELECT DATEPART('day', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(3L,
              v<int64_t>(
                  run_simple_agg("SELECT DATEPART('hour', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(49L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('minute', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(4L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('second', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(607L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('millisecond', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(607435L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('microsecond', m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEPART('nanosecond', m_9) FROM test limit 1;", dt)));
    /* ---DATE ADD --- */
    ASSERT_EQ(1177559344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('year',1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1153885744607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('quarter', 1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1148615344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146109744607435125L,
              v<int64_t>(
                  run_simple_agg("SELECT DATEADD('day',1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146026944607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('hour', 1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023404607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('minute', 1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023403607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('second', 59, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344932435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('millisecond', 325 , m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344607960125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('microsecond', 525, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344607436000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('nanosecond', 875, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1026396173874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('year',3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(955461773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('quarter', 3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(947599373874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 6, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(932824973874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('day',13, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931734173874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('hour', 9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931704053874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('minute', 38, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701783874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('second', 10, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773885533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('millisecond', 11 , m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874678L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('microsecond', 145, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('nanosecond', 875, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1734128595323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('year',10, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1450045395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('quarter', 4, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1423866195323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('month', 2, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1419805395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('day',15, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418516595323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('hour', 2, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418510055323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('minute', 11, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509415323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('second', 20, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395553,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('millisecond', 230 , m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('microsecond', 145, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('nanosecond', 875, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('nanosecond', 145000, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395553L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('microsecond', 230000, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509396553L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('millisecond', 1230, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(931701774885533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('millisecond', 1011 , m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701774874678L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('microsecond', 1000145, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(
        931701774874533L,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('nanosecond', 1000000875, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023345932435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('millisecond', 1325 , m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023345607960125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEADD('microsecond', 1000525, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(
        1146023345607436000L,
        v<int64_t>(run_simple_agg(
            "SELECT DATEADD('nanosecond', 1000000875, m_9) FROM test limit 1;", dt)));

    /* ---DATE DIFF --- */
    ASSERT_EQ(1146023344607435125L - 931701773874533000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874533000L - 1146023344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344607435125L - 1418509395323000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323000000L - 1146023344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023344607435125L - 1418509395000000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395000000000L - 1146023344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ((1146023344607435125L - 931701773874533000L) / 1000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533000L - 1146023344607435125L) / 1000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1146023344607435125L - 1418509395323000000L) / 1000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323000000L - 1146023344607435125L) / 1000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((1146023344607435125L - 1418509395000000000L) / 1000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395000000000L - 1146023344607435125L) / 1000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ((1146023344607435125L - 931701773874533000L) / (1000L * 1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533000L - 1146023344607435125L) / (1000L * 1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1146023344607435125L - 1418509395323000000L) / (1000L * 1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323000000L - 1146023344607435125L) / (1000L * 1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((1146023344607435125L - 1418509395000000000L) / (1000L * 1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395000000000L - 1146023344607435125L) / (1000L * 1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ(((1146023344607435125L / 1000000000) - (931701773874533L / 1000000)),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(((931701773874533L / 1000000) - (1146023344607435125L / 1000000000)),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(((1146023344607435125L / 1000000000) - (1418509395323L / 1000)),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(((1418509395323L / 1000) - (1146023344607435125L / 1000000000)),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(((1146023344607435125L / 1000000000) - 1418509395L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - (1146023344607435125L / 1000000000)),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ((3572026L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((-3572026L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(-4541434L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((4541434L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((-4541434L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((4541434L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ((59533L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((-59533L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((-75690L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((75690L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((-75690L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((75690L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ((2480),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((-2480),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(-3153,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((3153),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((-3153),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((3153),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ((81),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((-81),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((-104),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((104),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((-104),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((104),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ((7),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_6, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((-7),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(-8,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_3, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((8),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_9, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((-8),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ((8),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_9, m) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874533L - 1418509395323000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323000L - 931701773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(931701773874533L - 1418509395000000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395000000L - 931701773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L - 1418509395323000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323000L - 931701773874533L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L - 1418509395000000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395000000L - 931701773874533L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L - 1418509395323000L) / (1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323000L - 931701773874533L) / (1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L - 1418509395000000L) / (1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395000000L - 931701773874533L) / (1000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L / 1000000 - 1418509395323L / 1000),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L / 1000 - 931701773874533L / 1000000),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L / 1000000 - 1418509395L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - 931701773874533L / 1000000),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L / 1000000 - 1418509395323 / 1000L) / (60),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L / 1000 - 931701773874533L / 1000000) / (60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L / 1000000 - 1418509395L) / (60),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - 931701773874533L / 1000000) / (60),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L / 1000000 - 1418509395323L / 1000) / (60L * 60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L / 1000 - 931701773874533L / 1000000) / (60L * 60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L / 1000000 - 1418509395L) / (60L * 60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - 931701773874533L / 1000000) / (60L * 60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L / 1000000 - 1418509395323L / 1000) / (60L * 60L * 24L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L / 1000 - 931701773874533L / 1000000) / (60L * 60L * 24L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((931701773874533L / 1000000 - 1418509395L) / (60L * 60L * 24L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - 931701773874533L / 1000000) / (60L * 60L * 24L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ(185,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(-185,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(185,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ(-185,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_6, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(-15,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_6, m) FROM test limit 1;", dt)));
    ASSERT_EQ(-15,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395000L - 1418509395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509395323L - 1418509395000L,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('nanosecond', m, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395000L - 1418509395323L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L - 1418509395000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('microsecond', m, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395000L - 1418509395323L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L - 1418509395000L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('millisecond', m, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - 1418509395323L / 1000),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L / 1000 - 1418509395L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('second', m, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - 1418509395323L / 1000) / (60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L / 1000 - 1418509395L) / (60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('minute', m, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - 1418509395323L / 1000) / (60L * 60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L / 1000 - 1418509395L) / (60L * 60L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('hour', m, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395L - 1418509395323L / 1000) / (60L * 60L * 24L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ((1418509395323L / 1000 - 1418509395L) / (60L * 60L * 24L),
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('day', m, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('month', m, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m_3, m) FROM test limit 1;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT DATEDIFF('year', m, m_3) FROM test limit 1;", dt)));
    /* ---TIMESTAMP ADD --- */
    ASSERT_EQ(1177559344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR,1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1153885744607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(QUARTER, 1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1148615344607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, 1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146109744607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY,1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146026944607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023404607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, 1, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1146023403607435125L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, 59, m_9) FROM test limit 1;", dt)));
    ASSERT_EQ(1026396173874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR,3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(955461773874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(QUARTER, 3, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(947599373874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, 6, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(932824973874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY,13, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931734173874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 9, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931704053874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, 38, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(931701783874533L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, 10, m_6) FROM test limit 1;", dt)));
    ASSERT_EQ(1734128595323L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(YEAR,10, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1450045395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(QUARTER, 4, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1423866195323L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MONTH, 2, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1419805395323L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(DAY,15, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418516595323L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(HOUR, 2, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418510055323L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(MINUTE, 11, m_3) FROM test limit 1;", dt)));
    ASSERT_EQ(1418509415323L,
              v<int64_t>(run_simple_agg(
                  "SELECT TIMESTAMPADD(SECOND, 20, m_3) FROM test limit 1;", dt)));

    // Precisions Cast Tests
    // TODO(Wamsi): Add test for microsecond and nanosecond
    // once the calcite issue regarding support above milliseconds is resolved
    // Timestamps
    ASSERT_EQ(1418509395000,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(m as TIMESTAMP(3)) FROM test limit 1;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where CAST(m as TIMESTAMP(3)) < m_3", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where CAST(m as TIMESTAMP(3)) > m_3;", dt)));
    ASSERT_EQ(1418509395,
              v<int64_t>(run_simple_agg(
                  "SELECT CAST(m_3 as TIMESTAMP(0)) FROM test limit 1;", dt)));
    ASSERT_EQ(g_num_rows + g_num_rows,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where CAST(m_3 as TIMESTAMP(0)) = m", dt)));
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM test where cast(m as timestamp(0)) between "
            "TIMESTAMP(0) '2014-12-13 22:23:14' and TIMESTAMP(0) '2014-12-13 22:23:15'",
            dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(m as timestamp(3)) between "
                  "TIMESTAMP(3) '2014-12-12 22:23:15.320' and TIMESTAMP(3) '2014-12-13 "
                  "22:23:15.323'",
                  dt)));
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM test where cast(m_3 as timestamp(0)) between "
            "TIMESTAMP(0) '2014-12-13 22:23:14' and TIMESTAMP(3) '2014-12-13 22:23:15'",
            dt)));
    ASSERT_EQ(g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(m_6 as timestamp(3)) between "
                  "TIMESTAMP(3) '2014-12-13 22:23:15.870' and TIMESTAMP(3) '2014-12-13 "
                  "22:23:15.875'",
                  dt)));
    ASSERT_EQ(
        g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM test where cast(m_6 as timestamp(0)) between "
            "TIMESTAMP(0) '2014-12-13 22:23:14' and TIMESTAMP(3) '2014-12-13 22:23:15'",
            dt)));
    ASSERT_EQ(g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(m_9 as timestamp(3)) between "
                  "TIMESTAMP(3) '2014-12-13 22:23:15.607' and TIMESTAMP(3) '2014-12-13 "
                  "22:23:15.608'",
                  dt)));
    ASSERT_EQ(
        g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM test where cast(m_9 as timestamp(0)) between "
            "TIMESTAMP(0) '2014-12-13 22:23:14' and TIMESTAMP(0) '2014-12-13 22:23:15'",
            dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM test where cast(m_9 as "
                                  "timestamp(0)) >= TIMESTAMP(0) '2014-12-13 22:23:14';",
                                  dt)));
    ASSERT_EQ(
        10,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM test where cast(m_9 as "
                                  "timestamp(0)) <= TIMESTAMP(0) '2014-12-13 22:23:14';",
                                  dt)));
    ASSERT_EQ(10,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(m_9 as timestamp(3)) > "
                  "TIMESTAMP(3) '2014-12-13 22:23:14.607';",
                  dt)));
    ASSERT_EQ(10,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(m_9 as timestamp(3)) < "
                  "TIMESTAMP(3) '2014-12-13 22:23:14.607';",
                  dt)));
    ASSERT_EQ(10,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(m_6 as timestamp(3)) >= "
                  "TIMESTAMP(3) '2014-12-13 22:23:14.607';",
                  dt)));
    ASSERT_EQ(10,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(m_6 as timestamp(3)) <= "
                  "TIMESTAMP(3) '2014-12-13 22:23:14.607';",
                  dt)));
    ASSERT_EQ(
        5,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM test where cast(m_6 as "
                                  "timestamp(0)) >= TIMESTAMP(0) '2014-12-14 22:23:14';",
                                  dt)));
    ASSERT_EQ(
        15,
        v<int64_t>(run_simple_agg("SELECT count(*) FROM test where cast(m_6 as "
                                  "timestamp(0)) <= TIMESTAMP(0) '2014-12-14 22:23:14';",
                                  dt)));
    ASSERT_EQ(5,
              v<int64_t>(run_simple_agg("SELECT count(*) FROM test where m_3 >= "
                                        "TIMESTAMP(3) '2014-12-13 22:23:15.607';",
                                        dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT count(*) FROM test where m_3 <= "
                                        "TIMESTAMP(3) '2014-12-13 22:23:15.607';",
                                        dt)));
    ASSERT_EQ(
        20,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM test where m_3 >= TIMESTAMP(0) '2014-12-14 22:23:14';",
            dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM test where m_3 <= TIMESTAMP(0) '2014-12-14 22:23:14';",
            dt)));
    ASSERT_EQ(20,
              v<int64_t>(run_simple_agg(
                  "select count(*) from test where cast(m_3 as timestamp(0)) = m;", dt)));
    ASSERT_EQ(20,
              v<int64_t>(run_simple_agg("select count(*) from test where m_3 > m;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("select count(*) from test where m_3 = m;", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("select count(*) from test where m_3 < m;", dt)));
    ASSERT_EQ(
        5, v<int64_t>(run_simple_agg("select count(*) from test where m_6 > m_3;", dt)));
    ASSERT_EQ(
        15, v<int64_t>(run_simple_agg("select count(*) from test where m_6 < m_3;", dt)));
    ASSERT_EQ(
        0, v<int64_t>(run_simple_agg("select count(*) from test where m_6 = m_3;", dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "select count(*) from test where cast(m_6 as timestamp(3)) = m_3;", dt)));
    ASSERT_EQ(
        15, v<int64_t>(run_simple_agg("select count(*) from test where m_9 > m_6;", dt)));
    ASSERT_EQ(
        5, v<int64_t>(run_simple_agg("select count(*) from test where m_9 < m_6;", dt)));
    ASSERT_EQ(
        0, v<int64_t>(run_simple_agg("select count(*) from test where m_9 = m_6;", dt)));
    ASSERT_EQ(
        0,
        v<int64_t>(run_simple_agg(
            "select count(*) from test where cast(m_9 as timestamp(3)) = m_3;", dt)));
    ASSERT_EQ(15,
              v<int64_t>(run_simple_agg("SELECT count(*) FROM test where m_3 = "
                                        "TIMESTAMP(3) '2014-12-13 22:23:15.323';",
                                        dt)));
    // Dates
    ASSERT_EQ(
        g_num_rows + g_num_rows / 2,
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM test where cast(o as timestamp(0)) between "
            "TIMESTAMP(0) '1999-09-08 22:23:14' and TIMESTAMP(0) '1999-09-09 22:23:15'",
            dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(o as timestamp(3)) between "
                  "TIMESTAMP(3) '1999-09-08 12:12:31.500' and TIMESTAMP(3) '1999-09-09 "
                  "22:23:15'",
                  dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(o as timestamp(0)) between "
                  "TIMESTAMP(3) '1999-09-09 12:12:31.500' and TIMESTAMP(3) '1999-09-09 "
                  "22:23:15.500'",
                  dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(o as timestamp(3)) between "
                  "TIMESTAMP(0) '1999-09-09 12:12:31.500' and TIMESTAMP(0) '1999-09-09 "
                  "22:23:15.500'",
                  dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(o as timestamp(3)) >= "
                  "TIMESTAMP(3) '1999-09-09 12:12:31.500'",
                  dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(
                  run_simple_agg("SELECT count(*) FROM test where cast(o as "
                                 "timestamp(3)) < TIMESTAMP(3) '1999-09-09 12:12:31.500'",
                                 dt)));
    ASSERT_EQ(g_num_rows + g_num_rows / 2,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(o as timestamp(3)) < m_3", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM test where cast(o as timestamp(3)) >= m_3", dt)));

    ASSERT_EQ(
        1418509395000000000,
        v<int64_t>(run_simple_agg(
            "SELECT PG_DATE_TRUNC('second', m_9) FROM test where cast(m_9 as "
            "timestamp(0)) between "
            "TIMESTAMP(0) '2014-12-13 22:23:14' and TIMESTAMP(0) '2014-12-13 22:23:15'",
            dt)));
    ASSERT_EQ(
        1418509395000,
        v<int64_t>(run_simple_agg(
            "SELECT PG_DATE_TRUNC('second', m_3) FROM test where cast(m_3 as "
            "timestamp(0)) between "
            "TIMESTAMP(0) '2014-12-13 22:23:14' and TIMESTAMP(0) '2014-12-13 22:23:15'",
            dt)));
    ASSERT_EQ(
        1418509395000000,
        v<int64_t>(run_simple_agg(
            "SELECT PG_DATE_TRUNC('second', m_6) FROM test where cast(m_6 as "
            "timestamp(0)) between "
            "TIMESTAMP(0) '2014-12-13 22:23:14' and TIMESTAMP(0) '2014-12-13 22:23:15'",
            dt)));
    ASSERT_EQ(
        1418509395,
        v<int64_t>(run_simple_agg(
            "SELECT PG_DATE_TRUNC('second', m) FROM test where cast(m as "
            "timestamp(3)) between "
            "TIMESTAMP(3) '2014-12-13 22:23:14' and TIMESTAMP(3) '2014-12-13 22:23:15'",
            dt)));
    ASSERT_EQ(1418509395000000000,
              v<int64_t>(run_simple_agg(
                  "SELECT PG_DATE_TRUNC('second', m_9) FROM test where cast(m_9 as "
                  "timestamp(3)) between "
                  "TIMESTAMP(3) '2014-12-13 22:23:14.323' and TIMESTAMP(3) '2014-12-13 "
                  "22:23:15.999'",
                  dt)));
    ASSERT_EQ(1418509395000000,
              v<int64_t>(run_simple_agg(
                  "SELECT PG_DATE_TRUNC('second', m_6) FROM test where cast(m_6 as "
                  "timestamp(3)) between "
                  "TIMESTAMP(3) '2014-12-13 22:23:14.323' and TIMESTAMP(3) '2014-12-13 "
                  "22:23:15.999'",
                  dt)));
  }
}
#endif

#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
TEST(Select, TimestampPrecisionFormat) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    run_ddl_statement("DROP TABLE IF EXISTS ts_format;");
    EXPECT_NO_THROW(
        run_ddl_statement("CREATE TABLE ts_format (ts_3 TIMESTAMP(3), ts_6 TIMESTAMP(6), "
                          "ts_9 TIMESTAMP(9));"));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO ts_format VALUES('2012-05-22 01:02:03', "
                         "'2012-05-22 01:02:03', '2012-05-22 01:02:03');",
                         dt));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO ts_format VALUES('2012-05-22 01:02:03.', "
                         "'2012-05-22 01:02:03.', '2012-05-22 01:02:03.');",
                         dt));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO ts_format VALUES('2012-05-22 01:02:03.0', "
                         "'2012-05-22 01:02:03.0', '2012-05-22 01:02:03.0');",
                         dt));

    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO ts_format VALUES('2012-05-22 01:02:03.1', "
                         "'2012-05-22 01:02:03.1', '2012-05-22 01:02:03.1');",
                         dt));
    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO ts_format VALUES('2012-05-22 01:02:03.10', "
                         "'2012-05-22 01:02:03.10', '2012-05-22 01:02:03.10');",
                         dt));

    EXPECT_NO_THROW(
        run_multiple_agg("INSERT INTO ts_format VALUES('2012-05-22 01:02:03.03Z', "
                         "'2012-05-22 01:02:03.03Z', '2012-05-22 01:02:03.03Z');",
                         dt));
    EXPECT_NO_THROW(run_multiple_agg(
        "INSERT INTO ts_format VALUES('2012-05-22 01:02:03.003046777Z', '2012-05-22 "
        "01:02:03.000003046777Z', '2012-05-22 01:02:03.000000003046777Z');",
        dt));

    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg("SELECT count(ts_3) FROM ts_format where "
                                        "extract(epoch from ts_3) = 1337648523000;",
                                        dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg("SELECT count(ts_3) FROM ts_format where "
                                        "extract(epoch from ts_3) = 1337648523100;",
                                        dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg("SELECT count(ts_3) FROM ts_format where "
                                        "extract(epoch from ts_3) = 1337648523030;",
                                        dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg("SELECT count(ts_3) FROM ts_format where "
                                        "extract(epoch from ts_3) = 1337648523003;",
                                        dt)));

    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "SELECT count(ts_6) FROM ts_format where extract(epoch from ts_6) = "
                  "1337648523000000;",
                  dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "SELECT count(ts_6) FROM ts_format where extract(epoch from ts_6) = "
                  "1337648523100000;",
                  dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT count(ts_6) FROM ts_format where extract(epoch from ts_6) = "
                  "1337648523030000;",
                  dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT count(ts_6) FROM ts_format where extract(epoch from ts_6) = "
                  "1337648523000003;",
                  dt)));

    ASSERT_EQ(3L,
              v<int64_t>(run_simple_agg(
                  "SELECT count(ts_9) FROM ts_format where extract(epoch from ts_9) = "
                  "1337648523000000000;",
                  dt)));
    ASSERT_EQ(2L,
              v<int64_t>(run_simple_agg(
                  "SELECT count(ts_9) FROM ts_format where extract(epoch from ts_9) = "
                  "1337648523100000000;",
                  dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT count(ts_9) FROM ts_format where extract(epoch from ts_9) = "
                  "1337648523030000000;",
                  dt)));
    ASSERT_EQ(1L,
              v<int64_t>(run_simple_agg(
                  "SELECT count(ts_9) FROM ts_format where extract(epoch from ts_9) = "
                  "1337648523000000003;",
                  dt)));
  }
}
#endif

TEST(Truncate, Count) {
  run_ddl_statement("create table trunc_test (i1 integer, t1 text);");
  run_multiple_agg("insert into trunc_test values(1, '1');", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into trunc_test values(2, '2');", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(3),
            v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM trunc_test;",
                                      ExecutorDeviceType::CPU)));
  run_ddl_statement("truncate table trunc_test;");
  run_multiple_agg("insert into trunc_test values(3, '3');", ExecutorDeviceType::CPU);
  run_multiple_agg("insert into trunc_test values(4, '4');", ExecutorDeviceType::CPU);
  ASSERT_EQ(int64_t(7),
            v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM trunc_test;",
                                      ExecutorDeviceType::CPU)));
  run_ddl_statement("drop table trunc_test;");
}

TEST(Update, VarlenSmartSwitch) {
  if (!is_feature_enabled<VarlenUpdates>()) {
    return;
  }
  if (!is_feature_enabled<CalciteUpdatePathSelector>()) {
    return;
  }

  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists smartswitch;");
    run_ddl_statement("create table smartswitch (x int, y int[], z text );");

    run_multiple_agg("insert into smartswitch values (0, ARRAY[1,2,3,4], 'Flake');", dt);
    run_multiple_agg("insert into smartswitch values (1, ARRAY[5,6,7,8], 'Goofy');", dt);
    run_multiple_agg(
        "insert into smartswitch values (2, ARRAY[9,10,11,12,13], 'Lightweight');", dt);

    // In-place updates have no shift in rowid
    auto x_pre_rowid_0 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=0;", dt));
    auto x_pre_rowid_1 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=1;", dt));
    auto x_pre_rowid_2 =
        v<int64_t>(run_simple_agg("Select rowid from smartswitch where x=2;", dt));

    // Test RelProject-driven update
    run_multiple_agg("update smartswitch set x = x+1;", dt);
    ASSERT_EQ(int64_t(6),
              v<int64_t>(run_simple_agg("select sum(x) from smartswitch;", dt)));

    // Get post-update rowid
    auto x_post_rowid_1 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=1;", dt));
    auto x_post_rowid_2 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=2;", dt));
    auto x_post_rowid_3 =
        v<int64_t>(run_simple_agg("Select rowid from smartswitch where x=3;", dt));

    // Make sure the pre and post rowids are equal
    ASSERT_EQ(x_pre_rowid_0, x_post_rowid_1);
    ASSERT_EQ(x_pre_rowid_1, x_post_rowid_2);
    ASSERT_EQ(x_pre_rowid_2, x_post_rowid_3);

    // In-place updates have no shift in rowid
    x_pre_rowid_1 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=1;", dt));
    ;
    x_pre_rowid_2 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=2;", dt));
    ;
    auto x_pre_rowid_3 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=3;", dt));
    ;

    // Test RelCompound-driven update
    run_multiple_agg("update smartswitch set x=x+1 where x < 10;", dt);
    ASSERT_EQ(int64_t(9),
              v<int64_t>(run_simple_agg("select sum(x) from smartswitch;", dt)));

    x_post_rowid_2 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=2;", dt));
    ;
    x_post_rowid_3 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=3;", dt));
    ;
    auto x_post_rowid_4 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where x=4;", dt));
    ;

    // Make sure the pre and post rowids are equal
    // Completion of these assertions proves that in-place update was selected for this
    ASSERT_EQ(x_pre_rowid_1, x_post_rowid_2);
    ASSERT_EQ(x_pre_rowid_2, x_post_rowid_3);
    ASSERT_EQ(x_pre_rowid_3, x_post_rowid_4);

    // Test RelCompound-driven update

    auto y_pre_rowid_1 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where y[1]=1;", dt));
    run_multiple_agg("update smartswitch set y=ARRAY[9,10,11,12] where y[1]=1;", dt);
    auto y_post_rowid_1 =
        v<int64_t>(run_simple_agg("select rowid from smartswitch where y[1]=9 and "
                                  "y[2]=10 and y[3]=11 and y[4]=12 and z='Flake';",
                                  dt));

    // Internal insert-delete cycle should create a new rowid
    // This test validates that the CTAS varlen update path was used; the rowid change is
    // evidence
    ASSERT_NE(y_pre_rowid_1, y_post_rowid_1);

    ASSERT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg("select count(y) from smartswitch where y[1]=9 "
                                        "and y[2]=10 and y[3]=11 and y[4]=12;",
                                        dt)));
    ASSERT_EQ(int64_t(9),
              v<int64_t>(run_simple_agg("select sum(x) from smartswitch;", dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(z) from smartswitch where z='Flake';", dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(z) from smartswitch where z='Goofy';", dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(z) from smartswitch where z='Lightweight';", dt)));

    // Test RelProject-driven update
    run_multiple_agg("update smartswitch set y=ARRAY[2,3,5,7,11];", dt);
    ASSERT_EQ(int64_t(3),
              v<int64_t>(run_simple_agg("select count(y) from smartswitch where y[1]=2 "
                                        "and y[2]=3 and y[3]=5 and y[4]=7 and y[5]=11;",
                                        dt)));
    ASSERT_EQ(int64_t(9),
              v<int64_t>(run_simple_agg("select sum(x) from smartswitch;", dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(z) from smartswitch where z='Flake';", dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(z) from smartswitch where z='Goofy';", dt)));
    ASSERT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(z) from smartswitch where z='Lightweight';", dt)));

    run_ddl_statement("drop table smartswitch;");
  }

  g_enable_watchdog = save_watchdog;
}

TEST(Update, Text) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }
  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists text_default;");
    run_ddl_statement("create table text_default (t text) with (vacuum='delayed');");

    run_multiple_agg("insert into text_default values ('do');", dt);
    run_multiple_agg("insert into text_default values ('you');", dt);
    run_multiple_agg("insert into text_default values ('know');", dt);
    run_multiple_agg("insert into text_default values ('the');", dt);
    run_multiple_agg("insert into text_default values ('muffin');", dt);
    run_multiple_agg("insert into text_default values ('man');", dt);

    run_multiple_agg("update text_default set t='pizza' where char_length(t) <= 3;", dt);
    ASSERT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg(
                  "select count(t) from text_default where t='pizza';", dt)));
  }

  g_enable_watchdog = save_watchdog;
}

TEST(Update, TextINVariant) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }
  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists text_default;");
    run_ddl_statement("create table text_default (t text) with (vacuum='delayed');");

    run_multiple_agg("insert into text_default values ('do');", dt);
    run_multiple_agg("insert into text_default values ('you');", dt);
    run_multiple_agg("insert into text_default values ('know');", dt);
    run_multiple_agg("insert into text_default values ('the');", dt);
    run_multiple_agg("insert into text_default values ('muffin');", dt);
    run_multiple_agg("insert into text_default values ('man');", dt);

    run_multiple_agg(
        "update text_default set t='pizza' where t in ('do','you','the','man');", dt);
    ASSERT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg(
                  "select count(t) from text_default where t='pizza';", dt)));
  }

  g_enable_watchdog = save_watchdog;
}

TEST(Update, TextEncodingDict16) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }
  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists textenc16_default;");
    run_ddl_statement(
        "create table textenc16_default (t text encoding dict(16)) with "
        "(vacuum='delayed');");

    run_multiple_agg("insert into textenc16_default values ('do');", dt);
    run_multiple_agg("insert into textenc16_default values ('you');", dt);
    run_multiple_agg("insert into textenc16_default values ('know');", dt);
    run_multiple_agg("insert into textenc16_default values ('the');", dt);
    run_multiple_agg("insert into textenc16_default values ('muffin');", dt);
    run_multiple_agg("insert into textenc16_default values ('man');", dt);

    run_multiple_agg("update textenc16_default set t='pizza' where char_length(t) <= 3;",
                     dt);
    ASSERT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg(
                  "select count(t) from textenc16_default where t='pizza';", dt)));

    run_ddl_statement("drop table textenc16_default;");
  }

  g_enable_watchdog = save_watchdog;
}

TEST(Update, TextEncodingDict8) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }
  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists textenc8_default;");
    run_ddl_statement(
        "create table textenc8_default (t text encoding dict(8)) with "
        "(vacuum='delayed');");
    run_multiple_agg("insert into textenc8_default values ('do');", dt);
    run_multiple_agg("insert into textenc8_default values ('you');", dt);
    run_multiple_agg("insert into textenc8_default values ('know');", dt);
    run_multiple_agg("insert into textenc8_default values ('the');", dt);
    run_multiple_agg("insert into textenc8_default values ('muffin');", dt);
    run_multiple_agg("insert into textenc8_default values ('man');", dt);

    run_multiple_agg("update textenc8_default set t='pizza' where char_length(t) <= 3;",
                     dt);
    ASSERT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg(
                  "select count(t) from textenc8_default where t='pizza';", dt)));

    run_ddl_statement("drop table textenc8_default;");
  }
  g_enable_watchdog = save_watchdog;
}

TEST(Update, MultiColumnInteger) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table multicoltable (x integer, y integer, z integer) with "
        "(vacuum='delayed');");
    run_multiple_agg("insert into multicoltable values (2,3,4);", dt);
    run_multiple_agg("insert into multicoltable values (4,9,16);", dt);

    run_multiple_agg(
        "update multicoltable set x=-power(x,2),y=-power(y,2),z=-power(z,2) where x < 10 "
        "and y < 10 and z < 10;",
        dt);
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("select sum(x) from multicoltable;", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("select sum(y) from multicoltable;", dt)));
    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("select sum(z) from multicoltable;", dt)));

    run_ddl_statement("drop table multicoltable;");
  }
}

TEST(Update, TimestampUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table timestamp_default (t timestamp) with (vacuum='delayed');");

    run_multiple_agg("insert into timestamp_default values ('12/01/2013:000001');", dt);
    run_multiple_agg("insert into timestamp_default values ('12/01/2013:000002');", dt);
    run_multiple_agg("insert into timestamp_default values ('12/01/2013:000003');", dt);
    run_multiple_agg("insert into timestamp_default values ('12/01/2013:000004');", dt);
    run_multiple_agg(
        "update timestamp_default set t=timestampadd( second, 59, date_trunc( minute, t "
        ") ) where mod( extract( second "
        "from t ), 2 )=1;",
        dt);

    ASSERT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "select count(t) from timestamp_default where extract( second from t )=59;",
            dt)));

    run_ddl_statement("drop table timestamp_default;");
  }
}

TEST(Update, TimeUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table time_default (t time) with (vacuum='delayed');");

    run_multiple_agg("insert into time_default values('00:00:01');", dt);
    run_multiple_agg("insert into time_default values('00:01:00');", dt);
    run_multiple_agg("insert into time_default values('01:00:00');", dt);

    run_multiple_agg("update time_default set t='11:11:00' where t='00:00:01';", dt);
    run_multiple_agg("update time_default set t='11:00:11' where t='00:01:00';", dt);
    run_multiple_agg("update time_default set t='00:11:11' where t='01:00:00';", dt);

    ASSERT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "select count(t) from time_default where extract(hour from t)=11;", dt)));
    ASSERT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "select count(t) from time_default where extract(minute from t)=11;", dt)));
    ASSERT_EQ(
        int64_t(2),
        v<int64_t>(run_simple_agg(
            "select count(t) from time_default where extract(second from t)=11;", dt)));

    run_ddl_statement("drop table time_default;");
  }
}

TEST(Update, DateUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists date_default;");
    run_ddl_statement("create table date_default (d date) with (vacuum='delayed');");

    run_multiple_agg("insert into date_default values('01/01/1901');", dt);
    run_multiple_agg("insert into date_default values('02/02/1902');", dt);
    run_multiple_agg("insert into date_default values('03/03/1903');", dt);
    run_multiple_agg("insert into date_default values('04/04/1904');", dt);

    run_multiple_agg(
        "update date_default set d='12/25/2000' where mod(extract(day from d),2)=1;", dt);
    ASSERT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg(
                  "select count(d) from date_default where d='12/25/2000';", dt)));

    run_ddl_statement("drop table date_default;");
  }
}

TEST(Update, FloatUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table float_default (f float) with (vacuum='delayed');");

    run_multiple_agg("insert into float_default values(-0.01);", dt);
    run_multiple_agg("insert into float_default values( 0.02);", dt);
    run_multiple_agg("insert into float_default values(-0.03);", dt);
    run_multiple_agg("insert into float_default values( 0.04);", dt);

    run_multiple_agg("update float_default set f=ABS(f) where f < 0;", dt);
    ASSERT_FLOAT_EQ(static_cast<float>(0.01),
                    v<float>(run_simple_agg(
                        "select f from float_default where f > 0.0 and f < 0.02;", dt)));
    ASSERT_FLOAT_EQ(static_cast<float>(0.03),
                    v<float>(run_simple_agg(
                        "select f from float_default where f > 0.02 and f < 0.04;", dt)));

    run_ddl_statement("drop table float_default;");
  }
}

TEST(Update, IntegerUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table integer_default (i integer) with (vacuum='delayed');");

    run_multiple_agg("insert into integer_default values(-1);", dt);
    run_multiple_agg("insert into integer_default values( 2);", dt);
    run_multiple_agg("insert into integer_default values(-3);", dt);
    run_multiple_agg("insert into integer_default values( 4);", dt);

    run_multiple_agg("update integer_default set i=-i where i < 0;", dt);
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "select i from integer_default where i > 0 and i < 2;", dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg(
                  "select i from integer_default where i > 2 and i < 4;", dt)));
    run_ddl_statement("drop table integer_default;");
  }
}

TEST(Update, DoubleUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table double_default (d double) with (vacuum='delayed');");

    run_multiple_agg("insert into double_default values(-0.01);", dt);
    run_multiple_agg("insert into double_default values( 0.02);", dt);
    run_multiple_agg("insert into double_default values(-0.03);", dt);
    run_multiple_agg("insert into double_default values( 0.04);", dt);

    run_multiple_agg("update double_default set d=ABS(d) where d < 0;", dt);
    ASSERT_FLOAT_EQ(static_cast<double>(0.01),
                    v<double>(run_simple_agg(
                        "select d from double_default where d > 0.0 and d < 0.02;", dt)));
    ASSERT_FLOAT_EQ(
        static_cast<double>(0.03),
        v<double>(run_simple_agg(
            "select d from double_default where d > 0.02 and d < 0.04;", dt)));

    run_ddl_statement("drop table double_default;");
  }
}

TEST(Update, SmallIntUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table smallint_default (s smallint) with (vacuum='delayed');");

    run_multiple_agg("insert into smallint_default values(-1);", dt);
    run_multiple_agg("insert into smallint_default values( 2);", dt);
    run_multiple_agg("insert into smallint_default values(-3);", dt);
    run_multiple_agg("insert into smallint_default values( 4);", dt);

    run_multiple_agg("update smallint_default set s=-s where s < 0;", dt);
    run_multiple_agg("select s from smallint_default where s > 0 and s < 2;", dt);
    run_multiple_agg("select s from smallint_default where s > 2 and s < 4;", dt);

    run_ddl_statement("drop table smallint_default;");
  }
}

TEST(Update, BigIntUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table bigint_default (b bigint) with (vacuum='delayed');");
    run_multiple_agg("insert into bigint_default values(-1);", dt);
    run_multiple_agg("insert into bigint_default values( 2);", dt);
    run_multiple_agg("insert into bigint_default values(-3);", dt);
    run_multiple_agg("insert into bigint_default values( 4);", dt);

    run_multiple_agg("update bigint_default set b=-b where b < 0;", dt);
    run_multiple_agg("select b from bigint_default where b > 0 and b < 2;", dt);
    run_multiple_agg("select b from bigint_default where b > 2 and b < 4;", dt);

    run_ddl_statement("drop table bigint_default;");
  }
}

TEST(Update, DecimalUpdate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table decimal_default (d decimal(5)) with (vacuum='delayed');");
    run_multiple_agg("insert into decimal_default values(-1);", dt);
    run_multiple_agg("insert into decimal_default values( 2);", dt);
    run_multiple_agg("insert into decimal_default values(-3);", dt);
    run_multiple_agg("insert into decimal_default values( 4);", dt);

    run_multiple_agg("update decimal_default set d=-d where d < 0;", dt);

    run_simple_agg("select d from decimal_default where d > 0 and d < 2;", dt);
    run_simple_agg("select d from decimal_default where d > 2 and d < 4;", dt);
    ;

    run_ddl_statement("drop table decimal_default;");
  }
}

TEST(Update, JoinCacheInvalidationTest) {
  SKIP_ALL_ON_AGGREGATOR();

  if (!std::is_same<CalciteUpdatePathSelector, PreprocessorTrue>::value) {
    return;
  }
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table string_join1 ( t text ) with (vacuum='delayed')");
    run_ddl_statement("create table string_join2 ( t text ) with (vacuum='delayed')");

    run_multiple_agg("insert into string_join1 values ('muffin')", dt);
    run_multiple_agg("insert into string_join1 values ('pizza')", dt);
    run_multiple_agg("insert into string_join1 values ('ice cream')", dt);
    run_multiple_agg("insert into string_join1 values ('poutine')", dt);
    run_multiple_agg("insert into string_join1 values ('samosa')", dt);
    run_multiple_agg("insert into string_join2 values ('tomato')", dt);
    run_multiple_agg("insert into string_join2 values ('potato')", dt);
    run_multiple_agg("insert into string_join2 values ('apple')", dt);
    run_multiple_agg("insert into string_join2 values ('orange')", dt);
    run_multiple_agg("insert into string_join2 values ('chutney')", dt);
    run_multiple_agg("insert into string_join2 values ('poutine')", dt);

    run_simple_agg(
        "select count(string_join1.t) from string_join1 inner join string_join2 on "
        "string_join1.t = string_join2.t;",
        dt);
    run_multiple_agg("update string_join1 set t='not poutine' where t='poutine';", dt);

    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg(
            "select count(string_join1.t) from string_join1 inner join string_join2 on "
            "string_join1.t = string_join2.t;",
            dt)));

    run_ddl_statement("drop table string_join1;");
    run_ddl_statement("drop table string_join2;");
  }
}

TEST(Delete, WithoutVacuumAttribute) {
  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table no_deletes (x integer) with (vacuum='immediate');");
    run_multiple_agg("insert into no_deletes values (10);", dt);
    run_multiple_agg("insert into no_deletes values (11);", dt);
    EXPECT_THROW(run_multiple_agg("delete from no_deletes where x > 10;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("delete from no_deletes;", dt), std::runtime_error);
    run_ddl_statement("drop table no_deletes;");
  }
}

TEST(Update, ImplicitCastToDate4) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table datetab ( d1 date ) with ( vacuum='delayed' );");
    run_multiple_agg("insert into datetab values ('2001-04-05');", dt);

    EXPECT_THROW(run_multiple_agg("update datetab set d1='nonsense';", dt),
                 std::runtime_error);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab where d1='2001-04-05';", dt)));

    run_multiple_agg(
        "update datetab set d1=cast( '1999-12-31 23:59:59' as varchar(32) );", dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab where d1='1999-12-31';", dt)));

    run_multiple_agg("update datetab set d1=cast( '1990-12-31 13:59:59' as char(32) );",
                     dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab where d1='1990-12-31';", dt)));

    run_multiple_agg("update datetab set d1=cast( '1989-01-01 00:00:00' as timestamp );",
                     dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab where d1='1989-01-01';", dt)));

    run_multiple_agg("update datetab set d1=cast( '2000' as date );", dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab where d1='2000-01-01';", dt)));

    EXPECT_THROW(run_simple_agg("update datetab set d1=cast( 2000.00 as float );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab set d1=cast( 2123.444 as double );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab set d1=cast( 1235 as integer );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab set d1=cast( 12 as smallint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab set d1=cast( 9 as bigint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab set d1=cast( 'False' as boolean );", dt),
                 std::runtime_error);

    run_ddl_statement("drop table datetab;");
  }
}

TEST(Update, ImplicitCastToDate2) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table datetab4 ( d1 date encoding fixed(16)) with ( vacuum='delayed' );");
    run_multiple_agg("insert into datetab4 values ('2001-04-05');", dt);

    EXPECT_THROW(run_multiple_agg("update datetab4 set d1='nonsense';", dt),
                 std::runtime_error);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab4 where d1='2001-04-05';", dt)));

    run_multiple_agg(
        "update datetab4 set d1=cast( '1999-12-31 23:59:59' as varchar(32) );", dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab4 where d1='1999-12-31';", dt)));

    run_multiple_agg("update datetab4 set d1=cast( '1990-12-31 13:59:59' as char(32) );",
                     dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab4 where d1='1990-12-31';", dt)));

    run_multiple_agg("update datetab4 set d1=cast( '1989-01-01 00:00:00' as timestamp );",
                     dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab4 where d1='1989-01-01';", dt)));

    run_multiple_agg("update datetab4 set d1=cast( '2000' as date );", dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(d1) from datetab4 where d1='2000-01-01';", dt)));

    EXPECT_THROW(run_simple_agg("update datetab4 set d1=cast( 2000.00 as float );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab4 set d1=cast( 2123.444 as double );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab4 set d1=cast( 1235 as integer );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab4 set d1=cast( 12 as smallint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab4 set d1=cast( 9 as bigint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("update datetab4 set d1=cast( 'False' as boolean );", dt),
                 std::runtime_error);

    run_ddl_statement("drop table datetab4;");
  }
}

TEST(Update, ImplicitCastToEncodedString) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists textenc;");
    run_ddl_statement(
        "create table textenc ( s1 text encoding dict(32), s2 text encoding dict(16), s3 "
        "text encoding dict(8) ) with "
        "(vacuum='delayed');");
    run_multiple_agg("insert into textenc values ( 'kanye', 'omari', 'west' );", dt);

    run_multiple_agg("update textenc set s1 = 'the';", dt);
    EXPECT_EQ(NullableString("the"),
              v<NullableString>(
                  run_simple_agg("select s1 from textenc where s1 = 'the';", dt)));

    run_multiple_agg("update textenc set s2 = 'college';", dt);
    EXPECT_EQ(NullableString("college"),
              v<NullableString>(
                  run_simple_agg("select s2 from textenc where s2 = 'college';", dt)));

    run_multiple_agg("update textenc set s3 = 'dropout';", dt);
    EXPECT_EQ(NullableString("dropout"),
              v<NullableString>(
                  run_simple_agg("select s3 from textenc where s3 = 'dropout';", dt)));

    run_multiple_agg("update textenc set s1 = s2;", dt);
    EXPECT_EQ(NullableString("college"),
              v<NullableString>(
                  run_simple_agg("select s1 from textenc where s1 = 'college';", dt)));

    run_multiple_agg("update textenc set s2 = s3;", dt);
    EXPECT_EQ(NullableString("dropout"),
              v<NullableString>(
                  run_simple_agg("select s2 from textenc where s2='dropout';", dt)));

    run_multiple_agg("update textenc set s3 = s1;", dt);
    EXPECT_EQ(NullableString("college"),
              v<NullableString>(
                  run_simple_agg("select s3 from textenc where s3='college';", dt)));

    run_multiple_agg("update textenc set s1=cast('1977-06-08 00:00:00' as timestamp);",
                     dt);
    EXPECT_EQ(NullableString("1977-06-08 00:00:00"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast('12:34:56' as time);", dt);
    EXPECT_EQ(NullableString("12:34:56"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast('1977-06-08' as date);", dt);
    EXPECT_EQ(NullableString("1977-06-08"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast( 1234.00 as float );", dt);
    EXPECT_EQ(NullableString("1234.000000"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast( 12345.00 as double );", dt);
    EXPECT_EQ(NullableString("12345.000000"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast( 1234 as integer );", dt);
    EXPECT_EQ(NullableString("1234"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast( 12 as smallint );", dt);
    EXPECT_EQ(NullableString("12"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast( 123412341234 as bigint );", dt);
    EXPECT_EQ(NullableString("123412341234"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast( 'True' as boolean );", dt);
    EXPECT_EQ(NullableString("t"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s1=cast( 1234.56 as decimal );", dt);
    EXPECT_EQ(NullableString("               1235"),
              v<NullableString>(run_simple_agg("select s1 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast('1977-06-08 00:00:00' as timestamp);",
                     dt);
    EXPECT_EQ(NullableString("1977-06-08 00:00:00"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast('12:34:56' as time);", dt);
    EXPECT_EQ(NullableString("12:34:56"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast('1977-06-08' as date);", dt);
    EXPECT_EQ(NullableString("1977-06-08"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast( 1234.00 as float );", dt);
    EXPECT_EQ(NullableString("1234.000000"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast( 12345.00 as double );", dt);
    EXPECT_EQ(NullableString("12345.000000"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast( 1234 as integer );", dt);
    EXPECT_EQ(NullableString("1234"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast( 12 as smallint );", dt);
    EXPECT_EQ(NullableString("12"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast( 123412341234 as bigint );", dt);
    EXPECT_EQ(NullableString("123412341234"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast( 'True' as boolean );", dt);
    EXPECT_EQ(NullableString("t"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s2=cast( 1234.56 as decimal );", dt);
    EXPECT_EQ(NullableString("               1235"),
              v<NullableString>(run_simple_agg("select s2 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast('1977-06-08 00:00:00' as timestamp);",
                     dt);
    EXPECT_EQ(NullableString("1977-06-08 00:00:00"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast('12:34:56' as time);", dt);
    EXPECT_EQ(NullableString("12:34:56"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast('1977-06-08' as date);", dt);
    EXPECT_EQ(NullableString("1977-06-08"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast( 1234.00 as float );", dt);
    EXPECT_EQ(NullableString("1234.000000"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast( 12345.00 as double );", dt);
    EXPECT_EQ(NullableString("12345.000000"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast( 1234 as integer );", dt);
    EXPECT_EQ(NullableString("1234"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast( 12 as smallint );", dt);
    EXPECT_EQ(NullableString("12"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast( 123412341234 as bigint );", dt);
    EXPECT_EQ(NullableString("123412341234"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast( 'True' as boolean );", dt);
    EXPECT_EQ(NullableString("t"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_multiple_agg("update textenc set s3=cast( 1234.56 as decimal );", dt);
    EXPECT_EQ(NullableString("               1235"),
              v<NullableString>(run_simple_agg("select s3 from textenc;", dt)));

    run_ddl_statement("drop table textenc;");
  }
}

TEST(Update, ImplicitCastToNoneEncodedString) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  if (!is_feature_enabled<VarlenUpdates>()) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto execute_and_expect_string = [&dt](auto& query_to_execute,
                                           NullableString expected) {
      run_multiple_agg(query_to_execute, dt);
      EXPECT_EQ(
          v<NullableString>(run_simple_agg("select str from none_str limit 1;", dt)),
          expected);
    };

    run_ddl_statement(
        "create table none_str ( str text encoding none ) with (vacuum='delayed');");

    run_multiple_agg("insert into none_str values ('kanye');", dt);
    execute_and_expect_string("update none_str set str='yeezy';", "yeezy");
    execute_and_expect_string("update none_str set str='kanye' where str='yeezy';",
                              "kanye");
    execute_and_expect_string(
        "update none_str set str=cast('1977-06-08 00:00:00' as timestamp);",
        "1977-06-08 00:00:00");
    execute_and_expect_string("update none_str set str=cast('12:34:56' as time);",
                              "12:34:56");
    execute_and_expect_string("update none_str set str=cast('1977-06-08' as date);",
                              "1977-06-08");
    execute_and_expect_string("update none_str set str=cast(1234.00 as float);",
                              "1234.000000");
    execute_and_expect_string("update none_str set str=cast(12345.00 as double );",
                              "12345.000000");
    execute_and_expect_string("update none_str set str=cast(1234 as integer );", "1234");
    execute_and_expect_string("update none_str set str=cast( 12 as smallint );", "12");
    execute_and_expect_string("update none_str set str=cast( 123412341234 as bigint );",
                              "123412341234");
    execute_and_expect_string("update none_str set str=cast( 'True' as boolean );", "t");
    execute_and_expect_string("update none_str set str=cast( 1234.56 as decimal );",
                              "               1235");
    run_ddl_statement("drop table none_str;");
  }
}

TEST(Update, ImplicitCastToNumericTypes) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists floattest;");
    run_ddl_statement("create table floattest ( f float ) with (vacuum='delayed');");
    run_ddl_statement("drop table if exists doubletest;");
    run_ddl_statement("create table doubletest ( d double ) with (vacuum='delayed');");
    run_ddl_statement("drop table if exists inttest;");
    run_ddl_statement("create table inttest ( i integer ) with (vacuum='delayed');");
    run_ddl_statement("drop table if exists sinttest;");
    run_ddl_statement("create table sinttest ( i integer ) with (vacuum='delayed');");
    run_ddl_statement("drop table if exists binttest;");
    run_ddl_statement("create table binttest ( i integer ) with (vacuum='delayed');");
    run_ddl_statement("drop table if exists booltest;");
    run_ddl_statement("create table booltest ( b boolean ) with (vacuum='delayed');");
    run_ddl_statement("drop table if exists dectest;");
    run_ddl_statement("create table dectest ( d decimal(10) ) with (vacuum='delayed');");

    run_multiple_agg("insert into floattest values ( 0.1234 );", dt);
    run_multiple_agg("insert into doubletest values ( 0.1234 );", dt);
    run_multiple_agg("insert into inttest values ( 1234 );", dt);
    run_multiple_agg("insert into sinttest values ( 1234 );", dt);
    run_multiple_agg("insert into binttest values ( 1234 );", dt);
    run_multiple_agg("insert into booltest values ( 'True' );", dt);
    run_multiple_agg("insert into dectest values ( '1234.0' );", dt);

    EXPECT_THROW(
        run_multiple_agg("update floattest set f=cast( 'nonsense' as varchar );", dt),
        std::invalid_argument);

    run_multiple_agg("update floattest set f=cast( '128.90' as varchar );", dt);
    EXPECT_FLOAT_EQ(float(128.90),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast ('2000-01-01 10:11:12' as timestamp );",
                     dt);
    EXPECT_FLOAT_EQ(float(9.467215 * powf(10, 8)),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast ('12:34:56' as time );", dt);
    EXPECT_FLOAT_EQ(float(45296),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast ('1999-12-31' as date);", dt);
    EXPECT_FLOAT_EQ(float(9.465984 * powf(10, 8)),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast (1234.0 as float);", dt);
    EXPECT_FLOAT_EQ(float(1234.0),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast (1234.0 as double);", dt);
    EXPECT_FLOAT_EQ(float(1234.0),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast(56780 as integer);", dt);
    EXPECT_FLOAT_EQ(float(56780),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast(12345 as smallint);", dt);
    EXPECT_FLOAT_EQ(float(12345),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast(12345 as bigint);", dt);
    EXPECT_FLOAT_EQ(float(12345),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast('True' as boolean);", dt);
    EXPECT_FLOAT_EQ(float(1), v<float>(run_simple_agg("select f from floattest;", dt)));

    run_multiple_agg("update floattest set f=cast(1234.00 as decimal);", dt);
    EXPECT_FLOAT_EQ(float(1234),
                    v<float>(run_simple_agg("select f from floattest;", dt)));

    EXPECT_THROW(
        run_multiple_agg("update doubletest set d=cast( 'nonsense' as varchar );", dt),
        std::invalid_argument);

    run_multiple_agg("update doubletest set d=cast( '128.90' as varchar );", dt);
    EXPECT_DOUBLE_EQ(double(128.90),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg(
        "update doubletest set d=cast( '2000-01-01 10:11:12' as timestamp );", dt);
    EXPECT_DOUBLE_EQ(double(946721472),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( '12:34:56' as time );", dt);
    EXPECT_DOUBLE_EQ(double(45296),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( '1999-12-31' as date );", dt);
    EXPECT_DOUBLE_EQ(double(946598400),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( 1234.0 as float );", dt);
    EXPECT_DOUBLE_EQ(double(1234.0),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( 1234.0 as double );", dt);
    EXPECT_DOUBLE_EQ(double(1234.0),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( 56780 as integer );", dt);
    EXPECT_DOUBLE_EQ(double(56780),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( 12345 as smallint );", dt);
    EXPECT_DOUBLE_EQ(double(12345),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( 12345 as bigint );", dt);
    EXPECT_DOUBLE_EQ(double(12345),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( 'True' as boolean );", dt);
    EXPECT_DOUBLE_EQ(double(1),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    run_multiple_agg("update doubletest set d=cast( 1234.00 as decimal );", dt);
    EXPECT_DOUBLE_EQ(double(1234),
                     v<double>(run_simple_agg("select d from doubletest;", dt)));

    EXPECT_THROW(
        run_multiple_agg("update inttest set i=cast( 'nonsense' as varchar );", dt),
        std::invalid_argument);
    run_multiple_agg("update inttest set i=cast( '128.90' as varchar );", dt);
    EXPECT_EQ(int64_t(128), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( '2000-01-01 10:11:12' as timestamp );",
                     dt);
    EXPECT_EQ(int64_t(946721472),
              v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( '12:34:56' as time );", dt);
    EXPECT_EQ(int64_t(45296), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( '1999-12-31' as date );", dt);
    EXPECT_EQ(int64_t(946598400),
              v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( 1234.0 as float );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( 1234.0 as double );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( 56780 as integer );", dt);
    EXPECT_EQ(int64_t(56780), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( 12345 as smallint );", dt);
    EXPECT_EQ(int64_t(12345), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( 12345 as bigint );", dt);
    EXPECT_EQ(int64_t(12345), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( 'True' as boolean );", dt);
    EXPECT_EQ(int64_t(1), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    run_multiple_agg("update inttest set i=cast( 1234.00 as decimal );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from inttest;", dt)));

    EXPECT_THROW(
        run_multiple_agg("update sinttest set i=cast( 'nonsense' as varchar );", dt),
        std::invalid_argument);
    run_multiple_agg("update sinttest set i=cast( '128.90' as varchar );", dt);
    EXPECT_EQ(int64_t(128), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( '2000-01-01 10:11:12' as timestamp );",
                     dt);
    EXPECT_EQ(int64_t(946721472),
              v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( '12:34:56' as time );", dt);
    EXPECT_EQ(int64_t(45296), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( '1999-12-31' as date );", dt);
    EXPECT_EQ(int64_t(946598400),
              v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( 1234.0 as float );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( 1234.0 as double );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( 56780 as integer );", dt);
    EXPECT_EQ(int64_t(56780), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( 12345 as smallint );", dt);
    EXPECT_EQ(int64_t(12345), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( 12345 as bigint );", dt);
    EXPECT_EQ(int64_t(12345), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( 'True' as boolean );", dt);
    EXPECT_EQ(int64_t(1), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    run_multiple_agg("update sinttest set i=cast( 1234.00 as decimal );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from sinttest;", dt)));

    EXPECT_THROW(
        run_multiple_agg("update binttest set i=cast( 'nonsense' as varchar );", dt),
        std::invalid_argument);
    run_multiple_agg("update binttest set i=cast( '128.90' as varchar );", dt);
    EXPECT_EQ(int64_t(128), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( '2000-01-01 10:11:12' as timestamp );",
                     dt);
    EXPECT_EQ(int64_t(946721472),
              v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( '12:34:56' as time );", dt);
    EXPECT_EQ(int64_t(45296), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( '1999-12-31' as date );", dt);
    EXPECT_EQ(int64_t(946598400),
              v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( 1234.0 as float );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( 1234.0 as double );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( 56780 as integer );", dt);
    EXPECT_EQ(int64_t(56780), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( 12345 as smallint );", dt);
    EXPECT_EQ(int64_t(12345), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( 12345 as bigint );", dt);
    EXPECT_EQ(int64_t(12345), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( 'True' as boolean );", dt);
    EXPECT_EQ(int64_t(1), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    run_multiple_agg("update binttest set i=cast( 1234.00 as decimal );", dt);
    EXPECT_EQ(int64_t(1234), v<int64_t>(run_simple_agg("select i from binttest;", dt)));

    EXPECT_THROW(
        run_multiple_agg("update booltest set b=cast( 'nonsense' as varchar );", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update booltest set b=cast( '128.90' as varchar );", dt),
        std::runtime_error);

    EXPECT_THROW(
        run_multiple_agg(
            "update booltest set b=cast( '2000-01-01 10:11:12' as timestamp );", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update booltest set b=cast( '12:34:56' as time );", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update booltest set b=cast( '1999-12-31' as date );", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update booltest set b=cast( 1234.0 as float );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update booltest set b=cast( 1234.0 as double );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update booltest set b=cast( 56780 as integer );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update booltest set b=cast( 12345 as smallint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update booltest set b=cast( 12345 as bigint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update booltest set b=cast( 'True' as boolean );", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update booltest set b=cast( 1234.00 as decimal );", dt),
        std::runtime_error);

    EXPECT_THROW(
        run_multiple_agg("update dectest set d=cast( 'nonsense' as varchar );", dt),
        std::invalid_argument);
    run_multiple_agg("update dectest set d=cast( '128.90' as varchar );", dt);
    EXPECT_EQ(
        int64_t(128),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    run_multiple_agg("update dectest set d=cast( '2000-01-01 10:11:12' as timestamp );",
                     dt);
    EXPECT_EQ(
        int64_t(946721472),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    run_multiple_agg("update dectest set d=cast( '12:34:56' as time );", dt);
    EXPECT_EQ(
        int64_t(45296),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    run_multiple_agg("update dectest set d=cast( '1999-12-31' as date );", dt);
    EXPECT_EQ(
        int64_t(946598400),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    // If ConstantFoldingVisitor is allowed to fold FP to DECIMAL casts then
    // these two tests will pass, otherwise we'll keep hitting cast codegen assert.
    EXPECT_THROW(run_multiple_agg("update dectest set d=cast( 1234.0 as float );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update dectest set d=cast( 1234.0 as double );", dt),
                 std::runtime_error);

    run_multiple_agg("update dectest set d=cast( 56780 as integer );", dt);
    EXPECT_EQ(
        int64_t(56780),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    run_multiple_agg("update dectest set d=cast( 12345 as smallint );", dt);
    EXPECT_EQ(
        int64_t(12345),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    run_multiple_agg("update dectest set d=cast( 12345 as bigint );", dt);
    EXPECT_EQ(
        int64_t(12345),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    run_multiple_agg("update dectest set d=cast( 'True' as boolean );", dt);
    EXPECT_EQ(
        int64_t(1),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    run_multiple_agg("update dectest set d=cast( 1234.00 as decimal );", dt);
    EXPECT_EQ(
        int64_t(1234),
        v<int64_t>(run_simple_agg("select cast( d as integer ) from dectest;", dt)));

    run_ddl_statement("drop table floattest;");
    run_ddl_statement("drop table doubletest;");
    run_ddl_statement("drop table inttest;");
    run_ddl_statement("drop table sinttest;");
    run_ddl_statement("drop table binttest;");
    run_ddl_statement("drop table booltest;");
    run_ddl_statement("drop table dectest;");
  }
}

TEST(Update, ImplicitCastToTime4) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table time4 ( t1 time encoding fixed(32) ) with ( vacuum='delayed' );");
    run_multiple_agg("insert into time4 values ('01:23:45');", dt);

    EXPECT_THROW(run_multiple_agg("update time4 set t1='nonsense';", dt), std::exception);

    // todo(pavan):  The parser is wrong on this one; need to disable this conversion
    // run_multiple_agg("update time4 set t1=cast( '1999-12-31 23:59:59' as varchar(32)
    // );", dt); run_multiple_agg("select t1 from time4;", dt);

    // todo(pavan):  The parser is wrong on this one; need to disable this conversion
    // run_multiple_agg("update time4 set t1=cast( '1990-12-31 23:59:59' as char(32) );",
    // dt); run_multiple_agg("select t1 from time4;", dt);

    EXPECT_THROW(
        run_multiple_agg(
            "update time4 set t1=cast( '1989-01-01 00:00:00' as timestamp );", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update time4 set t1=cast( '2000' as date );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update time4 set t1=cast( 2000.00 as float );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update time4 set t1=cast( 2123.444 as double );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update time4 set t1=cast( 1235 as integer );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update time4 set t1=cast( 12 as smallint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update time4 set t1=cast( 9 as bigint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update time4 set t1=cast( 'False' as boolean );", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update time4 set t1=cast( '1234.00' as decimal );", dt),
        std::runtime_error);

    run_ddl_statement("drop table time4;");
  }
}

TEST(Update, ImplicitCastToTime8) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table timetab ( t1 time ) with ( vacuum='delayed' );");
    run_multiple_agg("insert into timetab values ('01:23:45');", dt);

    EXPECT_THROW(run_multiple_agg("update timetab set t1='nonsense';", dt),
                 std::exception);

    // todo(pavan): The parser is wrong on this one; need to disable this conversion
    // run_multiple_agg( "update timetab set t1=cast( '1999-12-31 23:59:59' as varchar(32)
    // );" , dt ); run_multiple_agg( "update timetab set t1=cast( '1990-12-31 23:59:59' as
    // char(32) );" , dt ); run_multiple_agg( "update timetab set t1=cast( '1989-01-01
    // 00:00:00' as timestamp );" , dt );

    EXPECT_THROW(run_multiple_agg("update timetab set t1=cast( '2000' as date );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update timetab set t1=cast( 2000.00 as float );", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update timetab set t1=cast( 2123.444 as double );", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update timetab set t1=cast( 1235 as integer );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update timetab set t1=cast( 12 as smallint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update timetab set t1=cast( 9 as bigint );", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update timetab set t1=cast( 'False' as boolean );", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update timetab set t1=cast( '1234.00' as decimal );", dt),
        std::runtime_error);

    run_ddl_statement("drop table timetab;");
  }
}

TEST(Update, ImplicitCastToTimestamp8) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table tstamp ( t1 timestamp ) with ( vacuum='delayed' );");
    run_multiple_agg("insert into tstamp values ('2000-01-01 00:00:00');", dt);

    EXPECT_THROW(run_multiple_agg("update tstamp set t1='nonsense';", dt),
                 std::exception);

    run_multiple_agg("update tstamp set t1=cast( '1999-12-31 23:59:59' as varchar(32) );",
                     dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(t1) from tstamp where t1='1999-12-31 23:59:59';", dt)));
    run_multiple_agg("update tstamp set t1=cast( '1990-12-31 23:59:59' as char(32) );",
                     dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(t1) from tstamp where t1='1990-12-31 23:59:59';", dt)));

    EXPECT_NO_THROW(run_multiple_agg(
        "update tstamp set t1=cast( '1989-01-01 00:00:00' as timestamp );", dt));
    EXPECT_THROW(run_multiple_agg("update tstamp set t1=cast( '2000' as date );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp set t1=cast( 2000.00 as float );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp set t1=cast( 2123.444 as double );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp set t1=cast( 1235 as integer );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp set t1=cast( 12 as smallint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp set t1=cast( 9 as bigint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp set t1=cast( 'False' as boolean );", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update tstamp set t1=cast( '1234.00' as decimal );", dt),
        std::runtime_error);

    run_ddl_statement("drop table tstamp;");
  }
}

TEST(Update, ImplicitCastToTimestamp4) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table tstamp4 ( t1 timestamp encoding fixed(32) ) with ( "
        "vacuum='delayed' );");
    run_multiple_agg("insert into tstamp4 values ('2000-01-01 00:00:00');", dt);

    EXPECT_THROW(run_multiple_agg("update tstamp4 set t1='nonsense';", dt),
                 std::exception);

    run_multiple_agg(
        "update tstamp4 set t1=cast( '1999-12-31 23:59:59' as varchar(32) );", dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(t1) from tstamp4 where t1='1999-12-31 23:59:59';", dt)));

    run_multiple_agg("update tstamp4 set t1=cast( '1990-12-31 23:59:59' as char(32) );",
                     dt);
    EXPECT_EQ(int64_t(1),
              v<int64_t>(run_simple_agg(
                  "select count(t1) from tstamp4 where t1='1990-12-31 23:59:59';", dt)));

    EXPECT_NO_THROW(run_multiple_agg(
        "update tstamp4 set t1=cast( '1989-01-01 00:00:00' as timestamp );", dt));
    EXPECT_THROW(run_multiple_agg("update tstamp4 set t1=cast( '2000' as date );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp4 set t1=cast( 2000.00 as float );", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update tstamp4 set t1=cast( 2123.444 as double );", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp4 set t1=cast( 1235 as integer );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp4 set t1=cast( 12 as smallint );", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update tstamp4 set t1=cast( 9 as bigint );", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update tstamp4 set t1=cast( 'False' as boolean );", dt),
        std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update tstamp4 set t1=cast( '1234.00' as decimal );", dt),
        std::runtime_error);

    run_ddl_statement("drop table tstamp4;");
  }
}

TEST(Update, ShardedTableShardKeyTest) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteUpdatePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement(
        "create table updateshardkey ( x integer, y integer, shard key (x) ) with "
        "(vacuum='delayed', shard_count=4);");

    run_multiple_agg("insert into updateshardkey values (1,2);", dt);
    run_multiple_agg("insert into updateshardkey values (3,4);", dt);
    run_multiple_agg("insert into updateshardkey values (5,6);", dt);
    run_multiple_agg("insert into updateshardkey values (7,8);", dt);
    run_multiple_agg("insert into updateshardkey values (9,10);", dt);
    run_multiple_agg("insert into updateshardkey values (11,12);", dt);
    run_multiple_agg("insert into updateshardkey values (13,14);", dt);
    run_multiple_agg("insert into updateshardkey values (15,16);", dt);
    run_multiple_agg("insert into updateshardkey values (17,18);", dt);

    EXPECT_THROW(run_multiple_agg("update updateshardkey set x=x-1;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update updateshardkey set x=x-1,y=y-1;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_multiple_agg("update updateshardkey set x=x-1 where x > 0;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("update updateshardkey set x=x-1,y=y-1 where x > 0;", dt),
        std::runtime_error);

    EXPECT_EQ(int64_t(2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18),
              v<int64_t>(run_simple_agg("select sum(y) from updateshardkey;", dt)));

    run_ddl_statement("drop table updateshardkey;");
  }
}

TEST(Update, UsingDateColumns) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    run_ddl_statement("drop table if exists chelsea_updates;");
    run_ddl_statement(
        "create table chelsea_updates (col_src date, col_dst_16 date encoding fixed(16), "
        "col_dst date, col_dst_ts timestamp(0), col_dst_ts_32 timestamp encoding "
        "fixed(32)) with ( vacuum='delayed' );");
    run_multiple_agg(
        "insert into chelsea_updates values('1911-01-01', null, null, null, null);", dt);
    run_multiple_agg(
        "insert into chelsea_updates values('1911-01-01', null, null, null, null);", dt);
    run_multiple_agg(
        "insert into chelsea_updates values('1911-01-01', null, null, null, null);", dt);
    run_multiple_agg(
        "insert into chelsea_updates values('1911-01-01', null, null, null, null);", dt);

    run_multiple_agg("update chelsea_updates set col_dst = col_src;", dt);
    EXPECT_EQ(
        int64_t(4),
        v<int64_t>(run_simple_agg(
            "select count(col_dst) from chelsea_updates where col_dst='1911-01-01';",
            dt)));
    run_multiple_agg("update chelsea_updates set col_dst_16 = col_src;", dt);
    EXPECT_EQ(
        int64_t(4),
        v<int64_t>(run_simple_agg(
            "select count(col_dst_16) from chelsea_updates where col_dst='1911-01-01';",
            dt)));
    run_multiple_agg("update chelsea_updates set col_dst_ts_32 = col_src;", dt);
    EXPECT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg("select count(col_dst) from chelsea_updates "
                                        "where col_dst='1911-01-01 00.00.00';",
                                        dt)));
    run_multiple_agg("update chelsea_updates set col_dst_ts = col_src;", dt);
    EXPECT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg("select count(col_dst_16) from chelsea_updates "
                                        "where col_dst='1911-01-01 00.00.00';",
                                        dt)));
  }
}

TEST(Delete, ShardedTableDeleteTest) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists shardkey;");
    run_ddl_statement(
        "create table shardkey ( x integer, y integer, shard key (x) ) with "
        "(vacuum='delayed', shard_count=4);");

    run_multiple_agg("insert into shardkey values (1,2);", dt);
    run_multiple_agg("insert into shardkey values (3,4);", dt);
    run_multiple_agg("insert into shardkey values (5,6);", dt);
    run_multiple_agg("insert into shardkey values (7,8);", dt);
    run_multiple_agg("insert into shardkey values (9,10);", dt);
    run_multiple_agg("insert into shardkey values (11,12);", dt);
    run_multiple_agg("insert into shardkey values (13,14);", dt);
    run_multiple_agg("insert into shardkey values (15,16);", dt);
    run_multiple_agg("insert into shardkey values (17,18);", dt);

    run_multiple_agg("select * from shardkey;", dt);
    run_multiple_agg("delete from shardkey where x <= 9;", dt);
    run_multiple_agg("select sum(x) from shardkey;", dt);

    ASSERT_EQ(int64_t(11 + 13 + 15 + 17),
              v<int64_t>(run_simple_agg("select sum(x) from shardkey;", dt)));

    run_ddl_statement("drop table shardkey;");
  }
}

TEST(Delete, JoinCacheInvalidationTest) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("create table string_join1 ( t text ) with (vacuum='delayed')");
    run_ddl_statement("create table string_join2 ( t text ) with (vacuum='delayed')");

    run_multiple_agg("insert into string_join1 values ('muffin')", dt);
    run_multiple_agg("insert into string_join1 values ('pizza')", dt);
    run_multiple_agg("insert into string_join1 values ('ice cream')", dt);
    run_multiple_agg("insert into string_join1 values ('poutine')", dt);
    run_multiple_agg("insert into string_join1 values ('samosa')", dt);
    run_multiple_agg("insert into string_join2 values ('tomato')", dt);
    run_multiple_agg("insert into string_join2 values ('potato')", dt);
    run_multiple_agg("insert into string_join2 values ('apple')", dt);
    run_multiple_agg("insert into string_join2 values ('orange')", dt);
    run_multiple_agg("insert into string_join2 values ('chutney')", dt);
    run_multiple_agg("insert into string_join2 values ('poutine')", dt);

    run_simple_agg(
        "select count(string_join1.t) from string_join1 inner join string_join2 on "
        "string_join1.t = string_join2.t;",
        dt);
    run_multiple_agg("delete from string_join1 where t='poutine';", dt);

    ASSERT_EQ(
        int64_t(0),
        v<int64_t>(run_simple_agg(
            "select count(string_join1.t) from string_join1 inner join string_join2 on "
            "string_join1.t = string_join2.t;",
            dt)));

    run_ddl_statement("drop table string_join1;");
    run_ddl_statement("drop table string_join2;");
  }
}

TEST(Delete, IntraFragment) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists vacuum_test;");
    run_ddl_statement(
        "create table vacuum_test (i1 integer, t1 text) with (vacuum='delayed');");
    run_multiple_agg("insert into vacuum_test values(1, '1');", dt);
    run_multiple_agg("insert into vacuum_test values(2, '2');", dt);
    run_multiple_agg("insert into vacuum_test values(3, '3');", dt);
    run_multiple_agg("insert into vacuum_test values(4, '4');", dt);
    run_multiple_agg("delete from vacuum_test where i1 <= 4;", dt);

    ASSERT_EQ(int64_t(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(i1) FROM vacuum_test;", dt)));

    run_ddl_statement("drop table vacuum_test;");
  }
}

#if 0
// FIX-ME:  Test failing on some systems with calcite exceptions, needs rewriting
TEST(Delete, Joins_EmptyTable) {
  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value)
    return;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT vacuum_test_alt.x, emptytab.x FROM vacuum_test_alt, emptytab WHERE vacuum_test_alt.x = emptytab.x;", dt);
    c("SELECT COUNT(*) FROM vacuum_test_alt, emptytab GROUP BY vacuum_test_alt.x;", dt);
    c("SELECT COUNT(*) FROM vacuum_test_alt, emptytab, test_inner where vacuum_test_alt.x = emptytab.x;", dt);
    c("SELECT vacuum_test_alt.x, emptytab.x FROM vacuum_test_alt LEFT JOIN emptytab ON vacuum_test_alt.y = emptytab.y "
      "ORDER BY "
      "vacuum_test_alt.x ASC;",
      dt);

    run_ddl_statement("drop table vacuum_test_alt;");
  }
}
#endif

TEST(Delete, Joins_InnerJoin_TwoTables) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    c("SELECT COUNT(*) FROM test a JOIN single_row_test b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) from test a JOIN single_row_test b ON a.ofd = b.x;", dt);
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.x = test_inner.x;", dt);
    c("SELECT a.y, z FROM test a JOIN test_inner b ON a.x = b.x order by a.y;", dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.str = b.dup_str;", dt);
    SKIP_ON_AGGREGATOR(
        c("SELECT COUNT(*) FROM test_inner_x a JOIN test_x b ON a.x = b.x;", dt));
    c("SELECT a.x FROM test a JOIN join_test b ON a.str = b.dup_str ORDER BY a.x;", dt);
    c("SELECT a.x FROM test_inner_x a JOIN test_x b ON a.x = b.x ORDER BY a.x;", dt);
    c("SELECT a.x FROM test a JOIN join_test b ON a.str = b.dup_str GROUP BY a.x ORDER "
      "BY a.x;",
      dt);
    c("SELECT a.x FROM test_inner_x a JOIN test_x b ON a.x = b.x GROUP BY a.x ORDER BY "
      "a.x;",
      dt);
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.x = test_inner.x AND test.rowid "
      "= test_inner.rowid;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.y = test_inner.y OR (test.y IS "
      "NULL AND test_inner.y IS NULL);",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.str = join_test.dup_str OR "
      "(test.str IS NULL AND "
      "join_test.dup_str IS NULL));",
      dt);
    c("SELECT t1.fixed_null_str FROM (SELECT fixed_null_str, SUM(x) n1 FROM test GROUP "
      "BY fixed_null_str) t1 INNER "
      "JOIN (SELECT fixed_null_str, SUM(y) n2 FROM test GROUP BY fixed_null_str) t2 ON "
      "((t1.fixed_null_str = "
      "t2.fixed_null_str) OR (t1.fixed_null_str IS NULL AND t2.fixed_null_str IS NULL));",
      dt);
  }
}

TEST(Delete, Joins_InnerJoin_AtLeastThreeTables) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON b.str = c.str;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON b.str = c.str JOIN "
      "join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT a.y, count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str "
      "GROUP BY a.y;",
      dt);
    c("SELECT a.x AS x, a.y, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = "
      "c.str "
      "ORDER BY a.y;",
      dt);
    c("SELECT a.x, b.x, b.str, c.str FROM test AS a JOIN join_test AS b ON a.x = b.x "
      "JOIN test_inner AS c ON b.x = c.x "
      "ORDER BY b.str;",
      dt);
    c("SELECT a.x, b.x, c.x FROM test a JOIN test_inner b ON a.x = b.x JOIN join_test c "
      "ON b.x = c.x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str JOIN "
      "hash_join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str JOIN "
      "join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON b.str = c.str JOIN "
      "hash_join_test AS d ON c.x = d.x;",
      dt);
    c("SELECT a.x AS x, a.y, b.str FROM test AS a JOIN hash_join_test AS b ON a.x = b.x "
      "JOIN test_inner AS c ON b.str "
      "= c.str "
      "ORDER BY a.y;",
      dt);
    c("SELECT a.x, b.x, c.x FROM test a JOIN test_inner b ON a.x = b.x JOIN "
      "hash_join_test c ON b.x = c.x;",
      dt);
    c("SELECT a.x, b.x FROM test_inner a JOIN test_inner b ON a.x = b.x ORDER BY a.x;",
      dt);
    c("SELECT a.x, b.x FROM join_test a JOIN join_test b ON a.x = b.x ORDER BY a.x;", dt);
    c("SELECT COUNT(1) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON a.t = c.x;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN test_inner b ON a.str = b.str JOIN "
      "hash_join_test c ON a.x = c.x JOIN "
      "join_test d ON a.x > d.x;",
      dt);
    c("SELECT a.x, b.str, c.str, d.y FROM hash_join_test a JOIN test b ON a.x = b.x JOIN "
      "join_test c ON b.x = c.x JOIN "
      "test_inner d ON b.x = d.x ORDER BY a.x, b.str;",
      dt);
  }

  g_enable_watchdog = save_watchdog;
}

TEST(Delete, Joins_InnerJoin_Filters) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT count(*) FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN test_inner "
      "AS c ON b.str = c.str WHERE a.y "
      "< 43;",
      dt);
    c("SELECT SUM(a.x), b.str FROM test AS a JOIN join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str "
      "WHERE a.y "
      "= 43 group by b.str;",
      dt);
    c("SELECT COUNT(*) FROM test JOIN test_inner ON test.str = test_inner.str AND test.x "
      "= 7;",
      dt);
    c("SELECT test.x, test_inner.str FROM test JOIN test_inner ON test.str = "
      "test_inner.str AND test.x <> 7;",
      dt);
    c("SELECT count(*) FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = c.str "
      "WHERE a.y "
      "< 43;",
      dt);
    c("SELECT SUM(a.x), b.str FROM test AS a JOIN hash_join_test AS b ON a.x = b.x JOIN "
      "test_inner AS c ON b.str = "
      "c.str "
      "WHERE a.y "
      "= 43 group by b.str;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.x = b.x JOIN test_inner c ON "
      "c.str = a.str WHERE c.str = "
      "'foo';",
      dt);
    c("SELECT COUNT(*) FROM test t1 JOIN test t2 ON t1.x = t2.x WHERE t1.y > t2.y;", dt);
    c("SELECT COUNT(*) FROM test t1 JOIN test t2 ON t1.x = t2.x WHERE t1.null_str = "
      "t2.null_str;",
      dt);
  }
}

TEST(Delete, Joins_LeftOuterJoin) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = "
      "test_inner.x ORDER BY test.x ASC;",
      dt);
    c("SELECT test.x key1, CASE WHEN test_inner.x IS NULL THEN 99 ELSE test_inner.x END "
      "key2 FROM test LEFT OUTER JOIN "
      "test_inner ON test.x = test_inner.x GROUP BY key1, key2 ORDER BY key1;",
      dt);
    c("SELECT test_inner.x key1 FROM test LEFT OUTER JOIN test_inner ON test.x = "
      "test_inner.x GROUP BY key1 HAVING "
      "key1 IS NOT NULL;",
      dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT JOIN test b ON a.x = b.x;", dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT OUTER JOIN test_x b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str;",
      dt);
    c("SELECT COUNT(*) FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str;",
      dt);
    c("SELECT a.x, b.str FROM test_inner a LEFT OUTER JOIN test_x b ON a.x = b.x ORDER "
      "BY a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str "
      "ORDER BY a.x, b.str IS NULL, "
      "b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT OUTER JOIN join_test b ON a.str = b.dup_str "
      "ORDER BY a.x, b.str IS NULL, "
      "b.str;",
      dt);
    c("SELECT COUNT(*) FROM test_inner_x a LEFT JOIN test_x b ON a.x = b.x;", dt);
    c("SELECT COUNT(*) FROM test a LEFT JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT COUNT(*) FROM test a LEFT JOIN join_test b ON a.str = b.dup_str;", dt);
    c("SELECT a.x, b.str FROM test_inner_x a LEFT JOIN test_x b ON a.x = b.x ORDER BY "
      "a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT JOIN join_test b ON a.str = b.dup_str ORDER BY "
      "a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a LEFT JOIN join_test b ON a.str = b.dup_str ORDER BY "
      "a.x, b.str IS NULL, b.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x = test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x < test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x > test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x >= test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test_inner.x <= test.x WHERE "
      "test_inner.str = test.str;",
      dt);
    c("SELECT test_inner.y, COUNT(*) n FROM test LEFT JOIN test_inner ON test_inner.x = "
      "test.x WHERE test_inner.str = "
      "'foo' GROUP BY test_inner.y ORDER BY n DESC;",
      dt);
    c("SELECT a.x, COUNT(b.y) FROM test a LEFT JOIN test_inner b ON b.x = a.x AND b.str "
      "NOT LIKE 'box' GROUP BY a.x "
      "ORDER BY a.x;",
      dt);
    c("SELECT a.x FROM test a LEFT OUTER JOIN test_inner b ON TRUE ORDER BY a.x ASC;",
      "SELECT a.x FROM test a LEFT OUTER JOIN test_inner b ON 1 ORDER BY a.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x > test_inner.x INNER "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x > test_inner.x INNER "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x = test_inner.x INNER "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test LEFT JOIN test_inner ON "
      "test.x = test_inner.x INNER "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner "
      "ON test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner "
      "ON test.x > test_inner.x LEFT "
      "JOIN hash_join_test ON test.str <> hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner "
      "ON test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC NULLS FIRST, hash_join_test.x ASC NULLS FIRST;",
      "SELECT test_inner.y, hash_join_test.x, COUNT(*) FROM test INNER JOIN test_inner "
      "ON test.x = test_inner.x LEFT "
      "JOIN hash_join_test ON test.str = hash_join_test.str GROUP BY test_inner.y, "
      "hash_join_test.x ORDER BY "
      "test_inner.y ASC, hash_join_test.x ASC;",
      dt);
    c("SELECT COUNT(*) FROM test LEFT JOIN test_inner ON test.str = test_inner.str AND "
      "test.x = test_inner.x;",
      dt);
  }

  g_enable_watchdog = save_watchdog;
}

TEST(Delete, Joins_LeftJoin_Filters) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = "
      "test_inner.x WHERE test.y > 40 "
      "ORDER BY test.x ASC;",
      dt);
    c("SELECT test.x, test_inner.x FROM test LEFT OUTER JOIN test_inner ON test.x = "
      "test_inner.x WHERE test.y > 42 "
      "ORDER BY test.x ASC;",
      dt);
    c("SELECT test.str AS foobar, test_inner.str FROM test LEFT OUTER JOIN test_inner ON "
      "test.x = test_inner.x WHERE "
      "test.y > 42 ORDER BY foobar DESC LIMIT 8;",
      dt);
    c("SELECT test.x AS foobar, test_inner.x AS inner_foobar, test.f AS f_foobar FROM "
      "test LEFT OUTER JOIN test_inner "
      "ON test.str = test_inner.str WHERE test.y > 40 ORDER BY foobar DESC, f_foobar "
      "DESC;",
      dt);
    c("SELECT test.str AS foobar, test_inner.str FROM test LEFT OUTER JOIN test_inner ON "
      "test.x = test_inner.x WHERE "
      "test_inner.str IS NOT NULL ORDER BY foobar DESC;",
      dt);
    c("SELECT COUNT(*) FROM test_inner a LEFT JOIN (SELECT * FROM test WHERE y > 40) b "
      "ON a.x = b.x;",
      dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN (SELECT * FROM test WHERE y > 40) b "
      "ON a.x = b.x ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT COUNT(*) FROM join_test a LEFT JOIN test b ON a.x = b.x AND a.x = 7;", dt);
    c("SELECT a.x, b.str FROM join_test a LEFT JOIN test b ON a.x = b.x AND a.x = 7 "
      "ORDER BY a.x, b.str;",
      dt);
    c("SELECT COUNT(*) FROM join_test a LEFT JOIN test b ON a.x = b.x WHERE a.x = 7;",
      dt);
    c("SELECT a.x FROM join_test a LEFT JOIN test b ON a.x = b.x WHERE a.x = 7;", dt);
  }
}

TEST(Delete, Joins_MultiCompositeColumns) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT a.x, b.str FROM test AS a JOIN join_test AS b ON a.str = b.str AND a.x = "
      "b.x ORDER BY a.x, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test AS a JOIN join_test AS b ON a.x = b.x AND a.str = "
      "b.str ORDER BY a.x, b.str;",
      dt);
    c("SELECT a.z, b.str FROM test a JOIN join_test b ON a.y = b.y AND a.x = b.x ORDER "
      "BY a.z, b.str;",
      dt);
    c("SELECT a.z, b.str FROM test a JOIN test_inner b ON a.y = b.y AND a.x = b.x ORDER "
      "BY a.z, b.str;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN join_test b ON a.x = b.x AND a.y = b.x JOIN "
      "test_inner c ON a.x = c.x WHERE "
      "c.str <> 'foo';",
      dt);
    c("SELECT a.x, b.x, d.str FROM test a JOIN test_inner b ON a.str = b.str JOIN "
      "hash_join_test c ON a.x = c.x JOIN "
      "join_test d ON a.x >= d.x AND a.x < d.x + 5 ORDER BY a.x, b.x;",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.x = join_test.x OR (test.x IS "
      "NULL AND join_test.x IS NULL)) "
      "AND (test.y = join_test.y OR (test.y IS NULL AND join_test.y IS NULL));",
      dt);
    c("SELECT COUNT(*) FROM test, join_test WHERE (test.str = join_test.dup_str OR "
      "(test.str IS NULL AND "
      "join_test.dup_str IS NULL)) AND (test.x = join_test.x OR (test.x IS NULL AND "
      "join_test.x IS NULL));",
      dt);
  }
}

TEST(Delete, Joins_BuildHashTable) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, join_test WHERE test.str = join_test.dup_str;", dt);
    // Intentionally duplicate previous string join to cover hash table building.
    c("SELECT COUNT(*) FROM test, join_test WHERE test.str = join_test.dup_str;", dt);
  }
}

TEST(Delete, Joins_ComplexQueries) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test a JOIN (SELECT * FROM test WHERE y < 43) b ON a.x = b.x "
      "JOIN join_test c ON a.x = c.x "
      "WHERE a.fixed_str = 'foo';",
      dt);
    c("SELECT * FROM (SELECT a.y, b.str FROM test a JOIN join_test b ON a.x = b.x) ORDER "
      "BY y, str;",
      dt);
    c("SELECT x, dup_str FROM (SELECT * FROM test a JOIN join_test b ON a.x = b.x) WHERE "
      "y > 40 ORDER BY x, dup_str;",
      dt);
    c("SELECT a.x FROM (SELECT * FROM test WHERE x = 8) AS a JOIN (SELECT * FROM "
      "test_inner WHERE x = 7) AS b ON a.str "
      "= b.str WHERE a.y < 42;",
      dt);
    c("SELECT a.str as key0,a.fixed_str as key1,COUNT(*) AS color FROM test a JOIN "
      "(select str,count(*) "
      "from test group by str order by COUNT(*) desc limit 40) b on a.str=b.str JOIN "
      "(select "
      "fixed_str,count(*) from test group by fixed_str order by count(*) desc limit 40) "
      "c on "
      "c.fixed_str=a.fixed_str GROUP BY key0, key1 ORDER BY key0,key1;",
      dt);
    c("SELECT COUNT(*) FROM test a JOIN (SELECT str FROM test) b ON a.str = b.str OR "
      "false;",
      "SELECT COUNT(*) FROM test a JOIN (SELECT str FROM test) b ON a.str = b.str OR 0;",
      dt);
  }
}

TEST(Delete, Joins_TimeAndDate) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test a, test b WHERE a.m = b.m;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.n = b.n;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o;", dt);
  }
}

TEST(Delete, Joins_OneOuterExpression) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

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

TEST(Delete, Joins_MultipleOuterExpressions) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x - 1 = test_inner.x AND "
      "test.str = test_inner.str;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 0 = test_inner.x AND "
      "test.str = test_inner.str;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str AND test.x "
      "+ 0 = test_inner.x;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 1 = test_inner.x AND "
      "test.str = test_inner.str;",
      dt);
    // The following query will fallback to loop join because we don't reorder the
    // expressions to be consistent with table order for composite equality yet.
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x + 0 = test_inner.x AND "
      "test_inner.str = test.str;",
      dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o + INTERVAL '0' DAY = b.o AND a.str "
      "= b.str;",
      "SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o AND a.str = b.str;",
      dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.o + INTERVAL '0' DAY = b.o AND a.x = "
      "b.x;",
      "SELECT COUNT(*) FROM test a, test b WHERE a.o = b.o AND a.x = b.x;",
      dt);
  }
}

TEST(Delete, ExtraFragment) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }

  auto insert_op = [](int random_val) -> std::string {
    std::ostringstream insert_string;
    insert_string << "insert into vacuum_test values (" << random_val << ", '"
                  << random_val << "');";
    return insert_string.str();
  };

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists vacuum_test;");
    run_ddl_statement(
        "create table vacuum_test (i1 integer, t1 text) with (vacuum='delayed', "
        "fragment_size=10);");
    for (int i = 1; i <= 100; i++) {
      run_multiple_agg(insert_op(i), dt);
    }
    run_multiple_agg("delete from vacuum_test where i1 > 50;", dt);
    ASSERT_EQ(int64_t(50),
              v<int64_t>(run_simple_agg("SELECT COUNT(i1) FROM vacuum_test;", dt)));
    run_ddl_statement("drop table vacuum_test;");
  }
}

TEST(Delete, Joins_ImplicitJoins) {
  SKIP_ALL_ON_AGGREGATOR();

  if (std::is_same<CalciteDeletePathSelector, PreprocessorFalse>::value) {
    return;
  }
  auto save_watchdog = g_enable_watchdog;
  g_enable_watchdog = false;

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("DELETE FROM test WHERE test.x = 8;", dt);

    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.x;", dt);
    c("SELECT COUNT(*) FROM test, hash_join_test WHERE test.t = hash_join_test.t;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x < test_inner.x + 1;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.real_str = test_inner.str;", dt);
    c("SELECT test_inner.x, COUNT(*) AS n FROM test, test_inner WHERE test.x = "
      "test_inner.x GROUP BY test_inner.x "
      "ORDER BY n;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str;", dt);
    c("SELECT test.str, COUNT(*) FROM test, test_inner WHERE test.str = test_inner.str "
      "GROUP BY test.str;",
      dt);
    c("SELECT test_inner.str, COUNT(*) FROM test, test_inner WHERE test.str = "
      "test_inner.str GROUP BY test_inner.str;",
      dt);
    c("SELECT test.str, COUNT(*) AS foobar FROM test, test_inner WHERE test.x = "
      "test_inner.x AND test.x > 6 GROUP BY "
      "test.str HAVING foobar > 5;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.real_str LIKE 'real_ba%' AND "
      "test.x = test_inner.x;",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE LENGTH(test.real_str) = 8 AND test.x "
      "= test_inner.x;",
      dt);
    c("SELECT a.x, b.str FROM test a, join_test b WHERE a.str = b.str GROUP BY a.x, "
      "b.str ORDER BY a.x, b.str;",
      dt);
    c("SELECT a.x, b.str FROM test a, join_test b WHERE a.str = b.str ORDER BY a.x, "
      "b.str;",
      dt);
    c("SELECT COUNT(1) FROM test a, join_test b, test_inner c WHERE a.str = b.str AND "
      "b.x = c.x",
      dt);
    c("SELECT COUNT(*) FROM test a, join_test b, test_inner c WHERE a.x = b.x AND a.y = "
      "b.x AND a.x = c.x AND c.str = "
      "'foo';",
      dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.x = b.x AND a.y = b.y;", dt);
    c("SELECT COUNT(*) FROM test a, test b WHERE a.x = b.x AND a.str = b.str;", dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE (test.x = test_inner.x AND test.y = "
      "42 AND test_inner.str = 'foo') "
      "OR (test.x = test_inner.x AND test.y = 43 AND test_inner.str = 'foo');",
      dt);
    c("SELECT COUNT(*) FROM test, test_inner WHERE test.x = test_inner.x OR test.x = "
      "test_inner.x;",
      dt);
    c("SELECT bar.str FROM test, bar WHERE test.str = bar.str;", dt);
    ASSERT_EQ(
        int64_t(3),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM test, join_test WHERE test.rowid = join_test.rowid;",
            dt)));
    ASSERT_EQ(7,
              v<int64_t>(run_simple_agg("SELECT test.x FROM test, test_inner WHERE "
                                        "test.x = test_inner.x AND test.rowid = 9;",
                                        dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test, test_inner WHERE "
                                        "test.x = test_inner.x AND test.rowid = 20;",
                                        dt)));
  }
  g_enable_watchdog = save_watchdog;
}

TEST(Create, Delete) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    run_ddl_statement("drop table if exists vacuum_test;");
    run_ddl_statement(
        "create table vacuum_test (i1 integer, t1 text) with (vacuum='delayed');");
    run_multiple_agg("insert into vacuum_test values(1, '1');", dt);
    run_multiple_agg("insert into vacuum_test values(2, '2');", dt);
    ASSERT_EQ(int64_t(3),
              v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM vacuum_test;", dt)));
    run_multiple_agg("insert into vacuum_test values(3, '3');", dt);
    SKIP_ON_AGGREGATOR(run_multiple_agg("insert into vacuum_test values(4, '4');", dt));
    SKIP_ON_AGGREGATOR(run_multiple_agg("delete from vacuum_test where i1 = 4;", dt));
    ASSERT_EQ(int64_t(6),
              v<int64_t>(run_simple_agg("SELECT SUM(i1) FROM vacuum_test;", dt)));
    run_ddl_statement("drop table vacuum_test;");
  }
}

#if 0
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
#endif

TEST(Select, GeoSpatial_Basics) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(p,p) < 0.1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT count(p) FROM geospatial_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT count(l) FROM geospatial_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT count(poly) FROM geospatial_test;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows),
        v<int64_t>(run_simple_agg("SELECT count(mpoly) FROM geospatial_test;", dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Distance('POINT(0 0)', p) < 100.0;",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(7),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                  "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 9;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(5),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Distance(p,l) < 2.0;", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance('LINESTRING(-1 0, 0 1)', p) < 0.8;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(2),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance('LINESTRING(-1 0, 0 1)', p) < 1.1;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test "
                                  "WHERE ST_Distance(p, 'LINESTRING(-1 0, 0 1)') < 2.5;",
                                  dt)));

    // Unsupported aggs
    EXPECT_THROW(run_simple_agg("SELECT MIN(p) FROM geospatial_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT MAX(p) FROM geospatial_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT AVG(p) FROM geospatial_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(run_simple_agg("SELECT SUM(p) FROM geospatial_test;", dt),
                 std::runtime_error);
  }
}

TEST(Select, GeoSpatial_Projection) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    // Select *
    {
      const auto rows =
          run_multiple_agg("SELECT * FROM geospatial_test WHERE id = 1", dt);
      const auto row = rows->getNextRow(false, false);
      ASSERT_EQ(row.size(), size_t(11));
    }

    // Projection (return GeoTargetValue)
    compare_geo_target(run_simple_agg("SELECT p FROM geospatial_test WHERE id = 1;", dt),
                       GeoPointTargetValue({1., 1.}));
    compare_geo_target(run_simple_agg("SELECT l FROM geospatial_test WHERE id = 1;", dt),
                       GeoLineStringTargetValue({1., 0., 2., 2., 3., 3.}));
    compare_geo_target(
        run_simple_agg("SELECT poly FROM geospatial_test WHERE id = 1;", dt),
        GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}));
    compare_geo_target(
        run_simple_agg("SELECT mpoly FROM geospatial_test WHERE id = 1;", dt),
        GeoMultiPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}, {1}));

    // Sample() version of above
    SKIP_ON_AGGREGATOR(
        compare_geo_target(
            run_simple_agg("SELECT SAMPLE(p) FROM geospatial_test WHERE id = 1;", dt),
            GeoPointTargetValue({1., 1.}));
        compare_geo_target(
            run_simple_agg("SELECT SAMPLE(l) FROM geospatial_test WHERE id = 1;", dt),
            GeoLineStringTargetValue({1., 0., 2., 2., 3., 3.}));
        compare_geo_target(
            run_simple_agg("SELECT SAMPLE(poly) FROM geospatial_test WHERE id = 1;", dt),
            GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}));
        compare_geo_target(
            run_simple_agg("SELECT SAMPLE(mpoly) FROM geospatial_test WHERE id = 1;", dt),
            GeoMultiPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}, {1})));

    // Sample() version of above with GROUP BY
    compare_geo_target(
        run_simple_agg("SELECT SAMPLE(p) FROM geospatial_test WHERE id = 1 GROUP BY id;",
                       dt),
        GeoPointTargetValue({1., 1.}));
    compare_geo_target(
        run_simple_agg("SELECT SAMPLE(l) FROM geospatial_test WHERE id = 1 GROUP BY id;",
                       dt),
        GeoLineStringTargetValue({1., 0., 2., 2., 3., 3.}));
    compare_geo_target(
        run_simple_agg(
            "SELECT SAMPLE(poly) FROM geospatial_test WHERE id = 1 GROUP BY id;", dt),
        GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}));
    compare_geo_target(
        run_simple_agg(
            "SELECT SAMPLE(mpoly) FROM geospatial_test WHERE id = 1 GROUP BY id;", dt),
        GeoMultiPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}, {1}));

    // Sample() with compression
    compare_geo_target(
        run_simple_agg(
            "SELECT SAMPLE(gp4326) FROM geospatial_test WHERE id = 1 GROUP BY id;", dt),
        GeoPointTargetValue({1., 1.}),
        0.01);
    compare_geo_target(
        run_simple_agg(
            "SELECT SAMPLE(gpoly4326) FROM geospatial_test WHERE id = 1 GROUP BY id;",
            dt),
        GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}),
        0.01);

    // Sample with multiple aggs
    {
      const auto rows = run_multiple_agg(
          "SELECT COUNT(*), SAMPLE(l) FROM geospatial_test WHERE id = 1 GROUP BY id;",
          dt);
      rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
      const auto row = rows->getNextRow(false, false);
      CHECK_EQ(row.size(), size_t(2));
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(row[0]));
      compare_geo_target(row[1], GeoLineStringTargetValue({1., 0., 2., 2., 3., 3.}));
    }
    {
      const auto rows = run_multiple_agg(
          "SELECT COUNT(*), SAMPLE(poly) FROM geospatial_test WHERE id = 1 GROUP BY id;",
          dt);
      rows->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
      const auto row = rows->getNextRow(false, false);
      CHECK_EQ(row.size(), size_t(2));
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(row[0]));
      compare_geo_target(row[1], GeoPolyTargetValue({0., 0., 2., 0., 0., 2.}, {3}));
    }

    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                  "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                                  dt)));
    compare_geo_target(
        run_simple_agg("SELECT p FROM geospatial_test WHERE "
                       "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                       dt),
        GeoPointTargetValue({0, 0}));

    compare_geo_target(
        get_first_target("SELECT p, l FROM geospatial_test WHERE "
                         "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                         dt),
        GeoPointTargetValue({0, 0}));
    compare_geo_target(
        get_first_target("SELECT p, ST_Distance(ST_GeomFromText('POINT(0 0)'), p), l "
                         "FROM geospatial_test "
                         "WHERE ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                         dt),
        GeoPointTargetValue({0, 0}));
    compare_geo_target(
        get_first_target("SELECT l, ST_Distance(ST_GeomFromText('POINT(0 0)'), p), p "
                         "FROM geospatial_test "
                         "WHERE ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                         dt),
        GeoLineStringTargetValue({0., 0., 0., 0.}));
    ASSERT_EQ(
        static_cast<double>(0.),
        v<double>(get_first_target("SELECT ST_Distance(ST_GeomFromText('POINT(0 0)'), "
                                   "p), p, l FROM geospatial_test WHERE "
                                   "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                                   dt)));

    compare_geo_target(
        run_simple_agg("SELECT l FROM geospatial_test WHERE "
                       "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                       dt),
        GeoLineStringTargetValue({0., 0., 0., 0.}));
    compare_geo_target(
        run_simple_agg("SELECT l FROM geospatial_test WHERE "
                       "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) BETWEEN 7 AND 8;",
                       dt),
        GeoLineStringTargetValue({5., 0., 10., 10., 11., 11.}));
    compare_geo_target(
        run_simple_agg("SELECT gp4326 FROM geospatial_test WHERE "
                       "ST_Distance(ST_GeomFromText('POINT(0 0)'), "
                       "p) > 1 AND ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 2",
                       dt),
        GeoPointTargetValue({0.9999, 0.9999}),
        0.01);

    // Projection (return WKT strings)
    ASSERT_EQ("POINT (1 1)",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT p FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("LINESTRING (1 0,2 2,3 3)",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT l FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("POLYGON ((0 0,2 0,0 2,0 0))",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT poly FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("MULTIPOLYGON (((0 0,2 0,0 2,0 0)))",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT mpoly FROM geospatial_test WHERE id = 1;", dt, false))));
    ASSERT_EQ("LINESTRING (5 0,10 10,11 11)",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT l FROM geospatial_test WHERE "
                  "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) BETWEEN 7 AND 8;",
                  dt,
                  false))));
    ASSERT_EQ("LINESTRING (0 0,0 0)",
              boost::get<std::string>(v<NullableString>(
                  get_first_target("SELECT l, p FROM geospatial_test WHERE "
                                   "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                                   dt,
                                   false))));
    ASSERT_EQ("POINT (0 0)",
              boost::get<std::string>(v<NullableString>(
                  get_first_target("SELECT p, l FROM geospatial_test WHERE "
                                   "ST_Distance(ST_GeomFromText('POINT(0 0)'), p) < 1;",
                                   dt,
                                   false))));

    // ST_Distance
    ASSERT_NEAR(static_cast<double>(2.0),
                v<double>(run_simple_agg(
                    "SELECT ST_Distance('LINESTRING(-2 2, 2 2)', 'LINESTRING(4 2, 4 3)') "
                    "from geospatial_test limit 1;",
                    dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(0.0),
                v<double>(run_simple_agg("SELECT ST_Distance('LINESTRING(-2 2, 2 2, 2 "
                                         "0)', 'LINESTRING(4 0, 0 -4, -4 0, 0 4)') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(0.31),
                v<double>(run_simple_agg("SELECT ST_Distance('LINESTRING(-2 2, 2 2, 2 "
                                         "0)', 'LINESTRING(4 0, 0 -4, -4 0, 0 5)') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(3.0),
                v<double>(run_simple_agg(
                    "SELECT ST_Distance(ST_GeomFromText('POINT(5 -1)'),"
                    "ST_GeomFromText('POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))')) "
                    "from geospatial_test limit 1;",
                    dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(0.0),
                v<double>(run_simple_agg("SELECT ST_Distance(ST_GeomFromText("
                                         "'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))'), "
                                         "ST_GeomFromText('POINT(0.5 0.5)')) "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.5),
        v<double>(run_simple_agg(
            "SELECT ST_Distance(ST_GeomFromText("
            "'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2), (1 1, -1 1, -1 -1, 1 -1, 1 1))'), "
            "ST_GeomFromText('POINT(0.5 0.5)')) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(0.0),
                v<double>(run_simple_agg("SELECT ST_Distance(ST_GeomFromText("
                                         "'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))'), "
                                         "ST_GeomFromText('LINESTRING(0.5 0.5, 0.7 0.75, "
                                         "-0.3 -0.3, -0.82 0.12, 0.3 0.64)')) "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.18),
        v<double>(run_simple_agg(
            "SELECT ST_Distance(ST_GeomFromText("
            "'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2), (1 1, -1 1, -1 -1, 1 -1, 1 1))'), "
            "ST_GeomFromText('LINESTRING(0.5 0.5, 0.7 0.75, -0.3 -0.3, -0.82 0.12, 0.3 "
            "0.64)')) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(0.0),
                v<double>(run_simple_agg(
                    "SELECT ST_Distance("
                    "'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2))', "
                    "'POLYGON((0.5 0.5, -0.5 0.5, -0.5 -0.5, 0.5 -0.5, 0.5 0.5))') "
                    "from geospatial_test limit 1;",
                    dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.5),
        v<double>(run_simple_agg(
            "SELECT ST_Distance("
            "'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2), (1 1, -1 1, -1 -1, 1 -1, 1 1))', "
            "'POLYGON((0.5 0.5, -0.5 0.5, -0.5 -0.5, 0.5 -0.5, 0.5 0.5))') "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(2.0),
        v<double>(run_simple_agg(
            "SELECT ST_Distance("
            "'POLYGON((2 2, -2 2, -2 -2, 2 -2, 2 2), (1 1, -1 1, -1 -1, 1 -1, 1 1))', "
            "'POLYGON((4 2, 5 2, 5 3, 4 3, 4 2))') "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(1.4142),
        v<double>(run_simple_agg(
            "SELECT ST_Distance("
            "'POLYGON((0 0, 4 0, 4 4, 2 5, 0 4, 0 0), (1 1, 1 3, 2 4, 3 3, 3 1, 1 1))', "
            "'POLYGON((5 5, 8 2, 8 4, 5 5))') "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            "SELECT ST_Distance("
            "'POLYGON((0 0, 4 0, 4 4, 2 5, 0 4, 0 0), (1 1, 1 3, 2 4, 3 3, 3 1, 1 1))', "
            "'POLYGON((3.5 3.5, 8 2, 8 4, 3.5 3.5))') "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(0.0),
        v<double>(run_simple_agg(
            "SELECT ST_Distance("
            "'POLYGON((0 0, 4 0, 4 4, 2 5, 0 4, 0 0), (1 1, 1 3, 2 4, 3 3, 3 1, 1 1))', "
            "'POLYGON((8 2, 8 4, 2 2, 8 2))') "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(2.0),
                v<double>(run_simple_agg("SELECT ST_Distance("
                                         "'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), "
                                         "((1 1, -1 1, -1 -1, 1 -1, 1 1)))', "
                                         "'POINT(4 2)') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(2.0),
                v<double>(run_simple_agg("SELECT ST_Distance("
                                         "'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), "
                                         "((1 1, -1 1, -1 -1, 1 -1, 1 1)))', "
                                         "'LINESTRING(4 2, 5 3)') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(2.0),
                v<double>(run_simple_agg("SELECT ST_Distance("
                                         "'LINESTRING(4 2, 5 3)', "
                                         "'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), "
                                         "((1 1, -1 1, -1 -1, 1 -1, 1 1)))') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(2.0),
                v<double>(run_simple_agg("SELECT ST_Distance("
                                         "'POLYGON((4 2, 5 3, 4 3))', "
                                         "'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), "
                                         "((1 1, -1 1, -1 -1, 1 -1, 1 1)))') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(2.0),
                v<double>(run_simple_agg("SELECT ST_Distance("
                                         "'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), "
                                         "((1 1, -1 1, -1 -1, 1 -1, 1 1)))', "
                                         "'POLYGON((4 2, 5 3, 4 3))') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(1.4142),
                v<double>(run_simple_agg("SELECT ST_Distance("
                                         "'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), "
                                         "((1 1, -1 1, -1 -1, 1 -1, 1 1)))', "
                                         "'MULTIPOLYGON(((4 2, 5 3, 4 3)), "
                                         "((3 3, 4 3, 3 4)))') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.01));

    ASSERT_NEAR(static_cast<double>(25.4558441),
                v<double>(run_simple_agg(
                    "SELECT ST_MaxDistance('POINT(1 1)', 'LINESTRING (9 0,18 18,19 19)') "
                    "FROM geospatial_test limit 1;",
                    dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(26.87005768),
                v<double>(run_simple_agg("SELECT Max(ST_MaxDistance(l, 'POINT(0 0)')) "
                                         "FROM geospatial_test;",
                                         dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(14.142135),
                v<double>(run_simple_agg(
                    " SELECT Max(ST_MaxDistance(p, l)) FROM geospatial_test;", dt)),
                static_cast<double>(0.01));

    // Geodesic distance between Paris and LA geographic points: ~9105km
    ASSERT_NEAR(
        static_cast<double>(9105643.0),
        v<double>(run_simple_agg(
            "SELECT ST_Distance(ST_GeogFromText('POINT(-118.4079 33.9434)', 4326), "
            "ST_GeogFromText('POINT(2.5559 49.0083)', 4326)) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(10000.0));
    // Geodesic distance between Paris and LA geometry points cast as geography points:
    // ~9105km
    ASSERT_NEAR(static_cast<double>(9105643.0),
                v<double>(run_simple_agg(
                    "SELECT ST_Distance(CastToGeography(ST_GeomFromText('POINT(-118.4079 "
                    "33.9434)', 4326)), "
                    "cast (ST_GeomFromText('POINT(2.5559 49.0083)', 4326) as geography)) "
                    "from geospatial_test limit 1;",
                    dt)),
                static_cast<double>(10000.0));
    // Cartesian distance between Paris and LA calculated from wgs84 degrees
    ASSERT_NEAR(
        static_cast<double>(121.89),
        v<double>(run_simple_agg(
            "SELECT ST_Distance(ST_GeomFromText('POINT(-118.4079 33.9434)', 4326), "
            "ST_GeomFromText('POINT(2.5559 49.0083)', 4326)) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(1.0));
    ASSERT_NEAR(
        static_cast<double>(121.89),
        v<double>(run_simple_agg(
            "SELECT ST_Distance('POINT(-118.4079 33.9434)', 'POINT(2.5559 49.0083)') "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(1.0));
    // Cartesian distance between Paris and LA wgs84 coords transformed to web merc
    ASSERT_NEAR(
        static_cast<double>(13653148.0),
        v<double>(run_simple_agg(
            "SELECT ST_Distance("
            "ST_Transform(ST_GeomFromText('POINT(-118.4079 33.9434)', 4326), 900913), "
            "ST_Transform(ST_GeomFromText('POINT(2.5559 49.0083)', 4326), 900913)) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(10000.0));

    // ST_Length
    // Cartesian length of a planar path
    ASSERT_NEAR(static_cast<double>(5.65685),
                v<double>(run_simple_agg(
                    "SELECT ST_Length('LINESTRING(1 0, 0 1, -1 0, 0 -1, 1 0)') "
                    "from geospatial_test limit 1;",
                    dt)),
                static_cast<double>(0.0001));
    // Geodesic length of a geographic path, in meters
    ASSERT_NEAR(
        static_cast<double>(617121.626),
        v<double>(run_simple_agg(
            "SELECT ST_Length(CAST (ST_GeomFromText('LINESTRING(-76.6168198439371 "
            "39.9703199555959, -80.5189990254673 40.6493554919257, -82.5189990254673 "
            "42.6493554919257)', 4326) as GEOGRAPHY)) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));

    // ST_Perimeter
    // Cartesian perimeter of a planar polygon
    ASSERT_NEAR(static_cast<double>(5.65685),
                v<double>(run_simple_agg("SELECT ST_Perimeter('POLYGON("
                                         "(1 0, 0 1, -1 0, 0 -1, 1 0),"
                                         "(0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0))') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.0001));
    // Geodesic perimeter of a polygon geography, in meters
    ASSERT_NEAR(
        static_cast<double>(1193066.02892),
        v<double>(run_simple_agg(
            "SELECT ST_Perimeter(ST_GeogFromText('POLYGON((-76.6168198439371 "
            "39.9703199555959, -80.5189990254673 40.6493554919257, -82.5189990254673 "
            "42.6493554919257, -76.6168198439371 39.9703199555959))', 4326)) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    // Cartesian perimeter of a planar multipolygon
    ASSERT_NEAR(static_cast<double>(4 * 1.41421 + 4 * 2.82842),
                v<double>(run_simple_agg("SELECT ST_Perimeter('MULTIPOLYGON("
                                         "((1 0, 0 1, -1 0, 0 -1, 1 0),"
                                         " (0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0)), "
                                         "((2 0, 0 2, -2 0, 0 -2, 2 0),"
                                         " (0.2 0, 0 0.2, -0.2 0, 0 -0.2, 0.2 0)))') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.0001));
    // Geodesic perimeter of a polygon geography, in meters
    ASSERT_NEAR(
        static_cast<double>(1193066.02892 + 1055903.62342 + 907463.55601),
        v<double>(run_simple_agg(
            "SELECT ST_Perimeter(ST_GeogFromText('MULTIPOLYGON("
            "((-76.6168198439371 39.9703199555959, -80.5189990254673 40.6493554919257,"
            "  -82.5189990254673 42.6493554919257, -76.6168198439371 39.9703199555959)), "
            "((-66.6168198439371 49.9703199555959, -70.5189990254673 50.6493554919257,"
            "  -72.5189990254673 52.6493554919257, -66.6168198439371 49.9703199555959)), "
            "((-56.6168198439371 59.9703199555959, -60.5189990254673 60.6493554919257,"
            "  -62.5189990254673 62.6493554919257, -56.6168198439371 59.9703199555959)))'"
            ", 4326)) from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));

    // ST_Area
    // Area of a planar polygon
    ASSERT_NEAR(static_cast<double>(2.0 - 0.02),
                v<double>(run_simple_agg("SELECT ST_Area('POLYGON("
                                         "(1 0, 0 1, -1 0, 0 -1, 1 0),"
                                         "(0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0))') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.0001));
    // Area of a planar multipolygon
    ASSERT_NEAR(static_cast<double>(2.0 - 0.02 + 8.0 - 0.08),
                v<double>(run_simple_agg("SELECT ST_Area('MULTIPOLYGON("
                                         "((1 0, 0 1, -1 0, 0 -1, 1 0),"
                                         " (0.1 0, 0 0.1, -0.1 0, 0 -0.1, 0.1 0)), "
                                         "((2 0, 0 2, -2 0, 0 -2, 2 0),"
                                         " (0.2 0, 0 0.2, -0.2 0, 0 -0.2, 0.2 0)))') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.0001));

    // ST_Intersects
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Intersects("
                  "ST_GeomFromText('POINT(0.9 0.9)'), "
                  "ST_GeomFromText('POINT(1.1 1.1)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("SELECT ST_Intersects("
                                        "ST_GeomFromText('POINT(1 1)'), "
                                        "ST_GeomFromText('LINESTRING(2 0, 0 2, -2 0, 0 "
                                        "-2)')) FROM geospatial_test limit 1;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Intersects("
                  "ST_GeomFromText('LINESTRING(2 0, 0 2, -2 0, 0 -2)'), "
                  "ST_GeomFromText('POINT(1 0)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg("SELECT ST_Intersects("
                                        "ST_GeomFromText('POINT(1 1)'), "
                                        "ST_GeomFromText('POLYGON((0 0, 1 0, 0 1, 0 "
                                        "0))')) FROM geospatial_test limit 1;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Intersects("
                  "ST_GeomFromText('POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))'), "
                  "ST_GeomFromText('POINT(1 1)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Intersects("
            "ST_GeomFromText('POINT(1 1)'), "
            "ST_GeomFromText('MULTIPOLYGON(((5 5, 6 6, 5 6)), ((0 0, 1 0, 0 1, 0 0)))')) "
            " FROM geospatial_test limit 1;",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Intersects("
                  "ST_GeomFromText('MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)), ((5 5, 6 "
                  "6, 5 6)))'), "
                  "ST_GeomFromText('POINT(1 1)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg("SELECT ST_Intersects("
                                  "ST_GeomFromText('LINESTRING(1 1, 0.5 1.5, 2 4)'), "
                                  "ST_GeomFromText('LINESTRING(2 0, 0 2, -2 0, 0 -2)')) "
                                  "FROM geospatial_test limit 1;",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Intersects("
                  "ST_GeomFromText('LINESTRING(1 1, 0.5 1.5, 1.5 1, 1.5 1.5)'), "
                  "ST_GeomFromText('LINESTRING(1 0, 0 1, -1 0, 0 -1)')) FROM "
                  "geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg("SELECT ST_Intersects("
                                  "ST_GeomFromText('LINESTRING(3 3, 3 2, 2.1 2.1)'), "
                                  "ST_GeomFromText('POLYGON((2 2, 0 1, -2 2, -2 0, 2 0, "
                                  "2 2))')) FROM geospatial_test limit 1;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Intersects("
            "ST_GeomFromText('POLYGON((2 2, 0 1, -2 2, -2 0, 2 0, 2 2))'), "
            "ST_GeomFromText('LINESTRING(3 3, 3 2, 2 2)')) FROM geospatial_test limit 1;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg("SELECT ST_Intersects("
                                  "ST_GeomFromText('LINESTRING(3 3, 3 2, 2.1 2.1)'), "
                                  "ST_GeomFromText('MULTIPOLYGON(((5 5, 6 6, 5 6)), ((2 "
                                  "2, 0 1, -2 2, -2 0, 2 0, 2 2)))')) "
                                  " FROM geospatial_test limit 1;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Intersects("
            "ST_GeomFromText('MULTIPOLYGON(((2 2, 0 1, -2 2, -2 0, 2 0, 2 2)), ((5 5, 6 "
            "6, 5 6)))'), "
            "ST_GeomFromText('LINESTRING(3 3, 3 2, 2 2)')) FROM geospatial_test limit 1;",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Intersects("
                  "ST_GeomFromText('POLYGON((-118.66313066279504 44.533565793694436,"
                  "-115.28301791070872 44.533565793694436,-115.28301791070872 "
                  "46.49961643537853,"
                  "-118.66313066279504 46.49961643537853,-118.66313066279504 "
                  "44.533565793694436))'),"
                  "ST_GeomFromText('LINESTRING (-118.526348964556 45.6369689645418,"
                  "-118.568716970537 45.552529965319,-118.604668964913 45.5192699867856,"
                  "-118.700612922525 45.4517749629224)')) from test limit 1;",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Intersects("
            "ST_GeomFromText('POLYGON((-165.27254008488316 60.286744877866084,"
            "-164.279755308478 60.286744877866084, -164.279755308478 60.818880025426154,"
            "-165.27254008488316 60.818880025426154))', 4326), "
            "ST_GeomFromText('MULTIPOLYGON (((-165.273152946156 60.5488599839382,"
            "-165.244307548387 60.4963022239955,-165.23881195357 60.4964759808483,"
            "-165.234271979534 60.4961199595109,-165.23165799921 60.496354988076,"
            "-165.229399998313 60.4973489979735,-165.225239975948 60.4977589987674,"
            "-165.217958113746 60.4974514248303,-165.21276192051 60.4972319866052)))', "
            "4326)) FROM geospatial_test limit "
            "1;",
            dt)));

    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Intersects(p,p);", dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(
                  run_simple_agg("SELECT count(*) FROM geospatial_test "
                                 "WHERE ST_Intersects(p, ST_GeomFromText('POINT(0 0)'));",
                                 dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(p, ST_GeomFromText('LINESTRING(0 0, 5 5)'));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(p, ST_GeomFromText('LINESTRING(0 0, 15 15)'));",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Intersects(l, ST_GeomFromText('LINESTRING(0.5 0.5, 6.5 0.5)'));",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Intersects(poly, ST_GeomFromText('LINESTRING(0 4.5, 7 0.5)'));",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Intersects(mpoly, ST_GeomFromText('LINESTRING(0 4.5, 7 0.5)'));",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                  "WHERE ST_Intersects(l, ST_GeomFromText('POLYGON((0.5 "
                                  "0.5, 6.5 0.5, 3 0.1))'));",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                  "WHERE ST_Intersects(poly, ST_GeomFromText('POLYGON((0 "
                                  "4.5, 7 0.5, 10 10))'));",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(6),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                  "WHERE ST_Intersects(mpoly, "
                                  "ST_GeomFromText('POLYGON((0 4.5, 7 0.5, 10 10))'));",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(l, ST_GeomFromText('MULTIPOLYGON(((0.5 0.5, 6.5 "
                  "0.5, 3 0.1)))'));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(poly, ST_GeomFromText('MULTIPOLYGON(((0 4.5, 7 "
                  "0.5, 10 10)))'));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Intersects(mpoly, ST_GeomFromText('MULTIPOLYGON(((0 4.5, 7 "
                  "0.5, 10 10)))'));",
                  dt)));

    // ST_Disjoint
    ASSERT_EQ(
        static_cast<int64_t>(0),
        v<int64_t>(run_simple_agg(
            "SELECT ST_Disjoint("
            "ST_GeomFromText('POLYGON((2 2, 0 1, -2 2, -2 0, 2 0, 2 2))'), "
            "ST_GeomFromText('LINESTRING(3 3, 3 2, 2 2)')) FROM geospatial_test limit 1;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg("SELECT ST_Disjoint("
                                  "ST_GeomFromText('LINESTRING(3 3, 3 2, 2.1 2.1)'), "
                                  "ST_GeomFromText('MULTIPOLYGON(((5 5, 6 6, 5 6)), ((2 "
                                  "2, 0 1, -2 2, -2 0, 2 0, 2 2)))')) "
                                  " FROM geospatial_test limit 1;",
                                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg("SELECT ST_Disjoint("
                                  "ST_GeomFromText('POLYGON((3 3, 3 2, 2.1 2.1))'), "
                                  "ST_GeomFromText('MULTIPOLYGON(((5 5, 6 6, 5 6)), ((2 "
                                  "2, 0 1, -2 2, -2 0, 2 0, 2 2)))')) "
                                  " FROM geospatial_test limit 1;",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Disjoint(p,p);", dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows - 1),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test "
                                  "WHERE ST_Disjoint(p, ST_GeomFromText('POINT(0 0)'));",
                                  dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows - 6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Disjoint(p, ST_GeomFromText('LINESTRING(0 0, 5 5)'));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Disjoint(p, ST_GeomFromText('LINESTRING(0 0, 15 15)'));",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows - 6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Disjoint(l, ST_GeomFromText('LINESTRING(0.5 0.5, 6.5 0.5)'));",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows - 6),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) FROM geospatial_test "
                  "WHERE ST_Disjoint(poly, ST_GeomFromText('LINESTRING(0 4.5, 7 0.5)'));",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(g_num_rows - 6),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test "
            "WHERE ST_Disjoint(mpoly, ST_GeomFromText('LINESTRING(0 4.5, 7 0.5)'));",
            dt)));

    // ST_Contains, ST_Within
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(p,p);", dt)));
    ASSERT_EQ(static_cast<int64_t>(2),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE "
                  "ST_Contains('POINT(0 0)', p) OR ST_Contains('POINT(1 1)', p);",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains('POINT(0 0)', p);",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(gp4326none, "
                  "ST_GeomFromText('POINT(1 1)', 4326));",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains('POINT(0 0)', l);",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Within('POINT(10.5 10.5)', l);",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(p,l);", dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(l,p);", dt)));

    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Contains(poly, 'POINT(-1 -1)');",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Contains(poly, 'POINT(0.1 0.1)');",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Within('POINT(0.1 0.1)', poly);",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE ST_Contains(gpoly4326, "
                  "ST_GeomFromText('POINT(0.1 0.1)', 4326));",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(1),  // polygon containing a point
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), "
                  "ST_GeomFromText('POINT(0 0)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(
            0),  // same polygon but with a hole in the middle that the point falls into
        v<int64_t>(run_simple_agg(
            "SELECT ST_Contains("
            "'POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0),(1 0, 0 1, -1 0, 0 -1, 1 0))', "
            "'POINT(0.1 0.1)') FROM geospatial_test limit 1;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // back to true if we combine the holed polygon with one
                                  // more in a multipolygon
        v<int64_t>(run_simple_agg(
            "SELECT ST_Contains("
            "'MULTIPOLYGON(((2 0, 0 2, -2 0, 0 -2, 2 0),(1 0, 0 1, -1 0, 0 -1, 1 0)), "
            "((2 0, 0 2, -2 0, 0 -2, 1 -2, 2 -1)))', "
            "'POINT(0.1 0.1)') FROM geospatial_test limit 1;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // last query but for 4326 objects
        v<int64_t>(run_simple_agg(
            "SELECT ST_Contains("
            "ST_GeomFromText('MULTIPOLYGON(((2 0, 0 2, -2 0, 0 -2, 2 0),(1 0, 0 1, -1 0, "
            "0 -1, 1 0)), "
            "((2 0, 0 2, -2 0, 0 -2, 1 -2, 2 -1)))', 4326), "
            "ST_GeomFromText('POINT(0.1 0.1)', 4326)) FROM geospatial_test limit 1;",
            dt)));

    ASSERT_EQ(static_cast<int64_t>(1),  // point in polygon, on left edge
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((0 -1, 2 1, 0 1))'), "
                  "ST_GeomFromText('POINT(0 0)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(1),  // point in polygon, on right edge
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((0 -1, 2 1, 0 1))'), "
                  "ST_GeomFromText('POINT(1 0)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(1),  // point in polygon, touch+leave
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 5 2, 0 2, -1 0))'), "
                  "ST_GeomFromText('POINT(0 0)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(1),  // point in polygon, touch+overlay+leave
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 4 0, 5 2, 0 2, -1 0))'), "
                  "ST_GeomFromText('POINT(0 0)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(1),  // point in polygon, touch+cross
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 4 -1, 5 2, 0 2, -1 0))'), "
                  "ST_GeomFromText('POINT(0 0)')) FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // point in polygon, touch+overlay+cross
        v<int64_t>(run_simple_agg(
            "SELECT ST_Contains("
            "ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 4 0, 4.5 -1, 5 2, 0 2, -1 0))'), "
            "ST_GeomFromText('POINT(0 0)')) FROM geospatial_test limit 1;",
            dt)));
    ASSERT_EQ(static_cast<int64_t>(0),  // point in polygon, check yray redundancy
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((0 -1, 2 1, 3 0, 5 2, 0 2, -1 0))'), "
                  "ST_GeomFromText('POINT(2 0)')) FROM geospatial_test limit 1;",
                  dt)));

    ASSERT_EQ(static_cast<int64_t>(1),  // polygon containing linestring
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), "
                  "ST_GeomFromText('LINESTRING(1 0, 0 1, -1 0, 0 -1, 1 0)')) "
                  "FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),  // polygon containing only a part of linestring
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), "
                  "ST_GeomFromText('LINESTRING(1 0, 0 1, -1 0, 0 -1, 3 0)')) "
                  "FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),    // polygon containing linestring vertices
              v<int64_t>(run_simple_agg(  // but not all of linestring's segments
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((2 2, 0 1, -2 2, -2 0, 2 0, 2 2))'), "
                  "ST_GeomFromText('LINESTRING(1.5 1.5, -1.5 1.5, 0 0.5, 1.5 1.5)')) "
                  "FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(1),  // polygon containing another polygon
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POLYGON((2 0, 0 2, -2 0, 0 -2, 2 0))'), "
                  "ST_GeomFromText('POLYGON((1 0, 0 1, -1 0, 0 -1, 1 0))')) "
                  "FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(1),  // multipolygon containing linestring
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('MULTIPOLYGON(((3 3, 4 3, 4 4)), "
                  "((2 0, 0 2, -2 0, 0 -2, 2 0)))'), "
                  "ST_GeomFromText('LINESTRING(1 0, 0 1, -1 0, 0 -1, 1 0)')) "
                  "FROM geospatial_test limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),    // multipolygon containing linestring vertices
              v<int64_t>(run_simple_agg(  // but not all of linestring's segments
                  "SELECT ST_Contains("
                  "ST_GeomFromText('MULTIPOLYGON(((2 2, 0 1, -2 2, -2 0, 2 0, 2 2)), "
                  "((3 3, 4 3, 4 4)))'), "
                  "ST_GeomFromText('LINESTRING(1.5 1.5, -1.5 1.5, 0 0.5, 1.5 1.5)')) "
                  "FROM geospatial_test limit 1;",
                  dt)));
    // Tolerance
    ASSERT_EQ(static_cast<int64_t>(1),  // point containing an extremely close point
              v<int64_t>(run_simple_agg(
                  "SELECT ST_Contains("
                  "ST_GeomFromText('POINT(2.1100000001 -1.7229999999    )'), "
                  "ST_GeomFromText('POINT(2.11         -1.723)')) FROM geospatial_test "
                  "limit 1;",
                  dt)));
    ASSERT_EQ(static_cast<int64_t>(0),  // point not containing a very close point
              v<int64_t>(run_simple_agg("SELECT ST_Contains("
                                        "ST_GeomFromText('POINT(2.11      -1.723    )'), "
                                        "ST_GeomFromText('POINT(2.1100001 -1.7229999)')) "
                                        "FROM geospatial_test limit 1;",
                                        dt)));
    ASSERT_EQ(
        static_cast<int64_t>(1),  // linestring containing an extremely close point
        v<int64_t>(run_simple_agg(
            "SELECT ST_Contains("
            "ST_GeomFromText('LINESTRING(1 -1.0000000001, 3 -1.0000000001)'), "
            "ST_GeomFromText('POINT(0.999999999 -1)')) FROM geospatial_test limit 1;",
            dt)));

    // ST_DWithin, ST_DFullyWithin
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("SELECT ST_DWithin("
                                        "'POLYGON((4 2, 5 3, 4 3))', "
                                        "'MULTIPOLYGON(((2 2, -2 2, -2 -2, 2 -2, 2 2)), "
                                        "((1 1, -1 1, -1 -1, 1 -1, 1 1)))', "
                                        "3.0) from geospatial_test limit 1;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg("SELECT ST_DFullyWithin("
                                        "'POINT(1 1)', 'LINESTRING (9 0,18 18,19 19)', "
                                        "26.0) AND NOT ST_DFullyWithin("
                                        "'LINESTRING (9 0,18 18,19 19)', 'POINT(1 1)', "
                                        "25.0)  from geospatial_test limit 1;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(7),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_DWithin(l, 'POINT(-1 -1)', 8.0);",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_DFullyWithin(l, 'POINT(-1 -1)', 8.0);",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(5),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_DWithin(poly, 'POINT(5 5)', 3.0);",
                                        dt)));
    // Check if Paris and LA are within a 10000km geodesic distance
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_DWithin(ST_GeogFromText('POINT(-118.4079 33.9434)', 4326), "
                  "ST_GeogFromText('POINT(2.5559 49.0083)', 4326), 10000000.0) "
                  "from geospatial_test limit 1;",
                  dt)));
    // TODO: ST_DWithin support for geographic paths, needs geodesic
    // ST_Distance(linestring)

    // Coord accessors
    ASSERT_NEAR(static_cast<double>(-118.4079),
                v<double>(run_simple_agg("SELECT ST_X('POINT(-118.4079 33.9434)') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.0));
    ASSERT_NEAR(static_cast<double>(33.9434),
                v<double>(run_simple_agg(
                    "SELECT ST_Y(ST_GeomFromText('POINT(-118.4079 33.9434)', 4326)) "
                    "from geospatial_test limit 1;",
                    dt)),
                static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(4021204.558),
                v<double>(run_simple_agg(
                    "SELECT ST_Y(ST_Transform("
                    "ST_GeomFromText('POINT(-118.4079 33.9434)', 4326), 900913)) "
                    "from geospatial_test limit 1;",
                    dt)),
                static_cast<double>(0.01));

    ASSERT_NEAR(static_cast<double>(-118.4079),
                v<double>(run_simple_agg("SELECT ST_XMax('POINT(-118.4079 33.9434)') "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(0.0));
    ASSERT_NEAR(
        static_cast<double>(3960189.382),
        v<double>(run_simple_agg(
            "SELECT ST_YMax('MULTIPOLYGON "
            "(((-13201820.2402333 3957482.147359,-13189665.9329505 3960189.38265416,"
            "-13176924.0813953 3949756.56479131)))') "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(static_cast<double>(4021204.558),
                v<double>(run_simple_agg(
                    "SELECT ST_YMin(ST_Transform(ST_GeomFromText("
                    "'LINESTRING (-118.4079 33.9434, 2.5559 49.0083)', 4326), 900913)) "
                    "from geospatial_test limit 1;",
                    dt)),
                static_cast<double>(0.01));

    SKIP_ON_AGGREGATOR(
        ASSERT_NEAR(static_cast<double>(5),
                    v<double>(run_simple_agg(
                        "SELECT ST_XMax(p) from geospatial_test limit 1 offset 5;", dt)),
                    static_cast<double>(0.0)));
    SKIP_ON_AGGREGATOR(ASSERT_NEAR(
        static_cast<double>(1.0),
        v<double>(run_simple_agg(
            "SELECT ST_YMin(gp4326) from geospatial_test limit 1 offset 1;", dt)),
        static_cast<double>(0.001)));
    SKIP_ON_AGGREGATOR(
        ASSERT_NEAR(static_cast<double>(2 * 7 + 1),
                    v<double>(run_simple_agg(
                        "SELECT ST_XMax(l) from geospatial_test limit 1 offset 7;", dt)),
                    static_cast<double>(0.0)));
    SKIP_ON_AGGREGATOR(ASSERT_NEAR(
        static_cast<double>(2 + 1),
        v<double>(run_simple_agg(
            "SELECT ST_YMax(mpoly) from geospatial_test limit 1 offset 2;", dt)),
        static_cast<double>(0.0)));

    // Point accessors, Linestring indexing
    ASSERT_NEAR(
        static_cast<double>(34.274647),
        v<double>(run_simple_agg(
            "SELECT ST_Y(ST_PointN(ST_GeomFromText('LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326), 2)) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(-122.446747),
        v<double>(run_simple_agg(
            "SELECT ST_X(ST_EndPoint(ST_GeomFromText('LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326))) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(
            557637.370),  // geodesic distance between first and end points: LA - SF trip
        v<double>(run_simple_agg(
            "SELECT ST_Distance(ST_PointN(ST_GeogFromText("
            "'LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326), 1), "
            "ST_EndPoint(ST_GeogFromText("
            "'LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326))) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(
            5.587),  // cartesian distance in degrees, same points: LA - SF trip
        v<double>(run_simple_agg(
            "SELECT ST_Distance(ST_PointN(ST_GeomFromText("
            "'LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326), 1), "
            "ST_EndPoint(ST_GeomFromText("
            "'LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326))) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    ASSERT_NEAR(
        static_cast<double>(689217.783),  // cartesian distance between merc-transformed
                                          // first and end points
        v<double>(run_simple_agg(
            "SELECT ST_Distance(ST_StartPoint(ST_Transform(ST_GeomFromText("
            "'LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326), 900913)), "
            "ST_EndPoint(ST_Transform(ST_GeomFromText("
            "'LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326), 900913))) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));
    // Linestring: check that runaway indices are controlled
    ASSERT_NEAR(
        static_cast<double>(-122.446747),  // stop at endpoint
        v<double>(run_simple_agg(
            "SELECT ST_X(ST_PointN(ST_GeomFromText("
            "'LINESTRING(-118.243683 34.052235, "
            "-119.229034 34.274647, -119.698189 34.420830, -121.898460 36.603954, "
            "-122.446747 37.733795)', 4326), 1000000)) "
            "from geospatial_test limit 1;",
            dt)),
        static_cast<double>(0.01));

    // Point geometries, literals in different spatial references, transforms
    ASSERT_EQ(static_cast<int64_t>(g_num_rows),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Distance('POINT(0 0)', gp) < 100.0;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(4),
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM geospatial_test WHERE "
                                        "ST_Distance(ST_GeogFromText('POINT(0 0)', "
                                        "4326), CastToGeography(gp4326)) < 500000.0;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(4),
              v<int64_t>(run_simple_agg(
                  "SELECT COUNT(*) FROM geospatial_test WHERE "
                  "ST_Distance(ST_GeomFromText('POINT(0 0)', 900913), gp900913) < 5.0;",
                  dt)));
    ASSERT_EQ(
        static_cast<int64_t>(4),
        v<int64_t>(run_simple_agg(
            "SELECT COUNT(*) FROM geospatial_test WHERE "
            "ST_Distance(ST_Transform(ST_GeomFromText('POINT(0 0)', 4326), 900913), "
            "ST_Transform(gp4326, 900913)) < 500000.0;",
            dt)));

    // Test some exceptions
    // Point coord accessor used on a non-POINT, in this case unindexed LINESTRING
    // (missing ST_POINTN)
    EXPECT_THROW(run_simple_agg(
                     "SELECT ST_Y(ST_GeogFromText("
                     "'LINESTRING(-118.243683 34.052235, -119.229034 34.274647)', 4326)) "
                     "from geospatial_test limit 1;",
                     dt),
                 std::runtime_error);
    // Two accessors in a row
    EXPECT_THROW(
        run_simple_agg(
            "SELECT ST_X(ST_Y(ST_GeogFromText('POINT(-118.243683 34.052235)', 4326))) "
            "from geospatial_test limit 1;",
            dt),
        std::runtime_error);
    // Coord order reversed, longitude value is out of latitude range
    EXPECT_THROW(run_simple_agg(
                     "SELECT ST_Y(ST_GeogFromText('POINT(34.052235 -118.243683)', 4326)) "
                     "from geospatial_test limit 1;",
                     dt),
                 std::runtime_error);
    // Linestring accessor on a non-LINESTRING
    EXPECT_THROW(
        run_simple_agg("SELECT ST_X(ST_ENDPOINT('POINT(-118.243683 34.052235)')) "
                       "from geospatial_test limit 1;",
                       dt),
        std::runtime_error);

    // ST_NRings
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NRings(poly) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NRings(mpoly) from geospatial_test limit 1;", dt)));

    // ST_NPoints
    ASSERT_EQ(static_cast<int64_t>(1),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_NPoints(p) from geospatial_test LIMIT 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT ST_NPoints(l) FROM geospatial_test ORDER "
                                        "BY ST_NPoints(l) DESC LIMIT 1;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT ST_NPoints(poly) FROM geospatial_test "
                                        "ORDER BY ST_NPoints(l) DESC LIMIT 1;",
                                        dt)));
    ASSERT_EQ(static_cast<int64_t>(3),
              v<int64_t>(run_simple_agg("SELECT ST_NPoints(mpoly) FROM geospatial_test "
                                        "ORDER BY ST_NPoints(l) DESC LIMIT 1;",
                                        dt)));

    // ST_SRID, ST_SetSRID
    ASSERT_EQ(static_cast<int64_t>(0),
              v<int64_t>(
                  run_simple_agg("SELECT ST_SRID(p) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(4326),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_SRID(gp4326) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(900913),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_SRID(gp900913) from geospatial_test limit 1;", dt)));
    ASSERT_EQ(static_cast<int64_t>(4326),
              v<int64_t>(run_simple_agg(
                  "SELECT ST_SRID(ST_GeogFromText('POINT(-118.243683 34.052235)', 4326)) "
                  "from geospatial_test limit 1;",
                  dt)));
    // Geodesic distance between Paris and LA: ~9105km
    ASSERT_NEAR(static_cast<double>(9105643.0),
                v<double>(run_simple_agg("SELECT ST_Distance("
                                         "CastToGeography(ST_SetSRID(ST_GeomFromText('"
                                         "POINT(-118.4079 33.9434)'), 4326)), "
                                         "CastToGeography(ST_SetSRID(ST_GeomFromText('"
                                         "POINT(2.5559 49.0083)'), 4326))) "
                                         "from geospatial_test limit 1;",
                                         dt)),
                static_cast<double>(10000.0));

    // SQLw/out geo support
    EXPECT_THROW(run_multiple_agg("SELECT count(distinct p) FROM geospatial_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT approx_count_distinct(p) FROM geospatial_test;", dt),
        std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT avg(p) FROM geospatial_test;", dt),
                 std::runtime_error);
    EXPECT_THROW(
        run_multiple_agg("SELECT p, count(*) FROM geospatial_test GROUP BY p;", dt),
        std::runtime_error);
  }
}

TEST(Select, GeoSpatial_GeoJoin) {
  SKIP_ALL_ON_AGGREGATOR();  // TODO(adb): if we replicate the poly table during table
                             // creation we should be able to lift this constraint

  // Test loop joins
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    ASSERT_NO_THROW(run_simple_agg(
        "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
        "b ON ST_Contains(b.poly, a.p);",
        dt,
        true,
        false));

    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
            "b ON ST_Contains(b.poly, a.p) WHERE b.id = 2 OFFSET 1;",
            dt,
            true,
            false)));

    const auto trivial_loop_join_state = g_trivial_loop_join_threshold;
    g_trivial_loop_join_threshold = 1;
    ScopeGuard reset_loop_join_state = [&trivial_loop_join_state] {
      g_trivial_loop_join_threshold = trivial_loop_join_state;
    };

    EXPECT_THROW(
        run_multiple_agg(
            "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
            "b ON ST_Contains(b.poly, a.p);",
            dt,
            false),
        std::runtime_error);
  }

  const auto enable_overlaps_hashjoin_state = g_enable_overlaps_hashjoin;
  g_enable_overlaps_hashjoin = true;
  ScopeGuard reset_overlaps_state = [&enable_overlaps_hashjoin_state] {
    g_enable_overlaps_hashjoin = enable_overlaps_hashjoin_state;
  };

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    // Test query rewrite for simple project
    ASSERT_NO_THROW(run_simple_agg(
        "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
        "b ON ST_Contains(b.poly, a.p);",
        dt));

    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
            "b ON ST_Contains(b.poly, a.p) WHERE b.id = 2 OFFSET 1;",
            dt)));
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test a INNER JOIN "
            "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p) WHERE b.id = 4",
            dt)));
    // re-run to test hash join cache (currently CPU only)
    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg(
            "SELECT count(*) FROM geospatial_test a INNER JOIN "
            "geospatial_inner_join_test b ON ST_Contains(b.poly, a.p) WHERE b.id = 4",
            dt)));

    // with compression
    ASSERT_EQ(
        static_cast<int64_t>(1),
        v<int64_t>(run_simple_agg(
            "SELECT a.id FROM geospatial_test a INNER JOIN geospatial_inner_join_test "
            "b ON ST_Contains(b.poly, a.gp4326) WHERE b.id = 2 OFFSET 1;",
            dt)));

    ASSERT_EQ(
        static_cast<int64_t>(3),
        v<int64_t>(run_simple_agg("SELECT count(*) FROM geospatial_test a INNER JOIN "
                                  "geospatial_inner_join_test b ON ST_Contains(b.poly, "
                                  "a.gp4326) WHERE b.id = 4;",
                                  dt)));
  }
}

TEST(Rounding, ROUND) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    {
      // Check no 2nd parameter
      // the cast is required. SQLite seems to only return FLOATs
      // clang-format off
      std::string sqlLite_select = "SELECT CAST(ROUND(s16) AS SMALLINT) AS r_s16, "
          "CAST(ROUND(s32) AS INT) AS r_s32, "
          "CAST(ROUND(s64) AS BIGINT) AS r_s64, "
          "ROUND(f32) AS r_f32, "
          "ROUND(f64) AS r_f64, "
          "ROUND(n64) AS r_n64, "
          "ROUND(d64) AS r_d64 FROM test_rounding ORDER BY f64 ASC";
      // clang-format on

      // clang-format off
      std::string select = "SELECT CAST(ROUND(s16) AS SMALLINT) AS r_s16, "
          "CAST(ROUND(s32) AS INT) AS r_s32, "
          "CAST(ROUND(s64) AS BIGINT) AS r_s64, "
          "ROUND(f32) AS r_f32, "
          "ROUND(f64) AS r_f64, "
          "ROUND(n64) AS r_n64, "
          "ROUND(d64) AS r_d64 FROM test_rounding ORDER BY f64 ASC NULLS FIRST";
      // clang-format on
      c(select, sqlLite_select, dt);
    }

    // Check negative 2nd parameter
    for (int n = -9; n < 0; n++) {
      std::string i = std::to_string(n);
      std::string rounding_base = std::to_string((int)pow(10, std::abs(n))) + ".0";

      // clang-format off
      std::string sqlLite_select = "SELECT CAST(ROUND((s16/"+rounding_base+")) * "+rounding_base+" AS SMALLINT) AS r_s16, "
              "CAST(ROUND((s32/"+rounding_base+")) * "+rounding_base+" AS INT) AS r_s32, "
              "CAST(ROUND((s64/"+rounding_base+")) * "+rounding_base+" AS BIGINT) AS r_s64, "
              "ROUND((f32/"+rounding_base+")) * "+rounding_base+" AS r_f32, "
              "ROUND((f64/"+rounding_base+")) * "+rounding_base+" AS r_f64, "
              "ROUND((n64/"+rounding_base+")) * "+rounding_base+" AS r_n64, "
              "ROUND((d64/"+rounding_base+")) * "+rounding_base+" AS r_d64 FROM test_rounding ORDER BY f64 ASC";
      // clang-format on

      // clang-format off
      std::string select = "SELECT ROUND(s16, "+i+") AS r_s16, "
              "ROUND(s32, "+i+") AS r_s32, "
              "ROUND(s64, "+i+") AS r_s64, "
              "ROUND(f32, "+i+") AS r_f32, "
              "ROUND(f64, "+i+") AS r_f64, "
              "ROUND(n64, "+i+") AS r_n64, "
              "ROUND(d64, "+i+") AS r_d64 FROM test_rounding ORDER BY f64 ASC NULLS FIRST";
      // clang-format on
      c(select, sqlLite_select, dt);
    }

    // check positive 2nd parameter
    for (int n = 0; n < 10; n++) {
      std::string i = std::to_string(n);

      // the cast is required. SQLite seems to only return FLOATs
      // clang-format off
      std::string sqlLite_select = "SELECT CAST(ROUND(s16, "+i+") AS SMALLINT) AS r_s16, "
              "CAST(ROUND(s32, "+i+") AS INT) AS r_s32, "
              "CAST(ROUND(s64, "+i+") AS BIGINT) AS r_s64, "
              "ROUND(f32, "+i+") AS r_f32, "
              "ROUND(f64, "+i+") AS r_f64, "
              "ROUND(n64, "+i+") AS r_n64, "
              "ROUND(d64, "+i+") AS r_d64 FROM test_rounding ORDER BY f64 ASC";
      // clang-format on

      // clang-format off
      std::string select = "SELECT CAST(ROUND(s16, "+i+") AS SMALLINT) AS r_s16, "
              "CAST(ROUND(s32, "+i+") AS INT) AS r_s32, "
              "CAST(ROUND(s64, "+i+") AS BIGINT) AS r_s64, "
              "ROUND(f32, "+i+") AS r_f32, "
              "ROUND(f64, "+i+") AS r_f64, "
              "ROUND(n64, "+i+") AS r_n64, "
              "ROUND(d64, "+i+") AS r_d64 FROM test_rounding ORDER BY f64 ASC NULLS FIRST";
      // clang-format on
      c(select, sqlLite_select, dt);
    }

    // check null 2nd parameter
    // the cast is required. SQLite seems to only return FLOATs
    // clang-format off
    std::string select = "SELECT CAST(ROUND(s16, (SELECT s16 FROM test_rounding WHERE s16 IS NULL)) AS SMALLINT) AS r_s16, "
        "CAST(ROUND(s32, (SELECT s16 FROM test_rounding WHERE s16 IS NULL)) AS INT) AS r_s32, "
        "CAST(ROUND(s64, (SELECT s16 FROM test_rounding WHERE s16 IS NULL)) AS BIGINT) AS r_s64, "
        "ROUND(f32, (SELECT s16 FROM test_rounding WHERE s16 IS NULL)) AS r_f32, "
        "ROUND(f64, (SELECT s16 FROM test_rounding WHERE s16 IS NULL)) AS r_f64, "
        "ROUND(n64, (SELECT s16 FROM test_rounding WHERE s16 IS NULL)) AS r_n64, "
        "ROUND(d64, (SELECT s16 FROM test_rounding WHERE s16 IS NULL)) AS r_d64 FROM test_rounding";
    c(select, dt);
    // clang-format on

    // check that no -0.0 (negative zero) is returned
    TargetValue val_s16 = run_simple_agg(
        "SELECT ROUND(CAST(-1.7 as SMALLINT), -1) as r_val FROM test_rounding WHERE s16 "
        "IS NULL;",
        dt);
    TargetValue val_s32 = run_simple_agg(
        "SELECT ROUND(CAST(-1.7 as INT), -1) as r_val FROM test_rounding WHERE s16 IS "
        "NULL;",
        dt);
    TargetValue val_s64 = run_simple_agg(
        "SELECT ROUND(CAST(-1.7 as BIGINT), -1) as r_val FROM test_rounding WHERE s16 IS "
        "NULL;",
        dt);
    TargetValue val_f32 = run_simple_agg(
        "SELECT ROUND(CAST(-1.7 as FLOAT), -1) as r_val FROM test_rounding WHERE s16 IS "
        "NULL;",
        dt);
    TargetValue val_f64 = run_simple_agg(
        "SELECT ROUND(CAST(-1.7 as DOUBLE), -1) as r_val FROM test_rounding WHERE s16 IS "
        "NULL;",
        dt);
    TargetValue val_n64 = run_simple_agg(
        "SELECT ROUND(CAST(-1.7 as NUMERIC(10,5)), -1) as r_val FROM test_rounding WHERE "
        "s16 IS NULL;",
        dt);
    TargetValue val_d64 = run_simple_agg(
        "SELECT ROUND(CAST(-1.7 as DECIMAL(10,5)), -1) as r_val FROM test_rounding WHERE "
        "s16 IS NULL;",
        dt);

    ASSERT_TRUE(0 == boost::get<int64_t>(boost::get<ScalarTargetValue>(val_s16)));
    ASSERT_TRUE(0 == boost::get<int64_t>(boost::get<ScalarTargetValue>(val_s32)));
    ASSERT_TRUE(0 == boost::get<int64_t>(boost::get<ScalarTargetValue>(val_s64)));

    ASSERT_FLOAT_EQ(0.0f, boost::get<float>(boost::get<ScalarTargetValue>(val_f32)));
    ASSERT_FALSE(std::signbit(boost::get<float>(boost::get<ScalarTargetValue>(val_f32))));

    ASSERT_DOUBLE_EQ(0.0, boost::get<double>(boost::get<ScalarTargetValue>(val_f64)));
    ASSERT_FALSE(
        std::signbit(boost::get<double>(boost::get<ScalarTargetValue>(val_f64))));

    ASSERT_DOUBLE_EQ(0.0, boost::get<double>(boost::get<ScalarTargetValue>(val_n64)));
    ASSERT_FALSE(
        std::signbit(boost::get<double>(boost::get<ScalarTargetValue>(val_f64))));

    ASSERT_DOUBLE_EQ(0.0, boost::get<double>(boost::get<ScalarTargetValue>(val_d64)));
    ASSERT_FALSE(
        std::signbit(boost::get<double>(boost::get<ScalarTargetValue>(val_f64))));
  }
}

TEST(Select, Sample) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ("else",
              boost::get<std::string>(v<NullableString>(run_simple_agg(
                  "SELECT SAMPLE(CASE WHEN x IN (9) THEN str ELSE 'else' END) FROM test;",
                  dt))));
    SKIP_ON_AGGREGATOR({
      const auto rows = run_multiple_agg(
          "SELECT SAMPLE(real_str), COUNT(*) FROM test WHERE x > 8;", dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(2), crt_row.size());
      const auto nullable_str = v<NullableString>(crt_row[0]);
      const auto null_ptr = boost::get<void*>(&nullable_str);
      ASSERT_TRUE(null_ptr && !*null_ptr);
      ASSERT_EQ(0, v<int64_t>(crt_row[1]));
      const auto empty_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(0), empty_row.size());
    });
    SKIP_ON_AGGREGATOR({
      const auto rows = run_multiple_agg(
          "SELECT SAMPLE(real_str), COUNT(*) FROM test WHERE x > 7;", dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(2), crt_row.size());
      const auto nullable_str = v<NullableString>(crt_row[0]);
      const auto str_ptr = boost::get<std::string>(&nullable_str);
      ASSERT_TRUE(str_ptr);
      ASSERT_EQ("real_bar", boost::get<std::string>(*str_ptr));
      ASSERT_EQ(g_num_rows / 2, v<int64_t>(crt_row[1]));
      const auto empty_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(0), empty_row.size());
    });
    {
      const auto rows = run_multiple_agg(
          "SELECT SAMPLE(real_str), COUNT(*) FROM test WHERE x > 7 GROUP BY x;", dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(2), crt_row.size());
      const auto nullable_str = v<NullableString>(crt_row[0]);
      const auto str_ptr = boost::get<std::string>(&nullable_str);
      ASSERT_TRUE(str_ptr);
      ASSERT_EQ("real_bar", boost::get<std::string>(*str_ptr));
      ASSERT_EQ(g_num_rows / 2, v<int64_t>(crt_row[1]));
      const auto empty_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(0), empty_row.size());
    }
    SKIP_ON_AGGREGATOR({
      const auto rows = run_multiple_agg(
          "SELECT SAMPLE(arr_i64), COUNT(*) FROM array_test WHERE x = 8;", dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(2), crt_row.size());
      compare_array(crt_row[0], std::vector<int64_t>{200, 300, 400});
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(crt_row[1]));
      const auto empty_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(0), empty_row.size());
    });
    {
      const auto rows = run_multiple_agg(
          "SELECT SAMPLE(arr_i64), COUNT(*) FROM array_test WHERE x = 8 GROUP BY x;", dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(2), crt_row.size());
      compare_array(crt_row[0], std::vector<int64_t>{200, 300, 400});
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(crt_row[1]));
      const auto empty_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(0), empty_row.size());
    }
    SKIP_ON_AGGREGATOR({
      const auto rows = run_multiple_agg(
          "SELECT SAMPLE(arr3_i64), COUNT(*) FROM array_test WHERE x = 8;", dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(2), crt_row.size());
      compare_array(crt_row[0], std::vector<int64_t>{200, 300, 400});
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(crt_row[1]));
      const auto empty_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(0), empty_row.size());
    });
    {
      const auto rows = run_multiple_agg(
          "SELECT SAMPLE(arr3_i64), COUNT(*) FROM array_test WHERE x = 8 GROUP BY x;",
          dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(2), crt_row.size());
      compare_array(crt_row[0], std::vector<int64_t>{200, 300, 400});
      ASSERT_EQ(static_cast<int64_t>(1), v<int64_t>(crt_row[1]));
      const auto empty_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(0), empty_row.size());
    }
    auto check_sample_rowid = [](const int64_t val) {
      const std::set<int64_t> valid_row_ids{15, 16, 17, 18, 19};
      ASSERT_TRUE(valid_row_ids.find(val) != valid_row_ids.end())
          << "Last sample rowid value " << val << " is invalid";
    };
    SKIP_ON_AGGREGATOR({
      const auto rows = run_multiple_agg(
          "SELECT AVG(d), AVG(f), str, SAMPLE(rowid) FROM test WHERE d > 2.4 GROUP "
          "BY str;",
          dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(4), crt_row.size());
      const auto d = v<double>(crt_row[0]);
      ASSERT_EQ(2.6, d);
      const auto f = v<double>(crt_row[1]);
      ASSERT_EQ(1.3, f);
      const auto nullable_str = v<NullableString>(crt_row[2]);
      const auto str_ptr = boost::get<std::string>(&nullable_str);
      ASSERT_TRUE(str_ptr);
      ASSERT_EQ("baz", boost::get<std::string>(*str_ptr));
      const auto rowid = v<int64_t>(crt_row[3]);
      check_sample_rowid(rowid);
    });
    SKIP_ON_AGGREGATOR({
      const auto rows = run_multiple_agg("SELECT SAMPLE(str) FROM test WHERE x > 8;", dt);
      const auto crt_row = rows->getNextRow(true, true);
      ASSERT_EQ(size_t(1), crt_row.size());
      const auto nullable_str = v<NullableString>(crt_row[0]);
      ASSERT_FALSE(boost::get<void*>(nullable_str));
    });
  }
}

void shard_key_test_runner(const std::string& shard_key_col,
                           const ExecutorDeviceType dt) {
  run_ddl_statement("drop table if exists shard_key_ddl_test;");
  run_ddl_statement(
      "CREATE TABLE shard_key_ddl_test (x INTEGER, y TEXT ENCODING DICT(32), pt "
      "POINT, z DOUBLE, a BIGINT NOT NULL, poly POLYGON, b SMALLINT, SHARD KEY(" +
      shard_key_col + ")) WITH (shard_count = 4)");

  run_multiple_agg(
      "INSERT INTO shard_key_ddl_test VALUES (1, 'foo', 'POINT(1 1)', 1.0, 1, "
      "'POLYGON((0 0, 1 1, 2 2, 3 3))', 1)",
      dt);
  run_multiple_agg(
      "INSERT INTO shard_key_ddl_test VALUES (2, 'bar', 'POINT(2 2)', 2.0, 2, "
      "'POLYGON((0 0, 1 1, 20 20, 3 3))', 2)",
      dt);
  run_multiple_agg(
      "INSERT INTO shard_key_ddl_test VALUES (3, 'hello', 'POINT(3 3)', 3.0, 3, "
      "'POLYGON((0 0, 1 1, 2 2, 30 30))', 3)",
      dt);
}

TEST(Select, ShardKeyDDL) {
  SKIP_ALL_ON_AGGREGATOR();

  for (auto dt : {ExecutorDeviceType::CPU}) {
    // Table creation / single row inserts
    EXPECT_NO_THROW(shard_key_test_runner("x", dt));
    EXPECT_NO_THROW(shard_key_test_runner("y", dt));
    EXPECT_NO_THROW(shard_key_test_runner("a", dt));
    EXPECT_NO_THROW(shard_key_test_runner("b", dt));

    // Unsupported DDL
    EXPECT_THROW(shard_key_test_runner("z", dt), std::runtime_error);
    EXPECT_THROW(shard_key_test_runner("pt", dt), std::runtime_error);
    EXPECT_THROW(shard_key_test_runner("poly", dt), std::runtime_error);

    // Unsupported update
    EXPECT_NO_THROW(shard_key_test_runner("a", dt));
    EXPECT_THROW(run_multiple_agg("UPDATE shard_key_ddl_test SET a = 2;", dt),
                 std::runtime_error);
  }
}

TEST(Create, DaysEncodingDDL) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    EXPECT_NO_THROW(run_ddl_statement("Drop table if exists chelsea;"));
    EXPECT_NO_THROW(run_ddl_statement(
        "create table chelsea(a date, b date encoding fixed(32), c date encoding "
        "fixed(16), d date encoding days(32), e date encoding days(16));"));

    EXPECT_NO_THROW(run_multiple_agg(
        "insert into "
        "chelsea "
        "values('1548712897','1548712897','1548712897','1548712897','1548712897')",
        dt));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values(null,null,null,null,null)", dt));
    EXPECT_NO_THROW(run_multiple_agg("select a,b,c,d,e from chelsea;", dt));

    ASSERT_EQ(int64_t(2),
              v<int64_t>(run_simple_agg("SELECT count(*) from chelsea;", dt)));
    ASSERT_EQ(
        int64_t(1548633600),
        v<int64_t>(run_simple_agg("SELECT d FROM chelsea where d is not null;", dt)));
    ASSERT_EQ(int64_t(1548633600),
              v<int64_t>(run_simple_agg(
                  "SELECT d FROM chelsea where d = DATE '2019-01-28';", dt)));
    ASSERT_EQ(
        int64_t(1548633600),
        v<int64_t>(run_simple_agg("SELECT e FROM chelsea where e is not null;", dt)));
    ASSERT_EQ(int64_t(1548633600),
              v<int64_t>(run_simple_agg(
                  "SELECT e FROM chelsea where e = DATE '2019-01-28';", dt)));

    EXPECT_THROW(
        run_ddl_statement("create table chelsea1(a timestamp encoding days(16))"),
        std::runtime_error);
  }
}

TEST(Select, DatesDaysEncodingTest) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    EXPECT_NO_THROW(run_ddl_statement("Drop table if exists chelsea;"));
    EXPECT_NO_THROW(run_ddl_statement(
        "create table chelsea(a date encoding days(32), b date encoding days(16));"));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values('-31496400','-31496400')", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values('-31536000','-31536000')", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values('-31492800','-31492800')", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values('31579200','31579200')", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values('31536000','31536000')", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values('31575600','31575600')", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values('1969-01-01','1969-01-01')", dt));
    EXPECT_NO_THROW(
        run_multiple_agg("insert into chelsea values('1971-01-01','1971-01-01')", dt));

    ASSERT_EQ(
        int64_t(8),
        v<int64_t>(run_simple_agg("SELECT count(*) from chelsea where a = b;", dt)));
    ASSERT_EQ(
        int64_t(8),
        v<int64_t>(run_simple_agg("SELECT count(*) from chelsea where b = a;", dt)));
    ASSERT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) from chelsea where a = '1969-01-01';", dt)));
    ASSERT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) from chelsea where a = '1971-01-01';", dt)));
    ASSERT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) from chelsea where b = '1969-01-01';", dt)));
    ASSERT_EQ(int64_t(4),
              v<int64_t>(run_simple_agg(
                  "SELECT count(*) from chelsea where b = '1971-01-01';", dt)));
  }
}

namespace {

int create_sharded_join_table(const std::string& table_name,
                              size_t fragment_size,
                              size_t num_rows,
                              const ShardInfo& shard_info,
                              bool with_delete_support = true) {
  std::string columns_definition{"i INTEGER, j INTEGER, s TEXT ENCODING DICT(32)"};

  try {
    std::string drop_ddl{"DROP TABLE IF EXISTS " + table_name + ";"};
    run_ddl_statement(drop_ddl);
    g_sqlite_comparator.query(drop_ddl);

    const auto create_ddl = build_create_table_statement(columns_definition,
                                                         table_name,
                                                         shard_info,
                                                         {},
                                                         fragment_size,
                                                         with_delete_support);
    run_ddl_statement(create_ddl);
    g_sqlite_comparator.query("CREATE TABLE " + table_name + "(i int, j int, s text);");

    const std::vector<std::string> alphabet{"a", "b", "c", "d", "e", "f", "g", "h", "i",
                                            "j", "k", "l", "m", "n", "o", "p", "q", "r",
                                            "s", "t", "u", "v", "w", "x", "y", "z"};
    const auto alphabet_sz = alphabet.size();

    int i = 0;
    int j = num_rows;
    for (size_t x = 0; x < num_rows; x++) {
      const std::string insert_query{"INSERT INTO " + table_name + " VALUES(" +
                                     std::to_string(i) + "," + std::to_string(j) + ",'" +
                                     alphabet[i % alphabet_sz] + "');"};
      LOG(INFO) << insert_query;

      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
      i++;
      j--;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to (re-)create tables for Inner Join sharded test: "
               << e.what();
    return -EEXIST;
  }
  return 0;
}

int create_and_populate_rounding_table() {
  try {
    const std::string drop_test_table{"DROP TABLE IF EXISTS test_rounding;"};
    run_ddl_statement(drop_test_table);
    g_sqlite_comparator.query(drop_test_table);

    const std::string create_test_table{
        "CREATE TABLE test_rounding (s16 SMALLINT, s32 INTEGER, s64 BIGINT, f32 FLOAT, "
        "f64 DOUBLE, n64 NUMERIC(10,5), "
        "d64 DECIMAL(10,5));"};
    run_ddl_statement(create_test_table);
    g_sqlite_comparator.query(create_test_table);

    const std::string inser_positive_test_data{
        "INSERT INTO test_rounding VALUES(3456, 234567, 3456789012, 3456.3456, "
        "34567.23456, 34567.23456, "
        "34567.23456);"};
    run_multiple_agg(inser_positive_test_data, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(inser_positive_test_data);

    const std::string inser_negative_test_data{
        "INSERT INTO test_rounding VALUES(-3456, -234567, -3456789012, -3456.3456, "
        "-34567.23456, -34567.23456, "
        "-34567.23456);"};
    run_multiple_agg(inser_negative_test_data, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(inser_negative_test_data);

    const std::string inser_null_test_data{
        "INSERT INTO test_rounding VALUES(NULL, NULL, NULL, NULL, NULL, NULL, NULL);"};
    run_multiple_agg(inser_null_test_data, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(inser_null_test_data);

  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_rounding'";
    return -EEXIST;
  }
  return 0;
}

int create_and_populate_tables(bool with_delete_support = true) {
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_inner;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{
        "x int not null, y int, xx smallint, str text encoding dict, dt DATE, dt32 DATE "
        "ENCODING FIXED(32), dt16 DATE ENCODING FIXED(16), ts TIMESTAMP"};
    const auto create_test_inner =
        build_create_table_statement(columns_definition,
                                     "test_inner",
                                     {g_shard_count ? "str" : "", g_shard_count},
                                     {},
                                     2,
                                     with_delete_support,
                                     g_aggregator);
    run_ddl_statement(create_test_inner);
    g_sqlite_comparator.query(
        "CREATE TABLE test_inner(x int not null, y int, xx smallint, str text, dt DATE, "
        "dt32 DATE, dt16 DATE, ts DATETIME);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_inner'";
    return -EEXIST;
  }
  {
    const std::string insert_query{
        "INSERT INTO test_inner VALUES(7, 43, 7, 'foo', '1999-09-09', '1999-09-09', "
        "'1999-09-09', '2014-12-13 22:23:15');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  {
    const std::string insert_query{
        "INSERT INTO test_inner VALUES(-9, 72, -9, 'bars', '2014-12-13', '2014-12-13', "
        "'2014-12-13', '1999-09-09 14:15:16');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS vacuum_test_alt;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{"x int not null, y int"};
    const auto create_vacuum_test_alt =
        build_create_table_statement(columns_definition,
                                     "vacuum_test_alt",
                                     {g_shard_count ? "x" : "", g_shard_count},
                                     {},
                                     2,
                                     with_delete_support);
    run_ddl_statement(create_vacuum_test_alt);
    g_sqlite_comparator.query("CREATE TABLE vacuum_test_alt(x int not null, y int );");

  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'vacuum_test_alt'";
    return -EEXIST;
  }
  {
    const std::string insert_query1{"INSERT INTO vacuum_test_alt VALUES(1,10);"};
    const std::string insert_query2{"INSERT INTO vacuum_test_alt VALUES(2,20);"};
    const std::string insert_query3{"INSERT INTO vacuum_test_alt VALUES(3,30);"};
    const std::string insert_query4{"INSERT INTO vacuum_test_alt VALUES(4,40);"};

    run_multiple_agg(insert_query1, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query1);
    run_multiple_agg(insert_query2, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query2);
    run_multiple_agg(insert_query3, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query3);
    run_multiple_agg(insert_query4, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query4);
  }
  try {
#if 0
    const std::string drop_old_test_inner_deleted{"DROP TABLE IF EXISTS test_inner_deleted;"};
    run_ddl_statement(drop_old_test_inner_deleted);
    g_sqlite_comparator.query(drop_old_test_inner_deleted);
    std::string columns_definition{"x int not null, y int, str text encoding dict, deleted boolean"};

    const auto create_test_inner_deleted =
        build_create_table_statement(columns_definition, "test_inner_deleted", {"", 0}, {}, 2, with_delete_support );
    run_ddl_statement(create_test_inner_deleted);
    auto& cat = g_session->getCatalog();
    const auto td = cat.getMetadataForTable("test_inner_deleted");
    CHECK(td);
    const auto cd = cat.getMetadataForColumn(td->tableId, "deleted");
    CHECK(cd);
    cat.setDeletedColumn(td, cd);

    g_sqlite_comparator.query("CREATE TABLE test_inner_deleted(x int not null, y int, str text);");
#endif
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_inner_deleted'";
    return -EEXIST;
  }
  {
#if 0
    const std::string insert_query{"INSERT INTO test_inner_deleted VALUES(7, 43, 'foo', 't');"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
#endif
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_inner_x;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{"x int not null, y int, str text encoding dict"};
    const auto create_test_inner =
        build_create_table_statement(columns_definition,
                                     "test_inner_x",
                                     {g_shard_count ? "x" : "", g_shard_count},
                                     {},
                                     2,
                                     with_delete_support,
                                     g_aggregator);
    run_ddl_statement(create_test_inner);
    g_sqlite_comparator.query(
        "CREATE TABLE test_inner_x(x int not null, y int, str text);");
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
    const auto create_bar = build_create_table_statement(
        columns_definition, "bar", {"", 0}, {}, 2, with_delete_support);
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
    const std::string drop_proj_top{"DROP TABLE IF EXISTS proj_top;"};
    run_ddl_statement(drop_proj_top);
    g_sqlite_comparator.query(drop_proj_top);
    const auto create_proj_top = "CREATE TABLE proj_top(str TEXT ENCODING NONE, x INT);";
    const auto create_proj_top_sqlite = "CREATE TABLE proj_top(str TEXT, x INT);";
    run_ddl_statement(create_proj_top);
    g_sqlite_comparator.query(create_proj_top_sqlite);
    {
      const auto insert_query = "INSERT INTO proj_top VALUES('a', 7);";
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
    {
      const auto insert_query = "INSERT INTO proj_top VALUES('b', 6);";
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
    {
      const auto insert_query = "INSERT INTO proj_top VALUES('c', 5);";
      run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
      g_sqlite_comparator.query(insert_query);
    }
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'proj_top'";
    return -EEXIST;
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
    std::string columns_definition{
        "x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn "
        "float, d double, dn double, str "
        "varchar(10), null_str text encoding dict, fixed_str text encoding dict(16), "
        "fixed_null_str text encoding "
        "dict(16), real_str text encoding none, shared_dict text, m timestamp(0), m_3 "
        "timestamp(3), m_6 timestamp(6), "
        "m_9 timestamp(9), n time(0), o date, o1 date encoding fixed(16), o2 date "
        "encoding fixed(32), fx int "
        "encoding fixed(16), dd decimal(10, 2), dd_notnull decimal(10, 2) not null, ss "
        "text encoding dict, u int, ofd "
        "int, ufd int not null, ofq bigint, ufq bigint not null, smallint_nulls "
        "smallint"};
#else
    std::string columns_definition{
        "x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn "
        "float, d double, dn double, str "
        "varchar(10), null_str text encoding dict, fixed_str text encoding dict(16), "
        "fixed_null_str text encoding "
        "dict(16), real_str text encoding none, shared_dict text, m timestamp(0), "
        "n time(0), o date, o1 date encoding fixed(16), o2 date "
        "encoding fixed(32), fx int "
        "encoding fixed(16), dd decimal(10, 2), dd_notnull decimal(10, 2) not null, ss "
        "text encoding dict, u int, ofd "
        "int, ufd int not null, ofq bigint, ufq bigint not null, smallint_nulls "
        "smallint"};
#endif
    const std::string create_test = build_create_table_statement(
        columns_definition,
        "test",
        {g_shard_count ? "str" : "", g_shard_count},
        {{"str", "test_inner", "str"}, {"shared_dict", "test", "str"}},
        2,
        with_delete_support);
    run_ddl_statement(create_test);
#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
    g_sqlite_comparator.query(
        "CREATE TABLE test(x int not null, y int, z smallint, t bigint, b boolean, f "
        "float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, "
        "fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), m_3 timestamp(3), m_6 timestamp(6), m_9 timestamp(9), n "
        "time(0), o date, o1 date, o2 date, "
        "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null, "
        "smallint_nulls smallint);");
#else
    g_sqlite_comparator.query(
        "CREATE TABLE test(x int not null, y int, z smallint, t bigint, b boolean, f "
        "float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, "
        "fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), n "
        "time(0), o date, o1 date, o2 date, "
        "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null, "
        "smallint_nulls smallint);");
#endif
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test'";
    return -EEXIST;
  }
  CHECK_EQ(g_num_rows % 2, 0);
  for (ssize_t i = 0; i < g_num_rows; ++i) {
#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, "
        "'foo', null, 'foo', null, "
        "'real_foo', 'foo',"
        "'2014-12-13 22:23:15', '2014-12-13 22:23:15.323', '1999-07-11 14:02:53.874533', "
        "'2006-04-26 "
        "03:49:04.607435125', "
        "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, 'fish', "
        "null, "
        "2147483647, -2147483648, null, -1, 32767);"};
#else
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, "
        "'foo', null, 'foo', null, "
        "'real_foo', 'foo',"
        "'2014-12-13 22:23:15', "
        "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, 'fish', "
        "null, "
        "2147483647, -2147483648, null, -1, 32767);"};
#endif
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
    const std::string insert_query{
        "INSERT INTO test VALUES(8, 43, -78, 1002, 'f', 1.2, 101.2, -101.2, 2.4, "
        "-2002.4, 'bar', null, 'bar', null, "
        "'real_bar', NULL, '2014-12-13 22:23:15', '2014-12-13 22:23:15.323', '2014-12-13 "
        "22:23:15.874533', "
        "'2014-12-13 22:23:15.607435763', '15:13:14', NULL, NULL, NULL, NULL, 222.2, "
        "222.2, "
        "null, null, null, "
        "-2147483647, "
        "9223372036854775807, -9223372036854775808, null);"};
#else
    const std::string insert_query{
        "INSERT INTO test VALUES(8, 43, -78, 1002, 'f', 1.2, 101.2, -101.2, 2.4, "
        "-2002.4, 'bar', null, 'bar', null, "
        "'real_bar', NULL, '2014-12-13 22:23:15', "
        "'15:13:14', NULL, NULL, NULL, NULL, 222.2, 222.2, "
        "null, null, null, "
        "-2147483647, "
        "9223372036854775807, -9223372036854775808, null);"};
#endif
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
#ifndef DISABLE_HIGH_PRECISION_TIMESTAMP
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 43, 102, 1002, 't', 1.3, 1000.3, -1000.3, 2.6, "
        "-220.6, 'baz', null, null, null, "
        "'real_baz', 'baz', '2014-12-14 22:23:15', '2014-12-14 22:23:15.750', "
        "'2014-12-14 22:23:15.437321', "
        "'2014-12-14 22:23:15.934567401', '15:13:14', '1999-09-09', '1999-09-09', "
        "'1999-09-09', 11, "
        "333.3, 333.3, "
        "'boat', null, 1, "
        "-1, 1, -9223372036854775808, 1);"};
#else
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 43, 102, 1002, 't', 1.3, 1000.3, -1000.3, 2.6, "
        "-220.6, 'baz', null, null, null, "
        "'real_baz', 'baz', '2014-12-14 22:23:15', "
        "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 11, "
        "333.3, 333.3, "
        "'boat', null, 1, "
        "-1, 1, -9223372036854775808, 1);"};
#endif
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_empty;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{
        "x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn "
        "float, d double, dn double, str "
        "varchar(10), null_str text encoding dict, fixed_str text encoding dict(16), "
        "fixed_null_str text encoding "
        "dict(16), real_str text encoding none, shared_dict text, m timestamp(0), "
        "n time(0), o date, o1 date encoding fixed(16), o2 date "
        "encoding fixed(32), fx int "
        "encoding fixed(16), dd decimal(10, 2), dd_notnull decimal(10, 2) not null, ss "
        "text encoding dict, u int, ofd "
        "int, ufd int not null, ofq bigint, ufq bigint not null"};
    const std::string create_test = build_create_table_statement(
        columns_definition,
        "test_empty",
        {g_shard_count ? "str" : "", g_shard_count},
        {{"str", "test_inner", "str"}, {"shared_dict", "test", "str"}},
        2,
        with_delete_support);
    run_ddl_statement(create_test);
    g_sqlite_comparator.query(
        "CREATE TABLE test_empty(x int not null, y int, z smallint, t bigint, b boolean, "
        "f "
        "float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, "
        "fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), n "
        "time(0), o date, o1 date, o2 date, "
        "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_empty'";
    return -EEXIST;
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_one_row;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{
        "x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn "
        "float, d double, dn double, str "
        "varchar(10), null_str text encoding dict, fixed_str text encoding dict(16), "
        "fixed_null_str text encoding "
        "dict(16), real_str text encoding none, shared_dict text, m timestamp(0), "
        "n time(0), o date, o1 date encoding fixed(16), o2 date "
        "encoding fixed(32), fx int "
        "encoding fixed(16), dd decimal(10, 2), dd_notnull decimal(10, 2) not null, ss "
        "text encoding dict, u int, ofd "
        "int, ufd int not null, ofq bigint, ufq bigint not null"};
    const std::string create_test = build_create_table_statement(
        columns_definition,
        "test_one_row",
        {g_shard_count ? "str" : "", g_shard_count},
        {{"str", "test_inner", "str"}, {"shared_dict", "test", "str"}},
        2,
        with_delete_support);
    run_ddl_statement(create_test);
    g_sqlite_comparator.query(
        "CREATE TABLE test_one_row(x int not null, y int, z smallint, t bigint, b "
        "boolean, "
        "f "
        "float, ff float, fn float, d "
        "double, dn double, str varchar(10), null_str text, fixed_str text, "
        "fixed_null_str text, real_str text, "
        "shared_dict "
        "text, m timestamp(0), n "
        "time(0), o date, o1 date, o2 date, "
        "fx int, dd decimal(10, 2), dd_notnull decimal(10, 2) not "
        "null, ss "
        "text, u int, ofd int, ufd int not null, ofq bigint, ufq bigint not null);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_one_row'";
    return -EEXIST;
  }
  {
    const std::string insert_query{
        "INSERT INTO test_one_row VALUES(8, 43, -78, 1002, 'f', 1.2, 101.2, -101.2, 2.4, "
        "-2002.4, 'bar', null, 'bar', null, "
        "'real_bar', NULL, '2014-12-13 22:23:15', "
        "'15:13:14', NULL, NULL, NULL, NULL, 222.2, 222.2, "
        "null, null, null, "
        "-2147483647, "
        "9223372036854775807, -9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test_x;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    std::string columns_definition{
        "x int not null, y int, z smallint, t bigint, b boolean, f float, ff float, fn "
        "float, d double, dn double, str "
        "text, null_str text encoding dict, fixed_str text encoding dict(16), real_str "
        "text encoding none, m "
        "timestamp(0), n time(0), o date, o1 date encoding fixed(16), "
        "o2 date encoding fixed(32), fx int encoding fixed(16), dd decimal(10, 2), "
        "dd_notnull decimal(10, 2) not null, ss text encoding dict, u int, ofd int, ufd "
        "int not null, ofq bigint, ufq "
        "bigint not null"};
    const std::string create_test =
        build_create_table_statement(columns_definition,
                                     "test_x",
                                     {g_shard_count ? "x" : "", g_shard_count},
                                     {},
                                     2,
                                     with_delete_support);
    run_ddl_statement(create_test);
    g_sqlite_comparator.query(
        "CREATE TABLE test_x(x int not null, y int, z smallint, t bigint, b boolean, f "
        "float, ff float, fn float, d "
        "double, dn double, str "
        "text, null_str text,"
        "fixed_str text, real_str text, m timestamp(0), n time(0), o date, o1 date, "
        "o2 date, fx int, dd decimal(10, 2), "
        "dd_notnull decimal(10, 2) not null, ss text, u int, ofd int, ufd int not null, "
        "ofq bigint, ufq bigint not "
        "null);");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test_x'";
    return -EEXIST;
  }
  CHECK_EQ(g_num_rows % 2, 0);
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    const std::string insert_query{
        "INSERT INTO test_x VALUES(7, 42, 101, 1001, 't', 1.1, 1.1, null, 2.2, null, "
        "'foo', null, 'foo', 'real_foo', "
        "'2014-12-13 "
        "22:23:15', "
        "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 9, 111.1, 111.1, 'fish', "
        "null, "
        "2147483647, -2147483648, null, -1);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test_x VALUES(8, 43, 102, 1002, 'f', 1.2, 101.2, -101.2, 2.4, "
        "-2002.4, 'bar', null, 'bar', "
        "'real_bar', "
        "'2014-12-13 "
        "22:23:15', "
        "'15:13:14', NULL, NULL, NULL, NULL, 222.2, 222.2, null, null, null, "
        "-2147483647, "
        "9223372036854775807, "
        "-9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test_x VALUES(7, 43, 102, 1002, 't', 1.3, 1000.3, -1000.3, 2.6, "
        "-220.6, 'baz', null, 'baz', "
        "'real_baz', "
        "'2014-12-13 "
        "22:23:15', "
        "'15:13:14', '1999-09-09', '1999-09-09', '1999-09-09', 11, 333.3, 333.3, 'boat', "
        "null, 1, -1, "
        "1, -9223372036854775808);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_array_test{"DROP TABLE IF EXISTS array_test;"};
    run_ddl_statement(drop_old_array_test);
    const std::string create_array_test{
        "CREATE TABLE array_test(x int not null, arr_i16 smallint[], arr_i32 int[], "
        "arr_i64 bigint[], arr_str text[] "
        "encoding dict, arr_float float[], arr_double double[], arr_bool boolean[], "
        "real_str text encoding none, "
        "arr3_i8 tinyint[3], arr3_i16 smallint[3], arr3_i32 int[3], arr3_i64 bigint[3], "
        "arr3_float float[3], "
        "arr3_double double[3], arr6_bool boolean[6]);"};
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
        "CREATE TABLE array_test_inner(x int, arr_i16 smallint[], arr_i32 int[], arr_i64 "
        "bigint[], arr_str text[] "
        "encoding "
        "dict, "
        "arr_float float[], arr_double double[], arr_bool boolean[], real_str text "
        "encoding none) WITH "
        "(fragment_size=4000000, partitions='REPLICATED');"};
    run_ddl_statement(create_array_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'array_test_inner'";
    return -EEXIST;
  }
  import_array_test("array_test_inner");
  if (!g_aggregator) {
    try {
      size_t num_shards = choose_shard_count();
      // check if the oversubscriptions to GPU for multiple Shard is correctly functional
      // or not.
      const size_t num_oversubscription = 10;

      ShardInfo shard_info{(num_shards) ? "i" : "", num_shards};
      size_t fragment_size = 2;
      bool delete_support = false;

      create_sharded_join_table("st1",
                                fragment_size,
                                num_oversubscription * num_shards,
                                shard_info,
                                delete_support);
      create_sharded_join_table(
          "st2", fragment_size, num_shards * fragment_size, shard_info, delete_support);
      create_sharded_join_table(
          "st3", fragment_size, 8 * num_shards, shard_info, delete_support);

    } catch (...) {
      LOG(ERROR) << "Failed to (re-)create table 'array_test_inner'";
      return -EEXIST;
    }
  }
  try {
    const std::string drop_old_single_row_test{"DROP TABLE IF EXISTS single_row_test;"};
    run_ddl_statement(drop_old_single_row_test);
    g_sqlite_comparator.query(drop_old_single_row_test);

    if (g_aggregator) {
      run_ddl_statement(
          "CREATE TABLE single_row_test(x int) WITH (PARTITIONS='REPLICATED');");
    } else {
      run_ddl_statement("CREATE TABLE single_row_test(x int);");
    }

    g_sqlite_comparator.query("CREATE TABLE single_row_test(x int);");
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
    import_decimal_compression_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'decimal_compression_test";
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
    import_join_test(with_delete_support);
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
    import_coalesce_cols_join_test(0, with_delete_support);
    import_coalesce_cols_join_test(1, with_delete_support);
    import_coalesce_cols_join_test(2, with_delete_support);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table for coalesce col join test "
                  "('coalesce_cols_join_0', "
                  "'coalesce_cols_join_1', 'coalesce_cols_join_2')";
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
    import_geospatial_test();
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'geospatial_test'";
    return -EEXIST;
  }
  try {
    import_geospatial_join_test(g_aggregator);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'geospatial_inner_join_test'";
  }
  try {
    const std::string drop_old_empty{"DROP TABLE IF EXISTS emptytab;"};
    run_ddl_statement(drop_old_empty);
    g_sqlite_comparator.query(drop_old_empty);
    const std::string create_empty{
        "CREATE TABLE emptytab(x int not null, y int, t bigint not null, f float not "
        "null, d double not null, dd "
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
    const std::string create_test_in_bitmap{
        "CREATE TABLE test_in_bitmap(str TEXT ENCODING DICT);"};
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

  int rc = create_and_populate_rounding_table();
  if (rc) {
    return rc;
  }

  return 0;
}

int create_views() {
  const std::string create_view_test{
      "CREATE VIEW view_test AS SELECT test.*, test_inner.* FROM test, test_inner WHERE "
      "test.str = test_inner.str;"};
  const std::string drop_old_view{"DROP VIEW IF EXISTS view_test;"};
  const std::string create_join_view_test{
      "CREATE VIEW join_view_test AS SELECT a.x AS x FROM test a JOIN test_inner b ON "
      "a.str = b.str;"};
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
        "CREATE TABLE ctas_test AS SELECT x, f, d, str, fixed_str FROM test WHERE x > "
        "7;"};
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
        "CREATE TABLE empty_ctas_test AS SELECT x, f, d, str, fixed_str FROM test WHERE "
        "x > 8;"};
    run_ddl_statement(create_ctas_test);
    g_sqlite_comparator.query(create_ctas_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'empty_ctas_test'";
    return -EEXIST;
  }
  return 0;
}

void drop_tables() {
  const std::string drop_vacuum_test_alt("DROP TABLE vacuum_test_alt;");
  g_sqlite_comparator.query(drop_vacuum_test_alt);
  const std::string drop_test_inner{"DROP TABLE test_inner;"};
  run_ddl_statement(drop_test_inner);
  g_sqlite_comparator.query(drop_test_inner);
  const std::string drop_test{"DROP TABLE test;"};
  run_ddl_statement(drop_test);
  g_sqlite_comparator.query(drop_test);
  const std::string drop_test_empty{"DROP TABLE test_empty;"};
  run_ddl_statement(drop_test_empty);
  g_sqlite_comparator.query(drop_test_empty);
  const std::string test_one_row{"DROP TABLE test_one_row;"};
  run_ddl_statement(test_one_row);
  g_sqlite_comparator.query(test_one_row);
  const std::string drop_test_inner_x{"DROP TABLE test_inner_x;"};
  run_ddl_statement(drop_test_inner_x);
  g_sqlite_comparator.query(drop_test_inner_x);
#if 0
  const std::string drop_test_inner_deleted{"DROP TABLE test_inner_deleted;"};
  run_ddl_statement(drop_test_inner_deleted);
  g_sqlite_comparator.query(drop_test_inner_deleted);
#endif
  const std::string drop_bar{"DROP TABLE bar;"};
  run_ddl_statement(drop_bar);
  g_sqlite_comparator.query(drop_bar);
  const std::string drop_proj_top{"DROP TABLE proj_top;"};
  run_ddl_statement(drop_proj_top);
  g_sqlite_comparator.query(drop_proj_top);
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
  const std::string drop_decimal_compression_test{"DROP TABLE decimal_compression_test;"};
  run_ddl_statement(drop_decimal_compression_test);
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
  run_ddl_statement(drop_join_test);
  g_sqlite_comparator.query(drop_join_test);
  const std::string drop_hash_join_test{"DROP TABLE hash_join_test;"};
  run_ddl_statement(drop_hash_join_test);
  g_sqlite_comparator.query(drop_hash_join_test);
  const std::string drop_coalesce_join_test_0{"DROP TABLE coalesce_cols_test_0"};
  run_ddl_statement(drop_coalesce_join_test_0);
  g_sqlite_comparator.query(drop_coalesce_join_test_0);
  const std::string drop_coalesce_join_test_1{"DROP TABLE coalesce_cols_test_1"};
  run_ddl_statement(drop_coalesce_join_test_1);
  g_sqlite_comparator.query(drop_coalesce_join_test_1);
  const std::string drop_coalesce_join_test_2{"DROP TABLE coalesce_cols_test_2"};
  run_ddl_statement(drop_coalesce_join_test_2);
  g_sqlite_comparator.query(drop_coalesce_join_test_2);
  const std::string drop_emp_table{"DROP TABLE emp;"};
  g_sqlite_comparator.query(drop_emp_table);
  run_ddl_statement(drop_emp_table);
  const std::string drop_dept_table{"DROP TABLE dept;"};
  g_sqlite_comparator.query(drop_dept_table);
  run_ddl_statement(drop_dept_table);
  run_ddl_statement("DROP TABLE geospatial_test;");
  const std::string drop_test_in_bitmap{"DROP TABLE test_in_bitmap;"};
  g_sqlite_comparator.query(drop_test_in_bitmap);
  run_ddl_statement(drop_test_in_bitmap);

  if (!g_aggregator) {
    const std::string drop_ctas_test{"DROP TABLE ctas_test;"};
    g_sqlite_comparator.query(drop_ctas_test);
    run_ddl_statement(drop_ctas_test);

    const std::string drop_empty_ctas_test{"DROP TABLE empty_ctas_test;"};
    g_sqlite_comparator.query(drop_empty_ctas_test);
    run_ddl_statement(drop_empty_ctas_test);
  }

  const std::string drop_test_table_rounding{"DROP TABLE test_rounding;"};
  run_ddl_statement(drop_test_table_rounding);
  g_sqlite_comparator.query(drop_test_table_rounding);
}

void drop_views() {
  const std::string drop_view_test{"DROP VIEW view_test;"};
  run_ddl_statement(drop_view_test);
  g_sqlite_comparator.query(drop_view_test);
  const std::string drop_join_view_test{"DROP VIEW join_view_test;"};
  run_ddl_statement(drop_join_view_test);
  g_sqlite_comparator.query(drop_join_view_test);
}

}  // namespace

LeafHostInfo to_leaf_host_info(std::string& server_info, NodeRole role) {
  size_t pos = server_info.find(':');
  if (pos == std::string::npos) {
    throw std::runtime_error("Invalid host:port -> " + server_info);
  }

  auto host = server_info.substr(0, pos);
  auto port = server_info.substr(pos + 1);

  return LeafHostInfo(host, std::stoi(port), role);
}

std::vector<LeafHostInfo> to_leaf_host_info(std::vector<std::string>& server_infos,
                                            NodeRole role) {
  std::vector<LeafHostInfo> host_infos;

  for (auto& server_info : server_infos) {
    host_infos.push_back(to_leaf_host_info(server_info, role));
  }

  return host_infos;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all test");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()("disable-literal-hoisting", "Disable literal hoisting");
  desc.add_options()("with-sharding", "Create sharded tables");
  desc.add_options()("from-table-reordering",
                     po::value<bool>(&g_from_table_reordering)
                         ->default_value(g_from_table_reordering)
                         ->implicit_value(true),
                     "Enable automatic table reordering in FROM clause");
  desc.add_options()("bigint-count",
                     po::value<bool>(&g_bigint_count)
                         ->default_value(g_bigint_count)
                         ->implicit_value(false),
                     "Use 64-bit count");
  desc.add_options()("disable-shared-mem-group-by",
                     po::value<bool>(&g_enable_smem_group_by)
                         ->default_value(g_enable_smem_group_by)
                         ->implicit_value(false),
                     "Enable/disable using GPU shared memory for GROUP BY.");
  desc.add_options()("enable-columnar-output",
                     po::value<bool>(&g_enable_columnar_output)
                         ->default_value(g_enable_columnar_output)
                         ->implicit_value(true),
                     "Enable/disable using columnar output format.");
  desc.add_options()("keep-data", "Don't drop tables at the end of the tests");
  desc.add_options()(
      "use-existing-data",
      "Don't create and drop tables and only run select tests (it implies --keep-data).");
  desc.add_options()("dump-ir",
                     po::value<std::string>(),
                     "Dump IR for all executed queries to file. Currently only supports "
                     "single node tests.");
  desc.add_options()(
      "test-help",
      "Print all ExecuteTest specific options (for gtest options use `--help`).");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: ExecuteTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  if (vm.count("disable-literal-hoisting")) {
    g_hoist_literals = false;
  }

  g_session.reset(QueryRunner::get_session(BASE_PATH));

  if (vm.count("with-sharding")) {
    g_shard_count = choose_shard_count();
  }
  if (vm.count("dump-ir")) {
    const auto filename = vm["dump-ir"].as<std::string>();
    g_ir_output_file = std::make_unique<QueryRunner::IROutputFile>(filename);
  }

  if (vm.count("keep-data")) {
    g_keep_test_data = true;
  }

  // insert artificial gap of columnId so as to test against the gap w/o
  // need of ALTER ADD/DROP COLUMN before doing query test.
  // Note: Temporarily disabling for distributed tests.
  g_test_against_columnId_gap = g_aggregator ? 0 : 99;

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
      SKIP_ON_AGGREGATOR(err = create_as_select());
    }
    if (!err) {
      SKIP_ON_AGGREGATOR(err = create_as_select_empty());
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
  if (!use_existing_data && !g_keep_test_data) {
    drop_tables();
    drop_views();
  }
  g_session.reset(nullptr);
  return err;
}
