#include "QueryRunner.h"

#include "../Parser/parser.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "../Import/Importer.h"

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <gtest/gtest.h>

using namespace std;

namespace {

std::unique_ptr<Catalog_Namespace::SessionInfo> g_session(get_session("/tmp"));
NVVMBackend g_nvvm_backend{NVVMBackend::CUDA};

ResultRows run_multiple_agg(const string& query_str, const ExecutorDeviceType device_type) {
  return run_multiple_agg(query_str, g_session, device_type, g_nvvm_backend);
}

TargetValue run_simple_agg(const string& query_str, const ExecutorDeviceType device_type) {
  auto rows = run_multiple_agg(query_str, device_type);
  auto crt_row = rows.getNextRow(true, true);
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
  list<Parser::Stmt*> parse_trees;
  string last_parsed;
  CHECK_EQ(parser.parse(create_table_stmt, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), 1);
  auto stmt = parse_trees.front();
  unique_ptr<Stmt> stmt_ptr(stmt);  // make sure it's deleted
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl);
  if (ddl != nullptr)
    ddl->execute(*g_session);
}

bool skip_tests(const ExecutorDeviceType device_type) {
  return device_type == ExecutorDeviceType::GPU && !g_session->get_catalog().get_dataMgr().gpusPresent();
}

bool approx_eq(const double v, const double target, const double eps = 0.01) {
  return target - eps < v && v < target + eps;
}

class SQLiteComparator {
 public:
  SQLiteComparator() : connector_("main", "") {}

  void query(const std::string& query_string) { connector_.query(query_string); }

  void compare(const std::string& query_string, const ExecutorDeviceType device_type) {
    connector_.query(query_string);
    const auto mapd_results = run_multiple_agg(query_string, device_type);
    ASSERT_EQ(connector_.getNumRows(), mapd_results.rowCount());
    const int num_rows{static_cast<int>(connector_.getNumRows())};
    if (mapd_results.definitelyHasNoRows()) {
      ASSERT_EQ(0, num_rows);
      return;
    }
    if (!num_rows) {
      return;
    }
    CHECK_EQ(connector_.getNumCols(), mapd_results.colCount());
    const int num_cols{static_cast<int>(connector_.getNumCols())};
    for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
      const auto crt_row = mapd_results.getNextRow(true, true);
      CHECK(!crt_row.empty());
      CHECK_EQ(static_cast<size_t>(num_cols), crt_row.size());
      for (int col_idx = 0; col_idx < num_cols; ++col_idx) {
        const auto ref_col_type = connector_.columnTypes[col_idx];
        const auto mapd_variant = crt_row[col_idx];
        const auto scalar_mapd_variant = boost::get<ScalarTargetValue>(&mapd_variant);
        CHECK(scalar_mapd_variant);
        switch (ref_col_type) {
          case SQLITE_INTEGER: {
            ASSERT_TRUE(mapd_results.getColType(col_idx).is_integer());
            const auto ref_val = connector_.getData<int64_t>(row_idx, col_idx);
            const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_int_p);
            const auto mapd_val = *mapd_as_int_p;
            ASSERT_EQ(ref_val, mapd_val);
            break;
          }
          case SQLITE_FLOAT: {
            ASSERT_TRUE(mapd_results.getColType(col_idx).is_integer() ||
                        mapd_results.getColType(col_idx).is_decimal() || mapd_results.getColType(col_idx).is_fp());
            const auto ref_val = connector_.getData<double>(row_idx, col_idx);
            const auto mapd_as_double_p = boost::get<double>(scalar_mapd_variant);
            ASSERT_NE(nullptr, mapd_as_double_p);
            const auto mapd_val = *mapd_as_double_p;
            ASSERT_TRUE(approx_eq(ref_val, mapd_val));
            break;
          }
          case SQLITE_TEXT: {
            ASSERT_TRUE(mapd_results.getColType(col_idx).is_string() || mapd_results.getColType(col_idx).is_time() ||
                        mapd_results.getColType(col_idx).is_boolean());
            const auto ref_val = connector_.getData<std::string>(row_idx, col_idx);
            if (mapd_results.getColType(col_idx).is_string()) {
              const auto mapd_as_str_p = boost::get<NullableString>(scalar_mapd_variant);
              ASSERT_NE(nullptr, mapd_as_str_p);
              const auto mapd_str_notnull = boost::get<std::string>(mapd_as_str_p);
              CHECK(mapd_str_notnull);
              const auto mapd_val = *mapd_str_notnull;
              ASSERT_EQ(ref_val, mapd_val);
            } else {
              const auto mapd_type = mapd_results.getColType(col_idx).get_type();
              switch (mapd_type) {
                case kTIMESTAMP:
                case kDATE: {
                  struct tm tm_struct {
                    0
                  };
                  const auto end_str =
                      strptime(ref_val.c_str(), mapd_type == kTIMESTAMP ? "%Y-%m-%dT%H%M%S" : "%Y-%m-%d", &tm_struct);
                  if (end_str != nullptr) {
                    ASSERT_EQ(0, *end_str);
                    ASSERT_EQ(ref_val.size(), static_cast<size_t>(end_str - ref_val.c_str()));
                  }
                  const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
                  ASSERT_EQ(*mapd_as_int_p, timegm(&tm_struct));
                  break;
                }
                case kBOOLEAN: {
                  const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
                  if (ref_val == "t") {
                    ASSERT_EQ(1, *mapd_as_int_p);
                  } else {
                    CHECK_EQ("f", ref_val);
                    ASSERT_EQ(0, *mapd_as_int_p);
                  }
                  break;
                }
                case kTIME: {
                  std::vector<std::string> time_tokens;
                  boost::split(time_tokens, ref_val, boost::is_any_of(":"));
                  ASSERT_EQ(size_t(3), time_tokens.size());
                  const auto mapd_as_int_p = boost::get<int64_t>(scalar_mapd_variant);
                  ASSERT_EQ(boost::lexical_cast<int64_t>(time_tokens[0]) * 3600 +
                                boost::lexical_cast<int64_t>(time_tokens[1]) * 60 +
                                boost::lexical_cast<int64_t>(time_tokens[2]),
                            *mapd_as_int_p);
                  break;
                }
                default:
                  CHECK(false);
              }
            }
            break;
          }
          case SQLITE_NULL:
            break;
          default:
            CHECK(false);
        }
      }
    }
  }

 private:
  SqliteConnector connector_;
};

const ssize_t g_num_rows{10};
SQLiteComparator g_sqlite_comparator;

void c(const std::string& query_string, const ExecutorDeviceType device_type) {
  g_sqlite_comparator.compare(query_string, device_type);
}
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }

TEST(Select, FilterAndSimpleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test;", dt);
    c("SELECT MIN(x) FROM test;", dt);
    c("SELECT MAX(x) FROM test;", dt);
    c("SELECT MIN(z) FROM test;", dt);
    c("SELECT MAX(z) FROM test;", dt);
    c("SELECT MIN(t) FROM test;", dt);
    c("SELECT MAX(t) FROM test;", dt);
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
    c("SELECT MIN(dec) FROM test;", dt);
    c("SELECT MAX(dec) FROM test;", dt);
    c("SELECT SUM(dec) FROM test;", dt);
    c("SELECT AVG(dec) FROM test;", dt);
    c("SELECT AVG(dec) FROM test WHERE x > 6 AND x < 8;", dt);
    c("SELECT COUNT(*) FROM test WHERE dec > 100;", dt);
    c("SELECT COUNT(*) FROM test WHERE dec > 200;", dt);
    c("SELECT COUNT(*) FROM test WHERE dec > 300;", dt);
    c("SELECT COUNT(*) FROM test WHERE dec > 111.0;", dt);
    c("SELECT COUNT(*) FROM test WHERE dec > 111.1;", dt);
    c("SELECT COUNT(*) FROM test WHERE dec > 222.2;", dt);
    c("SELECT MAX(x + dec) FROM test;", dt);
    c("SELECT MAX(x + 2 * dec), MIN(x + 2 * dec) FROM test;", dt);
    c("SELECT COUNT(*) FROM test where dec > CAST(111.0 AS decimal(10, 2));", dt);
    c("SELECT COUNT(*) FROM test where dec > CAST(222.0 AS decimal(10, 2));", dt);
    c("SELECT COUNT(*) FROM test where dec > CAST(333.0 AS decimal(10, 2));", dt);
    c("SELECT MIN(dec * dec) FROM test;", dt);
    c("SELECT MAX(dec * dec) FROM test;", dt);
    c("SELECT COUNT(*) FROM test WHERE x % 7 = 0;", dt);
    c("SELECT COUNT(*) FROM test WHERE x % 7 = 7;", dt);
    c("SELECT COUNT(*) FROM test WHERE x % 7 <> 0;", dt);
    c("SELECT COUNT(*) FROM test WHERE x % 7 <> 7;", dt);
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE x <> 7 AND x <> 8;", dt)),
              numeric_limits<int64_t>::max());
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE z <> 101 AND z <> 102;", dt)),
              numeric_limits<int64_t>::max());
    ASSERT_EQ(v<int64_t>(run_simple_agg("SELECT MIN(x) FROM test WHERE t <> 1001 AND t <> 1002;", dt)),
              numeric_limits<int64_t>::max());
  }
}

TEST(Select, LimitAndOffset) {
  CHECK(g_num_rows >= 4);
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 5;", dt);
      ASSERT_EQ(size_t(5), rows.rowCount());
    }
    {
      const auto rows = run_multiple_agg("SELECT * FROM test LIMIT 5 OFFSET 3;", dt);
      ASSERT_EQ(size_t(5), rows.rowCount());
    }
    {
      const auto rows = run_multiple_agg("SELECT * FROM test WHERE x <> 8 LIMIT 3 OFFSET 1;", dt);
      ASSERT_EQ(size_t(3), rows.rowCount());
    }
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
    c("SELECT COUNT(*) AS n FROM test GROUP BY d ORDER BY n;", dt);
    c("SELECT MIN(x + y) AS n FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY f + 1, f + d ORDER BY n;", dt);
    c("SELECT f + d AS s FROM test GROUP BY s ORDER BY s DESC;", dt);
  }
}

TEST(Select, FilterAndMultipleAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT AVG(x), AVG(y) FROM test;", dt);
    c("SELECT MIN(x), AVG(x * y), MAX(y + 7), COUNT(*) FROM test WHERE x + y > 47 AND x + y < 51;", dt);
  }
}

TEST(Select, FilterAndGroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x, y;", dt);
    c("SELECT MIN(x + y) FROM test WHERE x + y > 47 AND x + y < 53 GROUP BY x + 1, x + y;", dt);
    c("SELECT x, y, COUNT(*) FROM test GROUP BY x, y;", dt);
    c("SELECT x, dec, COUNT(*) FROM test GROUP BY x, dec ORDER BY x;", dt);
    c("SELECT dec as key, COUNT(*) AS value FROM test GROUP BY key HAVING key IS NOT NULL ORDER BY key, value DESC "
      "LIMIT 12;",
      dt);
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
    c("SELECT COUNT(distinct x + 1) FROM test;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x) FROM test GROUP BY y ORDER BY n;", dt);
    c("SELECT COUNT(*), MIN(x), MAX(x), AVG(y), SUM(z) AS n, COUNT(distinct x + 1) FROM test GROUP BY y ORDER BY n;",
      dt);
    c("SELECT COUNT(distinct dec) AS n FROM test GROUP BY y ORDER BY n;", dt);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct real_str) FROM test;", dt), std::runtime_error);
  }
}

TEST(Select, ScanNoAggregation) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT * FROM test;", dt);
    c("SELECT t.* FROM test t;", dt);
    c("SELECT x, z, t FROM test;", dt);
    c("SELECT x + z, t FROM test WHERE x <> 7 AND y > 42;", dt);
    c("SELECT * FROM test WHERE x > 8;", dt);
  }
}

TEST(Select, OrderBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    const auto rows = run_multiple_agg("SELECT x, y, z + t, x * y as m FROM test ORDER BY 3 desc LIMIT 5;", dt);
    CHECK_EQ(rows.rowCount(), std::min(5, static_cast<int>(g_num_rows)));
    CHECK_EQ(rows.colCount(), 4);
    for (size_t row_idx = 0; row_idx < rows.rowCount(); ++row_idx) {
      ASSERT_TRUE(v<int64_t>(rows.getRowAt(row_idx, 0, true)) == 8 || v<int64_t>(rows.getRowAt(row_idx, 0, true)) == 7);
      ASSERT_EQ(v<int64_t>(rows.getRowAt(row_idx, 1, true)), 43);
      ASSERT_EQ(v<int64_t>(rows.getRowAt(row_idx, 2, true)), 1104);
      ASSERT_TRUE(v<int64_t>(rows.getRowAt(row_idx, 3, true)) == 344 ||
                  v<int64_t>(rows.getRowAt(row_idx, 3, true)) == 301);
    }
    c("SELECT x, COUNT(distinct y) AS n FROM test GROUP BY x ORDER BY n DESC;", dt);
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
    const auto rows = run_multiple_agg(
        "SELECT x + y AS a, COUNT(*) * MAX(y) - SUM(z) AS b FROM test "
        "WHERE z BETWEEN 100 AND 200 GROUP BY x, y ORDER BY a DESC LIMIT 2;",
        dt);
    ASSERT_EQ(rows.rowCount(), size_t(2));
    {
      auto crt_row = rows.getNextRow(true, true);
      CHECK_EQ(size_t(2), crt_row.size());
      ASSERT_EQ(v<int64_t>(crt_row[0]), 51);
      ASSERT_EQ(v<int64_t>(crt_row[1]), -59 * g_num_rows / 2);
    }
    {
      auto crt_row = rows.getNextRow(true, true);
      CHECK_EQ(size_t(2), crt_row.size());
      ASSERT_EQ(v<int64_t>(crt_row[0]), 50);
      ASSERT_EQ(v<int64_t>(crt_row[1]), -59 * g_num_rows / 2);
    }
    auto empty_row = rows.getNextRow(true, true);
    CHECK(empty_row.empty());
  }
}

TEST(Select, GroupByExprNoFilterNoAggregate) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT x + y AS a FROM test GROUP BY a ORDER BY a;", dt);
  }
}

TEST(Select, Case) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) FROM test;", dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) "
      "FROM test WHERE CASE WHEN y BETWEEN 42 AND 43 THEN 5 ELSE 4 END > 4;",
      dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1 WHEN x BETWEEN 8 AND 9 THEN 2 ELSE 3 END) "
      "FROM test WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5 ELSE 4 END > 4;",
      dt);
    c("SELECT CASE WHEN x + y > 50 THEN 77 ELSE 88 END AS foo, COUNT(*) FROM test GROUP BY foo ORDER BY foo;", dt);
    c("SELECT SUM(CASE WHEN x BETWEEN 6 AND 7 THEN 1.1 WHEN x BETWEEN 8 AND 9 THEN 2.2 ELSE 3.3 END) "
      "FROM test WHERE CASE WHEN y BETWEEN 44 AND 45 THEN 5.1 ELSE 3.9 END > 4;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN 'oops 2' ELSE real_str END from "
      "test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 3 THEN 'oops 1' WHEN x BETWEEN 4 AND 6 THEN 'oops 2' ELSE str END from test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' end from "
      "test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN real_str ELSE 'ooops' END AS g "
      "FROM test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN str ELSE 'ooops' END FROM test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN 'seven' WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' END from "
      "test;",
      dt);
    c("SELECT CASE WHEN x BETWEEN 1 AND 7 THEN str WHEN x BETWEEN 7 AND 10 THEN 'eight' ELSE 'ooops' END AS g, "
      "COUNT(*) FROM test GROUP BY g ORDER BY g;",
      dt);
  }
}

TEST(Select, Strings) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT str, COUNT(*) FROM test GROUP BY str HAVING COUNT(*) > 5 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 4 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test WHERE str = 'bar' GROUP BY str HAVING COUNT(*) > 5 ORDER BY str;", dt);
    c("SELECT str, COUNT(*) FROM test GROUP BY str ORDER BY str;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str IS NOT NULL;", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE 'ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%eal_bar';", dt);
    c("SELECT COUNT(*) FROM test WHERE str LIKE '%ba%';", dt);
    c("SELECT * FROM test WHERE str LIKE '%';", dt);
    c("SELECT * FROM test WHERE str LIKE 'f%%';", dt);
    c("SELECT * FROM test WHERE str LIKE 'f%\%';", dt);
    c("SELECT * FROM test WHERE str LIKE '@f%%' ESCAPE '@';", dt);
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
    c("SELECT COUNT(*) FROM test WHERE LENGTH(str) = 3;", dt);
    ASSERT_EQ(2 * g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE CHAR_LENGTH(str) = 3;", dt)));
    ASSERT_EQ(g_num_rows, v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE str ILIKE 'f%%';", dt)));
  }
}

TEST(Select, StringsNoneEncoding) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_%%%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE 'real_ba%';", dt);
    c("SELECT COUNT(*) FROM test WHERE real_str LIKE '%eal_bar';", dt);
    c("SELECT * FROM test WHERE real_str LIKE '%';", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%%';", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_f%\%';", dt);
    c("SELECT * FROM test WHERE real_str LIKE 'real_@f%%' ESCAPE '@';", dt);
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
  }
}

TEST(Select, Time) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE m > timestamp(0) '2014-12-13T000000';", dt)));
    ASSERT_EQ(2 * g_num_rows,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE o > timestamp(0) '1999-09-08T160000';", dt)));
    ASSERT_EQ(0,
              v<int64_t>(run_simple_agg("SELECT COUNT(*) FROM test WHERE o > timestamp(0) '1999-09-10T160000';", dt)));
    ASSERT_EQ(14185093950L, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(20140, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(120, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(130, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM m) * 10) FROM test;", dt)));
    ASSERT_EQ(22, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM m)) FROM test;", dt)));
    ASSERT_EQ(23, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM m)) FROM test;", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM m)) FROM test;", dt)));
    ASSERT_EQ(6, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOW FROM m)) FROM test;", dt)));
    ASSERT_EQ(347, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DOY FROM m)) FROM test;", dt)));
    ASSERT_EQ(15, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(HOUR FROM n)) FROM test;", dt)));
    ASSERT_EQ(13, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MINUTE FROM n)) FROM test;", dt)));
    ASSERT_EQ(14, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(SECOND FROM n)) FROM test;", dt)));
    ASSERT_EQ(1999, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(YEAR FROM o)) FROM test;", dt)));
    ASSERT_EQ(9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(MONTH FROM o)) FROM test;", dt)));
    ASSERT_EQ(9, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(DAY FROM o)) FROM test;", dt)));
    ASSERT_EQ(4, v<int64_t>(run_simple_agg("SELECT EXTRACT(DOW FROM o) FROM test;", dt)));
    ASSERT_EQ(252, v<int64_t>(run_simple_agg("SELECT EXTRACT(DOY FROM o) FROM test;", dt)));
    ASSERT_EQ(936835200L, v<int64_t>(run_simple_agg("SELECT MAX(EXTRACT(EPOCH FROM o)) FROM test;", dt)));
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
  }
}

TEST(Select, DivByZero) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT x / 0 FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(distinct x / 0) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT d / 0. FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT f / (f - f) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY y / (x - x);", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY z, y / (x - x);", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT COUNT(*) FROM test GROUP BY y % (x - x);", dt), std::runtime_error);
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
    ASSERT_EQ(7, v<int64_t>(run_simple_agg("SELECT MAX(x) FROM test WHERE b = BOOLEAN 't';", dt)));
    ASSERT_EQ(3 * g_num_rows, v<int64_t>(run_simple_agg("SELECT SUM(2 * CAST(x = 7 AS INT)) FROM test;", dt)));
    c("SELECT COUNT(*) AS n FROM test GROUP BY x = 7, b ORDER BY n;", dt);
  }
}

TEST(Select, UnsupportedCast) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();
    EXPECT_THROW(run_multiple_agg("SELECT CAST(x AS text) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(arr_i32 AS text[]) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(arr_i32 AS float[]) FROM test;", dt), std::runtime_error);
    EXPECT_THROW(run_multiple_agg("SELECT CAST(arr_i32 AS double[]) FROM test;", dt), std::runtime_error);
  }
}

namespace Importer_NS {

ArrayDatum StringToArray(const std::string& s, const SQLTypeInfo& ti, const CopyParams& copy_params);
void parseStringArray(const std::string& s, const CopyParams& copy_params, std::vector<std::string>& string_vec);

}  // Importer_NS

namespace {

const size_t g_array_test_row_count{20};

void import_array_test(const std::string& table_name) {
  CHECK_EQ(size_t(0), g_array_test_row_count % 2);
  const auto& cat = g_session->get_catalog();
  const auto td = cat.getMetadataForTable(table_name);
  CHECK(td);
  Importer_NS::Loader loader(cat, td);
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  const auto col_descs = cat.getAllColumnMetadataForTable(td->tableId, false, false);
  for (const auto cd : col_descs) {
    import_buffers.emplace_back(
        new Importer_NS::TypedImportBuffer(cd,
                                           cd->columnType.get_compression() == kENCODING_DICT
                                               ? cat.getMetadataForDict(cd->columnType.get_comp_param())->stringDict
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
  loader.checkpoint();
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
      ASSERT_EQ(g_array_test_row_count + 2, result_rows.rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count + 2) * power10, v<int64_t>(result_rows.getRowAt(0, 1, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(0, 0, true)));
      ASSERT_EQ(power10, v<int64_t>(result_rows.getRowAt(g_array_test_row_count + 1, 1, true)));
      power10 *= 10;
    }
    for (const std::string float_type : {"float", "double"}) {
      auto result_rows = run_multiple_agg(
          "SELECT COUNT(*), UNNEST(arr_" + float_type + ") AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows.rowCount());
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(0, 0, true)));
    }
    {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_str) AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(g_array_test_row_count + 2, result_rows.rowCount());
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(g_array_test_row_count + 1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(0, 0, true)));
    }
    {
      auto result_rows =
          run_multiple_agg("SELECT COUNT(*), UNNEST(arr_bool) AS a FROM array_test GROUP BY a ORDER BY a DESC;", dt);
      ASSERT_EQ(size_t(2), result_rows.rowCount());
      ASSERT_EQ(int64_t(g_array_test_row_count * 3), v<int64_t>(result_rows.getRowAt(0, 0, true)));
      ASSERT_EQ(int64_t(g_array_test_row_count * 3), v<int64_t>(result_rows.getRowAt(1, 0, true)));
      ASSERT_EQ(1, v<int64_t>(result_rows.getRowAt(0, 1, true)));
      ASSERT_EQ(0, v<int64_t>(result_rows.getRowAt(1, 1, true)));
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
      ASSERT_EQ(g_array_test_row_count, result_rows.rowCount());
      for (size_t row_idx = 0; row_idx < g_array_test_row_count; ++row_idx) {
        ASSERT_EQ(3, v<int64_t>(result_rows.getRowAt(row_idx, 0, true)));
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
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");
  desc.add_options()("use-nvptx", "Use NVPTX instead of NVVM");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  if (vm.count("use-nvptx"))
    g_nvvm_backend = NVVMBackend::NVPTX;

  try {
    const std::string drop_old_test{"DROP TABLE IF EXISTS test;"};
    run_ddl_statement(drop_old_test);
    g_sqlite_comparator.query(drop_old_test);
    const std::string create_test{
        "CREATE TABLE test(x int not null, y int, z smallint, t bigint, b boolean, f float, d double, str text "
        "encoding dict, real_str text, m timestamp(0), n time(0), o date, fx int encoding fixed(16), dec decimal(10, "
        "2)) WITH (fragment_size=2);"};
    run_ddl_statement(create_test);
    g_sqlite_comparator.query(
        "CREATE TABLE test(x int not null, y int, z smallint, t bigint, b boolean, f float, d double, str text, "
        "real_str text, m timestamp(0), n time(0), o date, fx int, dec decimal(10, 2));");
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'test'";
    return -EEXIST;
  }
  CHECK_EQ(g_num_rows % 2, 0);
  for (ssize_t i = 0; i < g_num_rows; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 42, 101, 1001, 't', 1.1, 2.2, 'foo', 'real_foo', '2014-12-13T222315', '15:13:14', "
        "'1999-09-09', 9, 111.1);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(8, 43, 102, 1002, 'f', 1.2, 2.4, 'bar', 'real_bar', '2014-12-13T222315', '15:13:14', "
        "'1999-09-09', 10, 222.2);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  for (ssize_t i = 0; i < g_num_rows / 2; ++i) {
    const std::string insert_query{
        "INSERT INTO test VALUES(7, 43, 102, 1002, 't', 1.3, 2.6, 'baz', 'real_baz', '2014-12-13T222315', '15:13:14', "
        "'1999-09-09', 11, 333.3);"};
    run_multiple_agg(insert_query, ExecutorDeviceType::CPU);
    g_sqlite_comparator.query(insert_query);
  }
  try {
    const std::string drop_old_array_test{"DROP TABLE IF EXISTS array_test;"};
    run_ddl_statement(drop_old_array_test);
    const std::string create_array_test{
        "CREATE TABLE array_test(x int, arr_i16 smallint[], arr_i32 int[], arr_i64 bigint[], arr_str text[] encoding "
        "dict, "
        "arr_float float[], arr_double double[], arr_bool boolean[], real_str text) WITH (fragment_size=2);"};
    run_ddl_statement(create_array_test);
  } catch (...) {
    LOG(ERROR) << "Failed to (re-)create table 'array_test'";
    return -EEXIST;
  }
  import_array_test("array_test");
  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const exception& e) {
    LOG(ERROR) << e.what();
  }
  Executor::nukeCacheOfExecutors();
  const std::string drop_test{"DROP TABLE test;"};
  run_ddl_statement(drop_test);
  g_sqlite_comparator.query(drop_test);
  const std::string drop_array_test{"DROP TABLE array_test;"};
  run_ddl_statement(drop_array_test);
  g_session.reset(nullptr);
  return err;
}
