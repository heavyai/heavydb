/*
 * Copyright 2024 HEAVY.AI, Inc.
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

/**
 * @file    SqlTestFixture.h
 * @brief   Thin-client gtest fixture for tests that run SQL against a live heavydb server.
 *
 * Link deps (vs. the full EXECUTE_TEST_LIBS / THRIFT_HANDLER_TEST_LIBRARIES stack):
 *   gtest  Logger  Shared  mapd_thrift  ThriftClient
 *
 * Usage in main():
 *   SqlTestFixture::initConnection(argc, argv);   // parses --host/--port/--user/--pwd/--db
 *   ::testing::InitGoogleTest(&argc, argv);
 *   return RUN_ALL_TESTS();
 *
 * Usage in tests:
 *   class MyTest : public SqlTestFixture { ... };
 *   sql("CREATE TABLE t (x INT)");
 *   TQueryResult r;
 *   sql(r, "SELECT x FROM t");
 *   assertResultSetEqual({{int64_t{1}}}, r);
 */

#pragma once

#include <gtest/gtest.h>

#include <thrift/protocol/TBinaryProtocol.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>
#include <boost/variant.hpp>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "QueryEngine/TargetValue.h"
#include "Shared/SysDefinitions.h"
#include "Shared/scope.h"
#include "Shared/ThriftClient.h"
#include "gen-cpp/Heavy.h"

using namespace ::apache::thrift::protocol;

namespace po = boost::program_options;

using NullableTargetValue = boost::variant<TargetValue, void*>;
using ExpectedResult = std::vector<std::vector<NullableTargetValue>>;

// Global connection — set up by SqlTestFixture::initConnection() in main().
inline std::shared_ptr<HeavyClient> g_sql_test_client;
inline TSessionId g_sql_test_session_id;
// The admin session opened by initConnection(); switchToAdmin() reuses it because every
// connect() pays for password hashing.
inline TSessionId g_sql_admin_session_id;

// ---------------------------------------------------------------------------
// Visitor helpers (operate on Thrift TDatum / TRowSet only)
// ---------------------------------------------------------------------------

namespace sql_test_detail {

class AssertValueEqualsVisitor : public boost::static_visitor<> {
 public:
  AssertValueEqualsVisitor(const TDatum& datum,
                           const TColumnType& col_type,
                           size_t row,
                           size_t col)
      : datum_(datum), col_type_(col_type), row_(row), col_(col) {}

  template <typename T>
  void operator()(const T& value) const {
    throw std::runtime_error{std::string("Unexpected type in test assertion: ") +
                             typeid(value).name()};
  }

  void operator()(const int64_t value) const {
    EXPECT_EQ(datum_.val.int_val, value)
        << boost::format("At row: %d, column: %d") % row_ % col_;
  }

  void operator()(const double value) const {
    EXPECT_DOUBLE_EQ(datum_.val.real_val, value)
        << boost::format("At row: %d, column: %d") % row_ % col_;
  }

  void operator()(const float value) const {
    EXPECT_FLOAT_EQ(static_cast<float>(datum_.val.real_val), value)
        << boost::format("At row: %d, column: %d") % row_ % col_;
  }

  void operator()(const std::string& value) const {
    EXPECT_TRUE(!datum_.is_null)
        << boost::format("At row: %d, column: %d") % row_ % col_;
    const auto type = col_type_.col_type.type;
    if (isDateOrTime(type)) {
      auto type_info = SQLTypeInfo(getDatetimeSqlType(type),
                                   col_type_.col_type.precision,
                                   col_type_.col_type.scale);
      auto dt = StringToDatum(value, type_info);
      EXPECT_EQ(dt.bigintval, datum_.val.int_val)
          << boost::format("At row: %d, column: %d") % row_ % col_;
    } else {
      EXPECT_EQ(datum_.val.str_val, value)
          << boost::format("At row: %d, column: %d") % row_ % col_;
    }
  }

  void operator()(const ScalarTargetValue& value) const {
    boost::apply_visitor(AssertValueEqualsVisitor{datum_, col_type_, row_, col_}, value);
  }

  void operator()(const NullableString& value) const {
    if (boost::get<std::string>(&value)) {
      boost::apply_visitor(AssertValueEqualsVisitor{datum_, col_type_, row_, col_}, value);
    } else {
      EXPECT_TRUE(datum_.is_null)
          << boost::format("At row: %d, column: %d") % row_ % col_;
    }
  }

  void operator()(const ArrayTargetValue& values_opt) const {
    const auto& values = values_opt.get();
    ASSERT_EQ(values.size(), datum_.val.arr_val.size());
    for (size_t i = 0; i < values.size(); i++) {
      boost::apply_visitor(
          AssertValueEqualsVisitor{datum_.val.arr_val[i], col_type_, row_, col_},
          values[i]);
    }
  }

 private:
  bool isDateOrTime(TDatumType::type type) const {
    return type == TDatumType::TIME || type == TDatumType::TIMESTAMP ||
           type == TDatumType::DATE;
  }

  SQLTypes getDatetimeSqlType(TDatumType::type type) const {
    if (type == TDatumType::TIME) {
      return kTIME;
    }
    if (type == TDatumType::TIMESTAMP) {
      return kTIMESTAMP;
    }
    return kDATE;
  }

  const TDatum& datum_;
  const TColumnType& col_type_;
  size_t row_;
  size_t col_;
};

class AssertValueEqualsOrIsNullVisitor : public boost::static_visitor<> {
 public:
  AssertValueEqualsOrIsNullVisitor(const TDatum& datum,
                                   const TColumnType& col_type,
                                   size_t row,
                                   size_t col)
      : datum_(datum), col_type_(col_type), row_(row), col_(col) {}

  void operator()(const TargetValue& value) const {
    boost::apply_visitor(AssertValueEqualsVisitor{datum_, col_type_, row_, col_}, value);
  }

  void operator()(const void* /*null*/) const {
    EXPECT_TRUE(datum_.is_null)
        << boost::format("At row: %d, column: %d") % row_ % col_;
  }

  const TDatum& datum_;
  const TColumnType& col_type_;
  size_t row_;
  size_t col_;
};

// ---------------------------------------------------------------------------
// TRowSet traversal helpers
// ---------------------------------------------------------------------------

inline size_t getRowCount(const TRowSet& row_set) {
  if (row_set.is_columnar) {
    return row_set.columns.empty() ? 0 : row_set.columns[0].nulls.size();
  }
  return row_set.rows.size();
}

inline size_t getColumnCount(const TRowSet& row_set) {
  if (row_set.is_columnar) {
    return row_set.columns.size();
  }
  return row_set.rows.empty() ? 0 : row_set.rows[0].cols.size();
}

inline void setDatumArray(std::vector<TDatum>& out, const TColumn& col) {
  const auto& d = col.data;
  if (!d.int_col.empty()) {
    for (auto v : d.int_col) {
      TDatum item;
      item.val.int_val = v;
      out.push_back(item);
    }
  } else if (!d.real_col.empty()) {
    for (auto v : d.real_col) {
      TDatum item;
      item.val.real_val = v;
      out.push_back(item);
    }
  } else if (!d.str_col.empty()) {
    for (const auto& v : d.str_col) {
      TDatum item;
      item.val.str_val = v;
      out.push_back(item);
    }
  }
  CHECK(col.nulls.size() == out.size());
  for (size_t i = 0; i < col.nulls.size(); ++i) {
    out[i].is_null = col.nulls[i];
  }
}

inline void setDatum(TDatum& datum,
                     const TColumnData& data,
                     size_t idx,
                     bool is_null) {
  if (!data.int_col.empty()) {
    datum.val.int_val = data.int_col[idx];
  } else if (!data.real_col.empty()) {
    datum.val.real_val = data.real_col[idx];
  } else if (!data.str_col.empty()) {
    datum.val.str_val = data.str_col[idx];
  } else if (!data.arr_col.empty()) {
    std::vector<TDatum> arr;
    if (!is_null) {
      setDatumArray(arr, data.arr_col[idx]);
    }
    datum.val.arr_val = arr;
  } else {
    throw std::runtime_error{"Unexpected column data in getRow"};
  }
}

inline std::vector<TDatum> getRow(const TRowSet& row_set, size_t idx) {
  if (row_set.is_columnar) {
    std::vector<TDatum> row;
    for (const auto& col : row_set.columns) {
      TDatum datum;
      bool is_null = col.nulls[idx];
      setDatum(datum, col.data, idx, is_null);
      datum.is_null = is_null;
      row.push_back(datum);
    }
    return row;
  }
  return row_set.rows[idx].cols;
}

}  // namespace sql_test_detail

// ---------------------------------------------------------------------------
// Helpers used by SqlTestFixture (mirrored from DBHandlerTestHelpers.h)
// ---------------------------------------------------------------------------

constexpr void* Null = nullptr;
constexpr int64_t True = 1;
constexpr int64_t False = 0;

template <typename T, typename... Ts>
constexpr bool is_any() {
  return ((std::is_same_v<T, Ts> || ...));
}

class ToStringVisitor : public boost::static_visitor<std::string> {
 public:
  template <typename T>
  std::string operator()(const T& value) const {
    if constexpr (is_any<T, GeoTargetValue, GeoTargetValuePtr>()) {
      throw std::runtime_error("Geo types are currently not supported by ToStringVisitor");
    } else if constexpr (is_any<T, TargetValue, ScalarTargetValue, NullableString>()) {
      return boost::apply_visitor(ToStringVisitor{}, value);
    } else {
      std::stringstream ss;
      ss << value;
      return ss.str();
    }
  }

  std::string operator()(const ArrayTargetValue& value_optional) const {
    std::stringstream ss;
    if (value_optional.has_value()) {
      ss << "{";
      bool first_value{true};
      for (const auto& element : value_optional.value()) {
        if (first_value) {
          first_value = false;
        } else {
          ss << ", ";
        }
        ss << boost::apply_visitor(ToStringVisitor{}, element);
      }
      ss << "}";
    } else {
      ss << "NULL";
    }
    return ss.str();
  }

  std::string operator()(const std::string& value) const { return "'" + value + "'"; }

  std::string operator()(const void* /*null*/) const { return "NULL"; }
};

// ---------------------------------------------------------------------------
// ThinResultSet
// Wraps a TQueryResult and presents the same rowCount() / getNextRow() interface
// as the in-process ResultSet class, allowing EXECUTE_TEST_LIBS tests to be
// migrated to thin-client with minimal assertion-code changes.
// ---------------------------------------------------------------------------

class ThinResultSet {
 public:
  explicit ThinResultSet(const TQueryResult& result) : result_(result), row_idx_(0) {}

  size_t rowCount() const { return sql_test_detail::getRowCount(result_.row_set); }
  size_t colCount() const { return result_.row_set.row_desc.size(); }
  bool definitelyHasNoRows() const { return rowCount() == 0; }

  // Returns the Thrift column descriptor for column `idx`.
  // Use col_type.col_type.type (TDatumType) instead of SQLTypeInfo::get_type().
  const TColumnType& getColType(size_t idx) const {
    return result_.row_set.row_desc[idx];
  }

  std::vector<TargetValue> getNextRow(bool /*translate_strings*/,
                                      bool /*decimal_to_double*/) {
    const auto& row_set = result_.row_set;
    if (row_idx_ >= sql_test_detail::getRowCount(row_set)) {
      return {};
    }
    auto datums = sql_test_detail::getRow(row_set, row_idx_++);
    std::vector<TargetValue> row;
    row.reserve(datums.size());
    for (size_t c = 0; c < datums.size(); c++) {
      row.push_back(datumToTargetValue(datums[c], row_set.row_desc[c]));
    }
    return row;
  }

  void reset() { row_idx_ = 0; }

  // No-op: thin client returns geo as WKT strings; no server-side type conversion.
  template <typename T>
  void setGeoReturnType(T /*type*/) {}

 private:
  // Null sentinel constants matching in-process ResultSet behavior.
  // Tests that use v<int64_t>() or getIntValue() on nullable columns expect
  // these sentinels (not NullableString) for NULL rows — matching the in-process
  // ResultSet convention used by EXECUTE_TEST_LIBS tests.
  static constexpr int64_t kNullBigint = std::numeric_limits<int64_t>::min();
  static constexpr int64_t kNullInt =
      static_cast<int64_t>(std::numeric_limits<int32_t>::min());
  static constexpr int64_t kNullSmallint =
      static_cast<int64_t>(std::numeric_limits<int16_t>::min());
  static constexpr int64_t kNullTinyint =
      static_cast<int64_t>(std::numeric_limits<int8_t>::min());
  // Float/double sentinels match inline_fp_null_val() from Shared/InlineNullValues.h.
  static constexpr float kNullFloat = std::numeric_limits<float>::min();
  static constexpr double kNullDouble = std::numeric_limits<double>::min();

  static TargetValue datumToTargetValue(const TDatum& datum,
                                        const TColumnType& col_type) {
    auto type = col_type.col_type.type;
    bool is_array = col_type.col_type.is_array;

    if (is_array) {
      if (datum.is_null) {
        return ArrayTargetValue{boost::none};
      }
      std::vector<ScalarTargetValue> arr;
      for (const auto& elem : datum.val.arr_val) {
        arr.push_back(scalarFromDatum(elem, type));
      }
      return ArrayTargetValue{arr};
    }

    return ScalarTargetValue{scalarFromDatum(datum, type)};
  }

  static ScalarTargetValue scalarFromDatum(const TDatum& datum, TDatumType::type type) {
    if (type == TDatumType::FLOAT) {
      return datum.is_null ? ScalarTargetValue{kNullFloat}
                           : ScalarTargetValue{static_cast<float>(datum.val.real_val)};
    }
    if (type == TDatumType::DOUBLE || type == TDatumType::DECIMAL) {
      return datum.is_null ? ScalarTargetValue{kNullDouble}
                           : ScalarTargetValue{datum.val.real_val};
    }
    if (type == TDatumType::STR || type == TDatumType::POINT ||
        type == TDatumType::LINESTRING || type == TDatumType::POLYGON ||
        type == TDatumType::MULTIPOLYGON || type == TDatumType::GEOMETRY ||
        type == TDatumType::GEOGRAPHY || type == TDatumType::MULTILINESTRING) {
      return datum.is_null ? ScalarTargetValue{NullableString{static_cast<void*>(nullptr)}}
                           : ScalarTargetValue{NullableString{datum.val.str_val}};
    }
    // Integer and time types: return in-process sentinel for null (so v<int64_t>() works).
    if (datum.is_null) {
      switch (type) {
        case TDatumType::BIGINT:
        case TDatumType::TIME:
        case TDatumType::TIMESTAMP:
        case TDatumType::DATE:
          return kNullBigint;
        case TDatumType::SMALLINT:
          return kNullSmallint;
        case TDatumType::TINYINT:
        case TDatumType::BOOL:
          return kNullTinyint;
        default:  // INT
          return kNullInt;
      }
    }
    return datum.val.int_val;
  }

  TQueryResult result_;
  size_t row_idx_;
};

// Replacement for SKIP_NO_GPU() in thin-client tests.
// Requires `dt` (TExecuteMode::type) in scope, inside a for-loop.
#define SKIP_NO_GPU_THINCLIENT()                                                    \
  if ((dt) == TExecuteMode::GPU && !SqlTestFixture::gpusPresent()) { continue; }

// ---------------------------------------------------------------------------
// SqlTestFixture
// ---------------------------------------------------------------------------

class SqlTestFixture : public ::testing::Test {
 public:
  /**
   * Call once from main() before RUN_ALL_TESTS().
   * Parses --host, --port, --user, --pwd, --db from argv; remaining args
   * are left for gtest to consume.
   */
  void SetUp() override { switchToAdmin(); }

  static void initConnection(int argc, char** argv) {
    std::string host = "localhost";
    int port = 6274;
    std::string cert;
    std::string user = shared::kRootUsername;
    std::string pwd = shared::kDefaultRootPasswd;
    std::string db = shared::kDefaultDbName;

    po::options_description desc("SQL test server options");
    // clang-format off
    desc.add_options()
        ("host",  po::value<std::string>(&host)->default_value(host),  "heavydb hostname")
        ("port",  po::value<int>(&port)->default_value(port),          "heavydb TCP port")
        ("cert",  po::value<std::string>(&cert)->default_value(cert),  "TLS certificate path")
        ("user",  po::value<std::string>(&user)->default_value(user),  "DB username")
        ("pwd",   po::value<std::string>(&pwd)->default_value(pwd),    "DB password")
        ("db",    po::value<std::string>(&db)->default_value(db),      "DB name");
    // clang-format on

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(),
              vm);
    po::notify(vm);

    auto conn_mgr = std::make_shared<ThriftClientConnection>();
    auto transport = conn_mgr->open_buffered_client_transport(host, port, cert);
    transport->open();
    auto protocol = std::make_shared<TBinaryProtocol>(transport);
    g_sql_test_client = std::make_shared<HeavyClient>(protocol);
    g_sql_test_client->connect(g_sql_test_session_id, user, pwd, db);
    g_sql_admin_session_id = g_sql_test_session_id;
  }

  // -------------------------------------------------------------------------
  // SQL execution (matches DBHandlerTestFixture API)
  // -------------------------------------------------------------------------

  static void sql(const std::string& query) {
    TQueryResult result;
    sql(result, query);
  }

  static void sql(TQueryResult& result, const std::string& query) {
    g_sql_test_client->sql_execute(
        result, g_sql_test_session_id, boost::trim_copy(query), true, "", -1, -1);
  }

  // -------------------------------------------------------------------------
  // Result-set assertions
  // -------------------------------------------------------------------------

  static void assertResultSetEqual(const ExpectedResult& expected,
                                   const TQueryResult& actual) {
    const auto& row_set = actual.row_set;
    auto row_count = sql_test_detail::getRowCount(row_set);
    ASSERT_EQ(expected.size(), row_count)
        << "Returned result set does not have the expected number of rows";
    if (row_count == 0) {
      return;
    }
    size_t col_count = sql_test_detail::getColumnCount(row_set);
    ASSERT_EQ(expected[0].size(), col_count)
        << "Returned result set does not have the expected number of columns";

    for (size_t r = 0; r < row_count; r++) {
      auto row = sql_test_detail::getRow(row_set, r);
      for (size_t c = 0; c < col_count; c++) {
        boost::apply_visitor(
            sql_test_detail::AssertValueEqualsOrIsNullVisitor{
                row[c], row_set.row_desc[c], r, c},
            expected[r][c]);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Row/column counting (matches DBHandlerTestFixture API)
  // -------------------------------------------------------------------------

  static size_t getRowCount(const TQueryResult& result) {
    return sql_test_detail::getRowCount(result.row_set);
  }

  static size_t getColumnCount(const TQueryResult& result) {
    return sql_test_detail::getColumnCount(result.row_set);
  }

  static std::vector<TDatum> getRow(const TQueryResult& result, size_t index) {
    return sql_test_detail::getRow(result.row_set, index);
  }

  // -------------------------------------------------------------------------
  // Exception helpers (matches DBHandlerTestFixture API)
  // -------------------------------------------------------------------------

  template <typename Lambda>
  static void executeLambdaAndAssertException(Lambda lambda,
                                              const std::string& error_message,
                                              bool i_case = false) {
    try {
      lambda();
      FAIL() << "Expected exception with message: " << error_message;
    } catch (const TDBException& e) {
      assertExceptionMessage(e, error_message, i_case);
    } catch (const std::runtime_error& e) {
      assertExceptionMessage(e, error_message, i_case);
    }
  }

  static void assertExceptionMessage(const TDBException& e,
                                     const std::string& expected,
                                     bool i_case = false) {
    std::string actual = e.error_msg;
    std::string exp = expected;
    if (i_case) {
      boost::algorithm::to_lower(actual);
      boost::algorithm::to_lower(exp);
    }
    ASSERT_EQ(exp, actual);
  }

  static void assertExceptionMessage(const std::runtime_error& e,
                                     const std::string& expected,
                                     bool i_case = false) {
    std::string actual = e.what();
    std::string exp = expected;
    if (i_case) {
      boost::algorithm::to_lower(actual);
      boost::algorithm::to_lower(exp);
    }
    ASSERT_EQ(exp, actual);
  }

  // -------------------------------------------------------------------------
  // Execute-mode control (thin-client version of DBHandlerTestFixture::setExecuteMode)
  // -------------------------------------------------------------------------

  static bool setExecuteMode(const TExecuteMode::type mode) {
    try {
      g_sql_test_client->set_execution_mode(g_sql_test_session_id, mode);
      return true;
    } catch (...) {
      return false;
    }
  }

  // -------------------------------------------------------------------------
  // Combined execute + assert (mirrors DBHandlerTestFixture::sqlAndCompareResult)
  // -------------------------------------------------------------------------

  static void sqlAndCompareResult(
      const std::string& query,
      const std::vector<std::vector<NullableTargetValue>>& expected) {
    TQueryResult result;
    sql(result, query);
    assertResultSetEqual(expected, result);
  }

  // -------------------------------------------------------------------------
  // Exception-checking helpers (mirrors DBHandlerTestFixture API)
  // -------------------------------------------------------------------------

  static void queryAndAssertException(const std::string& query,
                                      const std::string& error_message,
                                      bool i_case = false) {
    executeLambdaAndAssertException([&] { sql(query); }, error_message, i_case);
  }

  static void queryAndAssertPartialException(const std::string& query,
                                             const std::string& error_message) {
    executeLambdaAndAssertPartialException([&] { sql(query); }, error_message);
  }

  template <typename Lambda>
  static void executeLambdaAndAssertPartialException(Lambda lambda,
                                                     const std::string& error_message) {
    try {
      lambda();
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const TDBException& e) {
      assertPartialExceptionMessage(e, error_message);
    } catch (const std::runtime_error& e) {
      assertPartialExceptionMessage(e, error_message);
    }
  }

  static void assertPartialExceptionMessage(const TDBException& e,
                                            const std::string& error_message) {
    ASSERT_TRUE(e.error_msg.find(error_message) != std::string::npos)
        << "Found:\n  " << e.error_msg << "\nExpected:\n  " << error_message;
  }

  static void assertPartialExceptionMessage(const std::runtime_error& e,
                                            const std::string& error_message) {
    ASSERT_TRUE(std::string(e.what()).find(error_message) != std::string::npos)
        << "Found:\n  " << e.what() << "\nExpected:\n  " << error_message;
  }

  // -------------------------------------------------------------------------
  // Literal helpers (mirrors DBHandlerTestFixture API)
  // -------------------------------------------------------------------------

  static constexpr int64_t i(int64_t v) { return v; }

  static boost::optional<std::vector<ScalarTargetValue>> array(
      std::vector<ScalarTargetValue> a) {
    return a;
  }

  // -------------------------------------------------------------------------
  // SQL execution with an explicit session (for multi-user tests)
  // -------------------------------------------------------------------------

  static void sql(TQueryResult& result,
                  const std::string& query,
                  const TSessionId& session) {
    g_sql_test_client->sql_execute(
        result, session, boost::trim_copy(query), true, "", -1, -1);
  }

  // -------------------------------------------------------------------------
  // Convenience: login / logout (for auth tests)
  // -------------------------------------------------------------------------

  static void login(const std::string& user,
                    const std::string& pwd,
                    const std::string& db = shared::kDefaultDbName) {
    g_sql_test_client->connect(g_sql_test_session_id, user, pwd, db);
  }

  // Login and return the new session — caller must logout(session) when done.
  static void login(const std::string& user,
                    const std::string& pwd,
                    const std::string& db,
                    TSessionId& session) {
    g_sql_test_client->connect(session, user, pwd, db);
  }

  static void logout() { g_sql_test_client->disconnect(g_sql_test_session_id); }

  static void logout(const TSessionId& session) {
    g_sql_test_client->disconnect(session);
  }

  // Make the global session the admin session opened by initConnection().
  static void switchToAdmin() { g_sql_test_session_id = g_sql_admin_session_id; }

  // Alias for switchToAdmin (mirrors DBHandlerTestFixture::loginAdmin).
  static void loginAdmin() { switchToAdmin(); }

  // createDBHandler() is a no-op in thin-client mode (connection handled by initConnection).
  static void createDBHandler() {}

  // Distributed mode is being deprecated; always false in thin-client tests.
  static bool isDistributedMode() { return false; }

  // -------------------------------------------------------------------------
  // GPU detection (replaces QR::get()->gpusPresent() in EXECUTE_TEST_LIBS tests)
  // -------------------------------------------------------------------------

  static bool gpusPresent() {
    TClusterHardwareInfo hw;
    g_sql_test_client->get_hardware_info(hw, g_sql_test_session_id);
    for (const auto& node : hw.hardware_info) {
      if (node.num_gpu_hw > 0) {
        return true;
      }
    }
    return false;
  }

  // -------------------------------------------------------------------------
  // run_multiple_agg / run_simple_agg
  // Drop-in replacements for QR::get()->runSQL() in EXECUTE_TEST_LIBS tests.
  // Returns a ThinResultSet whose getNextRow()/rowCount() match the in-process
  // ResultSet interface so v<T>() assertions work unchanged.
  // -------------------------------------------------------------------------

  static std::shared_ptr<ThinResultSet> run_multiple_agg(
      const std::string& query,
      const TExecuteMode::type device = TExecuteMode::CPU,
      const bool /*allow_loop_joins*/ = true) {
    setExecuteMode(device);
    TQueryResult result;
    sql(result, query);
    return std::make_shared<ThinResultSet>(result);
  }

  static TargetValue run_simple_agg(
      const std::string& query,
      const TExecuteMode::type device = TExecuteMode::CPU) {
    auto result = run_multiple_agg(query, device);
    if (result->rowCount() == 0) {
      return ScalarTargetValue{NullableString{static_cast<void*>(nullptr)}};
    }
    auto row = result->getNextRow(true, true);
    return row.empty() ? ScalarTargetValue{NullableString{static_cast<void*>(nullptr)}} : row[0];
  }

  // -------------------------------------------------------------------------
  // Direct Thrift client access (mirrors DBHandlerTestFixture::getDbHandlerAndSessionId)
  // Returns (HeavyClient*, TSessionId) — caller can invoke any Thrift API directly.
  // -------------------------------------------------------------------------

  static std::pair<HeavyClient*, TSessionId> getDbHandlerAndSessionId() {
    return {g_sql_test_client.get(), g_sql_test_session_id};
  }

  // -------------------------------------------------------------------------
  // Table / column metadata (replaces getCatalog() for schema inspection)
  // -------------------------------------------------------------------------

  static TTableDetails getTableDetails(const std::string& table_name) {
    TTableDetails details;
    g_sql_test_client->get_table_details(details, g_sql_test_session_id, table_name);
    return details;
  }

  // Returns the TTypeInfo for a named column (.encoding, .comp_param, .type, …).
  // Throws if table or column is not found.
  static TTypeInfo getColumnType(const std::string& table_name,
                                 const std::string& col_name) {
    auto details = getTableDetails(table_name);
    for (const auto& col : details.row_desc) {
      if (col.col_name == col_name) {
        return col.col_type;
      }
    }
    throw std::runtime_error{"Column not found: " + col_name + " in " + table_name};
  }

  // -------------------------------------------------------------------------
  // Running-query status (replaces Executor::getQuerySessionInfo for interrupt tests)
  // -------------------------------------------------------------------------

  static std::vector<TQueryInfo> getRunningQueries() {
    std::vector<TQueryInfo> queries;
    g_sql_test_client->get_queries_info(queries, g_sql_test_session_id);
    return queries;
  }

  // Poll until a query whose session matches `session` has `status_substr` in its
  // current_status or query_str field. Returns false on timeout.
  static bool waitForQueryInStatus(const std::string& session,
                                   const std::string& status_substr,
                                   int timeout_ms = 60000) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
      for (const auto& q : getRunningQueries()) {
        if (q.query_session_id == session &&
            (q.current_status.find(status_substr) != std::string::npos ||
             q.query_str.find(status_substr) != std::string::npos)) {
          return true;
        }
      }
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
      if (elapsed >= timeout_ms) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
};
