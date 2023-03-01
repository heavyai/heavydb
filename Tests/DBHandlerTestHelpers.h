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

#pragma once

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#include "Catalog/Catalog.h"
#include "QueryRunner/TestProcessSignalHandler.h"
#include "Shared/clean_boost_regex.hpp"
#include "ThriftHandler/DBHandler.h"

#include <gtest/gtest.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>

constexpr int64_t True = 1;
constexpr int64_t False = 0;
constexpr void* Null = nullptr;
constexpr int64_t Null_i = NULL_INT;

using NullableTargetValue = boost::variant<TargetValue, void*>;
using ExpectedResult = std::vector<std::vector<NullableTargetValue>>;
namespace po = boost::program_options;

extern size_t g_leaf_count;
extern bool g_cluster;
extern bool g_enable_system_tables;
extern bool g_read_only;

namespace {
using ColumnPair = std::pair<std::string, std::string>;

std::vector<std::string> split_on_regex(const std::string& in, const std::string& regex) {
  std::vector<std::string> tokens;
  boost::split_regex(tokens, in, boost::regex{regex});
  return tokens;
}

boost::regex make_regex(const std::string& pattern) {
  std::string whitespace_wrapper = "\\s*" + pattern + "\\s*";
  return boost::regex(whitespace_wrapper, boost::regex::icase);
}

const std::map<std::string, std::map<boost::regex, std::string>>
    k_rdms_column_type_substitutes = {
        {"sqlite",
         {{make_regex("TEXT.*"), "text"},
          {make_regex("DECIMAL\\s*\\(\\d+,\\s*\\d+\\)\\s*(\\[\\d*\\])?"), "double"},
          {make_regex("FLOAT"), "double"}}},
        {"postgres",
         {{make_regex("TEXT.*"), "text"},
          {make_regex("FLOAT"), "real"},
          {make_regex("DOUBLE"), "double precision"},
          {make_regex("TINYINT"), "smallint"},
          {make_regex("TIME\\b"), "time(0)"},
          {make_regex("TIMESTAMP"), "timestamp(0)"},
          {make_regex("TIMESTAMP\\s*\\(6\\)"), "timestamp"},
          {make_regex("(MULTI)?POINT"), "geometry"},
          {make_regex("(MULTI)?LINESTRING"), "geometry"},
          {make_regex("(MULTI)?POLYGON"), "geometry"}}},
        {"redshift",
         {{make_regex("TEXT.*"), "text"},
          {make_regex("FLOAT"), "real"},
          {make_regex("DOUBLE"), "double precision"},
          {make_regex("TIMESTAMP\\s*\\(\\d+\\)"), "timestamp"},
          {make_regex("TINYINT"), "smallint"},
          {make_regex("(MULTI)?POINT"), "geometry"},
          {make_regex("(MULTI)?LINESTRING"), "geometry"},
          {make_regex("(MULTI)?POLYGON"), "geometry"}}},
        {"snowflake",
         {{make_regex("TEXT.*"), "text"},
          {make_regex("TIME\\b"), "time(0)"},
          {make_regex("(MULTI)?POINT"), "geography"},
          {make_regex("(MULTI)?LINESTRING"), "geography"},
          {make_regex("(MULTI)?POLYGON"), "geography"}}},
        {"bigquery",
         {{make_regex("TEXT.*"), "string"},
          {make_regex("TIMESTAMP\\s*\\(\\d+\\)"), "timestamp"},
          {make_regex("TIME\\s*\\(\\d+\\)"), "time"},
          {make_regex("FLOAT"), "float64"},
          {make_regex("DOUBLE"), "float64"},
          {make_regex("(MULTI)?POINT"), "geography"},
          {make_regex("(MULTI)?LINESTRING"), "geography"},
          {make_regex("(MULTI)?POLYGON"), "geography"}}}};

const std::map<std::string, std::map<boost::regex, std::string>>
    k_rdms_column_type_prepend = {
        {"bigquery",
         {{make_regex("DECIMAL\\s*\\(\\d\\d+(,\\s*\\d\\d+)?\\)\\s*"), "BIG"}}}};
// `DECIMAL` types need to be upgraded to `BIGDECIMAL` if the
// scale exceeds the absolute range: 0 ≤ S ≤ 9 or if the precision
// exceeds the relative range: max(1, S) ≤ P ≤ S + 29

std::string get_col_type_for_rdms(std::string col_type, const std::string& rdms) {
  if (const auto& rdms_it = k_rdms_column_type_substitutes.find(rdms);
      rdms_it != k_rdms_column_type_substitutes.end()) {
    for (const auto& substitute : rdms_it->second) {
      if (boost::regex_match(col_type, substitute.first)) {
        col_type = substitute.second;
        break;
      }
    }
  }

  if (const auto& rdms_it = k_rdms_column_type_prepend.find(rdms);
      rdms_it != k_rdms_column_type_prepend.end()) {
    for (const auto& prepend : rdms_it->second) {
      if (boost::regex_match(col_type, prepend.first)) {
        col_type = prepend.second + col_type;
        break;
      }
    }
  }

  return col_type;
}

}  // namespace

/**
 * Helper class for asserting equality between a result set represented as a boost variant
 * and a thrift result set (TRowSet).
 */
class AssertValueEqualsVisitor : public boost::static_visitor<> {
 public:
  AssertValueEqualsVisitor(const TDatum& datum,
                           const TColumnType& column_type,
                           const size_t row,
                           const size_t column)
      : datum_(datum), column_type_(column_type), row_(row), column_(column) {}

  template <typename T>
  void operator()(const T& value) const {
    throw std::runtime_error{"Unexpected type used in test assertion. Type id: "s +
                             typeid(value).name()};
  }

  void operator()(const int64_t value) const {
    EXPECT_EQ(datum_.val.int_val, value)
        << boost::format("At row: %d, column: %d") % row_ % column_;
  }

  void operator()(const double value) const {
    EXPECT_DOUBLE_EQ(datum_.val.real_val, value)
        << boost::format("At row: %d, column: %d") % row_ % column_;
  }

  void operator()(const float value) const {
    EXPECT_FLOAT_EQ(datum_.val.real_val, value)
        << boost::format("At row: %d, column: %d") % row_ % column_;
  }

  void operator()(const std::string& value) const {
    auto str_value = datum_.val.str_val;
    EXPECT_TRUE(!datum_.is_null)
        << boost::format("At row: %d, column: %d") % row_ % column_;
    auto type = column_type_.col_type.type;
    if (isGeo(type) && !datum_.val.arr_val.empty()) {
      throw std::runtime_error{
          "Test assertions on non-WKT Geospatial data type projections are currently not "
          "supported"};
    } else if (isDateOrTime(type)) {
      auto type_info = SQLTypeInfo(getDatetimeSqlType(type),
                                   column_type_.col_type.precision,
                                   column_type_.col_type.scale);
      auto datetime_datum = StringToDatum(value, type_info);
      EXPECT_EQ(datetime_datum.bigintval, datum_.val.int_val)
          << boost::format("At row: %d, column: %d") % row_ % column_;
    } else {
      EXPECT_EQ(str_value, value)
          << boost::format("At row: %d, column: %d") % row_ % column_;
    }
  }

  void operator()(const ScalarTargetValue& value) const {
    boost::apply_visitor(AssertValueEqualsVisitor{datum_, column_type_, row_, column_},
                         value);
  }

  void operator()(const NullableString& value) const {
    if (value.which() == 0) {
      boost::apply_visitor(AssertValueEqualsVisitor{datum_, column_type_, row_, column_},
                           value);
    } else {
      EXPECT_TRUE(datum_.is_null)
          << boost::format("At row: %d, column: %d") % row_ % column_;
    }
  }

  void operator()(const ArrayTargetValue& values_optional) const {
    const auto& values = values_optional.get();
    ASSERT_EQ(values.size(), datum_.val.arr_val.size());
    for (size_t i = 0; i < values.size(); i++) {
      boost::apply_visitor(
          AssertValueEqualsVisitor{datum_.val.arr_val[i], column_type_, row_, column_},
          values[i]);
    }
  }

 private:
  bool isGeo(const TDatumType::type type) const {
    return (type == TDatumType::type::POINT || type == TDatumType::type::MULTIPOINT ||
            type == TDatumType::type::LINESTRING ||
            type == TDatumType::type::MULTILINESTRING ||
            type == TDatumType::type::POLYGON || type == TDatumType::type::MULTIPOLYGON);
  }

  bool isDateOrTime(const TDatumType::type type) const {
    return (type == TDatumType::type::TIME || type == TDatumType::type::TIMESTAMP ||
            type == TDatumType::type::DATE);
  }

  SQLTypes getDatetimeSqlType(const TDatumType::type type) const {
    if (type == TDatumType::type::TIME) {
      return kTIME;
    }
    if (type == TDatumType::type::TIMESTAMP) {
      return kTIMESTAMP;
    }
    if (type == TDatumType::type::DATE) {
      return kDATE;
    }
    throw std::runtime_error{"Unexpected type TDatumType::type : " +
                             std::to_string(type)};
  }

  const TDatum& datum_;
  const TColumnType& column_type_;
  const size_t row_;
  const size_t column_;
};

class AssertValueEqualsOrIsNullVisitor : public boost::static_visitor<> {
 public:
  AssertValueEqualsOrIsNullVisitor(const TDatum& datum,
                                   const TColumnType& column_type,
                                   const size_t row,
                                   const size_t column)
      : datum_(datum), column_type_(column_type), row_(row), column_(column) {}

  void operator()(const TargetValue& value) const {
    boost::apply_visitor(AssertValueEqualsVisitor{datum_, column_type_, row_, column_},
                         value);
  }

  void operator()(const void* null) const {
    EXPECT_TRUE(datum_.is_null)
        << boost::format("At row: %d, column: %d") % row_ % column_;
  }

  const TDatum& datum_;
  const TColumnType& column_type_;
  const size_t row_;
  const size_t column_;
};

/**
 * Helper gtest fixture class for executing SQL queries through DBHandler
 * and asserting result sets.
 */
class DBHandlerTestFixture : public testing::Test {
 public:
  static po::variables_map initTestArgs(int argc,
                                        char** argv,
                                        po::options_description& desc) {
    // Default options.  Addional options can be passed in as parameter.
    desc.add_options()("cluster",
                       po::value<std::string>(&cluster_config_file_path_),
                       "Path to data leaves list JSON file.");
    desc.add_options()("use-disk-cache", "Enable disk cache for all tables.");
    po::variables_map vm;
    po::store(
        po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);
    return vm;
  }

  static po::variables_map initTestArgs(int argc, char** argv) {
    po::options_description desc("Options");
    return initTestArgs(argc, argv, desc);
  }

  static void initTestArgs(const std::vector<LeafHostInfo>& string_servers,
                           const std::vector<LeafHostInfo>& leaf_servers) {
    string_leaves_ = string_servers;
    db_leaves_ = leaf_servers;
  }

  static bool isOdbc(const std::string& data_wrapper_type) {
    static const std::vector<std::string> odbc_wrappers{
        "sqlite", "postgres", "redshift", "snowflake", "bigquery"};
    return std::find(odbc_wrappers.begin(), odbc_wrappers.end(), data_wrapper_type) !=
           odbc_wrappers.end();
  }

  static bool isFileBased(const std::string& data_wrapper_type) {
    static const std::vector<std::string> file_based_wrappers{
        "regex_parser", "csv", "parquet"};
    return std::find(file_based_wrappers.begin(),
                     file_based_wrappers.end(),
                     data_wrapper_type) != file_based_wrappers.end();
  }

  static std::string getOdbcTableName(const std::string& table_name,
                                      const std::string& data_wrapper_type) {
    CHECK(false);
    return "";
  }
  // TODO(Misiu): Move all the visitor stuff to TestHelpers.h.  No need for it to be tied
  // to DBHandler.
  static void assertResultSetEqual(
      const std::vector<std::vector<NullableTargetValue>>& expected_result_set,
      const TQueryResult actual_result) {
    auto& row_set = actual_result.row_set;
    auto row_count = getRowCount(row_set);
    ASSERT_EQ(expected_result_set.size(), row_count)
        << "Returned result set does not have the expected number of rows";

    if (row_count == 0) {
      return;
    }

    auto expected_column_count = expected_result_set[0].size();
    size_t column_count = getColumnCount(row_set);
    ASSERT_EQ(expected_column_count, column_count)
        << "Returned result set does not have the expected number of columns";

    for (size_t r = 0; r < row_count; r++) {
      auto row = getRow(row_set, r);
      for (size_t c = 0; c < column_count; c++) {
        auto column_value = row[c];
        auto expected_column_value = expected_result_set[r][c];
        boost::apply_visitor(
            AssertValueEqualsOrIsNullVisitor{column_value, row_set.row_desc[c], r, c},
            expected_column_value);
      }
    }
  }

  static void assertExceptionMessage(const TDBException& e,
                                     const std::string& error_message,
                                     bool i_case = false) {
    std::string actual_err = e.error_msg;
    std::string expected_err = error_message;
    if (i_case) {
      boost::algorithm::to_lower(actual_err);
      boost::algorithm::to_lower(expected_err);
    }

    if (isDistributedMode()) {
      // In distributed mode, exception messages may be wrapped within
      // another thrift exception. In this case, do a substring check.
      if (actual_err.find(expected_err) == std::string::npos) {
        std::cerr << "recieved message: " << e.error_msg << "\n";
        std::cerr << "expected message: " << error_message << "\n";
      }
      ASSERT_TRUE(actual_err.find(expected_err) != std::string::npos);
    } else {
      ASSERT_EQ(expected_err, actual_err);
    }
  }

  static void assertExceptionMessage(const std::runtime_error& e,
                                     const std::string& error_message,
                                     bool i_case = false) {
    std::string actual_err = e.what();
    std::string expected_err = error_message;
    if (i_case) {
      boost::algorithm::to_lower(actual_err);
      boost::algorithm::to_lower(expected_err);
    }
    ASSERT_EQ(expected_err, actual_err);
  }

  static void assertExceptionMessage(const std::exception& e,
                                     const std::string& error_message,
                                     bool i_case = false) {
    std::string actual_err = e.what();
    std::string expected_err = error_message;
    if (i_case) {
      boost::algorithm::to_lower(actual_err);
      boost::algorithm::to_lower(expected_err);
    }
    ASSERT_EQ(expected_err, actual_err);
  }

  static void SetUpTestSuite() {}

  static void TearDownTestSuite() {}

  static void createDBHandler() {
    if (!db_handler_) {
      // Whitelist root path for tests by default
      ddl_utils::FilePathWhitelist::clear();
      ddl_utils::FilePathWhitelist::initialize(BASE_PATH, "[\"/\"]", "[\"/\"]");

      // Based on default values observed from starting up an OmniSci DB server.
      const bool allow_multifrag{true};
      const bool jit_debug{false};
      const bool intel_jit_profile{false};
      const bool allow_loop_joins{false};
      const bool enable_rendering{false};
      const bool renderer_prefer_igpu{false};
      const unsigned renderer_vulkan_timeout_ms{300000};
      const bool renderer_use_parallel_executors{false};
      const bool enable_auto_clear_render_mem{false};
      const int render_oom_retry_threshold{0};
      const size_t render_mem_bytes{500000000};
      const size_t max_concurrent_render_sessions{500};
      const bool render_compositor_use_last_gpu{false};
      const size_t reserved_gpu_mem{134217728};
      const size_t num_reader_threads{0};
      const bool legacy_syntax{true};
      const int idle_session_duration{60};
      const int max_session_duration{43200};
      system_parameters_.runtime_udf_registration_policy =
          SystemParameters::RuntimeUdfRegistrationPolicy::DISALLOWED;
      system_parameters_.omnisci_server_port = -1;
      system_parameters_.calcite_port = 3280;

      File_Namespace::DiskCacheConfig disk_cache_config{
          File_Namespace::DiskCacheConfig::getDefaultPath(std::string(BASE_PATH)),
          disk_cache_level_};

      db_handler_ = std::make_unique<DBHandler>(db_leaves_,
                                                string_leaves_,
                                                BASE_PATH,
                                                allow_multifrag,
                                                jit_debug,
                                                intel_jit_profile,
                                                g_read_only,
                                                allow_loop_joins,
                                                enable_rendering,
                                                renderer_prefer_igpu,
                                                renderer_vulkan_timeout_ms,
                                                renderer_use_parallel_executors,
                                                enable_auto_clear_render_mem,
                                                render_oom_retry_threshold,
                                                render_mem_bytes,
                                                max_concurrent_render_sessions,
                                                reserved_gpu_mem,
                                                render_compositor_use_last_gpu,
                                                num_reader_threads,
                                                auth_metadata_,
                                                system_parameters_,
                                                legacy_syntax,
                                                idle_session_duration,
                                                max_session_duration,
                                                udf_filename_,
                                                udf_compiler_path_,
                                                udf_compiler_options_,
#ifdef ENABLE_GEOS
                                                libgeos_so_filename_,
#endif
                                                disk_cache_config,
                                                false);
      loginAdmin();

      // Execute on CPU by default
      db_handler_->set_execution_mode(session_id_, TExecuteMode::CPU);
    }
  }

  static void destroyDBHandler() { db_handler_.reset(); }

 protected:
  friend class DBHandlerTestEnvironment;

  void SetUp() override { switchToAdmin(); }

  void TearDown() override {}

  static void sql(const std::string& query) {
    TQueryResult result;
    sql(result, query);
  }

  static TImportStatus getImportStatus(const std::string& import_id) {
    TImportStatus import_status;
    db_handler_->import_table_status(import_status, session_id_, import_id);
    return import_status;
  }

  static void sql(TQueryResult& result, const std::string& query) {
    db_handler_->sql_execute(
        result, session_id_, boost::trim_copy(query), true, "", -1, -1);
  }

  // Execute SQL with session_id
  static void sql(TQueryResult& result,
                  const std::string& query,
                  const TSessionId& sess_id) {
    db_handler_->sql_execute(result, sess_id, boost::trim_copy(query), true, "", -1, -1);
  }

  Catalog_Namespace::UserMetadata getCurrentUser() {
    return db_handler_->get_session_copy(session_id_).get_currentUser();
  }

  static Catalog_Namespace::Catalog& getCatalog() {
    return db_handler_->get_session_copy(session_id_).getCatalog();
  }

  static std::pair<DBHandler*, TSessionId&> getDbHandlerAndSessionId() {
    return {db_handler_.get(), session_id_};
  }

  static void resetCatalog() {
    auto& catalog = getCatalog();
    Catalog_Namespace::SysCatalog::instance().removeCatalog(
        catalog.getCurrentDB().dbName);
  }

  static void loginAdmin() {
    session_id_ = {};
    login(default_user_, "HyperInteractive", default_db_name_, session_id_);
    admin_session_id_ = session_id_;
  }
  static bool isDistributedMode() { return system_parameters_.aggregator; }
  static SystemParameters getSystemParameters() { return system_parameters_; }
  static void switchToAdmin() { session_id_ = admin_session_id_; }

  static void logout(const TSessionId& id) { db_handler_->disconnect(id); }

  static void login(const std::string& user,
                    const std::string& pass,
                    const std::string& db_name = default_db_name_) {
    session_id_ = {};
    login(user, pass, db_name, session_id_);
  }

  // Login and return the session id to logout later
  static void login(const std::string& user,
                    const std::string& pass,
                    const std::string& db,
                    TSessionId& result_id) {
    if (isDistributedMode()) {
      // Need to do full login here for distributed tests
      db_handler_->connect(result_id, user, pass, db);
    } else {
      db_handler_->internal_connect(result_id, user, db);
    }
  }

  static void setSessionId(const std::string& session_id) { session_id_ = session_id; }

  static std::vector<ColumnPair> schema_string_to_column_pairs(
      const std::string& schema) {
    auto schema_list = split_on_regex(schema, ",\\s+");
    std::vector<ColumnPair> result;
    for (const auto& token : schema_list) {
      auto tokens = split_on_regex(token, "\\s+");
      if (tokens[0] == "shard" &&
          tokens[1].substr(0, 3) == "key") {  // skip `shard key` specifier
        continue;
      }
      CHECK(tokens.size() >= 2);
      result.push_back({tokens[0], tokens[1]});
    }
    return result;
  }

  static std::string column_pairs_to_schema_string(
      const std::vector<ColumnPair>& column_pairs) {
    std::stringstream ss;
    for (size_t i = 0; i < column_pairs.size(); i++) {
      const auto& [col_name, col_type] = column_pairs[i];
      ss << col_name << " " << col_type;
      if (i < column_pairs.size() - 1) {
        ss << ", ";
      }
    }
    return ss.str();
  }

  static std::vector<ColumnPair> get_column_pairs_for_rdms(
      const std::vector<ColumnPair>& column_pairs,
      const std::string& rdms) {
    std::vector<ColumnPair> result;
    for (auto [col_name, col_type] : column_pairs) {
      result.emplace_back(col_name, get_col_type_for_rdms(col_type, rdms));
    }
    return result;
  }

  static void createODBCSourceTable(const std::string& table_name,
                                    const std::vector<ColumnPair>& column_pairs,
                                    const std::string& src_file,
                                    const std::string& data_wrapper_type,
                                    const bool is_odbc_geo = false) {}

  static const std::vector<LeafHostInfo>& getDbLeaves() { return db_leaves_; }

  template <typename Lambda>
  void executeLambdaAndAssertException(Lambda lambda,
                                       const std::string& error_message,
                                       const bool i_case = false) {
    try {
      lambda();
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const TDBException& e) {
      assertExceptionMessage(e, error_message, i_case);
    } catch (const std::runtime_error& e) {
      assertExceptionMessage(e, error_message, i_case);
    }
  }

  // sometime error message have non deterministic portions
  // used to check a meaningful portion of an error message
  template <typename Lambda>
  void executeLambdaAndAssertPartialException(Lambda lambda,
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

  void assertPartialExceptionMessage(const TDBException& e,
                                     const std::string& error_message) {
    ASSERT_TRUE(e.error_msg.find(error_message) != std::string::npos);
  }

  void assertPartialExceptionMessage(const std::runtime_error& e,
                                     const std::string& error_message) {
    ASSERT_TRUE(std::string(e.what()).find(error_message) != std::string::npos);
  }

  void queryAndAssertException(const std::string& sql_statement,
                               const std::string& error_message,
                               const bool i_case = false) {
    executeLambdaAndAssertException([&] { sql(sql_statement); }, error_message, i_case);
  }

  void queryAndAssertExceptionWithParam(
      const std::string& sql_statement,
      const std::string& key,
      const std::map<std::string, std::string>& error_message_map) {
    auto error_message_pair_it = error_message_map.find(key);
    ASSERT_TRUE(error_message_pair_it != error_message_map.end());
    auto& error_message = error_message_pair_it->second;
    queryAndAssertException(sql_statement, error_message);
  }

  void queryAndAssertPartialException(const std::string& sql_statement,
                                      const std::string& error_message) {
    executeLambdaAndAssertPartialException([&] { sql(sql_statement); }, error_message);
  }

  void sqlAndCompareResult(
      const std::string& sql_statement,
      const std::vector<std::vector<NullableTargetValue>>& expected_result_set) {
    TQueryResult result_set;
    sql(result_set, sql_statement);
    assertResultSetEqual(expected_result_set, result_set);
  }

  /**
   * Helper method used to cast a vector of scalars to an optional of the same object.
   */
  boost::optional<std::vector<ScalarTargetValue>> array(
      std::vector<ScalarTargetValue> array) {
    return array;
  }

  /**
   * Helper method used to cast an integer literal to an int64_t (in order to
   * avoid compiler ambiguity).
   */
  constexpr int64_t i(int64_t i) { return i; }

  bool setExecuteMode(const TExecuteMode::type mode) {
    if (db_handler_->cpu_mode_only_ && TExecuteMode::GPU) {
      return false;
    }
    db_handler_->set_execution_mode(session_id_, mode);
    return true;
  }

  TExecuteMode::type getExecuteMode() {
    return db_handler_->getExecutionMode(session_id_);
  }

  void resizeDispatchQueue(size_t queue_size) {
    db_handler_->resizeDispatchQueue(queue_size);
  }

  size_t getRowCount(const TQueryResult& result) { return getRowCount(result.row_set); }

 private:
  static size_t getRowCount(const TRowSet& row_set) {
    size_t row_count;
    if (row_set.is_columnar) {
      row_count = row_set.columns.empty() ? 0 : row_set.columns[0].nulls.size();
    } else {
      row_count = row_set.rows.size();
    }
    return row_count;
  }

  static size_t getColumnCount(const TRowSet& row_set) {
    size_t column_count;
    if (row_set.is_columnar) {
      column_count = row_set.columns.size();
    } else {
      column_count = row_set.rows.empty() ? 0 : row_set.rows[0].cols.size();
    }
    return column_count;
  }

  static void setDatumArray(std::vector<TDatum>& datum_array, const TColumn& column) {
    const auto& column_data = column.data;
    if (!column_data.int_col.empty()) {
      for (auto& item : column_data.int_col) {
        TDatum datum_item{};
        datum_item.val.int_val = item;
        datum_array.emplace_back(datum_item);
      }
    } else if (!column_data.real_col.empty()) {
      for (auto& item : column_data.real_col) {
        TDatum datum_item{};
        datum_item.val.real_val = item;
        datum_array.emplace_back(datum_item);
      }
    } else if (!column_data.str_col.empty()) {
      for (auto& item : column_data.str_col) {
        TDatum datum_item{};
        datum_item.val.str_val = item;
        datum_array.emplace_back(datum_item);
      }
    } else {
      // no-op: it is possible for the array to be empty
    }
    const auto& nulls = column.nulls;
    CHECK(nulls.size() == datum_array.size())
        << "mismatch of size between null data array and data read from array.";
    for (size_t i = 0; i < nulls.size(); ++i) {
      datum_array[i].is_null = nulls[i];
    }
  }

  static void setDatum(TDatum& datum,
                       const TColumnData& column_data,
                       const size_t index,
                       const bool is_null) {
    if (!column_data.int_col.empty()) {
      datum.val.int_val = column_data.int_col[index];
    } else if (!column_data.real_col.empty()) {
      datum.val.real_val = column_data.real_col[index];
    } else if (!column_data.str_col.empty()) {
      datum.val.str_val = column_data.str_col[index];
    } else if (!column_data.arr_col.empty()) {
      std::vector<TDatum> datum_array{};
      if (!is_null) {
        setDatumArray(datum_array, column_data.arr_col[index]);
      }
      datum.val.arr_val = datum_array;
    } else {
      throw std::runtime_error{"Unexpected column data"};
    }
  }

  static std::vector<TDatum> getRow(const TRowSet& row_set, const size_t index) {
    if (row_set.is_columnar) {
      std::vector<TDatum> row{};
      for (auto& column : row_set.columns) {
        TDatum datum{};
        auto is_null = column.nulls[index];
        setDatum(datum, column.data, index, is_null);
        if (is_null) {
          datum.is_null = true;
        }
        row.emplace_back(datum);
      }
      return row;
    } else {
      return row_set.rows[index].cols;
    }
  }

  static std::unique_ptr<DBHandler> db_handler_;
  static TSessionId session_id_;
  static TSessionId admin_session_id_;
  static std::vector<LeafHostInfo> db_leaves_;
  static std::vector<LeafHostInfo> string_leaves_;
  static AuthMetadata auth_metadata_;
  static SystemParameters system_parameters_;
  static std::string udf_filename_;
  static std::string udf_compiler_path_;
  static std::string default_user_;
  static std::string default_pass_;
  static std::vector<std::string> udf_compiler_options_;
#ifdef ENABLE_GEOS
  static std::string libgeos_so_filename_;
#endif

 public:
  static void setupSignalHandler() {
    TestProcessSignalHandler::registerSignalHandler();
    TestProcessSignalHandler::addShutdownCallback([]() {
      if (db_handler_) {
        db_handler_->shutdown();
      }
    });
  }

  static std::string default_db_name_;

  static std::string cluster_config_file_path_;
  static File_Namespace::DiskCacheLevel disk_cache_level_;
};

// https://google.github.io/googletest/advanced.html#global-set-up-and-tear-down
class DBHandlerTestEnvironment : public ::testing::Environment {
 public:
  ~DBHandlerTestEnvironment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override {
    DBHandlerTestFixture::setupSignalHandler();
    DBHandlerTestFixture::createDBHandler();
  }

  // Override this to define how to tear down the environment.
  void TearDown() override { DBHandlerTestFixture::destroyDBHandler(); }
};

TSessionId DBHandlerTestFixture::session_id_{};
TSessionId DBHandlerTestFixture::admin_session_id_{};
std::unique_ptr<DBHandler> DBHandlerTestFixture::db_handler_ = nullptr;
std::vector<LeafHostInfo> DBHandlerTestFixture::db_leaves_{};
std::vector<LeafHostInfo> DBHandlerTestFixture::string_leaves_{};
AuthMetadata DBHandlerTestFixture::auth_metadata_{};
std::string DBHandlerTestFixture::udf_filename_{};
std::string DBHandlerTestFixture::udf_compiler_path_{};
std::string DBHandlerTestFixture::default_user_{"admin"};
std::string DBHandlerTestFixture::default_pass_{"HyperInteractive"};
std::string DBHandlerTestFixture::default_db_name_{};
SystemParameters DBHandlerTestFixture::system_parameters_{};
std::vector<std::string> DBHandlerTestFixture::udf_compiler_options_{};
std::string DBHandlerTestFixture::cluster_config_file_path_{};
#ifdef ENABLE_GEOS
std::string DBHandlerTestFixture::libgeos_so_filename_{};
#endif
File_Namespace::DiskCacheLevel DBHandlerTestFixture::disk_cache_level_{
    File_Namespace::DiskCacheLevel::fsi};
