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

#pragma once

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#include <gtest/gtest.h>
#include <boost/format.hpp>
#include <boost/optional.hpp>

#include "Catalog/Catalog.h"
#include "QueryRunner/TestProcessSignalHandler.h"
#include "ThriftHandler/DBHandler.h"

constexpr int64_t True = 1;
constexpr int64_t False = 0;
constexpr void* Null = nullptr;
constexpr int64_t Null_i = NULL_INT;

using NullableTargetValue = boost::variant<TargetValue, void*>;

extern size_t g_leaf_count;
extern bool g_cluster;

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
    return (type == TDatumType::type::POINT || type == TDatumType::type::LINESTRING ||
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
  static void initTestArgs(int argc, char** argv) {
    namespace po = boost::program_options;

    po::options_description desc("Options");
    desc.add_options()("cluster",
                       po::value<std::string>(&cluster_config_file_path_),
                       "Path to data leaves list JSON file.");
    desc.add_options()("use-disk-cache", "Enable disk cache for all tables.");
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);

    use_disk_cache_ = (vm.count("use-disk-cache"));
  }

  static void initTestArgs(const std::vector<LeafHostInfo>& string_servers,
                           const std::vector<LeafHostInfo>& leaf_servers) {
    string_leaves_ = string_servers;
    db_leaves_ = leaf_servers;
  }

 protected:
  void SetUp() override {
    createDBHandler();
    switchToAdmin();
  }

  static void SetUpTestSuite() { createDBHandler(); }

  static void TearDownTestSuite() {}

  static void createDBHandler() {
    if (!db_handler_) {
      setupSignalHandler();

      // Whitelist root path for tests by default
      ddl_utils::FilePathWhitelist::clear();
      ddl_utils::FilePathWhitelist::initialize(BASE_PATH, "[\"/\"]", "[\"/\"]");

      // Based on default values observed from starting up an OmniSci DB server.
      const bool allow_multifrag{true};
      const bool jit_debug{false};
      const bool intel_jit_profile{false};
      const bool read_only{false};
      const bool allow_loop_joins{false};
      const bool enable_rendering{false};
      const bool renderer_use_vulkan_driver{false};
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
      const bool enable_runtime_udf_registration{false};
      system_parameters_.omnisci_server_port = -1;
      system_parameters_.calcite_port = 3280;

      File_Namespace::DiskCacheLevel cache_level{File_Namespace::DiskCacheLevel::fsi};
      if (use_disk_cache_) {
        cache_level = File_Namespace::DiskCacheLevel::all;
      }
      File_Namespace::DiskCacheConfig disk_cache_config{
          File_Namespace::DiskCacheConfig::getDefaultPath(std::string(BASE_PATH)),
          cache_level};

      db_handler_ = std::make_unique<DBHandler>(db_leaves_,
                                                string_leaves_,
                                                BASE_PATH,
                                                allow_multifrag,
                                                jit_debug,
                                                intel_jit_profile,
                                                read_only,
                                                allow_loop_joins,
                                                enable_rendering,
                                                renderer_use_vulkan_driver,
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
                                                enable_runtime_udf_registration,
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
  void TearDown() override {}

  static void sql(const std::string& query) {
    TQueryResult result;
    sql(result, query);
  }

  static void sql(TQueryResult& result, const std::string& query) {
    db_handler_->sql_execute(
        result, session_id_, boost::trim_copy(query), true, "", -1, -1);
  }

  // Execute SQL with session_id
  static void sql(TQueryResult& result, const std::string& query, TSessionId& sess_id) {
    db_handler_->sql_execute(result, sess_id, boost::trim_copy(query), true, "", -1, -1);
  }

  Catalog_Namespace::UserMetadata getCurrentUser() {
    return db_handler_->get_session_copy_ptr(session_id_)->get_currentUser();
  }

  static Catalog_Namespace::Catalog& getCatalog() {
    return db_handler_->get_session_copy_ptr(session_id_)->getCatalog();
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

  static const std::vector<LeafHostInfo>& getDbLeaves() { return db_leaves_; }

  template <typename Lambda>
  void executeLambdaAndAssertException(Lambda lambda, const std::string& error_message) {
    try {
      lambda();
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const TOmniSciException& e) {
      assertExceptionMessage(e, error_message);
    } catch (const std::runtime_error& e) {
      assertExceptionMessage(e, error_message);
    }
  }

  void assertExceptionMessage(const TOmniSciException& e,
                              const std::string& error_message) {
    if (isDistributedMode()) {
      // In distributed mode, exception messages may be wrapped within
      // another thrift exception. In this case, do a substring check.
      ASSERT_TRUE(e.error_msg.find(error_message) != std::string::npos);
    } else {
      ASSERT_EQ(error_message, e.error_msg);
    }
  }

  void assertExceptionMessage(const std::runtime_error& e,
                              const std::string& error_message) {
    ASSERT_EQ(error_message, e.what());
  }

  // sometime error message have non deterministic portions
  // used to check a meaningful portion of an error message
  template <typename Lambda>
  void executeLambdaAndAssertPartialException(Lambda lambda,
                                              const std::string& error_message) {
    try {
      lambda();
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const TOmniSciException& e) {
      assertPartialExceptionMessage(e, error_message);
    } catch (const std::runtime_error& e) {
      assertPartialExceptionMessage(e, error_message);
    }
  }

  void assertPartialExceptionMessage(const TOmniSciException& e,
                                     const std::string& error_message) {
    ASSERT_TRUE(e.error_msg.find(error_message) != std::string::npos);
  }

  void assertPartialExceptionMessage(const std::runtime_error& e,
                                     const std::string& error_message) {
    ASSERT_TRUE(std::string(e.what()).find(error_message) != std::string::npos);
  }

  void queryAndAssertException(const std::string& sql_statement,
                               const std::string& error_message) {
    executeLambdaAndAssertException([&] { sql(sql_statement); }, error_message);
  }

  void queryAndAssertPartialException(const std::string& sql_statement,
                                      const std::string& error_message) {
    executeLambdaAndAssertPartialException([&] { sql(sql_statement); }, error_message);
  }

  void assertResultSetEqual(
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

  void resizeDispatchQueue(size_t queue_size) {
    db_handler_->resizeDispatchQueue(queue_size);
  }

  size_t getRowCount(const TQueryResult& result) { return getRowCount(result.row_set); }

 private:
  size_t getRowCount(const TRowSet& row_set) {
    size_t row_count;
    if (row_set.is_columnar) {
      row_count = row_set.columns.empty() ? 0 : row_set.columns[0].nulls.size();
    } else {
      row_count = row_set.rows.size();
    }
    return row_count;
  }

  size_t getColumnCount(const TRowSet& row_set) {
    size_t column_count;
    if (row_set.is_columnar) {
      column_count = row_set.columns.size();
    } else {
      column_count = row_set.rows.empty() ? 0 : row_set.rows[0].cols.size();
    }
    return column_count;
  }

  void setDatumArray(std::vector<TDatum>& datum_array, const TColumnData& column_data) {
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
  }

  void setDatum(TDatum& datum,
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
        setDatumArray(datum_array, column_data.arr_col[index].data);
      }
      datum.val.arr_val = datum_array;
    } else {
      throw std::runtime_error{"Unexpected column data"};
    }
  }

  std::vector<TDatum> getRow(const TRowSet& row_set, const size_t index) {
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

  static void setupSignalHandler() {
    TestProcessSignalHandler::registerSignalHandler();
    TestProcessSignalHandler::addShutdownCallback([]() {
      if (db_handler_) {
        db_handler_->shutdown();
      }
    });
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
  static std::string default_db_name_;
  static std::vector<std::string> udf_compiler_options_;
#ifdef ENABLE_GEOS
  static std::string libgeos_so_filename_;
#endif

 public:
  static std::string cluster_config_file_path_;
  static bool use_disk_cache_;
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
bool DBHandlerTestFixture::use_disk_cache_{false};
