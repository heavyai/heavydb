/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

#include "Catalog/Catalog.h"
#ifdef EE_FSI_ODBC
#include "DataMgr/ForeignStorage/ODBC/OdbcDataWrapper.h"
#include "DataMgr/ForeignStorage/ODBC/odbc_utils.h"
#endif  // EE_FSI_ODBC
#include <boost/regex.hpp>
#include "QueryRunner/TestProcessSignalHandler.h"
#include "TestHelpers.h"
#include "ThriftHandler/DBHandler.h"
// TODO: remove OdbcFsiTestHelper.h dependency once all ODBC related functionality has
// been moved out of here and into OdbcFsiTestHelper
#include "OdbcFsiTestHelper.h"

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

extern bool g_enable_system_tables;
extern bool g_read_only;

namespace {

std::vector<std::string> split_on_regex(const std::string& in, const std::string& regex) {
  std::vector<std::string> tokens;
  boost::split_regex(tokens, in, boost::regex{regex});
  return tokens;
}

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
    if (boost::get<std::string>(&value)) {
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

template <typename T, typename... Ts>
constexpr bool is_any() {
  return ((std::is_same_v<T, Ts> || ...));
}

class ToStringVisitor : public boost::static_visitor<std::string> {
 public:
  template <typename T>
  std::string operator()(const T& value) const {
    if constexpr (is_any<T, GeoTargetValue, GeoTargetValuePtr>()) {
      throw std::runtime_error(
          "Geo types are currently not supported by ToStringVisitor");
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

  std::string operator()(const void* null) const { return "NULL"; }
};

/**
 * Helper gtest fixture class for executing SQL queries through DBHandler
 * and asserting result sets.
 */
class DBHandlerTestFixture : public TestHelpers::TbbPrivateServerKiller {
 public:
  static po::variables_map initTestArgs(int argc,
                                        char** argv,
                                        po::options_description& desc) {
    // Default options.  Addional options can be passed in as parameter.
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

  static bool isFileBased(const std::string& data_wrapper_type) {
    static const std::vector<std::string> file_based_wrappers{
        "regex_parser", "csv", "parquet"};
    return std::find(file_based_wrappers.begin(),
                     file_based_wrappers.end(),
                     data_wrapper_type) != file_based_wrappers.end();
  }

  static std::string getOdbcTableName(const std::string& table_name,
                                      const std::string& data_wrapper_type) {
#ifdef EE_FSI_ODBC
    if (data_wrapper_type == "postgres" || data_wrapper_type == "redshift" ||
        data_wrapper_type == "hive") {
      auto schema_name = getOdbcSchemaName(data_wrapper_type);
      return schema_name + "." + table_name;
    }
    if (data_wrapper_type == "snowflake") {
      auto schema_name = getOdbcSchemaName(data_wrapper_type);
      std::string database_name = "odbc_fsi_test";
      return database_name + "." + schema_name + "." + table_name;
    }
    if (data_wrapper_type == "bigquery") {
      auto schema_name = getOdbcSchemaName(data_wrapper_type);
      auto project_name = std::string(std::getenv("bigquery_project"));
      CHECK(project_name.size());
      return project_name + "." + schema_name + "." + table_name;
    }
    return table_name;
#else
    CHECK(false);
    return "";
#endif
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

    ASSERT_EQ(expected_err, actual_err);
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
      const bool renderer_enable_slab_allocation{false};
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

      db_handler_ = std::make_unique<DBHandler>(BASE_PATH,
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
                                                renderer_enable_slab_allocation,
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
#ifdef HAVE_TORCH_TFS
                                                torch_lib_path_,
#endif
                                                disk_cache_config,
                                                false);
      loginAdmin();

      // Execute on CPU by default
      db_handler_->set_execution_mode(session_id_, TExecuteMode::CPU);
    }
  }

  static void destroyDBHandler() {
    db_handler_.reset();
  }

  template <typename Lambda>
  static void executeLambdaAndAssertException(Lambda lambda,
                                              const std::string& error_message,
                                              const bool i_case = false) {
    try {
      lambda();
      FAIL() << "An exception should have been thrown for this test case, exception "
                "expected: "
             << error_message;
    } catch (const TDBException& e) {
      assertExceptionMessage(e, error_message, i_case);
    } catch (const std::runtime_error& e) {
      assertExceptionMessage(e, error_message, i_case);
    }
  }

  // sometime error message have non deterministic portions
  // used to check a meaningful portion of an error message
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

 protected:
  friend class DBHandlerTestEnvironment;

  void SetUp() override {
    TbbPrivateServerKiller::SetUp();
    switchToAdmin();
  }

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
  static SystemParameters getSystemParameters() {
    return system_parameters_;
  }
  static void switchToAdmin() {
    session_id_ = admin_session_id_;
  }

  static void logout(const TSessionId& id) {
    db_handler_->disconnect(id);
  }

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
    db_handler_->internal_connect(result_id, user, db);
  }

  static void setSessionId(const std::string& session_id) {
    session_id_ = session_id;
  }

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

#ifdef EE_FSI_ODBC
  static std::pair<std::string, std::string> getODBCCredentials(
      const std::string& data_wrapper_type) {
    CHECK(data_wrapper_type.size());
    static const std::map<std::string, std::pair<std::string, std::string>>
        odbc_credentials_environment{{"redshift",
                                      { "redshift_username",
                                        "redshift_password" }},
                                     {"snowflake",
                                      { "snowflake_username",
                                        "snowflake_password" }},
                                     {"postgres",
                                      { "postgres_username",
                                        "postgres_password" }},
                                     { "bigquery",
                                       { "bigquery_username",
                                         "bigquery_password" } }};

    if (auto it = odbc_credentials_environment.find(data_wrapper_type);
        it != odbc_credentials_environment.end()) {
      auto [username_environment, password_environment] = it->second;
      auto username_ptr = std::getenv(username_environment.c_str());
      auto password_ptr = std::getenv(password_environment.c_str());
      if (!username_ptr || !password_ptr) {
        return {"", ""};
      }
      return {std::string(username_ptr), std::string(password_ptr)};
    }
    return {"admin", "HyperInteractive"};
  }

  static auto getODBCCredentialString(const std::string& data_wrapper_type) {
    auto [username, password] = getODBCCredentials(data_wrapper_type);
    std::string credential_string = "Username=" + username + ";Password=" + password;
    return credential_string;
  }

  static auto createODBCConnection(const std::string& data_wrapper_type) {
    auto [username, password] = getODBCCredentials(data_wrapper_type);
    foreign_storage::UserMapping user_mapping{};
    user_mapping.setOptions({{foreign_storage::OdbcDataWrapper::ODBC_USERNAME, username},
                             { foreign_storage::OdbcDataWrapper::ODBC_PASSWORD,
                               password }});
    return foreign_storage::OdbcConnection::create({data_wrapper_type, std::nullopt},
                                                   &user_mapping);
  }

  static std::string getOdbcSchemaName(const std::string& data_wrapper_type) {
    CHECK(data_wrapper_type != "sqlite");
    auto [schema_name, _] = getODBCCredentials(data_wrapper_type);
    if (std::getenv("uuid")) {
      schema_name += "_" + std::string(std::getenv("uuid"));
    }
    boost::regex dot_or_hyphen("(\\.|-)");
    schema_name = boost::regex_replace(schema_name, dot_or_hyphen, "_");
    if (schema_name.empty()) {
      return "odbc_fsi_test";
    }
    return schema_name;
  }

  static void dropODBCSchema(const std::string& data_wrapper_type) {
    if (data_wrapper_type == "sqlite") {
      return;
    }
    auto odbc_connection = createODBCConnection(data_wrapper_type);
    auto schema_name = getOdbcSchemaName(data_wrapper_type);
    try {
      odbc_connection->runSqlAllowSuccessWithInfo("DROP SCHEMA IF EXISTS " + schema_name +
                                                  " CASCADE;");
    } catch (foreign_storage::ForeignStorageException& fes) {
      std::string msg = fes.what();
      boost::regex e_msg("(drop cascades to |schema (\"" + schema_name +
                             "\"|'odbc_fsi_test." + schema_name + "') does not exist)",
                         boost::regex::icase);
      if (!boost::regex_search(msg, e_msg)) {
        throw fes;
      }
    }
  }

  static void createODBCSchema(const std::string& data_wrapper_type) {
    if (data_wrapper_type == "sqlite") {
      return;
    }
    dropODBCSchema(data_wrapper_type);
    auto odbc_connection = createODBCConnection(data_wrapper_type);
    auto schema_name = getOdbcSchemaName(data_wrapper_type);
    odbc_connection->runSqlAllowSuccessWithInfo("CREATE SCHEMA " + schema_name + ";");
  }
#endif

  static void createODBCSourceTable(const std::string& table_name,
                                    const std::vector<ColumnPair>& column_pairs,
                                    const std::string& src_file,
                                    const std::string& data_wrapper_type) {
#ifdef EE_FSI_ODBC
    // Import a csv file into an odbc table.
    auto odbc_connection = createODBCConnection(data_wrapper_type);
    auto schema_name_table_name = getOdbcTableName(table_name, data_wrapper_type);
    try {
      odbc_connection->runSqlAllowSuccessWithInfo("drop table if exists " +
                                                  schema_name_table_name + ";");
    } catch (foreign_storage::ForeignStorageException& fes) {
      std::string msg = fes.what();
      boost::regex e_msg("table \"("s + schema_name_table_name + "|" + table_name +
                             ")\" does not exist"s,
                         boost::regex::icase);
      if (!boost::regex_search(msg, e_msg)) {
        throw fes;
      }
    }

    auto rdms_specific_column_pairs =
        get_column_pairs_for_rdms(column_pairs, data_wrapper_type);
    auto rdms_specific_schema = column_pairs_to_schema_string(rdms_specific_column_pairs);
    odbc_connection->runSqlAllowSuccessWithInfo("create table " + schema_name_table_name +
                                                " (" + rdms_specific_schema + ");");

    // Check for the availability of the table created in the remote database before
    // proceeding
    std::string last_exception_message;
    size_t num_tries = 200;
    while (num_tries > 0) {
      num_tries--;
      try {
        odbc_connection->runSql("SELECT * FROM " + schema_name_table_name + ";");
      } catch (foreign_storage::ForeignStorageException& fes) {
        last_exception_message = fes.what();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }
      break;
    }
    if (!num_tries) {
      ASSERT_TRUE(false) << "Timed out while attempting to poll for the foreign table \""
                         << schema_name_table_name
                         << "\", the last exception message was: "
                         << last_exception_message;
    }

    if (data_wrapper_type == "postgres") {
      odbc_connection->runSqlAllowSuccessWithInfo("SET datestyle TO \"SQL, MDY\";");
    }
    std::ifstream csv_file(src_file);
    std::stringstream insert_records;
    uint32_t line_num = 0;
    for (std::string line; std::getline(csv_file, line);) {
      if (line_num++ == 0) {
        // skip header
        continue;
      }
      apply_mods_to_insert_record(line, rdms_specific_column_pairs, data_wrapper_type);
      insert_records << (line_num > 2 ? ", (" : "(") << line << ")";
    }

    auto insert_records_str = insert_records.str();
    if (!insert_records_str.empty()) {
      odbc_connection->runSqlAllowSuccessWithInfo("insert into " +
                                                  schema_name_table_name + " values " +
                                                  insert_records_str + ";");
    }
#endif
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
  constexpr int64_t i(int64_t i) {
    return i;
  }

  bool setExecuteMode(const TExecuteMode::type mode) {
    if (db_handler_->cpu_mode_only_ && mode == TExecuteMode::GPU) {
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

  size_t getRowCount(const TQueryResult& result) {
    return getRowCount(result.row_set);
  }

  size_t getColumnCount(const TQueryResult& result) {
    return getColumnCount(result.row_set);
  }

  std::vector<TDatum> getRow(const TQueryResult& result, const size_t index) {
    return getRow(result.row_set, index);
  }

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
  static AuthMetadata auth_metadata_;
  static std::string udf_filename_;
  static std::string udf_compiler_path_;
  static std::string default_user_;
  static std::string default_pass_;
  static std::vector<std::string> udf_compiler_options_;
#ifdef ENABLE_GEOS
  static std::string libgeos_so_filename_;
#endif
#ifdef HAVE_TORCH_TFS
  static std::string torch_lib_path_;
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
  static File_Namespace::DiskCacheLevel disk_cache_level_;
  static SystemParameters system_parameters_;
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
AuthMetadata DBHandlerTestFixture::auth_metadata_{};
std::string DBHandlerTestFixture::udf_filename_{};
std::string DBHandlerTestFixture::udf_compiler_path_{};
std::string DBHandlerTestFixture::default_user_{"admin"};
std::string DBHandlerTestFixture::default_pass_{"HyperInteractive"};
std::string DBHandlerTestFixture::default_db_name_{};
SystemParameters DBHandlerTestFixture::system_parameters_{};
std::vector<std::string> DBHandlerTestFixture::udf_compiler_options_{};
#ifdef ENABLE_GEOS
std::string DBHandlerTestFixture::libgeos_so_filename_{};
#endif
#ifdef HAVE_TORCH_TFS
std::string DBHandlerTestFixture::torch_lib_path_{};
#endif
File_Namespace::DiskCacheLevel DBHandlerTestFixture::disk_cache_level_{
    File_Namespace::DiskCacheLevel::fsi};
