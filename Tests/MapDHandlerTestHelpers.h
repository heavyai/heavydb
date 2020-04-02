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

#include <gtest/gtest.h>

#include "Catalog/Catalog.h"
#include "ThriftHandler/MapDHandler.h"

/**
 * Helper gtest fixture class for executing SQL queries through MapDHandler.
 */
class MapDHandlerTestFixture : public testing::Test {
 public:
  static void initTestArgs(int argc, char** argv) {
    namespace po = boost::program_options;

    po::options_description desc("Options");
    desc.add_options()("cluster",
                       po::value<std::string>(&cluster_config_file_path_),
                       "Path to data leaves list JSON file.");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
  }

 protected:
  virtual void SetUp() override {
    if (!mapd_handler_) {
      // Based on default values observed from starting up an OmniSci DB server.
      const bool cpu_only{false};
      const bool allow_multifrag{true};
      const bool jit_debug{false};
      const bool intel_jit_profile{false};
      const bool read_only{false};
      const bool allow_loop_joins{false};
      const bool enable_rendering{false};
      const bool enable_auto_clear_render_mem{false};
      const int render_oom_retry_threshold{0};
      const size_t render_mem_bytes{500000000};
      const size_t max_concurrent_render_sessions{500};
      const int num_gpus{-1};
      const int start_gpu{0};
      const size_t reserved_gpu_mem{134217728};
      const size_t num_reader_threads{0};
      const bool legacy_syntax{true};
      const int idle_session_duration{60};
      const int max_session_duration{43200};
      const bool enable_runtime_udf_registration{false};
      mapd_parameters_.omnisci_server_port = -1;
      mapd_parameters_.calcite_port = 3280;

      mapd_handler_ = std::make_unique<MapDHandler>(db_leaves_,
                                                    string_leaves_,
                                                    BASE_PATH,
                                                    cpu_only,
                                                    allow_multifrag,
                                                    jit_debug,
                                                    intel_jit_profile,
                                                    read_only,
                                                    allow_loop_joins,
                                                    enable_rendering,
                                                    enable_auto_clear_render_mem,
                                                    render_oom_retry_threshold,
                                                    render_mem_bytes,
                                                    max_concurrent_render_sessions,
                                                    num_gpus,
                                                    start_gpu,
                                                    reserved_gpu_mem,
                                                    num_reader_threads,
                                                    auth_metadata_,
                                                    mapd_parameters_,
                                                    legacy_syntax,
                                                    idle_session_duration,
                                                    max_session_duration,
                                                    enable_runtime_udf_registration,
                                                    udf_filename_,
                                                    udf_compiler_path_,
                                                    udf_compiler_options_);
    }
    loginAdmin();
  }

  virtual void TearDown() override { logoutAdmin(); }

  void sql(const std::string& query) {
    TQueryResult result;
    sql(result, query);
  }

  void sql(TQueryResult& result, const std::string& query) {
    mapd_handler_->sql_execute(result, session_id_, query, true, "", -1, -1);
  }

  // Execute SQL with session_id
  void sql(TQueryResult& result, const std::string& query, TSessionId& sess_id) {
    mapd_handler_->sql_execute(result, sess_id, query, true, "", -1, -1);
  }

  Catalog_Namespace::UserMetadata getCurrentUser() {
    return mapd_handler_->get_session_copy_ptr(session_id_)->get_currentUser();
  }

  Catalog_Namespace::Catalog& getCatalog() {
    return mapd_handler_->get_session_copy_ptr(session_id_)->getCatalog();
  }

  void resetCatalog() {
    auto& catalog = getCatalog();
    catalog.remove(catalog.getCurrentDB().dbName);
  }

  void loginAdmin() {
    session_id_ = {};
    mapd_handler_->connect(session_id_, default_user_, default_pass_, default_db_name_);
    // Store admin session ID in seperate variable so we can always logout
    // the default admin on teardown
    admin_session_id_ = session_id_;
  }

  void logoutAdmin() { mapd_handler_->disconnect(admin_session_id_); }

  void logout(const TSessionId& id) { mapd_handler_->disconnect(id); }

  void login(const std::string& user,
             const std::string& pass,
             const std::string& db_name = default_db_name_) {
    session_id_ = {};
    mapd_handler_->connect(session_id_, user, pass, db_name);
  }

  // Login and return the session id to logout later
  void login(const std::string& user,
             const std::string& pass,
             const std::string& db,
             TSessionId& result_id) {
    mapd_handler_->connect(result_id, user, pass, db);
  }

  void queryAndAssertException(const std::string& sql_statement,
                               const std::string& error_message) {
    try {
      sql(sql_statement);
      FAIL() << "An exception should have been thrown for this test case.";
    } catch (const TMapDException& e) {
      ASSERT_EQ(error_message, e.error_msg);
    }
  }

 private:
  static std::unique_ptr<MapDHandler> mapd_handler_;
  static TSessionId session_id_;
  static TSessionId admin_session_id_;
  static std::vector<LeafHostInfo> db_leaves_;
  static std::vector<LeafHostInfo> string_leaves_;
  static AuthMetadata auth_metadata_;
  static MapDParameters mapd_parameters_;
  static std::string udf_filename_;
  static std::string udf_compiler_path_;
  static std::string default_user_;
  static std::string default_pass_;
  static std::string default_db_name_;
  static std::vector<std::string> udf_compiler_options_;
  static std::string cluster_config_file_path_;
};

TSessionId MapDHandlerTestFixture::session_id_{};
TSessionId MapDHandlerTestFixture::admin_session_id_{};
std::unique_ptr<MapDHandler> MapDHandlerTestFixture::mapd_handler_ = nullptr;
std::vector<LeafHostInfo> MapDHandlerTestFixture::db_leaves_{};
std::vector<LeafHostInfo> MapDHandlerTestFixture::string_leaves_{};
AuthMetadata MapDHandlerTestFixture::auth_metadata_{};
std::string MapDHandlerTestFixture::udf_filename_{};
std::string MapDHandlerTestFixture::udf_compiler_path_{};
std::string MapDHandlerTestFixture::default_user_{"admin"};
std::string MapDHandlerTestFixture::default_pass_{"HyperInteractive"};
std::string MapDHandlerTestFixture::default_db_name_{};
MapDParameters MapDHandlerTestFixture::mapd_parameters_{};
std::vector<std::string> MapDHandlerTestFixture::udf_compiler_options_{};
std::string MapDHandlerTestFixture::cluster_config_file_path_{};
