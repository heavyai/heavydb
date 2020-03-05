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
 protected:
  virtual void SetUp() override {
    if (!mapd_handler) {
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
      const int num_gpus{-1};
      const int start_gpu{0};
      const size_t reserved_gpu_mem{134217728};
      const size_t num_reader_threads{0};
      const bool legacy_syntax{true};
      const int idle_session_duration{60};
      const int max_session_duration{43200};
      const bool enable_runtime_udf_registration{false};
      mapd_parameters.omnisci_server_port = -1;

      mapd_handler = std::make_unique<MapDHandler>(db_leaves,
                                                   string_leaves,
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
                                                   num_gpus,
                                                   start_gpu,
                                                   reserved_gpu_mem,
                                                   num_reader_threads,
                                                   auth_metadata,
                                                   mapd_parameters,
                                                   legacy_syntax,
                                                   idle_session_duration,
                                                   max_session_duration,
                                                   enable_runtime_udf_registration,
                                                   udf_filename,
                                                   udf_compiler_path);
      mapd_handler->connect(session_id, default_user, default_pass, db_name);
    }
  }

  void sql(TQueryResult& result, const std::string& query) {
    mapd_handler->sql_execute(result, session_id, query, true, "", -1, -1);
  }

  Catalog_Namespace::Catalog& getCatalog() {
    return mapd_handler->get_session_copy_ptr(session_id)->getCatalog();
  }

 private:
  static std::unique_ptr<MapDHandler> mapd_handler;
  static TSessionId session_id;
  static std::vector<LeafHostInfo> db_leaves;
  static std::vector<LeafHostInfo> string_leaves;
  static AuthMetadata auth_metadata;
  static MapDParameters mapd_parameters;
  static std::string udf_filename;
  static std::string udf_compiler_path;
  static std::string default_user;
  static std::string default_pass;
  static std::string db_name;
};

TSessionId MapDHandlerTestFixture::session_id{};
std::unique_ptr<MapDHandler> MapDHandlerTestFixture::mapd_handler = nullptr;
std::vector<LeafHostInfo> MapDHandlerTestFixture::db_leaves{};
std::vector<LeafHostInfo> MapDHandlerTestFixture::string_leaves{};
AuthMetadata MapDHandlerTestFixture::auth_metadata{};
std::string MapDHandlerTestFixture::udf_filename{};
std::string MapDHandlerTestFixture::udf_compiler_path{};
std::string MapDHandlerTestFixture::default_user{"admin"};
std::string MapDHandlerTestFixture::default_pass{"HyperInteractive"};
std::string MapDHandlerTestFixture::db_name{};
MapDParameters MapDHandlerTestFixture::mapd_parameters{};
