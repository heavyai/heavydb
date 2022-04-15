/*
 * Copyright 2022 Intel Corporation.
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

#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"

#include "gen-cpp/calciteserver_types.h"

struct FilterPushDownInfo {
  int input_prev;
  int input_start;
  int input_next;
};

class CalciteJNI {
 public:
  CalciteJNI(const std::string& udf_filename = "", size_t calcite_max_mem_mb = 1024);
  ~CalciteJNI();

  std::string process(const std::string& user,
                      const std::string& db_name,
                      const std::string& sql_string,
                      const std::string& schema_json = "",
                      const std::string& session_id = "",
                      const std::vector<FilterPushDownInfo>& filter_push_down_info = {},
                      const bool legacy_syntax = false,
                      const bool is_explain = false,
                      const bool is_view_optimize = false);

  std::string getExtensionFunctionWhitelist();
  std::string getUserDefinedFunctionWhitelist();
  std::string getRuntimeExtensionFunctionWhitelist();
  void setRuntimeExtensionFunctions(
      const std::vector<ExtensionFunction>& udfs,
      const std::vector<table_functions::TableFunction>& udtfs,
      bool is_runtime = true);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
