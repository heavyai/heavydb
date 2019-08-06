/*
 * Copyright 2019 OmniSci, Inc.
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

#include <stdexcept>

#include "Descriptors/Types.h"

struct QueryExecutionProperties {
  QueryDescriptionType query_type;
  bool was_multifrag_kernel_launch;
};

class QueryExecutionError : public std::runtime_error {
 public:
  QueryExecutionError(const int32_t error_code, const std::string& e)
      : std::runtime_error("Query execution failed with error code " +
                           std::to_string(error_code) + "\n" + e)
      , error_code_(error_code) {}

  QueryExecutionError(const int32_t error_code,
                      const std::string& e,
                      const QueryExecutionProperties& execution_properties)
      : std::runtime_error("Query execution failed with error code " +
                           std::to_string(error_code) + "\n" + e)
      , error_code_(error_code)
      , execution_props_(execution_properties) {}

  QueryExecutionError(const int32_t error_code,
                      const QueryExecutionProperties& execution_properties)
      : std::runtime_error("Query execution failed with error code " +
                           std::to_string(error_code))
      , error_code_(error_code)
      , execution_props_(execution_properties) {}

  QueryExecutionError(const int32_t error_code)
      : std::runtime_error("Query execution failed with error code " +
                           std::to_string(error_code))
      , error_code_(error_code) {}

  int32_t getErrorCode() const { return error_code_; }

  bool wasMultifragKernelLaunch() const {
    return execution_props_ && (*execution_props_).was_multifrag_kernel_launch;
  }

 protected:
  int32_t error_code_;
  boost::optional<QueryExecutionProperties> execution_props_;
};
