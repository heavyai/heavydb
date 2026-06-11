/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "enums.h"

#include <stdexcept>

using heavyai::ErrorCode;

struct QueryExecutionProperties {
  heavyai::QueryDescriptionType query_type;
  bool was_multifrag_kernel_launch;
};

class QueryExecutionError : public std::runtime_error {
 public:
  QueryExecutionError(const ErrorCode error_code, const std::string& e)
      : std::runtime_error(std::string("Query execution failed with error code ") +
                           to_string(error_code) + "\n" + e)
      , error_code_(static_cast<int32_t>(error_code)) {}

  QueryExecutionError(const ErrorCode error_code,
                      const std::string& e,
                      const QueryExecutionProperties& execution_properties)
      : std::runtime_error(std::string("Query execution failed with error code ") +
                           to_string(error_code) + "\n" + e)
      , error_code_(static_cast<int32_t>(error_code))
      , execution_props_(execution_properties) {}

  QueryExecutionError(const ErrorCode error_code,
                      const QueryExecutionProperties& execution_properties)
      : std::runtime_error(std::string("Query execution failed with error code ") +
                           to_string(error_code))
      , error_code_(static_cast<int32_t>(error_code))
      , execution_props_(execution_properties) {}

  QueryExecutionError(const ErrorCode error_code)
      : std::runtime_error(std::string("Query execution failed with error code ") +
                           to_string(error_code))
      , error_code_(static_cast<int32_t>(error_code)) {}

  // Given error_code may not be in range of enum class ErrorCode.
  QueryExecutionError(const int32_t error_code)
      : std::runtime_error("Query execution failed with error code " +
                           QueryExecutionError::toString(error_code))
      , error_code_(error_code) {}

  int32_t getErrorCode() const { return error_code_; }

  bool hasErrorCode(ErrorCode const ec) const {
    return error_code_ == static_cast<int32_t>(ec);
  }

  inline static std::string toString(int32_t error_code) {
    if (size_t(error_code) < size_t(ErrorCode::N_)) {
      return to_string(static_cast<ErrorCode>(error_code));
    } else {
      return std::to_string(error_code);
    }
  }

  bool wasMultifragKernelLaunch() const {
    return execution_props_ && (*execution_props_).was_multifrag_kernel_launch;
  }

 protected:
  int32_t error_code_;  // May be out-of-range of enum class ErrorCode values.
  boost::optional<QueryExecutionProperties> execution_props_;
};

class ReductionRanOutOfSlots : public std::runtime_error {
 public:
  ReductionRanOutOfSlots() : std::runtime_error("ReductionRanOutOfSlots") {}
};
