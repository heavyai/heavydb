/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <exception>
#include <string>

#include "QueryRenderer/Data/Transforms/Aggregate/AggXformOp.h"

namespace QueryRenderer {

/**
 * Prepends additional tracking information to an existing exception's error message for
 * logging. It manages this by rethrowing the existing exception (passed as an
 * std::exception_ptr) to capture std::exception objects and grabs the error strings via
 * the std::exception::what() method.
 */
inline void log_agg_info_and_throw(const AggOp& op,
                                   const std::string& file,
                                   const int lineno,
                                   std::exception_ptr eptr) {
  // Attempting to grab the error msg via the std::exception::what() method.
  // The only way to retrive this via a std::exception_ptr is to rethrow. Casting
  // std::exception_ptr to the exception type you're expecting doesn't seem to work.
  //
  // std::exception_ptr is used because we want to maintain the original exception type to
  // rethrow.
  try {
    std::rethrow_exception(eptr);
  } catch (std::exception& err) {
    LOG(ERROR) << std::string(op)
               << ": Got an error during operator execution: " << err.what() << " "
               << gfx::RenderError::formatFileLineString(file, std::to_string(lineno));
  }
  std::rethrow_exception(eptr);
}

#define LOG_AGG_INFO_AND_THROW(op) \
  log_agg_info_and_throw(op, __FILE__, __LINE__, std::current_exception());

}  // namespace QueryRenderer
