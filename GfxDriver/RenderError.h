/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>
#include <string>

#include <boost/filesystem.hpp>

#include "Logger/Logger.h"

namespace filesys =
    boost::filesystem;  // TODO(croot): change this to std::filesystem when we go C++17

// @TODO simon.eves
// there's an issue with Ubuntu 17.10/gcc5.5/CUDA8.0.61 which breaks std::to_string
// so for now we string-ize __LINE__ the old-fashioned way
// instances of the constructor which takes __FILE__ and __LINE__ must be changed
// but there's only one so far (ColorInitializer.h) which I changed to use the
// NO_LOG version of the THROW macro
#define RENDERER_ERROR_STRINGIZE_DETAIL(x) #x
#define RENDERER_ERROR_STRINGIZE(x) RENDERER_ERROR_STRINGIZE_DETAIL(x)

namespace gfx {

class RenderError : public std::runtime_error {
 public:
  inline static std::string formatFileLineString(const std::string& file,
                                                 const std::string& lineno) {
    return "(" + filesys::path(file).filename().string() + ":" + lineno + ")";
  }

  explicit RenderError(const std::string& details = "")
      : std::runtime_error(details), details(details) {}
  explicit RenderError(const std::string& file,
                       const std::string& lineno,
                       const std::string& details)
      : std::runtime_error(details + " " + formatFileLineString(file, lineno))
      , details(details) {}
  std::string getDetails() const { return details; }

 private:
  std::string details;
};

class OutOfGpuMemoryError : public RenderError {
 public:
  OutOfGpuMemoryError(const std::string& details = "")
      : RenderError("OutOfGpuMemoryError" + (details.length() ? ": " + details : "")) {}
};

class DeviceLostError : public RenderError {
 public:
  DeviceLostError(const std::string& details = "")
      : RenderError("DeviceLostError" + (details.length() ? ": " + details : "")) {}
};

struct RenderErrorLogger {
  const bool logging_enabled = false;
  explicit RenderErrorLogger(const bool enable_logging)
      : logging_enabled(enable_logging) {}
  virtual ~RenderErrorLogger() {}
  virtual std::string getLogMsg() const { return ""; };
  virtual std::string getExceptionMsg() const = 0;
};

inline std::exception_ptr buildAndLogRenderError(const std::string& file,
                                                 const std::string& lineno,
                                                 const std::string& errstr) {
#ifndef __CUDACC__
  LOG(ERROR) << errstr << " " << RenderError::formatFileLineString(file, lineno);
#endif
  return std::make_exception_ptr(RenderError(errstr));
}

inline std::exception_ptr buildAndLogRenderError(const std::string& file,
                                                 const std::string& lineno,
                                                 const RenderErrorLogger& logger) {
  if (logger.logging_enabled) {
#ifndef __CUDACC__
    LOG(ERROR) << logger.getLogMsg() << " "
               << RenderError::formatFileLineString(file, lineno);
#endif
    return std::make_exception_ptr(RenderError(logger.getExceptionMsg()));
  }
  return std::make_exception_ptr(RenderError(file, lineno, logger.getExceptionMsg()));
}

inline std::exception_ptr buildAndLogRenderError(const std::string& file,
                                                 const std::string& lineno,
                                                 std::exception_ptr eptr) {
#ifndef __CUDACC__
  CHECK(eptr) << RenderError::formatFileLineString(file, lineno);
  try {
    std::rethrow_exception(eptr);
  } catch (std::exception& e) {
    LOG(ERROR) << e.what() << " " << RenderError::formatFileLineString(file, lineno);
  } catch (...) {
    // pass-thru
  }
  return eptr;
#else
  return eptr;
#endif  // __CUDACC__
}

}  // namespace gfx

#define RUNTIME_EX_ASSERT(condition, errobj)                    \
  if (!(condition)) {                                           \
    std::rethrow_exception(gfx::buildAndLogRenderError(         \
        __FILE__, RENDERER_ERROR_STRINGIZE(__LINE__), errobj)); \
  }

#define THROW_RUNTIME_EX(errobj)                      \
  std::rethrow_exception(gfx::buildAndLogRenderError( \
      __FILE__, RENDERER_ERROR_STRINGIZE(__LINE__), errobj));

#define THROW_RUNTIME_EX_NO_LOG(errstr) \
  throw gfx::RenderError(__FILE__, RENDERER_ERROR_STRINGIZE(__LINE__), errstr);
