/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Debug logging tailored to rendering
 *
 * Provides lightweight scope based logging for debugging render
 * code paths. When disabled globally all code is elided, but log
 * messages must still be valid (eases maintenance of permanent logging)
 *
 * Enable in cmake with -DENABLE_RENDER_LOGGER=on
 * Preprocessor definition is set in GfxDriver and QueryRenderer only
 *
 * Several sink options are provided for messages:
 * stdout (default), stderr, and LOG(INFO)
 *
 * Compile time options in RenderLogger.cpp:
 *  - Change default Sink, and max scope depth
 *  - Enable injection of logger DEBUG_TIMER objects
 *  - Enable logging of scope name upon closure
 *
 * Usage:
 *
 * Push a new scope level:
 * - RENDER_LOG_SCOPE() << "buffer size " << size;
 * - RENDER_LOG_SCOPE_P(gpu_id) << "rendering accumulation on gpu";
 * - RENDER_LOG_SCOPE_P(used_gpus) << "preparing for these gpus";
 *
 * Log at current scope level:
 * - RENDER_LOG() << "some more detail";
 *
 * RAII filter or redirect for child scopes
 * - render_logger::DepthFilter log_filter{4}; // only show scopes < 4 deep
 * - render_logger::SinkSwitch log_sink{render_logger::Sink::kLogInfo}
 *
 * Tips:
 * - Enable log scope closure when deep call stacks make tracking the stack
 *   unwind difficult to visually parse. Alternately drop the log into
 *   an editor that supports "scope" folding
 * - Set global max depth to 0, and add a scope filter of large value
 *   at the top of a call (e.g. UpdateShaders) to only log functions in
 *   a specific call stack
 * - Use a Sink switch to redirect specific calls of interest to LOG(INFO)
 *   for examination in log files
 *
 * Future Work:
 *  - Verbosity levels and filtering independent of scope depth
 *  - Sink to custom log channel
 *  - NVTX pofile marker / ranges support
 *  - Thread awareness. It is currently "thread safe" but output will
 *    be confusing if called from overlapping threads. This will need
 *    to be addressed once the renderer uses multiple threads.
 *    Since DEBUG_TIMERs are thread aware the optional timer injection
 *    can be used as a workaround if necessary
 * */

#pragma once

#include <memory>
#include <set>
#include <sstream>
#include <string>

#include <boost/noncopyable.hpp>

#include "Logger/Logger.h"

namespace render_logger {

enum class Sink { kNull, kStdOut, kStdErr, kLogInfo };

// Closure object is created by SCOPE macros
// Lifetime is current scope
class Closure {
 public:
  Closure() = default;
  ~Closure();

 private:
  std::string name_;
  std::unique_ptr<logger::DebugTimer> debug_timer_;

  friend class Logger;
};

using ClosureUqPtr = std::unique_ptr<Closure>;
using LogCallback = std::function<void(std::ostringstream& os)>;

// Lifetime of logger is current line
class Logger : boost::noncopyable {
 public:
  Logger();
  explicit Logger(const char* file, int line, const char* name, ClosureUqPtr& closure);
  explicit Logger(const char* file,
                  int line,
                  const char* name,
                  ClosureUqPtr& closure,
                  const std::vector<uint32_t>& gpu_vector);
  explicit Logger(const char* file,
                  int line,
                  const char* name,
                  ClosureUqPtr& closure,
                  const std::set<uint32_t>& gpu_set);
  explicit Logger(const char* file,
                  int line,
                  const char* name,
                  ClosureUqPtr& closure,
                  uint32_t gpu_id);
  explicit Logger(const char* file,
                  int line,
                  const char* name,
                  ClosureUqPtr& closure,
                  LogCallback callback);
  ~Logger();

  std::ostream& get();
  std::ostream& indent();

  static Sink& getSink();
  static int getDepth();
  static int& getMaxDepth();

 private:
  // Use pointer to limit inline object size
  std::unique_ptr<std::ostringstream> os_;
};

template <typename T>
std::string format_gpuid_container(T container) {
  std::ostringstream os;
  os << "gpus [";
  if (container.empty()) {
    os << "empty";
  } else {
    auto first = *container.begin();
    for (auto id : container) {
      if (id != first) {
        os << ' ';
      }
      os << id;
    }
  }
  os << "]";
  return os.str();
}

const auto format_gpuid_vector = format_gpuid_container<std::vector<uint32_t>>;
const auto format_gpuid_set = format_gpuid_container<std::set<uint32_t>>;

// Change the max scope depth for the current scope and child scopes
// Usage: render_logger::DepthFilter log_filter{4};
class DepthFilter {
  int prev_;

 public:
  explicit DepthFilter(int max) : prev_{Logger::getMaxDepth()} {
    Logger::getMaxDepth() = max;
  }
  ~DepthFilter() { Logger::getMaxDepth() = prev_; }
};

// Change the stream sink (log target) for the current scope and child scopes
// Usage: render::logger::SinkSwitch log_sink{render_logger::Sink::kLogInfo};
class SinkSwitch {
  Sink prev_;

 public:
  explicit SinkSwitch(Sink sink) : prev_{Logger::getSink()} { Logger::getSink() = sink; }
  ~SinkSwitch() { Logger::getSink() = prev_; }
};

inline bool fast_scope_depth_check() {
  extern int g_scope_depth;
  extern int g_max_scope_depth;
  return g_scope_depth >= g_max_scope_depth;
}

#ifdef ENABLE_RENDER_LOGGER

#define RENDER_LOG()                             \
  if (::render_logger::fast_scope_depth_check()) \
    ;                                            \
  else                                           \
    ::render_logger::Logger().indent()

#define RENDER_LOG_SCOPE()                               \
  ::render_logger::ClosureUqPtr _render_logger_closure_; \
  if (::render_logger::fast_scope_depth_check())         \
    ;                                                    \
  else                                                   \
    ::render_logger::Logger(__FILE__, __LINE__, __func__, _render_logger_closure_).get()

#define RENDER_LOG_SCOPE_P(...)                                             \
  ::render_logger::ClosureUqPtr _render_logger_closure_;                    \
  if (::render_logger::fast_scope_depth_check())                            \
    ;                                                                       \
  else                                                                      \
    ::render_logger::Logger(                                                \
        __FILE__, __LINE__, __func__, _render_logger_closure_, __VA_ARGS__) \
        .get()

#else
// Use if constexpr to ensure that log messages still compile, but are elided
#define RENDER_LOG()  \
  if constexpr (true) \
    ;                 \
  else                \
    ::render_logger::Logger().get()

#define RENDER_LOG_SCOPE() RENDER_LOG()
#define RENDER_LOG_SCOPE_P(...) RENDER_LOG()

#endif
}  // namespace render_logger
