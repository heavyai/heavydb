/*
 * Copyright 2022 HEAVY.AI, Inc.
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

/*
 * @file    Logger.h
 * @description Use Boost.Log for logging data compatible with previous API.
 *
 * Usage:
 * - Initialize a LogOptions object. E.g.
 *   logger::LogOptions log_options(argv[0]);
 * - LogOptions can optionally be added to boost::program_options:
 *   help_desc.add(log_options.get_options());
 * - Initialize global logger once per application:
 *   logger::init(log_options);
 * - From anywhere in the program:
 *    - LOG(INFO) << "Nice query!";
 *    - LOG(DEBUG4) << "x = " << x;
 *    - CHECK(condition);
 *    - CHECK_LE(x, xmax);
 *   Newlines are automatically appended to log messages.
 */

#ifndef SHARED_LOGGER_H
#define SHARED_LOGGER_H

#if !(defined(__CUDACC__) || defined(NO_BOOST))

#include <boost/config.hpp>
#include <boost/log/sources/record_ostream.hpp>

#include <memory>
#include <set>
#include <type_traits>

#endif

#include <array>
#include <sstream>
#include <string>
#include <thread>

#ifdef ERROR
// A common way for this to occur is with a #include <windows.h> which globally defines a
// number of macros, such as ERROR, that interferes with other headers. This may be
// resolved by locating the new #include that directly or indirectly includes windows.h
// (or something else that #defines ERROR) and placing a
// #include "Shared/cleanup_global_namespace.h" after it.
//
// Q: Why not just #include "Shared/cleanup_global_namespace.h" here?
//
// A: Two reasons:
//    * ERROR is not the only macro that windows.h defines, e.g. GetObject which
//      interferes with rapidjson.
//    * By not cleaning up the global macros at the source requires potential cleaning
//      on a much larger scale: all places that call LOG(ERROR). (Due to header guards,
//      #include "Logger.h" may not be included where it needs to in order to clean up.)
#error "ERROR must not be globally defined during preprocessing."
#endif

namespace boost {
// Forward declare boost types to avoid including expensive headers.
namespace program_options {
class options_description;
}
namespace filesystem {
class path;
}
}  // namespace boost

extern bool g_enable_debug_timer;

namespace logger {

// Channel, ChannelNames, and ChannelSymbols must be updated together.
// Each Channel has its own ChannelLogger declared below and defined in Logger.cpp.
enum Channel { IR = 0, PTX, ASM, _NCHANNELS };

constexpr std::array<char const*, 3> ChannelNames{"IR", "PTX", "ASM"};

constexpr std::array<char, 3> ChannelSymbols{'R', 'P', 'A'};

static_assert(Channel::_NCHANNELS == ChannelNames.size(),
              "Size of ChannelNames must equal number of Channels.");
static_assert(Channel::_NCHANNELS == ChannelSymbols.size(),
              "Size of ChannelSymbols must equal number of Channels.");

// Severity, SeverityNames, and SeveritySymbols must be updated together.
enum Severity {
  DEBUG4 = 0,
  DEBUG3,
  DEBUG2,
  DEBUG1,
  INFO,
  WARNING,
  ERROR,
  FATAL,
  _NSEVERITIES  // number of severity levels
};

constexpr std::array<char const*, 8> SeverityNames{"DEBUG4",
                                                   "DEBUG3",
                                                   "DEBUG2",
                                                   "DEBUG1",
                                                   "INFO",
                                                   "WARNING",
                                                   "ERROR",
                                                   "FATAL"};

constexpr std::array<char, 8> SeveritySymbols{'4', '3', '2', '1', 'I', 'W', 'E', 'F'};

static_assert(Severity::_NSEVERITIES == SeverityNames.size(),
              "Size of SeverityNames must equal number of Severity levels.");
static_assert(Severity::_NSEVERITIES == SeveritySymbols.size(),
              "Size of SeveritySymbols must equal number of Severity levels.");

using RequestId = uint64_t;
using ThreadId = uint64_t;

class LocalIdsScopeGuard;

struct ThreadLocalIds {
  RequestId request_id_;
  ThreadId thread_id_;
  ThreadLocalIds() = default;
  ThreadLocalIds(RequestId const request_id, ThreadId const thread_id)
      : request_id_(request_id), thread_id_(thread_id) {}
  // Call to assign a new thread_id.
  // The return value MUST be held otherwise g_thread_local_ids will immediately revert.
  LocalIdsScopeGuard setNewThreadId() const;
};

static_assert(std::is_standard_layout<ThreadLocalIds>::value);
static_assert(std::is_trivial<ThreadLocalIds>::value);

// On destruction, reset g_thread_local_ids to original value.
// May be needed by thread managers that recycle threads, such as TBB.
class [[nodiscard]] LocalIdsScopeGuard {
  ThreadLocalIds prev_local_ids_;
#ifndef NDEBUG
  // To verify g_thread_local_ids is unchanged during LocalIdsScopeGuard's lifetime.
  ThreadLocalIds thread_local_ids_;
#endif
  bool enabled_;  // Reset g_thread_local_ids iff enabled_.

  // Used by move constructor + assignment operator.
  // OK since all member variables are PODs.
  LocalIdsScopeGuard(LocalIdsScopeGuard const&) = default;
  LocalIdsScopeGuard& operator=(LocalIdsScopeGuard const&) = default;

 public:
#ifndef NDEBUG
  LocalIdsScopeGuard(ThreadLocalIds const prev_local_ids,
                     ThreadLocalIds const thread_local_ids)
      : prev_local_ids_(prev_local_ids)
      , thread_local_ids_(thread_local_ids)
      , enabled_(true) {}
#else
  LocalIdsScopeGuard(ThreadLocalIds const prev_local_ids, ThreadLocalIds)
      : prev_local_ids_(prev_local_ids), enabled_(true) {}
#endif
  // clang-format off
  LocalIdsScopeGuard(LocalIdsScopeGuard&& rhs) : LocalIdsScopeGuard(rhs) {
    // clang-format on
    rhs.enabled_ = false;
  }
  LocalIdsScopeGuard& operator=(LocalIdsScopeGuard&& rhs) {
    *this = rhs;
    rhs.enabled_ = false;
    return *this;
  }
  ~LocalIdsScopeGuard();
};

RequestId request_id();
ThreadId thread_id();
ThreadLocalIds thread_local_ids();
// Assign new request_id and return it.
// Assign new thread_id iff current thread_id is 0.
RequestId set_new_request_id();
void set_request_id(RequestId);

#if !(defined(__CUDACC__) || defined(NO_BOOST))

using Channels = std::set<Channel>;

// Filled by boost::program_options
class LogOptions {
  std::string base_path_{"."};  // ignored if log_dir_ is absolute.
  // boost::program_options::options_description is not copyable so unique_ptr
  // allows for modification after initialization (e.g. changing default values.)
  std::unique_ptr<boost::program_options::options_description> options_;

 public:
  // Initialize to default values
  std::unique_ptr<boost::filesystem::path> log_dir_;
  // file_name_pattern and symlink are prepended with base_name.
  std::string file_name_pattern_{".{SEVERITY}.%Y%m%d-%H%M%S.log"};
  std::string symlink_{".{SEVERITY}"};
  Severity severity_{Severity::INFO};
  Severity severity_clog_{Severity::ERROR};
  Channels channels_;
  bool auto_flush_{true};
  size_t max_files_{100};
  size_t min_free_space_{20 << 20};
  bool rotate_daily_{true};
  size_t rotation_size_{10 << 20};

  LogOptions(char const* argv0);
  ~LogOptions();  // Needed to allow forward declarations within std::unique_ptr.
  boost::filesystem::path full_log_dir() const;
  boost::program_options::options_description const& get_options() const;
  void parse_command_line(int, char const* const*);
  void set_base_path(std::string const& base_path);
  void set_options();
};

// Execute once in main() to initialize boost loggers.
void init(LogOptions const&);

// Flush all sinks.
// https://www.boost.org/libs/log/doc/html/log/rationale/why_crash_on_term.html
void shutdown();

struct LogShutdown {
  inline ~LogShutdown() { shutdown(); }
};

// Optional pointer to function to call on LOG(FATAL).
using FatalFunc = void (*)() noexcept;
void set_once_fatal_func(FatalFunc);

// Lifetime of Logger is each call to LOG().
class Logger {
  bool const is_channel_;
  int const enum_value_;
  // Pointers are used to minimize size of inline objects.
  std::unique_ptr<boost::log::record> record_;
  std::unique_ptr<boost::log::record_ostream> stream_;

 public:
  explicit Logger(Channel);
  explicit Logger(Severity);
  Logger(Logger&&) = default;
  ~Logger();
  operator bool() const;
  // Must check operator bool() first before calling stream().
  boost::log::record_ostream& stream(char const* file, int line);
};

inline bool fast_logging_check(Channel) {
  extern bool g_any_active_channels;
  return g_any_active_channels;
}

inline bool fast_logging_check(Severity severity) {
  extern Severity g_min_active_severity;
  return g_min_active_severity <= severity;
}

// These macros risk inadvertent else-matching to the if statements,
// which are fortunately prevented by our clang-tidy requirements.
// These can be changed to for/while loops with slight performance degradation.

#define SLOG(severity_or_channel)                                                     \
  if (auto _heavydb_logger_severity_or_channel_ = severity_or_channel;                \
      logger::fast_logging_check(_heavydb_logger_severity_or_channel_))               \
    if (auto _heavydb_logger_ = logger::Logger(_heavydb_logger_severity_or_channel_)) \
  _heavydb_logger_.stream(__FILE__, __LINE__)

#define LOG(tag) SLOG(logger::tag)

#define LOGGING(tag) logger::fast_logging_check(logger::tag)

#define VLOGGING(n) logger::fast_logging_check(logger::DEBUG##n)

#define CHECK(condition)            \
  if (BOOST_UNLIKELY(!(condition))) \
  LOG(FATAL) << "Check failed: " #condition " "

#define CHECK_OP(OP, x, y)                                      \
  if (std::string* fatal_msg = logger::Check##OP(x, y, #x, #y)) \
  LOG(FATAL) << *std::unique_ptr<std::string>(fatal_msg)

#define CHECK_EQ(x, y) CHECK_OP(EQ, x, y)
#define CHECK_NE(x, y) CHECK_OP(NE, x, y)
#define CHECK_LT(x, y) CHECK_OP(LT, x, y)
#define CHECK_LE(x, y) CHECK_OP(LE, x, y)
#define CHECK_GT(x, y) CHECK_OP(GT, x, y)
#define CHECK_GE(x, y) CHECK_OP(GE, x, y)

template <typename X, typename Y>
BOOST_NOINLINE std::string* check_failed(X const& x,
                                         Y const& y,
                                         char const* xstr,
                                         char const* ystr,
                                         char const* op_str) {
  std::stringstream ss;
  ss << "Check failed: " << xstr << op_str << ystr << " (" << x << op_str << y << ") ";
  return new std::string(ss.str());  // Deleted by CHECK_OP macro.
}

// Complexity comes from requirement that x and y be evaluated only once.
#define HEAVYDB_CHECKOP_FUNCTION(name, op)                          \
  template <typename X, typename Y>                                 \
  inline std::string* Check##name(                                  \
      X const& x, Y const& y, char const* xstr, char const* ystr) { \
    if (BOOST_LIKELY(x op y))                                       \
      return nullptr;                                               \
    else                                                            \
      return logger::check_failed(x, y, xstr, ystr, " " #op " ");   \
  }
HEAVYDB_CHECKOP_FUNCTION(EQ, ==)
HEAVYDB_CHECKOP_FUNCTION(NE, !=)
HEAVYDB_CHECKOP_FUNCTION(LT, <)
HEAVYDB_CHECKOP_FUNCTION(LE, <=)
HEAVYDB_CHECKOP_FUNCTION(GT, >)
HEAVYDB_CHECKOP_FUNCTION(GE, >=)
#undef HEAVYDB_CHECKOP_FUNCTION

#define UNREACHABLE() LOG(FATAL) << "UNREACHABLE "

#else  // __CUDACC__

// Provided for backward compatibility to allow code to compile.
// No logging is actually done, since cuda code should not log.
template <Severity severity>
class NullLogger {
 public:
  NullLogger() {
    if /*constexpr*/ (severity == Severity::FATAL) {
      abort();
    }
  }
  template <typename T>
#ifndef SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS
  [[deprecated]]
#endif
  // If you are here because of a deprecation warning, that is because the code
  // is attempting to log something in cuda (e.g. CHECK macros). It should
  // probably be fixed there.
  NullLogger&
  operator<<(const T&) {
    return *this;
  }
};

#define LOG(severity) logger::NullLogger<logger::Severity::severity>()

#define LOGGING(tag) false

#define VLOGGING(n) false

#define CHECK(condition) LOG_IF(FATAL, !(condition))

#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))

#define UNREACHABLE() LOG(FATAL)

#endif  // __CUDACC__

#define LOG_IF(severity, condition) \
  if (condition)                    \
  LOG(severity)

#define VLOG(n) LOG(DEBUG##n)

class Duration;

class DebugTimer {
  Duration* duration_;
  DebugTimer(DebugTimer const&) = delete;
  DebugTimer(DebugTimer&&) = delete;
  DebugTimer& operator=(DebugTimer const&) = delete;
  DebugTimer& operator=(DebugTimer&&) = delete;

 public:
  DebugTimer(Severity, char const* file, int line, char const* name);
  ~DebugTimer();
  void stop();
  // json is returned only when called on the root DurationTree.
  std::string stopAndGetJson();
};

boost::filesystem::path get_log_dir_path();

void debug_timer_new_thread(ThreadId const parent_thread_id);

// Typical usage: auto timer = DEBUG_TIMER(__func__);
#define DEBUG_TIMER(name) logger::DebugTimer(logger::INFO, __FILE__, __LINE__, name)

// This MUST NOT be called more than once per thread, otherwise a failed CHECK() occurs.
// Best practice is to call it from the point where the new thread is spawned.
// Beware of threads that are re-used.
#define DEBUG_TIMER_NEW_THREAD(parent_thread_id)        \
  do {                                                  \
    if (g_enable_debug_timer)                           \
      logger::debug_timer_new_thread(parent_thread_id); \
  } while (false)

}  // namespace logger

#endif  // SHARED_LOGGER_H
