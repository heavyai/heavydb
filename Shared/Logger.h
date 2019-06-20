/*
 * Copyright (c) 2019 OmniSci, Inc.
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
 * @author  Matt Pulver <matt.pulver@omnisci.com>
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

#ifndef __CUDACC__

#include <boost/config.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/common.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/sources/severity_feature.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/phoenix.hpp>
#include <boost/program_options.hpp>

#include <atomic>
#include <mutex>

#endif

#include <sstream>

namespace logger {

// Must match std::array<> SeverityNames and SeveritySymbols in Logger.cpp.
enum Severity {
  DEBUG4 = 0,
  DEBUG3,
  DEBUG2,
  DEBUG1,
  INFO,
  WARNING,
  ERROR,
  FATAL,
  NLEVELS  // number of severity levels
};

#ifndef __CUDACC__

namespace attr = boost::log::attributes;
namespace expr = boost::log::expressions;
namespace keywords = boost::log::keywords;
namespace sinks = boost::log::sinks;
namespace sources = boost::log::sources;
namespace po = boost::program_options;

// Filled by boost::program_options
class LogOptions {
  std::string base_path_{"."};  // ignored if log_dir_ is absolute.
  po::options_description options_;

 public:
  // Initialize to default values
  boost::filesystem::path log_dir_{"omnisci_log"};
  // file_name_pattern and symlink are prepended with base_name.
  std::string file_name_pattern_{".{SEVERITY}.%Y%m%d-%H%M%S.log"};
  std::string symlink_{".{SEVERITY}.log"};
  Severity severity_{Severity::INFO};
  Severity severity_clog_{Severity::WARNING};
  bool auto_flush_{true};
  size_t max_files_{100};
  size_t min_free_space_{20 << 20};
  bool rotate_daily_{true};
  size_t rotation_size_{10 << 20};

  LogOptions(char const* argv0);
  boost::filesystem::path full_log_dir() const;
  po::options_description const& get_options() const;
  void parse_command_line(int, char const* const*);
  void set_base_path(std::string const& base_path);
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
typedef void (*FatalFunc)();
void set_once_fatal_func(FatalFunc);

using logger_t = boost::log::sources::severity_logger_mt<Severity>;

BOOST_LOG_GLOBAL_LOGGER(g_logger, logger_t)

// Lifetime of Logger is each call to LOG().
template <Severity SEVERITY>
class Logger {
  // Pointers are used to minimize size of inline objects.
  std::unique_ptr<boost::log::record> record_;
  std::unique_ptr<boost::log::record_ostream> stream_;

 public:
  Logger();
  Logger(Logger&&) = default;
  ~Logger();
  inline operator bool() const { return static_cast<bool>(stream_); }
  // Must check operator bool() first before calling stream().
  boost::log::record_ostream& stream(char const* file, int line);
};

template <Severity SEVERITY>
Logger<SEVERITY>::Logger()
    : record_(std::make_unique<boost::log::record>(
          g_logger::get().open_record(boost::log::keywords::severity = SEVERITY))) {
  if (*record_) {
    stream_ = std::make_unique<boost::log::record_ostream>(*record_);
  }
}

template <Severity SEVERITY>
Logger<SEVERITY>::~Logger() {
  if (stream_) {
    g_logger::get().push_record(
        boost::move(stream_->get_record()));  // flushes stream first
  }
}

template <>
BOOST_NORETURN Logger<Severity::FATAL>::~Logger();

template <Severity SEVERITY>
boost::log::record_ostream& Logger<SEVERITY>::stream(char const* file, int line) {
  return *stream_ << boost::filesystem::path(file).filename().native() << ':' << line
                  << ' ';
}

extern Severity g_min_active_severity;

// These macros risk inadvertent else-matching to the if statements,
// which are fortunately prevented by our clang-tidy requirements.
// These can be changed to for/while loops with slight performance degradation.

#define LOG(severity)                                                         \
  if (logger::g_min_active_severity <= logger::Severity::severity)            \
    if (auto _omnisci_logger_ = logger::Logger<logger::Severity::severity>()) \
  _omnisci_logger_.stream(__FILE__, __LINE__)

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
#define OMINSCI_CHECKOP_FUNCTION(name, op)                          \
  template <typename X, typename Y>                                 \
  inline std::string* Check##name(                                  \
      X const& x, Y const& y, char const* xstr, char const* ystr) { \
    if (BOOST_LIKELY(x op y))                                       \
      return nullptr;                                               \
    else                                                            \
      return logger::check_failed(x, y, xstr, ystr, " " #op " ");   \
  }
OMINSCI_CHECKOP_FUNCTION(EQ, ==)
OMINSCI_CHECKOP_FUNCTION(NE, !=)
OMINSCI_CHECKOP_FUNCTION(LT, <)
OMINSCI_CHECKOP_FUNCTION(LE, <=)
OMINSCI_CHECKOP_FUNCTION(GT, >)
OMINSCI_CHECKOP_FUNCTION(GE, >=)
#undef OMINSCI_CHECKOP_FUNCTION

#define UNREACHABLE() LOG(FATAL) << "UNREACHABLE "

#else  // __CUDACC__

// Provided for backward compatibility to allow code to compile.
// No logging is actually done, since cuda code should not log.
template <Severity severity>
class NullLogger {
 public:
  NullLogger() {
    if /* constexpr */ (severity == Severity::FATAL) {
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

}  // namespace logger

#endif  // SHARED_LOGGER_H
