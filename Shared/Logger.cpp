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

#ifndef __CUDACC__

#include "Logger.h"
#include "StringTransform.h"

#include <boost/algorithm/string.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/sources/severity_feature.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/phoenix.hpp>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <regex>

namespace logger {

namespace attr = boost::log::attributes;
namespace expr = boost::log::expressions;
namespace keywords = boost::log::keywords;
namespace sinks = boost::log::sinks;
namespace sources = boost::log::sources;
namespace po = boost::program_options;

BOOST_LOG_ATTRIBUTE_KEYWORD(process_id, "ProcessID", attr::current_process_id::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(channel, "Channel", Channel)
BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", Severity)

BOOST_LOG_GLOBAL_LOGGER_DEFAULT(gChannelLogger, ChannelLogger)
BOOST_LOG_GLOBAL_LOGGER_DEFAULT(gSeverityLogger, SeverityLogger)

LogOptions::LogOptions(char const* argv0) : options_("Logging") {
  // Log file base_name matches name of program.
  std::string const base_name = argv0 == nullptr
                                    ? std::string("omnisci_server")
                                    : boost::filesystem::path(argv0).filename().string();
  file_name_pattern_ = base_name + file_name_pattern_;
  symlink_ = base_name + symlink_;
  std::string const channels = join(ChannelNames, " ");
  // Filter out DEBUG[1-4] severities from --help options
  std::string severities;
  for (auto const& name : SeverityNames) {
    if (!boost::algorithm::starts_with(name, "DEBUG")) {
      (severities += (severities.empty() ? "" : " ")) += name;
    }
  }
  options_.add_options()(
      "log-directory",
      po::value<boost::filesystem::path>(&log_dir_)->default_value(log_dir_),
      "Logging directory. May be relative to data directory, or absolute.");
  options_.add_options()(
      "log-file-name",
      po::value<std::string>(&file_name_pattern_)->default_value(file_name_pattern_),
      "Log file name relative to log-directory.");
  options_.add_options()("log-symlink",
                         po::value<std::string>(&symlink_)->default_value(symlink_),
                         "Symlink to active log.");
  options_.add_options()("log-severity",
                         po::value<Severity>(&severity_)->default_value(severity_),
                         ("Log to file severity level: " + severities).c_str());
  options_.add_options()(
      "log-severity-clog",
      po::value<Severity>(&severity_clog_)->default_value(severity_clog_),
      ("Log to console severity level: " + severities).c_str());
  options_.add_options()("log-channels",
                         po::value<Channels>(&channels_)->default_value(channels_),
                         ("Log channel debug info: " + channels).c_str());
  options_.add_options()("log-auto-flush",
                         po::value<bool>(&auto_flush_)->default_value(auto_flush_),
                         "Flush logging buffer to file after each message.");
  options_.add_options()("log-max-files",
                         po::value<size_t>(&max_files_)->default_value(max_files_),
                         "Maximum number of log files to keep.");
  options_.add_options()(
      "log-min-free-space",
      po::value<size_t>(&min_free_space_)->default_value(20 << 20),
      "Minimum number of bytes left on device before oldest log files are deleted.");
  options_.add_options()("log-rotate-daily",
                         po::value<bool>(&rotate_daily_)->default_value(true),
                         "Start new log files at midnight.");
  options_.add_options()("log-rotation-size",
                         po::value<size_t>(&rotation_size_)->default_value(10 << 20),
                         "Maximum file size in bytes before new log files are started.");
}

boost::filesystem::path LogOptions::full_log_dir() const {
  return log_dir_.has_root_directory() ? log_dir_ : base_path_ / log_dir_;
}

po::options_description const& LogOptions::get_options() const {
  return options_;
}

// Typical usage calls either get_options() or parse_command_line() but not both.
void LogOptions::parse_command_line(int argc, char const* const* argv) {
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(options_).allow_unregistered().run(),
      vm);
  po::notify(vm);  // Sets LogOptions member variables.
}

// Must be called before init() to take effect.
void LogOptions::set_base_path(std::string const& base_path) {
  base_path_ = base_path;
}

template <typename TAG>
std::string replace_braces(std::string const& str, TAG const tag) {
  constexpr std::regex::flag_type flags = std::regex::ECMAScript | std::regex::optimize;
  static std::regex const regex(R"(\{SEVERITY\})", flags);
  if /*constexpr*/ (std::is_same<TAG, Channel>::value) {
    return std::regex_replace(str, regex, ChannelNames[tag]);
  } else {
    return std::regex_replace(str, regex, SeverityNames[tag]);
  }
}

// Print decimal value for process_id (14620) instead of hex (0x0000391c)
boost::log::attributes::current_process_id::value_type::native_type get_native_process_id(
    boost::log::value_ref<boost::log::attributes::current_process_id::value_type,
                          tag::process_id> const& pid) {
  return pid ? pid->native_id() : 0;
}

template <typename SINK>
sinks::text_file_backend::open_handler_type create_or_replace_symlink(
    boost::weak_ptr<SINK> weak_ptr,
    std::string&& symlink) {
  namespace fs = boost::filesystem;
  return [weak_ptr,
          symlink = std::move(symlink)](sinks::text_file_backend::stream_type& stream) {
    if (boost::shared_ptr<SINK> sink = weak_ptr.lock()) {
      boost::system::error_code ec;
      fs::path const& file_name = sink->locked_backend()->get_current_file_name();
      fs::path const symlink_path = file_name.parent_path() / symlink;
      fs::remove(symlink_path, ec);
      if (ec) {
        stream << fs::path(__FILE__).filename().native() << ':' << __LINE__ << ' '
               << ec.message() << std::endl;
      }
      fs::create_symlink(file_name.filename(), symlink_path, ec);
      if (ec) {
        stream << fs::path(__FILE__).filename().native() << ':' << __LINE__ << ' '
               << ec.message() << std::endl;
      }
    }
  };
}

boost::log::formatting_ostream& operator<<(
    boost::log::formatting_ostream& strm,
    boost::log::to_log_manip<Channel, tag::channel> const& manip) {
  return strm << ChannelSymbols[manip.get()];
}

boost::log::formatting_ostream& operator<<(
    boost::log::formatting_ostream& strm,
    boost::log::to_log_manip<Severity, tag::severity> const& manip) {
  return strm << SeveritySymbols[manip.get()];
}

template <typename TAG, typename SINK>
void set_formatter(SINK& sink) {
  if /*constexpr*/ (std::is_same<TAG, Channel>::value) {
    sink->set_formatter(
        expr::stream << expr::format_date_time<boost::posix_time::ptime>(
                            "TimeStamp", "%Y-%m-%dT%H:%M:%S.%f")
                     << ' ' << channel << ' '
                     << boost::phoenix::bind(&get_native_process_id, process_id.or_none())
                     << ' ' << expr::smessage);
  } else {
    sink->set_formatter(
        expr::stream << expr::format_date_time<boost::posix_time::ptime>(
                            "TimeStamp", "%Y-%m-%dT%H:%M:%S.%f")
                     << ' ' << severity << ' '
                     << boost::phoenix::bind(&get_native_process_id, process_id.or_none())
                     << ' ' << expr::smessage);
  }
}

template <typename FILE_SINK, typename TAG>
boost::shared_ptr<FILE_SINK> make_sink(LogOptions const& log_opts,
                                       boost::filesystem::path const& full_log_dir,
                                       TAG const tag) {
  auto sink = boost::make_shared<FILE_SINK>(
      keywords::file_name =
          full_log_dir / replace_braces(log_opts.file_name_pattern_, tag),
      keywords::auto_flush = log_opts.auto_flush_,
      keywords::rotation_size = log_opts.rotation_size_);
  if /*constexpr*/ (std::is_same<TAG, Channel>::value) {
    sink->set_filter(channel == static_cast<Channel>(tag));
    set_formatter<Channel>(sink);
  } else {
    // INFO sink logs all other levels. Other sinks only log at their level or higher.
    Severity const min_filter_level = static_cast<Severity>(tag) == Severity::INFO
                                          ? log_opts.severity_
                                          : static_cast<Severity>(tag);
    sink->set_filter(min_filter_level <= severity);
    set_formatter<Severity>(sink);
  }
  typename FILE_SINK::locked_backend_ptr backend = sink->locked_backend();
  if (log_opts.rotate_daily_) {
    backend->set_time_based_rotation(sinks::file::rotation_at_time_point(0, 0, 0));
  }
  backend->set_file_collector(
      sinks::file::make_collector(keywords::target = full_log_dir,
                                  keywords::max_files = log_opts.max_files_,
                                  keywords::min_free_space = log_opts.min_free_space_));
  backend->set_open_handler(create_or_replace_symlink(
      boost::weak_ptr<FILE_SINK>(sink), replace_braces(log_opts.symlink_, tag)));
  backend->scan_for_files();
  return sink;
}

// Pointer to function to optionally call on LOG(FATAL).
std::atomic<FatalFunc> g_fatal_func{nullptr};
std::once_flag g_fatal_func_flag;

using ClogSync = sinks::synchronous_sink<sinks::text_ostream_backend>;
using FileSync = sinks::synchronous_sink<sinks::text_file_backend>;

template <typename CONSOLE_SINK>
boost::shared_ptr<CONSOLE_SINK> make_sink(LogOptions const& log_opts) {
  auto sink = boost::make_shared<CONSOLE_SINK>();
  boost::shared_ptr<std::ostream> clog(&std::clog, boost::null_deleter());
  sink->locked_backend()->add_stream(clog);
  sink->set_filter(log_opts.severity_clog_ <= severity);
  set_formatter<Severity>(sink);
  return sink;
}

// Locking/atomicity not needed for g_any_active_channels or g_min_active_severity
// as they are modifed by init() once.
bool g_any_active_channels{false};
Severity g_min_active_severity{Severity::FATAL};

void init(LogOptions const& log_opts) {
  boost::shared_ptr<boost::log::core> core = boost::log::core::get();
  // boost::log::add_common_attributes(); // LineID TimeStamp ProcessID ThreadID
  core->add_global_attribute("TimeStamp", attr::local_clock());
  core->add_global_attribute("ProcessID", attr::current_process_id());
  if (0 < log_opts.max_files_) {
    boost::filesystem::path const full_log_dir = log_opts.full_log_dir();
    bool const log_dir_was_created = boost::filesystem::create_directory(full_log_dir);
    // Don't create separate log sinks for anything less than Severity::INFO.
    Severity const min_sink_level = std::max(Severity::INFO, log_opts.severity_);
    for (int i = min_sink_level; i < Severity::_NSEVERITIES; ++i) {
      Severity const level = static_cast<Severity>(i);
      core->add_sink(make_sink<FileSync>(log_opts, full_log_dir, level));
    }
    g_min_active_severity = std::min(g_min_active_severity, log_opts.severity_);
    if (log_dir_was_created) {
      LOG(INFO) << "Log directory(" << full_log_dir.native() << ") created.";
    }
    for (auto const channel : log_opts.channels_) {
      core->add_sink(make_sink<FileSync>(log_opts, full_log_dir, channel));
    }
    g_any_active_channels = !log_opts.channels_.empty();
  }
  core->add_sink(make_sink<ClogSync>(log_opts));
  g_min_active_severity = std::min(g_min_active_severity, log_opts.severity_clog_);
}

void set_once_fatal_func(FatalFunc fatal_func) {
  if (g_fatal_func.exchange(fatal_func)) {
    throw std::runtime_error(
        "logger::set_once_fatal_func() should not be called more than once.");
  }
}

void shutdown() {
  boost::log::core::get()->remove_all_sinks();
}

// Used by boost::program_options when parsing enum Channel.
std::istream& operator>>(std::istream& in, Channels& channels) {
  std::string line;
  std::getline(in, line);
  std::regex const rex(R"(\w+)");
  using TokenItr = std::regex_token_iterator<std::string::iterator>;
  TokenItr const end;
  for (TokenItr tok(line.begin(), line.end(), rex); tok != end; ++tok) {
    auto itr = std::find(ChannelNames.cbegin(), ChannelNames.cend(), *tok);
    if (itr == ChannelNames.cend()) {
      in.setstate(std::ios_base::failbit);
      break;
    } else {
      channels.emplace(static_cast<Channel>(itr - ChannelNames.cbegin()));
    }
  }
  return in;
}

// Used by boost::program_options when stringifying Channels.
std::ostream& operator<<(std::ostream& out, Channels const& channels) {
  int i = 0;
  for (auto const channel : channels) {
    out << (i++ ? " " : "") << ChannelNames.at(channel);
  }
  return out;
}

// Used by boost::program_options when parsing enum Severity.
std::istream& operator>>(std::istream& in, Severity& sev) {
  std::string token;
  in >> token;
  auto itr = std::find(SeverityNames.cbegin(), SeverityNames.cend(), token);
  if (itr == SeverityNames.cend()) {
    in.setstate(std::ios_base::failbit);
  } else {
    sev = static_cast<Severity>(itr - SeverityNames.cbegin());
  }
  return in;
}

// Used by boost::program_options when stringifying enum Severity.
std::ostream& operator<<(std::ostream& out, Severity const& sev) {
  return out << SeverityNames.at(sev);
}

Logger::Logger(Channel channel)
    : is_channel_(true)
    , enum_value_(channel)
    , record_(std::make_unique<boost::log::record>(
          gChannelLogger::get().open_record(boost::log::keywords::channel = channel))) {
  if (*record_) {
    stream_ = std::make_unique<boost::log::record_ostream>(*record_);
  }
}

Logger::Logger(Severity severity)
    : is_channel_(false)
    , enum_value_(severity)
    , record_(std::make_unique<boost::log::record>(gSeverityLogger::get().open_record(
          boost::log::keywords::severity = severity))) {
  if (*record_) {
    stream_ = std::make_unique<boost::log::record_ostream>(*record_);
  }
}

Logger::~Logger() {
  if (stream_) {
    if (is_channel_) {
      gChannelLogger::get().push_record(boost::move(stream_->get_record()));
    } else {
      gSeverityLogger::get().push_record(boost::move(stream_->get_record()));
    }
  }
  if (!is_channel_ && static_cast<Severity>(enum_value_) == Severity::FATAL) {
    if (FatalFunc fatal_func = g_fatal_func.load()) {
      // set_once_fatal_func() prevents race condition.
      // Exceptions thrown by (*fatal_func)() are propagated here.
      std::call_once(g_fatal_func_flag, *fatal_func);
    }
    abort();
  }
}

Logger::operator bool() const {
  return static_cast<bool>(stream_);
}

boost::log::record_ostream& Logger::stream(char const* file, int line) {
  return *stream_ << boost::filesystem::path(file).filename().native() << ':' << line
                  << ' ';
}

}  // namespace logger

#endif  // #ifndef __CUDACC__
