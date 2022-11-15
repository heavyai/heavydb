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

bool g_enable_debug_timer{false};

#ifndef __CUDACC__

#include "Logger.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <boost/algorithm/string.hpp>
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
#include <boost/smart_ptr/weak_ptr.hpp>
#include <boost/variant.hpp>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <regex>

#include "Shared/SysDefinitions.h"
#include "Shared/nvtx_helpers.h"

namespace logger {
using ChannelLogger = boost::log::sources::channel_logger_mt<Channel>;
BOOST_LOG_GLOBAL_LOGGER(gChannelLogger_IR, ChannelLogger)
BOOST_LOG_GLOBAL_LOGGER(gChannelLogger_PTX, ChannelLogger)
BOOST_LOG_GLOBAL_LOGGER(gChannelLogger_ASM, ChannelLogger)

using SeverityLogger = boost::log::sources::severity_logger_mt<Severity>;
BOOST_LOG_GLOBAL_LOGGER(gSeverityLogger, SeverityLogger)

namespace attr = boost::log::attributes;
namespace expr = boost::log::expressions;
namespace fs = boost::filesystem;
namespace keywords = boost::log::keywords;
namespace sinks = boost::log::sinks;
namespace sources = boost::log::sources;
namespace po = boost::program_options;

BOOST_LOG_ATTRIBUTE_KEYWORD(process_id, "ProcessID", attr::current_process_id::value_type)
BOOST_LOG_ATTRIBUTE_KEYWORD(channel, "Channel", Channel)
BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", Severity)

BOOST_LOG_GLOBAL_LOGGER_CTOR_ARGS(gChannelLogger_IR,
                                  ChannelLogger,
                                  (keywords::channel = IR))
BOOST_LOG_GLOBAL_LOGGER_CTOR_ARGS(gChannelLogger_PTX,
                                  ChannelLogger,
                                  (keywords::channel = PTX))
BOOST_LOG_GLOBAL_LOGGER_CTOR_ARGS(gChannelLogger_ASM,
                                  ChannelLogger,
                                  (keywords::channel = ASM))
BOOST_LOG_GLOBAL_LOGGER_DEFAULT(gSeverityLogger, SeverityLogger)

// Return last component of path
std::string filename(char const* path) {
  return fs::path(path).filename().string();
}

LogOptions::LogOptions(char const* argv0)
    : log_dir_(std::make_unique<fs::path>(shared::kDefaultLogDirName)) {
  // Log file base_name matches name of program.
  std::string const base_name =
      argv0 == nullptr ? std::string("heavydb") : filename(argv0);
  file_name_pattern_ = base_name + file_name_pattern_;
  symlink_ = base_name + symlink_;
  set_options();
}

// Needed to allow forward declarations within std::unique_ptr.
LogOptions::~LogOptions() {}

fs::path LogOptions::full_log_dir() const {
  return log_dir_->has_root_directory() ? *log_dir_ : base_path_ / *log_dir_;
}

po::options_description const& LogOptions::get_options() const {
  return *options_;
}

// Typical usage calls either get_options() or parse_command_line() but not both.
void LogOptions::parse_command_line(int argc, char const* const* argv) {
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(*options_).allow_unregistered().run(),
      vm);
  po::notify(vm);  // Sets LogOptions member variables.
}

// Must be called before init() to take effect.
void LogOptions::set_base_path(std::string const& base_path) {
  base_path_ = base_path;
}

// May be called to update default values based on updated member variables.
void LogOptions::set_options() {
  options_ = std::make_unique<boost::program_options::options_description>("Logging");
  std::string const channels =
      std::accumulate(ChannelNames.cbegin() + 1,
                      ChannelNames.cend(),
                      std::string(ChannelNames.front()),
                      [](auto a, auto channel) { return a + ' ' + channel; });
  // Filter out DEBUG[1-4] severities from --help options
  std::string severities;
  for (auto const& name : SeverityNames) {
    if (!boost::algorithm::starts_with(name, "DEBUG")) {
      (severities += (severities.empty() ? "" : " ")) += name;
    }
  }
  options_->add_options()(
      "log-directory",
      po::value<fs::path>(&*log_dir_)->default_value(*log_dir_),
      "Logging directory. May be relative to data directory, or absolute.");
  options_->add_options()(
      "log-file-name",
      po::value<std::string>(&file_name_pattern_)->default_value(file_name_pattern_),
      "Log file name relative to log-directory.");
  options_->add_options()("log-symlink",
                          po::value<std::string>(&symlink_)->default_value(symlink_),
                          "Symlink to active log.");
  options_->add_options()("log-severity",
                          po::value<Severity>(&severity_)->default_value(severity_),
                          ("Log to file severity level: " + severities).c_str());
  options_->add_options()(
      "log-severity-clog",
      po::value<Severity>(&severity_clog_)->default_value(severity_clog_),
      ("Log to console severity level: " + severities).c_str());
  options_->add_options()("log-channels",
                          po::value<Channels>(&channels_)->default_value(channels_),
                          ("Log channel debug info: " + channels).c_str());
  options_->add_options()("log-auto-flush",
                          po::value<bool>(&auto_flush_)->default_value(auto_flush_),
                          "Flush logging buffer to file after each message.");
  options_->add_options()("log-max-files",
                          po::value<size_t>(&max_files_)->default_value(max_files_),
                          "Maximum number of log files to keep.");
  options_->add_options()(
      "log-min-free-space",
      po::value<size_t>(&min_free_space_)->default_value(min_free_space_),
      "Minimum number of bytes left on device before oldest log files are deleted.");
  options_->add_options()("log-rotate-daily",
                          po::value<bool>(&rotate_daily_)->default_value(rotate_daily_),
                          "Start new log files at midnight.");
  options_->add_options()(
      "log-rotation-size",
      po::value<size_t>(&rotation_size_)->default_value(rotation_size_),
      "Maximum file size in bytes before new log files are started.");
}

template <typename TAG>
std::string replace_braces(std::string const& str, TAG const tag) {
  constexpr std::regex::flag_type flags = std::regex::ECMAScript | std::regex::optimize;
  static std::regex const regex(R"(\{SEVERITY\})", flags);
  if constexpr (std::is_same<TAG, Channel>::value) {
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
  return [weak_ptr,
          symlink = std::move(symlink)](sinks::text_file_backend::stream_type& stream) {
    if (boost::shared_ptr<SINK> sink = weak_ptr.lock()) {
      boost::system::error_code ec;
      fs::path const& file_name = sink->locked_backend()->get_current_file_name();
      fs::path const symlink_path = file_name.parent_path() / symlink;
      fs::remove(symlink_path, ec);
      if (ec) {
        stream << filename(__FILE__) << ':' << __LINE__ << ' ' << ec.message() << '\n';
      }
      fs::create_symlink(file_name.filename(), symlink_path, ec);
      if (ec) {
        stream << filename(__FILE__) << ':' << __LINE__ << ' ' << ec.message() << '\n';
      }
    }
  };
}

// Remove symlink if referent file does not exist.
struct RemoveDeadLink {
  void operator()(fs::path const& symlink_path) const {
    boost::system::error_code ec;
    bool const exists = fs::exists(symlink_path, ec);
    // If the symlink or referent file doesn't exist, ec.message() = "No such file or
    // directory" even though it's not really an error, so we don't bother checking it.
    if (!exists) {
      fs::remove(symlink_path, ec);
      if (ec) {
        std::cerr << "Error removing " << symlink_path << ": " << ec.message()
                  << std::endl;
      }
    }
  }
};

// Custom file collector that also deletes invalid symlinks.
class Collector : public sinks::file::collector {
  boost::shared_ptr<sinks::file::collector> collector_;
  std::vector<fs::path> symlink_paths_;

 public:
  Collector(fs::path const& full_log_dir, LogOptions const& log_opts)
      : collector_(sinks::file::make_collector(
            keywords::target = full_log_dir,
            keywords::max_files = log_opts.max_files_,
            keywords::min_free_space = log_opts.min_free_space_)) {}
  // Remove dead symlinks after rotated files are deleted.
  void store_file(fs::path const& path) override {
    collector_->store_file(path);  // Deletes files that exceed rotation limits.
    std::for_each(symlink_paths_.begin(), symlink_paths_.end(), RemoveDeadLink{});
  }
#if 107900 <= BOOST_VERSION
  bool is_in_storage(fs::path const& path) const override {
    return collector_->is_in_storage(path);
  }
  sinks::file::scan_result scan_for_files(sinks::file::scan_method method,
                                          fs::path const& path = fs::path()) override {
    return collector_->scan_for_files(method, path);
  }
#else
  uintmax_t scan_for_files(sinks::file::scan_method method,
                           fs::path const& path = fs::path(),
                           unsigned* counter = nullptr) override {
    return collector_->scan_for_files(method, path, counter);
  }
#endif
  void track_symlink(fs::path symlink_path) {
    symlink_paths_.emplace_back(std::move(symlink_path));
  }
};

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
  if constexpr (std::is_same<TAG, Channel>::value) {
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
boost::shared_ptr<FILE_SINK> make_sink(boost::shared_ptr<Collector>& collector,
                                       LogOptions const& log_opts,
                                       fs::path const& full_log_dir,
                                       TAG const tag) {
  auto sink = boost::make_shared<FILE_SINK>(
      keywords::file_name =
          full_log_dir / replace_braces(log_opts.file_name_pattern_, tag),
      keywords::auto_flush = log_opts.auto_flush_,
      keywords::rotation_size = log_opts.rotation_size_);
  if constexpr (std::is_same<TAG, Channel>::value) {
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
  collector->track_symlink(full_log_dir / replace_braces(log_opts.symlink_, tag));
  backend->set_file_collector(collector);
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

static fs::path g_log_dir_path;

void init(LogOptions const& log_opts) {
  boost::shared_ptr<boost::log::core> core = boost::log::core::get();
  // boost::log::add_common_attributes(); // LineID TimeStamp ProcessID ThreadID
  core->add_global_attribute("TimeStamp", attr::local_clock());
  core->add_global_attribute("ProcessID", attr::current_process_id());
  if (0 < log_opts.max_files_) {
    fs::path const full_log_dir = log_opts.full_log_dir();
    auto collector = boost::make_shared<Collector>(full_log_dir, log_opts);
    bool const log_dir_was_created = fs::create_directory(full_log_dir);
    // Don't create separate log sinks for anything less than Severity::INFO.
    Severity const min_sink_level = std::max(Severity::INFO, log_opts.severity_);
    for (int i = min_sink_level; i < Severity::_NSEVERITIES; ++i) {
      Severity const level = static_cast<Severity>(i);
      core->add_sink(make_sink<FileSync>(collector, log_opts, full_log_dir, level));
    }
    g_min_active_severity = std::min(g_min_active_severity, log_opts.severity_);
    if (log_dir_was_created) {
      LOG(INFO) << "Log directory(" << full_log_dir.native() << ") created.";
    }
    for (auto const channel : log_opts.channels_) {
      core->add_sink(make_sink<FileSync>(collector, log_opts, full_log_dir, channel));
    }
    g_any_active_channels = !log_opts.channels_.empty();
  }
  core->add_sink(make_sink<ClogSync>(log_opts));
  g_min_active_severity = std::min(g_min_active_severity, log_opts.severity_clog_);
  nvtx_helpers::init();
  g_log_dir_path = log_opts.full_log_dir();
}

void set_once_fatal_func(FatalFunc fatal_func) {
  if (g_fatal_func.exchange(fatal_func)) {
    throw std::runtime_error(
        "logger::set_once_fatal_func() should not be called more than once.");
  }
}

void shutdown() {
  static std::once_flag logger_flag;
  std::call_once(logger_flag, []() {
    boost::log::core::get()->remove_all_sinks();
    nvtx_helpers::shutdown();
  });
}

namespace {

// Remove quotes if they match from beginning and end of string.
// Does not check for escaped quotes within string.
void unquote(std::string& str) {
  if (1 < str.size() && (str.front() == '\'' || str.front() == '"') &&
      str.front() == str.back()) {
    str.erase(str.size() - 1, 1);
    str.erase(0, 1);
  }
}

}  // namespace

// Used by boost::program_options when parsing enum Channel.
std::istream& operator>>(std::istream& in, Channels& channels) {
  std::string line;
  std::getline(in, line);
  unquote(line);
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
  unquote(token);
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

namespace {
std::atomic<RequestId> g_next_request_id{1};
std::atomic<ThreadId> g_next_thread_id{1};

thread_local ThreadLocalIds g_thread_local_ids(0, 0);

ChannelLogger& get_channel_logger(Channel const channel) {
  switch (channel) {
    default:
    case IR:
      return gChannelLogger_IR::get();
    case PTX:
      return gChannelLogger_PTX::get();
    case ASM:
      return gChannelLogger_ASM::get();
  }
}
}  // namespace

Logger::Logger(Channel channel)
    : is_channel_(true)
    , enum_value_(channel)
    , record_(std::make_unique<boost::log::record>(
          get_channel_logger(channel).open_record())) {
  if (*record_) {
    stream_ = std::make_unique<boost::log::record_ostream>(*record_);
  }
}

Logger::Logger(Severity severity)
    : is_channel_(false)
    , enum_value_(severity)
    , record_(std::make_unique<boost::log::record>(
          gSeverityLogger::get().open_record(keywords::severity = severity))) {
  if (*record_) {
    stream_ = std::make_unique<boost::log::record_ostream>(*record_);
  }
}

Logger::~Logger() {
  if (stream_) {
    if (is_channel_) {
      get_channel_logger(static_cast<Channel>(enum_value_))
          .push_record(boost::move(stream_->get_record()));
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

// Assume *this is from the parent thread. Set current g_thread_local_ids.
// The new thread is assumed to have the same request_id as the parent thread.
LocalIdsScopeGuard ThreadLocalIds::setNewThreadId() const {
  ThreadLocalIds const prev_thread_local_ids = g_thread_local_ids;
  g_thread_local_ids = ThreadLocalIds(request_id_, g_next_thread_id++);
  return {prev_thread_local_ids, g_thread_local_ids};
}

LocalIdsScopeGuard::~LocalIdsScopeGuard() {
  if (enabled_) {
#ifndef NDEBUG
    CHECK_EQ(thread_local_ids_.request_id_, g_thread_local_ids.request_id_);
    CHECK_EQ(thread_local_ids_.thread_id_, g_thread_local_ids.thread_id_);
#endif
    g_thread_local_ids = prev_local_ids_;
  }
}

boost::log::record_ostream& Logger::stream(char const* file, int line) {
  return *stream_ << g_thread_local_ids.request_id_ << ' '
                  << g_thread_local_ids.thread_id_ << ' ' << filename(file) << ':' << line
                  << ' ';
}

// DebugTimer-related classes and functions.
using Clock = std::chrono::steady_clock;

class DurationTree;

struct DebugTimerParams {
  Severity severity_;
  char const* file_;
  int line_;
  char const* name_;
};

class Duration {
  DurationTree* const duration_tree_;
  Clock::time_point const start_;
  Clock::time_point stop_;

 public:
  int const depth_;
  DebugTimerParams const debug_timer_params_;

  Duration(DurationTree* const duration_tree,
           int const depth,
           DebugTimerParams const debug_timer_params)
      : duration_tree_(duration_tree)
      , start_(Clock::now())
      , depth_(depth)
      , debug_timer_params_(debug_timer_params) {}
  bool stop();
  // Start time relative to parent DurationTree::start_.
  template <typename Units = std::chrono::milliseconds>
  typename Units::rep relative_start_time() const;
  // Duration value = stop_ - start_.
  template <typename Units = std::chrono::milliseconds>
  typename Units::rep value() const;
  DebugTimerParams const* operator->() const { return &debug_timer_params_; }
};

using DurationTreeNode = boost::variant<Duration, DurationTree&>;
using DurationTreeNodes = std::deque<DurationTreeNode>;

class DurationTree {
  DurationTreeNodes durations_;
  int current_depth_;  //< Depth of next DurationTreeNode.

 public:
  int const depth_;  //< Depth of tree within parent tree, 0 for root tree.
  Clock::time_point const start_;
  ThreadId const thread_id_;
  DurationTree(ThreadId thread_id, int start_depth)
      // Add +1 to current_depth_ for non-root DurationTrees for extra indentation.
      : current_depth_(start_depth + bool(start_depth))
      , depth_(start_depth)
      , start_(Clock::now())
      , thread_id_(thread_id) {}
  void pushDurationTree(DurationTree& duration_tree) {
    durations_.emplace_back(duration_tree);
  }
  const Duration& rootDuration() const {
    CHECK(!durations_.empty());
    return boost::get<Duration>(durations_.front());
  }
  int currentDepth() const { return current_depth_; }
  void decrementDepth() { --current_depth_; }
  DurationTreeNodes const& durations() const { return durations_; }
  Duration* newDuration(DebugTimerParams const debug_timer_params) {
    durations_.emplace_back(Duration(this, current_depth_++, debug_timer_params));
    return boost::get<Duration>(&durations_.back());
  }
};

/// Set stop_, decrement DurationTree::current_depth_.
/// Return true iff this Duration represents the root timer (see docs).
bool Duration::stop() {
  stop_ = Clock::now();
  duration_tree_->decrementDepth();
  return depth_ == 0;
}

template <typename Units>
typename Units::rep Duration::relative_start_time() const {
  return std::chrono::duration_cast<Units>(start_ - duration_tree_->start_).count();
}

template <typename Units>
typename Units::rep Duration::value() const {
  return std::chrono::duration_cast<Units>(stop_ - start_).count();
}

struct GetDepth : boost::static_visitor<int> {
  int operator()(Duration const& duration) const { return duration.depth_; }
  int operator()(DurationTree const& duration_tree) const { return duration_tree.depth_; }
};

using DurationTreeMap = std::unordered_map<ThreadId, std::unique_ptr<DurationTree>>;

std::mutex g_duration_tree_map_mutex;
DurationTreeMap g_duration_tree_map;

Duration* newDuration(DebugTimerParams const debug_timer_params) {
  if (g_enable_debug_timer) {
    if (g_thread_local_ids.thread_id_) {
      std::lock_guard<std::mutex> lock_guard(g_duration_tree_map_mutex);
      auto& duration_tree_ptr = g_duration_tree_map[g_thread_local_ids.thread_id_];
      if (!duration_tree_ptr) {
        duration_tree_ptr =
            std::make_unique<DurationTree>(g_thread_local_ids.thread_id_, 0);
      }
      return duration_tree_ptr->newDuration(debug_timer_params);
    }
    LOG(ERROR) << "DEBUG_TIMER(" << debug_timer_params.name_
               << ") must not be called from the root thread(0) at "
               << debug_timer_params.file_ << ':' << debug_timer_params.line_
               << ". New threads require DEBUG_TIMER_NEW_THREAD() to be called first.";
  }
  return nullptr;  // Inactive - don't measure or report timing.
}

std::ostream& operator<<(std::ostream& os, Duration const& duration) {
  return os << std::setw(2 * duration.depth_) << ' ' << duration.value() << "ms start("
            << duration.relative_start_time() << "ms) " << duration->name_ << ' '
            << filename(duration->file_) << ':' << duration->line_;
}

std::ostream& operator<<(std::ostream& os, DurationTree const& duration_tree) {
  os << std::setw(2 * duration_tree.depth_) << ' ' << "New thread("
     << duration_tree.thread_id_ << ')';
  for (auto const& duration_tree_node : duration_tree.durations()) {
    os << '\n' << duration_tree_node;
  }
  return os << '\n'
            << std::setw(2 * duration_tree.depth_) << ' ' << "End thread("
            << duration_tree.thread_id_ << ')';
}

// Only called by logAndEraseDurationTree() on root tree
boost::log::record_ostream& operator<<(boost::log::record_ostream& os,
                                       DurationTreeMap::const_reference kv_pair) {
  auto itr = kv_pair.second->durations().cbegin();
  auto const end = kv_pair.second->durations().cend();
  auto const& root_duration = boost::get<Duration>(*itr);
  os << "DEBUG_TIMER thread_id(" << kv_pair.first << ")\n"
     << root_duration.value() << "ms total duration for " << root_duration->name_;
  for (++itr; itr != end; ++itr) {
    os << '\n' << *itr;
  }
  return os;
}

// Encode DurationTree into json.
// DurationTrees encode parent/child relationships using a combination
// of ordered descendents, and an int depth member value for each node.
// Direct-child nodes are those which:
//  * do not come after a node of depth <= parent_depth
//  * have depth == parent_depth + 1
class JsonEncoder : boost::static_visitor<rapidjson::Value> {
  std::shared_ptr<rapidjson::Document> doc_;
  rapidjson::Document::AllocatorType& alloc_;
  // Iterators are used to determine children when visiting a Duration node.
  DurationTreeNodes::const_iterator begin_;
  DurationTreeNodes::const_iterator end_;

  JsonEncoder(JsonEncoder& json_encoder,
              DurationTreeNodes::const_iterator begin,
              DurationTreeNodes::const_iterator end)
      : doc_(json_encoder.doc_), alloc_(doc_->GetAllocator()), begin_(begin), end_(end) {}

 public:
  JsonEncoder()
      : doc_(std::make_shared<rapidjson::Document>(rapidjson::kObjectType))
      , alloc_(doc_->GetAllocator()) {}
  rapidjson::Value operator()(Duration const& duration) {
    rapidjson::Value retval(rapidjson::kObjectType);
    retval.AddMember("type", "duration", alloc_);
    retval.AddMember("duration_ms", rapidjson::Value(duration.value()), alloc_);
    retval.AddMember(
        "start_ms", rapidjson::Value(duration.relative_start_time()), alloc_);
    retval.AddMember("name", rapidjson::StringRef(duration->name_), alloc_);
    retval.AddMember("file", filename(duration->file_), alloc_);
    retval.AddMember("line", rapidjson::Value(duration->line_), alloc_);
    retval.AddMember("children", childNodes(duration.depth_), alloc_);
    return retval;
  }
  rapidjson::Value operator()(DurationTree const& duration_tree) {
    begin_ = duration_tree.durations().cbegin();
    end_ = duration_tree.durations().cend();
    rapidjson::Value retval(rapidjson::kObjectType);
    retval.AddMember("type", "duration_tree", alloc_);
    retval.AddMember("thread_id", std::to_string(duration_tree.thread_id_), alloc_);
    retval.AddMember("children", childNodes(duration_tree.depth_), alloc_);
    return retval;
  }
  rapidjson::Value childNodes(int const parent_depth) {
    GetDepth const get_depth;
    rapidjson::Value children(rapidjson::kArrayType);
    for (auto itr = begin_; itr != end_; ++itr) {
      int const depth = apply_visitor(get_depth, *itr);
      if (depth <= parent_depth) {
        break;
      }
      if (depth == parent_depth + 1) {
        JsonEncoder json_encoder(*this, std::next(itr), end_);
        children.PushBack(apply_visitor(json_encoder, *itr), alloc_);
      }
    }
    return children;
  }
  // The root Duration is the "timer" node in the top level debug json object.
  // Only root Duration has overall total_duration_ms.
  rapidjson::Value timer(DurationTreeMap::const_reference kv_pair) {
    begin_ = kv_pair.second->durations().cbegin();
    end_ = kv_pair.second->durations().cend();
    rapidjson::Value retval(rapidjson::kObjectType);
    if (begin_ != end_) {
      auto const& root_duration = boost::get<Duration>(*(begin_++));
      retval.AddMember("type", "root", alloc_);
      retval.AddMember("thread_id", std::to_string(kv_pair.first), alloc_);
      retval.AddMember(
          "total_duration_ms", rapidjson::Value(root_duration.value()), alloc_);
      retval.AddMember("name", rapidjson::StringRef(root_duration->name_), alloc_);
      retval.AddMember("children", childNodes(0), alloc_);
    }
    return retval;
  }
  // Assumes *doc_ is empty.
  std::string str(DurationTreeMap::const_reference kv_pair) {
    doc_->AddMember("timer", timer(kv_pair), alloc_);
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc_->Accept(writer);
    return {buffer.GetString(), buffer.GetSize()};
  }
};

/// Depth-first search and erase all DurationTrees. Not thread-safe.
struct EraseDurationTrees : boost::static_visitor<> {
  void operator()(DurationTreeMap::const_iterator const& itr) const {
    for (auto const& duration_tree_node : itr->second->durations()) {
      apply_visitor(*this, duration_tree_node);
    }
    g_duration_tree_map.erase(itr);
  }
  void operator()(Duration const&) const {}
  void operator()(DurationTree const& duration_tree) const {
    for (auto const& duration_tree_node : duration_tree.durations()) {
      apply_visitor(*this, duration_tree_node);
    }
    g_duration_tree_map.erase(duration_tree.thread_id_);
  }
};

void logAndEraseDurationTree(std::string* json_str) {
  std::lock_guard<std::mutex> lock_guard(g_duration_tree_map_mutex);
  DurationTreeMap::const_iterator const itr =
      g_duration_tree_map.find(g_thread_local_ids.thread_id_);
  CHECK(itr != g_duration_tree_map.cend());
  auto const& root_duration = itr->second->rootDuration();
  if (auto log = Logger(root_duration->severity_)) {
    log.stream(root_duration->file_, root_duration->line_) << *itr;
  }
  if (json_str) {
    JsonEncoder json_encoder;
    *json_str = json_encoder.str(*itr);
  }
  EraseDurationTrees erase_duration_trees;
  erase_duration_trees(itr);
}

DebugTimer::DebugTimer(Severity severity, char const* file, int line, char const* name)
    : duration_(newDuration({severity, file, line, name})) {
  nvtx_helpers::omnisci_range_push(nvtx_helpers::Category::kDebugTimer, name, file);
}

DebugTimer::~DebugTimer() {
  stop();
  nvtx_helpers::omnisci_range_pop();
}

void DebugTimer::stop() {
  if (duration_) {
    if (duration_->stop()) {
      logAndEraseDurationTree(nullptr);
    }
    duration_ = nullptr;
  }
}

std::string DebugTimer::stopAndGetJson() {
  std::string json_str;
  if (duration_) {
    if (duration_->stop()) {
      logAndEraseDurationTree(&json_str);
    }
    duration_ = nullptr;
  }
  return json_str;
}

/// Call this when a new thread is spawned that will have timers that need to be
/// associated with timers on the parent thread.
/// Required: g_thread_local_ids.thread_id_ is not yet in g_duration_tree_map.
void debug_timer_new_thread(ThreadId const parent_thread_id) {
  std::lock_guard<std::mutex> lock_guard(g_duration_tree_map_mutex);
  auto const parent_itr = g_duration_tree_map.find(parent_thread_id);
  CHECK(parent_itr != g_duration_tree_map.end()) << parent_thread_id;
  auto const current_depth = parent_itr->second->currentDepth();
  auto const emplaced = g_duration_tree_map.emplace(
      g_thread_local_ids.thread_id_,
      std::make_unique<DurationTree>(g_thread_local_ids.thread_id_, current_depth + 1));
  CHECK(emplaced.second) << "ThreadId " << g_thread_local_ids.thread_id_
                         << " already in map.";
  parent_itr->second->pushDurationTree(*emplaced.first->second);
}

RequestId request_id() {
  return g_thread_local_ids.request_id_;
}
ThreadId thread_id() {
  return g_thread_local_ids.thread_id_;
}
ThreadLocalIds thread_local_ids() {
  return g_thread_local_ids;
}

// For example KafkaMgr::sql_execute() calls
//  1. db_handler_->internal_connect()
//  2. db_handler_->sql_execute()
//  3. db_handler_->disconnect()
// sequentially on the same thread. The thread_id is assigned only on the first call.
RequestId set_new_request_id() {
  if (g_thread_local_ids.thread_id_ == 0) {
    g_thread_local_ids.thread_id_ = g_next_thread_id++;
  }
  g_thread_local_ids.request_id_ = g_next_request_id++;
  return g_thread_local_ids.request_id_;
}

void set_request_id(RequestId const request_id) {
  if (g_thread_local_ids.thread_id_ == 0) {
    g_thread_local_ids.thread_id_ = g_next_thread_id++;
  }
  g_thread_local_ids.request_id_ = request_id;
}

fs::path get_log_dir_path() {
  return fs::canonical(g_log_dir_path);
}
}  // namespace logger

#endif  // #ifndef __CUDACC__
