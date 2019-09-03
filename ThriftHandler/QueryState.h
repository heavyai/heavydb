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

/*
 * File:   QueryState.h
 * Author: matt.pulver@omnisci.com
 *
 */

#ifndef OMNISCI_THRIFTHANDLER_QUERYSTATE_H
#define OMNISCI_THRIFTHANDLER_QUERYSTATE_H

#include "Shared/Logger.h"
#include "Shared/StringTransform.h"
#include "gen-cpp/mapd_types.h"

#include <boost/circular_buffer.hpp>
#include <boost/filesystem.hpp>
#include <boost/noncopyable.hpp>
#include <boost/optional.hpp>
#include <boost/preprocessor.hpp>

#include <atomic>
#include <chrono>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

/* See docs/QueryState.md for documentation.
 *
 * Outline:
 *  - Consolidate information about current and past queries.
 *  - Time and record code blocks, and save to stdlog.
 *
 * Classes:
 *  - StdLog - logs to DEBUG1 on construction, INFO on destruction.
 *  - QueryState - hold info on the evolution of a query.
 *  - QueryStateProxy - Light-weight QueryState wrapper to track Timer/Event nesting.
 *  - Timer - Created by QueryStateProxy to log function/block lifetime.
 *
 * Basic API Usage:
 *  - auto stdlog = STDLOG() // May include session_info, name/value pairs.
 *  - auto query_state = query_states_.create(get_session_ptr(session), query_str);
 *  - stdlog.setQueryState(query_state); // Log query_state when stdlog destructs.
 *  - auto query_state_proxy = query_state.createQueryStateProxy();
 *  - auto timer = query_state_proxy.createTimer(__func__);
 */

/* Notes:
 *  - QueryStates holds many QueryState instances.
 *  - QueryState holds many Event instances.
 *  - QueryState and QueryStateProxy can create Timers.
 *  - QueryState and Timer can create QueryStateProxies.
 *  - Timers and Events are one-to-one.
 *  - When StdLog destructs, it logs all of the associated QueryState's Events.
 */

namespace Catalog_Namespace {
class SessionInfo;
}

namespace query_state {

using Clock = std::chrono::steady_clock;
using Id = uint64_t;

struct Event;
using Events = std::list<Event>;

struct Event {
  char const* const name;
  Events::iterator const parent;  // events_.end() = top level query_state
  std::thread::id const thread_id;
  Clock::duration const started;
  // duration is used instead of time_point since it is trivially copyable for atomic.
  std::atomic<Clock::duration> stopped;
  Event(char const* const name, Events::iterator parent);
  template <typename Units = std::chrono::milliseconds>
  boost::optional<typename Units::rep> duration() const {
    auto const stop_time = stopped.load();
    return boost::make_optional(
        stop_time != Clock::duration::zero(),
        std::chrono::duration_cast<Units>(stop_time - started).count());
  }
  void stop();
};

using EventFunction = std::function<void(Event const&)>;

class QueryStateProxy;
class QueryStates;

// SessionInfo can expire, so data is copied while available.
struct SessionData {
  std::weak_ptr<Catalog_Namespace::SessionInfo const> session_info;
  std::string db_name;
  std::string user_name;
  std::string public_session_id;

  SessionData() = default;
  SessionData(std::shared_ptr<Catalog_Namespace::SessionInfo const> const&);
};

class Timer;

class QueryState : public std::enable_shared_from_this<QueryState> {
  static std::atomic<Id> s_next_id;
  Id const id_;
  boost::optional<SessionData> session_data_;
  std::string const query_str_;
  Events events_;
  mutable std::mutex events_mutex_;
  std::atomic<bool> logged_;
  void logCallStack(std::stringstream&, unsigned const depth, Events::iterator parent);

  // Only shared_ptr instances are allowed due to call to shared_from_this().
  QueryState(std::shared_ptr<Catalog_Namespace::SessionInfo const> const&,
             std::string query_str);

 public:
  template <typename... ARGS>
  static std::shared_ptr<QueryState> create(ARGS&&... args) {
    // Trick to call std::make_shared with private constructors.
    struct EnableMakeShared : public QueryState {
      EnableMakeShared(ARGS&&... args) : QueryState(std::forward<ARGS>(args)...) {}
    };
    return std::make_shared<EnableMakeShared>(std::forward<ARGS>(args)...);
  }
  QueryStateProxy createQueryStateProxy();
  QueryStateProxy createQueryStateProxy(Events::iterator parent);
  Timer createTimer(char const* event_name, Events::iterator parent);
  inline bool empty_log() const { return events_.empty() && query_str_.empty(); }
  inline Id get_id() const { return id_; }
  inline std::string const& get_query_str() const { return query_str_; }
  // Will throw exception if session_data_.session_info.expired().
  std::shared_ptr<Catalog_Namespace::SessionInfo const> getConstSessionInfo() const;
  SessionData const* get_session_data() const;
  inline bool is_logged() const { return logged_.load(); }
  void logCallStack(std::stringstream&);
  inline void set_logged(bool logged) { logged_.store(logged); }
  friend class QueryStates;
};

// Light-weight class used as intermediary between Timer objects spawning child Timers.
// Assumes lifetime is less than original QueryState.
class QueryStateProxy {
  QueryState& query_state_;
  Events::iterator const parent_;

 public:
  QueryStateProxy(QueryState& query_state, Events::iterator parent)
      : query_state_(query_state), parent_(parent) {}
  Timer createTimer(char const* event_name);
  QueryState& getQueryState() { return query_state_; }
};

// At this point it is not clear how long we want to keep completed queries.
// The data structure and lifetime are not currently settled upon.
class QueryStates {
  using CircleBuffer = boost::circular_buffer<std::shared_ptr<QueryState>>;
  // constexpr size_t MAX_SIZE_BEFORE_OVERWRITE = 128; // C++17
  CircleBuffer circle_buffer_{128};
  std::mutex circle_mutex_;

 public:
  template <typename... ARGS>
  CircleBuffer::value_type create(ARGS&&... args) {
    std::lock_guard<std::mutex> lock(circle_mutex_);
    /* Logic for ensuring QueryState objects are logged before deleting them.
        if (circle_buffer_.full() && !circle_buffer_.front()->is_logged()) {
          constexpr size_t MAX_SIZE_BEFORE_OVERWRITE = 128;
          if (circle_buffer_.size() < MAX_SIZE_BEFORE_OVERWRITE) {
            circle_buffer_.set_capacity(2 * circle_buffer_.capacity());
          } else {
            LOG(ERROR) << "QueryStates is holding " << circle_buffer_.size()
                       << " queries but the oldest query_state has not completed
       logging.";
          }
        }
    */
    circle_buffer_.push_back(QueryState::create(std::forward<ARGS>(args)...));
    return circle_buffer_.back();
  }
};

class Timer {
  std::shared_ptr<QueryState> query_state_;
  Events::iterator event_;  // = pointer into QueryState::events_

 public:
  Timer(std::shared_ptr<QueryState>&&, Events::iterator event);
  Timer(Timer const&) = delete;
  Timer& operator=(Timer const&) = delete;
  Timer(Timer&&) = default;
  Timer& operator=(Timer&&) = default;
  ~Timer();
  QueryStateProxy createQueryStateProxy();
};

// Log Format:
// YYYY-MM-DDTHH:MM::SS.FFFFFF [S] [pid] [file]:[line] [label] [func] [match] [dur_ms]
//    [dbname] [user] [pubsessid] {[names]} {[values]}
// Call at both beginning(label="stdlog_begin") and end(label="stdlog") of Thrift call,
//    with dur_ms = current age of StdLog object in milliseconds.
// stdlog_begin is logged at DEBUG1 level, stdlog is logged at INFO level.
// All remaining optional parameters are name,value pairs that will be included in log.
#define STDLOG(...)                                              \
  BOOST_PP_IF(BOOST_PP_IS_EMPTY(__VA_ARGS__),                    \
              query_state::StdLog(__FILE__, __LINE__, __func__), \
              query_state::StdLog(__FILE__, __LINE__, __func__, __VA_ARGS__))

class StdLogData {
 protected:
  static std::atomic<int64_t> s_match;
  std::string const file_;
  unsigned const line_;
  char const* const func_;
  Clock::time_point const start_;
  int64_t const match_;  // Unique to each begin/end pair to match them together.
  std::list<std::string> name_value_pairs_;
  template <typename... Pairs>
  StdLogData(char const* file, unsigned line, char const* func, Pairs&&... pairs)
      : file_(boost::filesystem::path(file).filename().string())
      , line_(line)
      , func_(func)
      , start_(Clock::now())
      , match_(s_match++)
      , name_value_pairs_{to_string(std::forward<Pairs>(pairs))...} {
    static_assert(sizeof...(Pairs) % 2 == 0,
                  "StdLogData() requires an even number of name/value parameters.");
  }
};

class StdLog : public StdLogData {
  std::shared_ptr<Catalog_Namespace::SessionInfo> session_info_;
  std::shared_ptr<QueryState> query_state_;
  void log(logger::Severity, char const* label);
  void logCallStack(logger::Severity, char const* label);

 public:
  template <typename... Pairs>
  StdLog(char const* file,
         unsigned line,
         char const* func,
         std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
         Pairs&&... pairs)
      : StdLogData(file, line, func, std::forward<Pairs>(pairs)...)
      , session_info_(std::move(session_info)) {
    log(logger::Severity::DEBUG1, "stdlog_begin");
  }

  template <typename... Pairs>
  StdLog(char const* file,
         unsigned line,
         char const* func,
         std::shared_ptr<Catalog_Namespace::SessionInfo> session_info,
         std::shared_ptr<QueryState> query_state,
         Pairs&&... pairs)
      : StdLogData(file, line, func, std::forward<Pairs>(pairs)...)
      , session_info_(std::move(session_info))
      , query_state_(std::move(query_state)) {
    log(logger::Severity::DEBUG1, "stdlog_begin");
  }

  template <typename... Pairs>
  StdLog(char const* file,
         unsigned line,
         char const* func,
         std::shared_ptr<QueryState> query_state,
         Pairs&&... pairs)
      : StdLogData(file, line, func, std::forward<Pairs>(pairs)...)
      , query_state_(std::move(query_state)) {
    log(logger::Severity::DEBUG1, "stdlog_begin");
  }
  template <typename... Pairs>
  StdLog(char const* file, unsigned line, char const* func, Pairs&&... pairs)
      : StdLogData(file, line, func, std::forward<Pairs>(pairs)...) {
    log(logger::Severity::DEBUG1, "stdlog_begin");
  }
  StdLog(StdLog const&) = delete;
  StdLog& operator=(StdLog const&) = delete;
  StdLog(StdLog&&) = default;
  StdLog& operator=(StdLog&&) = default;
  ~StdLog();
  template <typename... Pairs>
  void appendNameValuePairs(Pairs&&... pairs) {
    static_assert(sizeof...(Pairs) % 2 == 0,
                  "appendNameValuePairs() requires an even number of parameters.");
    name_value_pairs_.splice(name_value_pairs_.cend(),
                             {to_string(std::forward<Pairs>(pairs))...});
  }
  template <typename Units = std::chrono::milliseconds>
  typename Units::rep duration() const {
    return std::chrono::duration_cast<Units>(Clock::now() - start_).count();
  }
  std::shared_ptr<Catalog_Namespace::SessionInfo const> getConstSessionInfo() const;
  std::shared_ptr<Catalog_Namespace::SessionInfo> getSessionInfo() const;
  void setQueryState(std::shared_ptr<QueryState>);
  void setSessionInfo(std::shared_ptr<Catalog_Namespace::SessionInfo>);
};

}  // namespace query_state

#endif  // OMNISCI_THRIFTHANDLER_QUERYSTATE_H
