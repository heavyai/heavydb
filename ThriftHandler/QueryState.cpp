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
 * File:   QueryState.cpp
 * Author: matt.pulver@omnisci.com
 *
 */

#include "QueryState.h"
#include "Catalog/Catalog.h"
#include "Catalog/SessionInfo.h"

#include <algorithm>
#include <iomanip>
#include <string_view>
#include <unordered_map>

namespace query_state {

Event::Event(char const* name, Events::iterator parent)
    : name(name)
    , parent(parent)
    , thread_id(std::this_thread::get_id())
    , started(Clock::now().time_since_epoch())
    , stopped(Clock::duration::zero()) {}

void Event::stop() {
  CHECK(stopped.exchange(Clock::now().time_since_epoch()) == Clock::duration::zero())
      << "stop() called more than once.";
}

SessionData::SessionData(
    std::shared_ptr<Catalog_Namespace::SessionInfo const> const& session_info)
    : session_info(session_info)
    , db_name(session_info->getCatalog().getCurrentDB().dbName)
    , user_name(session_info->get_currentUser().userName)
    , public_session_id(session_info->get_public_session_id()) {}

std::atomic<query_state::Id> QueryState::s_next_id{0};

QueryState::QueryState(
    std::shared_ptr<Catalog_Namespace::SessionInfo const> const& session_info,
    std::string query_str)
    : id_(s_next_id++)
    , session_data_(session_info ? boost::make_optional<SessionData>(session_info)
                                 : boost::none)
    , query_str_(std::move(query_str))
    , logged_(false) {}

QueryStateProxy QueryState::createQueryStateProxy() {
  return createQueryStateProxy(events_.end());
}

QueryStateProxy QueryState::createQueryStateProxy(Events::iterator parent) {
  return QueryStateProxy(*this, parent);
}

Timer QueryState::createTimer(char const* event_name, Events::iterator parent) {
  std::lock_guard<std::mutex> lock(events_mutex_);
  return Timer(shared_from_this(), events_.emplace(events_.end(), event_name, parent));
}

std::shared_ptr<Catalog_Namespace::SessionInfo const> QueryState::getConstSessionInfo()
    const {
  if (!session_data_) {
    throw std::runtime_error("session_info_ was not set for this QueryState.");
  }
  if (auto retval = session_data_->session_info.lock()) {
    return retval;
  } else {
    // This can happen for a query on a database that is simultaneously dropped.
    throw std::runtime_error("session_info requested but has expired.");
  }
}

// Assumes query_state_ is not null, and events_mutex_ is locked for this.
void QueryState::logCallStack(std::stringstream& ss,
                              unsigned const depth,
                              Events::iterator parent) {
  auto it = parent == events_.end() ? events_.begin() : std::next(parent);
  for (; it != events_.end(); ++it) {
    if (it->parent == parent) {
      auto duration = it->duration();  // std::optional, true if event completed
      ss << '\n'
         << std::setw(depth << 1) << ' ' << it->name << ' ' << it->thread_id
         << " - total time " << (duration ? *duration : -1) << " ms";
      logCallStack(ss, depth + 1, it);
      // If events_ is expendable, then this can be put here to speed-up large lists.
      // it = events_.erase(it); } else { ++it; } (and remove above ++it.)
    }
  }
}

void QueryState::logCallStack(std::stringstream& ss) {
  std::lock_guard<std::mutex> lock(events_mutex_);
  logCallStack(ss, 1, events_.end());
}

Timer QueryStateProxy::createTimer(char const* event_name) {
  return query_state_.createTimer(event_name, parent_);
}

Timer::Timer(std::shared_ptr<QueryState>&& query_state, Events::iterator event)
    : query_state_(std::move(query_state)), event_(event) {}

QueryStateProxy Timer::createQueryStateProxy() {
  return query_state_->createQueryStateProxy(event_);
}

Timer::~Timer() {
  event_->stop();
}

std::atomic<int64_t> StdLogData::s_match{0};

std::shared_ptr<Catalog_Namespace::SessionInfo const> StdLog::getConstSessionInfo()
    const {
  return session_info_;
}

std::shared_ptr<Catalog_Namespace::SessionInfo> StdLog::getSessionInfo() const {
  return session_info_;
}

struct QuoteFormatter {
  std::string const& str;
};

std::ostream& operator<<(std::ostream& os, QuoteFormatter const& quote_formatter) {
  if (quote_formatter.str.find_first_of(" \"") == std::string::npos) {
    return os << quote_formatter.str;
  } else {
    return os << std::quoted(quote_formatter.str, '"', '"');
  }
}

std::ostream& operator<<(std::ostream& os, SessionData const& session_data) {
  return os << QuoteFormatter{session_data.db_name} << ' '
            << QuoteFormatter{session_data.user_name} << ' '
            << session_data.public_session_id;
}

struct SessionInfoFormatter {
  Catalog_Namespace::SessionInfo const& session_info;
};

std::ostream& operator<<(std::ostream& os, SessionInfoFormatter const& formatter) {
  return os << QuoteFormatter{formatter.session_info.getCatalog().getCurrentDB().dbName}
            << ' ' << QuoteFormatter{formatter.session_info.get_currentUser().userName}
            << ' ' << formatter.session_info.get_public_session_id();
}

// Default severity for logging stdlog_begin lines is DEBUG1.
// Some functions, such as sql_execute, log them at INFO level.
logger::Severity StdLog::stdlogBeginSeverity(char const* func) {
  logger::Severity const defaultSeverity = logger::Severity::DEBUG1;
  static std::unordered_map<std::string_view, logger::Severity> const map{
      {"sql_execute", logger::Severity::INFO}};
  auto const itr = map.find(func);
  return itr == map.cend() ? defaultSeverity : itr->second;
}

void StdLog::log(logger::Severity severity, char const* label) {
  if (logger::fast_logging_check(severity)) {
    std::stringstream ss;
    ss << file_ << ':' << line_ << ' ' << label << ' ' << func_ << ' ' << match_ << ' '
       << duration<std::chrono::milliseconds>() << ' ';
    if (session_info_) {
      ss << SessionInfoFormatter{*session_info_} << ' ';
    } else if (query_state_ && query_state_->getSessionData()) {
      ss << *query_state_->getSessionData() << ' ';
    } else {
      ss << "   ";  // 3 spaces for 3 empty strings
    }
    auto const& nv = name_value_pairs_;
    if (nv.empty() && (!query_state_ || query_state_->emptyLog())) {
      ss << ' ';  // 1 space for final empty names/values arrays
    } else {
      // All values are logged after all names, so separate values stream is needed.
      std::stringstream values;
      unsigned nvalues = 0;
      if (query_state_ && !query_state_->getQueryStr().empty()) {
        ss << (nvalues ? ',' : '{') << std::quoted("query_str", '"', '"');
        values << (nvalues++ ? ',' : '{')
               << std::quoted(hide_sensitive_data_from_query(query_state_->getQueryStr()),
                              '"',
                              '"');
      }
      for (auto itr = nv.cbegin(); itr != nv.cend(); ++itr) {
        ss << (nvalues ? ',' : '{') << std::quoted(*itr, '"', '"');
        values << (nvalues++ ? ',' : '{') << std::quoted(*++itr, '"', '"');
      }
      ss << "} " << values.rdbuf() << '}';
    }
    BOOST_LOG_SEV(logger::gSeverityLogger::get(), severity) << ss.rdbuf();
  }
}

void StdLog::logCallStack(logger::Severity severity, char const* label) {
  if (logger::fast_logging_check(severity) && query_state_) {
    std::stringstream ss;
    ss << file_ << ':' << line_ << ' ' << label << ' ' << func_ << ' ' << match_
       << " total time " << duration<std::chrono::milliseconds>() << " ms";
    query_state_->logCallStack(ss);
    BOOST_LOG_SEV(logger::gSeverityLogger::get(), severity) << ss.rdbuf();
  }
}

StdLog::~StdLog() {
  log(logger::Severity::INFO, "stdlog");
  logCallStack(logger::Severity::DEBUG1, "stacked_times");
  if (query_state_) {
    query_state_->setLogged(true);
  }
  if (session_info_) {
    session_info_->update_last_used_time();
  }
}

void StdLog::setQueryState(std::shared_ptr<QueryState> query_state) {
  query_state_ = std::move(query_state);
}

void StdLog::setSessionInfo(
    std::shared_ptr<Catalog_Namespace::SessionInfo> session_info) {
  session_info_ = std::move(session_info);
}

}  // namespace query_state
