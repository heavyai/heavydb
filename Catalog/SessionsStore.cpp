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

#include "SessionsStore.h"
#include "Catalog.h"
#include "Shared/StringTransform.h"

#include <boost/algorithm/string.hpp>
#include <memory>
#include <thread>
#include <unordered_map>

using namespace Catalog_Namespace;

SessionInfo SessionsStore::getSessionCopy(const std::string& session_id) {
  auto origin = get(session_id);
  if (origin) {
    heavyai::shared_lock<heavyai::shared_mutex> lock(origin->getLock());
    return *origin;
  }
  throw std::runtime_error("No session with id " + session_id);
}

void SessionsStore::erase(const std::string& session_id) {
  heavyai::lock_guard<heavyai::shared_mutex> lock(getLock());
  eraseUnlocked(session_id);
}

void SessionsStore::eraseByUser(const std::string& user_name) {
  eraseIf([&user_name](const SessionInfoPtr& session_ptr) {
    return boost::iequals(user_name, session_ptr->get_currentUser().userName);
  });
}

void SessionsStore::eraseByDB(const std::string& db_name) {
  eraseIf([&db_name](const SessionInfoPtr& session_ptr) {
    return boost::iequals(db_name, session_ptr->getCatalog().getCurrentDB().dbName);
  });
}

void SessionsStore::disconnect(const std::string session_id) {
  heavyai::lock_guard<heavyai::shared_mutex> lock(getLock());
  auto session_ptr = getUnlocked(session_id);
  if (session_ptr) {
    const auto dbname = session_ptr->getCatalog().getCurrentDB().dbName;
    LOG(INFO) << "User " << session_ptr->get_currentUser().userLoggable()
              << " disconnected from database " << dbname
              << " with public_session_id: " << session_ptr->get_public_session_id();
    getDisconnectCallback()(session_ptr);
    eraseUnlocked(session_ptr->get_session_id());
  }
}

bool SessionsStore::isSessionExpired(const SessionInfoPtr& session_ptr,
                                     int idle_session_duration,
                                     int max_session_duration) {
  if (isSessionInUse(session_ptr)) {
    return false;
  }
  time_t last_used_time = session_ptr->get_last_used_time();
  time_t start_time = session_ptr->get_start_time();
  const auto current_session_duration = time(0) - last_used_time;
  if (current_session_duration > idle_session_duration) {
    LOG(INFO) << "Session " << session_ptr->get_public_session_id() << " idle duration "
              << current_session_duration << " seconds exceeds maximum idle duration "
              << idle_session_duration << " seconds. Invalidating session.";
    return true;
  }
  const auto total_session_duration = time(0) - start_time;
  if (total_session_duration > max_session_duration) {
    LOG(INFO) << "Session " << session_ptr->get_public_session_id() << " total duration "
              << total_session_duration
              << " seconds exceeds maximum total session duration "
              << max_session_duration << " seconds. Invalidating session.";
    return true;
  }
  return false;
}

std::vector<SessionInfoPtr> SessionsStore::getAllSessions() {
  return getIf([](const SessionInfoPtr&) { return true; });
}

std::vector<SessionInfoPtr> SessionsStore::getUserSessions(const std::string& user_name) {
  return getIf([&user_name](const SessionInfoPtr& session_ptr) {
    return session_ptr->get_currentUser().userName == user_name;
  });
}

SessionInfoPtr SessionsStore::getByPublicID(const std::string& public_id) {
  auto sessions = getIf([&public_id](const SessionInfoPtr& session_ptr) {
    return session_ptr->get_public_session_id() == public_id;
  });
  if (sessions.empty()) {
    return nullptr;
  }
  CHECK_EQ(sessions.size(), 1ul);
  return sessions[0];
}

class CachedSessionStore : public SessionsStore {
 public:
  CachedSessionStore(int idle_session_duration,
                     int max_session_duration,
                     int capacity,
                     DisconnectCallback disconnect_callback)
      : idle_session_duration_(idle_session_duration)
      , max_session_duration_(max_session_duration)
      , capacity_(capacity > 0 ? capacity : INT_MAX)
      , disconnect_callback_(disconnect_callback) {}

  SessionInfoPtr add(const Catalog_Namespace::UserMetadata& user_meta,
                     std::shared_ptr<Catalog> cat,
                     ExecutorDeviceType device) override {
    heavyai::lock_guard<heavyai::shared_mutex> lock(mtx_);
    if (int(sessions_.size()) >= capacity_) {
      std::vector<SessionInfoPtr> expired_sessions;
      for (auto it = sessions_.begin(); it != sessions_.end(); it++) {
        if (isSessionExpired(it->second, idle_session_duration_, max_session_duration_)) {
          expired_sessions.push_back(it->second);
        }
      }
      for (auto& session_ptr : expired_sessions) {
        try {
          disconnect_callback_(session_ptr);
          eraseUnlocked(session_ptr->get_session_id());
        } catch (const std::exception& e) {
          eraseUnlocked(session_ptr->get_session_id());
          throw e;
        }
      }
    }
    if (int(sessions_.size()) < capacity_) {
      do {
        auto session_id = generate_random_string(Catalog_Namespace::SESSION_ID_LENGTH);
        if (sessions_.count(session_id) != 0) {
          continue;
        }
        auto session_ptr = std::make_shared<Catalog_Namespace::SessionInfo>(
            cat, user_meta, device, session_id);
        sessions_[session_id] = session_ptr;
        return session_ptr;
      } while (true);
      UNREACHABLE();
    }
    throw std::runtime_error("Too many active sessions");
  }

  SessionInfoPtr get(const std::string& session_id) override {
    heavyai::lock_guard<heavyai::shared_mutex> lock(mtx_);
    auto session_ptr = getUnlocked(session_id);
    if (session_ptr) {
      if (SessionsStore::isSessionExpired(
              session_ptr, idle_session_duration_, max_session_duration_)) {
        try {
          disconnect_callback_(session_ptr);
          eraseUnlocked(session_ptr->get_session_id());
        } catch (const std::exception& e) {
          eraseUnlocked(session_ptr->get_session_id());
          throw e;
        }
        return nullptr;
      }
      session_ptr->update_last_used_time();
      return session_ptr;
    }
    return nullptr;
  }

  heavyai::shared_mutex& getLock() override { return mtx_; }

  void eraseIf(std::function<bool(const SessionInfoPtr&)> predicate) override {
    heavyai::lock_guard<heavyai::shared_mutex> lock(mtx_);
    for (auto it = sessions_.begin(); it != sessions_.end();) {
      if (predicate(it->second)) {
        it = sessions_.erase(it);
      } else {
        it++;
      }
    }
  }

  ~CachedSessionStore() override {
    std::lock_guard lg(mtx_);
    sessions_.clear();
  }

 protected:
  void eraseUnlocked(const std::string& session_id) override {
    sessions_.erase(session_id);
  }

  bool isSessionInUse(const SessionInfoPtr& session_ptr) override {
    return session_ptr.use_count() > 2;
  }

  SessionInfoPtr getUnlocked(const std::string& session_id) override {
    if (auto session_it = sessions_.find(session_id); session_it != sessions_.end()) {
      return session_it->second;
    }
    return nullptr;
  }

  DisconnectCallback getDisconnectCallback() override { return disconnect_callback_; }

  std::vector<SessionInfoPtr> getIf(
      std::function<bool(const SessionInfoPtr&)> predicate) override {
    std::vector<SessionInfoPtr> out;
    heavyai::shared_lock<heavyai::shared_mutex> sessions_lock(getLock());
    for (auto& [_, session] : sessions_) {
      heavyai::shared_lock<heavyai::shared_mutex> session_lock(session->getLock());
      if (predicate(session)) {
        out.push_back(session);
      }
    }
    return out;
  }

 private:
  std::unordered_map<std::string, SessionInfoPtr> sessions_;
  mutable heavyai::shared_mutex mtx_;
  const int idle_session_duration_;
  const int max_session_duration_;
  const int capacity_;
  DisconnectCallback disconnect_callback_;
};

std::unique_ptr<SessionsStore> SessionsStore::create(
    const std::string& base_path,
    size_t n_workers,
    int idle_session_duration,
    int max_session_duration,
    int capacity,
    DisconnectCallback disconnect_callback) {
  return std::make_unique<CachedSessionStore>(
      idle_session_duration, max_session_duration, capacity, disconnect_callback);
}
