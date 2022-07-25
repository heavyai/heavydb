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

#pragma once

#include "Catalog/SessionInfo.h"

#include <functional>
#include <memory>
#include <thread>

namespace Catalog_Namespace {

using SessionInfoPtr = std::shared_ptr<SessionInfo>;

using DisconnectCallback = std::function<void(SessionInfoPtr& session)>;

class SessionsStore {
 public:
  virtual SessionInfoPtr add(const Catalog_Namespace::UserMetadata& user_meta,
                             std::shared_ptr<Catalog> cat,
                             ExecutorDeviceType device) = 0;
  virtual SessionInfoPtr get(const std::string& session_id) = 0;
  void erase(const std::string& session_id);
  void eraseByUser(const std::string& user_name);
  void eraseByDB(const std::string& db_name);

  std::vector<SessionInfoPtr> getAllSessions();
  std::vector<SessionInfoPtr> getUserSessions(const std::string& user_name);
  SessionInfoPtr getByPublicID(const std::string& public_id);

  virtual ~SessionsStore() = default;

  SessionInfo getSessionCopy(const std::string& session_id);
  void disconnect(const std::string session_id);

  static std::unique_ptr<SessionsStore> create(const std::string& base_path,
                                               size_t n_workers,
                                               int idle_session_duration,
                                               int max_session_duration,
                                               int capacity,
                                               DisconnectCallback disconnect_callback);

 protected:
  bool isSessionExpired(const SessionInfoPtr& session_ptr,
                        int idle_session_duration,
                        int max_session_duration);
  virtual bool isSessionInUse(const SessionInfoPtr& session_ptr) = 0;
  virtual SessionInfoPtr getUnlocked(const std::string& session_id) = 0;
  virtual void eraseUnlocked(const std::string& session_id) = 0;
  virtual DisconnectCallback getDisconnectCallback() = 0;
  virtual std::vector<SessionInfoPtr> getIf(
      std::function<bool(const SessionInfoPtr&)> predicate) = 0;
  virtual void eraseIf(std::function<bool(const SessionInfoPtr&)> predicate) = 0;
  virtual heavyai::shared_mutex& getLock() = 0;
};

}  // namespace Catalog_Namespace
