/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  heavyai::shared_lock<heavyai::shared_mutex> getSharedLock() const;
  heavyai::lock_guard<heavyai::shared_mutex> getLockGuard() const;
  virtual bool isSessionInUse(const SessionInfoPtr& session_ptr) = 0;
  virtual SessionInfoPtr getUnlocked(const std::string& session_id) = 0;
  virtual void eraseUnlocked(const std::string& session_id) = 0;
  virtual DisconnectCallback getDisconnectCallback() = 0;
  virtual std::vector<SessionInfoPtr> getIf(
      std::function<bool(const SessionInfoPtr&)> predicate) = 0;
  virtual void eraseIf(std::function<bool(const SessionInfoPtr&)> predicate) = 0;
  virtual heavyai::shared_mutex& getMutex() const = 0;
};

}  // namespace Catalog_Namespace
