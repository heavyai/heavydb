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

#ifndef SESSION_INFO_H
#define SESSION_INFO_H

#include <atomic>
#include <cstdint>
#include <ctime>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Catalog/SysCatalog.h"
#include "QueryEngine/CompilationOptions.h"
#include "Shared/heavyai_shared_mutex.h"
#include "SqliteConnector/SqliteConnector.h"

#include "LeafHostInfo.h"

namespace Catalog_Namespace {

class Catalog;

/*
 * @type SessionInfo
 * @brief a user session
 */
class SessionInfo {
 public:
  SessionInfo(std::shared_ptr<Catalog> cat,
              const UserMetadata& user,
              const ExecutorDeviceType t,
              const std::string& sid)
      : catalog_(cat)
      , currentUser_(user)
      , executor_device_type_(t)
      , session_id_(sid)
      , last_used_time_(time(0))
      , start_time_(time(0))
      , public_session_id_(public_session_id()) {}
  SessionInfo(const SessionInfo& s)
      : catalog_(s.catalog_)
      , currentUser_(s.currentUser_)
      , executor_device_type_(static_cast<ExecutorDeviceType>(s.executor_device_type_))
      , session_id_(s.session_id_)
      , public_session_id_(s.public_session_id_) {}

  // From https://en.cppreference.com/w/cpp/memory/shared_ptr :
  // All member functions (including copy constructor and copy assignment) can be
  // called by multiple threads on different instances of shared_ptr without
  // additional synchronization even if these instances are copies and share ownership
  // of the same object. If multiple threads of execution access the same instance of
  // shared_ptr without synchronization and any of those accesses uses a non-const
  // member function of shared_ptr then a data race will occur; the shared_ptr
  // overloads of atomic functions can be used to prevent the data race.
  Catalog& getCatalog() const {
    heavyai::shared_lock<heavyai::shared_mutex> lock(mtx_);
    CHECK(catalog_);
    return *catalog_;
  }
  std::shared_ptr<Catalog> get_catalog_ptr() const {
    heavyai::shared_lock<heavyai::shared_mutex> lock(mtx_);
    return catalog_;
  }
  void set_catalog_ptr(std::shared_ptr<Catalog> c) {
    heavyai::lock_guard<heavyai::shared_mutex> lock(mtx_);
    catalog_ = c;
  }
  const UserMetadata& get_currentUser() const { return currentUser_; }
  const ExecutorDeviceType get_executor_device_type() const {
    return executor_device_type_;
  }
  void set_executor_device_type(ExecutorDeviceType t) { executor_device_type_ = t; }
  std::string get_session_id() const { return session_id_; }
  time_t get_last_used_time() const { return last_used_time_; }
  void update_last_used_time() { last_used_time_ = time(0); }
  // TODO(max): this method does not belong here
  bool checkDBAccessPrivileges(const DBObjectType& permissionType,
                               const AccessPrivileges& privs,
                               const std::string& objectName = "") const;
  time_t get_start_time() const { return start_time_; }
  std::string const& get_public_session_id() const { return public_session_id_; }
  operator std::string() const { return public_session_id_; }
  std::string const& get_connection_info() const { return connection_info_; }
  void set_connection_info(const std::string& connection) {
    connection_info_ = connection;
  }
  heavyai::shared_mutex& getLock() const { return mtx_; }

 private:
  // TODO(max): C++20 introduces atomic shared_ptr
  std::shared_ptr<Catalog> catalog_;
  UserMetadata currentUser_;
  std::atomic<ExecutorDeviceType> executor_device_type_;
  const std::string session_id_;
  std::atomic<time_t> last_used_time_;  // for tracking active session duration
  std::atomic<time_t> start_time_;      // for invalidating session after tolerance period
  const std::string public_session_id_;
  std::string
      connection_info_;  // String containing connection protocol (tcp/http) and address
  std::string public_session_id() const;
  mutable heavyai::shared_mutex mtx_;
};

std::ostream& operator<<(std::ostream& os, const SessionInfo& session_info);

const size_t CALCITE_SESSION_ID_LENGTH = 64;
const size_t SESSION_ID_LENGTH = 32;

}  // namespace Catalog_Namespace

#endif /* SESSION_INFO_H */
