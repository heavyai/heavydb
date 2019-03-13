/*
 * Copyright 2019 MapD Technologies, Inc.
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

#include "../QueryEngine/CompilationOptions.h"
#include "../SqliteConnector/SqliteConnector.h"
#include "LeafHostInfo.h"
#include "SysCatalog.h"

namespace Importer_NS {
class Loader;
class TypedImportBuffer;
}  // namespace Importer_NS

namespace Catalog_Namespace {

class Catalog;

// this class is defined to accommodate both Thrift and non-Thrift builds.
class MapDHandler {
 public:
  virtual void prepare_columnar_loader(
      const std::string& session,
      const std::string& table_name,
      size_t num_cols,
      std::unique_ptr<Importer_NS::Loader>* loader,
      std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>* import_buffers);
  virtual ~MapDHandler() {}
};

/*
 * @type SessionInfo
 * @brief a user session
 */
class SessionInfo {
 public:
  SessionInfo(std::shared_ptr<MapDHandler> mapdHandler,
              std::shared_ptr<Catalog> cat,
              const UserMetadata& user,
              const ExecutorDeviceType t,
              const std::string& sid)
      : mapdHandler_(mapdHandler)
      , catalog_(cat)
      , currentUser_(user)
      , executor_device_type_(t)
      , session_id(sid)
      , last_used_time(time(0))
      , start_time(time(0)) {}
  SessionInfo(std::shared_ptr<Catalog> cat,
              const UserMetadata& user,
              const ExecutorDeviceType t,
              const std::string& sid)
      : SessionInfo(std::make_shared<MapDHandler>(), cat, user, t, sid) {}
  SessionInfo(const SessionInfo& s)
      : mapdHandler_(s.mapdHandler_)
      , catalog_(s.catalog_)
      , currentUser_(s.currentUser_)
      , executor_device_type_(static_cast<ExecutorDeviceType>(s.executor_device_type_))
      , session_id(s.session_id) {}
  MapDHandler* get_mapdHandler() const { return mapdHandler_.get(); }
  Catalog& getCatalog() const { return *catalog_; }
  std::shared_ptr<Catalog> get_catalog_ptr() const { return catalog_; }
  void set_catalog_ptr(std::shared_ptr<Catalog> c) { catalog_ = c; }
  const UserMetadata& get_currentUser() const { return currentUser_; }
  const ExecutorDeviceType get_executor_device_type() const {
    return executor_device_type_;
  }
  void set_executor_device_type(ExecutorDeviceType t) { executor_device_type_ = t; }
  std::string get_session_id() const { return session_id; }
  time_t get_last_used_time() const { return last_used_time; }
  void update_last_used_time() { last_used_time = time(0); }
  void reset_superuser() { currentUser_.isSuper = currentUser_.isReallySuper; }
  void make_superuser() { currentUser_.isSuper = true; }
  bool checkDBAccessPrivileges(const DBObjectType& permissionType,
                               const AccessPrivileges& privs,
                               const std::string& objectName = "") const;
  time_t get_start_time() const { return start_time; }

  operator std::string() const;

 private:
  std::shared_ptr<MapDHandler> mapdHandler_;
  std::shared_ptr<Catalog> catalog_;
  UserMetadata currentUser_;
  std::atomic<ExecutorDeviceType> executor_device_type_;
  const std::string session_id;
  std::atomic<time_t> last_used_time;  // for cleaning up SessionInfo after client dies
  std::atomic<time_t> start_time;      // for invalidating session after tolerance period
};

}  // namespace Catalog_Namespace

std::ostream& operator<<(std::ostream& os,
                         const Catalog_Namespace::SessionInfo& session_info);

#endif /* SESSION_INFO_H */
