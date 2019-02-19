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

#include "SessionInfo.h"
#include "Catalog.h"

namespace Catalog_Namespace {

bool SessionInfo::checkDBAccessPrivileges(const DBObjectType& permissionType,
                                          const AccessPrivileges& privs,
                                          const std::string& objectName) const {
  auto& cat = getCatalog();
  if (!SysCatalog::instance().arePrivilegesOn()) {
    // run flow without DB object level access permission checks
    Privileges wants_privs;
    if (get_currentUser().isSuper) {
      wants_privs.super_ = true;
    } else {
      wants_privs.super_ = false;
    }
    wants_privs.select_ = false;
    wants_privs.insert_ = true;
    auto currentDB = static_cast<Catalog_Namespace::DBMetadata>(cat.getCurrentDB());
    auto currentUser = static_cast<Catalog_Namespace::UserMetadata>(get_currentUser());
    return SysCatalog::instance().checkPrivileges(currentUser, currentDB, wants_privs);
  } else {
    // run flow with DB object level access permission checks
    DBObject object(objectName, permissionType);
    if (permissionType == DBObjectType::DatabaseDBObjectType) {
      object.setName(cat.getCurrentDB().dbName);
    }

    object.loadKey(cat);
    object.setPrivileges(privs);
    std::vector<DBObject> privObjects;
    privObjects.push_back(object);
    return SysCatalog::instance().checkPrivileges(get_currentUser(), privObjects);
  }
}

SessionInfo::operator std::string() const {
  return get_currentUser().userName + "_" + session_id.substr(0, 3);
}

}  // namespace Catalog_Namespace

std::ostream& operator<<(std::ostream& os,
                         const Catalog_Namespace::SessionInfo& session_info) {
  os << std::string(session_info);
  return os;
}
