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
#include <iomanip>
#include <sstream>
#include "Catalog.h"

namespace Catalog_Namespace {

bool SessionInfo::checkDBAccessPrivileges(const DBObjectType& permissionType,
                                          const AccessPrivileges& privs,
                                          const std::string& objectName) const {
  auto& cat = getCatalog();
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

// start_time(3)-session_id(4) Example: 819-4RDo
// This shows 4 chars of the secret session key,
// leaving (32-4)*log2(62) > 166 bits secret.
std::string SessionInfo::public_session_id() const {
  const time_t start_time = get_start_time();
  struct tm st;
  localtime_r(&start_time, &st);
  std::ostringstream ss;
  ss << (st.tm_min % 10) << std::setfill('0') << std::setw(2) << st.tm_sec << '-'
     << session_id.substr(0, 4);
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const SessionInfo& session_info) {
  os << session_info.get_public_session_id();
  return os;
}

}  // namespace Catalog_Namespace
