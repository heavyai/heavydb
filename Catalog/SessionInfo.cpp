/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#ifdef __linux__
  localtime_r(&start_time, &st);
#else
  localtime_s(&st, &start_time);
#endif
  std::ostringstream ss;
  ss << (st.tm_min % 10) << std::setfill('0') << std::setw(2) << st.tm_sec << '-'
     << session_id_.substr(0, 4);
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const SessionInfo& session_info) {
  os << session_info.get_public_session_id();
  return os;
}

}  // namespace Catalog_Namespace
