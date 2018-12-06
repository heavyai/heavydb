/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * File:   DBObject.h
 * Author: norair
 * @brief  Class specification and related data structures for DBObject class.
 *
 * To support access privileges of DB users to DB entities (tables, columns, views, etc),
 * the users are granted roles and included in the corresponding object of the Role class,
 * and DB entities are being described as objects of DBObjects class
 *
 * Created on May 16, 2017, 03:30 PM
 */

#ifndef DBOBJECT_H
#define DBOBJECT_H

#include <glog/logging.h>
#include <string>
#include <unordered_set>

namespace Catalog_Namespace {
class Catalog;
}

// DB objects for which privileges are currently supported, only ever add enums, never
// remove as the nums are persisted in the catalog DB
enum DBObjectType {
  AbstractDBObjectType = 0,
  DatabaseDBObjectType,
  TableDBObjectType,
  DashboardDBObjectType,
  ViewDBObjectType
};

std::string DBObjectTypeToString(DBObjectType type);
DBObjectType DBObjectTypeFromString(const std::string& type);

struct DBObjectKey {
  int32_t permissionType = -1;
  int32_t dbId = -1;
  int32_t objectId = -1;

  static const size_t N_COLUMNS = 3;

  bool operator<(const DBObjectKey& key) const {
    int32_t ids_a[N_COLUMNS] = {permissionType, dbId, objectId};
    int32_t ids_b[N_COLUMNS] = {key.permissionType, key.dbId, key.objectId};
    return memcmp(ids_a, ids_b, N_COLUMNS * sizeof(int32_t)) < 0;
  }

  bool operator==(const DBObjectKey& key) const {
    return permissionType == key.permissionType && dbId == key.dbId &&
           objectId == key.objectId;
  }

  static DBObjectKey fromString(const std::vector<std::string>& key,
                                const DBObjectType& type);
};

// Access privileges currently supported

struct DatabasePrivileges {
  static const int32_t ALL = -1;
  static const int32_t CREATE_DATABASE = 1 << 0;
  static const int32_t DROP_DATABASE = 1 << 1;
  static const int32_t VIEW_SQL_EDITOR = 1 << 2;
  static const int32_t ACCESS = 1 << 3;
};

struct TablePrivileges {
  static const int32_t ALL = -1;
  static const int32_t CREATE_TABLE = 1 << 0;
  static const int32_t DROP_TABLE = 1 << 1;
  static const int32_t SELECT_FROM_TABLE = 1 << 2;
  static const int32_t INSERT_INTO_TABLE = 1 << 3;
  static const int32_t UPDATE_IN_TABLE = 1 << 4;
  static const int32_t DELETE_FROM_TABLE = 1 << 5;
  static const int32_t TRUNCATE_TABLE = 1 << 6;
  static const int32_t ALTER_TABLE = 1 << 7;

  static const int32_t ALL_MIGRATE =
      CREATE_TABLE | DROP_TABLE | SELECT_FROM_TABLE | INSERT_INTO_TABLE;
};

struct DashboardPrivileges {
  static const int32_t ALL = -1;
  static const int32_t CREATE_DASHBOARD = 1 << 0;
  static const int32_t DELETE_DASHBOARD = 1 << 1;
  static const int32_t VIEW_DASHBOARD = 1 << 2;
  static const int32_t EDIT_DASHBOARD = 1 << 3;

  static const int32_t ALL_MIGRATE =
      CREATE_DASHBOARD | DELETE_DASHBOARD | VIEW_DASHBOARD | EDIT_DASHBOARD;
};

struct ViewPrivileges {
  static const int32_t ALL = -1;
  static const int32_t CREATE_VIEW = 1 << 0;
  static const int32_t DROP_VIEW = 1 << 1;
  static const int32_t SELECT_FROM_VIEW = 1 << 2;
  static const int32_t INSERT_INTO_VIEW = 1 << 3;
  static const int32_t UPDATE_IN_VIEW = 1 << 4;
  static const int32_t DELETE_FROM_VIEW = 1 << 5;
  static const int32_t TRUNCATE_VIEW = 1 << 6;

  static const int32_t ALL_MIGRATE =
      CREATE_VIEW | DROP_VIEW | SELECT_FROM_VIEW | INSERT_INTO_VIEW;
};

struct AccessPrivileges {
  int64_t privileges;

  AccessPrivileges() : privileges(0) {}

  AccessPrivileges(int64_t priv) : privileges(priv) {}

  void reset() { privileges = 0L; }
  bool hasAny() const { return 0L != privileges; }
  bool hasPermission(int permission) const {
    return permission == (privileges & permission);
  }

  void add(AccessPrivileges newprivs) { privileges |= newprivs.privileges; }
  void remove(AccessPrivileges newprivs) { privileges &= ~(newprivs.privileges); }

  static const AccessPrivileges NONE;

  // database permissions
  static const AccessPrivileges ALL_DATABASE;
  static const AccessPrivileges VIEW_SQL_EDITOR;
  static const AccessPrivileges ACCESS;

  // table permissions
  static const AccessPrivileges ALL_TABLE_MIGRATE;
  static const AccessPrivileges ALL_TABLE;
  static const AccessPrivileges CREATE_TABLE;
  static const AccessPrivileges DROP_TABLE;
  static const AccessPrivileges SELECT_FROM_TABLE;
  static const AccessPrivileges INSERT_INTO_TABLE;
  static const AccessPrivileges UPDATE_IN_TABLE;
  static const AccessPrivileges DELETE_FROM_TABLE;
  static const AccessPrivileges TRUNCATE_TABLE;
  static const AccessPrivileges ALTER_TABLE;

  // dashboard permissions
  static const AccessPrivileges ALL_DASHBOARD_MIGRATE;
  static const AccessPrivileges ALL_DASHBOARD;
  static const AccessPrivileges CREATE_DASHBOARD;
  static const AccessPrivileges VIEW_DASHBOARD;
  static const AccessPrivileges EDIT_DASHBOARD;
  static const AccessPrivileges DELETE_DASHBOARD;

  // view permissions
  static const AccessPrivileges ALL_VIEW_MIGRATE;
  static const AccessPrivileges ALL_VIEW;
  static const AccessPrivileges CREATE_VIEW;
  static const AccessPrivileges DROP_VIEW;
  static const AccessPrivileges SELECT_FROM_VIEW;
  static const AccessPrivileges INSERT_INTO_VIEW;
  static const AccessPrivileges UPDATE_IN_VIEW;
  static const AccessPrivileges DELETE_FROM_VIEW;
  static const AccessPrivileges TRUNCATE_VIEW;
};

class DBObject {
 public:
  DBObject(const std::string& name, const DBObjectType& objectAndPermissionType);
  DBObject(const int32_t id, const DBObjectType& objectAndPermissionType);
  DBObject(DBObjectKey key, AccessPrivileges privs, int32_t owner)
      : objectName_("")
      , objectType_(AbstractDBObjectType)
      , objectKey_(key)
      , objectPrivs_(privs)
      , ownerId_(owner){};
  DBObject(const DBObject& object);
  ~DBObject() {}

  void setObjectType(const DBObjectType& objectType);
  void setName(std::string name) { objectName_ = name; }
  std::string getName() const { return objectName_; }
  DBObjectKey getObjectKey() const {
    CHECK(-1 != objectKey_.dbId); /** load key not called? */
    return objectKey_;
  }
  void setObjectKey(const DBObjectKey& objectKey) { objectKey_ = objectKey; }
  const AccessPrivileges& getPrivileges() const { return objectPrivs_; }
  void setPrivileges(const AccessPrivileges& privs) { objectPrivs_ = privs; }
  void resetPrivileges() { objectPrivs_.reset(); }
  void copyPrivileges(const DBObject& object);
  void updatePrivileges(const DBObject& object);
  void grantPrivileges(const DBObject& object) { updatePrivileges(object); }
  void revokePrivileges(const DBObject& object);
  void setPermissionType(const DBObjectType& permissionType);
  int32_t getOwner() const { return ownerId_; }
  void setOwner(int32_t userId) { ownerId_ = userId; }
  std::vector<std::string> toString() const;
  void loadKey(const Catalog_Namespace::Catalog& catalog);

 private:
  std::string objectName_;
  DBObjectType objectType_;
  DBObjectKey objectKey_;
  AccessPrivileges objectPrivs_;
  int32_t ownerId_;  // 0 - if not owned by user
};

#endif /* DBOBJECT_H */
