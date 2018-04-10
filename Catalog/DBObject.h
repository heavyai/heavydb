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

#include <unordered_set>
#include <string>
#include <glog/logging.h>

namespace Catalog_Namespace {
class Catalog;
}

// DB objects for which privileges are currently supported
enum DBObjectType {
  AbstractDBObjectType = 0,
  DatabaseDBObjectType,
  TableDBObjectType,
  ColumnDBObjectType,
  DashboardDBObjectType
};

std::string DBObjectTypeToString(DBObjectType type);
DBObjectType DBObjectTypeFromString(const std::string& type);

struct DBObjectKey {
  int32_t dbObjectType = -1;
  int32_t dbId = -1;
  int32_t tableId = -1;
  int32_t columnId = -1;

  static const size_t N_COLUMNS = 4;

  bool operator<(const DBObjectKey& key) const {
    int32_t ids_a[N_COLUMNS] = {dbObjectType, dbId, tableId, columnId};
    int32_t ids_b[N_COLUMNS] = {key.dbObjectType, key.dbId, key.tableId, key.columnId};
    return memcmp(ids_a, ids_b, N_COLUMNS * sizeof(int32_t)) < 0;
  }

  static DBObjectKey fromString(const std::vector<std::string>& key, const DBObjectType& type);
};

// Access privileges currently supported
struct AccessPrivileges {
  bool select = false;
  bool insert = false;
  bool create = false;
  bool truncate = false;
  bool create_dashboard = false;

  AccessPrivileges() {}

  AccessPrivileges(bool select, bool insert, bool create, bool truncate, bool create_dashboard)
      : select(select), insert(insert), create(create), truncate(truncate), create_dashboard(create_dashboard) {}

  void reset() { select = insert = create = truncate = create_dashboard = false; }
  bool hasAny() const { return select || insert || create || truncate || create_dashboard; }

  static const AccessPrivileges ALL;
  static const AccessPrivileges ALL_TABLE;
  static const AccessPrivileges ALL_DASHBOARD;
  static const AccessPrivileges SELECT;
  static const AccessPrivileges INSERT;
  static const AccessPrivileges CREATE;
  static const AccessPrivileges TRUNCATE;
  static const AccessPrivileges CREATE_DASHBOARD;
};

class DBObject {
 public:
  DBObject(const std::string& name, const DBObjectType& type);
  DBObject(const int32_t id, const DBObjectType& type);
  DBObject(const DBObject& object);
  ~DBObject() {}

  std::string getName() const { return objectName_; }
  int32_t getId() const { return objectId_; }
  void setId(int32_t id) { objectId_ = id; }
  DBObjectType getType() const { return objectType_; }
  DBObjectKey getObjectKey() const { return objectKey_; }
  void setObjectKey(const DBObjectKey& objectKey) { objectKey_ = objectKey; }
  const AccessPrivileges& getPrivileges() const { return objectPrivs_; }
  void setPrivileges(const AccessPrivileges& privs) { objectPrivs_ = privs; }
  void resetPrivileges() { objectPrivs_.reset(); }
  void copyPrivileges(const DBObject& object);
  void updatePrivileges(const DBObject& object);
  void grantPrivileges(const DBObject& object) { updatePrivileges(object); }
  void revokePrivileges(const DBObject& object);
  int32_t getOwner() const { return ownerId_; }
  void setOwner(int32_t userId) { ownerId_ = userId; }
  bool privsValid() const { return privsValid_; }
  void unvalidate() { privsValid_ = false; }
  std::vector<std::string> toString() const;
  void loadKey(const Catalog_Namespace::Catalog& catalog);

 private:
  std::string objectName_;
  int32_t objectId_;
  DBObjectType objectType_;
  DBObjectKey objectKey_;
  AccessPrivileges objectPrivs_;
  bool privsValid_;
  int32_t ownerId_;  // 0 - if not owned by user
};

#endif /* DBOBJECT_H */
