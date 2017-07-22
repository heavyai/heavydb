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

#include "../Shared/types.h"

#include <unordered_set>
#include <string>
#include <glog/logging.h>

// DB objects for which privileges are currently supported
enum DBObjectType {
  AbstractDBObjectType = 0,
  DatabaseDBObjectType,
  TableDBObjectType,
  ColumnDBObjectType,
  DashboardDBObjectType
};

// Access privileges currently supported
class AccessPrivileges {
 public:
  bool select_;
  bool insert_;
  bool create_;
  /* following privileges may be added on as needed basis
  bool update_;
  bool delete_;
  bool trancate_;
  bool references_;
  bool trigger_;
  bool connect_;
  bool temporary_;
  bool execute_;
  bool usage_;
  */
};

class DBObject {
 public:
  DBObject(const std::string& name, const DBObjectType& type);
  DBObject(const DBObject& object);
  ~DBObject();

  std::string getName() const;
  DBObjectType getType() const;
  DBObjectKey getObjectKey() const;
  std::vector<bool> getPrivileges() const;
  void setObjectKey(const DBObjectKey& objectKey);
  void setPrivileges(const std::vector<bool> priv);
  void copyPrivileges(const DBObject& object);
  void updatePrivileges(const DBObject& object);
  void grantPrivileges(const DBObject& object);
  void revokePrivileges(const DBObject& object);
  bool isUserPrivateObject() const;
  void setUserPrivateObject();
  int32_t getOwningUserId() const;
  void setOwningUserId(int32_t userId);

 private:
  std::string objectName_;
  DBObjectType objectType_;
  DBObjectKey objectKey_;
  AccessPrivileges objectPrivs_;
  bool privsValid_;
  bool userPrivateObject_;  // false if not use private
  int32_t owningUserId_;    // 0 - if not owned by user

  friend class UserRole;
};

#endif /* DBOBJECT_H */
