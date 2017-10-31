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
 * File:   Role.h
 * Author: norair
 * @brief  Class specification and related data structures for Role class.
 *
 * Users are granted same role and included in the Role class if they have
 * similar access privileges to the given set of tables, and/or other DB objects
 *
 * Created on May 16, 2017, 03:30 PM
 */

#ifndef ROLE_H
#define ROLE_H

#include "DBObject.h"
// #include "Catalog.h"

#include <map>
#include <unordered_set>
#include <boost/algorithm/string.hpp>
#include <string>
#include <glog/logging.h>

/* the mapd default roles */
#define MAPD_DEFAULT_ROOT_USER_ROLE "mapd_default_suser_role"
#define MAPD_DEFAULT_USER_ROLE "mapd_default_user_role"

// Abstract base class, includes access privileges to DB objects
class Role {
  /**
   * @type DBObjectMap
   * @brief Maps DBObject's object keys to pointers to DBObject class objects allocated on the heap
   */
  typedef std::map<DBObjectKey, DBObject*> DBObjectMap;

 public:
  Role(const std::string& name);
  Role(const Role& role);
  virtual ~Role();

  virtual size_t getMembershipSize() const = 0;
  virtual bool checkPrivileges(const DBObject& objectRequested) const = 0;
  virtual void copyRoles(const std::unordered_set<Role*>& roles) = 0;
  virtual void addRole(Role* role) = 0;
  virtual void removeRole(Role* role) = 0;

  virtual void grantPrivileges(const DBObject& object) = 0;
  virtual void revokePrivileges(const DBObject& object) = 0;
  virtual void getPrivileges(DBObject& object) = 0;
  virtual void grantRole(Role* role) = 0;
  virtual void revokeRole(Role* role) = 0;
  virtual bool hasRole(Role* role) = 0;
  virtual void updatePrivileges(Role* role) = 0;
  virtual void updatePrivileges() = 0;
  virtual std::string roleName(bool userName = false) const = 0;
  virtual bool isUserPrivateRole() const = 0;
  virtual std::vector<std::string> getRoles() const = 0;
  const DBObjectMap* getDbObject() const;
  DBObject* findDbObject(const DBObjectKey objectKey) const;

 protected:
  void copyDbObjects(const Role& role);

  std::string roleName_;
  DBObjectMap dbObjectMap_;
};

// Access privileges to DB objects for a specific user
class UserRole : public Role {
 public:
  UserRole(Role* role, const int32_t userId, const std::string& userName);
  UserRole(const UserRole& role);
  virtual ~UserRole();

  virtual size_t getMembershipSize() const;
  virtual bool checkPrivileges(const DBObject& objectRequested) const;

  virtual void copyRoles(const std::unordered_set<Role*>& roles);
  virtual void addRole(Role* role);
  virtual void removeRole(Role* role);

  virtual void grantPrivileges(const DBObject& object);
  virtual void revokePrivileges(const DBObject& object);
  virtual void getPrivileges(DBObject& object);
  virtual void grantRole(Role* role);
  virtual void revokeRole(Role* role);
  virtual bool hasRole(Role* role);
  virtual void updatePrivileges(Role* role);
  virtual void updatePrivileges();
  virtual std::string roleName(bool userName = false) const;
  virtual bool isUserPrivateRole() const;
  virtual std::vector<std::string> getRoles() const;

 private:
  int32_t userId_;
  std::string userName_;
  std::unordered_set<Role*> groupRole_;
};

// DB user roles and privileges, including DB users and their access privileges to DB objects
class GroupRole : public Role {
 public:
  GroupRole(const std::string& name, const bool& userPrivateRole = false);
  GroupRole(const GroupRole& role);
  virtual ~GroupRole();

  virtual size_t getMembershipSize() const;
  virtual bool checkPrivileges(const DBObject& objectRequested) const;
  virtual void copyRoles(const std::unordered_set<Role*>& roles);
  virtual void addRole(Role* role);
  virtual void removeRole(Role* role);

  virtual void grantPrivileges(const DBObject& object);
  virtual void revokePrivileges(const DBObject& object);
  virtual void getPrivileges(DBObject& object);
  virtual void grantRole(Role* role);
  virtual void revokeRole(Role* role);
  virtual bool hasRole(Role* role);
  virtual void updatePrivileges(Role* role);
  virtual void updatePrivileges();
  virtual std::string roleName(bool userName = false) const;
  virtual bool isUserPrivateRole() const;
  virtual std::vector<std::string> getRoles() const;

 private:
  bool userPrivateRole_;
  std::unordered_set<Role*> userRole_;
};

#endif /* ROLE_H */
