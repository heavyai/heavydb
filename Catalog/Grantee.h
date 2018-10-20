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

#ifndef GRANTEE_H
#define GRANTEE_H

#include "DBObject.h"

#include <glog/logging.h>
#include <boost/algorithm/string.hpp>
#include <boost/make_unique.hpp>
#include <map>
#include <string>
#include <unordered_set>

class User;
class Role;

class Grantee {
  typedef std::map<DBObjectKey, std::unique_ptr<DBObject>> DBObjectMap;

 public:
  Grantee(const std::string& name);
  virtual ~Grantee();
  virtual bool isUser() const = 0;
  virtual void grantPrivileges(const DBObject& object);
  virtual DBObject* revokePrivileges(const DBObject& object);
  virtual void grantRole(Role* role);
  virtual void revokeRole(Role* role);
  virtual bool hasAnyPrivileges(const DBObject& objectRequested, bool only_direct) const;
  virtual bool checkPrivileges(const DBObject& objectRequested) const;
  virtual void updatePrivileges();
  virtual void updatePrivileges(Role* role);
  virtual void revokeAllOnDatabase(int32_t dbId);
  virtual void renameDbObject(const DBObject& object);
  void getPrivileges(DBObject& object, bool only_direct);
  DBObject* findDbObject(const DBObjectKey& objectKey, bool only_direct) const;
  bool hasAnyPrivilegesOnDb(int32_t dbId, bool only_direct) const;
  const std::string& getName() const { return name_; }
  std::vector<std::string> getRoles() const;
  bool hasRole(Role* role, bool only_direct) const;
  const DBObjectMap* getDbObjects(bool only_direct) const {
    return only_direct ? &directPrivileges_ : &effectivePrivileges_;
  }
  void checkCycles(Role* newRole);

 protected:
  std::string name_;
  std::unordered_set<Role*> roles_;
  // tracks all privileges, including privileges from granted roles recursively
  DBObjectMap effectivePrivileges_;
  // tracks only privileges granted directly to this grantee
  DBObjectMap directPrivileges_;
};

class User : public Grantee {
 public:
  User(const std::string& name) : Grantee(name) {}
  virtual bool isUser() const { return true; }
};

class Role : public Grantee {
 public:
  Role(const std::string& name) : Grantee(name) {}
  virtual ~Role();

  virtual bool isUser() const { return false; }
  virtual void updatePrivileges();
  virtual void renameDbObject(const DBObject& object);

  // NOTE(max): To be used only from Grantee
  virtual void addGrantee(Grantee* grantee);
  virtual void removeGrantee(Grantee* grantee);

  virtual void revokeAllOnDatabase(int32_t dbId);
  std::vector<Grantee*> getGrantees() const;

 private:
  std::unordered_set<Grantee*> grantees_;
};

#endif /* GRANTEE_H */
