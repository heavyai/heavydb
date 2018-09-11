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
  Grantee(const Grantee& grantee);
  virtual ~Grantee();
  virtual bool isUser() const = 0;
  virtual void grantPrivileges(const DBObject& object);
  virtual DBObject* revokePrivileges(const DBObject& object);
  virtual void grantRole(Role* role);
  virtual void revokeRole(Role* role);
  virtual bool hasAnyPrivileges(const DBObject& objectRequested, bool recursive) const;
  virtual bool checkPrivileges(const DBObject& objectRequested) const;
  virtual void updatePrivileges();
  virtual void updatePrivileges(Role* role);
  virtual void revokeAllOnDatabase(int32_t dbId);
  void getPrivileges(DBObject& object, bool recursive);
  DBObject* findDbObject(const DBObjectKey& objectKey, bool recursive) const;
  const std::string& getName() const { return name_; }
  std::vector<std::string> getRoles() const;
  bool hasRole(Role* role, bool recursive) const;
  const DBObjectMap* getDbObjects(bool recursive) const {
    return recursive ? &cachedPrivileges_ : &privileges_;
  }
  void checkCycles(Role* newRole);

 protected:
  std::string name_;
  std::unordered_set<Role*> roles_;
  // tracks all privileges, including privileges from granted roles recursively
  DBObjectMap cachedPrivileges_;
  // tracks only privileges granted directly to this grantee
  DBObjectMap privileges_;
};

class User : public Grantee {
 public:
  User(const std::string& name) : Grantee(name) {}
  User(const User& user) : Grantee(user) {}
  virtual bool isUser() const { return true; }
};

class Role : public Grantee {
 public:
  Role(const std::string& name) : Grantee(name) {}
  Role(const Role& role) : Grantee(role) {}
  virtual ~Role();

  virtual bool isUser() const { return false; }
  virtual void updatePrivileges();

  // NOTE(max): To be used only from Grantee
  virtual void addGrantee(Grantee* grantee);
  virtual void removeGrantee(Grantee* grantee);

  virtual void revokeAllOnDatabase(int32_t dbId);
  std::vector<Grantee*> getGrantees() const;

 private:
  std::unordered_set<Grantee*> grantees_;
};

#endif /* GRANTEE_H */
