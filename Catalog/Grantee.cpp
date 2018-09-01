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

#include "Grantee.h"

using std::runtime_error;
using std::string;

Grantee::Grantee(const std::string& name) : name_(name) {}

Grantee::Grantee(const Grantee& grantee) : name_(grantee.name_), roles_(grantee.roles_) {
  for (auto it = grantee.cachedPrivileges_.begin(); it != grantee.cachedPrivileges_.end();
       ++it) {
    cachedPrivileges_[it->first] = boost::make_unique<DBObject>(*it->second.get());
  }
  for (auto it = grantee.privileges_.begin(); it != grantee.privileges_.end(); ++it) {
    privileges_[it->first] = boost::make_unique<DBObject>(*it->second.get());
  }
}

Grantee::~Grantee() {
  for (auto role : roles_) {
    role->removeGrantee(this);
  }
  cachedPrivileges_.clear();
  privileges_.clear();
  roles_.clear();
}

std::vector<std::string> Grantee::getRoles() const {
  std::vector<std::string> roles;
  for (const auto role : roles_) {
    roles.push_back(role->getName());
  }
  return roles;
}

bool Grantee::hasRole(Role* role) const {
  return roles_.find(role) != roles_.end();
}

void Grantee::getPrivileges(DBObject& object, bool recursive) {
  auto dbObject = findDbObject(object.getObjectKey(), recursive);
  if (!dbObject) {  // not found
    throw runtime_error("Can not get privileges because " + getName() +
                        " has no privileges to " + object.getName());
  }
  object.grantPrivileges(*dbObject);
}

DBObject* Grantee::findDbObject(const DBObjectKey& objectKey, bool recursive) const {
  const DBObjectMap& privs = recursive ? cachedPrivileges_ : privileges_;
  DBObject* dbObject = nullptr;
  auto dbObjectIt = privs.find(objectKey);
  if (dbObjectIt != privs.end()) {
    dbObject = dbObjectIt->second.get();
  }
  return dbObject;
}

void Grantee::grantPrivileges(const DBObject& object) {
  auto* dbObject = findDbObject(object.getObjectKey(), true);
  if (!dbObject) {  // not found
    cachedPrivileges_[object.getObjectKey()] = boost::make_unique<DBObject>(object);
  } else {  // found
    dbObject->grantPrivileges(object);
  }
  dbObject = findDbObject(object.getObjectKey(), false);
  if (!dbObject) {  // not found
    privileges_[object.getObjectKey()] = boost::make_unique<DBObject>(object);
  } else {  // found
    dbObject->grantPrivileges(object);
  }
  updatePrivileges();
}

// I think the sematics here are to send in a object to revoke
// if the revoke completely removed all permissions from the object get rid of it
// but then there is nothing to send back to catalog to have rest of delete for
// DB done
DBObject* Grantee::revokePrivileges(const DBObject& object) {
  auto dbObject = findDbObject(object.getObjectKey(), false);
  if (!dbObject ||
      !dbObject->getPrivileges().hasAny()) {  // not found or has none of privileges set
    throw runtime_error("Can not revoke privileges because " + getName() +
                        " has no privileges to " + object.getName());
  }
  bool object_removed = false;
  dbObject->revokePrivileges(object);
  if (!dbObject->getPrivileges().hasAny()) {
    privileges_.erase(object.getObjectKey());
    object_removed = true;
  }

  auto* cachedDbObject = findDbObject(object.getObjectKey(), true);
  if (cachedDbObject && cachedDbObject->getPrivileges().hasAny()) {
    cachedDbObject->revokePrivileges(object);
    if (!cachedDbObject->getPrivileges().hasAny()) {
      cachedPrivileges_.erase(object.getObjectKey());
    }
  }

  updatePrivileges();

  return object_removed ? nullptr : dbObject;
}

void Grantee::grantRole(Role* role) {
  bool found = false;
  for (const auto* granted_role : roles_) {
    if (role == granted_role) {
      found = true;
      break;
    }
  }
  if (found) {
    throw runtime_error("Role " + role->getName() + " have been granted to " + name_ +
                        " already.");
  }
  roles_.insert(role);
  role->addGrantee(this);
  updatePrivileges();
}

void Grantee::revokeRole(Role* role) {
  roles_.erase(role);
  role->removeGrantee(this);
  updatePrivileges();
}

static bool hasEnoughPrivs(const DBObject* real, const DBObject* requested) {
  if (real) {
    auto req = requested->getPrivileges().privileges;
    auto base = real->getPrivileges().privileges;

    // ensures that all requested privileges are present
    return req == (base & req);
  } else {
    return false;
  }
}

static bool hasAnyPrivs(const DBObject* real, const DBObject* /* requested*/) {
  if (real) {
    return real->getPrivileges().hasAny();
  } else {
    return false;
  }
}

bool Grantee::hasAnyPrivileges(const DBObject& objectRequested, bool recursive) const {
  DBObjectKey objectKey = objectRequested.getObjectKey();
  if (hasAnyPrivs(findDbObject(objectKey, recursive), &objectRequested)) {
    return true;
  }

  // if we have an object associated -> ignore it
  if (objectKey.objectId != -1) {
    objectKey.objectId = -1;
    if (hasAnyPrivs(findDbObject(objectKey, recursive), &objectRequested)) {
      return true;
    }
  }

  // if we have an
  if (objectKey.dbId != -1) {
    objectKey.dbId = -1;
    if (hasAnyPrivs(findDbObject(objectKey, recursive), &objectRequested)) {
      return true;
    }
  }
  return false;
}

bool Grantee::checkPrivileges(const DBObject& objectRequested) const {
  DBObjectKey objectKey = objectRequested.getObjectKey();
  if (hasEnoughPrivs(findDbObject(objectKey, true), &objectRequested)) {
    return true;
  }

  // if we have an object associated -> ignore it
  if (objectKey.objectId != -1) {
    objectKey.objectId = -1;
    if (hasEnoughPrivs(findDbObject(objectKey, true), &objectRequested)) {
      return true;
    }
  }

  // if we have an
  if (objectKey.dbId != -1) {
    objectKey.dbId = -1;
    if (hasEnoughPrivs(findDbObject(objectKey, true), &objectRequested)) {
      return true;
    }
  }
  return false;
}

void Grantee::updatePrivileges(Role* role) {
  for (auto& roleDbObject : *role->getDbObjects(false)) {
    auto dbObject = findDbObject(roleDbObject.first, true);
    if (dbObject) {  // found
      dbObject->updatePrivileges(*roleDbObject.second);
    } else {  // not found
      cachedPrivileges_[roleDbObject.first] =
          boost::make_unique<DBObject>(*roleDbObject.second.get());
    }
  }
}

void Grantee::updatePrivileges() {
  for (auto& dbObject : cachedPrivileges_) {
    dbObject.second->resetPrivileges();
  }
  for (auto it = privileges_.begin(); it != privileges_.end(); ++it) {
    if (cachedPrivileges_.find(it->first) != cachedPrivileges_.end()) {
      cachedPrivileges_[it->first]->updatePrivileges(*it->second);
    }
  }
  for (auto role : roles_) {
    if (role->getDbObjects(false)->size() > 0) {
      updatePrivileges(role);
    }
  }
  for (auto dbObjectIt = cachedPrivileges_.begin();
       dbObjectIt != cachedPrivileges_.end();) {
    if (!dbObjectIt->second->getPrivileges().hasAny()) {
      dbObjectIt = cachedPrivileges_.erase(dbObjectIt);
    } else {
      ++dbObjectIt;
    }
  }
}

void Grantee::revokeAllOnDatabase(int32_t dbId) {
  std::vector<DBObjectMap*> sources = {&cachedPrivileges_, &privileges_};
  for (auto privs : sources) {
    for (auto iter = privs->begin(); iter != privs->end();) {
      if (iter->first.dbId == dbId) {
        iter = privs->erase(iter);
      } else {
        ++iter;
      }
    }
  }
  updatePrivileges();
}

Role::~Role() {
  for (auto it = grantees_.begin(); it != grantees_.end();) {
    auto current_grantee = *it;
    ++it;
    current_grantee->revokeRole(this);
  }
  grantees_.clear();
}

void Role::addGrantee(Grantee* grantee) {
  if (grantees_.find(grantee) == grantees_.end()) {
    grantees_.insert(grantee);
  } else {
    throw runtime_error("Role " + getName() + " have been granted to " +
                        grantee->getName() + " already.");
  }
}

void Role::removeGrantee(Grantee* grantee) {
  if (grantees_.find(grantee) != grantees_.end()) {
    grantees_.erase(grantee);
  } else {
    throw runtime_error("Role " + getName() + " have not been granted to " +
                        grantee->getName() + " .");
  }
}

std::vector<std::string> Role::getGrantees() const {
  std::vector<std::string> grantees;
  for (const auto& role : grantees_) {
    grantees.push_back(role->getName());
  }
  return grantees;
}

void Role::revokeAllOnDatabase(int32_t dbId) {
  Grantee::revokeAllOnDatabase(dbId);
  for (auto grantee : grantees_) {
    grantee->revokeAllOnDatabase(dbId);
  }
}

void Role::updatePrivileges() {
  Grantee::updatePrivileges();
  for (auto grantee : grantees_) {
    grantee->updatePrivileges();
  }
}
