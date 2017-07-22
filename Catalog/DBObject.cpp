/*
 * File:   DBObject.cpp
 * Author: norair
 *
 * Created on May 16, 2017, 03:30 PM
 */

#include "DBObject.h"

DBObject::DBObject(const std::string& name, const DBObjectType& type) : objectName_(name), objectType_(type) {
  objectPrivs_.select_ = false;
  objectPrivs_.insert_ = false;
  objectPrivs_.create_ = false;
  privsValid_ = false;
  userPrivateObject_ = false;
  owningUserId_ = 0;
}

DBObject::DBObject(const DBObject& object)
    : objectName_(object.objectName_),
      objectType_(object.objectType_),
      privsValid_(object.privsValid_),
      userPrivateObject_(object.userPrivateObject_),
      owningUserId_(object.owningUserId_) {
  setObjectKey(object.objectKey_);
  copyPrivileges(object);
}

DBObject::~DBObject() {}

std::string DBObject::getName() const {
  return objectName_;
}

DBObjectType DBObject::getType() const {
  return objectType_;
}

DBObjectKey DBObject::getObjectKey() const {
  return objectKey_;
}

void DBObject::setObjectKey(const DBObjectKey& objectKey) {
  objectKey_ = objectKey;
}

std::vector<bool> DBObject::getPrivileges() const {
  std::vector<bool> privs;
  privs.push_back(objectPrivs_.select_);
  privs.push_back(objectPrivs_.insert_);
  privs.push_back(objectPrivs_.create_);
  return privs;
}

void DBObject::setPrivileges(std::vector<bool> priv) {
  for (size_t i = 0; i < priv.size(); i++) {
    if (priv[i]) {
      switch (i) {
        case (0): {
          objectPrivs_.select_ = true;
          break;
        }
        case (1): {
          objectPrivs_.insert_ = true;
          break;
        }
        case (2): {
          objectPrivs_.create_ = true;
          break;
        }
        default: { CHECK(false); }
      }
    }
  }
}

void DBObject::copyPrivileges(const DBObject& object) {
  // objectPrivs_ = object.objectPrivs_;
  objectPrivs_.select_ = object.objectPrivs_.select_;
  objectPrivs_.insert_ = object.objectPrivs_.insert_;
  objectPrivs_.create_ = object.objectPrivs_.create_;
  privsValid_ = true;
}

void DBObject::updatePrivileges(const DBObject& object) {
  objectPrivs_.select_ |= object.objectPrivs_.select_;
  objectPrivs_.insert_ |= object.objectPrivs_.insert_;
  objectPrivs_.create_ |= object.objectPrivs_.create_;
  privsValid_ = true;
}

void DBObject::grantPrivileges(const DBObject& object) {
  updatePrivileges(object);
}

void DBObject::revokePrivileges(const DBObject& object) {
  if (object.objectPrivs_.select_) {
    objectPrivs_.select_ = false;
  }
  if (object.objectPrivs_.insert_) {
    objectPrivs_.insert_ = false;
  }
  if (object.objectPrivs_.create_) {
    objectPrivs_.create_ = false;
  }
  privsValid_ = true;
}

bool DBObject::isUserPrivateObject() const {
  return userPrivateObject_;
}

void DBObject::setUserPrivateObject() {
  userPrivateObject_ = true;
}

int32_t DBObject::getOwningUserId() const {
  return owningUserId_;
}

void DBObject::setOwningUserId(int32_t userId) {
  owningUserId_ = userId;
}
