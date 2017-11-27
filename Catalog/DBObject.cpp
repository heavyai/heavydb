/*
 * File:   DBObject.cpp
 * Author: norair
 *
 * Created on May 16, 2017, 03:30 PM
 */

#include "DBObject.h"

const AccessPrivileges AccessPrivileges::ALL = AccessPrivileges(true, true, true, true);
const AccessPrivileges AccessPrivileges::ALL_NO_DB = AccessPrivileges(true, true, false, true);
const AccessPrivileges AccessPrivileges::SELECT = AccessPrivileges(true, false, false, false);
const AccessPrivileges AccessPrivileges::INSERT = AccessPrivileges(false, true, false, false);
const AccessPrivileges AccessPrivileges::CREATE = AccessPrivileges(false, false, true, false);
const AccessPrivileges AccessPrivileges::TRUNCATE = AccessPrivileges(false, false, false, true);

DBObject::DBObject(const std::string& name, const DBObjectType& type) : objectName_(name), objectType_(type) {
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

void DBObject::copyPrivileges(const DBObject& object) {
  objectPrivs_ = object.objectPrivs_;
  privsValid_ = true;
}

void DBObject::updatePrivileges(const DBObject& object) {
  objectPrivs_.select |= object.objectPrivs_.select;
  objectPrivs_.insert |= object.objectPrivs_.insert;
  objectPrivs_.create |= object.objectPrivs_.create;
  objectPrivs_.truncate |= object.objectPrivs_.truncate;
  privsValid_ = true;
}

void DBObject::revokePrivileges(const DBObject& object) {
  objectPrivs_.select &= !object.objectPrivs_.select;
  objectPrivs_.insert &= !object.objectPrivs_.insert;
  objectPrivs_.create &= !object.objectPrivs_.create;
  objectPrivs_.truncate &= !object.objectPrivs_.truncate;
  privsValid_ = true;
}
