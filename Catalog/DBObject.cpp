/*
 * File:   DBObject.cpp
 * Author: norair
 *
 * Created on May 16, 2017, 03:30 PM
 */

#include "DBObject.h"
#include "Catalog.h"

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

std::vector<std::string> DBObject::toString() const {
  std::vector<std::string> objectKey;
  switch (objectType_) {
    case (DatabaseDBObjectType): {
      objectKey.push_back(std::to_string(objectKey_.dbObjectType));
      objectKey.push_back(std::to_string(objectKey_.dbId));
      objectKey.push_back(std::to_string(-1));
      objectKey.push_back(std::to_string(-1));
      break;
    }
    case (TableDBObjectType): {
      objectKey.push_back(std::to_string(objectKey_.dbObjectType));
      objectKey.push_back(std::to_string(objectKey_.dbId));
      objectKey.push_back(std::to_string(objectKey_.tableId));
      objectKey.push_back(std::to_string(-1));
      break;
    }
    case (ColumnDBObjectType): {
      throw std::runtime_error("Privileges for columns are not supported in current release.");
      break;
    }
    case (DashboardDBObjectType): {
      throw std::runtime_error("Privileges for dashboards are not supported in current release.");
      break;
    }
    default: { CHECK(false); }
  }
  return objectKey;
}

void DBObject::loadKey(const Catalog_Namespace::Catalog& catalog) {
  DBObjectKey objectKey;
  switch (getType()) {
    case (DatabaseDBObjectType): {
      Catalog_Namespace::DBMetadata db;
      if (!Catalog_Namespace::SysCatalog::instance().getMetadataForDB(getName(), db)) {
        throw std::runtime_error("Failure generating DB object key. Database " + getName() + " does not exist.");
      }
      objectKey.dbObjectType = static_cast<int32_t>(DatabaseDBObjectType);
      objectKey.dbId = db.dbId;
      break;
    }
    case (TableDBObjectType): {
      if (!catalog.getMetadataForTable(getName())) {
        throw std::runtime_error("Failure generating DB object key. Table " + getName() + " does not exist.");
      }
      objectKey.dbObjectType = static_cast<int32_t>(TableDBObjectType);
      objectKey.dbId = catalog.get_currentDB().dbId;
      objectKey.tableId = catalog.getMetadataForTable(getName())->tableId;
      break;
    }
    case (ColumnDBObjectType): {
      break;
    }
    case (DashboardDBObjectType): {
      break;
    }
    default: { CHECK(false); }
  }
  setObjectKey(objectKey);
}

DBObjectKey DBObjectKey::fromString(const std::vector<std::string>& key, const DBObjectType& type) {
  DBObjectKey objectKey;
  switch (type) {
    case (DatabaseDBObjectType): {
      objectKey.dbObjectType = std::stoi(key[0]);
      objectKey.dbId = std::stoi(key[1]);
      break;
    }
    case (TableDBObjectType): {
      objectKey.dbObjectType = std::stoi(key[0]);
      objectKey.dbId = std::stoi(key[1]);
      objectKey.tableId = std::stoi(key[2]);
      break;
    }
    case (ColumnDBObjectType): {
      throw std::runtime_error("Privileges for columns are not supported in current release.");
      break;
    }
    case (DashboardDBObjectType): {
      throw std::runtime_error("Privileges for dashboards are not supported in current release.");
      break;
    }
    default: { CHECK(false); }
  }
  return objectKey;
}
