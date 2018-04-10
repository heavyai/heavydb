/*
 * File:   DBObject.cpp
 * Author: norair
 *
 * Created on May 16, 2017, 03:30 PM
 */

#include "DBObject.h"
#include "Catalog.h"

const AccessPrivileges AccessPrivileges::ALL = AccessPrivileges(true, true, true, true, true);
const AccessPrivileges AccessPrivileges::ALL_TABLE = AccessPrivileges(true, true, false, true, false);
const AccessPrivileges AccessPrivileges::ALL_DASHBOARD = AccessPrivileges(true, true, false, false, false);
const AccessPrivileges AccessPrivileges::SELECT = AccessPrivileges(true, false, false, false, false);
const AccessPrivileges AccessPrivileges::INSERT = AccessPrivileges(false, true, false, false, false);
const AccessPrivileges AccessPrivileges::CREATE = AccessPrivileges(false, false, true, false, false);
const AccessPrivileges AccessPrivileges::TRUNCATE = AccessPrivileges(false, false, false, true, false);
const AccessPrivileges AccessPrivileges::CREATE_DASHBOARD = AccessPrivileges(false, false, false, false, true);

std::string DBObjectTypeToString(DBObjectType type) {
  switch (type) {
    case DatabaseDBObjectType:
      return "DATABASE";
    case TableDBObjectType:
      return "TABLE";
    case DashboardDBObjectType:
      return "DASHBOARD";
    case ColumnDBObjectType:
      return "COLUMN";
    default:
      CHECK(false);
  }
}

DBObjectType DBObjectTypeFromString(const std::string& type) {
  if (type.compare("DATABASE") == 0) {
    return DatabaseDBObjectType;
  } else if (type.compare("TABLE") == 0) {
    return TableDBObjectType;
  } else if (type.compare("DASHBOARD") == 0) {
    return DashboardDBObjectType;
  } else {
    throw std::runtime_error("DB object type " + type + " is not supported.");
  }
}

DBObject::DBObject(const std::string& name, const DBObjectType& type) : objectName_(name), objectType_(type) {
  privsValid_ = false;
  ownerId_ = 0;
  objectId_ = 0;
}

DBObject::DBObject(const int32_t id, const DBObjectType& type) : objectId_(id), objectType_(type) {
  objectName_ = "";
  privsValid_ = false;
  ownerId_ = 0;
}

DBObject::DBObject(const DBObject& object)
    : objectName_(object.objectName_),
      objectId_(object.objectId_),
      objectType_(object.objectType_),
      privsValid_(object.privsValid_),
      ownerId_(object.ownerId_) {
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
  objectPrivs_.create_dashboard |= object.objectPrivs_.create_dashboard;
  privsValid_ = true;
}

void DBObject::revokePrivileges(const DBObject& object) {
  objectPrivs_.select &= !object.objectPrivs_.select;
  objectPrivs_.insert &= !object.objectPrivs_.insert;
  objectPrivs_.create &= !object.objectPrivs_.create;
  objectPrivs_.truncate &= !object.objectPrivs_.truncate;
  objectPrivs_.create_dashboard &= !object.objectPrivs_.create_dashboard;
  privsValid_ = true;
}

std::vector<std::string> DBObject::toString() const {
  std::vector<std::string> objectKey;
  switch (objectType_) {
    case DatabaseDBObjectType:
      objectKey.push_back(std::to_string(objectKey_.dbObjectType));
      objectKey.push_back(std::to_string(objectKey_.dbId));
      objectKey.push_back(std::to_string(-1));
      objectKey.push_back(std::to_string(-1));
      break;
    case TableDBObjectType:
    case DashboardDBObjectType:
      objectKey.push_back(std::to_string(objectKey_.dbObjectType));
      objectKey.push_back(std::to_string(objectKey_.dbId));
      objectKey.push_back(std::to_string(objectKey_.tableId));
      objectKey.push_back(std::to_string(-1));
      break;
    case ColumnDBObjectType:
      throw std::runtime_error("Privileges for columns are not supported in current release.");
      break;
    default: { CHECK(false); }
  }
  return objectKey;
}

void DBObject::loadKey(const Catalog_Namespace::Catalog& catalog) {
  DBObjectKey objectKey;
  switch (getType()) {
    case DatabaseDBObjectType: {
      Catalog_Namespace::DBMetadata db;
      if (!Catalog_Namespace::SysCatalog::instance().getMetadataForDB(getName(), db)) {
        throw std::runtime_error("Failure generating DB object key. Database " + getName() + " does not exist.");
      }
      objectKey.dbObjectType = static_cast<int32_t>(DatabaseDBObjectType);
      objectKey.dbId = db.dbId;
      objectId_ = db.dbId;
      ownerId_ = db.dbOwner;
      break;
    }
    case TableDBObjectType: {
      auto table = catalog.getMetadataForTable(getName());
      if (!table) {
        throw std::runtime_error("Failure generating DB object key. Table " + getName() + " does not exist.");
      }
      objectKey.dbObjectType = static_cast<int32_t>(TableDBObjectType);
      objectKey.dbId = catalog.get_currentDB().dbId;
      objectKey.tableId = table->tableId;
      objectId_ = table->tableId;
      ownerId_ = table->userId;
      break;
    }
    case DashboardDBObjectType: {
      auto dashboard = catalog.getMetadataForDashboard(getId());
      if (!dashboard) {
        throw std::runtime_error("Failure generating DB object key. Dashboard with ID " + std::to_string(getId()) +
                                 " does not exist.");
      }
      objectKey.dbObjectType = static_cast<int32_t>(TableDBObjectType);
      objectKey.dbId = catalog.get_currentDB().dbId;
      objectKey.tableId = dashboard->viewId;
      objectName_ = dashboard->viewName;
      ownerId_ = dashboard->userId;
      break;
    }
    case ColumnDBObjectType:
      throw std::runtime_error("Privileges for columns are not supported in current release.");
      break;
    default:
      CHECK(false);
  }
  setObjectKey(objectKey);
}

DBObjectKey DBObjectKey::fromString(const std::vector<std::string>& key, const DBObjectType& type) {
  DBObjectKey objectKey;
  switch (type) {
    case DatabaseDBObjectType:
      objectKey.dbObjectType = std::stoi(key[0]);
      objectKey.dbId = std::stoi(key[1]);
      break;
    case TableDBObjectType:
    case DashboardDBObjectType:
      objectKey.dbObjectType = std::stoi(key[0]);
      objectKey.dbId = std::stoi(key[1]);
      objectKey.tableId = std::stoi(key[2]);
      break;
    case ColumnDBObjectType:
      throw std::runtime_error("Privileges for columns are not supported in current release.");
      break;
    default:
      CHECK(false);
  }
  return objectKey;
}
