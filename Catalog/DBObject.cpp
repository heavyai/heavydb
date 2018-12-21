/*
 * File:   DBObject.cpp
 * Author: norair
 *
 * Created on May 16, 2017, 03:30 PM
 */

#include "DBObject.h"
#include "Catalog.h"

const AccessPrivileges AccessPrivileges::NONE = AccessPrivileges(0);

const AccessPrivileges AccessPrivileges::ALL_DATABASE =
    AccessPrivileges(DatabasePrivileges::ALL);
const AccessPrivileges AccessPrivileges::VIEW_SQL_EDITOR =
    AccessPrivileges(DatabasePrivileges::VIEW_SQL_EDITOR);
const AccessPrivileges AccessPrivileges::ACCESS =
    AccessPrivileges(DatabasePrivileges::ACCESS);

const AccessPrivileges AccessPrivileges::ALL_TABLE =
    AccessPrivileges(TablePrivileges::ALL);
const AccessPrivileges AccessPrivileges::ALL_TABLE_MIGRATE =
    AccessPrivileges(TablePrivileges::ALL_MIGRATE);
const AccessPrivileges AccessPrivileges::CREATE_TABLE =
    AccessPrivileges(TablePrivileges::CREATE_TABLE);
const AccessPrivileges AccessPrivileges::DROP_TABLE =
    AccessPrivileges(TablePrivileges::DROP_TABLE);
const AccessPrivileges AccessPrivileges::SELECT_FROM_TABLE =
    AccessPrivileges(TablePrivileges::SELECT_FROM_TABLE);
const AccessPrivileges AccessPrivileges::INSERT_INTO_TABLE =
    AccessPrivileges(TablePrivileges::INSERT_INTO_TABLE);
const AccessPrivileges AccessPrivileges::UPDATE_IN_TABLE =
    AccessPrivileges(TablePrivileges::UPDATE_IN_TABLE);
const AccessPrivileges AccessPrivileges::DELETE_FROM_TABLE =
    AccessPrivileges(TablePrivileges::DELETE_FROM_TABLE);
const AccessPrivileges AccessPrivileges::TRUNCATE_TABLE =
    AccessPrivileges(TablePrivileges::TRUNCATE_TABLE);
const AccessPrivileges AccessPrivileges::ALTER_TABLE =
    AccessPrivileges(TablePrivileges::ALTER_TABLE);

const AccessPrivileges AccessPrivileges::ALL_DASHBOARD =
    AccessPrivileges(DashboardPrivileges::ALL);
const AccessPrivileges AccessPrivileges::ALL_DASHBOARD_MIGRATE =
    AccessPrivileges(DashboardPrivileges::ALL_MIGRATE);
const AccessPrivileges AccessPrivileges::CREATE_DASHBOARD =
    AccessPrivileges(DashboardPrivileges::CREATE_DASHBOARD);
const AccessPrivileges AccessPrivileges::EDIT_DASHBOARD =
    AccessPrivileges(DashboardPrivileges::EDIT_DASHBOARD);
const AccessPrivileges AccessPrivileges::DELETE_DASHBOARD =
    AccessPrivileges(DashboardPrivileges::DELETE_DASHBOARD);
const AccessPrivileges AccessPrivileges::VIEW_DASHBOARD =
    AccessPrivileges(DashboardPrivileges::VIEW_DASHBOARD);

const AccessPrivileges AccessPrivileges::ALL_VIEW = AccessPrivileges(ViewPrivileges::ALL);
const AccessPrivileges AccessPrivileges::ALL_VIEW_MIGRATE =
    AccessPrivileges(ViewPrivileges::ALL_MIGRATE);
const AccessPrivileges AccessPrivileges::CREATE_VIEW =
    AccessPrivileges(ViewPrivileges::CREATE_VIEW);
const AccessPrivileges AccessPrivileges::DROP_VIEW =
    AccessPrivileges(ViewPrivileges::DROP_VIEW);
const AccessPrivileges AccessPrivileges::SELECT_FROM_VIEW =
    AccessPrivileges(ViewPrivileges::SELECT_FROM_VIEW);
const AccessPrivileges AccessPrivileges::INSERT_INTO_VIEW =
    AccessPrivileges(ViewPrivileges::INSERT_INTO_VIEW);
const AccessPrivileges AccessPrivileges::UPDATE_IN_VIEW =
    AccessPrivileges(ViewPrivileges::UPDATE_IN_VIEW);
const AccessPrivileges AccessPrivileges::DELETE_FROM_VIEW =
    AccessPrivileges(ViewPrivileges::DELETE_FROM_VIEW);
const AccessPrivileges AccessPrivileges::TRUNCATE_VIEW =
    AccessPrivileges(ViewPrivileges::TRUNCATE_VIEW);

std::string ObjectPermissionTypeToString(DBObjectType type) {
  switch (type) {
    case DatabaseDBObjectType:
      return "DATABASE";
    case TableDBObjectType:
      return "TABLE";
    case DashboardDBObjectType:
      return "DASHBOARD";
    case ViewDBObjectType:
      return "VIEW";
    default:
      CHECK(false);
  }
  return "not possible";
}

DBObjectType DBObjectTypeFromString(const std::string& type) {
  if (type.compare("DATABASE") == 0) {
    return DatabaseDBObjectType;
  } else if (type.compare("TABLE") == 0) {
    return TableDBObjectType;
  } else if (type.compare("DASHBOARD") == 0) {
    return DashboardDBObjectType;
  } else if (type.compare("VIEW") == 0) {
    return ViewDBObjectType;
  } else {
    throw std::runtime_error("DB object type " + type + " is not supported.");
  }
}

DBObject::DBObject(const std::string& name, const DBObjectType& objectAndPermissionType)
    : objectName_(name) {
  objectType_ = objectAndPermissionType;
  objectKey_.permissionType = objectAndPermissionType;
  ownerId_ = 0;
}

DBObject::DBObject(const int32_t id, const DBObjectType& objectAndPermissionType)
    : objectName_("") {
  objectType_ = objectAndPermissionType;
  objectKey_.permissionType = objectAndPermissionType;
  objectKey_.objectId = id;
  ownerId_ = 0;
}

DBObject::DBObject(const DBObject& object)
    : objectName_(object.objectName_), ownerId_(object.ownerId_) {
  objectType_ = object.objectType_;
  setObjectKey(object.objectKey_);
  copyPrivileges(object);
}

void DBObject::copyPrivileges(const DBObject& object) {
  objectPrivs_ = object.objectPrivs_;
}

void DBObject::updatePrivileges(const DBObject& object) {
  objectPrivs_.privileges |= object.objectPrivs_.privileges;
}

void DBObject::revokePrivileges(const DBObject& object) {
  objectPrivs_.privileges &= ~(object.objectPrivs_.privileges);
}

void DBObject::setPermissionType(const DBObjectType& permissionType) {
  objectKey_.permissionType = permissionType;
}
void DBObject::setObjectType(const DBObjectType& objectType) {
  objectType_ = objectType;
}

std::vector<std::string> DBObject::toString() const {
  std::vector<std::string> objectKey;
  switch (objectKey_.permissionType) {
    case DatabaseDBObjectType:
      objectKey.push_back(std::to_string(objectKey_.permissionType));
      objectKey.push_back(std::to_string(objectKey_.dbId));
      objectKey.push_back(std::to_string(-1));
      break;
    case TableDBObjectType:
    case DashboardDBObjectType:
    case ViewDBObjectType:
      objectKey.push_back(std::to_string(objectKey_.permissionType));
      objectKey.push_back(std::to_string(objectKey_.dbId));
      objectKey.push_back(std::to_string(objectKey_.objectId));
      break;
    default: { CHECK(false); }
  }
  return objectKey;
}

void DBObject::loadKey(const Catalog_Namespace::Catalog& catalog) {
  switch (objectType_) {
    case DatabaseDBObjectType: {
      // permissions at DB level

      if (!getName().empty()) {
        Catalog_Namespace::DBMetadata db;
        if (!Catalog_Namespace::SysCatalog::instance().getMetadataForDB(getName(), db)) {
          throw std::runtime_error("Failure generating DB object key. Database " +
                                   getName() + " does not exist.");
        }
        objectKey_.dbId = db.dbId;
        ownerId_ = db.dbOwner;
      } else {
        objectKey_.dbId = 0;  // very special case only used for initialisation of a role
      }
      break;
    }
    case ViewDBObjectType:
    case TableDBObjectType: {
      // permissions on tables

      objectKey_.dbId = catalog.getCurrentDB().dbId;

      if (!getName().empty()) {
        auto table =
            catalog.getMetadataForTable(getName(), false /* do not populate fragments */);
        if (!table) {
          throw std::runtime_error("Failure generating DB object key. Table/View " +
                                   getName() + " does not exist.");
        }
        objectKey_.objectId = table->tableId;
        ownerId_ = table->userId;
      } else {
        // table permission at db level
        ownerId_ = catalog.getCurrentDB().dbOwner;
      }

      break;
    }
    case DashboardDBObjectType: {
      // permissions on dashboards
      objectKey_.dbId = catalog.getCurrentDB().dbId;

      if (objectKey_.objectId > 0) {
        auto dashboard = catalog.getMetadataForDashboard(objectKey_.objectId);
        if (!dashboard) {
          throw std::runtime_error(
              "Failure generating DB object key. Dashboard with ID " +
              std::to_string(objectKey_.objectId) + " does not exist.");
        }
        objectName_ = dashboard->viewName;
        ownerId_ = dashboard->userId;
      } else {
        // dashboard permission at DB level
        ownerId_ = catalog.getCurrentDB().dbOwner;
      }

      break;
    }
    default:
      CHECK(false);
  }
}

DBObjectKey DBObjectKey::fromString(const std::vector<std::string>& key,
                                    const DBObjectType& type) {
  DBObjectKey objectKey;
  switch (type) {
    case DatabaseDBObjectType:
      objectKey.permissionType = std::stoi(key[0]);
      objectKey.dbId = std::stoi(key[1]);
      break;
    case TableDBObjectType:
    case ViewDBObjectType:
    case DashboardDBObjectType:
      objectKey.permissionType = std::stoi(key[0]);
      objectKey.dbId = std::stoi(key[1]);
      objectKey.objectId = std::stoi(key[2]);
      break;
    default:
      CHECK(false);
  }
  return objectKey;
}
