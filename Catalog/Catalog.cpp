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

/**
 * @file		Catalog.cpp
 * @author	Todd Mostak <todd@map-d.com>, Wei Hong <wei@map-d.com>
 * @brief		Functions for System Catalogs
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "Catalog.h"
#include <list>
#include <exception>
#include <cassert>
#include <memory>
#include <random>
#include <boost/filesystem.hpp>
#include <boost/uuid/sha1.hpp>
#include "../Fragmenter/Fragmenter.h"
#include "../Fragmenter/InsertOrderFragmenter.h"
#include "../Parser/ParserNode.h"
#include "../Shared/StringTransform.h"
#include "../Shared/measure.h"
#include "../StringDictionary/StringDictionaryClient.h"

using std::runtime_error;
using std::string;
using std::map;
using std::list;
using std::pair;
using std::vector;
using Chunk_NS::Chunk;
using Fragmenter_Namespace::InsertOrderFragmenter;

bool g_aggregator{false};

namespace Catalog_Namespace {

const std::string Catalog::physicalTableNameTag_("_shard_#");
Catalog_Namespace::SysCatalog* mapd_sys_cat(nullptr);

SysCatalog::~SysCatalog() {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  for (RoleMap::iterator roleIt = roleMap_.begin(); roleIt != roleMap_.end(); ++roleIt) {
    delete roleIt->second;
  }
  roleMap_.clear();
  for (UserRoleMap::iterator userRoleIt = userRoleMap_.begin(); userRoleIt != userRoleMap_.end(); ++userRoleIt) {
    delete userRoleIt->second;
  }
  userRoleMap_.clear();
}

void SysCatalog::initDB() {
  sqliteConnector_.query(
      "CREATE TABLE mapd_users (userid integer primary key, name text unique, passwd text, issuper boolean)");
  sqliteConnector_.query_with_text_params(
      "INSERT INTO mapd_users VALUES (?, ?, ?, 1)",
      std::vector<std::string>{MAPD_ROOT_USER_ID_STR, MAPD_ROOT_USER, MAPD_ROOT_PASSWD_DEFAULT});
  sqliteConnector_.query(
      "CREATE TABLE mapd_databases (dbid integer primary key, name text unique, owner integer references mapd_users)");
  sqliteConnector_.query(
      "CREATE TABLE mapd_privileges (userid integer references mapd_users, dbid integer references mapd_databases, "
      "select_priv boolean, insert_priv boolean, UNIQUE(userid, dbid))");
  createDatabase("mapd", MAPD_ROOT_USER_ID);
};

void SysCatalog::migrateSysCatalogSchema() {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "CREATE TABLE IF NOT EXISTS mapd_privileges (userid integer references mapd_users, dbid integer references "
        "mapd_databases, select_priv boolean, insert_priv boolean, UNIQUE(userid, dbid))");
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void SysCatalog::initObjectPrivileges() {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "CREATE TABLE IF NOT EXISTS mapd_roles(roleName text, userName text, UNIQUE(roleName, userName))");
    sqliteConnector_.query(
        "CREATE TABLE IF NOT EXISTS mapd_object_privileges(roleName text, roleType bool, "
        "objectName text, objectType integer, dbObjectType integer, dbId integer "
        "references mapd_databases, tableId integer references mapd_tables, columnId integer references "
        "mapd_columns, "
        "privSelect bool, privInsert bool, privCreate bool, privTruncate bool, privDelete bool, privUpdate bool, "
        "UNIQUE(roleName, dbObjectType, dbId, tableId, columnId))");
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void SysCatalog::createUser(const string& name, const string& passwd, bool issuper) {
  UserMetadata user;
  if (getMetadataForUser(name, user)) {
    throw runtime_error("User " + name + " already exists.");
  }
  if (access_priv_check_ && getMetadataForRole(name)) {
    throw runtime_error("User name " + name + " is same as one of role names. User and role names should be unique.");
  }
  sqliteConnector_.query_with_text_params("INSERT INTO mapd_users (name, passwd, issuper) VALUES (?, ?, ?)",
                                          std::vector<std::string>{name, passwd, std::to_string(issuper)});
  if (access_priv_check_) {
    createRole(name, true);
    grantDefaultPrivilegesToRole(name, issuper);
    grantRole(name, name);
    if (!getMetadataForRole(MAPD_DEFAULT_ROOT_USER_ROLE)) {
      createDefaultMapdRoles();
    }
    if (issuper) {
      grantRole(MAPD_DEFAULT_ROOT_USER_ROLE, name);
    } else {
      grantRole(MAPD_DEFAULT_USER_ROLE, name);
    }
  }
}

void SysCatalog::dropUser(const string& name) {
  if (access_priv_check_) {
    UserMetadata user;
    if (getMetadataForUser(name, user)) {
      dropRole(name);
      dropUserRole(name);
      const std::string& roleName(name);
      sqliteConnector_.query("BEGIN TRANSACTION");
      try {
        sqliteConnector_.query_with_text_param("DELETE FROM mapd_roles WHERE userName = ?", roleName);
      } catch (const std::exception& e) {
        sqliteConnector_.query("ROLLBACK TRANSACTION");
        throw;
      }
      sqliteConnector_.query("END TRANSACTION");
    }
  }
  UserMetadata user;
  if (!getMetadataForUser(name, user))
    throw runtime_error("User " + name + " does not exist.");
  sqliteConnector_.query("DELETE FROM mapd_users WHERE userid = " + std::to_string(user.userId));
  sqliteConnector_.query("DELETE FROM mapd_privileges WHERE userid = " + std::to_string(user.userId));
}

void SysCatalog::alterUser(const int32_t userid, const string* passwd, bool* is_superp) {
  if (passwd != nullptr && is_superp != nullptr)
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_users SET passwd = ?, issuper = ? WHERE userid = ?",
        std::vector<std::string>{*passwd, std::to_string(*is_superp), std::to_string(userid)});
  else if (passwd != nullptr)
    sqliteConnector_.query_with_text_params("UPDATE mapd_users SET passwd = ? WHERE userid = ?",
                                            std::vector<std::string>{*passwd, std::to_string(userid)});
  else if (is_superp != nullptr)
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_users SET issuper = ? WHERE userid = ?",
        std::vector<std::string>{std::to_string(*is_superp), std::to_string(userid)});
}

void SysCatalog::grantPrivileges(const int32_t userid, const int32_t dbid, const Privileges& privs) {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "INSERT OR REPLACE INTO mapd_privileges (userid, dbid, select_priv, insert_priv) VALUES (?1, ?2, ?3, ?4)",
        std::vector<std::string>{std::to_string(userid),
                                 std::to_string(dbid),
                                 std::to_string(privs.select_),
                                 std::to_string(privs.insert_)});
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

bool SysCatalog::checkPrivileges(UserMetadata& user, DBMetadata& db, const Privileges& wants_privs) {
  if (user.isSuper || user.userId == db.dbOwner) {
    return true;
  }
  sqliteConnector_.query_with_text_params(
      "SELECT select_priv, insert_priv FROM mapd_privileges "
      "WHERE userid = ?1 AND dbid = ?2;",
      std::vector<std::string>{std::to_string(user.userId), std::to_string(db.dbId)});
  int numRows = sqliteConnector_.getNumRows();
  if (numRows == 0) {
    return false;
  }
  Privileges has_privs;
  has_privs.select_ = sqliteConnector_.getData<bool>(0, 0);
  has_privs.insert_ = sqliteConnector_.getData<bool>(0, 1);

  if (wants_privs.select_ && wants_privs.select_ != has_privs.select_)
    return false;
  if (wants_privs.insert_ && wants_privs.insert_ != has_privs.insert_)
    return false;

  return true;
}

void SysCatalog::createDatabase(const string& name, int owner) {
  DBMetadata db;
  if (getMetadataForDB(name, db))
    throw runtime_error("Database " + name + " already exists.");
  sqliteConnector_.query_with_text_param(
      "INSERT INTO mapd_databases (name, owner) VALUES (?, " + std::to_string(owner) + ")", name);
  SqliteConnector dbConn(name, basePath_ + "/mapd_catalogs/");
  dbConn.query(
      "CREATE TABLE mapd_tables (tableid integer primary key, name text unique, ncolumns integer, isview boolean, "
      "fragments text, frag_type integer, max_frag_rows integer, max_chunk_size bigint, frag_page_size integer, "
      "max_rows bigint, partitions text, shard_column_id integer, shard integer, num_shards integer)");
  dbConn.query(
      "CREATE TABLE mapd_columns (tableid integer references mapd_tables, columnid integer, name text, coltype "
      "integer, colsubtype integer, coldim integer, colscale integer, is_notnull boolean, compression integer, "
      "comp_param integer, size integer, chunks text, is_systemcol boolean, is_virtualcol boolean, virtual_expr text, "
      "primary key(tableid, columnid), unique(tableid, name))");
  dbConn.query("CREATE TABLE mapd_views (tableid integer references mapd_tables, sql text)");
  dbConn.query(
      "CREATE TABLE mapd_frontend_views (viewid integer primary key, name text unique, userid integer references "
      "mapd_users, view_state text, image_hash text, update_time timestamp, view_metadata text)");
  dbConn.query(
      "CREATE TABLE mapd_links (linkid integer primary key, userid integer references mapd_users, "
      "link text unique, view_state text, update_time timestamp, view_metadata text)");
  dbConn.query(
      "CREATE TABLE mapd_dictionaries (dictid integer primary key, name text unique, nbits int, is_shared boolean, "
      "refcount int)");
  dbConn.query("CREATE TABLE mapd_logical_to_physical(logical_table_id integer, physical_table_id integer)");
}

void SysCatalog::dropDatabase(const int32_t dbid, const std::string& name) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query_with_text_param("DELETE FROM mapd_databases WHERE dbid = ?", std::to_string(dbid));
  boost::filesystem::remove(basePath_ + "/mapd_catalogs/" + name);
  ChunkKey chunkKeyPrefix = {dbid};
  calciteMgr_->updateMetadata(name, "");
  dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix);
  /* don't need to checkpoint as database is being dropped */
  // dataMgr_->checkpoint();
}

bool SysCatalog::checkPasswordForUser(const std::string& passwd, UserMetadata& user) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  {
    if (user.passwd != passwd) {
      return false;
    }
  }
  return true;
}

bool SysCatalog::getMetadataForUser(const string& name, UserMetadata& user) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query_with_text_param("SELECT userid, name, passwd, issuper FROM mapd_users WHERE name = ?", name);
  int numRows = sqliteConnector_.getNumRows();
  if (numRows == 0)
    return false;
  user.userId = sqliteConnector_.getData<int>(0, 0);
  user.userName = sqliteConnector_.getData<string>(0, 1);
  user.passwd = sqliteConnector_.getData<string>(0, 2);
  user.isSuper = sqliteConnector_.getData<bool>(0, 3);
  return true;
}

list<DBMetadata> SysCatalog::getAllDBMetadata() {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query("SELECT dbid, name, owner FROM mapd_databases");
  int numRows = sqliteConnector_.getNumRows();
  list<DBMetadata> db_list;
  for (int r = 0; r < numRows; ++r) {
    DBMetadata db;
    db.dbId = sqliteConnector_.getData<int>(r, 0);
    db.dbName = sqliteConnector_.getData<string>(r, 1);
    db.dbOwner = sqliteConnector_.getData<int>(r, 2);
    db_list.push_back(db);
  }
  return db_list;
}

list<UserMetadata> SysCatalog::getAllUserMetadata() {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query("SELECT userid, name, issuper FROM mapd_users");
  int numRows = sqliteConnector_.getNumRows();
  list<UserMetadata> user_list;
  for (int r = 0; r < numRows; ++r) {
    UserMetadata user;
    user.userId = sqliteConnector_.getData<int>(r, 0);
    user.userName = sqliteConnector_.getData<string>(r, 1);
    user.isSuper = sqliteConnector_.getData<bool>(r, 2);
    user_list.push_back(user);
  }
  return user_list;
}

bool SysCatalog::getMetadataForDB(const string& name, DBMetadata& db) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query_with_text_param("SELECT dbid, name, owner FROM mapd_databases WHERE name = ?", name);
  int numRows = sqliteConnector_.getNumRows();
  if (numRows == 0)
    return false;
  db.dbId = sqliteConnector_.getData<int>(0, 0);
  db.dbName = sqliteConnector_.getData<string>(0, 1);
  db.dbOwner = sqliteConnector_.getData<int>(0, 2);
  return true;
}

void SysCatalog::createDefaultMapdRoles() {
  DBObject dbObject(get_currentDB().dbName, DatabaseDBObjectType);
  populateDBObjectKey(dbObject, *this);

  // create default non suser role
  if (!mapd_sys_cat->getMetadataForRole(MAPD_DEFAULT_USER_ROLE)) {
    createRole(MAPD_DEFAULT_USER_ROLE);
    grantDBObjectPrivileges(MAPD_DEFAULT_USER_ROLE, dbObject, *this);
  }
  // create default suser role
  if (!mapd_sys_cat->getMetadataForRole(MAPD_DEFAULT_ROOT_USER_ROLE)) {
    createRole(MAPD_DEFAULT_ROOT_USER_ROLE);
    dbObject.setPrivileges({true, true, true, true});
    grantDBObjectPrivileges(MAPD_DEFAULT_ROOT_USER_ROLE, dbObject, *this);
    grantRole(MAPD_DEFAULT_ROOT_USER_ROLE, MAPD_ROOT_USER);
  }
}

void SysCatalog::grantDefaultPrivilegesToRole(const std::string& name, bool issuper) {
  DBObject dbObject(get_currentDB().dbName, DatabaseDBObjectType);
  populateDBObjectKey(dbObject, *this);
  if (issuper) {
    dbObject.setPrivileges({true, true, true, true});
  }
  grantDBObjectPrivileges(name, dbObject, *this);
}

std::vector<std::string> SysCatalog::convertObjectKeyToString(const DBObject& object) {
  std::vector<std::string> objectKey;
  switch (object.getType()) {
    case (DatabaseDBObjectType): {
      objectKey.push_back(std::to_string(object.getObjectKey()[0]));
      objectKey.push_back(std::to_string(object.getObjectKey()[1]));
      objectKey.push_back(std::to_string(0));
      objectKey.push_back(std::to_string(0));
      break;
    }
    case (TableDBObjectType): {
      objectKey.push_back(std::to_string(object.getObjectKey()[0]));
      objectKey.push_back(std::to_string(object.getObjectKey()[1]));
      objectKey.push_back(std::to_string(object.getObjectKey()[2]));
      objectKey.push_back(std::to_string(0));
      break;
    }
    case (ColumnDBObjectType): {
      throw runtime_error("Privileges for columns are not supported in current release.");
      break;
    }
    case (DashboardDBObjectType): {
      throw runtime_error("Privileges for dashboards are not supported in current release.");
      break;
    }
    default: { CHECK(false); }
  }
  return objectKey;
}

std::vector<int32_t> SysCatalog::convertObjectKeyFromString(const std::vector<std::string>& key,
                                                            const DBObjectType& type) {
  std::vector<int32_t> objectKey;
  switch (type) {
    case (DatabaseDBObjectType): {
      objectKey.push_back(std::stoi(key[0]));
      objectKey.push_back(std::stoi(key[1]));
      break;
    }
    case (TableDBObjectType): {
      objectKey.push_back(std::stoi(key[0]));
      objectKey.push_back(std::stoi(key[1]));
      objectKey.push_back(std::stoi(key[2]));
      break;
    }
    case (ColumnDBObjectType): {
      throw runtime_error("Privileges for columns are not supported in current release.");
      break;
    }
    case (DashboardDBObjectType): {
      throw runtime_error("Privileges for dashboards are not supported in current release.");
      break;
    }
    default: { CHECK(false); }
  }
  return objectKey;
}

void SysCatalog::populateDBObjectKey(DBObject& object, const Catalog_Namespace::Catalog& catalog) {
  DBObjectKey objectKey;
  switch (object.getType()) {
    case (DatabaseDBObjectType): {
      Catalog_Namespace::DBMetadata db;
      if (!mapd_sys_cat->getMetadataForDB(object.getName(), db)) {
        throw std::runtime_error("Failure generating DB object key. Database " + object.getName() + " does not exist.");
      }
      objectKey = {static_cast<int32_t>(DatabaseDBObjectType), db.dbId};
      break;
    }
    case (TableDBObjectType): {
      if (!catalog.getMetadataForTable(object.getName())) {
        throw std::runtime_error("Failure generating DB object key. Table " + object.getName() + " does not exist.");
      }
      objectKey = {static_cast<int32_t>(TableDBObjectType),
                   catalog.get_currentDB().dbId,
                   catalog.getMetadataForTable(object.getName())->tableId};
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
  object.setObjectKey(objectKey);
}

void SysCatalog::createDBObject(const UserMetadata& user,
                                const std::string& objectName,
                                const Catalog_Namespace::Catalog& catalog) {
  Role* user_rl = mapd_sys_cat->getMetadataForUserRole(user.userId);
  if (!user_rl) {
    if (!mapd_sys_cat->getMetadataForRole(MAPD_DEFAULT_ROOT_USER_ROLE)) {
      createDefaultMapdRoles();
    }
    if (user.isSuper) {
      grantRole(MAPD_DEFAULT_ROOT_USER_ROLE, user.userName);
    } else {
      grantRole(MAPD_DEFAULT_USER_ROLE, user.userName);
    }
    user_rl = mapd_sys_cat->getMetadataForUserRole(user.userId);
  }
  DBObject* object = new DBObject(objectName, TableDBObjectType);
  populateDBObjectKey(*object, catalog);
  object->setPrivileges(
      {true, true, false, true});  // as an owner/creator of this object, the user will have all access rights
  object->setUserPrivateObject();
  object->setOwningUserId(user.userId);
  user_rl->grantPrivileges(*object);
}

// GRANT INSERT ON TABLE payroll_table TO payroll_dept_role;
void SysCatalog::grantDBObjectPrivileges(const std::string& roleName,
                                         DBObject& object,
                                         const Catalog_Namespace::Catalog& catalog) {
  if (!roleName.compare(MAPD_ROOT_USER)) {
    throw runtime_error("Request to grant privileges to " + roleName +
                        " failed because mapd root user has all privileges by default.");
  }
  Role* rl = mapd_sys_cat->getMetadataForRole(roleName);
  if (!rl) {
    throw runtime_error("Request to grant privileges to " + roleName +
                        " failed because role or user with this name does not exist.");
  }
  populateDBObjectKey(object, catalog);
  rl->grantPrivileges(object);

  /* apply grant privileges statement to sqlite DB */
  std::vector<std::string> objectKey = convertObjectKeyToString(object);
  object.resetPrivileges();
  rl->getPrivileges(object);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "INSERT OR REPLACE INTO mapd_object_privileges(roleName, roleType, objectName, objectType, dbObjectType, dbId, "
        "tableId, "
        "columnId, "
        "privSelect, privInsert, privCreate, privTruncate, privDelete, privUpdate) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, "
        "?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        std::vector<std::string>{roleName,
                                 std::to_string(rl->isUserPrivateRole()),
                                 object.getName(),
                                 std::to_string(object.getType()),
                                 objectKey[0],
                                 objectKey[1],
                                 objectKey[2],
                                 objectKey[3],
                                 std::to_string(object.getPrivileges()[0]),
                                 std::to_string(object.getPrivileges()[1]),
                                 std::to_string(object.getPrivileges()[2]),
                                 std::to_string(object.getPrivileges()[3]),
                                 std::to_string(false),
                                 std::to_string(false)});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

// REVOKE INSERT ON TABLE payroll_table FROM payroll_dept_role;
void SysCatalog::revokeDBObjectPrivileges(const std::string& roleName,
                                          DBObject& object,
                                          const Catalog_Namespace::Catalog& catalog) {
  if (!roleName.compare(MAPD_ROOT_USER)) {
    throw runtime_error("Request to revoke privileges from " + roleName +
                        " failed because privileges can not be revoked from mapd root user.");
  }
  Role* rl = mapd_sys_cat->getMetadataForRole(roleName);
  if (!rl) {
    throw runtime_error("Request to revoke privileges from " + roleName +
                        " failed because role or user with this name does not exist.");
  }
  populateDBObjectKey(object, catalog);
  rl->revokePrivileges(object);

  /* apply revoke privileges statement to sqlite DB */
  std::vector<std::string> objectKey = convertObjectKeyToString(object);
  object.resetPrivileges();
  rl->getPrivileges(object);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "INSERT OR REPLACE INTO mapd_object_privileges(roleName, roleType, objectName, objectType, dbObjectType, dbId, "
        "tableId, "
        "columnId, "
        "privSelect, privInsert, privCreate, privTruncate, privDelete, privUpdate) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, "
        "?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        std::vector<std::string>{roleName,
                                 std::to_string(rl->isUserPrivateRole()),
                                 object.getName(),
                                 std::to_string(object.getType()),
                                 objectKey[0],
                                 objectKey[1],
                                 objectKey[2],
                                 objectKey[3],
                                 std::to_string(object.getPrivileges()[0]),
                                 std::to_string(object.getPrivileges()[1]),
                                 std::to_string(object.getPrivileges()[2]),
                                 std::to_string(object.getPrivileges()[3]),
                                 std::to_string(false),
                                 std::to_string(false)});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

bool SysCatalog::verifyDBObjectOwnership(const UserMetadata& user,
                                         DBObject object,
                                         const Catalog_Namespace::Catalog& catalog) {
  if (object.getType() == TableDBObjectType) {
    Role* rl = mapd_sys_cat->getMetadataForUserRole(user.userId);
    if (rl) {
      populateDBObjectKey(object, catalog);
      if (rl->findDbObject(object.getObjectKey()) &&
          (rl->findDbObject(object.getObjectKey())->getOwningUserId() == user.userId)) {
        return true;
      }
    }
  }
  return false;
}

void SysCatalog::getDBObjectPrivileges(const std::string& roleName,
                                       DBObject& object,
                                       const Catalog_Namespace::Catalog& catalog) {
  if (!roleName.compare(MAPD_ROOT_USER)) {
    throw runtime_error("Request to show privileges from " + roleName +
                        " failed because mapd root user has all privileges by default.");
  }
  Role* rl = mapd_sys_cat->getMetadataForRole(roleName);
  if (!rl) {
    throw runtime_error("Request to show privileges for " + roleName +
                        " failed because role or user with this name does not exist.");
  }
  populateDBObjectKey(object, catalog);
  rl->getPrivileges(object);
}

void SysCatalog::createRole(const std::string& roleName, const bool& userPrivateRole) {
  if (!userPrivateRole) {
    UserMetadata user;
    if (getMetadataForUser(roleName, user)) {
      throw runtime_error("Role name " + roleName +
                          " is same as one of user names. Role and user names should be unique.");
    }
  }
  if (getMetadataForRole(roleName) != nullptr) {
    throw std::runtime_error("CREATE ROLE " + roleName + " failed because role with this name already exists.");
  }
  Role* rl = getMetadataForRole(roleName);
  CHECK(!rl);  // it has been checked already in the calling proc that this role doesn't exist, fail otherwize
  rl = new GroupRole(to_upper(roleName), userPrivateRole);
  roleMap_[to_upper(roleName)] = rl;

  /* grant none privileges to this role and add it to sqlite DB */
  DBObject dbObject(get_currentDB().dbName, DatabaseDBObjectType);
  populateDBObjectKey(dbObject, static_cast<Catalog_Namespace::Catalog&>(*this));
  rl->grantPrivileges(dbObject);
  std::vector<std::string> objectKey = convertObjectKeyToString(dbObject);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_object_privileges(roleName, roleType, objectName, objectType, dbObjectType, dbId, tableId, "
        "columnId, "
        "privSelect, privInsert, privCreate, privTruncate, privDelete, privUpdate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?)",
        std::vector<std::string>{roleName,
                                 std::to_string(userPrivateRole),
                                 dbObject.getName(),
                                 std::to_string(dbObject.getType()),
                                 objectKey[0],
                                 objectKey[1],
                                 objectKey[2],
                                 objectKey[3],
                                 std::to_string(dbObject.getPrivileges()[0]),
                                 std::to_string(dbObject.getPrivileges()[1]),
                                 std::to_string(dbObject.getPrivileges()[2]),
                                 std::to_string(dbObject.getPrivileges()[3]),
                                 std::to_string(false),
                                 std::to_string(false)});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void SysCatalog::dropRole(const std::string& roleName) {
  Role* rl = getMetadataForRole(roleName);
  CHECK(rl);  // it has been checked already in the calling proc that this role exists, faiul otherwise
  delete rl;
  roleMap_.erase(to_upper(roleName));

  /* remove role and related data from sqlite DB */
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_param("DELETE FROM mapd_roles WHERE roleName = ?", roleName);
    sqliteConnector_.query_with_text_param("DELETE FROM mapd_object_privileges WHERE roleName = ?", roleName);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

// GRANT ROLE payroll_dept_role TO joe;
void SysCatalog::grantRole(const std::string& roleName, const std::string& userName) {
  Role* user_rl = nullptr;
  Role* rl = getMetadataForRole(roleName);
  if (!rl) {
    throw runtime_error("Request to grant role " + roleName + " failed because role with this name does not exist.");
  }
  UserMetadata user;
  if (!getMetadataForUser(userName, user)) {
    throw runtime_error("Request to grant role to user " + userName +
                        " failed because user with this name does not exist.");
  }
  user_rl = getMetadataForUserRole(user.userId);
  if (!user_rl) {
    // this user has never been granted roles before, so create new object
    user_rl = new UserRole(rl, user.userId, userName);
    std::lock_guard<std::mutex> lock(cat_mutex_);
    userRoleMap_[user.userId] = user_rl;
  }
  if (!user_rl->hasRole(rl)) {
    user_rl->grantRole(rl);
    /* apply grant role statement to sqlite DB */
    sqliteConnector_.query("BEGIN TRANSACTION");
    try {
      sqliteConnector_.query_with_text_params("INSERT INTO mapd_roles(roleName, userName) VALUES (?, ?)",
                                              std::vector<std::string>{roleName, userName});
    } catch (std::exception& e) {
      sqliteConnector_.query("ROLLBACK TRANSACTION");
      throw;
    }
    sqliteConnector_.query("END TRANSACTION");
  }
}

// REVOKE ROLE payroll_dept_role FROM joe;
void SysCatalog::revokeRole(const std::string& roleName, const std::string& userName) {
  Role* user_rl = nullptr;
  Role* rl = getMetadataForRole(roleName);
  if (!rl) {
    throw runtime_error("Request to revoke role " + roleName + " failed because role with this name does not exist.");
  }
  UserMetadata user;
  if (!getMetadataForUser(userName, user)) {
    throw runtime_error("Request to revoke role from user " + userName +
                        " failed because user with this name does not exist.");
  }
  user_rl = getMetadataForUserRole(user.userId);
  if (!user_rl || (user_rl && !user_rl->hasRole(rl))) {
    throw runtime_error("Request to revoke role " + roleName + " from user " + userName +
                        " failed because this role has not been granted to the user.");
  }
  user_rl->revokeRole(rl);
  if (user_rl->getMembershipSize() == 0) {
    delete user_rl;
    std::lock_guard<std::mutex> lock(cat_mutex_);
    userRoleMap_.erase(user.userId);
  }

  /* apply revoke role statement to sqlite DB */
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params("DELETE FROM mapd_roles WHERE roleName = ? AND userName = ?",
                                            std::vector<std::string>{roleName, userName});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

/*
 * delete object of UserRole class (delete all GroupRoles for this user,
 * i.e. delete pointers from all GroupRole objects referencing to this UserRole object)
 * called as a result of executing "DROP USER" command
 */
void SysCatalog::dropUserRole(const std::string& userName) {
  /* this proc is not being directly called from parser, so it should have been checked already
   * before calling this proc that the userName is valid (see call to proc "getMetadataForUser")
   */
  UserMetadata user;
  if (!getMetadataForUser(userName, user)) {
    throw runtime_error("Request to revoke roles from user " + userName +
                        " failed because user with this name does not exist.");
  }
  Role* user_rl = getMetadataForUserRole(user.userId);
  if (user_rl) {
    delete user_rl;
    std::lock_guard<std::mutex> lock(cat_mutex_);
    userRoleMap_.erase(user.userId);
  }
  // do nothing if userName was not found in userRoleMap_
}

bool SysCatalog::checkPrivileges(const UserMetadata& user, std::vector<DBObject>& privObjects) {
  if (user.isSuper) {
    return true;
  }
  Role* user_rl = mapd_sys_cat->getMetadataForUserRole(user.userId);
  if (!user_rl) {
    if (!mapd_sys_cat->getMetadataForRole(MAPD_DEFAULT_ROOT_USER_ROLE)) {
      createDefaultMapdRoles();
    }
    if (user.isSuper) {
      grantRole(MAPD_DEFAULT_ROOT_USER_ROLE, user.userName);
    } else {
      grantRole(MAPD_DEFAULT_USER_ROLE, user.userName);
    }
    user_rl = mapd_sys_cat->getMetadataForUserRole(user.userId);
  }
  for (std::vector<DBObject>::iterator objectIt = privObjects.begin(); objectIt != privObjects.end(); ++objectIt) {
    if (!user_rl->checkPrivileges(*objectIt)) {
      return false;
    }
  }
  return true;
}

bool SysCatalog::checkPrivileges(const std::string& userName, std::vector<DBObject>& privObjects) {
  UserMetadata user;
  if (!mapd_sys_cat->getMetadataForUser(userName, user)) {
    throw runtime_error("Request to check privileges for user " + userName +
                        " failed because user with this name does not exist.");
  }
  return (checkPrivileges(user, privObjects));
}

Role* SysCatalog::getMetadataForRole(const std::string& roleName) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto roleIt = roleMap_.find(to_upper(roleName));
  if (roleIt == roleMap_.end()) {  // check to make sure role exists
    return nullptr;
  }
  return roleIt->second;  // returns pointer to role
}

Role* SysCatalog::getMetadataForUserRole(int32_t userId) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto userRoleIt = userRoleMap_.find(userId);
  if (userRoleIt == userRoleMap_.end()) {  // check to make sure role exists
    return nullptr;
  }
  return userRoleIt->second;  // returns pointer to role
}

Catalog::Catalog(const string& basePath,
                 const string& dbname,
                 std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                 LdapMetadata ldapMetadata,
                 bool is_initdb,
                 std::shared_ptr<Calcite> calcite,
                 const bool access_priv_check)
    : basePath_(basePath),
      sqliteConnector_(dbname, basePath + "/mapd_catalogs/"),
      dataMgr_(dataMgr),
      calciteMgr_(calcite),
      nextTempTableId_(MAPD_TEMP_TABLE_START_ID),
      nextTempDictId_(MAPD_TEMP_DICT_START_ID),
      access_priv_check_(access_priv_check) {
  ldap_server_.reset(new LdapServer(ldapMetadata));
  if (!is_initdb)
    buildMaps();
}

Catalog::Catalog(const string& basePath,
                 const DBMetadata& curDB,
                 std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                 LdapMetadata ldapMetadata,
                 std::shared_ptr<Calcite> calcite,
                 const bool access_priv_check)
    : basePath_(basePath),
      sqliteConnector_(curDB.dbName, basePath + "/mapd_catalogs/"),
      currentDB_(curDB),
      dataMgr_(dataMgr),
      calciteMgr_(calcite),
      nextTempTableId_(MAPD_TEMP_TABLE_START_ID),
      nextTempDictId_(MAPD_TEMP_DICT_START_ID),
      access_priv_check_(access_priv_check) {
  ldap_server_.reset(new LdapServer(ldapMetadata));
  buildMaps();
}

Catalog::Catalog(const string& basePath,
                 const DBMetadata& curDB,
                 std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                 const std::vector<LeafHostInfo>& string_dict_hosts,
                 std::shared_ptr<Calcite> calcite,
                 const bool access_priv_check)
    : basePath_(basePath),
      sqliteConnector_(curDB.dbName, basePath + "/mapd_catalogs/"),
      currentDB_(curDB),
      dataMgr_(dataMgr),
      string_dict_hosts_(string_dict_hosts),
      calciteMgr_(calcite),
      nextTempTableId_(MAPD_TEMP_TABLE_START_ID),
      nextTempDictId_(MAPD_TEMP_DICT_START_ID),
      access_priv_check_(access_priv_check) {
  ldap_server_.reset(new LdapServer());
  buildMaps();
}

Catalog::~Catalog() {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  // must clean up heap-allocated TableDescriptor and ColumnDescriptor structs
  for (TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.begin(); tableDescIt != tableDescriptorMap_.end();
       ++tableDescIt) {
    if (tableDescIt->second->fragmenter != nullptr)
      delete tableDescIt->second->fragmenter;
    delete tableDescIt->second;
  }

  // TableDescriptorMapById points to the same descriptors.  No need to delete

  for (ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin();
       columnDescIt != columnDescriptorMap_.end();
       ++columnDescIt)
    delete columnDescIt->second;

  // ColumnDescriptorMapById points to the same descriptors.  No need to delete
}

void Catalog::updateTableDescriptorSchema() {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_tables)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("max_chunk_size")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD max_chunk_size BIGINT DEFAULT " +
                         std::to_string(DEFAULT_MAX_CHUNK_SIZE));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("shard_column_id")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD shard_column_id BIGINT DEFAULT " + std::to_string(0));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("shard")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD shard BIGINT DEFAULT " + std::to_string(-1));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("num_shards")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD num_shards BIGINT DEFAULT " + std::to_string(0));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("key_metainfo")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD key_metainfo TEXT DEFAULT '[]'");
      sqliteConnector_.query(queryString);
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateFrontendViewSchema() {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_frontend_views)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("image_hash")) == cols.end()) {
      sqliteConnector_.query("ALTER TABLE mapd_frontend_views ADD image_hash text");
    }
    if (std::find(cols.begin(), cols.end(), std::string("update_time")) == cols.end()) {
      sqliteConnector_.query("ALTER TABLE mapd_frontend_views ADD update_time timestamp");
    }
    if (std::find(cols.begin(), cols.end(), std::string("view_metadata")) == cols.end()) {
      sqliteConnector_.query("ALTER TABLE mapd_frontend_views ADD view_metadata text");
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateLinkSchema() {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "CREATE TABLE IF NOT EXISTS mapd_links (linkid integer primary key, userid integer references mapd_users, "
        "link text unique, view_state text, update_time timestamp, view_metadata text)");
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_links)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("view_metadata")) == cols.end()) {
      sqliteConnector_.query("ALTER TABLE mapd_links ADD view_metadata text");
    }
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateFrontendViewAndLinkUsers() {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("UPDATE mapd_links SET userid = 0 WHERE userid IS NULL");
    sqliteConnector_.query("UPDATE mapd_frontend_views SET userid = 0 WHERE userid IS NULL");
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

// introduce DB version into the dictionary tables
// if the DB does not have a version rename all dictionary tables

void Catalog::updateDictionaryNames() {
  if (currentDB_.dbName.length() == 0) {
    // updateDictionaryNames dbName length is zero nothing to do here
    return;
  }
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_dictionaries)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("version_num")) == cols.end()) {
      // No version number
      // need to rename dictionaries
      string dictQuery("SELECT dictid, name from mapd_dictionaries");
      sqliteConnector_.query(dictQuery);
      size_t numRows = sqliteConnector_.getNumRows();
      for (size_t r = 0; r < numRows; ++r) {
        int dictId = sqliteConnector_.getData<int>(r, 0);
        std::string dictName = sqliteConnector_.getData<string>(r, 1);

        std::string oldName = basePath_ + "/mapd_data/" + currentDB_.dbName + "_" + dictName;
        std::string newName =
            basePath_ + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) + "_DICT_" + std::to_string(dictId);

        int result = rename(oldName.c_str(), newName.c_str());

        if (result == 0)
          LOG(INFO) << "Dictionary upgrade: successfully renamed " << oldName << " to " << newName;
        else {
          LOG(ERROR) << "Failed to rename old dictionary directory " << oldName << " to " << newName + " dbname '"
                     << currentDB_.dbName << "' error code " << std::to_string(result);
        }
      }
      // need to add new version info
      string queryString("ALTER TABLE mapd_dictionaries ADD version_num BIGINT DEFAULT " +
                         std::to_string(DEFAULT_INITIAL_VERSION));
      sqliteConnector_.query(queryString);
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateLogicalToPhysicalTableLinkSchema() {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "CREATE TABLE IF NOT EXISTS mapd_logical_to_physical("
        "logical_table_id integer, physical_table_id integer)");
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateLogicalToPhysicalTableMap(const int32_t logical_tb_id) {
  /* this proc inserts/updates all pairs of (logical_tb_id, physical_tb_id) in
   * sqlite mapd_logical_to_physical table for given logical_tb_id as needed
   */

  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    const auto physicalTableIt = logicalToPhysicalTableMapById_.find(logical_tb_id);
    if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
      const auto physicalTables = physicalTableIt->second;
      CHECK(!physicalTables.empty());
      for (size_t i = 0; i < physicalTables.size(); i++) {
        int32_t physical_tb_id = physicalTables[i];
        sqliteConnector_.query_with_text_params(
            "INSERT OR REPLACE INTO mapd_logical_to_physical (logical_table_id, physical_table_id) VALUES (?1, ?2)",
            std::vector<std::string>{std::to_string(logical_tb_id), std::to_string(physical_tb_id)});
      }
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateDictionarySchema() {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_dictionaries)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("refcount")) == cols.end()) {
      sqliteConnector_.query("ALTER TABLE mapd_dictionaries ADD refcount DEFAULT 1");
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::CheckAndExecuteMigrations() {
  updateTableDescriptorSchema();
  updateFrontendViewAndLinkUsers();
  updateFrontendViewSchema();
  updateLinkSchema();
  updateDictionaryNames();
  updateLogicalToPhysicalTableLinkSchema();
  updateDictionarySchema();
}

void Catalog::buildRoleMap() {
  auto& sys_cat = static_cast<Catalog_Namespace::SysCatalog&>(*this);
  string roleQuery(
      "SELECT roleName, roleType, objectName, objectType, dbObjectType, dbId, tableId, columnId, privSelect, "
      "privInsert, privCreate, privTruncate, "
      "privDelete, privUpdate from mapd_object_privileges");
  sqliteConnector_.query(roleQuery);
  size_t numRows = sqliteConnector_.getNumRows();
  std::vector<std::string> objectKeyStr(4);
  DBObjectKey objectKey(4);
  std::vector<bool> privs(4);
  bool userPrivateRole(false);
  for (size_t r = 0; r < numRows; ++r) {
    std::string roleName = sqliteConnector_.getData<string>(r, 0);
    userPrivateRole = sqliteConnector_.getData<bool>(r, 1);
    std::string objectName = sqliteConnector_.getData<string>(r, 2);
    DBObjectType objectType = static_cast<DBObjectType>(sqliteConnector_.getData<int>(r, 3));
    objectKeyStr[0] = sqliteConnector_.getData<string>(r, 4);
    objectKeyStr[1] = sqliteConnector_.getData<string>(r, 5);
    objectKeyStr[2] = sqliteConnector_.getData<string>(r, 6);
    objectKeyStr[3] = sqliteConnector_.getData<string>(r, 7);
    objectKey = sys_cat.convertObjectKeyFromString(objectKeyStr, objectType);
    privs[0] = sqliteConnector_.getData<bool>(r, 8);
    privs[1] = sqliteConnector_.getData<bool>(r, 9);
    privs[2] = sqliteConnector_.getData<bool>(r, 10);
    privs[3] = sqliteConnector_.getData<bool>(r, 11);
    {
      DBObject dbObject(objectName, objectType);
      dbObject.setObjectKey(objectKey);
      dbObject.setPrivileges(privs);
      Role* rl = sys_cat.getMetadataForRole(roleName);
      if (!rl) {
        rl = new GroupRole(to_upper(roleName), userPrivateRole);
        roleMap_[to_upper(roleName)] = rl;
      }
      rl->grantPrivileges(dbObject);
    }
  }
}

void Catalog::buildUserRoleMap() {
  auto& sys_cat = static_cast<Catalog_Namespace::SysCatalog&>(*this);
  std::vector<std::pair<std::string, std::string>> userRoleVec;
  string userRoleQuery("SELECT roleName, userName from mapd_roles");
  sqliteConnector_.query(userRoleQuery);
  size_t numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    std::string roleName = sqliteConnector_.getData<string>(r, 0);
    std::string userName = sqliteConnector_.getData<string>(r, 1);
    Role* rl = sys_cat.getMetadataForRole(roleName);
    if (!rl) {
      throw runtime_error("Data inconsistency when building role map. Role " + roleName + " not found in the map.");
    }
    std::pair<std::string, std::string> roleVecElem(roleName, userName);
    userRoleVec.push_back(roleVecElem);
  }

  for (size_t i = 0; i < userRoleVec.size(); i++) {
    std::string roleName = userRoleVec[i].first;
    std::string userName = userRoleVec[i].second;
    UserMetadata user;
    if (!sys_cat.getMetadataForUser(userName, user)) {
      throw runtime_error("Data inconsistency when building role map. User " + userName + " not found in the map.");
    }
    Role* rl = sys_cat.getMetadataForRole(roleName);
    Role* user_rl = sys_cat.getMetadataForUserRole(user.userId);
    if (!user_rl) {
      // roles for this user have not been recovered from sqlite DB before, so create new object
      user_rl = new UserRole(rl, user.userId, userName);
      std::lock_guard<std::mutex> lock(cat_mutex_);
      userRoleMap_[user.userId] = user_rl;
    }
    user_rl->grantRole(rl);
  }
}

void Catalog::buildMaps() {
  CheckAndExecuteMigrations();

  string dictQuery("SELECT dictid, name, nbits, is_shared, refcount from mapd_dictionaries");
  sqliteConnector_.query(dictQuery);
  size_t numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    int dictId = sqliteConnector_.getData<int>(r, 0);
    std::string dictName = sqliteConnector_.getData<string>(r, 1);
    int dictNBits = sqliteConnector_.getData<int>(r, 2);
    bool is_shared = sqliteConnector_.getData<bool>(r, 3);
    int refcount = sqliteConnector_.getData<int>(r, 4);
    std::string fname =
        basePath_ + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) + "_DICT_" + std::to_string(dictId);
    DictDescriptor* dd = new DictDescriptor(dictId, dictName, dictNBits, is_shared, refcount, fname, false);
    dictDescriptorMapById_[dd->dictId].reset(dd);
  }

  string tableQuery(
      "SELECT tableid, name, ncolumns, isview, fragments, frag_type, max_frag_rows, max_chunk_size, frag_page_size, "
      "max_rows, partitions, shard_column_id, shard, num_shards, key_metainfo from mapd_tables");
  sqliteConnector_.query(tableQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    TableDescriptor* td = new TableDescriptor();
    td->tableId = sqliteConnector_.getData<int>(r, 0);
    td->tableName = sqliteConnector_.getData<string>(r, 1);
    td->nColumns = sqliteConnector_.getData<int>(r, 2);
    td->isView = sqliteConnector_.getData<bool>(r, 3);
    td->fragments = sqliteConnector_.getData<string>(r, 4);
    td->fragType = (Fragmenter_Namespace::FragmenterType)sqliteConnector_.getData<int>(r, 5);
    td->maxFragRows = sqliteConnector_.getData<int>(r, 6);
    td->maxChunkSize = sqliteConnector_.getData<int>(r, 7);
    td->fragPageSize = sqliteConnector_.getData<int>(r, 8);
    td->maxRows = sqliteConnector_.getData<int64_t>(r, 9);
    td->partitions = sqliteConnector_.getData<string>(r, 10);
    td->shardedColumnId = sqliteConnector_.getData<int>(r, 11);
    td->shard = sqliteConnector_.getData<int>(r, 12);
    td->nShards = sqliteConnector_.getData<int>(r, 13);
    td->keyMetainfo = sqliteConnector_.getData<string>(r, 14);
    if (!td->isView) {
      td->fragmenter = nullptr;
    }
    tableDescriptorMap_[to_upper(td->tableName)] = td;
    tableDescriptorMapById_[td->tableId] = td;
  }
  string columnQuery(
      "SELECT tableid, columnid, name, coltype, colsubtype, coldim, colscale, is_notnull, compression, comp_param, "
      "size, chunks, is_systemcol, is_virtualcol, virtual_expr from mapd_columns");
  sqliteConnector_.query(columnQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    ColumnDescriptor* cd = new ColumnDescriptor();
    cd->tableId = sqliteConnector_.getData<int>(r, 0);
    cd->columnId = sqliteConnector_.getData<int>(r, 1);
    cd->columnName = sqliteConnector_.getData<string>(r, 2);
    cd->columnType.set_type((SQLTypes)sqliteConnector_.getData<int>(r, 3));
    cd->columnType.set_subtype((SQLTypes)sqliteConnector_.getData<int>(r, 4));
    cd->columnType.set_dimension(sqliteConnector_.getData<int>(r, 5));
    cd->columnType.set_scale(sqliteConnector_.getData<int>(r, 6));
    cd->columnType.set_notnull(sqliteConnector_.getData<bool>(r, 7));
    cd->columnType.set_compression((EncodingType)sqliteConnector_.getData<int>(r, 8));
    cd->columnType.set_comp_param(sqliteConnector_.getData<int>(r, 9));
    cd->columnType.set_size(sqliteConnector_.getData<int>(r, 10));
    cd->chunks = sqliteConnector_.getData<string>(r, 11);
    cd->isSystemCol = sqliteConnector_.getData<bool>(r, 12);
    cd->isVirtualCol = sqliteConnector_.getData<bool>(r, 13);
    cd->virtualExpr = sqliteConnector_.getData<string>(r, 14);
    ColumnKey columnKey(cd->tableId, to_upper(cd->columnName));
    columnDescriptorMap_[columnKey] = cd;
    ColumnIdKey columnIdKey(cd->tableId, cd->columnId);
    columnDescriptorMapById_[columnIdKey] = cd;
  }
  string viewQuery("SELECT tableid, sql FROM mapd_views");
  sqliteConnector_.query(viewQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    int32_t tableId = sqliteConnector_.getData<int>(r, 0);
    TableDescriptor* td = tableDescriptorMapById_[tableId];
    td->viewSQL = sqliteConnector_.getData<string>(r, 1);
    td->fragmenter = nullptr;
  }

  string frontendViewQuery(
      "SELECT viewid, view_state, name, image_hash, strftime('%Y-%m-%dT%H:%M:%SZ', update_time), userid, view_metadata "
      "FROM mapd_frontend_views");
  sqliteConnector_.query(frontendViewQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    FrontendViewDescriptor* vd = new FrontendViewDescriptor();
    vd->viewId = sqliteConnector_.getData<int>(r, 0);
    vd->viewState = sqliteConnector_.getData<string>(r, 1);
    vd->viewName = sqliteConnector_.getData<string>(r, 2);
    vd->imageHash = sqliteConnector_.getData<string>(r, 3);
    vd->updateTime = sqliteConnector_.getData<string>(r, 4);
    vd->userId = sqliteConnector_.getData<int>(r, 5);
    vd->viewMetadata = sqliteConnector_.getData<string>(r, 6);
    frontendViewDescriptorMap_[std::to_string(vd->userId) + vd->viewName] = vd;
  }

  string linkQuery(
      "SELECT linkid, userid, link, view_state, strftime('%Y-%m-%dT%H:%M:%SZ', update_time), view_metadata "
      "FROM mapd_links");
  sqliteConnector_.query(linkQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    LinkDescriptor* ld = new LinkDescriptor();
    ld->linkId = sqliteConnector_.getData<int>(r, 0);
    ld->userId = sqliteConnector_.getData<int>(r, 1);
    ld->link = sqliteConnector_.getData<string>(r, 2);
    ld->viewState = sqliteConnector_.getData<string>(r, 3);
    ld->updateTime = sqliteConnector_.getData<string>(r, 4);
    ld->viewMetadata = sqliteConnector_.getData<string>(r, 5);
    linkDescriptorMap_[std::to_string(currentDB_.dbId) + ld->link] = ld;
    linkDescriptorMapById_[ld->linkId] = ld;
  }

  /* rebuild map linking logical tables to corresponding physical ones */
  string logicalToPhysicalTableMapQuery(
      "SELECT logical_table_id, physical_table_id "
      "FROM mapd_logical_to_physical");
  sqliteConnector_.query(logicalToPhysicalTableMapQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    int32_t logical_tb_id = sqliteConnector_.getData<int>(r, 0);
    int32_t physical_tb_id = sqliteConnector_.getData<int>(r, 1);
    const auto physicalTableIt = logicalToPhysicalTableMapById_.find(logical_tb_id);
    if (physicalTableIt == logicalToPhysicalTableMapById_.end()) {
      /* add new entity to the map logicalToPhysicalTableMapById_ */
      std::vector<int32_t> physicalTables;
      physicalTables.push_back(physical_tb_id);
      const auto it_ok = logicalToPhysicalTableMapById_.emplace(logical_tb_id, physicalTables);
      CHECK(it_ok.second);
    } else {
      /* update map logicalToPhysicalTableMapById_ */
      physicalTableIt->second.push_back(physical_tb_id);
    }
  }

  if (access_priv_check_) {
    /* rebuild role map (includes object privileges assignment) */
    buildRoleMap();
    /* rebuild map linking user IDs to granted them roles */
    buildUserRoleMap();
  }
}

void Catalog::addTableToMap(TableDescriptor& td,
                            const list<ColumnDescriptor>& columns,
                            const list<DictDescriptor>& dicts) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  TableDescriptor* new_td = new TableDescriptor();
  *new_td = td;
  tableDescriptorMap_[to_upper(td.tableName)] = new_td;
  tableDescriptorMapById_[td.tableId] = new_td;
  for (auto cd : columns) {
    ColumnDescriptor* new_cd = new ColumnDescriptor();
    *new_cd = cd;
    ColumnKey columnKey(new_cd->tableId, to_upper(new_cd->columnName));
    columnDescriptorMap_[columnKey] = new_cd;
    ColumnIdKey columnIdKey(new_cd->tableId, new_cd->columnId);
    columnDescriptorMapById_[columnIdKey] = new_cd;
  }
  std::unique_ptr<StringDictionaryClient> client;
  if (!string_dict_hosts_.empty()) {
    client.reset(new StringDictionaryClient(string_dict_hosts_.front(), -1, true));
  }
  for (auto dd : dicts) {
    if (!dd.dictId) {
      // Dummy entry created for a shard of a logical table, nothing to do.
      continue;
    }
    if (client) {
      client->create(dd.dictId, currentDB_.dbId, dd.dictIsTemp);
    }
    DictDescriptor* new_dd = new DictDescriptor(dd);
    dictDescriptorMapById_[dd.dictId].reset(new_dd);
    if (!dd.dictIsTemp)
      boost::filesystem::create_directory(new_dd->dictFolderPath);
  }
}

void Catalog::removeTableFromMap(const string& tableName, int tableId) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  TableDescriptorMapById::iterator tableDescIt = tableDescriptorMapById_.find(tableId);
  if (tableDescIt == tableDescriptorMapById_.end())
    throw runtime_error("Table " + tableName + " does not exist.");
  TableDescriptor* td = tableDescIt->second;
  int ncolumns = td->nColumns;
  tableDescriptorMapById_.erase(tableDescIt);
  tableDescriptorMap_.erase(to_upper(tableName));
  if (td->fragmenter != nullptr)
    delete td->fragmenter;
  bool isTemp = td->persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL;
  delete td;

  std::unique_ptr<StringDictionaryClient> client;
  if (g_aggregator) {
    CHECK(!string_dict_hosts_.empty());
    client.reset(new StringDictionaryClient(string_dict_hosts_.front(), -1, true));
  }

  // delete all column descriptors for the table
  for (int i = 1; i <= ncolumns; i++) {
    ColumnIdKey cidKey(tableId, i);
    ColumnDescriptorMapById::iterator colDescIt = columnDescriptorMapById_.find(cidKey);
    ColumnDescriptor* cd = colDescIt->second;
    columnDescriptorMapById_.erase(colDescIt);
    ColumnKey cnameKey(tableId, to_upper(cd->columnName));
    columnDescriptorMap_.erase(cnameKey);
    const int dictId = cd->columnType.get_comp_param();
    // Dummy dictionaries created for a shard of a logical table have the id set to zero.
    if (cd->columnType.get_compression() == kENCODING_DICT && dictId) {
      const auto dictIt = dictDescriptorMapById_.find(dictId);
      CHECK(dictIt != dictDescriptorMapById_.end());
      const auto& dd = dictIt->second;
      CHECK_GE(dd->refcount, 1);
      --dd->refcount;
      if (!dd->refcount) {
        if (!isTemp)
          boost::filesystem::remove_all(dd->dictFolderPath);
        if (client) {
          client->drop(dd->dictId, currentDB_.dbId);
        }
        dictDescriptorMapById_.erase(dictIt);
      }
    }
    delete cd;
  }
}

void Catalog::addFrontendViewToMap(FrontendViewDescriptor& vd) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  FrontendViewDescriptor* new_vd = new FrontendViewDescriptor();
  *new_vd = vd;
  frontendViewDescriptorMap_[std::to_string(vd.userId) + vd.viewName] = new_vd;
}

void Catalog::addLinkToMap(LinkDescriptor& ld) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  LinkDescriptor* new_ld = new LinkDescriptor();
  *new_ld = ld;
  linkDescriptorMap_[std::to_string(currentDB_.dbId) + ld.link] = new_ld;
  linkDescriptorMapById_[ld.linkId] = new_ld;
}

void Catalog::instantiateFragmenter(TableDescriptor* td) const {
  auto time_ms = measure<>::execution([&]() {
    // instanciate table fragmenter upon first use
    // assume only insert order fragmenter is supported
    assert(td->fragType == Fragmenter_Namespace::FragmenterType::INSERT_ORDER);
    vector<Chunk> chunkVec;
    list<const ColumnDescriptor*> columnDescs;
    getAllColumnMetadataForTable(td, columnDescs, true, false);
    Chunk::translateColumnDescriptorsToChunkVec(columnDescs, chunkVec);
    ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId};
    td->fragmenter = new InsertOrderFragmenter(chunkKeyPrefix,
                                               chunkVec,
                                               dataMgr_.get(),
                                               td->tableId,
                                               td->shard,
                                               td->maxFragRows,
                                               td->maxChunkSize,
                                               td->fragPageSize,
                                               td->maxRows,
                                               td->persistenceLevel);
  });
  LOG(INFO) << "Instantiating Fragmenter for table " << td->tableName << " took " << time_ms << "ms";
}

const TableDescriptor* Catalog::getMetadataForTable(const string& tableName) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto tableDescIt = tableDescriptorMap_.find(to_upper(tableName));
  if (tableDescIt == tableDescriptorMap_.end()) {  // check to make sure table exists
    return nullptr;
  }
  TableDescriptor* td = tableDescIt->second;
  if (td->fragmenter == nullptr && !td->isView)
    instantiateFragmenter(td);
  return td;  // returns pointer to table descriptor
}

const TableDescriptor* Catalog::getMetadataForTable(int tableId) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto tableDescIt = tableDescriptorMapById_.find(tableId);
  if (tableDescIt == tableDescriptorMapById_.end()) {  // check to make sure table exists
    return nullptr;
  }
  TableDescriptor* td = tableDescIt->second;
  if (td->fragmenter == nullptr && !td->isView)
    instantiateFragmenter(td);
  return td;  // returns pointer to table descriptor
}

const DictDescriptor* Catalog::getMetadataForDict(int dictId, bool loadDict) const {
  auto dictDescIt = dictDescriptorMapById_.find(dictId);
  if (dictDescIt == dictDescriptorMapById_.end()) {  // check to make sure dictionary exists
    return nullptr;
  }
  const auto& dd = dictDescIt->second;
  if (loadDict) {
    std::lock_guard<std::mutex> lock(cat_mutex_);
    if (!dd->stringDict) {
      auto time_ms = measure<>::execution([&]() {
        if (string_dict_hosts_.empty()) {
          if (dd->dictIsTemp)
            dd->stringDict = std::make_shared<StringDictionary>(dd->dictFolderPath, true, true);
          else
            dd->stringDict = std::make_shared<StringDictionary>(dd->dictFolderPath, false, true);
        } else {
          dd->stringDict = std::make_shared<StringDictionary>(string_dict_hosts_.front(), dd->dictId);
        }
      });
      LOG(INFO) << "Time to load Dictionary " << dictId << " was " << time_ms << "ms";
    }
  }
  return dd.get();
}

const std::vector<LeafHostInfo>& Catalog::getStringDictionaryHosts() const {
  return string_dict_hosts_;
}

const ColumnDescriptor* Catalog::getMetadataForColumn(int tableId, const string& columnName) const {
  ColumnKey columnKey(tableId, to_upper(columnName));
  auto colDescIt = columnDescriptorMap_.find(columnKey);
  if (colDescIt == columnDescriptorMap_.end()) {  // need to check to make sure column exists for table
    return nullptr;
  }
  return colDescIt->second;
}

const ColumnDescriptor* Catalog::getMetadataForColumn(int tableId, int columnId) const {
  ColumnIdKey columnIdKey(tableId, columnId);
  auto colDescIt = columnDescriptorMapById_.find(columnIdKey);
  if (colDescIt == columnDescriptorMapById_.end()) {  // need to check to make sure column exists for table
    return nullptr;
  }
  return colDescIt->second;
}

void Catalog::deleteMetadataForFrontendView(const std::string& userId, const std::string& viewName) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto viewDescIt = frontendViewDescriptorMap_.find(userId + viewName);
  if (viewDescIt == frontendViewDescriptorMap_.end()) {  // check to make sure view exists
    LOG(ERROR) << "deleteting view for user " << userId << " view " << viewName << " does not exist in map";
    return;
  }
  // found view in Map now remove it
  frontendViewDescriptorMap_.erase(viewDescIt);
  // remove from DB
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params("DELETE FROM mapd_frontend_views WHERE name = ? and userid = ?",
                                            std::vector<std::string>{viewName, userId});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

const FrontendViewDescriptor* Catalog::getMetadataForFrontendView(const string& userId, const string& viewName) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto viewDescIt = frontendViewDescriptorMap_.find(userId + viewName);
  if (viewDescIt == frontendViewDescriptorMap_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return viewDescIt->second;  // returns pointer to view descriptor
}

const LinkDescriptor* Catalog::getMetadataForLink(const string& link) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto linkDescIt = linkDescriptorMap_.find(link);
  if (linkDescIt == linkDescriptorMap_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return linkDescIt->second;  // returns pointer to view descriptor
}

const LinkDescriptor* Catalog::getMetadataForLink(int linkId) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto linkDescIt = linkDescriptorMapById_.find(linkId);
  if (linkDescIt == linkDescriptorMapById_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return linkDescIt->second;
}

void Catalog::getAllColumnMetadataForTable(const TableDescriptor* td,
                                           list<const ColumnDescriptor*>& columnDescriptors,
                                           const bool fetchSystemColumns,
                                           const bool fetchVirtualColumns) const {
  for (int i = 1; i <= td->nColumns; i++) {
    const ColumnDescriptor* cd = getMetadataForColumn(td->tableId, i);
    assert(cd != nullptr);
    if (!fetchSystemColumns && cd->isSystemCol)
      continue;
    if (!fetchVirtualColumns && cd->isVirtualCol)
      continue;
    columnDescriptors.push_back(cd);
  }
}

list<const ColumnDescriptor*> Catalog::getAllColumnMetadataForTable(const int tableId,
                                                                    const bool fetchSystemColumns,
                                                                    const bool fetchVirtualColumns) const {
  list<const ColumnDescriptor*> columnDescriptors;
  const TableDescriptor* td = getMetadataForTable(tableId);
  getAllColumnMetadataForTable(td, columnDescriptors, fetchSystemColumns, fetchVirtualColumns);
  return columnDescriptors;
}

list<const TableDescriptor*> Catalog::getAllTableMetadata() const {
  list<const TableDescriptor*> table_list;
  for (auto p : tableDescriptorMapById_)
    table_list.push_back(p.second);
  return table_list;
}

list<const FrontendViewDescriptor*> Catalog::getAllFrontendViewMetadata() const {
  list<const FrontendViewDescriptor*> view_list;
  for (auto p : frontendViewDescriptorMap_)
    view_list.push_back(p.second);
  return view_list;
}

void Catalog::createTable(TableDescriptor& td,
                          const list<ColumnDescriptor>& cols,
                          const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs,
                          bool isLogicalTable) {
  list<ColumnDescriptor> cds;
  list<DictDescriptor> dds;

  for (auto cd : cols) {
    if (cd.columnName == "rowid") {
      throw std::runtime_error("Cannot create column with name rowid. rowid is a system defined column.");
    }
  }

  list<ColumnDescriptor> columns(cols);
  // add row_id column
  ColumnDescriptor cd;
  cd.columnName = "rowid";
  cd.isSystemCol = true;
  cd.columnType = SQLTypeInfo(kBIGINT, true);
#ifdef MATERIALIZED_ROWID
  cd.isVirtualCol = false;
#else
  cd.isVirtualCol = true;
  cd.virtualExpr = "MAPD_FRAG_ID * MAPD_ROWS_PER_FRAG + MAPD_FRAG_ROW_ID";
#endif
  columns.push_back(cd);

  td.nColumns = columns.size();

  if (td.persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    sqliteConnector_.query("BEGIN TRANSACTION");
    try {
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_tables (name, ncolumns, isview, fragments, frag_type, max_frag_rows, max_chunk_size, "
          "frag_page_size, max_rows, partitions, shard_column_id, shard, num_shards, key_metainfo) VALUES (?, ?, ?, ?, "
          "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",

          std::vector<std::string>{td.tableName,
                                   std::to_string(columns.size()),
                                   std::to_string(td.isView),
                                   "",
                                   std::to_string(td.fragType),
                                   std::to_string(td.maxFragRows),
                                   std::to_string(td.maxChunkSize),
                                   std::to_string(td.fragPageSize),
                                   std::to_string(td.maxRows),
                                   td.partitions,
                                   std::to_string(td.shardedColumnId),
                                   std::to_string(td.shard),
                                   std::to_string(td.nShards),
                                   td.keyMetainfo});

      // now get the auto generated tableid
      sqliteConnector_.query_with_text_param("SELECT tableid FROM mapd_tables WHERE name = ?", td.tableName);
      td.tableId = sqliteConnector_.getData<int>(0, 0);
      int colId = 1;
      for (auto cd : columns) {
        if (cd.columnType.get_compression() == kENCODING_DICT) {
          const bool is_foreign_col = setColumnSharedDictionary(cd, shared_dict_defs);
          if (!is_foreign_col) {
            setColumnDictionary(cd, dds, td, isLogicalTable);
          }
        }
        sqliteConnector_.query_with_text_params(
            "INSERT INTO mapd_columns (tableid, columnid, name, coltype, colsubtype, coldim, colscale, is_notnull, "
            "compression, comp_param, size, chunks, is_systemcol, is_virtualcol, virtual_expr) VALUES (?, ?, ?, ?, ?, "
            "?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?)",
            std::vector<std::string>{std::to_string(td.tableId),
                                     std::to_string(colId),
                                     cd.columnName,
                                     std::to_string(cd.columnType.get_type()),
                                     std::to_string(cd.columnType.get_subtype()),
                                     std::to_string(cd.columnType.get_dimension()),
                                     std::to_string(cd.columnType.get_scale()),
                                     std::to_string(cd.columnType.get_notnull()),
                                     std::to_string(cd.columnType.get_compression()),
                                     std::to_string(cd.columnType.get_comp_param()),
                                     std::to_string(cd.columnType.get_size()),
                                     "",
                                     std::to_string(cd.isSystemCol),
                                     std::to_string(cd.isVirtualCol),
                                     cd.virtualExpr});
        cd.tableId = td.tableId;
        cd.columnId = colId++;
        cds.push_back(cd);
      }
      if (td.isView) {
        sqliteConnector_.query_with_text_params("INSERT INTO mapd_views (tableid, sql) VALUES (?,?)",
                                                std::vector<std::string>{std::to_string(td.tableId), td.viewSQL});
      }
    } catch (std::exception& e) {
      sqliteConnector_.query("ROLLBACK TRANSACTION");
      throw;
    }
    sqliteConnector_.query("END TRANSACTION");
  } else {  // Temporary table
    td.tableId = nextTempTableId_++;
    int colId = 1;
    for (auto cd : columns) {
      if (cd.columnType.get_compression() == kENCODING_DICT) {
        std::string fileName("");
        std::string folderPath("");
        int dictId = nextTempDictId_++;
        DictDescriptor dd(dictId,
                          fileName,
                          cd.columnType.get_comp_param(),
                          false,
                          1,
                          folderPath,
                          true);  // Is dictName (2nd argument) used?
        dds.push_back(dd);
        if (!cd.columnType.is_array()) {
          cd.columnType.set_size(cd.columnType.get_comp_param() / 8);
        }
        cd.columnType.set_comp_param(dictId);
      }
      cd.tableId = td.tableId;
      cd.columnId = colId++;
      cds.push_back(cd);
    }
  }
  addTableToMap(td, cds, dds);
  calciteMgr_->updateMetadata(currentDB_.dbName, td.tableName);
}

namespace {

const ColumnDescriptor* get_foreign_col(const Catalog& cat, const Parser::SharedDictionaryDef& shared_dict_def) {
  const auto& table_name = shared_dict_def.get_foreign_table();
  const auto td = cat.getMetadataForTable(table_name);
  CHECK(td);
  const auto& foreign_col_name = shared_dict_def.get_foreign_column();
  return cat.getMetadataForColumn(td->tableId, foreign_col_name);
}

}  // namespace

bool Catalog::setColumnSharedDictionary(ColumnDescriptor& cd,
                                        const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs) {
  if (shared_dict_defs.empty()) {
    return false;
  }
  for (const auto& shared_dict_def : shared_dict_defs) {
    const auto& column = shared_dict_def.get_column();
    if (cd.columnName == column) {
      const auto foreign_ref_col = get_foreign_col(*this, shared_dict_def);
      CHECK(foreign_ref_col);
      cd.columnType = foreign_ref_col->columnType;
      const int dict_id = cd.columnType.get_comp_param();
      const auto dictIt = dictDescriptorMapById_.find(dict_id);
      CHECK(dictIt != dictDescriptorMapById_.end());
      const auto& dd = dictIt->second;
      CHECK_GE(dd->refcount, 1);
      ++dd->refcount;
      sqliteConnector_.query_with_text_params("UPDATE mapd_dictionaries SET refcount = refcount + 1 WHERE dictid = ?",
                                              {std::to_string(dict_id)});
      return true;
    }
  }
  return false;
}

void Catalog::setColumnDictionary(ColumnDescriptor& cd,
                                  std::list<DictDescriptor>& dds,
                                  const TableDescriptor& td,
                                  const bool isLogicalTable) {
  std::string dictName{"Initial_key"};
  int dictId{0};
  std::string folderPath;
  if (isLogicalTable) {
    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_dictionaries (name, nbits, is_shared, refcount) VALUES (?, ?, ?, 1)",
        std::vector<std::string>{dictName, std::to_string(cd.columnType.get_comp_param()), "0"});
    sqliteConnector_.query_with_text_param("SELECT dictid FROM mapd_dictionaries WHERE name = ?", dictName);
    dictId = sqliteConnector_.getData<int>(0, 0);
    dictName = td.tableName + "_" + cd.columnName + "_dict" + std::to_string(dictId);
    sqliteConnector_.query_with_text_param("UPDATE mapd_dictionaries SET name = ? WHERE name = 'Initial_key'",
                                           dictName);
    folderPath = basePath_ + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) + "_DICT_" + std::to_string(dictId);
  }
  DictDescriptor dd(dictId, dictName, cd.columnType.get_comp_param(), false, 1, folderPath, false);
  dds.push_back(dd);
  if (!cd.columnType.is_array()) {
    cd.columnType.set_size(cd.columnType.get_comp_param() / 8);
  }
  cd.columnType.set_comp_param(dictId);
}

void Catalog::createShardedTable(TableDescriptor& td,
                                 const list<ColumnDescriptor>& cols,
                                 const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs) {
  if (td.nShards > 0 && (td.shardedColumnId <= 0 || static_cast<size_t>(td.shardedColumnId) > cols.size())) {
    std::string error_message{"Invalid sharding column for table " + td.tableName + " of database " +
                              currentDB_.dbName};
    throw runtime_error(error_message);
  }

  /* create logical table */
  TableDescriptor tdl(td);
  createTable(tdl, cols, shared_dict_defs, true);  // create logical table
  int32_t logical_tb_id = tdl.tableId;

  /* create physical tables and link them to the logical table */
  std::vector<int32_t> physicalTables;
  for (int32_t i = 1; i <= td.nShards; i++) {
    TableDescriptor tdp(td);
    tdp.tableName = generatePhysicalTableName(tdp.tableName, i);
    tdp.shard = i - 1;
    createTable(tdp, cols, shared_dict_defs, false);  // create physical table
    int32_t physical_tb_id = tdp.tableId;

    /* add physical table to the vector of physical tables */
    physicalTables.push_back(physical_tb_id);
  }

  if (!physicalTables.empty()) {
    /* add logical to physical tables correspondence to the map */
    const auto it_ok = logicalToPhysicalTableMapById_.emplace(logical_tb_id, physicalTables);
    CHECK(it_ok.second);
    /* update sqlite mapd_logical_to_physical in sqlite database */
    updateLogicalToPhysicalTableMap(logical_tb_id);
  }
}

void Catalog::truncateTable(const TableDescriptor* td) {
  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(td->tableId);
  if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
    // truncate all corresponding physical tables if this is a logical table
    const auto physicalTables = physicalTableIt->second;
    CHECK(!physicalTables.empty());
    for (size_t i = 0; i < physicalTables.size(); i++) {
      int32_t physical_tb_id = physicalTables[i];
      const TableDescriptor* phys_td = getMetadataForTable(physical_tb_id);
      CHECK(phys_td);
      doTruncateTable(phys_td);
    }
  }
  doTruncateTable(td);
}

void Catalog::doTruncateTable(const TableDescriptor* td) {
  const int tableId = td->tableId;
  // get a write lock on the table before trying to remove it
  ChunkKey chunkKey = {currentDB_.dbId, tableId};
  mapd_unique_lock<mapd_shared_mutex> tableLevelWriteLock(*dataMgr_->getMutexForChunkPrefix(chunkKey));
  // must destroy fragmenter before deleteChunks is called.
  if (td->fragmenter != nullptr) {
    auto tableDescIt = tableDescriptorMapById_.find(tableId);
    delete td->fragmenter;
    tableDescIt->second->fragmenter = nullptr;  // get around const-ness
  }
  ChunkKey chunkKeyPrefix = {currentDB_.dbId, tableId};
  // assuming deleteChunksWithPrefix is atomic
  dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix);
  // MAT TODO fix this
  // NOTE This is unsafe , if there are updates occuring at same time
  dataMgr_->checkpoint(currentDB_.dbId, tableId);
  dataMgr_->removeTableRelatedDS(currentDB_.dbId, tableId);

  std::unique_ptr<StringDictionaryClient> client;
  if (g_aggregator) {
    CHECK(!string_dict_hosts_.empty());
    client.reset(new StringDictionaryClient(string_dict_hosts_.front(), -1, true));
  }
  // clean up any dictionaries
  // delete all column descriptors for the table
  for (int i = 1; i <= td->nColumns; i++) {
    ColumnIdKey cidKey(tableId, i);
    ColumnDescriptorMapById::iterator colDescIt = columnDescriptorMapById_.find(cidKey);
    ColumnDescriptor* cd = colDescIt->second;
    const int dictId = cd->columnType.get_comp_param();
    // Dummy dictionaries created for a shard of a logical table have the id set to zero.
    if (cd->columnType.get_compression() == kENCODING_DICT && dictId) {
      const auto dictIt = dictDescriptorMapById_.find(dictId);
      CHECK(dictIt != dictDescriptorMapById_.end());
      const auto& dd = dictIt->second;
      CHECK_GE(dd->refcount, 1);
      // if this is the only table using this dict reset the dict
      if (dd->refcount == 1) {
        boost::filesystem::remove_all(dd->dictFolderPath);
        if (client) {
          client->drop(dd->dictId, currentDB_.dbId);
        }
        if (!dd->dictIsTemp)
          boost::filesystem::create_directory(dd->dictFolderPath);
      }

      DictDescriptor* new_dd = new DictDescriptor(
          dd->dictId, dd->dictName, dd->dictNBits, dd->dictIsShared, dd->refcount, dd->dictFolderPath, dd->dictIsTemp);
      dictDescriptorMapById_.erase(dictIt);
      // now create new Dict -- need to figure out what to do here for temp tables
      if (client) {
        client->create(new_dd->dictId, currentDB_.dbId, dd->dictIsTemp);
      }
      dictDescriptorMapById_[dictId].reset(new_dd);
      getMetadataForDict(dictId);
    }
  }
}

// used by rollback_table_epoch to clean up in memory artifcats after a rollback
void Catalog::removeChunks(const int table_id) {
  auto td = getMetadataForTable(table_id);

  if (td->fragmenter != nullptr) {
    auto tableDescIt = tableDescriptorMapById_.find(table_id);
    delete td->fragmenter;
    tableDescIt->second->fragmenter = nullptr;  // get around const-ness
  }
}

void Catalog::dropTable(const TableDescriptor* td) {
  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(td->tableId);
  if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
    // remove all corresponding physical tables if this is a logical table
    const auto physicalTables = physicalTableIt->second;
    CHECK(!physicalTables.empty());
    for (size_t i = 0; i < physicalTables.size(); i++) {
      int32_t physical_tb_id = physicalTables[i];
      const TableDescriptor* phys_td = getMetadataForTable(physical_tb_id);
      CHECK(phys_td);
      doDropTable(phys_td);
      removeTableFromMap(phys_td->tableName, physical_tb_id);
    }

    // remove corresponding record from the logicalToPhysicalTableMap in sqlite database
    sqliteConnector_.query("BEGIN TRANSACTION");
    try {
      sqliteConnector_.query_with_text_param("DELETE FROM mapd_logical_to_physical WHERE logical_table_id = ?",
                                             std::to_string(td->tableId));
    } catch (std::exception& e) {
      sqliteConnector_.query("ROLLBACK TRANSACTION");
      throw;
    }
    sqliteConnector_.query("END TRANSACTION");

    logicalToPhysicalTableMapById_.erase(td->tableId);
  }
  doDropTable(td);
  removeTableFromMap(td->tableName, td->tableId);
}

void Catalog::doDropTable(const TableDescriptor* td) {
  const int tableId = td->tableId;
  // get a write lock on the table before trying to remove it
  ChunkKey chunkKey = {currentDB_.dbId, tableId};
  mapd_unique_lock<mapd_shared_mutex> tableLevelWriteLock(*dataMgr_->getMutexForChunkPrefix(chunkKey));
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_param("DELETE FROM mapd_tables WHERE tableid = ?", std::to_string(tableId));
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_dictionaries SET refcount = refcount - 1 WHERE dictid in (select comp_param from mapd_columns "
        "where compression = ? and tableid = ?)",
        std::vector<std::string>{std::to_string(kENCODING_DICT), std::to_string(tableId)});
    sqliteConnector_.query_with_text_params(
        "DELETE FROM mapd_dictionaries WHERE dictid in (select comp_param from mapd_columns where compression = ? "
        "and tableid = ?) and refcount = 0",
        std::vector<std::string>{std::to_string(kENCODING_DICT), std::to_string(tableId)});
    sqliteConnector_.query_with_text_param("DELETE FROM mapd_columns WHERE tableid = ?", std::to_string(tableId));
    if (td->isView)
      sqliteConnector_.query_with_text_param("DELETE FROM mapd_views WHERE tableid = ?", std::to_string(tableId));
    // must destroy fragmenter before deleteChunks is called.
    if (td->fragmenter != nullptr) {
      auto tableDescIt = tableDescriptorMapById_.find(tableId);
      delete td->fragmenter;
      tableDescIt->second->fragmenter = nullptr;  // get around const-ness
    }
    ChunkKey chunkKeyPrefix = {currentDB_.dbId, tableId};
    // assuming deleteChunksWithPrefix is atomic
    dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix);
    // MAT TODO fix this
    // NOTE This is unsafe , if there are updates occuring at same time
    dataMgr_->checkpoint(currentDB_.dbId, tableId);
    dataMgr_->removeTableRelatedDS(currentDB_.dbId, tableId);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
}

void Catalog::renamePhysicalTable(const TableDescriptor* td, const string& newTableName) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params("UPDATE mapd_tables SET name = ? WHERE tableid = ?",
                                            std::vector<std::string>{newTableName, std::to_string(td->tableId)});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.find(to_upper(td->tableName));
  CHECK(tableDescIt != tableDescriptorMap_.end());
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
  // Get table descriptor to change it
  TableDescriptor* changeTd = tableDescIt->second;
  changeTd->tableName = newTableName;
  tableDescriptorMap_.erase(tableDescIt);  // erase entry under old name
  tableDescriptorMap_[to_upper(newTableName)] = changeTd;
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
}

void Catalog::renameTable(const TableDescriptor* td, const string& newTableName) {
  // rename all corresponding physical tables if this is a logical table
  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(td->tableId);
  if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
    const auto physicalTables = physicalTableIt->second;
    CHECK(!physicalTables.empty());
    for (size_t i = 0; i < physicalTables.size(); i++) {
      int32_t physical_tb_id = physicalTables[i];
      const TableDescriptor* phys_td = getMetadataForTable(physical_tb_id);
      CHECK(phys_td);
      std::string newPhysTableName = generatePhysicalTableName(newTableName, static_cast<int32_t>(i + 1));
      renamePhysicalTable(phys_td, newPhysTableName);
    }
  }
  renamePhysicalTable(td, newTableName);
}

void Catalog::renameColumn(const TableDescriptor* td, const ColumnDescriptor* cd, const string& newColumnName) {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_columns SET name = ? WHERE tableid = ? AND columnid = ?",
        std::vector<std::string>{newColumnName, std::to_string(td->tableId), std::to_string(cd->columnId)});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  ColumnDescriptorMap::iterator columnDescIt =
      columnDescriptorMap_.find(std::make_tuple(td->tableId, to_upper(cd->columnName)));
  CHECK(columnDescIt != columnDescriptorMap_.end());
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
  ColumnDescriptor* changeCd = columnDescIt->second;
  changeCd->columnName = newColumnName;
  columnDescriptorMap_.erase(columnDescIt);  // erase entry under old name
  columnDescriptorMap_[std::make_tuple(td->tableId, to_upper(newColumnName))] = changeCd;
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
}

void Catalog::createFrontendView(FrontendViewDescriptor& vd) {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    // TODO(andrew): this should be an upsert
    sqliteConnector_.query_with_text_params("SELECT viewid FROM mapd_frontend_views WHERE name = ? and userid = ?",
                                            std::vector<std::string>{vd.viewName, std::to_string(vd.userId)});
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_frontend_views SET view_state = ?, image_hash = ?, view_metadata = ?, update_time = "
          "datetime('now') where name = ? "
          "and userid = ?",
          std::vector<std::string>{
              vd.viewState, vd.imageHash, vd.viewMetadata, vd.viewName, std::to_string(vd.userId)});
    } else {
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_frontend_views (name, view_state, image_hash, view_metadata, update_time, userid) VALUES "
          "(?,?,?,?, "
          "datetime('now'), ?)",
          std::vector<std::string>{
              vd.viewName, vd.viewState, vd.imageHash, vd.viewMetadata, std::to_string(vd.userId)});
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");

  // now get the auto generated viewid
  try {
    sqliteConnector_.query_with_text_params(
        "SELECT viewid, strftime('%Y-%m-%dT%H:%M:%SZ', update_time) FROM mapd_frontend_views "
        "WHERE name = ? and userid = ?",
        std::vector<std::string>{vd.viewName, std::to_string(vd.userId)});
    vd.viewId = sqliteConnector_.getData<int>(0, 0);
    vd.updateTime = sqliteConnector_.getData<std::string>(0, 1);
  } catch (std::exception& e) {
    throw;
  }
  addFrontendViewToMap(vd);
}

std::string Catalog::calculateSHA1(const std::string& data) {
  boost::uuids::detail::sha1 sha1;
  unsigned int digest[5];
  sha1.process_bytes(data.c_str(), data.length());
  sha1.get_digest(digest);
  std::stringstream ss;
  for (size_t i = 0; i < 5; i++) {
    ss << std::hex << digest[i];
  }
  return ss.str();
}

std::string Catalog::createLink(LinkDescriptor& ld, size_t min_length) {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    ld.link = calculateSHA1(ld.viewState + ld.viewMetadata + std::to_string(ld.userId)).substr(0, 8);
    sqliteConnector_.query_with_text_params("SELECT linkid FROM mapd_links WHERE link = ? and userid = ?",
                                            std::vector<std::string>{ld.link, std::to_string(ld.userId)});
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_links SET update_time = datetime('now') WHERE userid = ? AND link = ?",
          std::vector<std::string>{std::to_string(ld.userId), ld.link});
    } else {
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_links (userid, link, view_state, view_metadata, update_time) VALUES (?,?,?,?, "
          "datetime('now'))",
          std::vector<std::string>{std::to_string(ld.userId), ld.link, ld.viewState, ld.viewMetadata});
    }
    // now get the auto generated viewid
    sqliteConnector_.query_with_text_param(
        "SELECT linkid, strftime('%Y-%m-%dT%H:%M:%SZ', update_time) FROM mapd_links WHERE link = ?", ld.link);
    ld.linkId = sqliteConnector_.getData<int>(0, 0);
    ld.updateTime = sqliteConnector_.getData<std::string>(0, 1);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  addLinkToMap(ld);
  return ld.link;
}

std::vector<const TableDescriptor*> Catalog::getPhysicalTablesDescriptors(
    const TableDescriptor* logicalTableDesc) const {
  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(logicalTableDesc->tableId);
  if (physicalTableIt == logicalToPhysicalTableMapById_.end()) {
    return {logicalTableDesc};
  }

  const auto physicalTablesIds = physicalTableIt->second;
  CHECK(!physicalTablesIds.empty());
  std::vector<const TableDescriptor*> physicalTables;
  for (size_t i = 0; i < physicalTablesIds.size(); i++) {
    physicalTables.push_back(getMetadataForTable(physicalTablesIds[i]));
  }

  return physicalTables;
}

std::string Catalog::generatePhysicalTableName(const std::string& logicalTableName, const int32_t& shardNumber) {
  std::string physicalTableName = logicalTableName + physicalTableNameTag_ + std::to_string(shardNumber);
  return (physicalTableName);
}

bool SessionInfo::checkDBAccessPrivileges(std::vector<bool> privs) const {
  auto& cat = get_catalog();
  auto& sys_cat = static_cast<Catalog_Namespace::SysCatalog&>(cat);
  if (!cat.isAccessPrivCheckEnabled()) {
    // run flow without DB object level access permission checks
    Privileges wants_privs;
    if (get_currentUser().isSuper) {
      wants_privs.super_ = true;
    } else {
      wants_privs.super_ = false;
    }
    wants_privs.select_ = false;
    wants_privs.insert_ = true;
    auto currentDB = static_cast<Catalog_Namespace::DBMetadata>(cat.get_currentDB());
    auto currentUser = static_cast<Catalog_Namespace::UserMetadata>(get_currentUser());
    return sys_cat.checkPrivileges(currentUser, currentDB, wants_privs);
  } else {
    // run flow with DB object level access permission checks
    DBObject object(cat.get_currentDB().dbName, DatabaseDBObjectType);
    sys_cat.populateDBObjectKey(object, cat);
    object.setPrivileges(privs);
    std::vector<DBObject> privObjects;
    privObjects.push_back(object);
    return sys_cat.checkPrivileges(get_currentUser(), privObjects);
  }
}

void SessionInfo::setSysCatalog(Catalog_Namespace::SysCatalog* sys_cat) {
  mapd_sys_cat = sys_cat;
}

Catalog_Namespace::SysCatalog* SessionInfo::getSysCatalog() const {
  return mapd_sys_cat;
}

}  // Catalog_Namespace
