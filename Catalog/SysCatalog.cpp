/*
 * Copyright 2019 MapD Technologies, Inc.
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
 * @file		SysCatalog.cpp
 * @author	Todd Mostak <todd@map-d.com>, Wei Hong <wei@map-d.com>
 * @brief		Functions for System Catalog
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "SysCatalog.h"
#include <algorithm>
#include <cassert>
#include <exception>
#include <list>
#include <memory>
#include <random>
#include <sstream>
#include "Catalog.h"

#include "Catalog/AuthMetadata.h"
#include "DataMgr/LockMgr.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/adaptor/map.hpp>
#include <boost/version.hpp>

#include "../Parser/ParserNode.h"
#include "../Shared/File.h"
#include "../Shared/StringTransform.h"
#include "../Shared/measure.h"
#include "MapDRelease.h"
#include "RWLocks.h"
#include "bcrypt.h"

using std::list;
using std::map;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;

namespace {

std::string hash_with_bcrypt(const std::string& pwd) {
  char salt[BCRYPT_HASHSIZE], hash[BCRYPT_HASHSIZE];
  CHECK(bcrypt_gensalt(-1, salt) == 0);
  CHECK(bcrypt_hashpw(pwd.c_str(), salt, hash) == 0);
  return std::string(hash, BCRYPT_HASHSIZE);
}

}  // namespace

namespace Catalog_Namespace {

thread_local bool SysCatalog::thread_holds_read_lock = false;

using sys_read_lock = read_lock<SysCatalog>;
using sys_write_lock = write_lock<SysCatalog>;
using sys_sqlite_lock = sqlite_lock<SysCatalog>;

void SysCatalog::init(const std::string& basePath,
                      std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                      AuthMetadata authMetadata,
                      std::shared_ptr<Calcite> calcite,
                      bool is_new_db,
                      bool aggregator,
                      const std::vector<LeafHostInfo>& string_dict_hosts) {
  {
    sys_write_lock write_lock(this);
    sys_sqlite_lock sqlite_lock(this);

    basePath_ = basePath;
    dataMgr_ = dataMgr;
    ldap_server_.reset(new LdapServer(authMetadata));
    rest_server_.reset(new RestServer(authMetadata));
    calciteMgr_ = calcite;
    string_dict_hosts_ = string_dict_hosts;
    aggregator_ = aggregator;
    bool db_exists =
        boost::filesystem::exists(basePath_ + "/mapd_catalogs/" + MAPD_SYSTEM_CATALOG);
    sqliteConnector_.reset(
        new SqliteConnector(MAPD_SYSTEM_CATALOG, basePath_ + "/mapd_catalogs/"));
    if (is_new_db) {
      initDB();
    } else {
      if (!db_exists) {
        importDataFromOldMapdDB();
      }
      checkAndExecuteMigrations();
    }
    buildRoleMap();
    buildUserRoleMap();
    buildObjectDescriptorMap();
  }
  // check if mapd db catalog is loaded in cat_map
  // if not, load mapd-cat here
  if (!Catalog::get(MAPD_DEFAULT_DB)) {
    Catalog_Namespace::DBMetadata db_meta;
    CHECK(getMetadataForDB(MAPD_DEFAULT_DB, db_meta));
    auto mapd_cat = std::make_shared<Catalog>(
        basePath_, db_meta, dataMgr_, string_dict_hosts_, calciteMgr_, is_new_db);
    Catalog::set(MAPD_DEFAULT_DB, mapd_cat);
  }
}

SysCatalog::~SysCatalog() {
  sys_write_lock write_lock(this);
  for (auto grantee = granteeMap_.begin(); grantee != granteeMap_.end(); ++grantee) {
    delete grantee->second;
  }
  granteeMap_.clear();
  for (ObjectRoleDescriptorMap::iterator objectIt = objectDescriptorMap_.begin();
       objectIt != objectDescriptorMap_.end();) {
    ObjectRoleDescriptorMap::iterator eraseIt = objectIt++;
    delete eraseIt->second;
  }
  objectDescriptorMap_.clear();
}

void SysCatalog::initDB() {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "CREATE TABLE mapd_users (userid integer primary key, name text unique, "
        "passwd_hash text, issuper boolean, default_db integer references "
        "mapd_databases)");
    sqliteConnector_->query_with_text_params(
        "INSERT INTO mapd_users VALUES (?, ?, ?, 1, NULL)",
        std::vector<std::string>{MAPD_ROOT_USER_ID_STR,
                                 MAPD_ROOT_USER,
                                 hash_with_bcrypt(MAPD_ROOT_PASSWD_DEFAULT)});
    sqliteConnector_->query(
        "CREATE TABLE mapd_databases (dbid integer primary key, name text unique, owner "
        "integer references mapd_users)");
    sqliteConnector_->query(
        "CREATE TABLE mapd_roles(roleName text, userName text, UNIQUE(roleName, "
        "userName))");
    sqliteConnector_->query(
        "CREATE TABLE mapd_object_permissions ("
        "roleName text, "
        "roleType bool, "
        "dbId integer references mapd_databases, "
        "objectName text, "
        "objectId integer, "
        "objectPermissionsType integer, "
        "objectPermissions integer, "
        "objectOwnerId integer, UNIQUE(roleName, objectPermissionsType, dbId, "
        "objectId))");
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
  createDatabase(MAPD_DEFAULT_DB, MAPD_ROOT_USER_ID);
}

void SysCatalog::checkAndExecuteMigrations() {
  migratePrivileged_old();
  createUserRoles();
  migratePrivileges();
  migrateDBAccessPrivileges();
  updateUserSchema();  // must come before updatePasswordsToHashes()
  updatePasswordsToHashes();
  updateBlankPasswordsToRandom();  // must come after updatePasswordsToHashes()
}

void SysCatalog::updateUserSchema() {
  sys_sqlite_lock sqlite_lock(this);

  // check to see if the new column already exists
  sqliteConnector_->query("PRAGMA TABLE_INFO(mapd_users)");
  for (size_t i = 0; i < sqliteConnector_->getNumRows(); i++) {
    const auto& col_name = sqliteConnector_->getData<std::string>(i, 1);
    if (col_name == "default_db") {
      return;  // new column already exists
    }
  }

  // create the new column
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "ALTER TABLE mapd_users ADD COLUMN default_db INTEGER REFERENCES mapd_databases");
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::importDataFromOldMapdDB() {
  sys_sqlite_lock sqlite_lock(this);
  std::string mapd_db_path = basePath_ + "/mapd_catalogs/" + MAPD_DEFAULT_DB;
  sqliteConnector_->query("ATTACH DATABASE `" + mapd_db_path + "` as old_cat");
  sqliteConnector_->query("BEGIN TRANSACTION");
  LOG(INFO) << "Moving global metadata into a separate catalog";
  try {
    auto moveTableIfExists = [conn = sqliteConnector_.get()](const std::string& tableName,
                                                             bool deleteOld = true) {
      conn->query("SELECT sql FROM old_cat.sqlite_master WHERE type='table' AND name='" +
                  tableName + "'");
      if (conn->getNumRows() != 0) {
        conn->query(conn->getData<string>(0, 0));
        conn->query("INSERT INTO " + tableName + " SELECT * FROM old_cat." + tableName);
        if (deleteOld) {
          conn->query("DROP TABLE old_cat." + tableName);
        }
      }
    };
    moveTableIfExists("mapd_users");
    moveTableIfExists("mapd_databases");
    moveTableIfExists("mapd_roles");
    moveTableIfExists("mapd_object_permissions");
    moveTableIfExists("mapd_privileges");
    moveTableIfExists("mapd_version_history", false);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to move global metadata into a separate catalog: " << e.what();
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    try {
      sqliteConnector_->query("DETACH DATABASE old_cat");
    } catch (const std::exception&) {
      // nothing to do here
    }
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
  const std::string sys_catalog_path =
      basePath_ + "/mapd_catalogs/" + MAPD_SYSTEM_CATALOG;
  LOG(INFO) << "Global metadata has been successfully moved into a separate catalog: "
            << sys_catalog_path
            << ". Using this database with an older version of omnisci_server "
               "is now impossible.";
  try {
    sqliteConnector_->query("DETACH DATABASE old_cat");
  } catch (const std::exception&) {
    // nothing to do here
  }
}

void SysCatalog::createUserRoles() {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='mapd_roles'");
    if (sqliteConnector_->getNumRows() != 0) {
      // already done
      sqliteConnector_->query("END TRANSACTION");
      return;
    }
    sqliteConnector_->query(
        "CREATE TABLE mapd_roles(roleName text, userName text, UNIQUE(roleName, "
        "userName))");
    sqliteConnector_->query("SELECT name FROM mapd_users WHERE name <> \'" +
                            MAPD_ROOT_USER + "\'");
    size_t numRows = sqliteConnector_->getNumRows();
    vector<string> user_names;
    for (size_t i = 0; i < numRows; ++i) {
      user_names.push_back(sqliteConnector_->getData<string>(i, 0));
    }
    for (const auto& user_name : user_names) {
      // for each user, create a fake role with the same name
      sqliteConnector_->query_with_text_params(
          "INSERT INTO mapd_roles(roleName, userName) VALUES (?, ?)",
          vector<string>{user_name, user_name});
    }
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void deleteObjectPrivileges(std::unique_ptr<SqliteConnector>& sqliteConnector,
                            std::string roleName,
                            bool userRole,
                            DBObject& object) {
  DBObjectKey key = object.getObjectKey();

  sqliteConnector->query_with_text_params(
      "DELETE FROM mapd_object_permissions WHERE roleName = ?1 and roleType = ?2 and "
      "objectPermissionsType = ?3 and "
      "dbId = "
      "?4 "
      "and objectId = ?5",
      std::vector<std::string>{roleName,
                               std::to_string(userRole),
                               std::to_string(key.permissionType),
                               std::to_string(key.dbId),
                               std::to_string(key.objectId)});
}

void insertOrUpdateObjectPrivileges(std::unique_ptr<SqliteConnector>& sqliteConnector,
                                    std::string roleName,
                                    bool userRole,
                                    DBObject& object) {
  DBObjectKey key = object.getObjectKey();

  sqliteConnector->query_with_text_params(
      "INSERT OR REPLACE INTO mapd_object_permissions("
      "roleName, "
      "roleType, "
      "objectPermissionsType, "
      "dbId, "
      "objectId, "
      "objectPermissions, "
      "objectOwnerId,"
      "objectName) "
      "VALUES (?1, ?2, ?3, "
      "?4, ?5, ?6, ?7, ?8)",
      std::vector<std::string>{
          roleName,                                           // roleName
          userRole ? "1" : "0",                               // roleType
          std::to_string(key.permissionType),                 // permissionType
          std::to_string(key.dbId),                           // dbId
          std::to_string(key.objectId),                       // objectId
          std::to_string(object.getPrivileges().privileges),  // objectPrivileges
          std::to_string(object.getOwner()),                  // objectOwnerId
          object.getName()                                    // name
      });
}

void SysCatalog::migratePrivileges() {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "SELECT name FROM sqlite_master WHERE type='table' AND "
        "name='mapd_object_permissions'");
    if (sqliteConnector_->getNumRows() != 0) {
      // already done
      sqliteConnector_->query("END TRANSACTION");
      return;
    }

    sqliteConnector_->query(
        "CREATE TABLE IF NOT EXISTS mapd_object_permissions ("
        "roleName text, "
        "roleType bool, "
        "dbId integer references mapd_databases, "
        "objectName text, "
        "objectId integer, "
        "objectPermissionsType integer, "
        "objectPermissions integer, "
        "objectOwnerId integer, UNIQUE(roleName, objectPermissionsType, dbId, "
        "objectId))");

    // get the list of databases and their grantees
    sqliteConnector_->query(
        "SELECT userid, dbid FROM mapd_privileges WHERE select_priv = 1 and insert_priv "
        "= 1");
    size_t numRows = sqliteConnector_->getNumRows();
    vector<pair<int, int>> db_grantees(numRows);
    for (size_t i = 0; i < numRows; ++i) {
      db_grantees[i].first = sqliteConnector_->getData<int>(i, 0);
      db_grantees[i].second = sqliteConnector_->getData<int>(i, 1);
    }
    // map user names to user ids
    sqliteConnector_->query("select userid, name from mapd_users");
    numRows = sqliteConnector_->getNumRows();
    std::unordered_map<int, string> users_by_id;
    std::unordered_map<int, bool> user_has_privs;
    for (size_t i = 0; i < numRows; ++i) {
      users_by_id[sqliteConnector_->getData<int>(i, 0)] =
          sqliteConnector_->getData<string>(i, 1);
      user_has_privs[sqliteConnector_->getData<int>(i, 0)] = false;
    }
    // map db names to db ids
    sqliteConnector_->query("select dbid, name from mapd_databases");
    numRows = sqliteConnector_->getNumRows();
    std::unordered_map<int, string> dbs_by_id;
    for (size_t i = 0; i < numRows; ++i) {
      dbs_by_id[sqliteConnector_->getData<int>(i, 0)] =
          sqliteConnector_->getData<string>(i, 1);
    }
    // migrate old privileges to new privileges: if user had insert access to database, he
    // was a grantee
    for (const auto& grantee : db_grantees) {
      user_has_privs[grantee.first] = true;
      auto dbName = dbs_by_id[grantee.second];
      {
        // table level permissions
        DBObjectKey key;
        key.permissionType = DBObjectType::TableDBObjectType;
        key.dbId = grantee.second;
        DBObject object(key, AccessPrivileges::ALL_TABLE_MIGRATE, MAPD_ROOT_USER_ID);
        object.setName(dbName);
        insertOrUpdateObjectPrivileges(
            sqliteConnector_, users_by_id[grantee.first], true, object);
      }

      {
        // dashboard level permissions
        DBObjectKey key;
        key.permissionType = DBObjectType::DashboardDBObjectType;
        key.dbId = grantee.second;
        DBObject object(key, AccessPrivileges::ALL_DASHBOARD_MIGRATE, MAPD_ROOT_USER_ID);
        object.setName(dbName);
        insertOrUpdateObjectPrivileges(
            sqliteConnector_, users_by_id[grantee.first], true, object);
      }

      {
        // view level permissions
        DBObjectKey key;
        key.permissionType = DBObjectType::ViewDBObjectType;
        key.dbId = grantee.second;
        DBObject object(key, AccessPrivileges::ALL_VIEW_MIGRATE, MAPD_ROOT_USER_ID);
        object.setName(dbName);
        insertOrUpdateObjectPrivileges(
            sqliteConnector_, users_by_id[grantee.first], true, object);
      }
    }
    for (auto user : user_has_privs) {
      auto dbName = dbs_by_id[0];
      if (user.second == false && user.first != MAPD_ROOT_USER_ID) {
        {
          DBObjectKey key;
          key.permissionType = DBObjectType::DatabaseDBObjectType;
          key.dbId = 0;
          DBObject object(key, AccessPrivileges::NONE, MAPD_ROOT_USER_ID);
          object.setName(dbName);
          insertOrUpdateObjectPrivileges(
              sqliteConnector_, users_by_id[user.first], true, object);
        }
      }
    }
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::updatePasswordsToHashes() {
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='mapd_users'");
    if (sqliteConnector_->getNumRows() == 0) {
      // Nothing to update
      sqliteConnector_->query("END TRANSACTION");
      return;
    }
    sqliteConnector_->query("PRAGMA TABLE_INFO(mapd_users)");
    for (size_t i = 0; i < sqliteConnector_->getNumRows(); i++) {
      const auto& col_name = sqliteConnector_->getData<std::string>(i, 1);
      if (col_name == "passwd_hash") {
        sqliteConnector_->query("END TRANSACTION");
        return;
      }
    }
    // Alas, SQLite can't drop columns so we have to recreate the table
    // (or, optionally, add the new column and reset the old one to a bunch of nulls)
    sqliteConnector_->query("SELECT userid, passwd FROM mapd_users");
    auto numRows = sqliteConnector_->getNumRows();
    vector<std::string> users, passwords;
    for (size_t i = 0; i < numRows; i++) {
      users.push_back(sqliteConnector_->getData<std::string>(i, 0));
      passwords.push_back(sqliteConnector_->getData<std::string>(i, 1));
    }
    sqliteConnector_->query(
        "CREATE TABLE mapd_users_tmp (userid integer primary key, name text unique, "
        "passwd_hash text, issuper boolean, default_db integer references "
        "mapd_databases)");
    sqliteConnector_->query(
        "INSERT INTO mapd_users_tmp(userid, name, passwd_hash, issuper, default_db) "
        "SELECT userid, name, null, issuper, default_db FROM mapd_users");
    for (size_t i = 0; i < users.size(); ++i) {
      sqliteConnector_->query_with_text_params(
          "UPDATE mapd_users_tmp SET passwd_hash = ? WHERE userid = ?",
          std::vector<std::string>{hash_with_bcrypt(passwords[i]), users[i]});
    }
    sqliteConnector_->query("DROP TABLE mapd_users");
    sqliteConnector_->query("ALTER TABLE mapd_users_tmp RENAME TO mapd_users");
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to hash passwords: " << e.what();
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
  sqliteConnector_->query("VACUUM");  // physically delete plain text passwords
  LOG(INFO) << "Passwords were successfully hashed";
}

void SysCatalog::updateBlankPasswordsToRandom() {
  const std::string UPDATE_BLANK_PASSWORDS_TO_RANDOM = "update_blank_passwords_to_random";
  sqliteConnector_->query_with_text_params(
      "SELECT migration_history FROM mapd_version_history WHERE migration_history = ?",
      std::vector<std::string>{UPDATE_BLANK_PASSWORDS_TO_RANDOM});
  if (sqliteConnector_->getNumRows()) {
    return;
  }

  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "SELECT userid, passwd_hash, name FROM mapd_users WHERE name <> 'mapd'");
    auto numRows = sqliteConnector_->getNumRows();
    vector<std::string> users, passwords, names;
    for (size_t i = 0; i < numRows; i++) {
      users.push_back(sqliteConnector_->getData<std::string>(i, 0));
      passwords.push_back(sqliteConnector_->getData<std::string>(i, 1));
      names.push_back(sqliteConnector_->getData<std::string>(i, 2));
    }
    for (size_t i = 0; i < users.size(); ++i) {
      int pwd_check_result = bcrypt_checkpw("", passwords[i].c_str());
      // if the check fails there is a good chance that data on disc is broken
      CHECK(pwd_check_result >= 0);
      if (pwd_check_result != 0) {
        continue;
      }
      LOG(WARNING) << "resetting blank password for user " << names[i] << " (" << users[i]
                   << ") to a random password";
      sqliteConnector_->query_with_text_params(
          "UPDATE mapd_users SET passwd_hash = ? WHERE userid = ?",
          std::vector<std::string>{hash_with_bcrypt(generate_random_string(15)),
                                   users[i]});
    }
    sqliteConnector_->query_with_text_params(
        "INSERT INTO mapd_version_history(version, migration_history) values(?,?)",
        std::vector<std::string>{std::to_string(MAPD_VERSION),
                                 UPDATE_BLANK_PASSWORDS_TO_RANDOM});
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to fix blank passwords: " << e.what();
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::migrateDBAccessPrivileges() {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "select name from sqlite_master WHERE type='table' AND "
        "name='mapd_version_history'");
    if (sqliteConnector_->getNumRows() == 0) {
      sqliteConnector_->query(
          "CREATE TABLE mapd_version_history(version integer, migration_history text "
          "unique)");
    } else {
      sqliteConnector_->query(
          "select * from mapd_version_history where migration_history = "
          "'db_access_privileges'");
      if (sqliteConnector_->getNumRows() != 0) {
        // both privileges migrated
        // no need for further execution
        sqliteConnector_->query("END TRANSACTION");
        return;
      }
    }
    // Insert check for migration
    sqliteConnector_->query_with_text_params(
        "INSERT INTO mapd_version_history(version, migration_history) values(?,?)",
        std::vector<std::string>{std::to_string(MAPD_VERSION), "db_access_privileges"});

    sqliteConnector_->query("select dbid, name from mapd_databases");
    std::unordered_map<int, string> databases;
    for (size_t i = 0; i < sqliteConnector_->getNumRows(); ++i) {
      databases[sqliteConnector_->getData<int>(i, 0)] =
          sqliteConnector_->getData<string>(i, 1);
    }

    sqliteConnector_->query("select userid, name from mapd_users");
    std::unordered_map<int, string> users;
    for (size_t i = 0; i < sqliteConnector_->getNumRows(); ++i) {
      users[sqliteConnector_->getData<int>(i, 0)] =
          sqliteConnector_->getData<string>(i, 1);
    }

    // All existing users by default will be granted DB Access permissions
    // and view sql editor privileges
    DBMetadata dbmeta;
    for (auto db_ : databases) {
      CHECK(SysCatalog::instance().getMetadataForDB(db_.second, dbmeta));
      for (auto user : users) {
        if (user.first != MAPD_ROOT_USER_ID) {
          {
            DBObjectKey key;
            key.permissionType = DBObjectType::DatabaseDBObjectType;
            key.dbId = dbmeta.dbId;

            // access permission;
            DBObject object_access(key, AccessPrivileges::ACCESS, dbmeta.dbOwner);
            object_access.setObjectType(DBObjectType::DatabaseDBObjectType);
            object_access.setName(dbmeta.dbName);
            // sql_editor permission
            DBObject object_editor(
                key, AccessPrivileges::VIEW_SQL_EDITOR, dbmeta.dbOwner);
            object_editor.setObjectType(DBObjectType::DatabaseDBObjectType);
            object_editor.setName(dbmeta.dbName);
            object_editor.updatePrivileges(object_access);
            insertOrUpdateObjectPrivileges(
                sqliteConnector_, user.second, true, object_editor);
          }
        }
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to migrate db access privileges: " << e.what();
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
  LOG(INFO) << "Successfully migrated db access privileges";
}

void SysCatalog::migratePrivileged_old() {
  sys_sqlite_lock sqlite_lock(this);

  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "CREATE TABLE IF NOT EXISTS mapd_privileges (userid integer references "
        "mapd_users, dbid integer references "
        "mapd_databases, select_priv boolean, insert_priv boolean, UNIQUE(userid, "
        "dbid))");
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

std::shared_ptr<Catalog> SysCatalog::login(std::string& dbname,
                                           std::string& username,
                                           const std::string& password,
                                           UserMetadata& user_meta,
                                           bool check_password) {
  // NOTE: The dbname can be reset by getMetadataWithDefault(). The username can be reset
  // by SamlServer's login()/authenticate_user().

  sys_write_lock write_lock(this);

  if (check_password) {
    {
      if (!checkPasswordForUser(password, username, user_meta)) {
        throw std::runtime_error("Invalid credentials.");
      }
    }
  } else {  // not checking for password so user must exist
    if (!getMetadataForUser(username, user_meta)) {
      throw std::runtime_error("Invalid credentials.");
    }
  }
  // we should have a user and user_meta by now

  Catalog_Namespace::DBMetadata db_meta;

  getMetadataWithDefault(dbname, username, db_meta, user_meta);

  return Catalog::get(
      basePath_, db_meta, dataMgr_, string_dict_hosts_, calciteMgr_, false);
}

std::shared_ptr<Catalog> SysCatalog::switchDatabase(std::string& dbname,
                                                    const std::string& username) {
  DBMetadata db_meta;
  UserMetadata user_meta;

  getMetadataWithDefault(dbname, username, db_meta, user_meta);

  // NOTE(max): register database in Catalog that early to allow ldap
  // and saml create default user and role privileges on databases
  auto cat =
      Catalog::get(basePath_, db_meta, dataMgr_, string_dict_hosts_, calciteMgr_, false);

  DBObject dbObject(dbname, DatabaseDBObjectType);
  dbObject.loadKey();
  dbObject.setPrivileges(AccessPrivileges::ACCESS);
  if (!checkPrivileges(user_meta, std::vector<DBObject>{dbObject})) {
    throw std::runtime_error("Invalid credentials.");
  }

  return cat;
}

void SysCatalog::createUser(const string& name,
                            const string& passwd,
                            bool issuper,
                            const std::string& dbname) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  UserMetadata user;
  if (getMetadataForUser(name, user)) {
    throw runtime_error("User " + name + " already exists.");
  }
  if (getGrantee(name)) {
    throw runtime_error(
        "User name " + name +
        " is same as one of existing grantees. User and role names should be unique.");
  }
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    std::vector<std::string> vals;
    if (!dbname.empty()) {
      DBMetadata db;
      if (!SysCatalog::instance().getMetadataForDB(dbname, db)) {
        throw runtime_error("DEFAULT_DB " + dbname + " not found.");
      }
      vals = {name,
              hash_with_bcrypt(passwd),
              std::to_string(issuper),
              std::to_string(db.dbId)};
      sqliteConnector_->query_with_text_params(
          "INSERT INTO mapd_users (name, passwd_hash, issuper, default_db) VALUES (?, ?, "
          "?, ?)",
          vals);
    } else {
      vals = {name, hash_with_bcrypt(passwd), std::to_string(issuper)};
      sqliteConnector_->query_with_text_params(
          "INSERT INTO mapd_users (name, passwd_hash, issuper) VALUES (?, ?, ?)", vals);
    }

    createRole_unsafe(name, true);
    // TODO: establish rational for this ...
    grantDefaultPrivilegesToRole_unsafe(name, issuper);
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::dropUser(const string& name) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    UserMetadata user;
    if (!getMetadataForUser(name, user)) {
      throw runtime_error("User " + name + " does not exist.");
    }
    dropRole_unsafe(name);
    deleteObjectDescriptorMap(name);
    const std::string& roleName(name);
    sqliteConnector_->query_with_text_param("DELETE FROM mapd_roles WHERE userName = ?",
                                            roleName);
    sqliteConnector_->query("DELETE FROM mapd_users WHERE userid = " +
                            std::to_string(user.userId));
    sqliteConnector_->query("DELETE FROM mapd_privileges WHERE userid = " +
                            std::to_string(user.userId));
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

namespace {  // anonymous namespace

auto append_with_commas = [](string& s, const string& t) {
  if (!s.empty()) {
    s += ", ";
  }
  s += t;
};

}  // anonymous namespace

void SysCatalog::alterUser(const int32_t userid,
                           const string* passwd,
                           bool* issuper,
                           const string* dbname) {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    string sql;
    std::vector<std::string> values;

    if (passwd != nullptr) {
      append_with_commas(sql, "passwd_hash = ?");
      values.push_back(hash_with_bcrypt(*passwd));
    }

    if (issuper != nullptr) {
      append_with_commas(sql, "issuper = ?");
      values.push_back(std::to_string(*issuper));
    }

    if (dbname != nullptr) {
      if (!dbname->empty()) {
        append_with_commas(sql, "default_db = ?");
        DBMetadata db;
        if (!SysCatalog::instance().getMetadataForDB(*dbname, db)) {
          throw runtime_error(string("DEFAULT_DB ") + *dbname + " not found.");
        }
        values.push_back(std::to_string(db.dbId));
      } else {
        append_with_commas(sql, "default_db = NULL");
      }
    }

    sql = "UPDATE mapd_users SET " + sql + " WHERE userid = ?";
    values.push_back(std::to_string(userid));

    sqliteConnector_->query_with_text_params(sql, values);
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::createDatabase(const string& name, int owner) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  DBMetadata db;
  if (getMetadataForDB(name, db)) {
    throw runtime_error("Database " + name + " already exists.");
  }
  if (to_upper(name) == to_upper(MAPD_SYSTEM_CATALOG)) {
    throw runtime_error("Database name " + name + " is reserved.");
  }

  std::unique_ptr<SqliteConnector> dbConn(
      new SqliteConnector(name, basePath_ + "/mapd_catalogs/"));
  // NOTE(max): it's okay to run this in a separate transaction. If we fail later
  // we delete the database anyways.
  // If we run it in the same transaction as SysCatalog functions, then Catalog
  // constructor won't find the tables we have just created.
  dbConn->query("BEGIN TRANSACTION");
  try {
    dbConn->query(
        "CREATE TABLE mapd_tables (tableid integer primary key, name text unique, userid "
        "integer, ncolumns integer, "
        "isview boolean, "
        "fragments text, frag_type integer, max_frag_rows integer, max_chunk_size "
        "bigint, "
        "frag_page_size integer, "
        "max_rows bigint, partitions text, shard_column_id integer, shard integer, "
        "num_shards integer, key_metainfo TEXT, version_num "
        "BIGINT DEFAULT 1) ");
    dbConn->query(
        "CREATE TABLE mapd_columns (tableid integer references mapd_tables, columnid "
        "integer, name text, coltype "
        "integer, colsubtype integer, coldim integer, colscale integer, is_notnull "
        "boolean, compression integer, "
        "comp_param integer, size integer, chunks text, is_systemcol boolean, "
        "is_virtualcol boolean, virtual_expr "
        "text, is_deletedcol boolean, version_num BIGINT, "
        "primary key(tableid, columnid), unique(tableid, name))");
    dbConn->query(
        "CREATE TABLE mapd_views (tableid integer references mapd_tables, sql text)");
    dbConn->query(
        "CREATE TABLE mapd_dashboards (id integer primary key autoincrement, name text , "
        "userid integer references mapd_users, state text, image_hash text, update_time "
        "timestamp, "
        "metadata text, UNIQUE(userid, name) )");
    dbConn->query(
        "CREATE TABLE mapd_links (linkid integer primary key, userid integer references "
        "mapd_users, "
        "link text unique, view_state text, update_time timestamp, view_metadata text)");
    dbConn->query(
        "CREATE TABLE mapd_dictionaries (dictid integer primary key, name text unique, "
        "nbits int, is_shared boolean, "
        "refcount int, version_num BIGINT DEFAULT 1)");
    dbConn->query(
        "CREATE TABLE mapd_logical_to_physical(logical_table_id integer, "
        "physical_table_id "
        "integer)");
    dbConn->query("CREATE TABLE mapd_record_ownership_marker (dummy integer)");
    dbConn->query_with_text_params(
        "INSERT INTO mapd_record_ownership_marker (dummy) VALUES (?1)",
        std::vector<std::string>{std::to_string(owner)});
  } catch (const std::exception&) {
    dbConn->query("ROLLBACK TRANSACTION");
    boost::filesystem::remove(basePath_ + "/mapd_catalogs/" + name);
    throw;
  }
  dbConn->query("END TRANSACTION");

  // Now update SysCatalog with privileges and the new database
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query_with_text_param(
        "INSERT INTO mapd_databases (name, owner) VALUES (?, " + std::to_string(owner) +
            ")",
        name);
    CHECK(getMetadataForDB(name, db));
    auto cat =
        Catalog::get(basePath_, db, dataMgr_, string_dict_hosts_, calciteMgr_, true);
    if (owner != MAPD_ROOT_USER_ID) {
      DBObject object(name, DBObjectType::DatabaseDBObjectType);
      object.loadKey(*cat);
      UserMetadata user;
      CHECK(getMetadataForUserById(owner, user));
      grantAllOnDatabase_unsafe(user.userName, object, *cat);
    }
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    boost::filesystem::remove(basePath_ + "/mapd_catalogs/" + name);
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::dropDatabase(const DBMetadata& db) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  auto cat =
      Catalog::get(basePath_, db, dataMgr_, string_dict_hosts_, calciteMgr_, false);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    // remove this database ID from any users that have it set as their default database
    sqliteConnector_->query_with_text_param(
        "UPDATE mapd_users SET default_db = NULL WHERE default_db = ?",
        std::to_string(db.dbId));
    /* revoke object privileges to all tables of the database being dropped */
    const auto tables = cat->getAllTableMetadata();
    for (const auto table : tables) {
      if (table->shard >= 0) {
        // skip shards, they're not standalone tables
        continue;
      }
      revokeDBObjectPrivilegesFromAll_unsafe(
          DBObject(table->tableName, TableDBObjectType), cat.get());
    }
    const auto dashboards = cat->getAllDashboardsMetadata();
    for (const auto dashboard : dashboards) {
      revokeDBObjectPrivilegesFromAll_unsafe(
          DBObject(dashboard->dashboardId, DashboardDBObjectType), cat.get());
    }
    /* revoke object privileges to the database being dropped */
    for (const auto& grantee : granteeMap_) {
      if (grantee.second->hasAnyPrivilegesOnDb(db.dbId, true)) {
        revokeAllOnDatabase_unsafe(grantee.second->getName(), db.dbId, grantee.second);
      }
    }
    sqliteConnector_->query_with_text_param("DELETE FROM mapd_databases WHERE dbid = ?",
                                            std::to_string(db.dbId));
    cat->eraseDBData();
    Catalog::remove(db.dbName);
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

bool SysCatalog::checkPasswordForUser(const std::string& passwd,
                                      std::string& name,
                                      UserMetadata& user) {
  {
    if (!getMetadataForUser(name, user)) {
      // Check password against some fake hash just to waste time so that response times
      // for invalid password and invalid user are similar and a caller can't say the
      // difference
      char fake_hash[BCRYPT_HASHSIZE];
      CHECK(bcrypt_gensalt(-1, fake_hash) == 0);
      bcrypt_checkpw(passwd.c_str(), fake_hash);
      return false;
    }
    int pwd_check_result = bcrypt_checkpw(passwd.c_str(), user.passwd_hash.c_str());
    // if the check fails there is a good chance that data on disc is broken
    CHECK(pwd_check_result >= 0);
    if (pwd_check_result != 0) {
      return false;
    }
  }
  return true;
}

bool SysCatalog::getMetadataForUser(const string& name, UserMetadata& user) {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_param(
      "SELECT userid, name, passwd_hash, issuper, default_db FROM mapd_users WHERE name "
      "= ?",
      name);
  int numRows = sqliteConnector_->getNumRows();
  if (numRows == 0) {
    return false;
  }
  user.userId = sqliteConnector_->getData<int>(0, 0);
  user.userName = sqliteConnector_->getData<string>(0, 1);
  user.passwd_hash = sqliteConnector_->getData<string>(0, 2);
  user.isSuper = sqliteConnector_->getData<bool>(0, 3);
  user.isReallySuper = user.isSuper;
  user.defaultDbId =
      sqliteConnector_->isNull(0, 4) ? -1 : sqliteConnector_->getData<int>(0, 4);
  return true;
}

bool SysCatalog::getMetadataForUserById(const int32_t idIn, UserMetadata& user) {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_param(
      "SELECT userid, name, passwd_hash, issuper, default_db FROM mapd_users WHERE "
      "userid = ?",
      std::to_string(idIn));
  int numRows = sqliteConnector_->getNumRows();
  if (numRows == 0) {
    return false;
  }
  user.userId = sqliteConnector_->getData<int>(0, 0);
  user.userName = sqliteConnector_->getData<string>(0, 1);
  user.passwd_hash = sqliteConnector_->getData<string>(0, 2);
  user.isSuper = sqliteConnector_->getData<bool>(0, 3);
  user.isReallySuper = user.isSuper;
  user.defaultDbId =
      sqliteConnector_->isNull(0, 4) ? -1 : sqliteConnector_->getData<int>(0, 4);
  return true;
}

list<DBMetadata> SysCatalog::getAllDBMetadata() {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("SELECT dbid, name, owner FROM mapd_databases");
  int numRows = sqliteConnector_->getNumRows();
  list<DBMetadata> db_list;
  for (int r = 0; r < numRows; ++r) {
    DBMetadata db;
    db.dbId = sqliteConnector_->getData<int>(r, 0);
    db.dbName = sqliteConnector_->getData<string>(r, 1);
    db.dbOwner = sqliteConnector_->getData<int>(r, 2);
    db_list.push_back(db);
  }
  return db_list;
}

list<UserMetadata> SysCatalog::getAllUserMetadata(long dbId) {
  // this call is to return users that have some form of permissions to objects in the db
  // sadly mapd_object_permissions table is also misused to manage user roles.
  sys_sqlite_lock sqlite_lock(this);
  std::string sql = "SELECT userid, name, issuper FROM mapd_users";
  if (dbId >= 0) {
    sql =
        "SELECT userid, name, issuper FROM mapd_users WHERE name IN (SELECT roleName "
        "FROM mapd_object_permissions "
        "WHERE "
        "objectPermissions<>0 AND roleType=1 AND dbId=" +
        std::to_string(dbId) + ")";
  }
  sqliteConnector_->query(sql);
  int numRows = sqliteConnector_->getNumRows();
  list<UserMetadata> user_list;
  for (int r = 0; r < numRows; ++r) {
    UserMetadata user;
    user.userId = sqliteConnector_->getData<int>(r, 0);
    user.userName = sqliteConnector_->getData<string>(r, 1);
    user.isSuper = sqliteConnector_->getData<bool>(r, 2);
    user_list.push_back(user);
  }
  return user_list;
}

list<UserMetadata> SysCatalog::getAllUserMetadata() {
  return getAllUserMetadata(-1);
}

void SysCatalog::getMetadataWithDefault(std::string& dbname,
                                        const std::string& username,
                                        Catalog_Namespace::DBMetadata& db_meta,
                                        UserMetadata& user_meta) {
  if (!getMetadataForUser(username, user_meta)) {
    throw std::runtime_error("Invalid credentials.");
  }

  if (!dbname.empty()) {
    if (!getMetadataForDB(dbname, db_meta)) {
      throw std::runtime_error("Database name " + dbname + " does not exist.");
    }
    // loaded the requested database
  } else {
    if (user_meta.defaultDbId != -1) {
      if (!getMetadataForDBById(user_meta.defaultDbId, db_meta)) {
        throw std::runtime_error(
            "Server error: User #" + std::to_string(user_meta.userId) + " " +
            user_meta.userName + " has invalid default_db #" +
            std::to_string(user_meta.defaultDbId) + " which does not exist.");
      }
      dbname = db_meta.dbName;
      // loaded the user's default database
    } else {
      if (!getMetadataForDB(MAPD_DEFAULT_DB, db_meta)) {
        throw std::runtime_error(std::string("Database ") + MAPD_DEFAULT_DB +
                                 " does not exist.");
      }
      dbname = MAPD_DEFAULT_DB;
      // loaded the mapd database by default
    }
  }
}

bool SysCatalog::getMetadataForDB(const string& name, DBMetadata& db) {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_param(
      "SELECT dbid, name, owner FROM mapd_databases WHERE name = ?", name);
  int numRows = sqliteConnector_->getNumRows();
  if (numRows == 0) {
    return false;
  }
  db.dbId = sqliteConnector_->getData<int>(0, 0);
  db.dbName = sqliteConnector_->getData<string>(0, 1);
  db.dbOwner = sqliteConnector_->getData<int>(0, 2);
  return true;
}

bool SysCatalog::getMetadataForDBById(const int32_t idIn, DBMetadata& db) {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_param(
      "SELECT dbid, name, owner FROM mapd_databases WHERE dbid = ?",
      std::to_string(idIn));
  int numRows = sqliteConnector_->getNumRows();
  if (numRows == 0) {
    return false;
  }
  db.dbId = sqliteConnector_->getData<int>(0, 0);
  db.dbName = sqliteConnector_->getData<string>(0, 1);
  db.dbOwner = sqliteConnector_->getData<int>(0, 2);
  return true;
}

DBSummaryList SysCatalog::getDatabaseListForUser(const UserMetadata& user) {
  DBSummaryList ret;

  std::list<Catalog_Namespace::DBMetadata> db_list = getAllDBMetadata();
  std::list<Catalog_Namespace::UserMetadata> user_list = getAllUserMetadata();

  for (auto d : db_list) {
    DBObject dbObject(d.dbName, DatabaseDBObjectType);
    dbObject.loadKey();
    dbObject.setPrivileges(AccessPrivileges::ACCESS);
    if (!checkPrivileges(user, std::vector<DBObject>{dbObject})) {
      continue;
    }
    for (auto u : user_list) {
      if (d.dbOwner == u.userId) {
        ret.emplace_back(DBSummary{d.dbName, u.userName});
        break;
      }
    }
  }

  return ret;
}

// Note (max): I wonder why this one is necessary
void SysCatalog::grantDefaultPrivilegesToRole_unsafe(const std::string& name,
                                                     bool issuper) {
  DBObject dbObject(MAPD_DEFAULT_DB, DatabaseDBObjectType);
  auto* catalog = Catalog::get(MAPD_DEFAULT_DB).get();
  CHECK(catalog);
  dbObject.loadKey(*catalog);

  if (issuper) {
    // dont do this, user is super
    // dbObject.setPrivileges(AccessPrivileges::ALL_DATABASE);
  }

  grantDBObjectPrivileges_unsafe(name, dbObject, *catalog);
}

void SysCatalog::createDBObject(const UserMetadata& user,
                                const std::string& objectName,
                                DBObjectType type,
                                const Catalog_Namespace::Catalog& catalog,
                                int32_t objectId) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  DBObject object =
      objectId == -1 ? DBObject(objectName, type) : DBObject(objectId, type);
  object.loadKey(catalog);
  switch (type) {
    case TableDBObjectType:
      object.setPrivileges(AccessPrivileges::ALL_TABLE);
      break;
    case DashboardDBObjectType:
      object.setPrivileges(AccessPrivileges::ALL_DASHBOARD);
      break;
    default:
      object.setPrivileges(AccessPrivileges::ALL_DATABASE);
      break;
  }
  object.setOwner(user.userId);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    if (user.userName.compare(
            MAPD_ROOT_USER)) {  // no need to grant to suser, has all privs by default
      grantDBObjectPrivileges_unsafe(user.userName, object, catalog);
      auto* grantee = instance().getUserGrantee(user.userName);
      if (!grantee) {
        throw runtime_error("User " + user.userName + "  does not exist.");
      }
      grantee->grantPrivileges(object);
    }
  } catch (std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::grantDBObjectPrivilegesBatch_unsafe(
    const vector<string>& grantees,
    const vector<DBObject>& objects,
    const Catalog_Namespace::Catalog& catalog) {
  for (const auto& grantee : grantees) {
    for (const auto& object : objects) {
      grantDBObjectPrivileges_unsafe(grantee, object, catalog);
    }
  }
}

// GRANT INSERT ON TABLE payroll_table TO payroll_dept_role;
void SysCatalog::grantDBObjectPrivileges_unsafe(
    const std::string& granteeName,
    DBObject object,
    const Catalog_Namespace::Catalog& catalog) {
  object.loadKey(catalog);
  if (object.getPrivileges().hasPermission(DatabasePrivileges::ALL) &&
      object.getObjectKey().permissionType == DatabaseDBObjectType) {
    return grantAllOnDatabase_unsafe(granteeName, object, catalog);
  }

  sys_write_lock write_lock(this);

  if (!granteeName.compare(MAPD_ROOT_USER)) {
    throw runtime_error("Request to grant privileges to " + granteeName +
                        " failed because mapd root user has all privileges by default.");
  }
  auto* grantee = instance().getGrantee(granteeName);
  if (!grantee) {
    throw runtime_error("Request to grant privileges to " + granteeName +
                        " failed because role or user with this name does not exist.");
  }
  grantee->grantPrivileges(object);

  /* apply grant privileges statement to sqlite DB */
  std::vector<std::string> objectKey = object.toString();
  object.resetPrivileges();
  grantee->getPrivileges(object, true);

  sys_sqlite_lock sqlite_lock(this);
  insertOrUpdateObjectPrivileges(
      sqliteConnector_, granteeName, grantee->isUser(), object);
  updateObjectDescriptorMap(granteeName, object, grantee->isUser(), catalog);
}

void SysCatalog::grantAllOnDatabase_unsafe(const std::string& roleName,
                                           DBObject& object,
                                           const Catalog_Namespace::Catalog& catalog) {
  // It's a separate use case because it's easier for implementation to convert ALL ON
  // DATABASE into ALL ON DASHBOARDS, ALL ON VIEWS and ALL ON TABLES
  // Add DB Access privileges
  DBObject tmp_object = object;
  tmp_object.setPrivileges(AccessPrivileges::ACCESS);
  tmp_object.setPermissionType(DatabaseDBObjectType);
  grantDBObjectPrivileges_unsafe(roleName, tmp_object, catalog);
  tmp_object.setPrivileges(AccessPrivileges::VIEW_SQL_EDITOR);
  tmp_object.setPermissionType(DatabaseDBObjectType);
  grantDBObjectPrivileges_unsafe(roleName, tmp_object, catalog);
  tmp_object.setPrivileges(AccessPrivileges::ALL_TABLE);
  tmp_object.setPermissionType(TableDBObjectType);
  grantDBObjectPrivileges_unsafe(roleName, tmp_object, catalog);
  tmp_object.setPrivileges(AccessPrivileges::ALL_VIEW);
  tmp_object.setPermissionType(ViewDBObjectType);
  grantDBObjectPrivileges_unsafe(roleName, tmp_object, catalog);
  tmp_object.setPrivileges(AccessPrivileges::ALL_DASHBOARD);
  tmp_object.setPermissionType(DashboardDBObjectType);
  grantDBObjectPrivileges_unsafe(roleName, tmp_object, catalog);
  return;
}

void SysCatalog::revokeDBObjectPrivilegesBatch_unsafe(
    const vector<string>& grantees,
    const vector<DBObject>& objects,
    const Catalog_Namespace::Catalog& catalog) {
  for (const auto& grantee : grantees) {
    for (const auto& object : objects) {
      revokeDBObjectPrivileges_unsafe(grantee, object, catalog);
    }
  }
}

// REVOKE INSERT ON TABLE payroll_table FROM payroll_dept_role;
void SysCatalog::revokeDBObjectPrivileges_unsafe(
    const std::string& granteeName,
    DBObject object,
    const Catalog_Namespace::Catalog& catalog) {
  sys_write_lock write_lock(this);

  if (!granteeName.compare(MAPD_ROOT_USER)) {
    throw runtime_error(
        "Request to revoke privileges from " + granteeName +
        " failed because privileges can not be revoked from mapd root user.");
  }
  auto* grantee = getGrantee(granteeName);
  if (!grantee) {
    throw runtime_error("Request to revoke privileges from " + granteeName +
                        " failed because role or user with this name does not exist.");
  }
  object.loadKey(catalog);

  if (object.getPrivileges().hasPermission(DatabasePrivileges::ALL) &&
      object.getObjectKey().permissionType == DatabaseDBObjectType) {
    return revokeAllOnDatabase_unsafe(granteeName, object.getObjectKey().dbId, grantee);
  }

  auto ret_object = grantee->revokePrivileges(object);
  if (ret_object) {
    sys_sqlite_lock sqlite_lock(this);
    insertOrUpdateObjectPrivileges(
        sqliteConnector_, granteeName, grantee->isUser(), *ret_object);
    updateObjectDescriptorMap(granteeName, *ret_object, grantee->isUser(), catalog);
  } else {
    sys_sqlite_lock sqlite_lock(this);
    deleteObjectPrivileges(sqliteConnector_, granteeName, grantee->isUser(), object);
    deleteObjectDescriptorMap(granteeName, object, catalog);
  }
}

void SysCatalog::revokeAllOnDatabase_unsafe(const std::string& roleName,
                                            int32_t dbId,
                                            Grantee* grantee) {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_params(
      "DELETE FROM mapd_object_permissions WHERE roleName = ?1 and dbId = ?2",
      std::vector<std::string>{roleName, std::to_string(dbId)});
  grantee->revokeAllOnDatabase(dbId);
  for (auto d = objectDescriptorMap_.begin(); d != objectDescriptorMap_.end();) {
    if (d->second->roleName == roleName && d->second->dbId == dbId) {
      delete d->second;
      d = objectDescriptorMap_.erase(d);
    } else {
      d++;
    }
  }
}

void SysCatalog::revokeDBObjectPrivilegesFromAll_unsafe(DBObject dbObject,
                                                        Catalog* catalog) {
  sys_write_lock write_lock(this);
  dbObject.loadKey(*catalog);
  auto privs = (dbObject.getObjectKey().permissionType == TableDBObjectType)
                   ? AccessPrivileges::ALL_TABLE
                   : (dbObject.getObjectKey().permissionType == DashboardDBObjectType)
                         ? AccessPrivileges::ALL_DASHBOARD
                         : AccessPrivileges::ALL_TABLE;
  dbObject.setPrivileges(privs);
  for (const auto& grantee : granteeMap_) {
    if (grantee.second->findDbObject(dbObject.getObjectKey(), true)) {
      revokeDBObjectPrivileges_unsafe(grantee.second->getName(), dbObject, *catalog);
    }
  }
}

bool SysCatalog::verifyDBObjectOwnership(const UserMetadata& user,
                                         DBObject object,
                                         const Catalog_Namespace::Catalog& catalog) {
  sys_read_lock read_lock(this);

  auto* grantee = instance().getUserGrantee(user.userName);
  if (grantee) {
    object.loadKey(catalog);
    auto* found_object = grantee->findDbObject(object.getObjectKey(), false);
    if (found_object && found_object->getOwner() == user.userId) {
      return true;
    }
  }
  return false;
}

void SysCatalog::getDBObjectPrivileges(const std::string& granteeName,
                                       DBObject& object,
                                       const Catalog_Namespace::Catalog& catalog) const {
  sys_read_lock read_lock(this);
  if (!granteeName.compare(MAPD_ROOT_USER)) {
    throw runtime_error("Request to show privileges from " + granteeName +
                        " failed because mapd root user has all privileges by default.");
  }
  auto* grantee = instance().getGrantee(granteeName);
  if (!grantee) {
    throw runtime_error("Request to show privileges for " + granteeName +
                        " failed because role or user with this name does not exist.");
  }
  object.loadKey(catalog);
  grantee->getPrivileges(object, true);
}

void SysCatalog::createRole_unsafe(const std::string& roleName,
                                   const bool& userPrivateRole) {
  sys_write_lock write_lock(this);

  auto* grantee = getGrantee(roleName);
  if (grantee) {
    throw std::runtime_error("CREATE ROLE " + roleName +
                             " failed because grantee with this name already exists.");
  }
  if (userPrivateRole) {
    grantee = new User(roleName);
  } else {
    grantee = new Role(roleName);
  }
  granteeMap_[to_upper(roleName)] = grantee;

  // NOTE (max): Why create an empty privileges record for a role?
  /* grant none privileges to this role and add it to sqlite DB */
  DBObject dbObject(MAPD_DEFAULT_DB, DatabaseDBObjectType);
  DBObjectKey objKey;
  // 0 is an id that does not exist
  objKey.dbId = 0;
  objKey.permissionType = DatabaseDBObjectType;
  dbObject.setObjectKey(objKey);
  grantee->grantPrivileges(dbObject);

  sys_sqlite_lock sqlite_lock(this);
  insertOrUpdateObjectPrivileges(sqliteConnector_, roleName, userPrivateRole, dbObject);
}

void SysCatalog::dropRole_unsafe(const std::string& roleName) {
  sys_write_lock write_lock(this);

  // it may very well be a user "role", so keep it generic
  auto* rl = getGrantee(roleName);
  CHECK(rl);  // it has been checked already in the calling proc that this role exists,
              // faiul otherwise
  delete rl;
  granteeMap_.erase(to_upper(roleName));

  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_param("DELETE FROM mapd_roles WHERE roleName = ?",
                                          roleName);
  sqliteConnector_->query_with_text_param(
      "DELETE FROM mapd_object_permissions WHERE roleName = ?", roleName);
}

void SysCatalog::grantRoleBatch_unsafe(const std::vector<std::string>& roles,
                                       const std::vector<std::string>& grantees) {
  for (const auto& role : roles) {
    for (const auto& grantee : grantees) {
      grantRole_unsafe(role, grantee);
    }
  }
}

// GRANT ROLE payroll_dept_role TO joe;
void SysCatalog::grantRole_unsafe(const std::string& roleName,
                                  const std::string& granteeName) {
  auto* rl = getRoleGrantee(roleName);
  if (!rl) {
    throw runtime_error("Request to grant role " + roleName +
                        " failed because role with this name does not exist.");
  }
  auto* grantee = getGrantee(granteeName);
  if (!grantee) {
    throw runtime_error("Request to grant role " + roleName + " failed because grantee " +
                        granteeName + " does not exist.");
  }
  sys_write_lock write_lock(this);
  if (!grantee->hasRole(rl, true)) {
    grantee->grantRole(rl);
    sys_sqlite_lock sqlite_lock(this);
    sqliteConnector_->query_with_text_params(
        "INSERT INTO mapd_roles(roleName, userName) VALUES (?, ?)",
        std::vector<std::string>{roleName, granteeName});
  }
}

void SysCatalog::revokeRoleBatch_unsafe(const std::vector<std::string>& roles,
                                        const std::vector<std::string>& grantees) {
  for (const auto& role : roles) {
    for (const auto& grantee : grantees) {
      revokeRole_unsafe(role, grantee);
    }
  }
}

// REVOKE ROLE payroll_dept_role FROM joe;
void SysCatalog::revokeRole_unsafe(const std::string& roleName,
                                   const std::string& granteeName) {
  auto* rl = getRoleGrantee(roleName);
  if (!rl) {
    throw runtime_error("Request to revoke role " + roleName +
                        " failed because role with this name does not exist.");
  }
  auto* grantee = getGrantee(granteeName);
  if (!grantee) {
    throw runtime_error("Request to revoke role from " + granteeName +
                        " failed because grantee with this name does not exist.");
  }
  sys_write_lock write_lock(this);
  grantee->revokeRole(rl);
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_params(
      "DELETE FROM mapd_roles WHERE roleName = ? AND userName = ?",
      std::vector<std::string>{roleName, granteeName});
}

// Update or add element in ObjectRoleDescriptorMap
void SysCatalog::updateObjectDescriptorMap(const std::string& roleName,
                                           DBObject& object,
                                           bool roleType,
                                           const Catalog_Namespace::Catalog& cat) {
  bool present = false;
  auto privs = object.getPrivileges();
  sys_write_lock write_lock(this);
  auto range = objectDescriptorMap_.equal_range(
      std::to_string(cat.getCurrentDB().dbId) + ":" +
      std::to_string(object.getObjectKey().permissionType) + ":" +
      std::to_string(object.getObjectKey().objectId));
  for (auto d = range.first; d != range.second; ++d) {
    if (d->second->roleName == roleName) {
      // overwrite permissions
      d->second->privs = privs;
      present = true;
    }
  }
  if (!present) {
    ObjectRoleDescriptor* od = new ObjectRoleDescriptor();
    od->roleName = roleName;
    od->roleType = roleType;
    od->objectType = object.getObjectKey().permissionType;
    od->dbId = object.getObjectKey().dbId;
    od->objectId = object.getObjectKey().objectId;
    od->privs = object.getPrivileges();
    od->objectOwnerId = object.getOwner();
    od->objectName = object.getName();
    objectDescriptorMap_.insert(ObjectRoleDescriptorMap::value_type(
        std::to_string(od->dbId) + ":" + std::to_string(od->objectType) + ":" +
            std::to_string(od->objectId),
        od));
  }
}

// rename object descriptors
void SysCatalog::renameObjectsInDescriptorMap(DBObject& object,
                                              const Catalog_Namespace::Catalog& cat) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  auto range = objectDescriptorMap_.equal_range(
      std::to_string(cat.getCurrentDB().dbId) + ":" +
      std::to_string(object.getObjectKey().permissionType) + ":" +
      std::to_string(object.getObjectKey().objectId));
  for (auto d = range.first; d != range.second; ++d) {
    // rename object
    d->second->objectName = object.getName();
  }

  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query_with_text_params(
        "UPDATE mapd_object_permissions SET objectName = ?1 WHERE "
        "dbId = ?2 AND objectId = ?3",
        std::vector<std::string>{object.getName(),
                                 std::to_string(cat.getCurrentDB().dbId),
                                 std::to_string(object.getObjectKey().objectId)});
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

// Remove user/role from ObjectRoleDescriptorMap
void SysCatalog::deleteObjectDescriptorMap(const std::string& roleName) {
  sys_write_lock write_lock(this);

  for (auto d = objectDescriptorMap_.begin(); d != objectDescriptorMap_.end();) {
    if (d->second->roleName == roleName) {
      delete d->second;
      d = objectDescriptorMap_.erase(d);
    } else {
      d++;
    }
  }
}

// Remove element from ObjectRoleDescriptorMap
void SysCatalog::deleteObjectDescriptorMap(const std::string& roleName,
                                           DBObject& object,
                                           const Catalog_Namespace::Catalog& cat) {
  sys_write_lock write_lock(this);
  auto range = objectDescriptorMap_.equal_range(
      std::to_string(cat.getCurrentDB().dbId) + ":" +
      std::to_string(object.getObjectKey().permissionType) + ":" +
      std::to_string(object.getObjectKey().objectId));
  for (auto d = range.first; d != range.second;) {
    // remove the entry
    if (d->second->roleName == roleName) {
      delete d->second;
      d = objectDescriptorMap_.erase(d);
    } else {
      d++;
    }
  }
}

bool SysCatalog::hasAnyPrivileges(const UserMetadata& user,
                                  std::vector<DBObject>& privObjects) {
  sys_read_lock read_lock(this);
  if (user.isSuper) {
    return true;
  }
  auto* user_rl = instance().getUserGrantee(user.userName);
  if (!user_rl) {
    throw runtime_error("User " + user.userName + "  does not exist.");
  }
  for (std::vector<DBObject>::iterator objectIt = privObjects.begin();
       objectIt != privObjects.end();
       ++objectIt) {
    if (!user_rl->hasAnyPrivileges(*objectIt, false)) {
      return false;
    }
  }
  return true;
}

bool SysCatalog::checkPrivileges(const UserMetadata& user,
                                 const std::vector<DBObject>& privObjects) const {
  sys_read_lock read_lock(this);
  if (user.isSuper) {
    return true;
  }

  auto* user_rl = instance().getUserGrantee(user.userName);
  if (!user_rl) {
    throw runtime_error("User " + user.userName + "  does not exist.");
  }
  for (auto& object : privObjects) {
    if (!user_rl->checkPrivileges(object)) {
      return false;
    }
  }
  return true;
}

bool SysCatalog::checkPrivileges(const std::string& userName,
                                 const std::vector<DBObject>& privObjects) const {
  UserMetadata user;
  if (!instance().getMetadataForUser(userName, user)) {
    throw runtime_error("Request to check privileges for user " + userName +
                        " failed because user with this name does not exist.");
  }
  return (checkPrivileges(user, privObjects));
}

Grantee* SysCatalog::getGrantee(const std::string& name) const {
  sys_read_lock read_lock(this);
  auto grantee = granteeMap_.find(to_upper(name));
  if (grantee == granteeMap_.end()) {  // check to make sure role exists
    return nullptr;
  }
  return grantee->second;  // returns pointer to role
}

Role* SysCatalog::getRoleGrantee(const std::string& name) const {
  return dynamic_cast<Role*>(getGrantee(name));
}

User* SysCatalog::getUserGrantee(const std::string& name) const {
  return dynamic_cast<User*>(getGrantee(name));
}

std::vector<ObjectRoleDescriptor*>
SysCatalog::getMetadataForObject(int32_t dbId, int32_t dbType, int32_t objectId) const {
  sys_read_lock read_lock(this);
  std::vector<ObjectRoleDescriptor*> objectsList;

  auto range = objectDescriptorMap_.equal_range(std::to_string(dbId) + ":" +
                                                std::to_string(dbType) + ":" +
                                                std::to_string(objectId));
  for (auto d = range.first; d != range.second; ++d) {
    objectsList.push_back(d->second);
  }
  return objectsList;  // return pointers to objects
}

bool SysCatalog::isRoleGrantedToGrantee(const std::string& granteeName,
                                        const std::string& roleName,
                                        bool only_direct) const {
  sys_read_lock read_lock(this);
  if (roleName == granteeName) {
    return true;
  }
  bool rc = false;
  auto* user_rl = instance().getUserGrantee(granteeName);
  if (user_rl) {
    auto* rl = instance().getRoleGrantee(roleName);
    if (rl && user_rl->hasRole(rl, only_direct)) {
      rc = true;
    }
  }
  return rc;
}

bool SysCatalog::isDashboardSystemRole(const std::string& roleName) {
  return boost::algorithm::ends_with(roleName, SYSTEM_ROLE_TAG);
}

std::vector<std::string> SysCatalog::getRoles(const std::string& userName,
                                              const int32_t dbId) {
  sys_sqlite_lock sqlite_lock(this);
  std::string sql =
      "SELECT DISTINCT roleName FROM mapd_object_permissions WHERE objectPermissions<>0 "
      "AND roleType=0 AND dbId=" +
      std::to_string(dbId);
  sqliteConnector_->query(sql);
  int numRows = sqliteConnector_->getNumRows();
  std::vector<std::string> roles(0);
  for (int r = 0; r < numRows; ++r) {
    auto roleName = sqliteConnector_->getData<string>(r, 0);
    if (isRoleGrantedToGrantee(userName, roleName, false) &&
        !isDashboardSystemRole(roleName)) {
      roles.push_back(roleName);
    }
  }
  return roles;
}

std::vector<std::string> SysCatalog::getRoles(bool userPrivateRole,
                                              bool isSuper,
                                              const std::string& userName) {
  sys_read_lock read_lock(this);
  std::vector<std::string> roles;
  for (auto& grantee : granteeMap_) {
    if (!userPrivateRole && grantee.second->isUser()) {
      continue;
    }
    if (!isSuper && !isRoleGrantedToGrantee(userName, grantee.second->getName(), false)) {
      continue;
    }
    if (isDashboardSystemRole(grantee.second->getName())) {
      continue;
    }
    roles.push_back(grantee.second->getName());
  }
  return roles;
}

void SysCatalog::revokeDashboardSystemRole(const std::string roleName,
                                           const std::vector<std::string> grantees) {
  auto* rl = getRoleGrantee(roleName);
  for (auto granteeName : grantees) {
    const auto* grantee = SysCatalog::instance().getGrantee(granteeName);
    if (rl && grantee->hasRole(rl, true)) {
      // Grantees existence have been already validated
      SysCatalog::instance().revokeRole(roleName, granteeName);
    }
  }
}

void SysCatalog::buildRoleMap() {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  string roleQuery(
      "SELECT roleName, roleType, objectPermissionsType, dbId, objectId, "
      "objectPermissions, objectOwnerId, objectName "
      "from mapd_object_permissions");
  sqliteConnector_->query(roleQuery);
  size_t numRows = sqliteConnector_->getNumRows();
  std::vector<std::string> objectKeyStr(4);
  DBObjectKey objectKey;
  AccessPrivileges privs;
  bool userPrivateRole(false);
  for (size_t r = 0; r < numRows; ++r) {
    std::string roleName = sqliteConnector_->getData<string>(r, 0);
    userPrivateRole = sqliteConnector_->getData<bool>(r, 1);
    DBObjectType permissionType =
        static_cast<DBObjectType>(sqliteConnector_->getData<int>(r, 2));
    objectKeyStr[0] = sqliteConnector_->getData<string>(r, 2);
    objectKeyStr[1] = sqliteConnector_->getData<string>(r, 3);
    objectKeyStr[2] = sqliteConnector_->getData<string>(r, 4);
    objectKey = DBObjectKey::fromString(objectKeyStr, permissionType);
    privs.privileges = sqliteConnector_->getData<int>(r, 5);
    int32_t owner = sqliteConnector_->getData<int>(r, 6);
    std::string name = sqliteConnector_->getData<string>(r, 7);

    DBObject dbObject(objectKey, privs, owner);
    dbObject.setName(name);
    if (-1 == objectKey.objectId) {
      dbObject.setObjectType(DBObjectType::DatabaseDBObjectType);
    } else {
      dbObject.setObjectType(permissionType);
    }

    auto* rl = getGrantee(roleName);
    if (!rl) {
      if (userPrivateRole) {
        rl = new User(roleName);
      } else {
        rl = new Role(roleName);
      }
      granteeMap_[to_upper(roleName)] = rl;
    }
    rl->grantPrivileges(dbObject);
  }
}

void SysCatalog::populateRoleDbObjects(const std::vector<DBObject>& objects) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    for (auto dbobject : objects) {
      UserMetadata user;
      CHECK(getMetadataForUserById(dbobject.getOwner(), user));
      auto* grantee = getUserGrantee(user.userName);
      if (grantee) {
        insertOrUpdateObjectPrivileges(
            sqliteConnector_, grantee->getName(), true, dbobject);
        grantee->grantPrivileges(dbobject);
      }
    }

  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::buildUserRoleMap() {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  std::vector<std::pair<std::string, std::string>> granteeRooles;
  string userRoleQuery("SELECT roleName, userName from mapd_roles");
  sqliteConnector_->query(userRoleQuery);
  size_t numRows = sqliteConnector_->getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    std::string roleName = sqliteConnector_->getData<string>(r, 0);
    std::string userName = sqliteConnector_->getData<string>(r, 1);
    // required for declared nomenclature before v4.0.0
    if ((boost::equals(roleName, "mapd_default_suser_role") &&
         boost::equals(userName, MAPD_ROOT_USER)) ||
        (boost::equals(roleName, "mapd_default_user_role") &&
         !boost::equals(userName, "mapd_default_user_role"))) {
      // grouprole already exists with roleName==userName in mapd_roles table
      // ignore duplicate instances of userRole which exists before v4.0.0
      continue;
    }
    auto* rl = getGrantee(roleName);
    if (!rl) {
      throw runtime_error("Data inconsistency when building role map. Role " + roleName +
                          " not found in the map.");
    }
    std::pair<std::string, std::string> roleVecElem(roleName, userName);
    granteeRooles.push_back(roleVecElem);
  }

  for (size_t i = 0; i < granteeRooles.size(); i++) {
    std::string roleName = granteeRooles[i].first;
    std::string granteeName = granteeRooles[i].second;
    auto* grantee = getGrantee(granteeName);
    if (!grantee) {
      throw runtime_error("Data inconsistency when building role map. Grantee " +
                          granteeName + " not found in the map.");
    }
    if (granteeName == roleName) {
      continue;
    }
    Role* rl = dynamic_cast<Role*>(getGrantee(roleName));
    if (!rl) {
      throw runtime_error("Data inconsistency when building role map. Role " + roleName +
                          " not found in the map.");
    }
    grantee->grantRole(rl);
  }
}

void SysCatalog::buildObjectDescriptorMap() {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  string objectQuery(
      "SELECT roleName, roleType, objectPermissionsType, dbId, objectId, "
      "objectPermissions, objectOwnerId, objectName "
      "from mapd_object_permissions");
  sqliteConnector_->query(objectQuery);
  size_t numRows = sqliteConnector_->getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    ObjectRoleDescriptor* od = new ObjectRoleDescriptor();
    od->roleName = sqliteConnector_->getData<string>(r, 0);
    od->roleType = sqliteConnector_->getData<bool>(r, 1);
    od->objectType = sqliteConnector_->getData<int>(r, 2);
    od->dbId = sqliteConnector_->getData<int>(r, 3);
    od->objectId = sqliteConnector_->getData<int>(r, 4);
    od->privs.privileges = sqliteConnector_->getData<int>(r, 5);
    od->objectOwnerId = sqliteConnector_->getData<int>(r, 6);
    od->objectName = sqliteConnector_->getData<string>(r, 7);
    objectDescriptorMap_.insert(ObjectRoleDescriptorMap::value_type(
        std::to_string(od->dbId) + ":" + std::to_string(od->objectType) + ":" +
            std::to_string(od->objectId),
        od));
  }
}

template <typename F, typename... Args>
void SysCatalog::execInTransaction(F&& f, Args&&... args) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    (this->*f)(std::forward<Args>(args)...);
  } catch (std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::createRole(const std::string& roleName, const bool& userPrivateRole) {
  execInTransaction(&SysCatalog::createRole_unsafe, roleName, userPrivateRole);
}

void SysCatalog::dropRole(const std::string& roleName) {
  execInTransaction(&SysCatalog::dropRole_unsafe, roleName);
}

void SysCatalog::grantRoleBatch(const std::vector<std::string>& roles,
                                const std::vector<std::string>& grantees) {
  execInTransaction(&SysCatalog::grantRoleBatch_unsafe, roles, grantees);
}

void SysCatalog::grantRole(const std::string& role, const std::string& grantee) {
  execInTransaction(&SysCatalog::grantRole_unsafe, role, grantee);
}

void SysCatalog::revokeRoleBatch(const std::vector<std::string>& roles,
                                 const std::vector<std::string>& grantees) {
  execInTransaction(&SysCatalog::revokeRoleBatch_unsafe, roles, grantees);
}

void SysCatalog::revokeRole(const std::string& role, const std::string& grantee) {
  execInTransaction(&SysCatalog::revokeRole_unsafe, role, grantee);
}

void SysCatalog::grantDBObjectPrivileges(const string& grantee,
                                         const DBObject& object,
                                         const Catalog_Namespace::Catalog& catalog) {
  execInTransaction(
      &SysCatalog::grantDBObjectPrivileges_unsafe, grantee, object, catalog);
}

void SysCatalog::grantDBObjectPrivilegesBatch(const vector<string>& grantees,
                                              const vector<DBObject>& objects,
                                              const Catalog_Namespace::Catalog& catalog) {
  execInTransaction(
      &SysCatalog::grantDBObjectPrivilegesBatch_unsafe, grantees, objects, catalog);
}

void SysCatalog::revokeDBObjectPrivileges(const string& grantee,
                                          const DBObject& object,
                                          const Catalog_Namespace::Catalog& catalog) {
  execInTransaction(
      &SysCatalog::revokeDBObjectPrivileges_unsafe, grantee, object, catalog);
}

void SysCatalog::revokeDBObjectPrivilegesBatch(
    const vector<string>& grantees,
    const vector<DBObject>& objects,
    const Catalog_Namespace::Catalog& catalog) {
  execInTransaction(
      &SysCatalog::revokeDBObjectPrivilegesBatch_unsafe, grantees, objects, catalog);
}

void SysCatalog::revokeDBObjectPrivilegesFromAll(DBObject object, Catalog* catalog) {
  execInTransaction(&SysCatalog::revokeDBObjectPrivilegesFromAll_unsafe, object, catalog);
}

void SysCatalog::syncUserWithRemoteProvider(const std::string& user_name,
                                            const std::vector<std::string>& roles,
                                            bool* is_super) {
  UserMetadata user_meta;
  if (!getMetadataForUser(user_name, user_meta)) {
    createUser(user_name, generate_random_string(15), is_super ? *is_super : false, "");
  } else if (is_super && *is_super != user_meta.isSuper) {
    alterUser(user_meta.userId, nullptr, is_super, nullptr);
  }
  std::vector<std::string> current_roles = {};
  auto* user_rl = getUserGrantee(user_name);
  if (user_rl) {
    current_roles = user_rl->getRoles();
  }
  // first remove obsolete ones
  for (auto& current_role_name : current_roles) {
    if (current_role_name != user_name) {
      if (std::find(roles.begin(), roles.end(), current_role_name) == roles.end()) {
        revokeRole(current_role_name, user_name);
      }
    }
  }
  // now re-add them
  std::stringstream ss;
  ss << "Roles synchronized for user " << user_name << ": ";
  for (auto& role_name : roles) {
    auto* rl = getRoleGrantee(role_name);
    if (rl) {
      grantRole(role_name, user_name);
      ss << role_name << " ";
    } else {
      LOG(WARNING) << "Error synchronizing roles for user " << user_name << ": role "
                   << role_name << " does not exist.";
    }
  }
  LOG(INFO) << ss.str();
}

std::unordered_map<std::string, std::vector<std::string>>
SysCatalog::getGranteesOfSharedDashboards(const std::vector<std::string>& dashboard_ids) {
  sys_sqlite_lock sqlite_lock(this);
  std::unordered_map<std::string, std::vector<std::string>> active_grantees;
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    for (auto dash : dashboard_ids) {
      std::vector<std::string> grantees = {};
      sqliteConnector_->query_with_text_params(
          "SELECT roleName FROM mapd_object_permissions WHERE objectPermissions NOT IN "
          "(0,1) AND objectPermissionsType = ? AND objectId = ?",
          std::vector<std::string>{
              std::to_string(static_cast<int32_t>(DashboardDBObjectType)), dash});
      int num_rows = sqliteConnector_->getNumRows();
      if (num_rows == 0) {
        // no grantees
        continue;
      } else {
        for (size_t i = 0; i < sqliteConnector_->getNumRows(); ++i) {
          grantees.push_back(sqliteConnector_->getData<string>(i, 0));
        }
        active_grantees[dash] = grantees;
      }
    }
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
  return active_grantees;
}

}  // namespace Catalog_Namespace
