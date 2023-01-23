/*
 * Copyright 2022 HEAVY.AI, Inc.
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
 * @brief		Functions for System Catalog
 *
 */

#include "SysCatalog.h"
#include <algorithm>
#include <cassert>
#include <exception>
#include <filesystem>
#include <list>
#include <memory>
#include <random>
#include <sstream>
#include <string_view>
#include "Catalog.h"

#include "Catalog/AuthMetadata.h"
#include "QueryEngine/ExternalCacheInvalidators.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/adaptor/map.hpp>
#include <boost/version.hpp>

#include "MapDRelease.h"
#include "Parser/ParserNode.h"
#include "RWLocks.h"
#include "Shared/File.h"
#include "Shared/StringTransform.h"
#include "Shared/SysDefinitions.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "include/bcrypt.h"

using std::list;
using std::map;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;

using namespace std::string_literals;

std::string g_base_path;
bool g_enable_idp_temporary_users{true};
bool g_enable_system_tables{true};

extern bool g_enable_fsi;
extern bool g_read_only;

namespace {

std::string hash_with_bcrypt(const std::string& pwd) {
  char salt[BCRYPT_HASHSIZE], hash[BCRYPT_HASHSIZE];
  CHECK(bcrypt_gensalt(-1, salt) == 0);
  CHECK(bcrypt_hashpw(pwd.c_str(), salt, hash) == 0);
  return std::string(hash, BCRYPT_HASHSIZE);
}

// This catalog copy must take place before any other catalog accesses.
std::filesystem::path copy_catalog_if_read_only(std::filesystem::path base_data_path) {
  std::filesystem::path catalog_base_data_path;

  // For a read-only server, make a temporary copy of the catalog directory.
  // This catalog copy must take place before any other catalog accesses.
  if (!g_read_only) {
    // Catalog directory will be in the normal location.
    catalog_base_data_path = base_data_path;
  } else {
    // Catalog directory will be in a temporary location.
    catalog_base_data_path = base_data_path / "temporary";

    // Delete the temporary directory if it exists.
    // The name "temporary" is hardcoded so nobody should object to its deletion!
    CHECK_NE(catalog_base_data_path.string().find("temporary"), std::string::npos);
    CHECK_NE(catalog_base_data_path, base_data_path);
    if (std::filesystem::exists(catalog_base_data_path)) {
      std::filesystem::remove_all(catalog_base_data_path);
    }
    std::filesystem::create_directories(catalog_base_data_path);

    // Make the temporary copy of the catalog.
    const auto normal_catalog_path = base_data_path / shared::kCatalogDirectoryName;
    const auto temporary_catalog_path =
        catalog_base_data_path / shared::kCatalogDirectoryName;
    LOG(INFO) << "copying catalog from " << normal_catalog_path << " to "
              << temporary_catalog_path << " for read-only server";
    std::filesystem::copy(normal_catalog_path,
                          temporary_catalog_path,
                          std::filesystem::copy_options::recursive);

    // Create a temporary empty directory structure similar to how initheavy would.
    // Not expected to be used. Created just in case any code tries to access them.
    try {
      std::filesystem::create_directories(catalog_base_data_path /
                                          shared::kDataDirectoryName);
    } catch (...) {
    }
    try {
      std::filesystem::create_directories(catalog_base_data_path /
                                          shared::kDefaultExportDirName);
    } catch (...) {
    }
    try {
      std::filesystem::create_directories(catalog_base_data_path /
                                          shared::kLockfilesDirectoryName);
    } catch (...) {
    }
    try {
      std::filesystem::create_directories(catalog_base_data_path /
                                          shared::kLockfilesDirectoryName /
                                          shared::kCatalogDirectoryName);
    } catch (...) {
    }
    try {
      std::filesystem::create_directories(catalog_base_data_path /
                                          shared::kLockfilesDirectoryName /
                                          shared::kDataDirectoryName);
    } catch (...) {
    }
  }

  return catalog_base_data_path;
}

}  // namespace

namespace Catalog_Namespace {

thread_local bool SysCatalog::thread_holds_read_lock = false;
std::mutex SysCatalog::instance_mutex_;
std::unique_ptr<SysCatalog> SysCatalog::instance_;

using sys_read_lock = read_lock<SysCatalog>;
using sys_write_lock = write_lock<SysCatalog>;
using sys_sqlite_lock = sqlite_lock<SysCatalog>;

bool g_log_user_id{false};  // --log-user-id

std::string UserMetadata::userLoggable() const {
  return g_log_user_id ? std::to_string(userId) : userName;
}

auto CommonFileOperations::assembleCatalogName(std::string const& name) {
  return base_path_ + "/" + shared::kCatalogDirectoryName + "/" + name;
};

void CommonFileOperations::removeCatalogByFullPath(std::string const& full_path) {
  boost::filesystem::remove(full_path);
}

void CommonFileOperations::removeCatalogByName(std::string const& name) {
  boost::filesystem::remove(assembleCatalogName(name));
};

auto CommonFileOperations::duplicateAndRenameCatalog(std::string const& current_name,
                                                     std::string const& new_name) {
  auto full_current_path = assembleCatalogName(current_name);
  auto full_new_path = assembleCatalogName(new_name);

  try {
    boost::filesystem::copy_file(full_current_path, full_new_path);
  } catch (std::exception& e) {
    std::string err_message{"Could not copy file " + full_current_path + " to " +
                            full_new_path + " exception was " + e.what()};
    LOG(ERROR) << err_message;
    throw std::runtime_error(err_message);
  }

  return std::make_pair(full_current_path, full_new_path);
};

void SysCatalog::init(const std::string& basePath,
                      std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                      const AuthMetadata& authMetadata,
                      std::shared_ptr<Calcite> calcite,
                      bool is_new_db,
                      bool aggregator,
                      const std::vector<LeafHostInfo>& string_dict_hosts) {
  basePath_ = !g_multi_instance ? copy_catalog_if_read_only(basePath).string() : basePath;
  sqliteConnector_.reset(new SqliteConnector(
      shared::kSystemCatalogName, basePath_ + "/" + shared::kCatalogDirectoryName + "/"));
  dcatalogMutex_ = std::make_unique<heavyai::DistributedSharedMutex>(
      std::filesystem::path(basePath_) / shared::kLockfilesDirectoryName /
          shared::kCatalogDirectoryName / (shared::kSystemCatalogName + ".lockfile"),
      [this](size_t) {
        heavyai::unique_lock<heavyai::DistributedSharedMutex> dsqlite_lock(
            *dsqliteMutex_);
        buildMapsUnlocked();
      });
  dsqliteMutex_ = std::make_unique<heavyai::DistributedSharedMutex>(
      std::filesystem::path(basePath_) / shared::kLockfilesDirectoryName /
      shared::kCatalogDirectoryName / (shared::kSystemCatalogName + ".sqlite.lockfile"));
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  dataMgr_ = dataMgr;
  authMetadata_ = &authMetadata;
  pki_server_.reset(new PkiServer(*authMetadata_));
  calciteMgr_ = calcite;
  string_dict_hosts_ = string_dict_hosts;
  aggregator_ = aggregator;
  migrations::MigrationMgr::takeMigrationLock(basePath_);
  if (is_new_db) {
    initDB();
  } else {
    bool db_exists =
        boost::filesystem::exists(basePath_ + "/" + shared::kCatalogDirectoryName + "/" +
                                  shared::kSystemCatalogName);
    if (!db_exists) {
      importDataFromOldMapdDB();
    }
    if (migrations::MigrationMgr::migrationEnabled()) {
      checkAndExecuteMigrations();
      migrations::MigrationMgr::relaxMigrationLock();
    }
  }
  buildMaps(is_new_db);
  is_initialized_ = true;
}

bool SysCatalog::isInitialized() const {
  return is_initialized_;
};

void SysCatalog::buildMaps(bool is_new_db) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  buildMapsUnlocked(is_new_db);
}

void SysCatalog::buildMapsUnlocked(bool is_new_db) {
  VLOG(2) << "reloading catalog caches for: " << shared::kSystemCatalogName;

  // Store permissions for temporary users.
  std::map<std::string, std::vector<std::string>> tu_map;
  for (auto& pair : temporary_users_by_name_) {
    CHECK(pair.second);
    UserMetadata& user = *pair.second;
    auto it = granteeMap_.find(to_upper(user.userName));
    CHECK(it != granteeMap_.end()) << to_upper(user.userName) << " not found";

    auto user_rl = dynamic_cast<User*>(it->second.get());
    CHECK(user_rl);
    std::vector<std::string> current_roles = user_rl->getRoles();
    auto result = tu_map.emplace(user.userName, std::move(current_roles));
    CHECK(result.second);
  }

  // Forget permissions and reload them from file storage.
  buildRoleMapUnlocked();
  buildUserRoleMapUnlocked();
  buildObjectDescriptorMapUnlocked();
  if (!is_new_db) {
    // We don't want to create the information schema db during database initialization
    // because we don't have the appropriate context to intialize the tables.  For
    // instance if the server is intended to run in distributed mode, initializing the
    // table as part of initdb will be missing information such as the location of the
    // string dictionary server.
    initializeInformationSchemaDb();
  }

  // Restore permissions for temporary users that were stored above.
  for (auto& pair : temporary_users_by_name_) {
    CHECK(pair.second);
    UserMetadata& user = *pair.second;

    createRole_unsafe(user.userName, /*user_private_role*/ true, /*is_temporary*/ true);

    auto it = tu_map.find(user.userName);
    CHECK(it != tu_map.end()) << user.userName << " not found";
    for (const auto& r : it->second) {
      grantRole_unsafe(r, user.userName, /*is_temporary*/ true);
    }
  }
}

SysCatalog::SysCatalog()
    : CommonFileOperations{basePath_}
    , aggregator_{false}
    , sqliteMutex_{}
    , sharedMutex_{}
    , thread_holding_sqlite_lock{std::thread::id()}
    , thread_holding_write_lock{std::thread::id()}
    , dummyCatalog_{std::make_shared<Catalog>()} {}

SysCatalog::~SysCatalog() {
  // TODO(sy): Need to lock here to wait for other threads to complete before pulling out
  // the rug from under them. Unfortunately this lock was seen to deadlock because the
  // HeavyDB shutdown sequence needs cleanup. Do we even need these clear()'s anymore?
  // sys_write_lock write_lock(this);
  granteeMap_.clear();
  objectDescriptorMap_.clear();
  cat_map_.clear();
}

void SysCatalog::initDB() {
  if (g_read_only) {
    throw std::runtime_error("can't init a new database in read-only mode");
  }
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "CREATE TABLE mapd_users (userid integer primary key, name text unique, "
        "passwd_hash text, issuper boolean, default_db integer references "
        "mapd_databases, can_login boolean)");
    sqliteConnector_->query_with_text_params(
        "INSERT INTO mapd_users VALUES (?, ?, ?, 1, NULL, 1)",
        std::vector<std::string>{shared::kRootUserIdStr,
                                 shared::kRootUsername,
                                 hash_with_bcrypt(shared::kDefaultRootPasswd)});
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
  createDatabase(shared::kDefaultDbName, shared::kRootUserId);
  createRole_unsafe(
      shared::kRootUsername, /*userPrivateRole=*/true, /*is_temporary=*/false);
}

void SysCatalog::checkAndExecuteMigrations() {
  migratePrivileged_old();
  createRoles();
  fixRolesMigration();
  migratePrivileges();
  migrateDBAccessPrivileges();
  updateUserSchema();  // must come before updatePasswordsToHashes()
  updatePasswordsToHashes();
  updateBlankPasswordsToRandom();  // must come after updatePasswordsToHashes()
  updateSupportUserDeactivation();
  addAdminUserRole();
  checkDuplicateCaseInsensitiveDbNames();
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
  std::string mapd_db_path = basePath_ + "/" + shared::kCatalogDirectoryName + "/mapd";
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
      basePath_ + "/" + shared::kCatalogDirectoryName + "/" + shared::kSystemCatalogName;
  LOG(INFO) << "Global metadata has been successfully moved into a separate catalog: "
            << sys_catalog_path
            << ". Using this database with an older version of heavydb "
               "is now impossible.";
  try {
    sqliteConnector_->query("DETACH DATABASE old_cat");
  } catch (const std::exception&) {
    // nothing to do here
  }
}

void SysCatalog::createRoles() {
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
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

/*
  There was an error in how we migrated users from pre-4.0 versions where we would copy
  all user names into the mapd_roles table.  This table should never have usernames in it
  (the correct migration was to copy users into the mapd_object_permissions table instead)
  so this migration function prunes such cases out.
 */
void SysCatalog::fixRolesMigration() {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query("SELECT name FROM mapd_users");
    auto num_rows = sqliteConnector_->getNumRows();
    std::vector<std::string> user_names;
    for (size_t i = 0; i < num_rows; ++i) {
      user_names.push_back(sqliteConnector_->getData<std::string>(i, 0));
    }
    for (const auto& user_name : user_names) {
      sqliteConnector_->query_with_text_param("DELETE FROM mapd_roles WHERE roleName = ?",
                                              user_name);
    }
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

namespace {

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
                                    const DBObject& object) {
  CHECK(object.valid());
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

}  // namespace

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
        auto type = DBObjectType::TableDBObjectType;
        DBObjectKey key{type, grantee.second};
        DBObject object(
            dbName, type, key, AccessPrivileges::ALL_TABLE_MIGRATE, shared::kRootUserId);
        insertOrUpdateObjectPrivileges(
            sqliteConnector_, users_by_id[grantee.first], true, object);
      }

      {
        // dashboard level permissions
        auto type = DBObjectType::DashboardDBObjectType;
        DBObjectKey key{type, grantee.second};
        DBObject object(dbName,
                        type,
                        key,
                        AccessPrivileges::ALL_DASHBOARD_MIGRATE,
                        shared::kRootUserId);
        insertOrUpdateObjectPrivileges(
            sqliteConnector_, users_by_id[grantee.first], true, object);
      }

      {
        // view level permissions
        auto type = DBObjectType::ViewDBObjectType;
        DBObjectKey key{type, grantee.second};
        DBObject object(
            dbName, type, key, AccessPrivileges::ALL_VIEW_MIGRATE, shared::kRootUserId);
        insertOrUpdateObjectPrivileges(
            sqliteConnector_, users_by_id[grantee.first], true, object);
      }
    }
    for (auto user : user_has_privs) {
      auto dbName = dbs_by_id[0];
      if (user.second == false && user.first != shared::kRootUserId) {
        {
          auto type = DBObjectType::DatabaseDBObjectType;
          DBObjectKey key{type, 0};
          DBObject object(dbName, type, key, AccessPrivileges::NONE, shared::kRootUserId);
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

void SysCatalog::addAdminUserRole() {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query(
        "SELECT roleName FROM mapd_object_permissions WHERE roleName = \'" +
        shared::kRootUsername + "\'");
    if (sqliteConnector_->getNumRows() != 0) {
      // already done
      sqliteConnector_->query("END TRANSACTION");
      return;
    }

    createRole_unsafe(
        shared::kRootUsername, /*userPrivateRole=*/true, /*is_temporary=*/false);
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::updatePasswordsToHashes() {
  sys_sqlite_lock sqlite_lock(this);
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

  sys_sqlite_lock sqlite_lock(this);
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
          std::vector<std::string>{hash_with_bcrypt(generate_random_string(72)),
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

void SysCatalog::updateSupportUserDeactivation() {
  const std::string UPDATE_SUPPORT_USER_DEACTIVATION = "update_support_user_deactivation";
  sys_sqlite_lock sqlite_lock(this);
  // check to see if the new column already exists
  sqliteConnector_->query("PRAGMA TABLE_INFO(mapd_users)");
  for (size_t i = 0; i < sqliteConnector_->getNumRows(); i++) {
    const auto& col_name = sqliteConnector_->getData<std::string>(i, 1);
    if (col_name == "can_login") {
      return;  // new column already exists
    }
  }
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query("ALTER TABLE mapd_users ADD COLUMN can_login BOOLEAN");
    sqliteConnector_->query("UPDATE mapd_users SET can_login = true");
    sqliteConnector_->query_with_text_params(
        "INSERT INTO mapd_version_history(version, migration_history) values(?,?)",
        std::vector<std::string>{std::to_string(MAPD_VERSION),
                                 UPDATE_SUPPORT_USER_DEACTIVATION});
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to add support for user deactivation: " << e.what();
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
      CHECK(getMetadataForDB(db_.second, dbmeta));
      for (auto user : users) {
        if (user.first != shared::kRootUserId) {
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

void SysCatalog::checkDuplicateCaseInsensitiveDbNames() const {
  static const string duplicate_check_migration{
      "check_duplicate_case_insensitive_db_names"};
  if (hasExecutedMigration(duplicate_check_migration)) {
    return;
  }
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query(
      "SELECT UPPER(name) AS db_name, COUNT(*) AS name_count "
      "FROM mapd_databases GROUP BY db_name HAVING name_count > 1");
  auto num_rows = sqliteConnector_->getNumRows();
  if (num_rows > 0) {
    std::stringstream error_message;
    error_message << "Duplicate case insensitive database names encountered:\n";
    for (size_t row = 0; row < num_rows; row++) {
      error_message << sqliteConnector_->getData<string>(row, 0) << " ("
                    << sqliteConnector_->getData<int>(row, 1) << ")\n";
    }
    throw std::runtime_error{error_message.str()};
  }
  recordExecutedMigration(duplicate_check_migration);
}

std::shared_ptr<Catalog> SysCatalog::login(std::string& dbname,
                                           std::string& username,
                                           const std::string& password,
                                           UserMetadata& user_meta,
                                           bool check_password) {
  // NOTE(sy): The dbname isn't const because getMetadataWithDefaultDB()
  // can reset it. The username isn't const because SamlServer's
  // login()/authenticate_user() can reset it.

  if (check_password) {
    loginImpl(username, password, user_meta);
  } else {  // not checking for password so user must exist
    if (!getMetadataForUser(username, user_meta)) {
      throw std::runtime_error("Invalid credentials.");
    }
  }
  // we should have a user and user_meta by now
  if (!user_meta.can_login) {
    throw std::runtime_error("Unauthorized Access: User " + username + " is deactivated");
  }
  Catalog_Namespace::DBMetadata db_meta;
  getMetadataWithDefaultDB(dbname, username, db_meta, user_meta);
  return getCatalog(db_meta, false);
}

// loginImpl() with no EE code and no SAML code
void SysCatalog::loginImpl(std::string& username,
                           const std::string& password,
                           UserMetadata& user_meta) {
  if (!checkPasswordForUser(password, username, user_meta)) {
    throw std::runtime_error("Authentication failure");
  }
}

std::shared_ptr<Catalog> SysCatalog::switchDatabase(std::string& dbname,
                                                    const std::string& username) {
  DBMetadata db_meta;
  UserMetadata user_meta;

  getMetadataWithDefaultDB(dbname, username, db_meta, user_meta);

  // NOTE(max): register database in Catalog that early to allow ldap
  // and saml create default user and role privileges on databases
  auto cat = getCatalog(db_meta, false);

  DBObject dbObject(dbname, DatabaseDBObjectType);
  dbObject.loadKey();
  dbObject.setPrivileges(AccessPrivileges::ACCESS);
  if (!checkPrivileges(user_meta, std::vector<DBObject>{dbObject})) {
    throw std::runtime_error("Unauthorized Access: user " + user_meta.userLoggable() +
                             " is not allowed to access database " + dbname + ".");
  }

  return cat;
}

void SysCatalog::check_for_session_encryption(const std::string& pki_cert,
                                              std::string& session) {
  if (!pki_server_->inUse()) {
    return;
  }
  pki_server_->encrypt_session(pki_cert, session);
}

UserMetadata SysCatalog::createUser(const string& name,
                                    UserAlterations alts,
                                    bool is_temporary) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  if (!alts.passwd) {
    alts.passwd = "";
  }
  if (!alts.is_super) {
    alts.is_super = false;
  }
  if (!alts.default_db) {
    alts.default_db = "";
  }
  if (!alts.can_login) {
    alts.can_login = true;
  }

  UserMetadata user;
  if (getMetadataForUser(name, user)) {
    throw runtime_error("User " + user.userLoggable() + " already exists.");
  }
  if (getGrantee(name)) {
    std::string const loggable = g_log_user_id ? std::string("") : name + ' ';
    throw runtime_error(
        "User " + loggable +
        "is same as one of existing grantees. User and role names should be unique.");
  }
  DBMetadata db;
  if (!alts.default_db->empty()) {
    if (!getMetadataForDB(*alts.default_db, db)) {
      throw runtime_error("DEFAULT_DB " + *alts.default_db + " not found.");
    }
  }

  // Temporary user.
  if (is_temporary) {
    if (!g_read_only) {
      throw std::runtime_error("Temporary users require read-only mode.");
      // NOTE(sy): We can remove this restriction when we're confident that
      // nothing permanent can depend on a temporary user.
    }
    auto user2 = std::make_shared<UserMetadata>(next_temporary_user_id_++,
                                                name,
                                                hash_with_bcrypt(*alts.passwd),
                                                *alts.is_super,
                                                !alts.default_db->empty() ? db.dbId : -1,
                                                *alts.can_login,
                                                true);
    temporary_users_by_name_[name] = user2;
    temporary_users_by_id_[user2->userId] = user2;
    createRole_unsafe(name, /*userPrivateRole=*/true, /*is_temporary=*/true);
    VLOG(1) << "Created temporary user: " << user2->userLoggable();
    return *user2;
  }

  // Normal user.
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    std::vector<std::string> vals;
    if (!alts.default_db->empty()) {
      vals = {name,
              hash_with_bcrypt(*alts.passwd),
              std::to_string(*alts.is_super),
              std::to_string(db.dbId),
              std::to_string(*alts.can_login)};
      sqliteConnector_->query_with_text_params(
          "INSERT INTO mapd_users (name, passwd_hash, issuper, default_db, can_login) "
          "VALUES (?, ?, ?, ?, ?)",
          vals);
    } else {
      vals = {name,
              hash_with_bcrypt(*alts.passwd),
              std::to_string(*alts.is_super),
              std::to_string(*alts.can_login)};
      sqliteConnector_->query_with_text_params(
          "INSERT INTO mapd_users (name, passwd_hash, issuper, can_login) "
          "VALUES (?, ?, ?, ?)",
          vals);
    }
    createRole_unsafe(name, /*userPrivateRole=*/true, /*is_temporary=*/false);
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
  auto u = getUser(name);
  CHECK(u);
  VLOG(1) << "Created user: " << u->userLoggable();
  return *u;
}

// Can be invoked directly to drop users without sanitization for testing.
void SysCatalog::dropUserUnchecked(const std::string& name, const UserMetadata& user) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  // Temporary user.
  if (user.is_temporary) {
    auto it1 = temporary_users_by_name_.find(name);
    CHECK(it1 != temporary_users_by_name_.end());
    auto it2 = temporary_users_by_id_.find(it1->second->userId);
    CHECK(it2 != temporary_users_by_id_.end());
    dropRole_unsafe(name, /*is_temporary=*/true);
    deleteObjectDescriptorMap(name);
    temporary_users_by_name_.erase(it1);
    temporary_users_by_id_.erase(it2);
    return;
  }

  // Normal user.

  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    dropRole_unsafe(name, /*is_temporary=*/false);
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

void SysCatalog::dropUser(const string& name) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  std::string const loggable = g_log_user_id ? std::string("") : name + ' ';

  UserMetadata user;
  if (!getMetadataForUser(name, user)) {
    throw runtime_error("Cannot drop user. User " + loggable + "does not exist.");
  }

  if (user.userId == shared::kRootUserId) {
    throw runtime_error("Cannot drop user. User " + loggable + "is required to exist.");
  }

  auto dbs = getAllDBMetadata();
  for (const auto& db : dbs) {
    if (db.dbOwner == user.userId) {
      throw runtime_error("Cannot drop user. User " + loggable + "owns database " +
                          db.dbName);
    }
  }

  dropUserUnchecked(name, user);
}

std::vector<Catalog*> SysCatalog::getCatalogsForAllDbs() {
  std::vector<Catalog*> catalogs{};
  const auto& db_metadata_list = getAllDBMetadata();
  for (const auto& db_metadata : db_metadata_list) {
    catalogs.emplace_back(getCatalog(db_metadata, false).get());
  }
  return catalogs;
}

namespace {  // anonymous namespace

auto append_with_commas = [](string& s, const string& t) {
  if (!s.empty()) {
    s += ", ";
  }
  s += t;
};

}  // anonymous namespace

bool UserAlterations::wouldChange(UserMetadata const& user) const {
  if (passwd && hash_with_bcrypt(*passwd) != user.passwd_hash) {
    return true;
  }
  if (is_super && *is_super != user.isSuper) {
    return true;
  }
  if (default_db) {
    DBMetadata db;
    if (!default_db->empty()) {
      if (!SysCatalog::instance().getMetadataForDB(*default_db, db)) {
        throw std::runtime_error(string("DEFAULT_DB ") + *default_db + " not found.");
      }
    } else {
      db.dbId = -1;
    }
    if (db.dbId != user.defaultDbId) {
      return true;
    }
  }
  if (can_login && *can_login != user.can_login) {
    return true;
  }
  return false;
}

std::string UserAlterations::toString(bool hide_password) const {
  std::stringstream ss;
  if (passwd) {
    if (hide_password) {
      ss << "PASSWORD='XXXXXXXX'";
    } else {
      ss << "PASSWORD='" << *passwd << "'";
    }
  }
  if (is_super) {
    if (!ss.str().empty()) {
      ss << ", ";
    }
    ss << "IS_SUPER='" << (*is_super ? "TRUE" : "FALSE") << "'";
  }
  if (default_db) {
    if (!ss.str().empty()) {
      ss << ", ";
    }
    ss << "DEFAULT_DB='" << *default_db << "'";
  }
  if (can_login) {
    if (!ss.str().empty()) {
      ss << ", ";
    }
    ss << "CAN_LOGIN='" << *can_login << "'";
  }
  return ss.str();
}

UserMetadata SysCatalog::alterUser(string const& name, UserAlterations alts) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  UserMetadata user;
  if (!getMetadataForUser(name, user)) {
    std::string const loggable = g_log_user_id ? std::string("") : name + ' ';
    throw runtime_error("Cannot alter user. User " + loggable + "does not exist.");
  }
  if (!alts.wouldChange(user)) {
    return user;
  }

  // Temporary user.
  if (user.is_temporary) {
    if (alts.passwd) {
      user.passwd_hash = hash_with_bcrypt(*alts.passwd);
    }
    if (alts.is_super) {
      user.isSuper = *alts.is_super;
    }
    if (alts.default_db) {
      if (!alts.default_db->empty()) {
        DBMetadata db;
        if (!getMetadataForDB(*alts.default_db, db)) {
          throw runtime_error(string("DEFAULT_DB ") + *alts.default_db + " not found.");
        }
        user.defaultDbId = db.dbId;
      } else {
        user.defaultDbId = -1;
      }
    }
    if (alts.can_login) {
      user.can_login = *alts.can_login;
    }
    *temporary_users_by_name_[name] = user;
    return user;
  }

  // Normal user.
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    string sql;
    std::vector<std::string> values;
    if (alts.passwd) {
      append_with_commas(sql, "passwd_hash = ?");
      values.push_back(hash_with_bcrypt(*alts.passwd));
    }
    if (alts.is_super) {
      append_with_commas(sql, "issuper = ?");
      values.push_back(std::to_string(*alts.is_super));
    }
    if (alts.default_db) {
      if (!alts.default_db->empty()) {
        append_with_commas(sql, "default_db = ?");
        DBMetadata db;
        if (!getMetadataForDB(*alts.default_db, db)) {
          throw runtime_error(string("DEFAULT_DB ") + *alts.default_db + " not found.");
        }
        values.push_back(std::to_string(db.dbId));
      } else {
        append_with_commas(sql, "default_db = NULL");
      }
    }
    if (alts.can_login) {
      append_with_commas(sql, "can_login = ?");
      values.push_back(std::to_string(*alts.can_login));
    }

    sql = "UPDATE mapd_users SET " + sql + " WHERE userid = ?";
    values.push_back(std::to_string(user.userId));

    sqliteConnector_->query_with_text_params(sql, values);
  } catch (const std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
  auto u = getUser(name);
  CHECK(u);
  VLOG(1) << "Altered user: " << u->userLoggable();
  return *u;
}

auto SysCatalog::yieldTransactionStreamer() {
  return
      [](auto& db_connector, auto on_success, auto on_failure, auto&&... query_requests) {
        auto query_runner = [&db_connector](auto&&... query_reqs) {
          [[gnu::unused]] int throw_away[] = {
              (db_connector->query_with_text_params(
                   std::forward<decltype(query_reqs)>(query_reqs)),
               0)...};
        };

        db_connector->query("BEGIN TRANSACTION");
        try {
          query_runner(std::forward<decltype(query_requests)>(query_requests)...);
          on_success();
        } catch (std::exception&) {
          db_connector->query("ROLLBACK TRANSACTION");
          on_failure();
          throw;
        }
        db_connector->query("END TRANSACTION");
      };
}

void SysCatalog::updateUserRoleName(const std::string& roleName,
                                    const std::string& newName) {
  sys_write_lock write_lock(this);

  auto it = granteeMap_.find(to_upper(roleName));
  if (it != granteeMap_.end()) {
    it->second->setName(newName);
    std::swap(granteeMap_[to_upper(newName)], it->second);
    granteeMap_.erase(it);
  }

  // Also rename in objectDescriptorMap_
  for (auto d = objectDescriptorMap_.begin(); d != objectDescriptorMap_.end(); ++d) {
    if (d->second->roleName == roleName) {
      d->second->roleName = newName;
    }
  }
}

void SysCatalog::renameUser(std::string const& old_name, std::string const& new_name) {
  using namespace std::string_literals;
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  UserMetadata old_user;
  if (!getMetadataForUser(old_name, old_user)) {
    std::string const loggable = g_log_user_id ? std::string("") : old_name + ' ';
    throw std::runtime_error("User " + loggable + "doesn't exist.");
  }

  UserMetadata new_user;
  if (getMetadataForUser(new_name, new_user)) {
    throw std::runtime_error("User " + new_user.userLoggable() + " already exists.");
  }

  if (getGrantee(new_name)) {
    std::string const loggable = g_log_user_id ? std::string("") : new_name + ' ';
    throw runtime_error(
        "Username " + loggable +
        "is same as one of existing grantees. User and role names should be unique.");
  }

  // Temporary user.
  if (old_user.is_temporary) {
    auto userit = temporary_users_by_name_.find(old_name);
    CHECK(userit != temporary_users_by_name_.end());
    auto node = temporary_users_by_name_.extract(userit);
    node.key() = new_name;
    temporary_users_by_name_.insert(std::move(node));
    userit->second->userName = new_name;
    updateUserRoleName(old_name, new_name);
    return;
  }

  // Normal user.
  auto transaction_streamer = yieldTransactionStreamer();
  auto failure_handler = [] {};
  auto success_handler = [this, &old_name, &new_name] {
    updateUserRoleName(old_name, new_name);
  };
  auto q1 = {"UPDATE mapd_users SET name=?1 where name=?2;"s, new_name, old_name};
  auto q2 = {"UPDATE mapd_object_permissions set roleName=?1 WHERE roleName=?2;"s,
             new_name,
             old_name};
  auto q3 = {"UPDATE mapd_roles set userName=?1 WHERE userName=?2;"s, new_name, old_name};
  transaction_streamer(sqliteConnector_, success_handler, failure_handler, q1, q2, q3);
}

void SysCatalog::changeDatabaseOwner(std::string const& dbname,
                                     const std::string& new_owner) {
  using namespace std::string_literals;
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  DBMetadata db;
  if (!getMetadataForDB(dbname, db)) {
    throw std::runtime_error("Database " + dbname + " does not exists.");
  }

  Catalog_Namespace::UserMetadata user, original_owner;
  if (!getMetadataForUser(new_owner, user)) {
    throw std::runtime_error("User with username \"" + new_owner + "\" does not exist. " +
                             "Database with name \"" + dbname +
                             "\" can not have owner changed.");
  }

  bool original_owner_exists = getMetadataForUserById(db.dbOwner, original_owner);
  auto cat = getCatalog(db, true);
  DBObject db_object(db.dbName, DBObjectType::DatabaseDBObjectType);
  runUpdateQueriesAndChangeOwnership(
      user,
      original_owner,
      db_object,
      *cat,
      UpdateQueries{{"UPDATE mapd_databases SET owner=?1 WHERE name=?2;",
                     {std::to_string(user.userId), db.dbName}}},
      original_owner_exists);
}

void SysCatalog::renameDatabase(std::string const& old_name,
                                std::string const& new_name) {
  using namespace std::string_literals;
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  DBMetadata new_db;
  if (getMetadataForDB(new_name, new_db)) {
    throw std::runtime_error("Database " + new_name + " already exists.");
  }
  if (to_upper(new_name) == to_upper(shared::kSystemCatalogName)) {
    throw std::runtime_error("Database name " + new_name + "is reserved.");
  }

  DBMetadata old_db;
  if (!getMetadataForDB(old_name, old_db)) {
    throw std::runtime_error("Database " + old_name + " does not exists.");
  }

  removeCatalog(old_db.dbName);

  std::string old_catalog_path, new_catalog_path;
  std::tie(old_catalog_path, new_catalog_path) =
      duplicateAndRenameCatalog(old_db.dbName, new_name);

  auto transaction_streamer = yieldTransactionStreamer();
  auto failure_handler = [this, new_catalog_path] {
    removeCatalogByFullPath(new_catalog_path);
  };
  auto success_handler = [this, old_catalog_path] {
    removeCatalogByFullPath(old_catalog_path);
  };

  auto q1 = {
      "UPDATE mapd_databases SET name=?1 WHERE name=?2;"s, new_name, old_db.dbName};
  auto q2 = {
      "UPDATE mapd_object_permissions SET objectName=?1 WHERE objectNAME=?2 and (objectPermissionsType=?3 or objectId = -1) and dbId=?4;"s,
      new_name,
      old_db.dbName,
      std::to_string(static_cast<int>(DBObjectType::DatabaseDBObjectType)),
      std::to_string(old_db.dbId)};

  transaction_streamer(sqliteConnector_, success_handler, failure_handler, q1, q2);
}

void SysCatalog::createDatabase(const string& name, int owner) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  DBMetadata db;
  if (getMetadataForDB(name, db)) {
    throw runtime_error("Database " + name + " already exists.");
  }
  if (to_upper(name) == to_upper(shared::kSystemCatalogName)) {
    throw runtime_error("Database name " + name + " is reserved.");
  }

  std::unique_ptr<SqliteConnector> dbConn(
      new SqliteConnector(name, basePath_ + "/" + shared::kCatalogDirectoryName + "/"));
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
        "sort_column_id integer default 0, storage_type text default '', "
        "max_rollback_epochs integer default -1, "
        "is_system_table boolean default 0, "
        "num_shards integer, key_metainfo TEXT, version_num "
        "BIGINT DEFAULT 1) ");
    dbConn->query(
        "CREATE TABLE mapd_columns (tableid integer references mapd_tables, columnid "
        "integer, name text, coltype "
        "integer, colsubtype integer, coldim integer, colscale integer, is_notnull "
        "boolean, compression integer, "
        "comp_param integer, size integer, chunks text, is_systemcol boolean, "
        "is_virtualcol boolean, virtual_expr "
        "text, is_deletedcol boolean, version_num BIGINT, default_value text, "
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

    if (g_enable_fsi) {
      dbConn->query(Catalog::getForeignServerSchema());
      dbConn->query(Catalog::getForeignTableSchema());
    }
    dbConn->query(Catalog::getCustomExpressionsSchema());
  } catch (const std::exception&) {
    dbConn->query("ROLLBACK TRANSACTION");
    boost::filesystem::remove(basePath_ + "/" + shared::kCatalogDirectoryName + "/" +
                              name);
    throw;
  }
  dbConn->query("END TRANSACTION");

  std::shared_ptr<Catalog> cat;
  // Now update SysCatalog with privileges and the new database
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    sqliteConnector_->query_with_text_param(
        "INSERT INTO mapd_databases (name, owner) VALUES (?, " + std::to_string(owner) +
            ")",
        name);
    CHECK(getMetadataForDB(name, db));

    cat = getCatalog(db, true);

    if (owner != shared::kRootUserId) {
      DBObject object(name, DBObjectType::DatabaseDBObjectType);
      object.loadKey(*cat);
      UserMetadata user;
      CHECK(getMetadataForUserById(owner, user));
      grantAllOnDatabase_unsafe(user.userName, object, *cat);
    }
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    boost::filesystem::remove(basePath_ + "/" + shared::kCatalogDirectoryName + "/" +
                              name);
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");

  // force a migration on the new database
  removeCatalog(name);
  cat = getCatalog(db, false);

  if (g_enable_fsi) {
    try {
      cat->createDefaultServersIfNotExists();
    } catch (...) {
      boost::filesystem::remove(basePath_ + "/" + shared::kCatalogDirectoryName + "/" +
                                name);
      throw;
    }
  }
}

void SysCatalog::dropDatabase(const DBMetadata& db) {
  auto cat = getCatalog(db, false);
  cat->eraseDbPhysicalData();
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
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
        revokeAllOnDatabase_unsafe(
            grantee.second->getName(), db.dbId, grantee.second.get());
      }
    }
    sqliteConnector_->query_with_text_param("DELETE FROM mapd_databases WHERE dbid = ?",
                                            std::to_string(db.dbId));
    cat->eraseDbMetadata();
    removeCatalog(db.dbName);
  } catch (const std::exception&) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

// checkPasswordForUser() with no EE code
bool SysCatalog::checkPasswordForUser(const std::string& passwd,
                                      std::string& name,
                                      UserMetadata& user) {
  return checkPasswordForUserImpl(passwd, name, user);
}

bool SysCatalog::checkPasswordForUserImpl(const std::string& passwd,
                                          std::string& name,
                                          UserMetadata& user) {
  sys_read_lock read_lock(this);
  if (!getMetadataForUser(name, user)) {
    // Check password against some fake hash just to waste time so that response times
    // for invalid password and invalid user are similar and a caller can't say the
    // difference
    char fake_hash[BCRYPT_HASHSIZE];
    CHECK(bcrypt_gensalt(-1, fake_hash) == 0);
    bcrypt_checkpw(passwd.c_str(), fake_hash);
    LOG(WARNING) << "Local login failed";
    return false;
  }
  int pwd_check_result = bcrypt_checkpw(passwd.c_str(), user.passwd_hash.c_str());
  // if the check fails there is a good chance that data on disc is broken
  CHECK(pwd_check_result >= 0);
  return pwd_check_result == 0;
}

static bool parseUserMetadataFromSQLite(const std::unique_ptr<SqliteConnector>& conn,
                                        UserMetadata& user,
                                        int row) {
  int numRows = conn->getNumRows();
  if (numRows == 0) {
    return false;
  }
  user.userId = conn->getData<int>(row, 0);
  user.userName = conn->getData<string>(row, 1);
  user.passwd_hash = conn->getData<string>(row, 2);
  user.isSuper = conn->getData<bool>(row, 3);
  user.defaultDbId = conn->isNull(row, 4) ? -1 : conn->getData<int>(row, 4);
  if (conn->isNull(row, 5)) {
    LOG(WARNING)
        << "User property 'can_login' not set for user " << user.userLoggable()
        << ". Disabling login ability. Set the users login ability with \"ALTER USER "
        << (g_log_user_id ? std::string("[username]") : user.userName)
        << " (can_login='true');\".";
  }
  user.can_login = conn->isNull(row, 5) ? false : conn->getData<bool>(row, 5);
  return true;
}

bool SysCatalog::getMetadataForUser(const string& name, UserMetadata& user) {
  sys_read_lock read_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_param(
      "SELECT userid, name, passwd_hash, issuper, default_db, can_login FROM mapd_users "
      "WHERE name = ?",
      name);
  int numRows = sqliteConnector_->getNumRows();
  if (numRows == 0) {
    auto userit = temporary_users_by_name_.find(name);
    if (userit != temporary_users_by_name_.end()) {
      user = *userit->second;
      return true;
    } else {
      return false;
    }
  }
  return parseUserMetadataFromSQLite(sqliteConnector_, user, 0);
}

bool SysCatalog::getMetadataForUserById(const int32_t idIn, UserMetadata& user) {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_param(
      "SELECT userid, name, passwd_hash, issuper, default_db, can_login FROM mapd_users "
      "WHERE userid = ?",
      std::to_string(idIn));
  int numRows = sqliteConnector_->getNumRows();
  if (numRows == 0) {
    auto userit = temporary_users_by_id_.find(idIn);
    if (userit != temporary_users_by_id_.end()) {
      user = *userit->second;
      return true;
    } else {
      return false;
    }
  }
  return parseUserMetadataFromSQLite(sqliteConnector_, user, 0);
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

namespace {

auto get_users(SysCatalog& syscat,
               std::unique_ptr<SqliteConnector>& sqliteConnector,
               const int32_t dbId = -1) {
  // Normal users.
  sqliteConnector->query(
      "SELECT userid, name, passwd_hash, issuper, default_db, can_login FROM mapd_users");
  int numRows = sqliteConnector->getNumRows();
  list<UserMetadata> user_list;
  const bool return_all_users = dbId == -1;
  auto has_any_privilege = [&return_all_users, &dbId, &syscat](const std::string& name) {
    if (!return_all_users) {
      const auto grantee = syscat.getUserGrantee(name);
      return grantee ? grantee->hasAnyPrivilegesOnDb(dbId, false) : false;
    }
    return true;
  };
  for (int r = 0; r < numRows; ++r) {
    Catalog_Namespace::UserMetadata user;
    parseUserMetadataFromSQLite(sqliteConnector, user, r);
    if (has_any_privilege(user.userName)) {
      user_list.emplace_back(std::move(user));
    }
  }

  // Temporary users.
  for (const auto& [id, userptr] : syscat.temporary_users_by_id_) {
    if (has_any_privilege(userptr->userName)) {
      user_list.emplace_back(*userptr);
    }
  }

  return user_list;
}

}  // namespace

list<UserMetadata> SysCatalog::getAllUserMetadata(const int64_t dbId) {
  // this call is to return users that have some form of permissions to objects in the db
  // sadly mapd_object_permissions table is also misused to manage user roles.
  sys_sqlite_lock sqlite_lock(this);
  return get_users(*this, sqliteConnector_, dbId);
}

list<UserMetadata> SysCatalog::getAllUserMetadata() {
  sys_sqlite_lock sqlite_lock(this);
  return get_users(*this, sqliteConnector_);
}

void SysCatalog::getMetadataWithDefaultDB(std::string& dbname,
                                          const std::string& username,
                                          Catalog_Namespace::DBMetadata& db_meta,
                                          UserMetadata& user_meta) {
  sys_read_lock read_lock(this);
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
        std::string loggable = g_log_user_id ? std::string("") : ' ' + user_meta.userName;
        throw std::runtime_error(
            "Server error: User #" + std::to_string(user_meta.userId) + loggable +
            " has invalid default_db #" + std::to_string(user_meta.defaultDbId) +
            " which does not exist.");
      }
      dbname = db_meta.dbName;
      // loaded the user's default database
    } else {
      if (!getMetadataForDB(shared::kDefaultDbName, db_meta)) {
        throw std::runtime_error(std::string("Database ") + shared::kDefaultDbName +
                                 " does not exist.");
      }
      dbname = shared::kDefaultDbName;
      // loaded the mapd database by default
    }
  }
}

bool SysCatalog::getMetadataForDB(const string& name, DBMetadata& db) {
  sys_read_lock read_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_param(
      "SELECT dbid, name, owner FROM mapd_databases WHERE UPPER(name) = ?",
      to_upper(name));
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

  std::map<int32_t, std::string> user_id_to_name_map;
  for (const auto& user : user_list) {
    user_id_to_name_map.emplace(user.userId, user.userName);
  }

  for (auto d : db_list) {
    DBObject dbObject(d.dbName, DatabaseDBObjectType);
    dbObject.loadKey();
    dbObject.setPrivileges(AccessPrivileges::ACCESS);
    if (!checkPrivileges(user, std::vector<DBObject>{dbObject})) {
      continue;
    }

    if (auto it = user_id_to_name_map.find(d.dbOwner); it != user_id_to_name_map.end()) {
      ret.emplace_back(DBSummary{d.dbName, it->second});
    } else {
      ret.emplace_back(DBSummary{d.dbName, "<DELETED>"});
    }
  }

  return ret;
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
    case ServerDBObjectType:
      object.setPrivileges(AccessPrivileges::ALL_SERVER);
      break;
    default:
      object.setPrivileges(AccessPrivileges::ALL_DATABASE);
      break;
  }
  object.setOwner(user.userId);
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    if (!user.isSuper) {  // no need to grant to suser, has all privs by default
      grantDBObjectPrivileges_unsafe(user.userName, object, catalog);
      auto* grantee = instance().getUserGrantee(user.userName);
      if (!grantee) {
        throw runtime_error("Cannot create DBObject. User " + user.userLoggable() +
                            " does not exist.");
      }
      grantee->grantPrivileges(object);
    }
  } catch (std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::renameDBObject(const std::string& objectName,
                                const std::string& newName,
                                DBObjectType type,
                                int32_t objectId,
                                const Catalog_Namespace::Catalog& catalog) {
  sys_write_lock write_lock(this);
  DBObject new_object(newName, type);
  DBObjectKey key;
  key.dbId = catalog.getCurrentDB().dbId;
  key.objectId = objectId;
  key.permissionType = type;
  new_object.setObjectKey(key);
  auto objdescs =
      getMetadataForObject(key.dbId, static_cast<int32_t>(type), key.objectId);
  for (auto obj : objdescs) {
    Grantee* grnt = getGrantee(obj->roleName);
    if (grnt) {
      grnt->renameDbObject(new_object);
    }
  }
  renameObjectsInDescriptorMap(new_object, catalog);
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
  CHECK(object.valid());
  if (object.getPrivileges().hasPermission(DatabasePrivileges::ALL) &&
      object.getObjectKey().permissionType == DatabaseDBObjectType) {
    return grantAllOnDatabase_unsafe(granteeName, object, catalog);
  }

  sys_write_lock write_lock(this);

  UserMetadata user_meta;
  bool is_temporary_user{false};
  if (instance().getMetadataForUser(granteeName, user_meta)) {
    if (user_meta.isSuper) {
      // super doesn't have explicit privileges so nothing to do
      return;
    }
    is_temporary_user = user_meta.is_temporary;
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

  if (!is_temporary_user) {
    sys_sqlite_lock sqlite_lock(this);
    insertOrUpdateObjectPrivileges(
        sqliteConnector_, granteeName, grantee->isUser(), object);
  }
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

  if (g_enable_fsi) {
    tmp_object.setPrivileges(AccessPrivileges::ALL_SERVER);
    tmp_object.setPermissionType(ServerDBObjectType);
    grantDBObjectPrivileges_unsafe(roleName, tmp_object, catalog);
  }

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

void SysCatalog::revokeDBObjectPrivilegesFromAllBatch_unsafe(
    vector<DBObject>& objects,
    Catalog_Namespace::Catalog* catalog) {
  for (const auto& object : objects) {
    revokeDBObjectPrivilegesFromAll_unsafe(object, catalog);
  }
}

// REVOKE INSERT ON TABLE payroll_table FROM payroll_dept_role;
void SysCatalog::revokeDBObjectPrivileges_unsafe(
    const std::string& granteeName,
    DBObject object,
    const Catalog_Namespace::Catalog& catalog) {
  sys_write_lock write_lock(this);

  UserMetadata user_meta;
  bool is_temporary_user{false};
  if (instance().getMetadataForUser(granteeName, user_meta)) {
    if (user_meta.isSuper) {
      // super doesn't have explicit privileges so nothing to do
      return;
    }
    is_temporary_user = user_meta.is_temporary;
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
    if (!is_temporary_user) {
      sys_sqlite_lock sqlite_lock(this);
      insertOrUpdateObjectPrivileges(
          sqliteConnector_, granteeName, grantee->isUser(), *ret_object);
    }
    updateObjectDescriptorMap(granteeName, *ret_object, grantee->isUser(), catalog);
  } else {
    if (!is_temporary_user) {
      sys_sqlite_lock sqlite_lock(this);
      deleteObjectPrivileges(sqliteConnector_, granteeName, grantee->isUser(), object);
    }
    deleteObjectDescriptorMap(granteeName, object, catalog);
  }
}

void SysCatalog::revokeAllOnDatabase_unsafe(const std::string& roleName,
                                            int32_t dbId,
                                            Grantee* grantee) {
  bool is_temporary =
      (temporary_users_by_name_.find(roleName) != temporary_users_by_name_.end());
  if (!is_temporary) {
    sys_sqlite_lock sqlite_lock(this);
    sqliteConnector_->query_with_text_params(
        "DELETE FROM mapd_object_permissions WHERE roleName = ?1 and dbId = ?2",
        std::vector<std::string>{roleName, std::to_string(dbId)});
  }
  grantee->revokeAllOnDatabase(dbId);
  for (auto d = objectDescriptorMap_.begin(); d != objectDescriptorMap_.end();) {
    if (d->second->roleName == roleName && d->second->dbId == dbId) {
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

void SysCatalog::runUpdateQueriesAndChangeOwnership(
    const UserMetadata& new_owner,
    const UserMetadata& previous_owner,
    DBObject object,
    const Catalog_Namespace::Catalog& catalog,
    const SysCatalog::UpdateQueries& update_queries,
    bool revoke_privileges) {
  sys_write_lock write_lock(this);
  if (new_owner.is_temporary || previous_owner.is_temporary) {
    throw std::runtime_error("ownership change not allowed for temporary user(s)");
  }
  sys_sqlite_lock sqlite_lock(this);
  object.loadKey(catalog);
  switch (object.getType()) {
    case TableDBObjectType:
      object.setPrivileges(AccessPrivileges::ALL_TABLE);
      break;
    case DashboardDBObjectType:
      object.setPrivileges(AccessPrivileges::ALL_DASHBOARD);
      break;
    case ServerDBObjectType:
      object.setPrivileges(AccessPrivileges::ALL_SERVER);
      break;
    case DatabaseDBObjectType:
      object.setPrivileges(AccessPrivileges::ALL_DATABASE);
      break;
    case ViewDBObjectType:
      object.setPrivileges(AccessPrivileges::ALL_VIEW);
      break;
    default:
      UNREACHABLE();  // unkown object type
      break;
  }
  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    if (!new_owner.isSuper) {  // no need to grant to suser, has all privs by default
      grantDBObjectPrivileges_unsafe(new_owner.userName, object, catalog);
    }
    if (!previous_owner.isSuper && revoke_privileges) {  // no need to revoke from suser
      revokeDBObjectPrivileges_unsafe(previous_owner.userName, object, catalog);
    }

    // run update queries if specified
    for (const auto& update_query : update_queries) {
      sqliteConnector_->query_with_text_params(update_query.query,
                                               update_query.text_params);
    }

    auto object_key = object.getObjectKey();
    sqliteConnector_->query_with_text_params(
        "UPDATE mapd_object_permissions SET objectOwnerId = ? WHERE dbId = ? AND "
        "objectId = ? AND objectPermissionsType = ?",
        std::vector<std::string>{std::to_string(new_owner.userId),
                                 std::to_string(object_key.dbId),
                                 std::to_string(object_key.objectId),
                                 std::to_string(object_key.permissionType)});

    for (const auto& [user_or_role, grantee] : granteeMap_) {
      grantee->reassignObjectOwner(object_key, new_owner.userId);
    }

    for (const auto& [map_object_key, map_object_descriptor] : objectDescriptorMap_) {
      if (map_object_descriptor->objectId == object_key.objectId &&
          map_object_descriptor->objectType == object_key.permissionType &&
          map_object_descriptor->dbId == object_key.dbId) {
        map_object_descriptor->objectOwnerId = new_owner.userId;
      }
    }
  } catch (std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    rebuildObjectMapsUnlocked();
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::changeDBObjectOwnership(const UserMetadata& new_owner,
                                         const UserMetadata& previous_owner,
                                         DBObject object,
                                         const Catalog_Namespace::Catalog& catalog,
                                         bool revoke_privileges) {
  runUpdateQueriesAndChangeOwnership(
      new_owner, previous_owner, object, catalog, {}, revoke_privileges);
}

void SysCatalog::getDBObjectPrivileges(const std::string& granteeName,
                                       DBObject& object,
                                       const Catalog_Namespace::Catalog& catalog) const {
  sys_read_lock read_lock(this);
  UserMetadata user_meta;

  if (instance().getMetadataForUser(granteeName, user_meta)) {
    if (user_meta.isSuper) {
      throw runtime_error(
          "Request to show privileges from " + granteeName +
          " failed because user is super user and has all privileges by default.");
    }
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
                                   const bool user_private_role,
                                   const bool is_temporary) {
  sys_write_lock write_lock(this);

  auto* grantee = getGrantee(roleName);
  if (grantee) {
    throw std::runtime_error("CREATE ROLE " + roleName +
                             " failed because grantee with this name already exists.");
  }
  std::unique_ptr<Grantee> g;
  if (user_private_role) {
    g.reset(new User(roleName));
  } else {
    g.reset(new Role(roleName));
  }
  grantee = g.get();
  granteeMap_[to_upper(roleName)] = std::move(g);

  // NOTE (max): Why create an empty privileges record for a role?
  /* grant none privileges to this role and add it to sqlite DB */
  DBObject dbObject(shared::kDefaultDbName, DatabaseDBObjectType);
  DBObjectKey objKey;
  // 0 is an id that does not exist
  objKey.dbId = 0;
  objKey.permissionType = DatabaseDBObjectType;
  dbObject.setObjectKey(objKey);
  grantee->grantPrivileges(dbObject);

  if (!is_temporary) {
    sys_sqlite_lock sqlite_lock(this);
    insertOrUpdateObjectPrivileges(
        sqliteConnector_, roleName, user_private_role, dbObject);
  }
}

void SysCatalog::dropRole_unsafe(const std::string& roleName, const bool is_temporary) {
  sys_write_lock write_lock(this);

  for (auto d = objectDescriptorMap_.begin(); d != objectDescriptorMap_.end();) {
    if (d->second->roleName == roleName) {
      d = objectDescriptorMap_.erase(d);
    } else {
      d++;
    }
  }
  // it may very well be a user "role", so keep it generic
  granteeMap_.erase(to_upper(roleName));

  if (!is_temporary) {
    sys_sqlite_lock sqlite_lock(this);
    sqliteConnector_->query_with_text_param("DELETE FROM mapd_roles WHERE roleName = ?",
                                            roleName);
    sqliteConnector_->query_with_text_param(
        "DELETE FROM mapd_object_permissions WHERE roleName = ?", roleName);
  }
}

void SysCatalog::grantRoleBatch_unsafe(const std::vector<std::string>& roles,
                                       const std::vector<std::string>& grantees) {
  for (const auto& role : roles) {
    for (const auto& grantee : grantees) {
      bool is_temporary_user{false};
      UserMetadata user;
      if (getMetadataForUser(grantee, user)) {
        is_temporary_user = user.is_temporary;
      }
      grantRole_unsafe(role, grantee, is_temporary_user);
    }
  }
}

// GRANT ROLE payroll_dept_role TO joe;
void SysCatalog::grantRole_unsafe(const std::string& roleName,
                                  const std::string& granteeName,
                                  const bool is_temporary) {
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
    if (!is_temporary) {
      sys_sqlite_lock sqlite_lock(this);
      sqliteConnector_->query_with_text_params(
          "INSERT INTO mapd_roles(roleName, userName) VALUES (?, ?)",
          std::vector<std::string>{rl->getName(), grantee->getName()});
    }
  }
}

void SysCatalog::revokeRoleBatch_unsafe(const std::vector<std::string>& roles,
                                        const std::vector<std::string>& grantees) {
  for (const auto& role : roles) {
    for (const auto& grantee : grantees) {
      bool is_temporary_user{false};
      UserMetadata user;
      if (getMetadataForUser(grantee, user)) {
        is_temporary_user = user.is_temporary;
      }
      revokeRole_unsafe(role, grantee, is_temporary_user);
    }
  }
}

// REVOKE ROLE payroll_dept_role FROM joe;
void SysCatalog::revokeRole_unsafe(const std::string& roleName,
                                   const std::string& granteeName,
                                   const bool is_temporary) {
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
  if (!is_temporary) {
    sys_sqlite_lock sqlite_lock(this);
    sqliteConnector_->query_with_text_params(
        "DELETE FROM mapd_roles WHERE roleName = ? AND userName = ?",
        std::vector<std::string>{rl->getName(), grantee->getName()});
  }
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
    auto od = std::make_unique<ObjectRoleDescriptor>();
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
        std::move(od)));
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
    throw runtime_error("Cannot check privileges. User " + user.userLoggable() +
                        " does not exist.");
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
    throw runtime_error("Cannot check privileges. User " + user.userLoggable() +
                        " does not exist.");
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
    std::string const loggable = g_log_user_id ? std::string("") : userName + ' ';
    throw runtime_error("Request to check privileges for user " + loggable +
                        "failed because user with this name does not exist.");
  }
  return (checkPrivileges(user, privObjects));
}

Grantee* SysCatalog::getGrantee(const std::string& name) const {
  sys_read_lock read_lock(this);
  auto grantee = granteeMap_.find(to_upper(name));
  if (grantee == granteeMap_.end()) {  // check to make sure role exists
    return nullptr;
  }
  return grantee->second.get();  // returns pointer to role
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
    objectsList.push_back(d->second.get());
  }
  return objectsList;  // return pointers to objects
}

std::vector<ObjectRoleDescriptor> SysCatalog::getMetadataForAllObjects() const {
  sys_read_lock read_lock(this);
  std::vector<ObjectRoleDescriptor> objects;
  for (const auto& entry : objectDescriptorMap_) {
    auto object_role = entry.second.get();
    if (object_role->dbId != 0 && !isDashboardSystemRole(object_role->roleName)) {
      objects.emplace_back(*object_role);
    }
  }
  return objects;
}

bool SysCatalog::isRoleGrantedToGrantee(const std::string& granteeName,
                                        const std::string& roleName,
                                        bool only_direct) const {
  sys_read_lock read_lock(this);
  if (roleName == granteeName) {
    return true;
  }
  bool is_role_granted = false;
  auto* target_role = instance().getRoleGrantee(roleName);
  auto has_role = [&](auto grantee_rl) {
    is_role_granted = target_role && grantee_rl->hasRole(target_role, only_direct);
  };
  if (auto* user_role = instance().getUserGrantee(granteeName); user_role) {
    has_role(user_role);
  } else if (auto* role = instance().getRoleGrantee(granteeName); role) {
    has_role(role);
  } else {
    CHECK(false);
  }
  return is_role_granted;
}

bool SysCatalog::isDashboardSystemRole(const std::string& roleName) const {
  return boost::algorithm::ends_with(roleName, SYSTEM_ROLE_TAG);
}

std::vector<std::string> SysCatalog::getRoles(const std::string& user_name,
                                              bool effective) {
  sys_read_lock read_lock(this);
  auto* grantee = getGrantee(user_name);
  if (!grantee) {
    throw std::runtime_error("user or role not found");
  }
  return grantee->getRoles(/*only_direct=*/!effective);
}

std::vector<std::string> SysCatalog::getRoles(const std::string& userName,
                                              const int32_t dbId) {
  sys_sqlite_lock sqlite_lock(this);
  std::string sql =
      "SELECT DISTINCT roleName FROM mapd_object_permissions WHERE "
      "objectPermissions<>0 "
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

std::vector<std::string> SysCatalog::getRoles(bool include_user_private_role,
                                              bool is_super,
                                              const std::string& user_name,
                                              bool ignore_deleted_user) {
  sys_read_lock read_lock(this);
  if (ignore_deleted_user) {
    // In certain cases, it is possible to concurrently call this method while the user is
    // being dropped. In such a case, return an empty result.
    UserMetadata user;
    if (!getMetadataForUser(user_name, user)) {
      return {};
    }
  }
  std::vector<std::string> roles;
  for (auto& grantee : granteeMap_) {
    if (!include_user_private_role && grantee.second->isUser()) {
      continue;
    }
    if (!is_super &&
        !isRoleGrantedToGrantee(user_name, grantee.second->getName(), false)) {
      continue;
    }
    if (isDashboardSystemRole(grantee.second->getName())) {
      continue;
    }
    roles.push_back(grantee.second->getName());
  }
  return roles;
}

std::set<std::string> SysCatalog::getCreatedRoles() const {
  sys_read_lock read_lock(this);
  std::set<std::string> roles;  // Sorted for human readers.
  for (const auto& [key, grantee] : granteeMap_) {
    if (!grantee->isUser() && !isDashboardSystemRole(grantee->getName())) {
      roles.emplace(grantee->getName());
    }
  }
  return roles;
}

void SysCatalog::buildRoleMapUnlocked() {
  granteeMap_.clear();
  string roleQuery(
      "SELECT roleName, roleType, objectPermissionsType, dbId, objectId, "
      "objectPermissions, objectOwnerId, objectName "
      "from mapd_object_permissions");
  sqliteConnector_->query(roleQuery);
  size_t numRows = sqliteConnector_->getNumRows();
  std::vector<std::string> objectKeyStr(4);
  DBObjectKey objectKey;
  AccessPrivileges privs;
  bool userPrivateRole{false};
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
      std::unique_ptr<Grantee> g;
      if (userPrivateRole) {
        g.reset(new User(roleName));
      } else {
        g.reset(new Role(roleName));
      }
      rl = g.get();
      granteeMap_[to_upper(roleName)] = std::move(g);
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

void SysCatalog::buildUserRoleMapUnlocked() {
  std::vector<std::pair<std::string, std::string>> granteeRoles;
  string userRoleQuery("SELECT roleName, userName from mapd_roles");
  sqliteConnector_->query(userRoleQuery);
  size_t numRows = sqliteConnector_->getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    std::string roleName = sqliteConnector_->getData<string>(r, 0);
    std::string userName = sqliteConnector_->getData<string>(r, 1);
    // required for declared nomenclature before v4.0.0
    if ((boost::equals(roleName, "mapd_default_suser_role") &&
         boost::equals(userName, shared::kRootUsername)) ||
        (boost::equals(roleName, "mapd_default_user_role") &&
         !boost::equals(userName, "mapd_default_user_role"))) {
      // grouprole already exists with roleName==userName in mapd_roles table
      // ignore duplicate instances of userRole which exists before v4.0.0
      continue;
    }
    auto* rl = getGrantee(roleName);
    if (!rl) {
      throw runtime_error("Data inconsistency when building role map. Role " + roleName +
                          " from db not found in the map.");
    }
    std::pair<std::string, std::string> roleVecElem(roleName, userName);
    granteeRoles.push_back(roleVecElem);
  }

  for (const auto& [roleName, granteeName] : granteeRoles) {
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

void SysCatalog::buildObjectDescriptorMapUnlocked() {
  objectDescriptorMap_.clear();
  string objectQuery(
      "SELECT roleName, roleType, objectPermissionsType, dbId, objectId, "
      "objectPermissions, objectOwnerId, objectName "
      "from mapd_object_permissions");
  sqliteConnector_->query(objectQuery);
  size_t numRows = sqliteConnector_->getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    auto od = std::make_unique<ObjectRoleDescriptor>();
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
        std::move(od)));
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

void SysCatalog::createRole(const std::string& roleName,
                            const bool user_private_role,
                            const bool is_temporary) {
  execInTransaction(
      &SysCatalog::createRole_unsafe, roleName, user_private_role, is_temporary);
}

void SysCatalog::dropRole(const std::string& roleName, const bool is_temporary) {
  execInTransaction(&SysCatalog::dropRole_unsafe, roleName, is_temporary);
}

void SysCatalog::grantRoleBatch(const std::vector<std::string>& roles,
                                const std::vector<std::string>& grantees) {
  execInTransaction(&SysCatalog::grantRoleBatch_unsafe, roles, grantees);
}

void SysCatalog::grantRole(const std::string& role,
                           const std::string& grantee,
                           const bool is_temporary) {
  execInTransaction(&SysCatalog::grantRole_unsafe, role, grantee, is_temporary);
}

void SysCatalog::revokeRoleBatch(const std::vector<std::string>& roles,
                                 const std::vector<std::string>& grantees) {
  execInTransaction(&SysCatalog::revokeRoleBatch_unsafe, roles, grantees);
}

void SysCatalog::revokeRole(const std::string& role,
                            const std::string& grantee,
                            const bool is_temporary) {
  execInTransaction(&SysCatalog::revokeRole_unsafe, role, grantee, is_temporary);
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

void SysCatalog::revokeDBObjectPrivilegesFromAllBatch(vector<DBObject>& objects,
                                                      Catalog* catalog) {
  execInTransaction(
      &SysCatalog::revokeDBObjectPrivilegesFromAllBatch_unsafe, objects, catalog);
}

void SysCatalog::syncUserWithRemoteProvider(const std::string& user_name,
                                            std::vector<std::string> idp_roles,
                                            UserAlterations alts) {
  // need to escalate to a write lock
  // need to unlock the read lock
  sys_read_lock read_lock(this);
  read_lock.unlock();
  sys_write_lock write_lock(this);
  bool enable_idp_temporary_users{g_enable_idp_temporary_users && g_read_only};
  if (auto user = getUser(user_name); !user) {
    if (!alts.passwd) {
      alts.passwd = generate_random_string(72);
    }
    user = createUser(user_name, alts, /*is_temporary=*/enable_idp_temporary_users);
    LOG(INFO) << "Remote identity provider created user [" << user->userLoggable()
              << "] with (" << alts.toString() << ")";
  } else if (alts.wouldChange(*user)) {
    user = alterUser(user->userName, alts);
    LOG(INFO) << "Remote identity provider altered user [" << user->userLoggable()
              << "] with (" << alts.toString() << ")";
  }
  std::vector<std::string> current_roles = {};
  auto* user_rl = getUserGrantee(user_name);
  if (user_rl) {
    current_roles = user_rl->getRoles();
  }
  std::transform(
      current_roles.begin(), current_roles.end(), current_roles.begin(), to_upper);
  std::transform(idp_roles.begin(), idp_roles.end(), idp_roles.begin(), to_upper);
  std::list<std::string> roles_revoked, roles_granted;
  // first remove obsolete ones
  for (auto& current_role_name : current_roles) {
    if (std::find(idp_roles.begin(), idp_roles.end(), current_role_name) ==
        idp_roles.end()) {
      revokeRole(current_role_name,
                 user_name,
                 /*is_temporary=*/enable_idp_temporary_users);
      roles_revoked.push_back(current_role_name);
    }
  }
  for (auto& role_name : idp_roles) {
    if (std::find(current_roles.begin(), current_roles.end(), role_name) ==
        current_roles.end()) {
      auto* rl = getRoleGrantee(role_name);
      if (rl) {
        grantRole(role_name,
                  user_name,
                  /*is_temporary=*/enable_idp_temporary_users);
        roles_granted.push_back(role_name);
      } else {
        LOG(WARNING) << "Error synchronizing roles for user " << user_name << ": role "
                     << role_name << " does not exist";
      }
    }
  }
  if (roles_granted.empty() && roles_revoked.empty()) {
    LOG(INFO) << "Roles for user " << user_name
              << " are up to date with remote identity provider";
  } else {
    if (!roles_revoked.empty()) {
      LOG(INFO) << "Roles revoked during synchronization with identity provider for user "
                << user_name << ": " << join(roles_revoked, " ");
    }
    if (!roles_granted.empty()) {
      LOG(INFO) << "Roles granted during synchronization with identity provider for user "
                << user_name << ": " << join(roles_granted, " ");
    }
  }
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

std::shared_ptr<Catalog> SysCatalog::getCatalog(const std::string& dbName) {
  dbid_to_cat_map::const_accessor cata;
  if (cat_map_.find(cata, to_upper(dbName))) {
    return cata->second;
  } else {
    Catalog_Namespace::DBMetadata db_meta;
    if (getMetadataForDB(dbName, db_meta)) {
      return getCatalog(db_meta, false);
    } else {
      return nullptr;
    }
  }
}

std::shared_ptr<Catalog> SysCatalog::getCatalog(const int32_t db_id) {
  dbid_to_cat_map::const_accessor cata;
  for (dbid_to_cat_map::iterator cat_it = cat_map_.begin(); cat_it != cat_map_.end();
       ++cat_it) {
    if (cat_it->second->getDatabaseId() == db_id) {
      return cat_it->second;
    }
  }
  return nullptr;
}

std::shared_ptr<Catalog> SysCatalog::getCatalog(const DBMetadata& curDB, bool is_new_db) {
  const auto key = to_upper(curDB.dbName);
  {
    dbid_to_cat_map::const_accessor cata;
    if (cat_map_.find(cata, key)) {
      return cata->second;
    }
  }

  // Catalog doesnt exist
  // has to be made outside of lock as migration uses cat
  auto cat = std::make_shared<Catalog>(
      basePath_, curDB, dataMgr_, string_dict_hosts_, calciteMgr_, is_new_db);

  dbid_to_cat_map::accessor cata;

  if (cat_map_.find(cata, key)) {
    return cata->second;
  }

  cat_map_.insert(cata, key);
  cata->second = cat;

  return cat;
}

void SysCatalog::removeCatalog(const std::string& dbName) {
  cat_map_.erase(to_upper(dbName));
}

void SysCatalog::reassignObjectOwners(
    const std::map<int32_t, std::vector<DBObject>>& old_owner_db_objects,
    int32_t new_owner_id,
    const Catalog_Namespace::Catalog& catalog) {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);

  sqliteConnector_->query("BEGIN TRANSACTION");
  try {
    UserMetadata new_owner;
    CHECK(getMetadataForUserById(new_owner_id, new_owner));
    for (const auto& [old_owner_id, db_objects] : old_owner_db_objects) {
      UserMetadata old_owner;
      CHECK(getMetadataForUserById(old_owner_id, old_owner));
      if (!old_owner.isSuper) {
        revokeDBObjectPrivilegesBatch_unsafe({old_owner.userName}, db_objects, catalog);
      }
      if (!new_owner.isSuper) {
        grantDBObjectPrivilegesBatch_unsafe({new_owner.userName}, db_objects, catalog);
      }
    }

    std::set<int32_t> old_owner_ids;
    for (const auto& [old_owner_id, db_objects] : old_owner_db_objects) {
      old_owner_ids.emplace(old_owner_id);
    }

    auto db_id = catalog.getDatabaseId();
    for (const auto old_user_id : old_owner_ids) {
      sqliteConnector_->query_with_text_params(
          "UPDATE mapd_object_permissions SET objectOwnerId = ? WHERE objectOwnerId = ? "
          "AND dbId = ? AND objectId != -1",
          std::vector<std::string>{std::to_string(new_owner_id),
                                   std::to_string(old_user_id),
                                   std::to_string(db_id)});
    }

    for (const auto& [user_or_role, grantee] : granteeMap_) {
      grantee->reassignObjectOwners(old_owner_ids, new_owner_id, db_id);
    }

    for (const auto& [object_key, object_descriptor] : objectDescriptorMap_) {
      if (object_descriptor->objectId != -1 && object_descriptor->dbId == db_id &&
          shared::contains(old_owner_ids, object_descriptor->objectOwnerId)) {
        object_descriptor->objectOwnerId = new_owner_id;
      }
    }
  } catch (std::exception& e) {
    sqliteConnector_->query("ROLLBACK TRANSACTION");
    rebuildObjectMapsUnlocked();
    throw;
  }
  sqliteConnector_->query("END TRANSACTION");
}

void SysCatalog::initializeInformationSchemaDb() {
  if (g_enable_system_tables && !hasExecutedMigration(shared::kInfoSchemaMigrationName)) {
    sys_write_lock write_lock(this);
    DBMetadata db_metadata;
    if (getMetadataForDB(shared::kInfoSchemaDbName, db_metadata)) {
      LOG(WARNING) << "A database with name \"" << shared::kInfoSchemaDbName
                   << "\" already exists. System table creation will be skipped. Rename "
                      "this database in order to use system tables.";
    } else {
      createDatabase(shared::kInfoSchemaDbName, shared::kRootUserId);
      try {
        recordExecutedMigration(shared::kInfoSchemaMigrationName);
      } catch (...) {
        getMetadataForDB(shared::kInfoSchemaDbName, db_metadata);
        dropDatabase(db_metadata);
        throw;
      }
    }
  }
}

bool SysCatalog::hasExecutedMigration(const std::string& migration_name) const {
  if (hasVersionHistoryTable()) {
    sys_sqlite_lock sqlite_lock(this);
    sqliteConnector_->query_with_text_params(
        "SELECT migration_history FROM mapd_version_history WHERE migration_history = ?",
        std::vector<std::string>{migration_name});
    return (sqliteConnector_->getNumRows() > 0);
  }
  return false;
}

void SysCatalog::recordExecutedMigration(const std::string& migration_name) const {
  if (!hasVersionHistoryTable()) {
    createVersionHistoryTable();
  }
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query_with_text_params(
      "INSERT INTO mapd_version_history(version, migration_history) values(?, ?)",
      std::vector<std::string>{std::to_string(MAPD_VERSION), migration_name});
}

bool SysCatalog::hasVersionHistoryTable() const {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query(
      "select name from sqlite_master WHERE type='table' AND "
      "name='mapd_version_history'");
  return (sqliteConnector_->getNumRows() > 0);
}

void SysCatalog::createVersionHistoryTable() const {
  sys_sqlite_lock sqlite_lock(this);
  sqliteConnector_->query(
      "CREATE TABLE mapd_version_history(version integer, migration_history text "
      "unique)");
}

void SysCatalog::rebuildObjectMapsUnlocked() {
  // Rebuild updated maps from storage
  granteeMap_.clear();
  buildRoleMapUnlocked();
  objectDescriptorMap_.clear();
  buildObjectDescriptorMapUnlocked();
}

void SysCatalog::checkDropRenderGroupColumnsMigration() const {
  sys_write_lock write_lock(this);
  sys_sqlite_lock sqlite_lock(this);
  static const std::string drop_render_groups_migration{"drop_render_groups"};
  if (!hasExecutedMigration(drop_render_groups_migration)) {
    auto cats = Catalog_Namespace::SysCatalog::instance().getCatalogsForAllDbs();
    for (auto& cat : cats) {
      cat->checkDropRenderGroupColumnsMigration();
    }
    recordExecutedMigration(drop_render_groups_migration);
  }
}

const TableDescriptor* get_metadata_for_table(const ::shared::TableKey& table_key,
                                              bool populate_fragmenter) {
  const auto catalog = SysCatalog::instance().getCatalog(table_key.db_id);
  CHECK(catalog);
  return catalog->getMetadataForTable(table_key.table_id, populate_fragmenter);
}

const ColumnDescriptor* get_metadata_for_column(const ::shared::ColumnKey& column_key) {
  const auto catalog = SysCatalog::instance().getCatalog(column_key.db_id);
  CHECK(catalog);
  return catalog->getMetadataForColumn(column_key.table_id, column_key.column_id);
}
}  // namespace Catalog_Namespace
