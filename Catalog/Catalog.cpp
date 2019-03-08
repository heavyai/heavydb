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
 * @brief		Functions for database metadata access
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "Catalog.h"
#include <algorithm>
#include <cassert>
#include <exception>
#include <list>
#include <memory>
#include <random>
#include <sstream>
#include "SysCatalog.h"

#include "DataMgr/LockMgr.h"
#include "SharedDictionaryValidator.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/adaptor/map.hpp>
#include <boost/version.hpp>
#if BOOST_VERSION >= 106600
#include <boost/uuid/detail/sha1.hpp>
#else
#include <boost/uuid/sha1.hpp>
#endif

#include "../QueryEngine/Execute.h"
#include "../QueryEngine/TableOptimizer.h"

#include "../Fragmenter/Fragmenter.h"
#include "../Fragmenter/InsertOrderFragmenter.h"
#include "../Parser/ParserNode.h"
#include "../Shared/File.h"
#include "../Shared/StringTransform.h"
#include "../Shared/measure.h"
#include "../StringDictionary/StringDictionaryClient.h"
#include "MapDRelease.h"
#include "RWLocks.h"
#include "SharedDictionaryValidator.h"

using Chunk_NS::Chunk;
using Fragmenter_Namespace::InsertOrderFragmenter;
using std::list;
using std::map;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;

int g_test_against_columnId_gap = 0;

namespace Catalog_Namespace {

const int DEFAULT_INITIAL_VERSION = 1;  // start at version 1
const int MAPD_TEMP_TABLE_START_ID =
    1073741824;  // 2^30, give room for over a billion non-temp tables
const int MAPD_TEMP_DICT_START_ID =
    1073741824;  // 2^30, give room for over a billion non-temp dictionaries

const std::string Catalog::physicalTableNameTag_("_shard_#");
std::map<std::string, std::shared_ptr<Catalog>> Catalog::mapd_cat_map_;

thread_local bool Catalog::thread_holds_read_lock = false;

using cat_read_lock = read_lock<Catalog>;
using cat_write_lock = write_lock<Catalog>;
using cat_sqlite_lock = sqlite_lock<Catalog>;

// migration will be done as two step process this release
// will create and use new table
// next release will remove old table, doing this to have fall back path
// incase of migration failure
void Catalog::updateFrontendViewsToDashboards() {
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='mapd_dashboards'");
    if (sqliteConnector_.getNumRows() != 0) {
      // already done
      sqliteConnector_.query("END TRANSACTION");
      return;
    }
    sqliteConnector_.query(
        "CREATE TABLE mapd_dashboards (id integer primary key autoincrement, name text , "
        "userid integer references mapd_users, state text, image_hash text, update_time "
        "timestamp, "
        "metadata text, UNIQUE(userid, name) )");
    // now copy content from old table to new table
    sqliteConnector_.query(
        "insert into mapd_dashboards (id, name , "
        "userid, state, image_hash, update_time , "
        "metadata) "
        "SELECT viewid , name , userid, view_state, image_hash, update_time, "
        "view_metadata "
        "from mapd_frontend_views");
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

Catalog::Catalog(const string& basePath,
                 const DBMetadata& curDB,
                 std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                 const std::vector<LeafHostInfo>& string_dict_hosts,
                 std::shared_ptr<Calcite> calcite,
                 bool is_new_db)
    : basePath_(basePath)
    , sqliteConnector_(curDB.dbName, basePath + "/mapd_catalogs/")
    , currentDB_(curDB)
    , dataMgr_(dataMgr)
    , string_dict_hosts_(string_dict_hosts)
    , calciteMgr_(calcite)
    , nextTempTableId_(MAPD_TEMP_TABLE_START_ID)
    , nextTempDictId_(MAPD_TEMP_DICT_START_ID)
    , sqliteMutex_()
    , sharedMutex_()
    , thread_holding_sqlite_lock()
    , thread_holding_write_lock() {
  if (!is_new_db) {
    CheckAndExecuteMigrations();
  }
  buildMaps();
  if (!is_new_db) {
    CheckAndExecuteMigrationsPostBuildMaps();
  }
}

Catalog::~Catalog() {
  cat_write_lock write_lock(this);
  // must clean up heap-allocated TableDescriptor and ColumnDescriptor structs
  for (TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.begin();
       tableDescIt != tableDescriptorMap_.end();
       ++tableDescIt) {
    if (tableDescIt->second->fragmenter != nullptr) {
      delete tableDescIt->second->fragmenter;
    }
    delete tableDescIt->second;
  }

  // TableDescriptorMapById points to the same descriptors.  No need to delete

  for (ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin();
       columnDescIt != columnDescriptorMap_.end();
       ++columnDescIt) {
    delete columnDescIt->second;
  }

  // ColumnDescriptorMapById points to the same descriptors.  No need to delete
}

void Catalog::updateTableDescriptorSchema() {
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_tables)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("max_chunk_size")) ==
        cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD max_chunk_size BIGINT DEFAULT " +
                         std::to_string(DEFAULT_MAX_CHUNK_SIZE));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("shard_column_id")) ==
        cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD shard_column_id BIGINT DEFAULT " +
                         std::to_string(0));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("shard")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD shard BIGINT DEFAULT " +
                         std::to_string(-1));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("num_shards")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD num_shards BIGINT DEFAULT " +
                         std::to_string(0));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("key_metainfo")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD key_metainfo TEXT DEFAULT '[]'");
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("userid")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD userid integer DEFAULT " +
                         std::to_string(MAPD_ROOT_USER_ID));
      sqliteConnector_.query(queryString);
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateFrontendViewSchema() {
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    // check table still exists
    sqliteConnector_.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND "
        "name='mapd_frontend_views'");
    if (sqliteConnector_.getNumRows() == 0) {
      // table does not exists
      // no need to migrate
      sqliteConnector_.query("END TRANSACTION");
      return;
    }
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
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "CREATE TABLE IF NOT EXISTS mapd_links (linkid integer primary key, userid "
        "integer references mapd_users, "
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
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("UPDATE mapd_links SET userid = 0 WHERE userid IS NULL");
    // check table still exists
    sqliteConnector_.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND "
        "name='mapd_frontend_views'");
    if (sqliteConnector_.getNumRows() == 0) {
      // table does not exists
      // no need to migrate
      sqliteConnector_.query("END TRANSACTION");
      return;
    }
    sqliteConnector_.query(
        "UPDATE mapd_frontend_views SET userid = 0 WHERE userid IS NULL");
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

// introduce DB version into the tables table
// if the DB does not have a version reset all pagesizes to 2097152 to be compatible with
// old value

void Catalog::updatePageSize() {
  cat_sqlite_lock sqlite_lock(this);
  if (currentDB_.dbName.length() == 0) {
    // updateDictionaryNames dbName length is zero nothing to do here
    return;
  }
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_tables)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("version_num")) == cols.end()) {
      LOG(INFO) << "Updating mapd_tables updatePageSize";
      // No version number
      // need to update the defaul tpagesize to old correct value
      sqliteConnector_.query("UPDATE mapd_tables SET frag_page_size = 2097152 ");
      // need to add new version info
      string queryString("ALTER TABLE mapd_tables ADD version_num BIGINT DEFAULT " +
                         std::to_string(DEFAULT_INITIAL_VERSION));
      sqliteConnector_.query(queryString);
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateDeletedColumnIndicator() {
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_columns)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("version_num")) == cols.end()) {
      LOG(INFO) << "Updating mapd_columns updateDeletedColumnIndicator";
      // need to add new version info
      string queryString("ALTER TABLE mapd_columns ADD version_num BIGINT DEFAULT " +
                         std::to_string(DEFAULT_INITIAL_VERSION));
      sqliteConnector_.query(queryString);
      // need to add new column to table defintion to indicate deleted column, column used
      // as bitmap for deleted rows.
      sqliteConnector_.query(
          "ALTER TABLE mapd_columns  ADD is_deletedcol boolean default 0 ");
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

// introduce DB version into the dictionary tables
// if the DB does not have a version rename all dictionary tables

void Catalog::updateDictionaryNames() {
  cat_sqlite_lock sqlite_lock(this);
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

        std::string oldName =
            basePath_ + "/mapd_data/" + currentDB_.dbName + "_" + dictName;
        std::string newName = basePath_ + "/mapd_data/DB_" +
                              std::to_string(currentDB_.dbId) + "_DICT_" +
                              std::to_string(dictId);

        int result = rename(oldName.c_str(), newName.c_str());

        if (result == 0) {
          LOG(INFO) << "Dictionary upgrade: successfully renamed " << oldName << " to "
                    << newName;
        } else {
          LOG(ERROR) << "Failed to rename old dictionary directory " << oldName << " to "
                     << newName + " dbname '" << currentDB_.dbName << "' error code "
                     << std::to_string(result);
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
  cat_sqlite_lock sqlite_lock(this);
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

  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    const auto physicalTableIt = logicalToPhysicalTableMapById_.find(logical_tb_id);
    if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
      const auto physicalTables = physicalTableIt->second;
      CHECK(!physicalTables.empty());
      for (size_t i = 0; i < physicalTables.size(); i++) {
        int32_t physical_tb_id = physicalTables[i];
        sqliteConnector_.query_with_text_params(
            "INSERT OR REPLACE INTO mapd_logical_to_physical (logical_table_id, "
            "physical_table_id) VALUES (?1, ?2)",
            std::vector<std::string>{std::to_string(logical_tb_id),
                                     std::to_string(physical_tb_id)});
      }
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateDictionarySchema() {
  cat_sqlite_lock sqlite_lock(this);
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

void Catalog::recordOwnershipOfObjectsInObjectPermissions() {
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  std::vector<DBObject> objects;
  try {
    sqliteConnector_.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND "
        "name='mapd_record_ownership_marker'");
    // check if mapd catalog - marker exists
    if (sqliteConnector_.getNumRows() != 0 && currentDB_.dbId == 1) {
      // already done
      sqliteConnector_.query("END TRANSACTION");
      return;
    }
    // check if different catalog - marker exists
    else if (sqliteConnector_.getNumRows() != 0 && currentDB_.dbId != 1) {
      sqliteConnector_.query("SELECT dummy FROM mapd_record_ownership_marker");
      // Check if migration is being performed on existing non mapd catalogs
      // Older non mapd dbs will have table but no record in them
      if (sqliteConnector_.getNumRows() != 0) {
        // already done
        sqliteConnector_.query("END TRANSACTION");
        return;
      }
    }
    // marker not exists - create one
    else {
      sqliteConnector_.query("CREATE TABLE mapd_record_ownership_marker (dummy integer)");
    }

    DBMetadata db;
    CHECK(SysCatalog::instance().getMetadataForDB(currentDB_.dbName, db));
    // place dbId as a refernce for migration being performed
    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_record_ownership_marker (dummy) VALUES (?1)",
        std::vector<std::string>{std::to_string(db.dbOwner)});

    static const std::map<const DBObjectType, const AccessPrivileges>
        object_level_all_privs_lookup{
            {DatabaseDBObjectType, AccessPrivileges::ALL_DATABASE},
            {TableDBObjectType, AccessPrivileges::ALL_TABLE},
            {DashboardDBObjectType, AccessPrivileges::ALL_DASHBOARD},
            {ViewDBObjectType, AccessPrivileges::ALL_VIEW}};

    // grant owner all permissions on DB
    DBObjectKey key;
    key.dbId = currentDB_.dbId;
    auto _key_place = [&key](auto type) {
      key.permissionType = type;
      return key;
    };
    for (auto& it : object_level_all_privs_lookup) {
      objects.emplace_back(_key_place(it.first), it.second, db.dbOwner);
      objects.back().setName(currentDB_.dbName);
    }

    {
      // other users tables and views
      string tableQuery(
          "SELECT tableid, name, userid, isview FROM mapd_tables WHERE userid > 0");
      sqliteConnector_.query(tableQuery);
      size_t numRows = sqliteConnector_.getNumRows();
      for (size_t r = 0; r < numRows; ++r) {
        int32_t tableid = sqliteConnector_.getData<int>(r, 0);
        std::string tableName = sqliteConnector_.getData<string>(r, 1);
        int32_t ownerid = sqliteConnector_.getData<int>(r, 2);
        bool isview = sqliteConnector_.getData<bool>(r, 3);

        DBObjectType type =
            isview ? DBObjectType::ViewDBObjectType : DBObjectType::TableDBObjectType;
        DBObjectKey key;
        key.dbId = currentDB_.dbId;
        key.objectId = tableid;
        key.permissionType = type;

        DBObject obj(tableName, type);
        obj.setObjectKey(key);
        obj.setOwner(ownerid);
        obj.setPrivileges(isview ? AccessPrivileges::ALL_VIEW
                                 : AccessPrivileges::ALL_TABLE);

        objects.push_back(obj);
      }
    }

    {
      // other users dashboards
      string tableQuery("SELECT id, name, userid FROM mapd_dashboards WHERE userid > 0");
      sqliteConnector_.query(tableQuery);
      size_t numRows = sqliteConnector_.getNumRows();
      for (size_t r = 0; r < numRows; ++r) {
        int32_t dashId = sqliteConnector_.getData<int>(r, 0);
        std::string dashName = sqliteConnector_.getData<string>(r, 1);
        int32_t ownerid = sqliteConnector_.getData<int>(r, 2);

        DBObjectType type = DBObjectType::DashboardDBObjectType;
        DBObjectKey key;
        key.dbId = currentDB_.dbId;
        key.objectId = dashId;
        key.permissionType = type;

        DBObject obj(dashName, type);
        obj.setObjectKey(key);
        obj.setOwner(ownerid);
        obj.setPrivileges(AccessPrivileges::ALL_DASHBOARD);

        objects.push_back(obj);
      }
    }
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");

  // now apply the objects to the syscat to track the permisisons
  // moved outside transaction to avoid lock in sqlite
  try {
    SysCatalog::instance().populateRoleDbObjects(objects);
  } catch (const std::exception& e) {
    LOG(ERROR) << " Issue during migration of DB " << name() << " issue was " << e.what();
    throw std::runtime_error(" Issue during migration of DB " + name() + " issue was " +
                             e.what());
    // will need to remove the mapd_record_ownership_marker table and retry
  }
}

void Catalog::checkDateInDaysColumnMigration() {
  cat_sqlite_lock sqlite_lock(this);
  std::vector<int> tables_migrated = {};
  std::unordered_map<int, std::vector<std::string>> tables_to_migrate;
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "select name from sqlite_master WHERE type='table' AND "
        "name='mapd_version_history'");
    if (sqliteConnector_.getNumRows() == 0) {
      sqliteConnector_.query(
          "CREATE TABLE mapd_version_history(version integer, migration_history text "
          "unique)");
      sqliteConnector_.query(
          "CREATE TABLE mapd_date_in_days_column_migration_tmp(table_id integer primary "
          "key)");
    } else {
      sqliteConnector_.query(
          "select * from mapd_version_history where migration_history = "
          "'date_in_days_column'");
      if (sqliteConnector_.getNumRows() != 0) {
        // no need for further execution
        sqliteConnector_.query("END TRANSACTION");
        return;
      }
      LOG(INFO) << "Performing Date in days columns migration.";
      sqliteConnector_.query(
          "select name from sqlite_master where type='table' AND "
          "name='mapd_date_in_days_column_migration_tmp'");
      if (sqliteConnector_.getNumRows() != 0) {
        sqliteConnector_.query(
            "select table_id from mapd_date_in_days_column_migration_tmp");
        if (sqliteConnector_.getNumRows() != 0) {
          for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
            tables_migrated.push_back(sqliteConnector_.getData<int>(i, 0));
          }
        }
      } else {
        sqliteConnector_.query(
            "CREATE TABLE mapd_date_in_days_column_migration_tmp(table_id integer "
            "primary key)");
      }
    }
    sqliteConnector_.query_with_text_params(
        "SELECT tables.tableid, tables.name, columns.name FROM mapd_tables tables, "
        "mapd_columns columns where tables.tableid = columns.tableid AND "
        "columns.coltype = ?1 AND columns.compression = ?2",
        std::vector<std::string>{
            std::to_string(static_cast<int>(SQLTypes::kDATE)),
            std::to_string(static_cast<int>(EncodingType::kENCODING_DATE_IN_DAYS))});
    if (sqliteConnector_.getNumRows() != 0) {
      for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
        tables_to_migrate[sqliteConnector_.getData<int>(i, 0)] = {
            sqliteConnector_.getData<std::string>(i, 1),
            sqliteConnector_.getData<std::string>(i, 2)};
      }
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to complete migraion on date in days column: " << e.what();
    sqliteConnector_.query("ROLLBACK");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");

  for (auto& id_names : tables_to_migrate) {
    if (std::find(tables_migrated.begin(), tables_migrated.end(), id_names.first) ==
        tables_migrated.end()) {
      sqliteConnector_.query("BEGIN TRANSACTION");
      try {
        LOG(INFO) << "Table: " << id_names.second[0]
                  << " may suffer from issues with DATE column: " << id_names.second[1]
                  << ". Running an OPTIMIZE command to solve any issues with metadata.";

        auto executor = Executor::getExecutor(getCurrentDB().dbId);
        TableDescriptorMapById::iterator tableDescIt =
            tableDescriptorMapById_.find(id_names.first);
        if (tableDescIt == tableDescriptorMapById_.end()) {
          throw runtime_error("Table descriptor does not exist for table " +
                              id_names.second[0] + " does not exist.");
        }
        auto td = tableDescIt->second;
        TableOptimizer optimizer(td, executor.get(), *this);
        optimizer.recomputeMetadata();

        sqliteConnector_.query_with_text_params(
            "INSERT INTO mapd_date_in_days_column_migration_tmp VALUES(?)",
            std::vector<std::string>{std::to_string(id_names.first)});
      } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to complete migraion on date in days column: " << e.what();
        sqliteConnector_.query("ROLLBACK");
        throw;
      }
      sqliteConnector_.query("COMMIT");
    }
  }

  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("DROP TABLE mapd_date_in_days_column_migration_tmp");
    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_version_history(version, migration_history) values(?,?)",
        std::vector<std::string>{std::to_string(MAPD_VERSION), "date_in_days_column"});
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to complete migraion on date in days column: " << e.what();
    sqliteConnector_.query("ROLLBACK");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  LOG(INFO) << "Migration successfull on Date in days columns";
}

void Catalog::createDashboardSystemRoles() {
  std::unordered_map<std::string, std::pair<int, std::string>> dashboards;
  std::vector<std::string> dashboard_ids;
  static const std::string migration_name{"dashboard_roles_migration"};
  {
    cat_sqlite_lock sqlite_lock(this);
    sqliteConnector_.query("BEGIN TRANSACTION");
    try {
      // migration_history should be present in all catalogs by now
      // if not then would be created before this migration
      sqliteConnector_.query(
          "select * from mapd_version_history where migration_history = '" +
          migration_name + "'");
      if (sqliteConnector_.getNumRows() != 0) {
        // no need for further execution
        sqliteConnector_.query("END TRANSACTION");
        return;
      }
      LOG(INFO) << "Performing dashboard internal roles Migration.";
      sqliteConnector_.query("select id, userid, metadata from mapd_dashboards");
      for (size_t i = 0; i < sqliteConnector_.getNumRows(); ++i) {
        if (SysCatalog::instance().getRoleGrantee(
                generate_dash_system_rolename(std::to_string(currentDB_.dbId),
                                              sqliteConnector_.getData<string>(i, 0)))) {
          // Successfully created roles during previous migration/crash
          // No need to include them
          continue;
        }
        dashboards[sqliteConnector_.getData<string>(i, 0)] = std::make_pair(
            sqliteConnector_.getData<int>(i, 1), sqliteConnector_.getData<string>(i, 2));
        dashboard_ids.push_back(sqliteConnector_.getData<string>(i, 0));
      }
    } catch (const std::exception& e) {
      sqliteConnector_.query("ROLLBACK TRANSACTION");
      throw;
    }
    sqliteConnector_.query("END TRANSACTION");
  }
  // All current grantees with shared dashboards.
  const auto active_grantees =
      SysCatalog::instance().getGranteesOfSharedDashboards(dashboard_ids);

  try {
    // NOTE(wamsi): Transactionally unsafe
    for (auto dash : dashboards) {
      createOrUpdateDashboardSystemRole(
          dash.second.second,
          dash.second.first,
          generate_dash_system_rolename(std::to_string(currentDB_.dbId), dash.first));
      auto result = active_grantees.find(dash.first);
      if (result != active_grantees.end()) {
        SysCatalog::instance().grantRoleBatch(
            {generate_dash_system_rolename(std::to_string(currentDB_.dbId), dash.first)},
            result->second);
      }
    }
    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_version_history(version, migration_history) values(?,?)",
        std::vector<std::string>{std::to_string(MAPD_VERSION), migration_name});
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to create dashboard system roles during migration: "
               << e.what();
    throw;
  }
  LOG(INFO) << "Successfully created dashboard system roles during migration.";
}

void Catalog::CheckAndExecuteMigrations() {
  updateTableDescriptorSchema();
  updateFrontendViewAndLinkUsers();
  updateFrontendViewSchema();
  updateLinkSchema();
  updateDictionaryNames();
  updateLogicalToPhysicalTableLinkSchema();
  updateDictionarySchema();
  updatePageSize();
  updateDeletedColumnIndicator();
  updateFrontendViewsToDashboards();
  recordOwnershipOfObjectsInObjectPermissions();
}

void Catalog::CheckAndExecuteMigrationsPostBuildMaps() {
  checkDateInDaysColumnMigration();
  createDashboardSystemRoles();
}

namespace {
std::string getUserFromId(const int32_t id) {
  UserMetadata user;
  if (SysCatalog::instance().getMetadataForUserById(id, user)) {
    return user.userName;
  }
  // a user could be deleted and a dashboard still exist?
  return "Unknown";
}
}  // namespace

void Catalog::buildMaps() {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);

  string dictQuery(
      "SELECT dictid, name, nbits, is_shared, refcount from mapd_dictionaries");
  sqliteConnector_.query(dictQuery);
  size_t numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    int dictId = sqliteConnector_.getData<int>(r, 0);
    std::string dictName = sqliteConnector_.getData<string>(r, 1);
    int dictNBits = sqliteConnector_.getData<int>(r, 2);
    bool is_shared = sqliteConnector_.getData<bool>(r, 3);
    int refcount = sqliteConnector_.getData<int>(r, 4);
    std::string fname = basePath_ + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) +
                        "_DICT_" + std::to_string(dictId);
    DictRef dict_ref(currentDB_.dbId, dictId);
    DictDescriptor* dd = new DictDescriptor(
        dict_ref, dictName, dictNBits, is_shared, refcount, fname, false);
    dictDescriptorMapByRef_[dict_ref].reset(dd);
  }

  string tableQuery(
      "SELECT tableid, name, ncolumns, isview, fragments, frag_type, max_frag_rows, "
      "max_chunk_size, frag_page_size, "
      "max_rows, partitions, shard_column_id, shard, num_shards, key_metainfo, userid "
      "from mapd_tables");
  sqliteConnector_.query(tableQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    TableDescriptor* td = new TableDescriptor();
    td->tableId = sqliteConnector_.getData<int>(r, 0);
    td->tableName = sqliteConnector_.getData<string>(r, 1);
    td->nColumns = sqliteConnector_.getData<int>(r, 2);
    td->isView = sqliteConnector_.getData<bool>(r, 3);
    td->fragments = sqliteConnector_.getData<string>(r, 4);
    td->fragType =
        (Fragmenter_Namespace::FragmenterType)sqliteConnector_.getData<int>(r, 5);
    td->maxFragRows = sqliteConnector_.getData<int>(r, 6);
    td->maxChunkSize = sqliteConnector_.getData<int>(r, 7);
    td->fragPageSize = sqliteConnector_.getData<int>(r, 8);
    td->maxRows = sqliteConnector_.getData<int64_t>(r, 9);
    td->partitions = sqliteConnector_.getData<string>(r, 10);
    td->shardedColumnId = sqliteConnector_.getData<int>(r, 11);
    td->shard = sqliteConnector_.getData<int>(r, 12);
    td->nShards = sqliteConnector_.getData<int>(r, 13);
    td->keyMetainfo = sqliteConnector_.getData<string>(r, 14);
    td->userId = sqliteConnector_.getData<int>(r, 15);
    if (!td->isView) {
      td->fragmenter = nullptr;
    }
    td->hasDeletedCol = false;
    tableDescriptorMap_[to_upper(td->tableName)] = td;
    tableDescriptorMapById_[td->tableId] = td;
  }
  string columnQuery(
      "SELECT tableid, columnid, name, coltype, colsubtype, coldim, colscale, "
      "is_notnull, compression, comp_param, "
      "size, chunks, is_systemcol, is_virtualcol, virtual_expr, is_deletedcol from "
      "mapd_columns ORDER BY tableid, "
      "columnid");
  sqliteConnector_.query(columnQuery);
  numRows = sqliteConnector_.getNumRows();
  int32_t skip_physical_cols = 0;
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
    cd->isDeletedCol = sqliteConnector_.getData<bool>(r, 15);
    cd->isGeoPhyCol = skip_physical_cols > 0;
    ColumnKey columnKey(cd->tableId, to_upper(cd->columnName));
    columnDescriptorMap_[columnKey] = cd;
    ColumnIdKey columnIdKey(cd->tableId, cd->columnId);
    columnDescriptorMapById_[columnIdKey] = cd;

    if (skip_physical_cols <= 0) {
      skip_physical_cols = cd->columnType.get_physical_cols();
    }

    auto td_itr = tableDescriptorMapById_.find(cd->tableId);
    CHECK(td_itr != tableDescriptorMapById_.end());

    if (cd->isDeletedCol) {
      td_itr->second->hasDeletedCol = true;
      setDeletedColumnUnlocked(td_itr->second, cd);
    } else if (cd->columnType.is_geometry() || skip_physical_cols-- <= 0) {
      tableDescriptorMapById_[cd->tableId]->columnIdBySpi_.push_back(cd->columnId);
    }
  }
  // sort columnIdBySpi_ based on columnId
  for (auto& tit : tableDescriptorMapById_) {
    std::sort(tit.second->columnIdBySpi_.begin(),
              tit.second->columnIdBySpi_.end(),
              [](const size_t a, const size_t b) -> bool { return a < b; });
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
      "SELECT id, state, name, image_hash, strftime('%Y-%m-%dT%H:%M:%SZ', update_time), "
      "userid, "
      "metadata "
      "FROM mapd_dashboards");
  sqliteConnector_.query(frontendViewQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    std::shared_ptr<FrontendViewDescriptor> vd =
        std::make_shared<FrontendViewDescriptor>();
    vd->viewId = sqliteConnector_.getData<int>(r, 0);
    vd->viewState = sqliteConnector_.getData<string>(r, 1);
    vd->viewName = sqliteConnector_.getData<string>(r, 2);
    vd->imageHash = sqliteConnector_.getData<string>(r, 3);
    vd->updateTime = sqliteConnector_.getData<string>(r, 4);
    vd->userId = sqliteConnector_.getData<int>(r, 5);
    vd->viewMetadata = sqliteConnector_.getData<string>(r, 6);
    vd->user = getUserFromId(vd->userId);
    vd->viewSystemRoleName = generate_dash_system_rolename(
        std::to_string(currentDB_.dbId), sqliteConnector_.getData<string>(r, 0));
    dashboardDescriptorMap_[std::to_string(vd->userId) + ":" + vd->viewName] = vd;
  }

  string linkQuery(
      "SELECT linkid, userid, link, view_state, strftime('%Y-%m-%dT%H:%M:%SZ', "
      "update_time), view_metadata "
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
      const auto it_ok =
          logicalToPhysicalTableMapById_.emplace(logical_tb_id, physicalTables);
      CHECK(it_ok.second);
    } else {
      /* update map logicalToPhysicalTableMapById_ */
      physicalTableIt->second.push_back(physical_tb_id);
    }
  }
}

void Catalog::addTableToMap(TableDescriptor& td,
                            const list<ColumnDescriptor>& columns,
                            const list<DictDescriptor>& dicts) {
  cat_write_lock write_lock(this);
  TableDescriptor* new_td = new TableDescriptor();
  *new_td = td;
  new_td->mutex_ = std::make_shared<std::mutex>();
  tableDescriptorMap_[to_upper(td.tableName)] = new_td;
  tableDescriptorMapById_[td.tableId] = new_td;
  for (auto cd : columns) {
    ColumnDescriptor* new_cd = new ColumnDescriptor();
    *new_cd = cd;
    ColumnKey columnKey(new_cd->tableId, to_upper(new_cd->columnName));
    columnDescriptorMap_[columnKey] = new_cd;
    ColumnIdKey columnIdKey(new_cd->tableId, new_cd->columnId);
    columnDescriptorMapById_[columnIdKey] = new_cd;

    // Add deleted column to the map
    if (cd.isDeletedCol) {
      CHECK(new_td->hasDeletedCol);
      setDeletedColumnUnlocked(new_td, new_cd);
    }
  }

  std::sort(new_td->columnIdBySpi_.begin(),
            new_td->columnIdBySpi_.end(),
            [](const size_t a, const size_t b) -> bool { return a < b; });

  std::unique_ptr<StringDictionaryClient> client;
  DictRef dict_ref(currentDB_.dbId, -1);
  if (!string_dict_hosts_.empty()) {
    client.reset(new StringDictionaryClient(string_dict_hosts_.front(), dict_ref, true));
  }
  for (auto dd : dicts) {
    if (!dd.dictRef.dictId) {
      // Dummy entry created for a shard of a logical table, nothing to do.
      continue;
    }
    dict_ref.dictId = dd.dictRef.dictId;
    if (client) {
      client->create(dict_ref, dd.dictIsTemp);
    }
    DictDescriptor* new_dd = new DictDescriptor(dd);
    dictDescriptorMapByRef_[dict_ref].reset(new_dd);
    if (!dd.dictIsTemp) {
      boost::filesystem::create_directory(new_dd->dictFolderPath);
    }
  }
}

void Catalog::removeTableFromMap(const string& tableName, int tableId) {
  cat_write_lock write_lock(this);
  TableDescriptorMapById::iterator tableDescIt = tableDescriptorMapById_.find(tableId);
  if (tableDescIt == tableDescriptorMapById_.end()) {
    throw runtime_error("Table " + tableName + " does not exist.");
  }

  TableDescriptor* td = tableDescIt->second;

  if (td->hasDeletedCol) {
    const auto ret = deletedColumnPerTable_.erase(td);
    CHECK_EQ(ret, size_t(1));
  }

  tableDescriptorMapById_.erase(tableDescIt);
  tableDescriptorMap_.erase(to_upper(tableName));
  if (td->fragmenter != nullptr) {
    delete td->fragmenter;
  }
  bool isTemp = td->persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL;
  delete td;

  std::unique_ptr<StringDictionaryClient> client;
  if (SysCatalog::instance().isAggregator()) {
    CHECK(!string_dict_hosts_.empty());
    DictRef dict_ref(currentDB_.dbId, -1);
    client.reset(new StringDictionaryClient(string_dict_hosts_.front(), dict_ref, true));
  }

  // delete all column descriptors for the table
  // no more link columnIds to sequential indexes!
  for (auto cit = columnDescriptorMapById_.begin();
       cit != columnDescriptorMapById_.end();) {
    if (tableId != std::get<0>(cit->first)) {
      ++cit;
    } else {
      int i = std::get<1>(cit++->first);
      ColumnIdKey cidKey(tableId, i);
      ColumnDescriptorMapById::iterator colDescIt = columnDescriptorMapById_.find(cidKey);
      ColumnDescriptor* cd = colDescIt->second;
      columnDescriptorMapById_.erase(colDescIt);
      ColumnKey cnameKey(tableId, to_upper(cd->columnName));
      columnDescriptorMap_.erase(cnameKey);
      const int dictId = cd->columnType.get_comp_param();
      // Dummy dictionaries created for a shard of a logical table have the id set to
      // zero.
      if (cd->columnType.get_compression() == kENCODING_DICT && dictId) {
        INJECT_TIMER(removingDicts);
        DictRef dict_ref(currentDB_.dbId, dictId);
        const auto dictIt = dictDescriptorMapByRef_.find(dict_ref);
        CHECK(dictIt != dictDescriptorMapByRef_.end());
        const auto& dd = dictIt->second;
        CHECK_GE(dd->refcount, 1);
        --dd->refcount;
        if (!dd->refcount) {
          dd->stringDict.reset();
          if (!isTemp) {
            File_Namespace::renameForDelete(dd->dictFolderPath);
          }
          if (client) {
            client->drop(dict_ref);
          }
          dictDescriptorMapByRef_.erase(dictIt);
        }
      }

      delete cd;
    }
  }
}

void Catalog::addFrontendViewToMap(FrontendViewDescriptor& vd) {
  cat_write_lock write_lock(this);
  addFrontendViewToMapNoLock(vd);
}

void Catalog::addFrontendViewToMapNoLock(FrontendViewDescriptor& vd) {
  cat_write_lock write_lock(this);
  dashboardDescriptorMap_[std::to_string(vd.userId) + ":" + vd.viewName] =
      std::make_shared<FrontendViewDescriptor>(vd);
}

std::vector<DBObject> Catalog::parseDashboardObjects(const std::string& view_meta,
                                                     const int& user_id) {
  std::vector<DBObject> objects;
  DBObjectKey key;
  key.dbId = currentDB_.dbId;
  auto _key_place = [&key](auto type, auto id) {
    key.permissionType = type;
    key.objectId = id;
    return key;
  };
  for (auto object_name : parse_underlying_dash_objects(view_meta)) {
    auto td = getMetadataForTable(object_name);
    if (!td) {
      // Parsed object source is not present in current database
      // LOG the info and ignore
      LOG(INFO) << "Ignoring dashboard source Table/View: " << object_name
                << " no longer exists in current DB.";
      continue;
    }
    // Dashboard source can be Table or View
    const auto object_type = td->isView ? ViewDBObjectType : TableDBObjectType;
    const auto priv = td->isView ? AccessPrivileges::SELECT_FROM_VIEW
                                 : AccessPrivileges::SELECT_FROM_TABLE;
    objects.emplace_back(_key_place(object_type, td->tableId), priv, user_id);
    objects.back().setObjectType(td->isView ? ViewDBObjectType : TableDBObjectType);
    objects.back().setName(td->tableName);
  }
  return objects;
}

void Catalog::createOrUpdateDashboardSystemRole(const std::string& view_meta,
                                                const int32_t& user_id,
                                                const std::string& dash_role_name) {
  auto objects = parseDashboardObjects(view_meta, user_id);
  Role* rl = SysCatalog::instance().getRoleGrantee(dash_role_name);
  if (!rl) {
    // Dashboard role does not exist
    // create role and grant privileges
    // NOTE(wamsi): Transactionally unsafe
    SysCatalog::instance().createRole(dash_role_name, false);
    SysCatalog::instance().grantDBObjectPrivilegesBatch({dash_role_name}, objects, *this);
  } else {
    // Dashboard system role already exists
    // Add/remove privileges on objects
    auto ex_objects = rl->getDbObjects(true);
    for (auto key : *ex_objects | boost::adaptors::map_keys) {
      if (key.permissionType != TableDBObjectType &&
          key.permissionType != ViewDBObjectType) {
        continue;
      }
      bool found = false;
      for (auto obj : objects) {
        found = key == obj.getObjectKey() ? true : false;
        if (found) {
          break;
        }
      }
      if (!found) {
        // revoke privs on object since the object is no
        // longer used by the dashboard as source
        // NOTE(wamsi): Transactionally unsafe
        SysCatalog::instance().revokeDBObjectPrivileges(
            dash_role_name, *rl->findDbObject(key, true), *this);
      }
    }
    // Update privileges on remaining objects
    // NOTE(wamsi): Transactionally unsafe
    SysCatalog::instance().grantDBObjectPrivilegesBatch({dash_role_name}, objects, *this);
  }
}

void Catalog::addLinkToMap(LinkDescriptor& ld) {
  cat_write_lock write_lock(this);
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
    getAllColumnMetadataForTable(td, columnDescs, true, false, true);
    Chunk::translateColumnDescriptorsToChunkVec(columnDescs, chunkVec);
    ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId};
    td->fragmenter = new InsertOrderFragmenter(chunkKeyPrefix,
                                               chunkVec,
                                               dataMgr_.get(),
                                               const_cast<Catalog*>(this),
                                               td->tableId,
                                               td->shard,
                                               td->maxFragRows,
                                               td->maxChunkSize,
                                               td->fragPageSize,
                                               td->maxRows,
                                               td->persistenceLevel);
  });
  LOG(INFO) << "Instantiating Fragmenter for table " << td->tableName << " took "
            << time_ms << "ms";
}

const TableDescriptor* Catalog::getMetadataForTable(const string& tableName,
                                                    const bool populateFragmenter) const {
  // we give option not to populate fragmenter (default true/yes) as it can be heavy for
  // pure metadata calls
  cat_read_lock read_lock(this);
  auto tableDescIt = tableDescriptorMap_.find(to_upper(tableName));
  if (tableDescIt == tableDescriptorMap_.end()) {  // check to make sure table exists
    return nullptr;
  }
  TableDescriptor* td = tableDescIt->second;
  std::unique_lock<std::mutex> td_lock(*td->mutex_.get());
  if (populateFragmenter && td->fragmenter == nullptr && !td->isView) {
    instantiateFragmenter(td);
  }
  return td;  // returns pointer to table descriptor
}

const TableDescriptor* Catalog::getMetadataForTableImpl(
    int tableId,
    const bool populateFragmenter) const {
  cat_read_lock read_lock(this);
  auto tableDescIt = tableDescriptorMapById_.find(tableId);
  if (tableDescIt == tableDescriptorMapById_.end()) {  // check to make sure table exists
    return nullptr;
  }
  TableDescriptor* td = tableDescIt->second;
  std::unique_lock<std::mutex> td_lock(*td->mutex_.get());
  if (populateFragmenter && td->fragmenter == nullptr && !td->isView) {
    instantiateFragmenter(td);
  }
  return td;  // returns pointer to table descriptor
}

const TableDescriptor* Catalog::getMetadataForTable(int tableId) const {
  return getMetadataForTableImpl(tableId, true);
}

const DictDescriptor* Catalog::getMetadataForDict(const int dictId,
                                                  const bool loadDict) const {
  const DictRef dictRef(currentDB_.dbId, dictId);
  cat_read_lock read_lock(this);
  auto dictDescIt = dictDescriptorMapByRef_.find(dictRef);
  if (dictDescIt ==
      dictDescriptorMapByRef_.end()) {  // check to make sure dictionary exists
    return nullptr;
  }
  auto& dd = dictDescIt->second;

  if (loadDict) {
    cat_sqlite_lock sqlite_lock(this);
    if (!dd->stringDict) {
      auto time_ms = measure<>::execution([&]() {
        if (string_dict_hosts_.empty()) {
          if (dd->dictIsTemp) {
            dd->stringDict =
                std::make_shared<StringDictionary>(dd->dictFolderPath, true, true);
          } else {
            dd->stringDict =
                std::make_shared<StringDictionary>(dd->dictFolderPath, false, true);
          }
        } else {
          dd->stringDict =
              std::make_shared<StringDictionary>(string_dict_hosts_.front(), dd->dictRef);
        }
      });
      LOG(INFO) << "Time to load Dictionary " << dd->dictRef.dbId << "_"
                << dd->dictRef.dictId << " was " << time_ms << "ms";
    }
  }

  return dd.get();
}

const std::vector<LeafHostInfo>& Catalog::getStringDictionaryHosts() const {
  return string_dict_hosts_;
}

const ColumnDescriptor* Catalog::getMetadataForColumn(int tableId,
                                                      const string& columnName) const {
  cat_read_lock read_lock(this);

  ColumnKey columnKey(tableId, to_upper(columnName));
  auto colDescIt = columnDescriptorMap_.find(columnKey);
  if (colDescIt ==
      columnDescriptorMap_.end()) {  // need to check to make sure column exists for table
    return nullptr;
  }
  return colDescIt->second;
}

const ColumnDescriptor* Catalog::getMetadataForColumn(int tableId, int columnId) const {
  cat_read_lock read_lock(this);

  ColumnIdKey columnIdKey(tableId, columnId);
  auto colDescIt = columnDescriptorMapById_.find(columnIdKey);
  if (colDescIt == columnDescriptorMapById_
                       .end()) {  // need to check to make sure column exists for table
    return nullptr;
  }
  return colDescIt->second;
}

const int Catalog::getColumnIdBySpiUnlocked(const int table_id, const size_t spi) const {
  const auto tabDescIt = tableDescriptorMapById_.find(table_id);
  CHECK(tableDescriptorMapById_.end() != tabDescIt);
  const auto& columnIdBySpi = tabDescIt->second->columnIdBySpi_;

  auto spx = spi;
  int phi = 0;
  if (spx >= SPIMAP_MAGIC1)  // see Catalog.h
  {
    phi = (spx - SPIMAP_MAGIC1) % SPIMAP_MAGIC2;
    spx = (spx - SPIMAP_MAGIC1) / SPIMAP_MAGIC2;
  }

  CHECK(0 < spx && spx <= columnIdBySpi.size());
  return columnIdBySpi[spx - 1] + phi;
}

const int Catalog::getColumnIdBySpi(const int table_id, const size_t spi) const {
  cat_read_lock read_lock(this);
  return getColumnIdBySpiUnlocked(table_id, spi);
}

const ColumnDescriptor* Catalog::getMetadataForColumnBySpi(const int tableId,
                                                           const size_t spi) const {
  cat_read_lock read_lock(this);

  const auto columnId = getColumnIdBySpiUnlocked(tableId, spi);
  ColumnIdKey columnIdKey(tableId, columnId);
  const auto colDescIt = columnDescriptorMapById_.find(columnIdKey);
  return columnDescriptorMapById_.end() == colDescIt ? nullptr : colDescIt->second;
}

void Catalog::deleteMetadataForFrontendView(const std::string& userId,
                                            const std::string& viewName) {
  cat_write_lock write_lock(this);

  auto viewDescIt = dashboardDescriptorMap_.find(userId + ":" + viewName);
  if (viewDescIt == dashboardDescriptorMap_.end()) {  // check to make sure view exists
    LOG(ERROR) << "No metadata for dashboard for user " << userId << " dashboard "
               << viewName << " does not exist in map";
    throw runtime_error("No metadata for dashboard for user " + userId + " dashboard " +
                        viewName + " does not exist in map");
  }
  // found view in Map now remove it
  dashboardDescriptorMap_.erase(viewDescIt);
  // remove from DB
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "DELETE FROM mapd_dashboards WHERE name = ? and userid = ?",
        std::vector<std::string>{viewName, userId});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

const FrontendViewDescriptor* Catalog::getMetadataForFrontendView(
    const string& userId,
    const string& viewName) const {
  cat_read_lock read_lock(this);

  auto viewDescIt = dashboardDescriptorMap_.find(userId + ":" + viewName);
  if (viewDescIt == dashboardDescriptorMap_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return viewDescIt->second.get();  // returns pointer to view descriptor
}

const FrontendViewDescriptor* Catalog::getMetadataForDashboard(const int32_t id) const {
  cat_read_lock read_lock(this);
  std::string userId;
  std::string name;
  bool found{false};
  {
    for (auto descp : dashboardDescriptorMap_) {
      auto dash = descp.second.get();
      if (dash->viewId == id) {
        userId = std::to_string(dash->userId);
        name = dash->viewName;
        found = true;
        break;
      }
    }
  }
  if (found) {
    return getMetadataForFrontendView(userId, name);
  }
  return nullptr;
}

void Catalog::deleteMetadataForDashboard(const int32_t id) {
  std::string userId;
  std::string name;
  bool found{false};
  {
    cat_read_lock read_lock(this);
    for (auto descp : dashboardDescriptorMap_) {
      auto dash = descp.second.get();
      if (dash->viewId == id) {
        userId = std::to_string(dash->userId);
        name = dash->viewName;
        found = true;
        break;
      }
    }
  }
  if (found) {
    // TODO: transactionally unsafe
    SysCatalog::instance().revokeDBObjectPrivilegesFromAll(
        DBObject(id, DashboardDBObjectType), this);
    deleteMetadataForFrontendView(userId, name);
  }
}

const LinkDescriptor* Catalog::getMetadataForLink(const string& link) const {
  cat_read_lock read_lock(this);
  auto linkDescIt = linkDescriptorMap_.find(link);
  if (linkDescIt == linkDescriptorMap_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return linkDescIt->second;  // returns pointer to view descriptor
}

const LinkDescriptor* Catalog::getMetadataForLink(int linkId) const {
  cat_read_lock read_lock(this);
  auto linkDescIt = linkDescriptorMapById_.find(linkId);
  if (linkDescIt == linkDescriptorMapById_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return linkDescIt->second;
}

void Catalog::getAllColumnMetadataForTable(
    const TableDescriptor* td,
    list<const ColumnDescriptor*>& columnDescriptors,
    const bool fetchSystemColumns,
    const bool fetchVirtualColumns,
    const bool fetchPhysicalColumns) const {
  cat_read_lock read_lock(this);
  int32_t skip_physical_cols = 0;
  for (const auto& columnDescriptor : columnDescriptorMapById_) {
    if (!fetchPhysicalColumns && skip_physical_cols > 0) {
      --skip_physical_cols;
      continue;
    }
    auto cd = columnDescriptor.second;
    if (cd->tableId != td->tableId) {
      continue;
    }
    if (!fetchSystemColumns && cd->isSystemCol) {
      continue;
    }
    if (!fetchVirtualColumns && cd->isVirtualCol) {
      continue;
    }
    if (!fetchPhysicalColumns) {
      const auto& col_ti = cd->columnType;
      skip_physical_cols = col_ti.get_physical_cols();
    }
    columnDescriptors.push_back(cd);
  }
}

list<const ColumnDescriptor*> Catalog::getAllColumnMetadataForTable(
    const int tableId,
    const bool fetchSystemColumns,
    const bool fetchVirtualColumns,
    const bool fetchPhysicalColumns) const {
  cat_read_lock read_lock(this);
  list<const ColumnDescriptor*> columnDescriptors;
  const TableDescriptor* td =
      getMetadataForTableImpl(tableId, false);  // dont instantiate fragmenter
  getAllColumnMetadataForTable(td,
                               columnDescriptors,
                               fetchSystemColumns,
                               fetchVirtualColumns,
                               fetchPhysicalColumns);
  return columnDescriptors;
}

list<const TableDescriptor*> Catalog::getAllTableMetadata() const {
  cat_read_lock read_lock(this);
  list<const TableDescriptor*> table_list;
  for (auto p : tableDescriptorMapById_) {
    table_list.push_back(p.second);
  }
  return table_list;
}

list<const FrontendViewDescriptor*> Catalog::getAllFrontendViewMetadata() const {
  list<const FrontendViewDescriptor*> view_list;
  for (auto p : dashboardDescriptorMap_) {
    view_list.push_back(p.second.get());
  }
  return view_list;
}

DictRef Catalog::addDictionary(ColumnDescriptor& cd) {
  const auto& td = *tableDescriptorMapById_[cd.tableId];
  list<DictDescriptor> dds;
  setColumnDictionary(cd, dds, td, true);
  auto& dd = dds.back();
  CHECK(dd.dictRef.dictId);

  std::unique_ptr<StringDictionaryClient> client;
  if (!string_dict_hosts_.empty()) {
    client.reset(new StringDictionaryClient(
        string_dict_hosts_.front(), DictRef(currentDB_.dbId, -1), true));
  }
  if (client) {
    client->create(dd.dictRef, dd.dictIsTemp);
  }

  DictDescriptor* new_dd = new DictDescriptor(dd);
  dictDescriptorMapByRef_[dd.dictRef].reset(new_dd);
  if (!dd.dictIsTemp) {
    boost::filesystem::create_directory(new_dd->dictFolderPath);
  }
  return dd.dictRef;
}

// TODO this all looks incorrect there is no reference count being checked here
// this code needs to be fixed
void Catalog::delDictionary(const ColumnDescriptor& cd) {
  if (!(cd.columnType.is_string() || cd.columnType.is_string_array())) {
    return;
  }
  if (!(cd.columnType.get_compression() == kENCODING_DICT)) {
    return;
  }
  if (!(cd.columnType.get_comp_param() > 0)) {
    return;
  }

  const auto& td = *tableDescriptorMapById_[cd.tableId];
  const auto dictId = cd.columnType.get_comp_param();
  const DictRef dictRef(currentDB_.dbId, dictId);
  const auto dictName =
      td.tableName + "_" + cd.columnName + "_dict" + std::to_string(dictId);
  sqliteConnector_.query_with_text_param("DELETE FROM mapd_dictionaries WHERE name = ?",
                                         dictName);
  File_Namespace::renameForDelete(basePath_ + "/mapd_data/DB_" +
                                  std::to_string(currentDB_.dbId) + "_DICT_" +
                                  std::to_string(dictId));

  std::unique_ptr<StringDictionaryClient> client;
  if (!string_dict_hosts_.empty()) {
    client.reset(new StringDictionaryClient(string_dict_hosts_.front(), dictRef, true));
  }
  if (client) {
    client->drop(dictRef);
  }

  dictDescriptorMapByRef_.erase(dictRef);
}

void Catalog::getDictionary(const ColumnDescriptor& cd,
                            std::map<int, StringDictionary*>& stringDicts) {
  // learn 'committed' ColumnDescriptor of this column
  auto cit = columnDescriptorMap_.find(ColumnKey(cd.tableId, to_upper(cd.columnName)));
  CHECK(cit != columnDescriptorMap_.end());
  auto& ccd = *cit->second;

  if (!(ccd.columnType.is_string() || ccd.columnType.is_string_array())) {
    return;
  }
  if (!(ccd.columnType.get_compression() == kENCODING_DICT)) {
    return;
  }
  if (!(ccd.columnType.get_comp_param() > 0)) {
    return;
  }

  auto dictId = ccd.columnType.get_comp_param();
  getMetadataForDict(dictId);

  const DictRef dictRef(currentDB_.dbId, dictId);
  auto dit = dictDescriptorMapByRef_.find(dictRef);
  CHECK(dit != dictDescriptorMapByRef_.end());
  CHECK(dit->second);
  CHECK(dit->second.get()->stringDict);
  stringDicts[ccd.columnId] = dit->second.get()->stringDict.get();
}

void Catalog::addColumn(const TableDescriptor& td, ColumnDescriptor& cd) {
  // caller must handle sqlite/chunk transaction TOGETHER
  cd.tableId = td.tableId;
  if (cd.columnType.get_compression() == kENCODING_DICT) {
    addDictionary(cd);
  }

  sqliteConnector_.query_with_text_params(
      "INSERT INTO mapd_columns (tableid, columnid, name, coltype, colsubtype, coldim, "
      "colscale, is_notnull, "
      "compression, comp_param, size, chunks, is_systemcol, is_virtualcol, virtual_expr, "
      "is_deletedcol) "
      "VALUES (?, "
      "(SELECT max(columnid) + 1 FROM mapd_columns WHERE tableid = ?), "
      "?, ?, ?, "
      "?, "
      "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      std::vector<std::string>{std::to_string(td.tableId),
                               std::to_string(td.tableId),
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
                               cd.virtualExpr,
                               std::to_string(cd.isDeletedCol)});

  sqliteConnector_.query_with_text_params(
      "UPDATE mapd_tables SET ncolumns = ncolumns + 1 WHERE tableid = ?",
      std::vector<std::string>{std::to_string(td.tableId)});

  sqliteConnector_.query_with_text_params(
      "SELECT columnid FROM mapd_columns WHERE tableid = ? AND name = ?",
      std::vector<std::string>{std::to_string(td.tableId), cd.columnName});
  cd.columnId = sqliteConnector_.getData<int>(0, 0);

  ++tableDescriptorMapById_[td.tableId]->nColumns;
  auto ncd = new ColumnDescriptor(cd);
  columnDescriptorMap_[ColumnKey(cd.tableId, to_upper(cd.columnName))] = ncd;
  columnDescriptorMapById_[ColumnIdKey(cd.tableId, cd.columnId)] = ncd;
  columnDescriptorsForRoll.emplace_back(nullptr, ncd);
}

void Catalog::roll(const bool forward) {
  std::set<const TableDescriptor*> tds;

  for (const auto& cdr : columnDescriptorsForRoll) {
    auto ocd = cdr.first;
    auto ncd = cdr.second;
    CHECK(ocd || ncd);
    auto tabDescIt = tableDescriptorMapById_.find((ncd ? ncd : ocd)->tableId);
    CHECK(tableDescriptorMapById_.end() != tabDescIt);
    auto td = tabDescIt->second;
    auto& vc = td->columnIdBySpi_;
    if (forward) {
      if (ocd) {
        if (nullptr == ncd ||
            ncd->columnType.get_comp_param() != ocd->columnType.get_comp_param()) {
          delDictionary(*ocd);
        }

        vc.erase(std::remove(vc.begin(), vc.end(), ocd->columnId), vc.end());

        delete ocd;
      }
      if (ncd) {
        // append columnId if its new and not phy geo
        if (vc.end() == std::find(vc.begin(), vc.end(), ncd->columnId)) {
          if (!ncd->isGeoPhyCol) {
            vc.push_back(ncd->columnId);
          }
        }
      }
      tds.insert(td);
    } else {
      if (ocd) {
        columnDescriptorMap_[ColumnKey(ocd->tableId, to_upper(ocd->columnName))] = ocd;
        columnDescriptorMapById_[ColumnIdKey(ocd->tableId, ocd->columnId)] = ocd;
      }
      // roll back the dict of new column
      if (ncd) {
        columnDescriptorMap_.erase(ColumnKey(ncd->tableId, to_upper(ncd->columnName)));
        columnDescriptorMapById_.erase(ColumnIdKey(ncd->tableId, ncd->columnId));
        if (nullptr == ocd ||
            ocd->columnType.get_comp_param() != ncd->columnType.get_comp_param()) {
          delDictionary(*ncd);
        }
        delete ncd;
      }
    }
  }
  columnDescriptorsForRoll.clear();

  if (forward) {
    for (const auto td : tds) {
      calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
    }
  }
}

void Catalog::expandGeoColumn(const ColumnDescriptor& cd,
                              list<ColumnDescriptor>& columns) {
  const auto& col_ti = cd.columnType;
  if (IS_GEO(col_ti.get_type())) {
    switch (col_ti.get_type()) {
      case kPOINT: {
        ColumnDescriptor physical_cd_coords(true);
        physical_cd_coords.columnName = cd.columnName + "_coords";
        SQLTypeInfo coords_ti = SQLTypeInfo(kARRAY, true);
        // Raw data: compressed/uncompressed coords
        coords_ti.set_subtype(kTINYINT);
        size_t unit_size;
        if (col_ti.get_compression() == kENCODING_GEOINT &&
            col_ti.get_comp_param() == 32) {
          unit_size = 4 * sizeof(int8_t);
        } else {
          CHECK(col_ti.get_compression() == kENCODING_NONE);
          unit_size = 8 * sizeof(int8_t);
        }
        coords_ti.set_size(2 * unit_size);
        physical_cd_coords.columnType = coords_ti;
        columns.push_back(physical_cd_coords);

        // If adding more physical columns - update SQLTypeInfo::get_physical_cols()

        break;
      }
      case kLINESTRING: {
        ColumnDescriptor physical_cd_coords(true);
        physical_cd_coords.columnName = cd.columnName + "_coords";
        SQLTypeInfo coords_ti = SQLTypeInfo(kARRAY, true);
        // Raw data: compressed/uncompressed coords
        coords_ti.set_subtype(kTINYINT);
        physical_cd_coords.columnType = coords_ti;
        columns.push_back(physical_cd_coords);

        ColumnDescriptor physical_cd_bounds(true);
        physical_cd_bounds.columnName = cd.columnName + "_bounds";
        SQLTypeInfo bounds_ti = SQLTypeInfo(kARRAY, true);
        bounds_ti.set_subtype(kDOUBLE);
        bounds_ti.set_size(4 * sizeof(double));
        physical_cd_bounds.columnType = bounds_ti;
        columns.push_back(physical_cd_bounds);

        // If adding more physical columns - update SQLTypeInfo::get_physical_cols()

        break;
      }
      case kPOLYGON: {
        ColumnDescriptor physical_cd_coords(true);
        physical_cd_coords.columnName = cd.columnName + "_coords";
        SQLTypeInfo coords_ti = SQLTypeInfo(kARRAY, true);
        // Raw data: compressed/uncompressed coords
        coords_ti.set_subtype(kTINYINT);
        physical_cd_coords.columnType = coords_ti;
        columns.push_back(physical_cd_coords);

        ColumnDescriptor physical_cd_ring_sizes(true);
        physical_cd_ring_sizes.columnName = cd.columnName + "_ring_sizes";
        SQLTypeInfo ring_sizes_ti = SQLTypeInfo(kARRAY, true);
        ring_sizes_ti.set_subtype(kINT);
        physical_cd_ring_sizes.columnType = ring_sizes_ti;
        columns.push_back(physical_cd_ring_sizes);

        ColumnDescriptor physical_cd_bounds(true);
        physical_cd_bounds.columnName = cd.columnName + "_bounds";
        SQLTypeInfo bounds_ti = SQLTypeInfo(kARRAY, true);
        bounds_ti.set_subtype(kDOUBLE);
        bounds_ti.set_size(4 * sizeof(double));
        physical_cd_bounds.columnType = bounds_ti;
        columns.push_back(physical_cd_bounds);

        ColumnDescriptor physical_cd_render_group(true);
        physical_cd_render_group.columnName = cd.columnName + "_render_group";
        SQLTypeInfo render_group_ti = SQLTypeInfo(kINT, true);
        physical_cd_render_group.columnType = render_group_ti;
        columns.push_back(physical_cd_render_group);

        // If adding more physical columns - update SQLTypeInfo::get_physical_cols()

        break;
      }
      case kMULTIPOLYGON: {
        ColumnDescriptor physical_cd_coords(true);
        physical_cd_coords.columnName = cd.columnName + "_coords";
        SQLTypeInfo coords_ti = SQLTypeInfo(kARRAY, true);
        // Raw data: compressed/uncompressed coords
        coords_ti.set_subtype(kTINYINT);
        physical_cd_coords.columnType = coords_ti;
        columns.push_back(physical_cd_coords);

        ColumnDescriptor physical_cd_ring_sizes(true);
        physical_cd_ring_sizes.columnName = cd.columnName + "_ring_sizes";
        SQLTypeInfo ring_sizes_ti = SQLTypeInfo(kARRAY, true);
        ring_sizes_ti.set_subtype(kINT);
        physical_cd_ring_sizes.columnType = ring_sizes_ti;
        columns.push_back(physical_cd_ring_sizes);

        ColumnDescriptor physical_cd_poly_rings(true);
        physical_cd_poly_rings.columnName = cd.columnName + "_poly_rings";
        SQLTypeInfo poly_rings_ti = SQLTypeInfo(kARRAY, true);
        poly_rings_ti.set_subtype(kINT);
        physical_cd_poly_rings.columnType = poly_rings_ti;
        columns.push_back(physical_cd_poly_rings);

        ColumnDescriptor physical_cd_bounds(true);
        physical_cd_bounds.columnName = cd.columnName + "_bounds";
        SQLTypeInfo bounds_ti = SQLTypeInfo(kARRAY, true);
        bounds_ti.set_subtype(kDOUBLE);
        bounds_ti.set_size(4 * sizeof(double));
        physical_cd_bounds.columnType = bounds_ti;
        columns.push_back(physical_cd_bounds);

        ColumnDescriptor physical_cd_render_group(true);
        physical_cd_render_group.columnName = cd.columnName + "_render_group";
        SQLTypeInfo render_group_ti = SQLTypeInfo(kINT, true);
        physical_cd_render_group.columnType = render_group_ti;
        columns.push_back(physical_cd_render_group);

        // If adding more physical columns - update SQLTypeInfo::get_physical_cols()

        break;
      }
      default:
        throw runtime_error("Unrecognized geometry type.");
        break;
    }
  }
}

void Catalog::createTable(
    TableDescriptor& td,
    const list<ColumnDescriptor>& cols,
    const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs,
    bool isLogicalTable) {
  cat_write_lock write_lock(this);
  list<ColumnDescriptor> cds;
  list<DictDescriptor> dds;
  std::set<std::string> toplevel_column_names;
  list<ColumnDescriptor> columns;
  for (auto cd : cols) {
    if (cd.columnName == "rowid") {
      throw std::runtime_error(
          "Cannot create column with name rowid. rowid is a system defined column.");
    }
    columns.push_back(cd);
    toplevel_column_names.insert(cd.columnName);
    if (cd.columnType.is_geometry()) {
      expandGeoColumn(cd, columns);
    }
  }

  ColumnDescriptor cd;
  // add row_id column -- Must be last column in the table
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
  toplevel_column_names.insert(cd.columnName);

  if (td.hasDeletedCol) {
    ColumnDescriptor cd_del;
    cd_del.columnName = "$deleted$";
    cd_del.isSystemCol = true;
    cd_del.isVirtualCol = false;
    cd_del.columnType = SQLTypeInfo(kBOOLEAN, true);
    cd_del.isDeletedCol = true;

    columns.push_back(cd_del);
  }

  td.nColumns = columns.size();
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  if (td.persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    try {
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_tables (name, userid, ncolumns, isview, fragments, "
          "frag_type, max_frag_rows, "
          "max_chunk_size, "
          "frag_page_size, max_rows, partitions, shard_column_id, shard, num_shards, "
          "key_metainfo) VALUES (?, ?, ?, "
          "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",

          std::vector<std::string>{td.tableName,
                                   std::to_string(td.userId),
                                   std::to_string(td.nColumns),
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
      sqliteConnector_.query_with_text_param(
          "SELECT tableid FROM mapd_tables WHERE name = ?", td.tableName);
      td.tableId = sqliteConnector_.getData<int>(0, 0);
      int colId = 1;
      for (auto cd : columns) {
        if (cd.columnType.get_compression() == kENCODING_DICT) {
          const bool is_foreign_col =
              setColumnSharedDictionary(cd, cds, dds, td, shared_dict_defs);
          if (!is_foreign_col) {
            setColumnDictionary(cd, dds, td, isLogicalTable);
          }
        }

        if (toplevel_column_names.count(cd.columnName)) {
          // make up colId gap for sanity test (begin with 1 bc much code depends on it!)
          if (colId > 1) {
            colId += g_test_against_columnId_gap;
          }
          if (!cd.isGeoPhyCol) {
            td.columnIdBySpi_.push_back(colId);
          }
        }

        sqliteConnector_.query_with_text_params(
            "INSERT INTO mapd_columns (tableid, columnid, name, coltype, colsubtype, "
            "coldim, colscale, is_notnull, "
            "compression, comp_param, size, chunks, is_systemcol, is_virtualcol, "
            "virtual_expr, is_deletedcol) "
            "VALUES (?, ?, ?, ?, ?, "
            "?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                                     cd.virtualExpr,
                                     std::to_string(cd.isDeletedCol)});
        cd.tableId = td.tableId;
        cd.columnId = colId++;
        cds.push_back(cd);
      }
      if (td.isView) {
        sqliteConnector_.query_with_text_params(
            "INSERT INTO mapd_views (tableid, sql) VALUES (?,?)",
            std::vector<std::string>{std::to_string(td.tableId), td.viewSQL});
      }
    } catch (std::exception& e) {
      sqliteConnector_.query("ROLLBACK TRANSACTION");
      throw e;
    }

  } else {  // Temporary table
    td.tableId = nextTempTableId_++;
    int colId = 1;
    for (auto cd : columns) {
      auto col_ti = cd.columnType;
      if (IS_GEO(col_ti.get_type())) {
        throw runtime_error("Geometry types in temporary tables are not supported.");
      }

      if (cd.columnType.get_compression() == kENCODING_DICT) {
        // TODO(vraj) : create shared dictionary for temp table if needed
        std::string fileName("");
        std::string folderPath("");
        DictRef dict_ref(currentDB_.dbId, nextTempDictId_);
        nextTempDictId_++;
        DictDescriptor dd(dict_ref,
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
        cd.columnType.set_comp_param(dict_ref.dictId);
      }
      cd.tableId = td.tableId;
      cd.columnId = colId++;
      cds.push_back(cd);
    }
  }
  try {
    addTableToMap(td, cds, dds);
    calciteMgr_->updateMetadata(currentDB_.dbName, td.tableName);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    removeTableFromMap(td.tableName, td.tableId);
    throw e;
  }

  sqliteConnector_.query("END TRANSACTION");
}

// returns the table epoch or -1 if there is something wrong with the shared epoch
int32_t Catalog::getTableEpoch(const int32_t db_id, const int32_t table_id) const {
  cat_read_lock read_lock(this);
  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(table_id);
  if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
    // check all shards have same checkpoint
    const auto physicalTables = physicalTableIt->second;
    CHECK(!physicalTables.empty());
    size_t curr_epoch = 0;
    for (size_t i = 0; i < physicalTables.size(); i++) {
      int32_t physical_tb_id = physicalTables[i];
      const TableDescriptor* phys_td = getMetadataForTable(physical_tb_id);
      CHECK(phys_td);
      if (i == 0) {
        curr_epoch = dataMgr_->getTableEpoch(db_id, physical_tb_id);
      } else {
        if (curr_epoch != dataMgr_->getTableEpoch(db_id, physical_tb_id)) {
          // oh dear the leaves do not agree on the epoch for this table
          LOG(ERROR) << "Epochs on shards do not all agree on table id " << table_id
                     << " db id  " << db_id << " epoch " << curr_epoch << " leaf_epoch "
                     << dataMgr_->getTableEpoch(db_id, physical_tb_id);
          return -1;
        }
      }
    }
    return curr_epoch;
  } else {
    return dataMgr_->getTableEpoch(db_id, table_id);
  }
}

void Catalog::setTableEpoch(const int db_id, const int table_id, int new_epoch) {
  cat_read_lock read_lock(this);
  LOG(INFO) << "Set table epoch db:" << db_id << " Table ID  " << table_id
            << " back to new epoch " << new_epoch;
  removeChunks(table_id);
  dataMgr_->setTableEpoch(db_id, table_id, new_epoch);

  // check if sharded
  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(table_id);
  if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
    const auto physicalTables = physicalTableIt->second;
    CHECK(!physicalTables.empty());
    for (size_t i = 0; i < physicalTables.size(); i++) {
      int32_t physical_tb_id = physicalTables[i];
      const TableDescriptor* phys_td = getMetadataForTable(physical_tb_id);
      CHECK(phys_td);
      LOG(INFO) << "Set sharded table epoch db:" << db_id << " Table ID  "
                << physical_tb_id << " back to new epoch " << new_epoch;
      removeChunks(physical_tb_id);
      dataMgr_->setTableEpoch(db_id, physical_tb_id, new_epoch);
    }
  }
}

const ColumnDescriptor* Catalog::getDeletedColumn(const TableDescriptor* td) const {
  cat_read_lock read_lock(this);
  const auto it = deletedColumnPerTable_.find(td);
  return it != deletedColumnPerTable_.end() ? it->second : nullptr;
}

const bool Catalog::checkMetadataForDeletedRecs(int dbId,
                                                int tableId,
                                                int columnId) const {
  // check if there are rows deleted by examining metadata for the deletedColumn metadata
  ChunkKey chunkKeyPrefix = {dbId, tableId, columnId};
  std::vector<std::pair<ChunkKey, ChunkMetadata>> chunkMetadataVec;
  dataMgr_->getChunkMetadataVecForKeyPrefix(chunkMetadataVec, chunkKeyPrefix);
  int64_t chunk_max{0};

  for (auto cm : chunkMetadataVec) {
    chunk_max = cm.second.chunkStats.max.tinyintval;
    // delete has occured
    if (chunk_max == 1) {
      return true;
    }
  }
  return false;
}

const ColumnDescriptor* Catalog::getDeletedColumnIfRowsDeleted(
    const TableDescriptor* td) const {
  cat_read_lock read_lock(this);

  const auto it = deletedColumnPerTable_.find(td);
  // if not a table that supports delete return nullptr,  nothing more to do
  if (it == deletedColumnPerTable_.end()) {
    return nullptr;
  }
  const ColumnDescriptor* cd = it->second;

  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(td->tableId);

  if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
    // check all shards
    const auto physicalTables = physicalTableIt->second;
    CHECK(!physicalTables.empty());
    for (size_t i = 0; i < physicalTables.size(); i++) {
      int32_t physical_tb_id = physicalTables[i];
      const TableDescriptor* phys_td = getMetadataForTable(physical_tb_id);
      CHECK(phys_td);
      if (checkMetadataForDeletedRecs(currentDB_.dbId, phys_td->tableId, cd->columnId)) {
        return cd;
      }
    }
  } else {
    if (checkMetadataForDeletedRecs(currentDB_.dbId, td->tableId, cd->columnId)) {
      return cd;
    }
  }
  // no deletes so far recorded in metadata
  return nullptr;
}

void Catalog::setDeletedColumn(const TableDescriptor* td, const ColumnDescriptor* cd) {
  cat_write_lock write_lock(this);
  setDeletedColumnUnlocked(td, cd);
}

void Catalog::setDeletedColumnUnlocked(const TableDescriptor* td,
                                       const ColumnDescriptor* cd) {
  cat_write_lock write_lock(this);
  const auto it_ok = deletedColumnPerTable_.emplace(td, cd);
  CHECK(it_ok.second);
}

namespace {

const ColumnDescriptor* get_foreign_col(
    const Catalog& cat,
    const Parser::SharedDictionaryDef& shared_dict_def) {
  const auto& table_name = shared_dict_def.get_foreign_table();
  const auto td = cat.getMetadataForTable(table_name);
  CHECK(td);
  const auto& foreign_col_name = shared_dict_def.get_foreign_column();
  return cat.getMetadataForColumn(td->tableId, foreign_col_name);
}

}  // namespace

void Catalog::addReferenceToForeignDict(ColumnDescriptor& referencing_column,
                                        Parser::SharedDictionaryDef shared_dict_def) {
  cat_write_lock write_lock(this);
  const auto foreign_ref_col = get_foreign_col(*this, shared_dict_def);
  CHECK(foreign_ref_col);
  referencing_column.columnType = foreign_ref_col->columnType;
  const int dict_id = referencing_column.columnType.get_comp_param();
  const DictRef dict_ref(currentDB_.dbId, dict_id);
  const auto dictIt = dictDescriptorMapByRef_.find(dict_ref);
  CHECK(dictIt != dictDescriptorMapByRef_.end());
  const auto& dd = dictIt->second;
  CHECK_GE(dd->refcount, 1);
  ++dd->refcount;
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query_with_text_params(
      "UPDATE mapd_dictionaries SET refcount = refcount + 1 WHERE dictid = ?",
      {std::to_string(dict_id)});
}

bool Catalog::setColumnSharedDictionary(
    ColumnDescriptor& cd,
    std::list<ColumnDescriptor>& cdd,
    std::list<DictDescriptor>& dds,
    const TableDescriptor td,
    const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);

  if (shared_dict_defs.empty()) {
    return false;
  }
  for (const auto& shared_dict_def : shared_dict_defs) {
    // check if the current column is a referencing column
    const auto& column = shared_dict_def.get_column();
    if (cd.columnName == column) {
      if (!shared_dict_def.get_foreign_table().compare(td.tableName)) {
        // Dictionaries are being shared in table to be created
        const auto& ref_column = shared_dict_def.get_foreign_column();
        auto colIt =
            std::find_if(cdd.begin(), cdd.end(), [ref_column](const ColumnDescriptor it) {
              return !ref_column.compare(it.columnName);
            });
        CHECK(colIt != cdd.end());
        cd.columnType = colIt->columnType;

        sqliteConnector_.query_with_text_params(
            "SELECT dictid FROM mapd_dictionaries WHERE dictid in (select comp_param "
            "from "
            "mapd_columns "
            "where compression = ? and tableid = ? and columnid = ?)",
            std::vector<std::string>{std::to_string(kENCODING_DICT),
                                     std::to_string(td.tableId),
                                     std::to_string(colIt->columnId)});
        const auto dict_id = sqliteConnector_.getData<int>(0, 0);
        auto dictIt = std::find_if(
            dds.begin(), dds.end(), [this, dict_id](const DictDescriptor it) {
              return it.dictRef.dbId == this->currentDB_.dbId &&
                     it.dictRef.dictId == dict_id;
            });
        if (dictIt != dds.end()) {
          // There exists dictionary definition of a dictionary column
          CHECK_GE(dictIt->refcount, 1);
          ++dictIt->refcount;
          sqliteConnector_.query_with_text_params(
              "UPDATE mapd_dictionaries SET refcount = refcount + 1 WHERE dictid = ?",
              {std::to_string(dict_id)});
        } else {
          // The dictionary is referencing a column which is referencing a column in
          // diffrent table
          auto root_dict_def = compress_reference_path(shared_dict_def, shared_dict_defs);
          addReferenceToForeignDict(cd, root_dict_def);
        }
      } else {
        addReferenceToForeignDict(cd, shared_dict_def);
      }
      return true;
    }
  }
  return false;
}

void Catalog::setColumnDictionary(ColumnDescriptor& cd,
                                  std::list<DictDescriptor>& dds,
                                  const TableDescriptor& td,
                                  const bool isLogicalTable) {
  cat_write_lock write_lock(this);

  std::string dictName{"Initial_key"};
  int dictId{0};
  std::string folderPath;
  if (isLogicalTable) {
    cat_sqlite_lock sqlite_lock(this);

    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_dictionaries (name, nbits, is_shared, refcount) VALUES (?, ?, "
        "?, 1)",
        std::vector<std::string>{
            dictName, std::to_string(cd.columnType.get_comp_param()), "0"});
    sqliteConnector_.query_with_text_param(
        "SELECT dictid FROM mapd_dictionaries WHERE name = ?", dictName);
    dictId = sqliteConnector_.getData<int>(0, 0);
    dictName = td.tableName + "_" + cd.columnName + "_dict" + std::to_string(dictId);
    sqliteConnector_.query_with_text_param(
        "UPDATE mapd_dictionaries SET name = ? WHERE name = 'Initial_key'", dictName);
    folderPath = basePath_ + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) +
                 "_DICT_" + std::to_string(dictId);
  }
  DictDescriptor dd(currentDB_.dbId,
                    dictId,
                    dictName,
                    cd.columnType.get_comp_param(),
                    false,
                    1,
                    folderPath,
                    false);
  dds.push_back(dd);
  if (!cd.columnType.is_array()) {
    cd.columnType.set_size(cd.columnType.get_comp_param() / 8);
  }
  cd.columnType.set_comp_param(dictId);
}

void Catalog::createShardedTable(
    TableDescriptor& td,
    const list<ColumnDescriptor>& cols,
    const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs) {
  cat_write_lock write_lock(this);

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
    const auto it_ok =
        logicalToPhysicalTableMapById_.emplace(logical_tb_id, physicalTables);
    CHECK(it_ok.second);
    /* update sqlite mapd_logical_to_physical in sqlite database */
    updateLogicalToPhysicalTableMap(logical_tb_id);
  }
}

void Catalog::truncateTable(const TableDescriptor* td) {
  cat_write_lock write_lock(this);

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
  cat_write_lock write_lock(this);

  const int tableId = td->tableId;
  // must destroy fragmenter before deleteChunks is called.
  if (td->fragmenter != nullptr) {
    auto tableDescIt = tableDescriptorMapById_.find(tableId);
    delete td->fragmenter;
    tableDescIt->second->fragmenter = nullptr;  // get around const-ness
  }
  ChunkKey chunkKeyPrefix = {currentDB_.dbId, tableId};
  // assuming deleteChunksWithPrefix is atomic
  dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix, MemoryLevel::CPU_LEVEL);
  dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix, MemoryLevel::GPU_LEVEL);

  dataMgr_->removeTableRelatedDS(currentDB_.dbId, tableId);

  std::unique_ptr<StringDictionaryClient> client;
  if (SysCatalog::instance().isAggregator()) {
    CHECK(!string_dict_hosts_.empty());
    DictRef dict_ref(currentDB_.dbId, -1);
    client.reset(new StringDictionaryClient(string_dict_hosts_.front(), dict_ref, true));
  }
  // clean up any dictionaries
  // delete all column descriptors for the table
  for (const auto& columnDescriptor : columnDescriptorMapById_) {
    auto cd = columnDescriptor.second;
    if (cd->tableId != td->tableId) {
      continue;
    }
    const int dict_id = cd->columnType.get_comp_param();
    // Dummy dictionaries created for a shard of a logical table have the id set to zero.
    if (cd->columnType.get_compression() == kENCODING_DICT && dict_id) {
      const DictRef dict_ref(currentDB_.dbId, dict_id);
      const auto dictIt = dictDescriptorMapByRef_.find(dict_ref);
      CHECK(dictIt != dictDescriptorMapByRef_.end());
      const auto& dd = dictIt->second;
      CHECK_GE(dd->refcount, 1);
      // if this is the only table using this dict reset the dict
      if (dd->refcount == 1) {
        // close the dictionary
        dd->stringDict.reset();
        File_Namespace::renameForDelete(dd->dictFolderPath);
        if (client) {
          client->drop(dd->dictRef);
        }
        if (!dd->dictIsTemp) {
          boost::filesystem::create_directory(dd->dictFolderPath);
        }
      }

      DictDescriptor* new_dd = new DictDescriptor(dd->dictRef,
                                                  dd->dictName,
                                                  dd->dictNBits,
                                                  dd->dictIsShared,
                                                  dd->refcount,
                                                  dd->dictFolderPath,
                                                  dd->dictIsTemp);
      dictDescriptorMapByRef_.erase(dictIt);
      // now create new Dict -- need to figure out what to do here for temp tables
      if (client) {
        client->create(new_dd->dictRef, new_dd->dictIsTemp);
      }
      dictDescriptorMapByRef_[new_dd->dictRef].reset(new_dd);
      getMetadataForDict(new_dd->dictRef.dictId);
    }
  }
}

// used by rollback_table_epoch to clean up in memory artifacts after a rollback
void Catalog::removeChunks(const int table_id) {
  auto td = getMetadataForTable(table_id);

  if (td->fragmenter != nullptr) {
    cat_sqlite_lock sqlite_lock(this);
    if (td->fragmenter != nullptr) {
      auto tableDescIt = tableDescriptorMapById_.find(table_id);
      delete td->fragmenter;
      tableDescIt->second->fragmenter = nullptr;  // get around const-ness
    }
  }

  // remove the chunks from in memory structures
  ChunkKey chunkKey = {currentDB_.dbId, table_id};

  dataMgr_->deleteChunksWithPrefix(chunkKey, MemoryLevel::CPU_LEVEL);
  dataMgr_->deleteChunksWithPrefix(chunkKey, MemoryLevel::GPU_LEVEL);
}

void Catalog::dropTable(const TableDescriptor* td) {
  SysCatalog::instance().revokeDBObjectPrivilegesFromAll(
      DBObject(td->tableName, td->isView ? ViewDBObjectType : TableDBObjectType), this);
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);
  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(td->tableId);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
      // remove all corresponding physical tables if this is a logical table
      const auto physicalTables = physicalTableIt->second;
      CHECK(!physicalTables.empty());
      for (size_t i = 0; i < physicalTables.size(); i++) {
        int32_t physical_tb_id = physicalTables[i];
        const TableDescriptor* phys_td = getMetadataForTable(physical_tb_id);
        CHECK(phys_td);
        doDropTable(phys_td);
      }

      // remove corresponding record from the logicalToPhysicalTableMap in sqlite database
      sqliteConnector_.query_with_text_param(
          "DELETE FROM mapd_logical_to_physical WHERE logical_table_id = ?",
          std::to_string(td->tableId));
      logicalToPhysicalTableMapById_.erase(td->tableId);
    }
    doDropTable(td);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::doDropTable(const TableDescriptor* td) {
  const int tableId = td->tableId;
  sqliteConnector_.query_with_text_param("DELETE FROM mapd_tables WHERE tableid = ?",
                                         std::to_string(tableId));
  sqliteConnector_.query_with_text_params(
      "select comp_param from mapd_columns where compression = ? and tableid = ?",
      std::vector<std::string>{std::to_string(kENCODING_DICT), std::to_string(tableId)});
  int numRows = sqliteConnector_.getNumRows();
  std::vector<int> dict_id_list;
  for (int r = 0; r < numRows; ++r) {
    dict_id_list.push_back(sqliteConnector_.getData<int>(r, 0));
  }
  for (auto dict_id : dict_id_list) {
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_dictionaries SET refcount = refcount - 1 WHERE dictid = ?",
        std::vector<std::string>{std::to_string(dict_id)});
  }
  sqliteConnector_.query_with_text_params(
      "DELETE FROM mapd_dictionaries WHERE dictid in (select comp_param from "
      "mapd_columns where compression = ? "
      "and tableid = ?) and refcount = 0",
      std::vector<std::string>{std::to_string(kENCODING_DICT), std::to_string(tableId)});
  sqliteConnector_.query_with_text_param("DELETE FROM mapd_columns WHERE tableid = ?",
                                         std::to_string(tableId));
  if (td->isView) {
    sqliteConnector_.query_with_text_param("DELETE FROM mapd_views WHERE tableid = ?",
                                           std::to_string(tableId));
  }
  eraseTablePhysicalData(td);
}

void Catalog::renamePhysicalTable(const TableDescriptor* td, const string& newTableName) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);

  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_tables SET name = ? WHERE tableid = ?",
        std::vector<std::string>{newTableName, std::to_string(td->tableId)});
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  TableDescriptorMap::iterator tableDescIt =
      tableDescriptorMap_.find(to_upper(td->tableName));
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
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);
  // rename all corresponding physical tables if this is a logical table
  const auto physicalTableIt = logicalToPhysicalTableMapById_.find(td->tableId);
  if (physicalTableIt != logicalToPhysicalTableMapById_.end()) {
    const auto physicalTables = physicalTableIt->second;
    CHECK(!physicalTables.empty());
    for (size_t i = 0; i < physicalTables.size(); i++) {
      int32_t physical_tb_id = physicalTables[i];
      const TableDescriptor* phys_td = getMetadataForTable(physical_tb_id);
      CHECK(phys_td);
      std::string newPhysTableName =
          generatePhysicalTableName(newTableName, static_cast<int32_t>(i + 1));
      renamePhysicalTable(phys_td, newPhysTableName);
    }
  }
  renamePhysicalTable(td, newTableName);

  // update table name in direct and effective priv map
  DBObject object(newTableName, TableDBObjectType);
  DBObjectKey key;
  key.dbId = currentDB_.dbId;
  key.objectId = td->tableId;
  key.permissionType = static_cast<int>(DBObjectType::TableDBObjectType);
  object.setObjectKey(key);
  auto objdescs = SysCatalog::instance().getMetadataForObject(
      currentDB_.dbId, static_cast<int>(DBObjectType::TableDBObjectType), td->tableId);
  for (auto obj : objdescs) {
    Grantee* grnt = SysCatalog::instance().getGrantee(obj->roleName);
    if (grnt) {
      grnt->renameDbObject(object);
    }
  }
  SysCatalog::instance().renameObjectsInDescriptorMap(object, *this);
}

void Catalog::renameColumn(const TableDescriptor* td,
                           const ColumnDescriptor* cd,
                           const string& newColumnName) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    for (int i = 0; i <= cd->columnType.get_physical_cols(); ++i) {
      auto cdx = getMetadataForColumn(td->tableId, cd->columnId + i);
      CHECK(cdx);
      std::string new_column_name = cdx->columnName;
      new_column_name.replace(0, cd->columnName.size(), newColumnName);
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_columns SET name = ? WHERE tableid = ? AND columnid = ?",
          std::vector<std::string>{new_column_name,
                                   std::to_string(td->tableId),
                                   std::to_string(cdx->columnId)});
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
  for (int i = 0; i <= cd->columnType.get_physical_cols(); ++i) {
    auto cdx = getMetadataForColumn(td->tableId, cd->columnId + i);
    CHECK(cdx);
    ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.find(
        std::make_tuple(td->tableId, to_upper(cdx->columnName)));
    CHECK(columnDescIt != columnDescriptorMap_.end());
    ColumnDescriptor* changeCd = columnDescIt->second;
    changeCd->columnName.replace(0, cd->columnName.size(), newColumnName);
    columnDescriptorMap_.erase(columnDescIt);  // erase entry under old name
    columnDescriptorMap_[std::make_tuple(td->tableId, to_upper(changeCd->columnName))] =
        changeCd;
  }
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
}

int32_t Catalog::createFrontendView(FrontendViewDescriptor& vd) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    // TODO(andrew): this should be an upsert
    sqliteConnector_.query_with_text_params(
        "SELECT id FROM mapd_dashboards WHERE name = ? and userid = ?",
        std::vector<std::string>{vd.viewName, std::to_string(vd.userId)});
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_dashboards SET state = ?, image_hash = ?, metadata = ?, "
          "update_time = "
          "datetime('now') where name = ? "
          "and userid = ?",
          std::vector<std::string>{vd.viewState,
                                   vd.imageHash,
                                   vd.viewMetadata,
                                   vd.viewName,
                                   std::to_string(vd.userId)});
    } else {
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_dashboards (name, state, image_hash, metadata, update_time, "
          "userid) "
          "VALUES "
          "(?,?,?,?, "
          "datetime('now'), ?)",
          std::vector<std::string>{vd.viewName,
                                   vd.viewState,
                                   vd.imageHash,
                                   vd.viewMetadata,
                                   std::to_string(vd.userId)});
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");

  // now get the auto generated viewid
  try {
    sqliteConnector_.query_with_text_params(
        "SELECT id, strftime('%Y-%m-%dT%H:%M:%SZ', update_time) FROM mapd_dashboards "
        "WHERE name = ? and userid = ?",
        std::vector<std::string>{vd.viewName, std::to_string(vd.userId)});
    vd.viewId = sqliteConnector_.getData<int>(0, 0);
    vd.updateTime = sqliteConnector_.getData<std::string>(0, 1);
  } catch (std::exception& e) {
    throw;
  }
  vd.viewSystemRoleName = generate_dash_system_rolename(std::to_string(currentDB_.dbId),
                                                        std::to_string(vd.viewId));
  addFrontendViewToMap(vd);
  // NOTE(wamsi): Transactionally unsafe
  createOrUpdateDashboardSystemRole(vd.viewMetadata, vd.userId, vd.viewSystemRoleName);
  return vd.viewId;
}

void Catalog::replaceDashboard(FrontendViewDescriptor& vd) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);

  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "SELECT id FROM mapd_dashboards WHERE id = ?",
        std::vector<std::string>{std::to_string(vd.viewId)});
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_dashboards SET name = ?, state = ?, image_hash = ?, metadata = ?, "
          "update_time = "
          "datetime('now') where id = ? ",
          std::vector<std::string>{vd.viewName,
                                   vd.viewState,
                                   vd.imageHash,
                                   vd.viewMetadata,
                                   std::to_string(vd.viewId)});
    } else {
      LOG(ERROR) << "Error replacing dashboard id " << vd.viewId
                 << " does not exist in db";
      throw runtime_error("Error replacing dashboard id " + std::to_string(vd.viewId) +
                          " does not exist in db");
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");

  bool found{false};
  for (auto descp : dashboardDescriptorMap_) {
    auto dash = descp.second.get();
    if (dash->viewId == vd.viewId) {
      found = true;
      auto viewDescIt = dashboardDescriptorMap_.find(std::to_string(dash->userId) + ":" +
                                                     dash->viewName);
      if (viewDescIt ==
          dashboardDescriptorMap_.end()) {  // check to make sure view exists
        LOG(ERROR) << "No metadata for dashboard for user " << dash->userId
                   << " dashboard " << dash->viewName << " does not exist in map";
        throw runtime_error("No metadata for dashboard for user " +
                            std::to_string(dash->userId) + " dashboard " +
                            dash->viewName + " does not exist in map");
      }
      dashboardDescriptorMap_.erase(viewDescIt);
      break;
    }
  }
  if (!found) {
    LOG(ERROR) << "Error replacing dashboard id " << vd.viewId
               << " does not exist in map";
    throw runtime_error("Error replacing dashboard id " + std::to_string(vd.viewId) +
                        " does not exist in map");
  }

  // now reload the object
  sqliteConnector_.query_with_text_params(
      "SELECT id, strftime('%Y-%m-%dT%H:%M:%SZ', update_time)  FROM "
      "mapd_dashboards "
      "WHERE id = ?",
      std::vector<std::string>{std::to_string(vd.viewId)});
  vd.updateTime = sqliteConnector_.getData<string>(0, 1);
  vd.viewSystemRoleName = generate_dash_system_rolename(std::to_string(currentDB_.dbId),
                                                        std::to_string(vd.viewId));
  addFrontendViewToMapNoLock(vd);
  // NOTE(wamsi): Transactionally unsafe
  createOrUpdateDashboardSystemRole(vd.viewMetadata, vd.userId, vd.viewSystemRoleName);
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
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(this);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    ld.link = calculateSHA1(ld.viewState + ld.viewMetadata + std::to_string(ld.userId))
                  .substr(0, 8);
    sqliteConnector_.query_with_text_params(
        "SELECT linkid FROM mapd_links WHERE link = ? and userid = ?",
        std::vector<std::string>{ld.link, std::to_string(ld.userId)});
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_links SET update_time = datetime('now') WHERE userid = ? AND link "
          "= ?",
          std::vector<std::string>{std::to_string(ld.userId), ld.link});
    } else {
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_links (userid, link, view_state, view_metadata, update_time) "
          "VALUES (?,?,?,?, "
          "datetime('now'))",
          std::vector<std::string>{
              std::to_string(ld.userId), ld.link, ld.viewState, ld.viewMetadata});
    }
    // now get the auto generated viewid
    sqliteConnector_.query_with_text_param(
        "SELECT linkid, strftime('%Y-%m-%dT%H:%M:%SZ', update_time) FROM mapd_links "
        "WHERE link = ?",
        ld.link);
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

const ColumnDescriptor* Catalog::getShardColumnMetadataForTable(
    const TableDescriptor* td) const {
  const auto column_descriptors =
      getAllColumnMetadataForTable(td->tableId, false, true, true);

  const ColumnDescriptor* shard_cd{nullptr};
  int i = 1;
  for (auto cd_itr = column_descriptors.begin(); cd_itr != column_descriptors.end();
       ++cd_itr, ++i) {
    if (i == td->shardedColumnId) {
      shard_cd = *cd_itr;
    }
  }
  return shard_cd;
}

std::vector<const TableDescriptor*> Catalog::getPhysicalTablesDescriptors(
    const TableDescriptor* logicalTableDesc) const {
  cat_read_lock read_lock(this);
  const auto physicalTableIt =
      logicalToPhysicalTableMapById_.find(logicalTableDesc->tableId);
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

int Catalog::getLogicalTableId(const int physicalTableId) const {
  for (const auto& l : logicalToPhysicalTableMapById_) {
    if (l.second.end() != std::find_if(l.second.begin(),
                                       l.second.end(),
                                       [&](decltype(*l.second.begin()) tid) -> bool {
                                         return physicalTableId == tid;
                                       })) {
      return l.first;
    }
  }
  return physicalTableId;
}

void Catalog::checkpoint(const int logicalTableId) const {
  const auto td = getMetadataForTable(logicalTableId);
  const auto shards = getPhysicalTablesDescriptors(td);
  for (const auto shard : shards) {
    getDataMgr().checkpoint(getCurrentDB().dbId, shard->tableId);
  }
}

void Catalog::eraseDBData() {
  cat_write_lock write_lock(this);
  // Physically erase all tables and dictionaries from disc and memory
  const auto tables = getAllTableMetadata();
  for (const auto table : tables) {
    eraseTablePhysicalData(table);
  }
  // Physically erase database metadata
  boost::filesystem::remove(basePath_ + "/mapd_catalogs/" + currentDB_.dbName);
  calciteMgr_->updateMetadata(currentDB_.dbName, "");
}

void Catalog::eraseTablePhysicalData(const TableDescriptor* td) {
  cat_write_lock write_lock(this);
  const int tableId = td->tableId;
  // must destroy fragmenter before deleteChunks is called.
  if (td->fragmenter != nullptr) {
    auto tableDescIt = tableDescriptorMapById_.find(tableId);
    {
      INJECT_TIMER(deleting_fragmenter);
      delete td->fragmenter;
    }
    tableDescIt->second->fragmenter = nullptr;  // get around const-ness
  }
  ChunkKey chunkKeyPrefix = {currentDB_.dbId, tableId};
  {
    INJECT_TIMER(deleteChunksWithPrefix);
    // assuming deleteChunksWithPrefix is atomic
    dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix, MemoryLevel::CPU_LEVEL);
    dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix, MemoryLevel::GPU_LEVEL);
  }
  if (!td->isView) {
    INJECT_TIMER(Remove_Table);
    dataMgr_->removeTableRelatedDS(currentDB_.dbId, tableId);
  }
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
  {
    INJECT_TIMER(removeTableFromMap_);
    removeTableFromMap(td->tableName, tableId);
  }
}

std::string Catalog::generatePhysicalTableName(const std::string& logicalTableName,
                                               const int32_t& shardNumber) {
  std::string physicalTableName =
      logicalTableName + physicalTableNameTag_ + std::to_string(shardNumber);
  return (physicalTableName);
}

void Catalog::set(const std::string& dbName, std::shared_ptr<Catalog> cat) {
  mapd_cat_map_[dbName] = cat;
}

std::shared_ptr<Catalog> Catalog::get(const std::string& dbName) {
  auto cat_it = mapd_cat_map_.find(dbName);
  if (cat_it != mapd_cat_map_.end()) {
    return cat_it->second;
  }
  return nullptr;
}

std::shared_ptr<Catalog> Catalog::get(const string& basePath,
                                      const DBMetadata& curDB,
                                      std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                                      const std::vector<LeafHostInfo>& string_dict_hosts,
                                      std::shared_ptr<Calcite> calcite,
                                      bool is_new_db) {
  auto cat = Catalog::get(curDB.dbName);

  if (cat) {
    return cat;
  } else {
    cat = std::make_shared<Catalog>(
        basePath, curDB, dataMgr, string_dict_hosts, calcite, is_new_db);
    Catalog::set(curDB.dbName, cat);
    return cat;
  }
}

void Catalog::remove(const std::string& dbName) {
  mapd_cat_map_.erase(dbName);
}

void Catalog::optimizeTable(const TableDescriptor* td) const {
  cat_read_lock read_lock(this);
  // "if not a table that supports delete return nullptr,  nothing more to do"
  const ColumnDescriptor* cd = getDeletedColumn(td);
  if (nullptr == cd) {
    return;
  }
  // vacuum chunks which show sign of deleted rows in metadata
  ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId, cd->columnId};
  std::vector<std::pair<ChunkKey, ChunkMetadata>> chunkMetadataVec;
  dataMgr_->getChunkMetadataVecForKeyPrefix(chunkMetadataVec, chunkKeyPrefix);
  for (auto cm : chunkMetadataVec) {
    // "delete has occured"
    if (cm.second.chunkStats.max.tinyintval == 1) {
      UpdelRoll updel_roll;
      updel_roll.catalog = this;
      updel_roll.logicalTableId = getLogicalTableId(td->tableId);
      updel_roll.memoryLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
      const auto cd = getMetadataForColumn(td->tableId, cm.first[2]);
      const auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                                   &getDataMgr(),
                                                   cm.first,
                                                   updel_roll.memoryLevel,
                                                   0,
                                                   cm.second.numBytes,
                                                   cm.second.numElements);
      td->fragmenter->compactRows(this,
                                  td,
                                  cm.first[3],
                                  td->fragmenter->getVacuumOffsets(chunk),
                                  updel_roll.memoryLevel,
                                  updel_roll);
      updel_roll.commitUpdate();
    }
  }
}

}  // namespace Catalog_Namespace
