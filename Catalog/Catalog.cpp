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

#include "Catalog/Catalog.h"

#include <algorithm>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/adaptor/map.hpp>
#include <boost/version.hpp>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <list>
#include <memory>
#include <random>
#include <regex>
#include <sstream>

#if BOOST_VERSION >= 106600
#include <boost/uuid/detail/sha1.hpp>
#else
#include <boost/uuid/sha1.hpp>
#endif
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include "Catalog/SysCatalog.h"

#include "QueryEngine/Execute.h"
#include "QueryEngine/TableOptimizer.h"

#include "DataMgr/FileMgr/FileMgr.h"
#include "DataMgr/FileMgr/GlobalFileMgr.h"
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"
#include "Fragmenter/Fragmenter.h"
#include "Fragmenter/SortedOrderFragmenter.h"
#include "LockMgr/LockMgr.h"
#include "MigrationMgr/MigrationMgr.h"
#include "Parser/ParserNode.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/TableOptimizer.h"
#include "Shared/DateTimeParser.h"
#include "Shared/File.h"
#include "Shared/StringTransform.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/sqltypes_geo.h"
#include "StringDictionary/StringDictionaryClient.h"

#include "MapDRelease.h"
#include "RWLocks.h"
#include "SharedDictionaryValidator.h"

using Chunk_NS::Chunk;
using Fragmenter_Namespace::InsertOrderFragmenter;
using Fragmenter_Namespace::SortedOrderFragmenter;
using std::list;
using std::map;
using std::pair;
using std::runtime_error;
using std::string;
using std::vector;

int g_test_against_columnId_gap{0};
bool g_enable_fsi{false};
bool g_enable_s3_fsi{false};
extern bool g_cache_string_hash;

// Serialize temp tables to a json file in the Catalogs directory for Calcite parsing
// under unit testing.
bool g_serialize_temp_tables{true};

namespace Catalog_Namespace {

const int DEFAULT_INITIAL_VERSION = 1;  // start at version 1
const int MAPD_TEMP_TABLE_START_ID =
    1073741824;  // 2^30, give room for over a billion non-temp tables
const int MAPD_TEMP_DICT_START_ID =
    1073741824;  // 2^30, give room for over a billion non-temp dictionaries

thread_local bool Catalog::thread_holds_read_lock = false;

using sys_read_lock = read_lock<SysCatalog>;
using cat_read_lock = read_lock<Catalog>;
using cat_write_lock = write_lock<Catalog>;
using cat_sqlite_lock = sqlite_lock<Catalog>;

// migration will be done as two step process this release
// will create and use new table
// next release will remove old table, doing this to have fall back path
// incase of migration failure
void Catalog::updateFrontendViewsToDashboards() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
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

namespace {

inline auto table_json_filepath(const std::string& base_path,
                                const std::string& db_name) {
  return boost::filesystem::path(base_path + "/mapd_catalogs/" + db_name +
                                 "_temp_tables.json");
}

}  // namespace

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
  if (g_serialize_temp_tables) {
    boost::filesystem::remove(table_json_filepath(basePath_, currentDB_.dbName));
  }
  // once all initialized use real object
  initialized_ = true;
}

Catalog::Catalog() {}

Catalog::~Catalog() {
  cat_write_lock write_lock(this);
  // must clean up heap-allocated TableDescriptor and ColumnDescriptor structs
  for (TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.begin();
       tableDescIt != tableDescriptorMap_.end();
       ++tableDescIt) {
    tableDescIt->second->fragmenter = nullptr;
    delete tableDescIt->second;
  }

  // TableDescriptorMapById points to the same descriptors.  No need to delete

  for (ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin();
       columnDescIt != columnDescriptorMap_.end();
       ++columnDescIt) {
    delete columnDescIt->second;
  }

  // ColumnDescriptorMapById points to the same descriptors.  No need to delete

  if (g_serialize_temp_tables) {
    boost::filesystem::remove(table_json_filepath(basePath_, currentDB_.dbName));
  }
}

const Catalog* Catalog::getObjForLock() {
  if (initialized_) {
    return this;
  } else {
    return SysCatalog::instance().getDummyCatalog().get();
  }
}

void Catalog::updateTableDescriptorSchema() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
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
                         std::to_string(OMNISCI_ROOT_USER_ID));
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("sort_column_id")) ==
        cols.end()) {
      sqliteConnector_.query(
          "ALTER TABLE mapd_tables ADD sort_column_id INTEGER DEFAULT 0");
    }
    if (std::find(cols.begin(), cols.end(), std::string("storage_type")) == cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD storage_type TEXT DEFAULT ''");
      sqliteConnector_.query(queryString);
    }
    if (std::find(cols.begin(), cols.end(), std::string("max_rollback_epochs")) ==
        cols.end()) {
      string queryString("ALTER TABLE mapd_tables ADD max_rollback_epochs INT DEFAULT " +
                         std::to_string(-1));
      sqliteConnector_.query(queryString);
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateFixlenArrayColumns() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "select name from sqlite_master WHERE type='table' AND "
        "name='mapd_version_history'");
    if (sqliteConnector_.getNumRows() == 0) {
      sqliteConnector_.query(
          "CREATE TABLE mapd_version_history(version integer, migration_history text "
          "unique)");
    } else {
      sqliteConnector_.query(
          "select * from mapd_version_history where migration_history = "
          "'notnull_fixlen_arrays'");
      if (sqliteConnector_.getNumRows() != 0) {
        // legacy fixlen arrays had migrated
        // no need for further execution
        sqliteConnector_.query("END TRANSACTION");
        return;
      }
    }
    // Insert check for migration
    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_version_history(version, migration_history) values(?,?)",
        std::vector<std::string>{std::to_string(MAPD_VERSION), "notnull_fixlen_arrays"});
    LOG(INFO) << "Updating mapd_columns, legacy fixlen arrays";
    // Upating all fixlen array columns
    string queryString("UPDATE mapd_columns SET is_notnull=1 WHERE coltype=" +
                       std::to_string(kARRAY) + " AND size>0;");
    sqliteConnector_.query(queryString);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateGeoColumns() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "select name from sqlite_master WHERE type='table' AND "
        "name='mapd_version_history'");
    if (sqliteConnector_.getNumRows() == 0) {
      sqliteConnector_.query(
          "CREATE TABLE mapd_version_history(version integer, migration_history text "
          "unique)");
    } else {
      sqliteConnector_.query(
          "select * from mapd_version_history where migration_history = "
          "'notnull_geo_columns'");
      if (sqliteConnector_.getNumRows() != 0) {
        // legacy geo columns had migrated
        // no need for further execution
        sqliteConnector_.query("END TRANSACTION");
        return;
      }
    }
    // Insert check for migration
    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_version_history(version, migration_history) values(?,?)",
        std::vector<std::string>{std::to_string(MAPD_VERSION), "notnull_geo_columns"});
    LOG(INFO) << "Updating mapd_columns, legacy geo columns";
    // Upating all geo columns
    string queryString(
        "UPDATE mapd_columns SET is_notnull=1 WHERE coltype=" + std::to_string(kPOINT) +
        " OR coltype=" + std::to_string(kLINESTRING) + " OR coltype=" +
        std::to_string(kPOLYGON) + " OR coltype=" + std::to_string(kMULTIPOLYGON) + ";");
    sqliteConnector_.query(queryString);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::updateFrontendViewSchema() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
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
  cat_sqlite_lock sqlite_lock(getObjForLock());
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
  cat_sqlite_lock sqlite_lock(getObjForLock());
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
  cat_sqlite_lock sqlite_lock(getObjForLock());
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
  cat_sqlite_lock sqlite_lock(getObjForLock());
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

void Catalog::updateDefaultColumnValues() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("PRAGMA TABLE_INFO(mapd_columns)");
    std::vector<std::string> cols;
    for (size_t i = 0; i < sqliteConnector_.getNumRows(); i++) {
      cols.push_back(sqliteConnector_.getData<std::string>(i, 1));
    }
    if (std::find(cols.begin(), cols.end(), std::string("default_value")) == cols.end()) {
      LOG(INFO) << "Adding support for default values to mapd_columns";
      sqliteConnector_.query("ALTER TABLE mapd_columns ADD default_value TEXT");
    }
  } catch (std::exception& e) {
    LOG(ERROR) << "Failed to make metadata update for default values` support";
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

// introduce DB version into the dictionary tables
// if the DB does not have a version rename all dictionary tables

void Catalog::updateDictionaryNames() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
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
            g_base_path + "/mapd_data/" + currentDB_.dbName + "_" + dictName;
        std::string newName = g_base_path + "/mapd_data/DB_" +
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

void Catalog::updateDictionarySchema() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
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

void Catalog::updateCustomExpressionsSchema() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(getCustomExpressionsSchema(true));
  } catch (const std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

const std::string Catalog::getCustomExpressionsSchema(bool if_not_exists) {
  return "CREATE TABLE " + (if_not_exists ? std::string{"IF NOT EXISTS "} : "") +
         "omnisci_custom_expressions(id integer primary key, name text, " +
         "expression_json text, data_source_type text, " +
         "data_source_id integer, is_deleted boolean)";
}

void Catalog::recordOwnershipOfObjectsInObjectPermissions() {
  cat_sqlite_lock sqlite_lock(getObjForLock());
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
  cat_sqlite_lock sqlite_lock(getObjForLock());
  migrations::MigrationMgr::migrateDateInDaysMetadata(
      tableDescriptorMapById_, getCurrentDB().dbId, this, sqliteConnector_);
}

void Catalog::createDashboardSystemRoles() {
  std::unordered_map<std::string, std::pair<int, std::string>> dashboards;
  std::vector<std::string> dashboard_ids;
  static const std::string migration_name{"dashboard_roles_migration"};
  {
    cat_sqlite_lock sqlite_lock(getObjForLock());
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
        if (SysCatalog::instance().getRoleGrantee(generate_dashboard_system_rolename(
                std::to_string(currentDB_.dbId),
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
      createOrUpdateDashboardSystemRole(dash.second.second,
                                        dash.second.first,
                                        generate_dashboard_system_rolename(
                                            std::to_string(currentDB_.dbId), dash.first));
      auto result = active_grantees.find(dash.first);
      if (result != active_grantees.end()) {
        SysCatalog::instance().grantRoleBatch(
            {generate_dashboard_system_rolename(std::to_string(currentDB_.dbId),
                                                dash.first)},
            result->second);
      }
    }
    cat_sqlite_lock sqlite_lock(getObjForLock());
    // check if this has already been completed
    sqliteConnector_.query(
        "select * from mapd_version_history where migration_history = '" +
        migration_name + "'");
    if (sqliteConnector_.getNumRows() != 0) {
      return;
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
  updateFixlenArrayColumns();
  updateGeoColumns();
  updateFrontendViewAndLinkUsers();
  updateFrontendViewSchema();
  updateLinkSchema();
  updateDictionaryNames();
  updateDictionarySchema();
  updatePageSize();
  updateDeletedColumnIndicator();
  updateFrontendViewsToDashboards();
  recordOwnershipOfObjectsInObjectPermissions();
  updateCustomExpressionsSchema();
  updateDefaultColumnValues();
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
  cat_sqlite_lock sqlite_lock(getObjForLock());

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
    std::string fname = g_base_path + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) +
                        "_DICT_" + std::to_string(dictId);
    DictRef dict_ref(currentDB_.dbId, dictId);
    DictDescriptor* dd = new DictDescriptor(
        dict_ref, dictName, dictNBits, is_shared, refcount, fname, false);
    dictDescriptorMapByRef_[dict_ref].reset(dd);
  }

  string tableQuery(
      "SELECT tableid, name, ncolumns, isview, fragments, frag_type, max_frag_rows, "
      "max_chunk_size, frag_page_size, "
      "max_rows, partitions, shard_column_id, shard, num_shards, key_metainfo, userid, "
      "sort_column_id, storage_type, max_rollback_epochs "
      "from mapd_tables");
  sqliteConnector_.query(tableQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    TableDescriptor* td;
    const auto& storage_type = sqliteConnector_.getData<string>(r, 17);
    if (!storage_type.empty()) {
      const auto table_id = sqliteConnector_.getData<int>(r, 0);
      const auto& table_name = sqliteConnector_.getData<string>(r, 1);
      LOG(FATAL) << "Unable to read Catalog metadata: storage type is currently not a "
                    "supported table option (table "
                 << table_name << " [" << table_id << "] in database "
                 << currentDB_.dbName << ").";
    }

    td = new TableDescriptor();
    td->storageType = storage_type;
    td->tableId = sqliteConnector_.getData<int>(r, 0);
    td->tableName = sqliteConnector_.getData<string>(r, 1);
    td->nColumns = sqliteConnector_.getData<int>(r, 2);
    td->isView = sqliteConnector_.getData<bool>(r, 3);
    td->fragments = sqliteConnector_.getData<string>(r, 4);
    td->fragType =
        (Fragmenter_Namespace::FragmenterType)sqliteConnector_.getData<int>(r, 5);
    td->maxFragRows = sqliteConnector_.getData<int>(r, 6);
    td->maxChunkSize = sqliteConnector_.getData<int64_t>(r, 7);
    td->fragPageSize = sqliteConnector_.getData<int>(r, 8);
    td->maxRows = sqliteConnector_.getData<int64_t>(r, 9);
    td->partitions = sqliteConnector_.getData<string>(r, 10);
    CHECK_EQ(sqliteConnector_.getData<int>(r, 11), 0);
    CHECK_EQ(sqliteConnector_.getData<int>(r, 12), -1);
    CHECK_EQ(sqliteConnector_.getData<int>(r, 13), 0);
    td->keyMetainfo = sqliteConnector_.getData<string>(r, 14);
    td->userId = sqliteConnector_.getData<int>(r, 15);
    td->sortedColumnId =
        sqliteConnector_.isNull(r, 16) ? 0 : sqliteConnector_.getData<int>(r, 16);
    if (!td->isView) {
      td->fragmenter = nullptr;
    }
    td->maxRollbackEpochs = sqliteConnector_.getData<int>(r, 18);
    td->hasDeletedCol = false;

    tableDescriptorMap_[to_upper(td->tableName)] = td;
    tableDescriptorMapById_[td->tableId] = td;
  }

  string columnQuery(
      "SELECT tableid, columnid, name, coltype, colsubtype, coldim, colscale, "
      "is_notnull, compression, comp_param, "
      "size, chunks, is_systemcol, is_virtualcol, virtual_expr, is_deletedcol, "
      "default_value from "
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
    if (sqliteConnector_.isNull(r, 16)) {
      cd->default_value = std::nullopt;
    } else {
      cd->default_value = std::make_optional(sqliteConnector_.getData<string>(r, 16));
    }
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
    std::shared_ptr<DashboardDescriptor> vd = std::make_shared<DashboardDescriptor>();
    vd->dashboardId = sqliteConnector_.getData<int>(r, 0);
    vd->dashboardState = sqliteConnector_.getData<string>(r, 1);
    vd->dashboardName = sqliteConnector_.getData<string>(r, 2);
    vd->imageHash = sqliteConnector_.getData<string>(r, 3);
    vd->updateTime = sqliteConnector_.getData<string>(r, 4);
    vd->userId = sqliteConnector_.getData<int>(r, 5);
    vd->dashboardMetadata = sqliteConnector_.getData<string>(r, 6);
    vd->user = getUserFromId(vd->userId);
    vd->dashboardSystemRoleName = generate_dashboard_system_rolename(
        std::to_string(currentDB_.dbId), sqliteConnector_.getData<string>(r, 0));
    dashboardDescriptorMap_[std::to_string(vd->userId) + ":" + vd->dashboardName] = vd;
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

  buildCustomExpressionsMap();
}

void Catalog::buildCustomExpressionsMap() {
  sqliteConnector_.query(
      "SELECT id, name, expression_json, data_source_type, data_source_id, "
      "is_deleted "
      "FROM omnisci_custom_expressions");
  auto num_rows = sqliteConnector_.getNumRows();
  for (size_t row = 0; row < num_rows; row++) {
    auto custom_expr = getCustomExpressionFromConnector(row);
    custom_expr_map_by_id_[custom_expr->id] = std::move(custom_expr);
  }
}

std::unique_ptr<CustomExpression> Catalog::getCustomExpressionFromConnector(size_t row) {
  auto id = sqliteConnector_.getData<int>(row, 0);
  auto name = sqliteConnector_.getData<string>(row, 1);
  auto expression_json = sqliteConnector_.getData<string>(row, 2);
  auto data_source_type_str = sqliteConnector_.getData<string>(row, 3);
  auto data_source_id = sqliteConnector_.getData<int>(row, 4);
  auto is_deleted = sqliteConnector_.getData<bool>(row, 5);
  return std::make_unique<CustomExpression>(
      id,
      name,
      expression_json,
      CustomExpression::dataSourceTypeFromString(data_source_type_str),
      data_source_id,
      is_deleted);
}

void Catalog::addTableToMap(const TableDescriptor* td,
                            const list<ColumnDescriptor>& columns,
                            const list<DictDescriptor>& dicts) {
  cat_write_lock write_lock(this);
  TableDescriptor* new_td = new TableDescriptor();
  *new_td = *td;

  new_td->mutex_ = std::make_shared<std::mutex>();
  tableDescriptorMap_[to_upper(td->tableName)] = new_td;
  tableDescriptorMapById_[td->tableId] = new_td;
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

void Catalog::removeTableFromMap(const string& tableName,
                                 const int tableId,
                                 const bool is_on_error) {
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
  td->fragmenter = nullptr;

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
        // If we're removing this table due to an error, it is possible that the string
        // dictionary reference was never populated. Don't crash, just continue cleaning
        // up the TableDescriptor and ColumnDescriptors
        if (!is_on_error) {
          CHECK(dictIt != dictDescriptorMapByRef_.end());
        } else {
          if (dictIt == dictDescriptorMapByRef_.end()) {
            continue;
          }
        }
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

void Catalog::addFrontendViewToMap(DashboardDescriptor& vd) {
  cat_write_lock write_lock(this);
  addFrontendViewToMapNoLock(vd);
}

void Catalog::addFrontendViewToMapNoLock(DashboardDescriptor& vd) {
  cat_write_lock write_lock(this);
  dashboardDescriptorMap_[std::to_string(vd.userId) + ":" + vd.dashboardName] =
      std::make_shared<DashboardDescriptor>(vd);
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
  for (auto object_name : parse_underlying_dashboard_objects(view_meta)) {
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
    SysCatalog::instance().createRole(
        dash_role_name, /*user_private_role=*/false, /*is_temporary=*/false);
    SysCatalog::instance().grantDBObjectPrivilegesBatch({dash_role_name}, objects, *this);
  } else {
    // Dashboard system role already exists
    // Add/remove privileges on objects
    std::set<DBObjectKey> revoke_keys;
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
        revoke_keys.insert(key);
      }
    }
    for (auto& key : revoke_keys) {
      // revoke privs on object since the object is no
      // longer used by the dashboard as source
      // NOTE(wamsi): Transactionally unsafe
      SysCatalog::instance().revokeDBObjectPrivileges(
          dash_role_name, *rl->findDbObject(key, true), *this);
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
    CHECK_EQ(td->fragType, Fragmenter_Namespace::FragmenterType::INSERT_ORDER);
    vector<Chunk> chunkVec;
    list<const ColumnDescriptor*> columnDescs;
    getAllColumnMetadataForTableImpl(td, columnDescs, true, false, true);
    for (auto& cd : columnDescs) {
      chunkVec.emplace_back(cd->makeInfo(currentDB_.dbId));
    }
    ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId};
    if (td->sortedColumnId > 0) {
      td->fragmenter = std::make_shared<SortedOrderFragmenter>(chunkKeyPrefix,
                                                               chunkVec,
                                                               dataMgr_.get(),
                                                               const_cast<Catalog*>(this),
                                                               td->tableId,
                                                               td->maxFragRows,
                                                               td->maxChunkSize,
                                                               td->fragPageSize,
                                                               td->maxRows,
                                                               td->persistenceLevel);
    } else {
      td->fragmenter = std::make_shared<InsertOrderFragmenter>(chunkKeyPrefix,
                                                               chunkVec,
                                                               dataMgr_.get(),
                                                               const_cast<Catalog*>(this),
                                                               td->tableId,
                                                               td->maxFragRows,
                                                               td->maxChunkSize,
                                                               td->fragPageSize,
                                                               td->maxRows,
                                                               td->persistenceLevel,
                                                               !td->storageType.empty());
    }
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
  auto tableDescIt = tableDescriptorMapById_.find(tableId);
  if (tableDescIt == tableDescriptorMapById_.end()) {  // check to make sure table exists
    return nullptr;
  }
  TableDescriptor* td = tableDescIt->second;
  if (populateFragmenter) {
    std::unique_lock<std::mutex> td_lock(*td->mutex_.get());
    if (td->fragmenter == nullptr && !td->isView) {
      instantiateFragmenter(td);
    }
  }
  return td;  // returns pointer to table descriptor
}

const TableDescriptor* Catalog::getMetadataForTable(int tableId,
                                                    bool populateFragmenter) const {
  cat_read_lock read_lock(this);
  return getMetadataForTableImpl(tableId, populateFragmenter);
}

const DictDescriptor* Catalog::getMetadataForDict(const int dict_id,
                                                  const bool load_dict) const {
  cat_read_lock read_lock(this);
  return getMetadataForDictUnlocked(dict_id, load_dict);
}

TableDescriptor* Catalog::getMutableMetadataForTableUnlocked(int tableId) {
  auto tableDescIt = tableDescriptorMapById_.find(tableId);
  if (tableDescIt == tableDescriptorMapById_.end()) {  // check to make sure table exists
    return nullptr;
  }
  return tableDescIt->second;
}

const DictDescriptor* Catalog::getMetadataForDictUnlocked(const int dictId,
                                                          const bool loadDict) const {
  const DictRef dictRef(currentDB_.dbId, dictId);
  auto dictDescIt = dictDescriptorMapByRef_.find(dictRef);
  if (dictDescIt ==
      dictDescriptorMapByRef_.end()) {  // check to make sure dictionary exists
    return nullptr;
  }
  auto& dd = dictDescIt->second;

  if (loadDict) {
    std::lock_guard string_dict_lock(*dd->string_dict_mutex);
    if (!dd->stringDict) {
      auto time_ms = measure<>::execution([&]() {
        if (string_dict_hosts_.empty()) {
          if (dd->dictIsTemp) {
            dd->stringDict = std::make_shared<StringDictionary>(
                dd->dictFolderPath, true, true, g_cache_string_hash);
          } else {
            dd->stringDict = std::make_shared<StringDictionary>(
                dd->dictFolderPath, false, true, g_cache_string_hash);
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

const ColumnDescriptor* Catalog::getMetadataForColumn(int table_id, int column_id) const {
  cat_read_lock read_lock(this);
  return getMetadataForColumnUnlocked(table_id, column_id);
}

const ColumnDescriptor* Catalog::getMetadataForColumnUnlocked(int tableId,
                                                              int columnId) const {
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

void Catalog::deleteMetadataForDashboards(const std::vector<int32_t> dashboard_ids,
                                          const UserMetadata& user) {
  std::stringstream invalid_ids, restricted_ids;

  for (int32_t dashboard_id : dashboard_ids) {
    if (!getMetadataForDashboard(dashboard_id)) {
      invalid_ids << (!invalid_ids.str().empty() ? ", " : "") << dashboard_id;
      continue;
    }
    DBObject object(dashboard_id, DashboardDBObjectType);
    object.loadKey(*this);
    object.setPrivileges(AccessPrivileges::DELETE_DASHBOARD);
    std::vector<DBObject> privs = {object};
    if (!SysCatalog::instance().checkPrivileges(user, privs))
      restricted_ids << (!restricted_ids.str().empty() ? ", " : "") << dashboard_id;
  }

  if (invalid_ids.str().size() > 0 || restricted_ids.str().size() > 0) {
    std::stringstream error_message;
    error_message << "Delete dashboard(s) failed with error(s):";
    if (invalid_ids.str().size() > 0)
      error_message << "\nDashboard id: " << invalid_ids.str()
                    << " - Dashboard id does not exist";
    if (restricted_ids.str().size() > 0)
      error_message
          << "\nDashboard id: " << restricted_ids.str()
          << " - User should be either owner of dashboard or super user to delete it";
    throw std::runtime_error(error_message.str());
  }
  std::vector<DBObject> dash_objs;

  for (int32_t dashboard_id : dashboard_ids) {
    dash_objs.push_back(DBObject(dashboard_id, DashboardDBObjectType));
  }
  // BE-5245: Transactionally unsafe (like other combined Catalog/Syscatalog operations)
  SysCatalog::instance().revokeDBObjectPrivilegesFromAllBatch(dash_objs, this);
  {
    cat_write_lock write_lock(this);
    cat_sqlite_lock sqlite_lock(getObjForLock());

    sqliteConnector_.query("BEGIN TRANSACTION");
    try {
      for (int32_t dashboard_id : dashboard_ids) {
        auto dash = getMetadataForDashboard(dashboard_id);
        // Dash should still exist if revokeDBObjectPrivileges passed but throw and
        // rollback if already deleted
        if (!dash) {
          throw std::runtime_error(
              std::string("Delete dashboard(s) failed with error(s):\nDashboard id: ") +
              std::to_string(dashboard_id) + " - Dashboard id does not exist ");
        }
        std::string user_id = std::to_string(dash->userId);
        std::string dash_name = dash->dashboardName;
        auto viewDescIt = dashboardDescriptorMap_.find(user_id + ":" + dash_name);
        dashboardDescriptorMap_.erase(viewDescIt);
        sqliteConnector_.query_with_text_params(
            "DELETE FROM mapd_dashboards WHERE name = ? and userid = ?",
            std::vector<std::string>{dash_name, user_id});
      }
    } catch (std::exception& e) {
      sqliteConnector_.query("ROLLBACK TRANSACTION");
      throw;
    }
    sqliteConnector_.query("END TRANSACTION");
  }
}

const DashboardDescriptor* Catalog::getMetadataForDashboard(
    const string& userId,
    const string& dashName) const {
  cat_read_lock read_lock(this);

  auto viewDescIt = dashboardDescriptorMap_.find(userId + ":" + dashName);
  if (viewDescIt == dashboardDescriptorMap_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return viewDescIt->second.get();  // returns pointer to view descriptor
}

const DashboardDescriptor* Catalog::getMetadataForDashboard(const int32_t id) const {
  cat_read_lock read_lock(this);
  std::string userId;
  std::string name;
  bool found{false};
  {
    for (auto descp : dashboardDescriptorMap_) {
      auto dash = descp.second.get();
      if (dash->dashboardId == id) {
        userId = std::to_string(dash->userId);
        name = dash->dashboardName;
        found = true;
        break;
      }
    }
  }
  if (found) {
    return getMetadataForDashboard(userId, name);
  }
  return nullptr;
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

void Catalog::getAllColumnMetadataForTableImpl(
    const TableDescriptor* td,
    list<const ColumnDescriptor*>& columnDescriptors,
    const bool fetchSystemColumns,
    const bool fetchVirtualColumns,
    const bool fetchPhysicalColumns) const {
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

std::list<const ColumnDescriptor*> Catalog::getAllColumnMetadataForTable(
    const int tableId,
    const bool fetchSystemColumns,
    const bool fetchVirtualColumns,
    const bool fetchPhysicalColumns) const {
  cat_read_lock read_lock(this);
  return getAllColumnMetadataForTableUnlocked(
      tableId, fetchSystemColumns, fetchVirtualColumns, fetchPhysicalColumns);
}

std::list<const ColumnDescriptor*> Catalog::getAllColumnMetadataForTableUnlocked(
    const int tableId,
    const bool fetchSystemColumns,
    const bool fetchVirtualColumns,
    const bool fetchPhysicalColumns) const {
  std::list<const ColumnDescriptor*> columnDescriptors;
  const TableDescriptor* td =
      getMetadataForTableImpl(tableId, false);  // dont instantiate fragmenter
  getAllColumnMetadataForTableImpl(td,
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

std::vector<TableDescriptor> Catalog::getAllTableMetadataCopy() const {
  cat_read_lock read_lock(this);
  std::vector<TableDescriptor> tables;
  tables.reserve(tableDescriptorMapById_.size());
  for (auto table_entry : tableDescriptorMapById_) {
    tables.emplace_back(*table_entry.second);
  }
  return tables;
}

list<const DashboardDescriptor*> Catalog::getAllDashboardsMetadata() const {
  list<const DashboardDescriptor*> dashboards;
  for (auto dashboard_entry : dashboardDescriptorMap_) {
    dashboards.push_back(dashboard_entry.second.get());
  }
  return dashboards;
}

std::vector<DashboardDescriptor> Catalog::getAllDashboardsMetadataCopy() const {
  std::vector<DashboardDescriptor> dashboards;
  dashboards.reserve(dashboardDescriptorMap_.size());
  for (auto dashboard_entry : dashboardDescriptorMap_) {
    dashboards.emplace_back(*dashboard_entry.second);
  }
  return dashboards;
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

void Catalog::delDictionary(const ColumnDescriptor& cd) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());
  if (!(cd.columnType.is_string() || cd.columnType.is_string_array())) {
    return;
  }
  if (!(cd.columnType.get_compression() == kENCODING_DICT)) {
    return;
  }
  const auto dictId = cd.columnType.get_comp_param();
  CHECK_GT(dictId, 0);
  // decrement and zero check dict ref count
  const auto td = getMetadataForTable(cd.tableId);
  CHECK(td);
  sqliteConnector_.query_with_text_param(
      "UPDATE mapd_dictionaries SET refcount = refcount - 1 WHERE dictid = ?",
      std::to_string(dictId));
  sqliteConnector_.query_with_text_param(
      "SELECT refcount FROM mapd_dictionaries WHERE dictid = ?", std::to_string(dictId));
  const auto refcount = sqliteConnector_.getData<int>(0, 0);
  VLOG(3) << "Dictionary " << dictId << "from dropped table has reference count "
          << refcount;
  if (refcount > 0) {
    return;
  }
  const DictRef dictRef(currentDB_.dbId, dictId);
  sqliteConnector_.query_with_text_param("DELETE FROM mapd_dictionaries WHERE dictid = ?",
                                         std::to_string(dictId));
  File_Namespace::renameForDelete(g_base_path + "/mapd_data/DB_" +
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
  cat_write_lock write_lock(this);
  // caller must handle sqlite/chunk transaction TOGETHER
  cd.tableId = td.tableId;
  if (cd.columnType.get_compression() == kENCODING_DICT) {
    addDictionary(cd);
  }

  using BindType = SqliteConnector::BindType;
  std::vector<BindType> types(17, BindType::TEXT);
  if (!cd.default_value.has_value()) {
    types[16] = BindType::NULL_TYPE;
  }
  sqliteConnector_.query_with_text_params(
      "INSERT INTO mapd_columns (tableid, columnid, name, coltype, colsubtype, coldim, "
      "colscale, is_notnull, "
      "compression, comp_param, size, chunks, is_systemcol, is_virtualcol, virtual_expr, "
      "is_deletedcol, default_value) "
      "VALUES (?, "
      "(SELECT max(columnid) + 1 FROM mapd_columns WHERE tableid = ?), "
      "?, ?, ?, "
      "?, "
      "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                               std::to_string(cd.isDeletedCol),
                               cd.default_value.value_or("NULL")},
      types);

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

void Catalog::dropColumn(const TableDescriptor& td, const ColumnDescriptor& cd) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());
  // caller must handle sqlite/chunk transaction TOGETHER
  sqliteConnector_.query_with_text_params(
      "DELETE FROM mapd_columns where tableid = ? and columnid = ?",
      std::vector<std::string>{std::to_string(td.tableId), std::to_string(cd.columnId)});

  sqliteConnector_.query_with_text_params(
      "UPDATE mapd_tables SET ncolumns = ncolumns - 1 WHERE tableid = ?",
      std::vector<std::string>{std::to_string(td.tableId)});

  ColumnDescriptorMap::iterator columnDescIt =
      columnDescriptorMap_.find(ColumnKey(cd.tableId, to_upper(cd.columnName)));
  CHECK(columnDescIt != columnDescriptorMap_.end());

  columnDescriptorsForRoll.emplace_back(columnDescIt->second, nullptr);

  columnDescriptorMap_.erase(columnDescIt);
  columnDescriptorMapById_.erase(ColumnIdKey(cd.tableId, cd.columnId));
  --tableDescriptorMapById_[td.tableId]->nColumns;
}

void Catalog::roll(const bool forward) {
  cat_write_lock write_lock(this);
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
        physical_cd_coords.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 0);
        physical_cd_coords.columnType = get_geo_physical_col_type(col_ti, 0);
        columns.push_back(physical_cd_coords);

        // If adding more physical columns - update SQLTypeInfo::get_physical_cols()

        break;
      }
      case kLINESTRING: {
        ColumnDescriptor physical_cd_coords(true);
        physical_cd_coords.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 0);
        physical_cd_coords.columnType = get_geo_physical_col_type(col_ti, 0);
        columns.push_back(physical_cd_coords);

        ColumnDescriptor physical_cd_bounds(true);
        physical_cd_bounds.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 1);
        physical_cd_bounds.columnType = get_geo_physical_col_type(col_ti, 1);
        columns.push_back(physical_cd_bounds);

        // If adding more physical columns - update SQLTypeInfo::get_physical_cols()

        break;
      }
      case kPOLYGON: {
        ColumnDescriptor physical_cd_coords(true);
        physical_cd_coords.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 0);
        physical_cd_coords.columnType = get_geo_physical_col_type(col_ti, 0);
        columns.push_back(physical_cd_coords);

        ColumnDescriptor physical_cd_ring_sizes(true);
        physical_cd_ring_sizes.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 1);
        physical_cd_ring_sizes.columnType = get_geo_physical_col_type(col_ti, 1);
        columns.push_back(physical_cd_ring_sizes);

        ColumnDescriptor physical_cd_bounds(true);
        physical_cd_bounds.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 2);
        physical_cd_bounds.columnType = get_geo_physical_col_type(col_ti, 2);
        columns.push_back(physical_cd_bounds);

        ColumnDescriptor physical_cd_render_group(true);
        physical_cd_render_group.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 3);
        physical_cd_render_group.columnType = get_geo_physical_col_type(col_ti, 3);
        columns.push_back(physical_cd_render_group);

        // If adding more physical columns - update SQLTypeInfo::get_physical_cols()

        break;
      }
      case kMULTIPOLYGON: {
        ColumnDescriptor physical_cd_coords(true);
        physical_cd_coords.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 0);
        physical_cd_coords.columnType = get_geo_physical_col_type(col_ti, 0);
        columns.push_back(physical_cd_coords);

        ColumnDescriptor physical_cd_ring_sizes(true);
        physical_cd_ring_sizes.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 1);
        physical_cd_ring_sizes.columnType = get_geo_physical_col_type(col_ti, 1);
        columns.push_back(physical_cd_ring_sizes);

        ColumnDescriptor physical_cd_poly_rings(true);
        physical_cd_poly_rings.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 2);
        physical_cd_poly_rings.columnType = get_geo_physical_col_type(col_ti, 2);
        columns.push_back(physical_cd_poly_rings);

        ColumnDescriptor physical_cd_bounds(true);
        physical_cd_bounds.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 3);
        physical_cd_bounds.columnType = get_geo_physical_col_type(col_ti, 3);
        columns.push_back(physical_cd_bounds);

        ColumnDescriptor physical_cd_render_group(true);
        physical_cd_render_group.columnName =
            get_geo_physical_col_name(cd.columnName, col_ti, 4);
        physical_cd_render_group.columnType = get_geo_physical_col_type(col_ti, 4);
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
  list<ColumnDescriptor> cds = cols;
  list<DictDescriptor> dds;
  std::set<std::string> toplevel_column_names;
  list<ColumnDescriptor> columns;

  if (!td.storageType.empty()) {
    if (td.persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
      throw std::runtime_error("Only temporary tables can be backed by foreign storage.");
    }
    dataMgr_->getForeignStorageInterface()->prepareTable(getCurrentDB().dbId, td, cds);
  }

  for (auto cd : cds) {
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
  cds.clear();

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
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  if (td.persistenceLevel == Data_Namespace::MemoryLevel::DISK_LEVEL) {
    try {
      sqliteConnector_.query_with_text_params(
          R"(INSERT INTO mapd_tables (name, userid, ncolumns, isview, fragments, frag_type, max_frag_rows, max_chunk_size, frag_page_size, max_rows, partitions, shard_column_id, shard, num_shards, sort_column_id, storage_type, max_rollback_epochs, key_metainfo) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?))",
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
                                   "0",
                                   "-1",
                                   "0",
                                   std::to_string(td.sortedColumnId),
                                   td.storageType,
                                   std::to_string(td.maxRollbackEpochs),
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

        using BindType = SqliteConnector::BindType;
        std::vector<BindType> types(17, BindType::TEXT);
        if (!cd.default_value.has_value()) {
          types[16] = BindType::NULL_TYPE;
        }
        sqliteConnector_.query_with_text_params(
            "INSERT INTO mapd_columns (tableid, columnid, name, coltype, colsubtype, "
            "coldim, colscale, is_notnull, "
            "compression, comp_param, size, chunks, is_systemcol, is_virtualcol, "
            "virtual_expr, is_deletedcol, default_value) "
            "VALUES (?, ?, ?, ?, ?, "
            "?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                                     std::to_string(cd.isDeletedCol),
                                     cd.default_value.value_or("NULL")},
            types);
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
      throw;
    }
  } else {  // Temporary table
    td.tableId = nextTempTableId_++;
    int colId = 1;
    for (auto cd : columns) {
      if (cd.columnType.get_compression() == kENCODING_DICT) {
        const bool is_foreign_col =
            setColumnSharedDictionary(cd, cds, dds, td, shared_dict_defs);

        if (!is_foreign_col) {
          // Create a new temporary dictionary
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
      cd.tableId = td.tableId;
      cd.columnId = colId++;
      cds.push_back(cd);
    }

    if (g_serialize_temp_tables) {
      serializeTableJsonUnlocked(&td, cds);
    }
  }

  try {
    auto cache = dataMgr_->getPersistentStorageMgr()->getDiskCache();
    if (cache) {
      CHECK(!cache->hasCachedMetadataForKeyPrefix({getCurrentDB().dbId, td.tableId}))
          << "Disk cache at " + cache->getCacheDirectory()
          << " contains preexisting data for new table.  Please "
             "delete or clear cache before continuing";
    }

    addTableToMap(&td, cds, dds);
    calciteMgr_->updateMetadata(currentDB_.dbName, td.tableName);
    if (!td.storageType.empty()) {
      dataMgr_->getForeignStorageInterface()->registerTable(this, td, cds);
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    removeTableFromMap(td.tableName, td.tableId, true);
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");

  sqlite_lock.unlock();
  getMetadataForTable(td.tableName,
                      true);  // cause instantiateFragmenter() to be called
}

void Catalog::serializeTableJsonUnlocked(const TableDescriptor* td,
                                         const std::list<ColumnDescriptor>& cds) const {
  // relies on the catalog write lock
  using namespace rapidjson;

  VLOG(1) << "Serializing temporary table " << td->tableName << " to JSON for Calcite.";

  const auto db_name = currentDB_.dbName;
  const auto file_path = table_json_filepath(basePath_, db_name);

  Document d;
  if (boost::filesystem::exists(file_path)) {
    // look for an existing file for this database
    std::ifstream reader(file_path.string());
    CHECK(reader.is_open());
    IStreamWrapper json_read_wrapper(reader);
    d.ParseStream(json_read_wrapper);
  } else {
    d.SetObject();
  }
  CHECK(d.IsObject());
  CHECK(!d.HasMember(StringRef(td->tableName.c_str())));

  Value table(kObjectType);
  table.AddMember(
      "name", Value().SetString(StringRef(td->tableName.c_str())), d.GetAllocator());
  table.AddMember("id", Value().SetInt(td->tableId), d.GetAllocator());
  table.AddMember("columns", Value(kArrayType), d.GetAllocator());

  for (const auto& cd : cds) {
    Value column(kObjectType);
    column.AddMember(
        "name", Value().SetString(StringRef(cd.columnName)), d.GetAllocator());
    column.AddMember("coltype",
                     Value().SetInt(static_cast<int>(cd.columnType.get_type())),
                     d.GetAllocator());
    column.AddMember("colsubtype",
                     Value().SetInt(static_cast<int>(cd.columnType.get_subtype())),
                     d.GetAllocator());
    column.AddMember(
        "coldim", Value().SetInt(cd.columnType.get_dimension()), d.GetAllocator());
    column.AddMember(
        "colscale", Value().SetInt(cd.columnType.get_scale()), d.GetAllocator());
    column.AddMember(
        "is_notnull", Value().SetBool(cd.columnType.get_notnull()), d.GetAllocator());
    column.AddMember("is_systemcol", Value().SetBool(cd.isSystemCol), d.GetAllocator());
    column.AddMember("is_virtualcol", Value().SetBool(cd.isVirtualCol), d.GetAllocator());
    column.AddMember("is_deletedcol", Value().SetBool(cd.isDeletedCol), d.GetAllocator());
    table["columns"].PushBack(column, d.GetAllocator());
  }
  d.AddMember(StringRef(td->tableName.c_str()), table, d.GetAllocator());

  // Overwrite the existing file
  std::ofstream writer(file_path.string(), std::ios::trunc | std::ios::out);
  CHECK(writer.is_open());
  OStreamWrapper json_wrapper(writer);

  Writer<OStreamWrapper> json_writer(json_wrapper);
  d.Accept(json_writer);
  writer.close();
}

void Catalog::dropTableFromJsonUnlocked(const std::string& table_name) const {
  // relies on the catalog write lock
  using namespace rapidjson;

  VLOG(1) << "Dropping temporary table " << table_name << " to JSON for Calcite.";

  const auto db_name = currentDB_.dbName;
  const auto file_path = table_json_filepath(basePath_, db_name);

  CHECK(boost::filesystem::exists(file_path));
  Document d;

  std::ifstream reader(file_path.string());
  CHECK(reader.is_open());
  IStreamWrapper json_read_wrapper(reader);
  d.ParseStream(json_read_wrapper);

  CHECK(d.IsObject());
  auto table_name_ref = StringRef(table_name.c_str());
  CHECK(d.HasMember(table_name_ref));
  CHECK(d.RemoveMember(table_name_ref));

  // Overwrite the existing file
  std::ofstream writer(file_path.string(), std::ios::trunc | std::ios::out);
  CHECK(writer.is_open());
  OStreamWrapper json_wrapper(writer);

  Writer<OStreamWrapper> json_writer(json_wrapper);
  d.Accept(json_writer);
  writer.close();
}

// returns the table epoch or -1 if there is something wrong with the shared epoch
int32_t Catalog::getTableEpoch(const int32_t db_id, const int32_t table_id) const {
  cat_read_lock read_lock(this);
  const auto td = getMetadataForTable(table_id, false);
  if (!td) {
    std::stringstream table_not_found_error_message;
    table_not_found_error_message << "Table (" << db_id << "," << table_id
                                  << ") not found";
    throw std::runtime_error(table_not_found_error_message.str());
  }
  auto epoch = dataMgr_->getTableEpoch(db_id, table_id);
  LOG(INFO) << "Got table epoch for db id: " << db_id << ", table id: " << table_id
            << ", epoch: " << epoch;
  return epoch;
}

void Catalog::setTableEpoch(const int db_id, const int table_id, int new_epoch) {
  cat_read_lock read_lock(this);
  LOG(INFO) << "Set table epoch db:" << db_id << " Table ID  " << table_id
            << " back to new epoch " << new_epoch;
  const auto td = getMetadataForTable(table_id, false);
  if (!td) {
    std::stringstream table_not_found_error_message;
    table_not_found_error_message << "Table (" << db_id << "," << table_id
                                  << ") not found";
    throw std::runtime_error(table_not_found_error_message.str());
  }
  if (td->persistenceLevel != Data_Namespace::MemoryLevel::DISK_LEVEL) {
    std::stringstream is_temp_table_error_message;
    is_temp_table_error_message << "Cannot set epoch on temporary table";
    throw std::runtime_error(is_temp_table_error_message.str());
  }
  File_Namespace::FileMgrParams file_mgr_params;
  file_mgr_params.epoch = new_epoch;
  file_mgr_params.max_rollback_epochs = td->maxRollbackEpochs;

  // Should have table lock from caller so safe to do this after, avoids
  // having to repopulate data on error
  removeChunksUnlocked(table_id);
  dataMgr_->getGlobalFileMgr()->setFileMgrParams(db_id, table_id, file_mgr_params);
}

void Catalog::alterPhysicalTableMetadata(
    const TableDescriptor* td,
    const TableDescriptorUpdateParams& table_update_params) {
  // Only called from parent alterTableParamMetadata, expect already to have catalog and
  // sqlite write locks

  // Sqlite transaction should have already been begun in parent alterTableCatalogMetadata

  TableDescriptor* mutable_td = getMutableMetadataForTableUnlocked(td->tableId);
  CHECK(mutable_td);
  if (td->maxRollbackEpochs != table_update_params.max_rollback_epochs) {
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_tables SET max_rollback_epochs = ? WHERE tableid = ?",
        std::vector<std::string>{std::to_string(table_update_params.max_rollback_epochs),
                                 std::to_string(td->tableId)});
    mutable_td->maxRollbackEpochs = table_update_params.max_rollback_epochs;
  }

  if (td->maxRows != table_update_params.max_rows) {
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_tables SET max_rows = ? WHERE tableid = ?",
        std::vector<std::string>{std::to_string(table_update_params.max_rows),
                                 std::to_string(td->tableId)});
    mutable_td->maxRows = table_update_params.max_rows;
  }
}

void Catalog::alterTableMetadata(const TableDescriptor* td,
                                 const TableDescriptorUpdateParams& table_update_params) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    alterPhysicalTableMetadata(td, table_update_params);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    LOG(FATAL) << "Table '" << td->tableName << "' catalog update failed";
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::setMaxRollbackEpochs(const int32_t table_id,
                                   const int32_t max_rollback_epochs) {
  // Must be called from AlterTableParamStmt or other method that takes executor and
  // TableSchema locks
  cat_write_lock write_lock(this);  // Consider only taking read lock for potentially
                                    // heavy table storage metadata rolloff operations
  if (max_rollback_epochs <= -1) {
    throw std::runtime_error("Cannot set max_rollback_epochs < 0.");
  }
  const auto td = getMetadataForTable(
      table_id, false);  // Deep copy as there will be gap between read and write locks
  CHECK(td);             // Existence should have already been checked in
                         // ParserNode::AlterTableParmStmt
  TableDescriptorUpdateParams table_update_params(td);
  table_update_params.max_rollback_epochs = max_rollback_epochs;
  if (table_update_params == td) {  // Operator is overloaded to test for equality
    LOG(INFO) << "Setting max_rollback_epochs for table " << table_id
              << " to existing value, skipping operation";
    return;
  }
  File_Namespace::FileMgrParams file_mgr_params;
  file_mgr_params.epoch = -1;  // Use existing epoch
  file_mgr_params.max_rollback_epochs = max_rollback_epochs;
  setTableFileMgrParams(table_id, file_mgr_params);
  // Unlock as alterTableCatalogMetadata will take write lock, and Catalog locks are not
  // upgradeable Should be safe as we have schema lock on this table
  /// read_lock.unlock();
  alterTableMetadata(td, table_update_params);
}

void Catalog::setMaxRows(const int32_t table_id, const int64_t max_rows) {
  if (max_rows < 0) {
    throw std::runtime_error("Max rows cannot be a negative number.");
  }
  const auto td = getMetadataForTable(table_id);
  CHECK(td);
  TableDescriptorUpdateParams table_update_params(td);
  table_update_params.max_rows = max_rows;
  if (table_update_params == td) {
    LOG(INFO) << "Max rows value of " << max_rows
              << " is the same as the existing value. Skipping update.";
    return;
  }
  alterTableMetadata(td, table_update_params);
  CHECK(td->fragmenter);
  td->fragmenter->dropFragmentsToSize(max_rows);
}

// For testing purposes only
void Catalog::setUncappedTableEpoch(const std::string& table_name) {
  cat_write_lock write_lock(this);
  auto td_entry = tableDescriptorMap_.find(to_upper(table_name));
  CHECK(td_entry != tableDescriptorMap_.end());
  auto td = td_entry->second;
  TableDescriptorUpdateParams table_update_params(td);
  table_update_params.max_rollback_epochs = -1;
  alterTableMetadata(td, table_update_params);

  File_Namespace::FileMgrParams file_mgr_params;
  file_mgr_params.max_rollback_epochs = -1;
  setTableFileMgrParams(td->tableId, file_mgr_params);
}

void Catalog::setTableFileMgrParams(
    const int table_id,
    const File_Namespace::FileMgrParams& file_mgr_params) {
  // Expects parent to have write lock
  const auto td = getMetadataForTable(table_id, false);
  const auto db_id = this->getDatabaseId();
  if (!td) {
    std::stringstream table_not_found_error_message;
    table_not_found_error_message << "Table (" << db_id << "," << table_id
                                  << ") not found";
    throw std::runtime_error(table_not_found_error_message.str());
  }
  if (td->persistenceLevel != Data_Namespace::MemoryLevel::DISK_LEVEL) {
    std::stringstream is_temp_table_error_message;
    is_temp_table_error_message << "Cannot set storage params on temporary table";
    throw std::runtime_error(is_temp_table_error_message.str());
  }
  removeChunksUnlocked(table_id);
  dataMgr_->getGlobalFileMgr()->setFileMgrParams(db_id, table_id, file_mgr_params);
}

std::vector<TableEpochInfo> Catalog::getTableEpochs(const int32_t db_id,
                                                    const int32_t table_id) const {
  cat_read_lock read_lock(this);
  std::vector<TableEpochInfo> table_epochs;
  auto epoch = dataMgr_->getTableEpoch(db_id, table_id);
  LOG(INFO) << "Got table epoch for db id: " << db_id << ", table id:  " << table_id
            << ", epoch: " << epoch;
  table_epochs.emplace_back(table_id, epoch);
  return table_epochs;
}

void Catalog::setTableEpochs(const int32_t db_id,
                             const std::vector<TableEpochInfo>& table_epochs) const {
  const auto td = getMetadataForTable(table_epochs[0].table_id, false);
  CHECK(td);
  File_Namespace::FileMgrParams file_mgr_params;
  file_mgr_params.max_rollback_epochs = td->maxRollbackEpochs;

  cat_read_lock read_lock(this);
  for (const auto& table_epoch_info : table_epochs) {
    removeChunksUnlocked(table_epoch_info.table_id);
    file_mgr_params.epoch = table_epoch_info.table_epoch;
    dataMgr_->getGlobalFileMgr()->setFileMgrParams(
        db_id, table_epoch_info.table_id, file_mgr_params);
    LOG(INFO) << "Set table epoch for db id: " << db_id
              << ", table id: " << table_epoch_info.table_id
              << ", back to epoch: " << table_epoch_info.table_epoch;
  }
}

namespace {
std::string table_epochs_to_string(const std::vector<TableEpochInfo>& table_epochs) {
  std::string table_epochs_str{"["};
  bool first_entry{true};
  for (const auto& table_epoch : table_epochs) {
    if (first_entry) {
      first_entry = false;
    } else {
      table_epochs_str += ", ";
    }
    table_epochs_str += "(table_id: " + std::to_string(table_epoch.table_id) +
                        ", epoch: " + std::to_string(table_epoch.table_epoch) + ")";
  }
  table_epochs_str += "]";
  return table_epochs_str;
}
}  // namespace

void Catalog::setTableEpochsLogExceptions(
    const int32_t db_id,
    const std::vector<TableEpochInfo>& table_epochs) const {
  try {
    setTableEpochs(db_id, table_epochs);
  } catch (std::exception& e) {
    LOG(ERROR) << "An error occurred when attempting to set table epochs. DB id: "
               << db_id << ", Table epochs: " << table_epochs_to_string(table_epochs)
               << ", Error: " << e.what();
  }
}

const ColumnDescriptor* Catalog::getDeletedColumn(const TableDescriptor* td) const {
  cat_read_lock read_lock(this);
  const auto it = deletedColumnPerTable_.find(td);
  return it != deletedColumnPerTable_.end() ? it->second : nullptr;
}

const bool Catalog::checkMetadataForDeletedRecs(const TableDescriptor* td,
                                                int delete_column_id) const {
  // check if there are rows deleted by examining the deletedColumn metadata
  CHECK(td);
  auto fragmenter = td->fragmenter;
  if (fragmenter) {
    return fragmenter->hasDeletedRows(delete_column_id);
  } else {
    return false;
  }
}

const ColumnDescriptor* Catalog::getDeletedColumnIfRowsDeleted(
    const TableDescriptor* td) const {
  const ColumnDescriptor* cd;
  {
    cat_read_lock read_lock(this);

    const auto it = deletedColumnPerTable_.find(td);
    // if not a table that supports delete return nullptr,  nothing more to do
    if (it == deletedColumnPerTable_.end()) {
      return nullptr;
    }
    cd = it->second;
  }
  // individual tables are still protected by higher level locks
  if (checkMetadataForDeletedRecs(td, cd->columnId)) {
    return cd;
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
                                        Parser::SharedDictionaryDef shared_dict_def,
                                        const bool persist_reference) {
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
  if (persist_reference) {
    cat_sqlite_lock sqlite_lock(getObjForLock());
    sqliteConnector_.query_with_text_params(
        "UPDATE mapd_dictionaries SET refcount = refcount + 1 WHERE dictid = ?",
        {std::to_string(dict_id)});
  }
}

bool Catalog::setColumnSharedDictionary(
    ColumnDescriptor& cd,
    std::list<ColumnDescriptor>& cdd,
    std::list<DictDescriptor>& dds,
    const TableDescriptor td,
    const std::vector<Parser::SharedDictionaryDef>& shared_dict_defs) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());

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

        const int dict_id = colIt->columnType.get_comp_param();
        CHECK_GE(dict_id, 1);
        auto dictIt = std::find_if(
            dds.begin(), dds.end(), [this, dict_id](const DictDescriptor it) {
              return it.dictRef.dbId == this->currentDB_.dbId &&
                     it.dictRef.dictId == dict_id;
            });
        if (dictIt != dds.end()) {
          // There exists dictionary definition of a dictionary column
          CHECK_GE(dictIt->refcount, 1);
          ++dictIt->refcount;
          if (!table_is_temporary(&td)) {
            // Persist reference count
            sqliteConnector_.query_with_text_params(
                "UPDATE mapd_dictionaries SET refcount = refcount + 1 WHERE dictid = ?",
                {std::to_string(dict_id)});
          }
        } else {
          // The dictionary is referencing a column which is referencing a column in
          // diffrent table
          auto root_dict_def = compress_reference_path(shared_dict_def, shared_dict_defs);
          addReferenceToForeignDict(cd, root_dict_def, !table_is_temporary(&td));
        }
      } else {
        const auto& foreign_table_name = shared_dict_def.get_foreign_table();
        const auto foreign_td = getMetadataForTable(foreign_table_name, false);
        if (table_is_temporary(foreign_td)) {
          if (!table_is_temporary(&td)) {
            throw std::runtime_error(
                "Only temporary tables can share dictionaries with other temporary "
                "tables.");
          }
          addReferenceToForeignDict(cd, shared_dict_def, false);
        } else {
          addReferenceToForeignDict(cd, shared_dict_def, !table_is_temporary(&td));
        }
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
    cat_sqlite_lock sqlite_lock(getObjForLock());

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
    folderPath = g_base_path + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) +
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
  createTable(td, cols, shared_dict_defs, true);  // create logical table
}

void Catalog::truncateTable(const TableDescriptor* td) {
  cat_write_lock write_lock(this);
  doTruncateTable(td);
}

void Catalog::doTruncateTable(const TableDescriptor* td) {
  cat_write_lock write_lock(this);

  const int tableId = td->tableId;
  // must destroy fragmenter before deleteChunks is called.
  if (td->fragmenter != nullptr) {
    auto tableDescIt = tableDescriptorMapById_.find(tableId);
    CHECK(tableDescIt != tableDescriptorMapById_.end());
    tableDescIt->second->fragmenter = nullptr;
    CHECK(td->fragmenter == nullptr);
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

void Catalog::removeFragmenterForTable(const int table_id) const {
  cat_write_lock write_lock(this);
  auto td = getMetadataForTable(table_id, false);
  if (td->fragmenter != nullptr) {
    auto tableDescIt = tableDescriptorMapById_.find(table_id);
    CHECK(tableDescIt != tableDescriptorMapById_.end());
    tableDescIt->second->fragmenter = nullptr;
    CHECK(td->fragmenter == nullptr);
  }
}

// used by rollback_table_epoch to clean up in memory artifacts after a rollback
void Catalog::removeChunksUnlocked(const int table_id) const {
  auto td = getMetadataForTable(table_id);
  CHECK(td);

  if (td->fragmenter != nullptr) {
    auto tableDescIt = tableDescriptorMapById_.find(table_id);
    CHECK(tableDescIt != tableDescriptorMapById_.end());
    tableDescIt->second->fragmenter = nullptr;
    CHECK(td->fragmenter == nullptr);
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
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    doDropTable(td);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::doDropTable(const TableDescriptor* td) {
  executeDropTableSqliteQueries(td);
  if (g_serialize_temp_tables && table_is_temporary(td)) {
    dropTableFromJsonUnlocked(td->tableName);
  }
  eraseTablePhysicalData(td);
}

void Catalog::executeDropTableSqliteQueries(const TableDescriptor* td) {
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
}

void Catalog::renamePhysicalTable(const TableDescriptor* td, const string& newTableName) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());

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
  {
    cat_write_lock write_lock(this);
    cat_sqlite_lock sqlite_lock(getObjForLock());
    renamePhysicalTable(td, newTableName);
  }
  {
    DBObject object(newTableName, TableDBObjectType);
    // update table name in direct and effective priv map
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
}

void Catalog::renamePhysicalTable(std::vector<std::pair<std::string, std::string>>& names,
                                  std::vector<int>& tableIds) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());

  // execute the SQL query
  try {
    for (size_t i = 0; i < names.size(); i++) {
      int tableId = tableIds[i];
      std::string& newTableName = names[i].second;

      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_tables SET name = ? WHERE tableid = ?",
          std::vector<std::string>{newTableName, std::to_string(tableId)});
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }

  // reset the table descriptors, give Calcite a kick
  for (size_t i = 0; i < names.size(); i++) {
    std::string& curTableName = names[i].first;
    std::string& newTableName = names[i].second;

    TableDescriptorMap::iterator tableDescIt =
        tableDescriptorMap_.find(to_upper(curTableName));
    CHECK(tableDescIt != tableDescriptorMap_.end());
    calciteMgr_->updateMetadata(currentDB_.dbName, curTableName);

    // Get table descriptor to change it
    TableDescriptor* changeTd = tableDescIt->second;
    changeTd->tableName = newTableName;
    tableDescriptorMap_.erase(tableDescIt);  // erase entry under old name
    tableDescriptorMap_[to_upper(newTableName)] = changeTd;
    calciteMgr_->updateMetadata(currentDB_.dbName, curTableName);
  }
}

// Collect an 'overlay' mapping of the tableNames->tableId
//    to account for possible chained renames
//    (for swap:  a->b, b->c, c->d, d->a)

const TableDescriptor* lookupTableDescriptor(Catalog* cat,
                                             std::map<std::string, int>& cachedTableMap,
                                             std::string& curTableName) {
  auto iter = cachedTableMap.find(curTableName);
  if ((iter != cachedTableMap.end())) {
    // get the cached tableId
    //   and use that to lookup the TableDescriptor
    int tableId = (*iter).second;
    if (tableId == -1)
      return NULL;
    else
      return cat->getMetadataForTable(tableId);
  }

  // else ... lookup in standard location
  return cat->getMetadataForTable(curTableName);
}

void replaceTableName(std::map<std::string, int>& cachedTableMap,
                      std::string& curTableName,
                      std::string& newTableName,
                      int tableId) {
  // mark old/cur name as deleted
  cachedTableMap[curTableName] = -1;

  // insert the 'new' name
  cachedTableMap[newTableName] = tableId;
}

void Catalog::renameTable(std::vector<std::pair<std::string, std::string>>& names) {
  // tableId of all tables being renamed
  //    ... in matching order to 'names'
  std::vector<int> tableIds;

  // (sorted & unique) list of tables ids for locking
  //     (with names index of src in case of error)
  //     <tableId, strIndex>
  //  std::map is by definition/implementation sorted
  //  std::map current usage below tests to avoid over-write
  std::map<int, size_t> uniqueOrderedTableIds;

  // mapping of modified tables names -> tableId
  std::map<std::string, int> cachedTableMap;

  // -------- Setup --------

  // gather tableIds pre-execute; build maps
  for (size_t i = 0; i < names.size(); i++) {
    std::string& curTableName = names[i].first;
    std::string& newTableName = names[i].second;

    // make sure the table being renamed exists,
    //    or will exist when executed in 'name' order
    auto td = lookupTableDescriptor(this, cachedTableMap, curTableName);
    CHECK(td);

    tableIds.push_back(td->tableId);
    if (uniqueOrderedTableIds.find(td->tableId) == uniqueOrderedTableIds.end()) {
      // don't overwrite as it should map to the first names index 'i'
      uniqueOrderedTableIds[td->tableId] = i;
    }
    replaceTableName(cachedTableMap, curTableName, newTableName, td->tableId);
  }

  CHECK_EQ(tableIds.size(), names.size());

  // The outer Stmt created a write lock before calling the catalog rename table
  //   -> TODO: might want to sort out which really should set the lock :
  //      the comment in the outer scope indicates it should be in here
  //      but it's not clear if the access done there *requires* it out there
  //
  // Lock tables pre-execute (may/will be in different order than rename occurs)
  // const auto execute_write_lock = mapd_unique_lock<mapd_shared_mutex>(
  //     *legacylockmgr::LockMgr<mapd_shared_mutex, bool>::getMutex(
  //        legacylockmgr::ExecutorOuterLock, true));

  // acquire the locks for all tables being renamed
  lockmgr::LockedTableDescriptors tableLocks;
  for (auto& idPair : uniqueOrderedTableIds) {
    std::string& tableName = names[idPair.second].first;
    tableLocks.emplace_back(
        std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>>(
            lockmgr::TableSchemaLockContainer<lockmgr::WriteLock>::acquireTableDescriptor(
                *this, tableName, false)));
  }

  // -------- Rename --------

  {
    cat_write_lock write_lock(this);
    cat_sqlite_lock sqlite_lock(getObjForLock());

    sqliteConnector_.query("BEGIN TRANSACTION");

    // collect all (tables + physical tables) into a single list
    std::vector<std::pair<std::string, std::string>> allNames;
    std::vector<int> allTableIds;

    for (size_t i = 0; i < names.size(); i++) {
      int tableId = tableIds[i];
      std::string& curTableName = names[i].first;
      std::string& newTableName = names[i].second;

      allNames.push_back(std::pair<std::string, std::string>(curTableName, newTableName));
      allTableIds.push_back(tableId);
    }
    // rename all tables in one shot
    renamePhysicalTable(allNames, allTableIds);

    sqliteConnector_.query("END TRANSACTION");
    //  cat write/sqlite locks are released when they go out scope
  }
  {
    // now update the SysCatalog
    for (size_t i = 0; i < names.size(); i++) {
      int tableId = tableIds[i];
      std::string& newTableName = names[i].second;
      {
        // update table name in direct and effective priv map
        DBObjectKey key;
        key.dbId = currentDB_.dbId;
        key.objectId = tableId;
        key.permissionType = static_cast<int>(DBObjectType::TableDBObjectType);

        DBObject object(newTableName, TableDBObjectType);
        object.setObjectKey(key);

        auto objdescs = SysCatalog::instance().getMetadataForObject(
            currentDB_.dbId, static_cast<int>(DBObjectType::TableDBObjectType), tableId);
        for (auto obj : objdescs) {
          Grantee* grnt = SysCatalog::instance().getGrantee(obj->roleName);
          if (grnt) {
            grnt->renameDbObject(object);
          }
        }
        SysCatalog::instance().renameObjectsInDescriptorMap(object, *this);
      }
    }
  }

  // -------- Cleanup --------

  // table locks are released when 'tableLocks' goes out of scope
}

void Catalog::renameColumn(const TableDescriptor* td,
                           const ColumnDescriptor* cd,
                           const string& newColumnName) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());
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

int32_t Catalog::createDashboard(DashboardDescriptor& vd) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    // TODO(andrew): this should be an upsert
    sqliteConnector_.query_with_text_params(
        "SELECT id FROM mapd_dashboards WHERE name = ? and userid = ?",
        std::vector<std::string>{vd.dashboardName, std::to_string(vd.userId)});
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_dashboards SET state = ?, image_hash = ?, metadata = ?, "
          "update_time = "
          "datetime('now') where name = ? "
          "and userid = ?",
          std::vector<std::string>{vd.dashboardState,
                                   vd.imageHash,
                                   vd.dashboardMetadata,
                                   vd.dashboardName,
                                   std::to_string(vd.userId)});
    } else {
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_dashboards (name, state, image_hash, metadata, "
          "update_time, "
          "userid) "
          "VALUES "
          "(?,?,?,?, "
          "datetime('now'), ?)",
          std::vector<std::string>{vd.dashboardName,
                                   vd.dashboardState,
                                   vd.imageHash,
                                   vd.dashboardMetadata,
                                   std::to_string(vd.userId)});
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");

  // now get the auto generated dashboardId
  try {
    sqliteConnector_.query_with_text_params(
        "SELECT id, strftime('%Y-%m-%dT%H:%M:%SZ', update_time) FROM mapd_dashboards "
        "WHERE name = ? and userid = ?",
        std::vector<std::string>{vd.dashboardName, std::to_string(vd.userId)});
    vd.dashboardId = sqliteConnector_.getData<int>(0, 0);
    vd.updateTime = sqliteConnector_.getData<std::string>(0, 1);
  } catch (std::exception& e) {
    throw;
  }
  vd.dashboardSystemRoleName = generate_dashboard_system_rolename(
      std::to_string(currentDB_.dbId), std::to_string(vd.dashboardId));
  addFrontendViewToMap(vd);
  sqlite_lock.unlock();
  write_lock.unlock();
  // NOTE(wamsi): Transactionally unsafe
  createOrUpdateDashboardSystemRole(
      vd.dashboardMetadata, vd.userId, vd.dashboardSystemRoleName);
  return vd.dashboardId;
}

void Catalog::replaceDashboard(DashboardDescriptor& vd) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());

  CHECK(sqliteConnector_.getSqlitePtr());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "SELECT id FROM mapd_dashboards WHERE id = ?",
        std::vector<std::string>{std::to_string(vd.dashboardId)});
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_dashboards SET name = ?, state = ?, image_hash = ?, metadata = "
          "?, userid = ?, update_time = datetime('now') where id = ? ",
          std::vector<std::string>{vd.dashboardName,
                                   vd.dashboardState,
                                   vd.imageHash,
                                   vd.dashboardMetadata,
                                   std::to_string(vd.userId),
                                   std::to_string(vd.dashboardId)});
    } else {
      LOG(ERROR) << "Error replacing dashboard id " << vd.dashboardId
                 << " does not exist in db";
      throw runtime_error("Error replacing dashboard id " +
                          std::to_string(vd.dashboardId) + " does not exist in db");
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");

  bool found{false};
  for (auto descp : dashboardDescriptorMap_) {
    auto dash = descp.second.get();
    if (dash->dashboardId == vd.dashboardId) {
      found = true;
      auto viewDescIt = dashboardDescriptorMap_.find(std::to_string(dash->userId) + ":" +
                                                     dash->dashboardName);
      if (viewDescIt ==
          dashboardDescriptorMap_.end()) {  // check to make sure view exists
        LOG(ERROR) << "No metadata for dashboard for user " << dash->userId
                   << " dashboard " << dash->dashboardName << " does not exist in map";
        throw runtime_error("No metadata for dashboard for user " +
                            std::to_string(dash->userId) + " dashboard " +
                            dash->dashboardName + " does not exist in map");
      }
      dashboardDescriptorMap_.erase(viewDescIt);
      break;
    }
  }
  if (!found) {
    LOG(ERROR) << "Error replacing dashboard id " << vd.dashboardId
               << " does not exist in map";
    throw runtime_error("Error replacing dashboard id " + std::to_string(vd.dashboardId) +
                        " does not exist in map");
  }

  // now reload the object
  sqliteConnector_.query_with_text_params(
      "SELECT id, strftime('%Y-%m-%dT%H:%M:%SZ', update_time)  FROM "
      "mapd_dashboards "
      "WHERE id = ?",
      std::vector<std::string>{std::to_string(vd.dashboardId)});
  vd.updateTime = sqliteConnector_.getData<string>(0, 1);
  vd.dashboardSystemRoleName = generate_dashboard_system_rolename(
      std::to_string(currentDB_.dbId), std::to_string(vd.dashboardId));
  addFrontendViewToMapNoLock(vd);
  sqlite_lock.unlock();
  write_lock.unlock();
  // NOTE(wamsi): Transactionally unsafe
  createOrUpdateDashboardSystemRole(
      vd.dashboardMetadata, vd.userId, vd.dashboardSystemRoleName);
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
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    ld.link = calculateSHA1(ld.viewState + ld.viewMetadata + std::to_string(ld.userId))
                  .substr(0, 8);
    sqliteConnector_.query_with_text_params(
        "SELECT linkid FROM mapd_links WHERE link = ? and userid = ?",
        std::vector<std::string>{ld.link, std::to_string(ld.userId)});
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query_with_text_params(
          "UPDATE mapd_links SET update_time = datetime('now') WHERE userid = ? AND "
          "link = ?",
          std::vector<std::string>{std::to_string(ld.userId), ld.link});
    } else {
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_links (userid, link, view_state, view_metadata, "
          "update_time) VALUES (?,?,?,?, datetime('now'))",
          std::vector<std::string>{
              std::to_string(ld.userId), ld.link, ld.viewState, ld.viewMetadata});
    }
    // now get the auto generated dashid
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
  return nullptr;
}

const std::map<int, const ColumnDescriptor*> Catalog::getDictionaryToColumnMapping() {
  cat_read_lock read_lock(this);

  std::map<int, const ColumnDescriptor*> mapping;

  const auto tables = getAllTableMetadata();
  for (const auto td : tables) {
    for (auto& cd : getAllColumnMetadataForTable(td->tableId, false, false, true)) {
      const auto& ti = cd->columnType;
      if (ti.is_string()) {
        if (ti.get_compression() == kENCODING_DICT) {
          // if foreign reference, get referenced tab.col
          const auto dict_id = ti.get_comp_param();

          // ignore temp (negative) dictionaries
          if (dict_id > 0 && mapping.end() == mapping.find(dict_id)) {
            mapping[dict_id] = cd;
          }
        }
      }
    }
  }

  return mapping;
}

bool Catalog::filterTableByTypeAndUser(const TableDescriptor* td,
                                       const UserMetadata& user_metadata,
                                       const GetTablesType get_tables_type) const {
  switch (get_tables_type) {
    case GET_PHYSICAL_TABLES: {
      if (td->isView) {
        return false;
      }
      break;
    }
    case GET_VIEWS: {
      if (!td->isView) {
        return false;
      }
      break;
    }
    default:
      break;
  }
  DBObject dbObject(td->tableName, td->isView ? ViewDBObjectType : TableDBObjectType);
  dbObject.loadKey(*this);
  std::vector<DBObject> privObjects = {dbObject};
  if (!SysCatalog::instance().hasAnyPrivileges(user_metadata, privObjects)) {
    // skip table, as there are no privileges to access it
    return false;
  }
  return true;
}

std::vector<std::string> Catalog::getTableNamesForUser(
    const UserMetadata& user_metadata,
    const GetTablesType get_tables_type) const {
  sys_read_lock syscat_read_lock(&SysCatalog::instance());
  cat_read_lock read_lock(this);

  std::vector<std::string> table_names;
  const auto tables = getAllTableMetadata();
  for (const auto td : tables) {
    if (filterTableByTypeAndUser(td, user_metadata, get_tables_type)) {
      table_names.push_back(td->tableName);
    }
  }
  return table_names;
}

std::vector<TableMetadata> Catalog::getTablesMetadataForUser(
    const UserMetadata& user_metadata,
    const GetTablesType get_tables_type,
    const std::string& filter_table_name) const {
  sys_read_lock syscat_read_lock(&SysCatalog::instance());
  cat_read_lock read_lock(this);

  std::vector<TableMetadata> tables_metadata;
  const auto tables = getAllTableMetadata();
  for (const auto td : tables) {
    if (filterTableByTypeAndUser(td, user_metadata, get_tables_type)) {
      if (!filter_table_name.empty()) {
        if (td->tableName != filter_table_name) {
          continue;
        }
      }
      TableMetadata table_metadata(td);  // Makes a copy, not safe to access raw table
                                         // descriptor outside catalog lock
      tables_metadata.emplace_back(table_metadata);
    }
  }
  return tables_metadata;
}

void Catalog::checkpoint(const int logicalTableId) const {
  const auto td = getMetadataForTable(logicalTableId);
  getDataMgr().checkpoint(getCurrentDB().dbId, td->tableId);
}

void Catalog::checkpointWithAutoRollback(const int logical_table_id) const {
  auto table_epochs = getTableEpochs(getDatabaseId(), logical_table_id);
  try {
    checkpoint(logical_table_id);
  } catch (...) {
    setTableEpochsLogExceptions(getDatabaseId(), table_epochs);
    throw;
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
  const int tableId = td->tableId;
  // must destroy fragmenter before deleteChunks is called.
  if (td->fragmenter != nullptr) {
    auto tableDescIt = tableDescriptorMapById_.find(tableId);
    CHECK(tableDescIt != tableDescriptorMapById_.end());
    {
      INJECT_TIMER(deleting_fragmenter);
      tableDescIt->second->fragmenter = nullptr;
    }
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

// prepare a fresh file reload on next table access
void Catalog::setForReload(const int32_t tableId) {
  cat_read_lock read_lock(this);
  const auto td = getMetadataForTable(tableId);
  const auto tableEpoch = getTableEpoch(currentDB_.dbId, td->tableId);
  setTableEpoch(currentDB_.dbId, td->tableId, tableEpoch);
}

// get a table's data dirs
std::vector<std::string> Catalog::getTableDataDirectories(
    const TableDescriptor* td) const {
  const auto global_file_mgr = getDataMgr().getGlobalFileMgr();
  std::vector<std::string> file_paths;
  const auto file_mgr = dynamic_cast<File_Namespace::FileMgr*>(
      global_file_mgr->getFileMgr(currentDB_.dbId, td->tableId));
  boost::filesystem::path file_path(file_mgr->getFileMgrBasePath());
  file_paths.push_back(file_path.filename().string());
  return file_paths;
}

// get a column's dict dir basename
std::string Catalog::getColumnDictDirectory(const ColumnDescriptor* cd) const {
  if ((cd->columnType.is_string() || cd->columnType.is_string_array()) &&
      cd->columnType.get_compression() == kENCODING_DICT &&
      cd->columnType.get_comp_param() > 0) {
    const auto dictId = cd->columnType.get_comp_param();
    const DictRef dictRef(currentDB_.dbId, dictId);
    const auto dit = dictDescriptorMapByRef_.find(dictRef);
    CHECK(dit != dictDescriptorMapByRef_.end());
    CHECK(dit->second);
    boost::filesystem::path file_path(dit->second->dictFolderPath);
    return file_path.filename().string();
  }
  return std::string();
}

// get a table's dict dirs
std::vector<std::string> Catalog::getTableDictDirectories(
    const TableDescriptor* td) const {
  std::vector<std::string> file_paths;
  for (auto cd : getAllColumnMetadataForTable(td->tableId, false, false, true)) {
    auto file_base = getColumnDictDirectory(cd);
    if (!file_base.empty() &&
        file_paths.end() == std::find(file_paths.begin(), file_paths.end(), file_base)) {
      file_paths.push_back(file_base);
    }
  }
  return file_paths;
}

// returns table schema in a string
// NOTE(sy): Might be able to replace dumpSchema() later with
//           dumpCreateTable() after a deeper review of the TableArchiver code.
std::string Catalog::dumpSchema(const TableDescriptor* td) const {
  cat_read_lock read_lock(this);

  std::ostringstream os;
  os << "CREATE TABLE @T (";
  // gather column defines
  const auto cds = getAllColumnMetadataForTable(td->tableId, false, false, false);
  std::string comma;
  std::vector<std::string> shared_dicts;
  std::map<const std::string, const ColumnDescriptor*> dict_root_cds;
  for (const auto cd : cds) {
    if (!(cd->isSystemCol || cd->isVirtualCol)) {
      const auto& ti = cd->columnType;
      os << comma << cd->columnName;
      // CHAR is perculiar... better dump it as TEXT(32) like \d does
      if (ti.get_type() == SQLTypes::kCHAR) {
        os << " "
           << "TEXT";
      } else if (ti.get_subtype() == SQLTypes::kCHAR) {
        os << " "
           << "TEXT[]";
      } else {
        os << " " << ti.get_type_name();
      }
      os << (ti.get_notnull() ? " NOT NULL" : "");
      if (cd->default_value.has_value()) {
        os << " DEFAULT " << cd->getDefaultValueLiteral();
      }
      if (ti.is_string() || (ti.is_array() && ti.get_subtype() == kTEXT)) {
        auto size = ti.is_array() ? ti.get_logical_size() : ti.get_size();
        if (ti.get_compression() == kENCODING_DICT) {
          // if foreign reference, get referenced tab.col
          const auto dict_id = ti.get_comp_param();
          const DictRef dict_ref(currentDB_.dbId, dict_id);
          const auto dict_it = dictDescriptorMapByRef_.find(dict_ref);
          CHECK(dict_it != dictDescriptorMapByRef_.end());
          const auto dict_name = dict_it->second->dictName;
          // when migrating a table, any foreign dict ref will be dropped
          // and the first cd of a dict will become root of the dict
          if (dict_root_cds.end() == dict_root_cds.find(dict_name)) {
            dict_root_cds[dict_name] = cd;
            os << " ENCODING " << ti.get_compression_name() << "(" << (size * 8) << ")";
          } else {
            const auto dict_root_cd = dict_root_cds[dict_name];
            shared_dicts.push_back("SHARED DICTIONARY (" + cd->columnName +
                                   ") REFERENCES @T(" + dict_root_cd->columnName + ")");
            // "... shouldn't specify an encoding, it borrows from the referenced
            // column"
          }
        } else {
          os << " ENCODING NONE";
        }
      } else if (ti.is_date_in_days() ||
                 (ti.get_size() > 0 && ti.get_size() != ti.get_logical_size())) {
        const auto comp_param = ti.get_comp_param() ? ti.get_comp_param() : 32;
        os << " ENCODING " << ti.get_compression_name() << "(" << comp_param << ")";
      } else if (ti.is_geometry()) {
        if (ti.get_compression() == kENCODING_GEOINT) {
          os << " ENCODING " << ti.get_compression_name() << "(" << ti.get_comp_param()
             << ")";
        } else {
          os << " ENCODING NONE";
        }
      }
      comma = ", ";
    }
  }
  // gather SHARED DICTIONARYs
  if (shared_dicts.size()) {
    os << ", " << boost::algorithm::join(shared_dicts, ", ");
  }
  // gather WITH options ...
  std::vector<std::string> with_options;
  with_options.push_back("FRAGMENT_SIZE=" + std::to_string(td->maxFragRows));
  with_options.push_back("MAX_CHUNK_SIZE=" + std::to_string(td->maxChunkSize));
  with_options.push_back("PAGE_SIZE=" + std::to_string(td->fragPageSize));
  with_options.push_back("MAX_ROWS=" + std::to_string(td->maxRows));
  with_options.emplace_back(td->hasDeletedCol ? "VACUUM='DELAYED'"
                                              : "VACUUM='IMMEDIATE'");
  if (!td->partitions.empty()) {
    with_options.push_back("PARTITIONS='" + td->partitions + "'");
  }
  if (td->sortedColumnId > 0) {
    const auto sort_cd = getMetadataForColumn(td->tableId, td->sortedColumnId);
    CHECK(sort_cd);
    with_options.push_back("SORT_COLUMN='" + sort_cd->columnName + "'");
  }
  if (td->maxRollbackEpochs != DEFAULT_MAX_ROLLBACK_EPOCHS &&
      td->maxRollbackEpochs != -1) {
    with_options.push_back("MAX_ROLLBACK_EPOCHS=" +
                           std::to_string(td->maxRollbackEpochs));
  }
  os << ") WITH (" + boost::algorithm::join(with_options, ", ") + ");";
  return os.str();
}

#include "Parser/ReservedKeywords.h"

//! returns true if the string contains one or more spaces
inline bool contains_spaces(std::string_view str) {
  return std::find_if(str.begin(), str.end(), [](const unsigned char& ch) {
           return std::isspace(ch);
         }) != str.end();
}

//! returns true if the string contains one or more OmniSci SQL reserved characters
inline bool contains_sql_reserved_chars(
    std::string_view str,
    std::string_view chars = "`~!@#$%^&*()-=+[{]}\\|;:'\",<.>/?") {
  return str.find_first_of(chars) != std::string_view::npos;
}

//! returns true if the string equals an OmniSci SQL reserved keyword
inline bool is_reserved_sql_keyword(std::string_view str) {
  return reserved_keywords.find(to_upper(std::string(str))) != reserved_keywords.end();
}

// returns a "CREATE TABLE" statement in a string for "SHOW CREATE TABLE"
std::string Catalog::dumpCreateTable(const TableDescriptor* td,
                                     bool multiline_formatting,
                                     bool dump_defaults) const {
  cat_read_lock read_lock(this);

  std::ostringstream os;

  if (!td->isView) {
    os << "CREATE ";
    if (td->persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL) {
      os << "TEMPORARY ";
    }
    os << "TABLE " + td->tableName + " (";
  } else {
    os << "CREATE VIEW " + td->tableName + " AS " << td->viewSQL;
    return os.str();
  }
  // scan column defines
  std::vector<std::string> additional_info;
  std::set<std::string> shared_dict_column_names;

  gatherAdditionalInfo(additional_info, shared_dict_column_names, td);

  // gather column defines
  const auto cds = getAllColumnMetadataForTable(td->tableId, false, false, false);
  std::map<const std::string, const ColumnDescriptor*> dict_root_cds;
  bool first = true;
  for (const auto cd : cds) {
    if (!(cd->isSystemCol || cd->isVirtualCol)) {
      const auto& ti = cd->columnType;
      if (!first) {
        os << ",";
        if (!multiline_formatting) {
          os << " ";
        }
      } else {
        first = false;
      }
      if (multiline_formatting) {
        os << "\n  ";
      }
      // column name
      os << quoteIfRequired(cd->columnName);
      // CHAR is perculiar... better dump it as TEXT(32) like \d does
      if (ti.get_type() == SQLTypes::kCHAR) {
        os << " "
           << "TEXT";
      } else if (ti.get_subtype() == SQLTypes::kCHAR) {
        os << " "
           << "TEXT[]";
      } else {
        os << " " << ti.get_type_name();
      }
      os << (ti.get_notnull() ? " NOT NULL" : "");
      if (cd->default_value.has_value()) {
        os << " DEFAULT " << cd->getDefaultValueLiteral();
      }
      if (shared_dict_column_names.find(cd->columnName) ==
          shared_dict_column_names.end()) {
        // avoids "Column ... shouldn't specify an encoding, it borrows it
        //    from the referenced column"
        if (ti.is_string() || (ti.is_array() && ti.get_subtype() == kTEXT)) {
          auto size = ti.is_array() ? ti.get_logical_size() : ti.get_size();
          if (ti.get_compression() == kENCODING_DICT) {
            os << " ENCODING " << ti.get_compression_name() << "(" << (size * 8) << ")";
          } else {
            os << " ENCODING NONE";
          }
        } else if (ti.is_date_in_days() ||
                   (ti.get_size() > 0 && ti.get_size() != ti.get_logical_size())) {
          const auto comp_param = ti.get_comp_param() ? ti.get_comp_param() : 32;
          os << " ENCODING " << ti.get_compression_name() << "(" << comp_param << ")";
        } else if (ti.is_geometry()) {
          if (ti.get_compression() == kENCODING_GEOINT) {
            os << " ENCODING " << ti.get_compression_name() << "(" << ti.get_comp_param()
               << ")";
          } else {
            os << " ENCODING NONE";
          }
        }
      }
    }
  }
  // gather SHARED DICTIONARYs
  if (additional_info.size()) {
    std::string comma;
    if (!multiline_formatting) {
      comma = ", ";
    } else {
      comma = ",\n  ";
    }
    os << comma;
    os << boost::algorithm::join(additional_info, comma);
  }
  os << ")";

  std::vector<std::string> with_options;
  if (dump_defaults || td->maxFragRows != DEFAULT_FRAGMENT_ROWS) {
    with_options.push_back("FRAGMENT_SIZE=" + std::to_string(td->maxFragRows));
  }
  if (dump_defaults || td->maxChunkSize != DEFAULT_MAX_CHUNK_SIZE) {
    with_options.push_back("MAX_CHUNK_SIZE=" + std::to_string(td->maxChunkSize));
  }
  if (dump_defaults || td->fragPageSize != DEFAULT_PAGE_SIZE) {
    with_options.push_back("PAGE_SIZE=" + std::to_string(td->fragPageSize));
  }
  if (dump_defaults || td->maxRows != DEFAULT_MAX_ROWS) {
    with_options.push_back("MAX_ROWS=" + std::to_string(td->maxRows));
  }
  if ((dump_defaults || td->maxRollbackEpochs != DEFAULT_MAX_ROLLBACK_EPOCHS) &&
      td->maxRollbackEpochs != -1) {
    with_options.push_back("MAX_ROLLBACK_EPOCHS=" +
                           std::to_string(td->maxRollbackEpochs));
  }
  if (dump_defaults || !td->hasDeletedCol) {
    with_options.push_back(td->hasDeletedCol ? "VACUUM='DELAYED'" : "VACUUM='IMMEDIATE'");
  }
  if (!td->partitions.empty()) {
    with_options.push_back("PARTITIONS='" + td->partitions + "'");
  }
  if (td->sortedColumnId > 0) {
    const auto sort_cd = getMetadataForColumn(td->tableId, td->sortedColumnId);
    CHECK(sort_cd);
    with_options.push_back("SORT_COLUMN='" + sort_cd->columnName + "'");
  }

  if (!with_options.empty()) {
    if (!multiline_formatting) {
      os << " ";
    } else {
      os << "\n";
    }
    os << "WITH (" + boost::algorithm::join(with_options, ", ") + ")";
  }
  os << ";";
  return os.str();
}

bool Catalog::validateNonExistentTableOrView(const std::string& name,
                                             const bool if_not_exists) {
  if (getMetadataForTable(name, false)) {
    if (if_not_exists) {
      return false;
    }
    throw std::runtime_error("Table or View with name \"" + name + "\" already exists.");
  }
  return true;
}

std::string Catalog::quoteIfRequired(const std::string& column_name) const {
  if (is_reserved_sql_keyword(column_name) || contains_spaces(column_name) ||
      contains_sql_reserved_chars(column_name)) {
    return get_quoted_string(column_name, '"', '"');
  } else {
    return column_name;
  }
}

// this will gather information that represents the shared dictionary columns
// as they are on the table NOW not at original creation
void Catalog::gatherAdditionalInfo(std::vector<std::string>& additional_info,
                                   std::set<std::string>& shared_dict_column_names,
                                   const TableDescriptor* td) const {
  const auto cds = getAllColumnMetadataForTable(td->tableId, false, false, false);
  for (const auto cd : cds) {
    if (!(cd->isSystemCol || cd->isVirtualCol)) {
      const SQLTypeInfo& ti = cd->columnType;
      if (ti.get_compression() != kENCODING_DICT) {
        continue;
      }
      auto dictId = ti.get_comp_param();

      // now we need to check how many other users have this dictionary

      DictRef dict_ref(currentDB_.dbId, dictId);
      const auto dictIt = dictDescriptorMapByRef_.find(dict_ref);
      if (dictIt == dictDescriptorMapByRef_.end()) {
        LOG(ERROR) << "missing dictionary " << dictId << " for table " << td->tableName;
        continue;
      }

      const auto& dd = dictIt->second;
      if (dd->refcount > 1) {
        auto lowest_table = td->tableId;
        auto lowest_column = cd->columnId;
        std::string lowest_column_name;
        // we have multiple tables using this dictionary
        // find the other occurances and keep the "lowest"
        for (auto const& [key, val] : columnDescriptorMap_) {
          if (val->columnType.get_compression() == kENCODING_DICT &&
              val->columnType.get_comp_param() == dictId &&
              !(val->tableId == td->tableId && val->columnId == cd->columnId)) {
            if (val->tableId < lowest_table) {
              lowest_table = val->tableId;
              lowest_column = val->columnId;
              lowest_column_name = val->columnName;
            }
            if (val->columnId < lowest_column) {
              lowest_column = val->columnId;
              lowest_column_name = val->columnName;
            }
          }
        }
        if (lowest_table != td->tableId || lowest_column != cd->columnId) {
          // we are referencing a different tables dictionary
          auto lowest_td = tableDescriptorMapById_.find(lowest_table)->second;
          CHECK(lowest_td);
          std::string txt = "SHARED DICTIONARY (" + quoteIfRequired(cd->columnName) +
                            ") REFERENCES " + lowest_td->tableName + "(" +
                            quoteIfRequired(lowest_column_name) + ")";

          additional_info.emplace_back(txt);
          shared_dict_column_names.insert(cd->columnName);
        }
      }
    }
  }
}

int32_t Catalog::createCustomExpression(
    std::unique_ptr<CustomExpression> custom_expression) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  int32_t custom_expression_id{-1};
  try {
    auto data_source_type_str =
        CustomExpression::dataSourceTypeToString(custom_expression->data_source_type);
    auto data_source_id_str = std::to_string(custom_expression->data_source_id);
    std::string custom_expr_select_query{
        "SELECT id FROM omnisci_custom_expressions WHERE name = ? and data_source_type = "
        "? and data_source_id = ? and is_deleted = ?"};
    std::vector<std::string> custom_expr_select_params{custom_expression->name,
                                                       data_source_type_str,
                                                       data_source_id_str,
                                                       std::to_string(false)};
    sqliteConnector_.query_with_text_params(custom_expr_select_query,
                                            custom_expr_select_params);
    if (sqliteConnector_.getNumRows() > 0) {
      throw std::runtime_error{
          "A custom expression with the given "
          "name and data source already exists."};
    }
    sqliteConnector_.query_with_text_params(
        "INSERT INTO omnisci_custom_expressions(name, expression_json, "
        "data_source_type, data_source_id, is_deleted) VALUES (?,?,?,?,?)",
        std::vector<std::string>{custom_expression->name,
                                 custom_expression->expression_json,
                                 data_source_type_str,
                                 data_source_id_str,
                                 std::to_string(false)});
    sqliteConnector_.query_with_text_params(custom_expr_select_query,
                                            custom_expr_select_params);
    CHECK_EQ(sqliteConnector_.getNumRows(), static_cast<size_t>(1));
    custom_expression->id = sqliteConnector_.getData<int32_t>(0, 0);
    custom_expression_id = custom_expression->id;
    CHECK(custom_expr_map_by_id_.find(custom_expression->id) ==
          custom_expr_map_by_id_.end());
    custom_expr_map_by_id_[custom_expression->id] = std::move(custom_expression);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  CHECK_GT(custom_expression_id, 0);
  return custom_expression_id;
}

const CustomExpression* Catalog::getCustomExpression(int32_t custom_expression_id) const {
  cat_read_lock read_lock(this);
  auto it = custom_expr_map_by_id_.find(custom_expression_id);
  if (it != custom_expr_map_by_id_.end()) {
    return it->second.get();
  }
  return nullptr;
}

const std::unique_ptr<const CustomExpression> Catalog::getCustomExpressionFromStorage(
    int32_t custom_expression_id) {
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query_with_text_params(
      "SELECT id, name, expression_json, data_source_type, data_source_id, "
      "is_deleted FROM omnisci_custom_expressions WHERE id = ?",
      std::vector<std::string>{to_string(custom_expression_id)});
  if (sqliteConnector_.getNumRows() > 0) {
    CHECK_EQ(sqliteConnector_.getNumRows(), static_cast<size_t>(1));
    return getCustomExpressionFromConnector(0);
  }
  return nullptr;
}

std::vector<const CustomExpression*> Catalog::getCustomExpressionsForUser(
    const UserMetadata& user) const {
  std::vector<const CustomExpression*> all_custom_expressions;
  {
    // Get custom expression pointers separately in order to avoid holding the catalog
    // read lock while checking privileges (which may cause a deadlock).
    cat_read_lock read_lock(this);
    for (const auto& [id, custom_expression] : custom_expr_map_by_id_) {
      all_custom_expressions.emplace_back(custom_expression.get());
    }
  }

  std::vector<const CustomExpression*> filtered_custom_expressions;
  for (const auto custom_expression : all_custom_expressions) {
    CHECK(custom_expression->data_source_type == DataSourceType::TABLE);
    DBObject db_object{custom_expression->data_source_id, TableDBObjectType};
    db_object.loadKey(*this);
    db_object.setPrivileges(AccessPrivileges::SELECT_FROM_TABLE);
    if (SysCatalog::instance().checkPrivileges(user, {db_object})) {
      filtered_custom_expressions.emplace_back(custom_expression);
    }
  }
  return filtered_custom_expressions;
}

void Catalog::updateCustomExpression(int32_t custom_expression_id,
                                     const std::string& expression_json) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());
  auto it = custom_expr_map_by_id_.find(custom_expression_id);
  if (it == custom_expr_map_by_id_.end() || it->second->is_deleted) {
    throw std::runtime_error{"Custom expression with id \"" +
                             std::to_string(custom_expression_id) + "\" does not exist."};
  }
  auto old_expression_json = it->second->expression_json;
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "SELECT id FROM omnisci_custom_expressions WHERE id = ?",
        std::vector<std::string>{std::to_string(custom_expression_id)});
    CHECK_EQ(sqliteConnector_.getNumRows(), static_cast<size_t>(1));
    sqliteConnector_.query_with_text_params(
        "UPDATE omnisci_custom_expressions SET expression_json = ? WHERE id = ?",
        std::vector<std::string>{expression_json, std::to_string(custom_expression_id)});
    it->second->expression_json = expression_json;
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    it->second->expression_json = old_expression_json;
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::deleteCustomExpressions(const std::vector<int32_t>& custom_expression_ids,
                                      bool do_soft_delete) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());

  std::vector<int32_t> invalid_ids;
  for (const auto id : custom_expression_ids) {
    if (custom_expr_map_by_id_.find(id) == custom_expr_map_by_id_.end()) {
      invalid_ids.emplace_back(id);
    }
  }
  if (!invalid_ids.empty()) {
    throw std::runtime_error{"Custom expressions with ids: " + join(invalid_ids, ",") +
                             " do not exist."};
  }
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    for (const auto id : custom_expression_ids) {
      sqliteConnector_.query_with_text_params(
          "SELECT id FROM omnisci_custom_expressions WHERE id = ?",
          std::vector<std::string>{std::to_string(id)});
      CHECK_EQ(sqliteConnector_.getNumRows(), static_cast<size_t>(1));
      if (do_soft_delete) {
        sqliteConnector_.query_with_text_params(
            "UPDATE omnisci_custom_expressions SET is_deleted = ? WHERE id = ?",
            std::vector<std::string>{std::to_string(true), std::to_string(id)});
      } else {
        sqliteConnector_.query_with_text_params(
            "DELETE FROM omnisci_custom_expressions WHERE id = ?",
            std::vector<std::string>{std::to_string(id)});
      }
    }

    for (const auto id : custom_expression_ids) {
      if (do_soft_delete) {
        auto it = custom_expr_map_by_id_.find(id);
        CHECK(it != custom_expr_map_by_id_.end());
        it->second->is_deleted = true;
      } else {
        custom_expr_map_by_id_.erase(id);
      }
    }
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
}

namespace {
int32_t validate_and_get_user_id(const std::string& user_name) {
  UserMetadata user;
  if (!SysCatalog::instance().getMetadataForUser(user_name, user)) {
    throw std::runtime_error{"User with username \"" + user_name + "\" does not exist."};
  }
  return user.userId;
}

std::string convert_object_owners_map_to_string(
    int32_t db_id,
    int32_t new_owner_id,
    const std::map<int32_t, std::vector<DBObject>>& old_owner_db_objects) {
  std::stringstream result;
  for (const auto& [old_owner_id, db_objects] : old_owner_db_objects) {
    result << "db_id: " << db_id << ", new_owner_user_id: " << new_owner_id
           << ", old_owner_user_id: " << old_owner_id << ", db_objects: [";
    bool first_object{true};
    for (const auto& db_object : db_objects) {
      if (first_object) {
        first_object = false;
      } else {
        result << ", ";
      }
      result << "\"object_id: " << db_object.getObjectKey().objectId
             << ", object_type: " << DBObjectTypeToString(db_object.getType()) << "\"";
    }
    result << "]\n";
  }
  return result.str();
}

void add_db_object(const std::string& object_name,
                   DBObjectType object_type,
                   int32_t user_id,
                   const AccessPrivileges& privileges,
                   std::map<int32_t, std::vector<DBObject>>& db_objects) {
  DBObject db_object{object_name, object_type};
  db_object.setPrivileges(privileges);
  db_objects[user_id].emplace_back(db_object);
}
}  // namespace

void Catalog::reassignOwners(const std::set<std::string>& old_owners,
                             const std::string& new_owner) {
  CHECK(!old_owners.empty());
  int32_t new_owner_id = validate_and_get_user_id(new_owner);
  std::set<int32_t> old_owner_ids;
  for (const auto& old_owner : old_owners) {
    auto old_owner_id = validate_and_get_user_id(old_owner);
    if (old_owner_id != new_owner_id) {
      old_owner_ids.emplace(old_owner_id);
    }
  }

  // An empty set after the above loop implies reassignment to the same user (i.e. all
  // users in the old_owners set is the same as new_owner). Do nothing in this case.
  if (old_owner_ids.empty()) {
    return;
  }

  std::map<int32_t, std::vector<DBObject>> old_owner_db_objects;
  {
    cat_write_lock write_lock(this);
    cat_sqlite_lock sqlite_lock(getObjForLock());
    sqliteConnector_.query("BEGIN TRANSACTION");
    try {
      for (const auto old_user_id : old_owner_ids) {
        sqliteConnector_.query_with_text_params(
            "UPDATE mapd_tables SET userid = ? WHERE userid = ?",
            std::vector<std::string>{std::to_string(new_owner_id),
                                     std::to_string(old_user_id)});

        sqliteConnector_.query_with_text_params(
            "UPDATE mapd_dashboards SET userid = ? WHERE userid = ?",
            std::vector<std::string>{std::to_string(new_owner_id),
                                     std::to_string(old_user_id)});

        if (g_enable_fsi) {
          sqliteConnector_.query_with_text_params(
              "UPDATE omnisci_foreign_servers SET owner_user_id = ? "
              "WHERE owner_user_id = ?",
              std::vector<std::string>{std::to_string(new_owner_id),
                                       std::to_string(old_user_id)});
        }
      }

      for (const auto& [table_name, td] : tableDescriptorMap_) {
        if (shared::contains(old_owner_ids, td->userId)) {
          if (td->isView) {
            add_db_object(td->tableName,
                          DBObjectType::ViewDBObjectType,
                          td->userId,
                          AccessPrivileges::ALL_VIEW,
                          old_owner_db_objects);
          } else {
            add_db_object(td->tableName,
                          DBObjectType::TableDBObjectType,
                          td->userId,
                          AccessPrivileges::ALL_TABLE,
                          old_owner_db_objects);
          }
          td->userId = new_owner_id;
        }
      }

      DashboardDescriptorMap new_owner_dashboard_map;
      for (auto it = dashboardDescriptorMap_.begin();
           it != dashboardDescriptorMap_.end();) {
        if (auto dashboard = it->second;
            shared::contains(old_owner_ids, dashboard->userId)) {
          DBObject db_object{dashboard->dashboardId, DBObjectType::DashboardDBObjectType};
          db_object.setPrivileges(AccessPrivileges::ALL_DASHBOARD);
          old_owner_db_objects[dashboard->userId].emplace_back(db_object);

          // Dashboards in the dashboardDescriptorMap_ use keys with the format
          // "{user id}:{dashboard name}". Ensure that map entries are replaced
          // with the new owner's user id.
          std::string old_key{std::to_string(dashboard->userId) + ":" +
                              dashboard->dashboardName};
          CHECK_EQ(it->first, old_key);
          std::string new_key{std::to_string(new_owner_id) + ":" +
                              dashboard->dashboardName};
          CHECK(dashboardDescriptorMap_.find(new_key) == dashboardDescriptorMap_.end());
          new_owner_dashboard_map[new_key] = dashboard;
          dashboard->userId = new_owner_id;
          it = dashboardDescriptorMap_.erase(it);
        } else {
          it++;
        }
      }
      dashboardDescriptorMap_.merge(new_owner_dashboard_map);

      // Ensure new owner is set in the DB objects.
      for (auto& [old_owner_id, db_objects] : old_owner_db_objects) {
        for (auto& db_object : db_objects) {
          db_object.loadKey(*this);
          CHECK_EQ(db_object.getOwner(), new_owner_id);
          const auto& object_key = db_object.getObjectKey();
          CHECK_EQ(object_key.dbId, getDatabaseId());
          CHECK_NE(object_key.objectId, -1);
        }
      }
    } catch (std::exception& e) {
      sqliteConnector_.query("ROLLBACK TRANSACTION");
      restoreOldOwnersInMemory(old_owner_db_objects, new_owner_id);
      throw;
    }
    sqliteConnector_.query("END TRANSACTION");
  }

  try {
    SysCatalog::instance().reassignObjectOwners(
        old_owner_db_objects, new_owner_id, *this);
  } catch (std::exception& e) {
    restoreOldOwners(old_owner_db_objects, new_owner_id);
    throw;
  }
}

void Catalog::restoreOldOwners(
    const std::map<int32_t, std::vector<DBObject>>& old_owner_db_objects,
    int32_t new_owner_id) {
  cat_write_lock write_lock(this);
  cat_sqlite_lock sqlite_lock(getObjForLock());
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    for (const auto& [old_owner_id, db_objects] : old_owner_db_objects) {
      for (const auto& db_object : db_objects) {
        auto object_id = db_object.getObjectKey().objectId;
        CHECK_GT(object_id, 0);
        std::vector<std::string> query_params{std::to_string(old_owner_id),
                                              std::to_string(new_owner_id),
                                              std::to_string(object_id)};
        auto object_type = db_object.getType();
        if (object_type == DBObjectType::TableDBObjectType ||
            object_type == DBObjectType::ViewDBObjectType) {
          sqliteConnector_.query_with_text_params(
              "UPDATE mapd_tables SET userid = ? WHERE userid = ? AND tableid = ?",
              query_params);
        } else if (object_type == DBObjectType::DashboardDBObjectType) {
          sqliteConnector_.query_with_text_params(
              "UPDATE mapd_dashboards SET userid = ? WHERE userid = ? AND id = ?",
              query_params);
        } else {
          UNREACHABLE() << "Unexpected DB object type: " << static_cast<int>(object_type);
        }
      }
    }
    restoreOldOwnersInMemory(old_owner_db_objects, new_owner_id);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    LOG(FATAL)
        << "Unable to restore database objects ownership after an error occurred. "
           "Database object ownership information may be in an inconsistent state. " +
               convert_object_owners_map_to_string(
                   getDatabaseId(), new_owner_id, old_owner_db_objects);
  }
  sqliteConnector_.query("END TRANSACTION");
}

void Catalog::restoreOldOwnersInMemory(
    const std::map<int32_t, std::vector<DBObject>>& old_owner_db_objects,
    int32_t new_owner_id) {
  for (const auto& [old_owner_id, db_objects] : old_owner_db_objects) {
    for (const auto& db_object : db_objects) {
      auto object_id = db_object.getObjectKey().objectId;
      auto object_type = db_object.getType();
      if (object_type == DBObjectType::TableDBObjectType ||
          object_type == DBObjectType::ViewDBObjectType) {
        auto it = tableDescriptorMapById_.find(object_id);
        CHECK(it != tableDescriptorMapById_.end());
        CHECK(it->second);
        it->second->userId = old_owner_id;
      } else if (object_type == DBObjectType::DashboardDBObjectType) {
        auto it = dashboardDescriptorMap_.find(std::to_string(new_owner_id) + ":" +
                                               db_object.getName());
        CHECK(it != dashboardDescriptorMap_.end());
        CHECK(it->second);
        it->second->userId = old_owner_id;
        dashboardDescriptorMap_[std::to_string(old_owner_id) + ":" +
                                db_object.getName()] = it->second;
        dashboardDescriptorMap_.erase(it);
      } else {
        UNREACHABLE() << "Unexpected DB object type: " << static_cast<int>(object_type);
      }
    }
  }
}

}  // namespace Catalog_Namespace
