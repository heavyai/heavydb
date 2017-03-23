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
#include "../Shared/StringTransform.h"

using std::runtime_error;
using std::string;
using std::map;
using std::list;
using std::pair;
using std::vector;
using Chunk_NS::Chunk;
using Fragmenter_Namespace::InsertOrderFragmenter;

namespace Catalog_Namespace {

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

void SysCatalog::createUser(const string& name, const string& passwd, bool issuper) {
  UserMetadata user;
  if (getMetadataForUser(name, user))
    throw runtime_error("User " + name + " already exists.");
  sqliteConnector_.query_with_text_params("INSERT INTO mapd_users (name, passwd, issuper) VALUES (?, ?, ?)",
                                          std::vector<std::string>{name, passwd, std::to_string(issuper)});
}

void SysCatalog::dropUser(const string& name) {
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
      "max_rows bigint, partitions "
      "text)");
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
      "CREATE TABLE mapd_dictionaries (dictid integer primary key, name text unique, nbits int, is_shared boolean)");
}

void SysCatalog::dropDatabase(const int32_t dbid, const std::string& name) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query_with_text_param("DELETE FROM mapd_databases WHERE dbid = ?", std::to_string(dbid));
  boost::filesystem::remove(basePath_ + "/mapd_catalogs/" + name);
  ChunkKey chunkKeyPrefix = {dbid};
#ifdef HAVE_CALCITE
  calciteMgr_->updateMetadata(name, "");
#endif  // HAVE_CALCITE
  dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix);
  /* don't need to checkpoint as database is being dropped */
  // dataMgr_->checkpoint();
}

bool SysCatalog::checkPasswordForUser(const std::string& passwd, UserMetadata& user) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  if (ldap_server_->inUse()) {
    return ldap_server_->authenticate_user(user.userName, passwd);
  } else {
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

Catalog::Catalog(const string& basePath,
                 const string& dbname,
                 std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                 LdapMetadata ldapMetadata,
                 bool is_initdb
#ifdef HAVE_CALCITE
                 ,
                 std::shared_ptr<Calcite> calcite
#endif  // HAVE_CALCITE
                 )
    : basePath_(basePath),
      sqliteConnector_(dbname, basePath + "/mapd_catalogs/"),
      dataMgr_(dataMgr)
#ifdef HAVE_CALCITE
      ,
      calciteMgr_(calcite)
#endif  // HAVE_CALCITE
{
  ldap_server_.reset(new LdapServer(ldapMetadata));
  if (!is_initdb)
    buildMaps();
}

Catalog::Catalog(const string& basePath,
                 const DBMetadata& curDB,
                 std::shared_ptr<Data_Namespace::DataMgr> dataMgr,
                 LdapMetadata ldapMetadata
#ifdef HAVE_CALCITE
                 ,
                 std::shared_ptr<Calcite> calcite
#endif  // HAVE_CALCITE
                 )
    : basePath_(basePath),
      sqliteConnector_(curDB.dbName, basePath + "/mapd_catalogs/"),
      currentDB_(curDB),
      dataMgr_(dataMgr)
#ifdef HAVE_CALCITE
      ,
      calciteMgr_(calcite)
#endif  // HAVE_CALCITE
{
  ldap_server_.reset(new LdapServer(ldapMetadata));
  buildMaps();
}

Catalog::Catalog(const string& basePath,
                 const DBMetadata& curDB,
                 std::shared_ptr<Data_Namespace::DataMgr> dataMgr
#ifdef HAVE_CALCITE
                 ,
                 const std::vector<LeafHostInfo>& string_dict_hosts,
                 std::shared_ptr<Calcite> calcite
#endif  // HAVE_CALCITE
                 )
    : basePath_(basePath),
      sqliteConnector_(curDB.dbName, basePath + "/mapd_catalogs/"),
      currentDB_(curDB),
      dataMgr_(dataMgr)
#ifdef HAVE_CALCITE
      ,
      string_dict_hosts_(string_dict_hosts),
      calciteMgr_(calcite)
#endif  // HAVE_CALCITE
{
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

void Catalog::CheckAndExecuteMigrations() {
  updateTableDescriptorSchema();
  updateFrontendViewAndLinkUsers();
  updateFrontendViewSchema();
  updateLinkSchema();
  updateDictionaryNames();
}

void Catalog::buildMaps() {
  string dictQuery("SELECT dictid, name, nbits, is_shared from mapd_dictionaries");
  sqliteConnector_.query(dictQuery);
  size_t numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    int dictId = sqliteConnector_.getData<int>(r, 0);
    std::string dictName = sqliteConnector_.getData<string>(r, 1);
    int dictNBits = sqliteConnector_.getData<int>(r, 2);
    bool is_shared = sqliteConnector_.getData<bool>(r, 3);
    std::string fname =
        basePath_ + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) + "_DICT_" + std::to_string(dictId);
    DictDescriptor* dd = new DictDescriptor(dictId, dictName, dictNBits, is_shared, fname);
    dictDescriptorMapById_[dd->dictId].reset(dd);
  }

  CheckAndExecuteMigrations();

  string tableQuery(
      "SELECT tableid, name, ncolumns, isview, fragments, frag_type, max_frag_rows, max_chunk_size, frag_page_size, "
      "max_rows, "
      "partitions from mapd_tables");
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
  for (auto dd : dicts) {
    DictDescriptor* new_dd = new DictDescriptor(dd);
    dictDescriptorMapById_[dd.dictId].reset(new_dd);
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
  delete td;

  // delete all column descriptors for the table
  for (int i = 1; i <= ncolumns; i++) {
    ColumnIdKey cidKey(tableId, i);
    ColumnDescriptorMapById::iterator colDescIt = columnDescriptorMapById_.find(cidKey);
    ColumnDescriptor* cd = colDescIt->second;
    columnDescriptorMapById_.erase(colDescIt);
    ColumnKey cnameKey(tableId, to_upper(cd->columnName));
    columnDescriptorMap_.erase(cnameKey);
    if (cd->columnType.get_compression() == kENCODING_DICT) {
      DictDescriptorMapById::iterator dictIt = dictDescriptorMapById_.find(cd->columnType.get_comp_param());
      const auto& dd = dictIt->second;
      boost::filesystem::remove_all(dd->dictFolderPath);
      dictDescriptorMapById_.erase(dictIt);
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
  // instatiion table fragmenter upon first use
  // assume only insert order fragmenter is supported
  assert(td->fragType == Fragmenter_Namespace::FragmenterType::INSERT_ORDER);
  vector<Chunk> chunkVec;
  list<const ColumnDescriptor*> columnDescs;
  getAllColumnMetadataForTable(td, columnDescs, true, false);
  Chunk::translateColumnDescriptorsToChunkVec(columnDescs, chunkVec);
  ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId};
  td->fragmenter = new InsertOrderFragmenter(
      chunkKeyPrefix, chunkVec, dataMgr_.get(), td->maxFragRows, td->maxChunkSize, td->fragPageSize, td->maxRows);
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
      if (string_dict_hosts_.empty()) {
        dd->stringDict = std::make_shared<StringDictionary>(dd->dictFolderPath);
      } else {
        dd->stringDict = std::make_shared<StringDictionary>(string_dict_hosts_.front(), dd->dictId);
      }
    }
  }
  return dd.get();
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

void Catalog::createTable(TableDescriptor& td, const list<ColumnDescriptor>& cols) {
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

  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_params(
        "INSERT INTO mapd_tables (name, ncolumns, isview, fragments, frag_type, max_frag_rows, max_chunk_size, "
        "frag_page_size, "
        "max_rows, partitions) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        std::vector<std::string>{td.tableName,
                                 std::to_string(columns.size()),
                                 std::to_string(td.isView),
                                 "",
                                 std::to_string(td.fragType),
                                 std::to_string(td.maxFragRows),
                                 std::to_string(td.maxChunkSize),
                                 std::to_string(td.fragPageSize),
                                 std::to_string(td.maxRows),
                                 td.partitions});
    // now get the auto generated tableid
    sqliteConnector_.query_with_text_param("SELECT tableid FROM mapd_tables WHERE name = ?", td.tableName);
    td.tableId = sqliteConnector_.getData<int>(0, 0);
    int colId = 1;
    for (auto cd : columns) {
      if (cd.columnType.get_compression() == kENCODING_DICT) {
        // std::string dictName = td.tableName + "_" + cd.columnName + "_dict";
        std::string dictName = "Initial_key";
        sqliteConnector_.query_with_text_params(
            "INSERT INTO mapd_dictionaries (name, nbits, is_shared) VALUES (?, ?, ?)",
            std::vector<std::string>{dictName, std::to_string(cd.columnType.get_comp_param()), "0"});
        sqliteConnector_.query_with_text_param("SELECT dictid FROM mapd_dictionaries WHERE name = ?", dictName);
        int dictId = sqliteConnector_.getData<int>(0, 0);
        dictName = td.tableName + "_" + cd.columnName + "_dict" + std::to_string(dictId);
        sqliteConnector_.query_with_text_param("UPDATE mapd_dictionaries SET name = ? WHERE name = 'Initial_key'",
                                               dictName);
        // std::string folderPath = basePath_ + "/mapd_data/" + currentDB_.dbName + "_" + dictName;
        std::string folderPath =
            basePath_ + "/mapd_data/DB_" + std::to_string(currentDB_.dbId) + "_DICT_" + std::to_string(dictId);
        DictDescriptor dd(dictId, dictName, cd.columnType.get_comp_param(), false, folderPath);
        dds.push_back(dd);
        if (!cd.columnType.is_array()) {
          cd.columnType.set_size(cd.columnType.get_comp_param() / 8);
        }
        cd.columnType.set_comp_param(dictId);
      }
      sqliteConnector_.query_with_text_params(
          "INSERT INTO mapd_columns (tableid, columnid, name, coltype, colsubtype, coldim, colscale, is_notnull, "
          "compression, comp_param, size, chunks, is_systemcol, is_virtualcol, virtual_expr) VALUES (?, ?, ?, ?, ?, ?, "
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
  addTableToMap(td, cds, dds);
}

void Catalog::dropTable(const TableDescriptor* td) {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query_with_text_param("DELETE FROM mapd_tables WHERE tableid = ?", std::to_string(td->tableId));
    sqliteConnector_.query_with_text_params(
        "DELETE FROM mapd_dictionaries WHERE dictid in (select comp_param from mapd_columns where compression = ? and "
        "tableid = ?)",
        std::vector<std::string>{std::to_string(kENCODING_DICT), std::to_string(td->tableId)});
    sqliteConnector_.query_with_text_param("DELETE FROM mapd_columns WHERE tableid = ?", std::to_string(td->tableId));
    if (td->isView)
      sqliteConnector_.query_with_text_param("DELETE FROM mapd_views WHERE tableid = ?", std::to_string(td->tableId));
    // must destroy fragmenter before deleteChunks is called.
    if (td->fragmenter != nullptr) {
      auto tableDescIt = tableDescriptorMapById_.find(td->tableId);
      delete td->fragmenter;
      tableDescIt->second->fragmenter = nullptr;  // get around const-ness
    }
    ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId};
    // assuming deleteChunksWithPrefix is atomic
    dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix);
    dataMgr_->checkpoint(currentDB_.dbId, td->tableId);
    dataMgr_->removeTableRelatedDS(currentDB_.dbId, td->tableId);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
#ifdef HAVE_CALCITE
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
#endif  // HAVE_CALCITE
  removeTableFromMap(td->tableName, td->tableId);
}

void Catalog::renameTable(const TableDescriptor* td, const string& newTableName) {
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
#ifdef HAVE_CALCITE
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
#endif  // HAVE_CALCITE
  // Get table descriptor to change it
  TableDescriptor* changeTd = tableDescIt->second;
  changeTd->tableName = newTableName;
  tableDescriptorMap_.erase(tableDescIt);  // erase entry under old name
  tableDescriptorMap_[to_upper(newTableName)] = changeTd;
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
#ifdef HAVE_CALCITE
  calciteMgr_->updateMetadata(currentDB_.dbName, td->tableName);
#endif  // HAVE_CALCITE
  ColumnDescriptor* changeCd = columnDescIt->second;
  changeCd->columnName = newColumnName;
  columnDescriptorMap_.erase(columnDescIt);  // erase entry under old name
  columnDescriptorMap_[std::make_tuple(td->tableId, to_upper(newColumnName))] = changeCd;
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

}  // Catalog_Namespace
