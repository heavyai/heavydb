/**
 * @file		Catalog.cpp
 * @author	Todd Mostak <todd@map-d.com>, Wei Hong <wei@map-d.com>
 * @brief		Functions for System Catalogs
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <list>
#include <exception>
#include <cassert>
#include <memory>
#include "boost/filesystem.hpp"
#include "Catalog.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Fragmenter/InsertOrderFragmenter.h"

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
  sqliteConnector_.query(string("INSERT INTO mapd_users VALUES (") + MAPD_ROOT_USER_ID_STR + ", '" + MAPD_ROOT_USER +
                         "', '" + MAPD_ROOT_PASSWD_DEFAULT + "', 1)");
  sqliteConnector_.query(
      "CREATE TABLE mapd_databases (dbid integer primary key, name text unique, owner integer references mapd_users)");
  createDatabase("mapd", MAPD_ROOT_USER_ID);
};

void SysCatalog::createUser(const string& name, const string& passwd, bool issuper) {
  UserMetadata user;
  if (getMetadataForUser(name, user))
    throw runtime_error("User " + name + " already exists.");
  sqliteConnector_.query("INSERT INTO mapd_users (name, passwd, issuper) VALUES ('" + name + "', '" + passwd + "', " +
                         std::to_string(issuper) + ")");
}

void SysCatalog::dropUser(const string& name) {
  UserMetadata user;
  if (!getMetadataForUser(name, user))
    throw runtime_error("User " + name + " does not exist.");
  sqliteConnector_.query("DELETE FROM mapd_users WHERE userid = " + std::to_string(user.userId));
}

void SysCatalog::alterUser(const int32_t userid, const string* passwd, bool* is_superp) {
  if (passwd != nullptr && is_superp != nullptr)
    sqliteConnector_.query("UPDATE mapd_users SET passwd = '" + *passwd + "', issuper = " + std::to_string(*is_superp) +
                           " WHERE userid = " + std::to_string(userid));
  else if (passwd != nullptr)
    sqliteConnector_.query("UPDATE mapd_users SET passwd = '" + *passwd + "' WHERE userid = " + std::to_string(userid));
  else if (is_superp != nullptr)
    sqliteConnector_.query("UPDATE mapd_users SET issuper = " + std::to_string(*is_superp) + " WHERE userid = " +
                           std::to_string(userid));
}

void SysCatalog::createDatabase(const string& name, int owner) {
  DBMetadata db;
  if (getMetadataForDB(name, db))
    throw runtime_error("Database " + name + " already exists.");
  sqliteConnector_.query("INSERT INTO mapd_databases (name, owner) VALUES ('" + name + "', " + std::to_string(owner) +
                         ")");
  SqliteConnector dbConn(name, basePath_ + "/mapd_catalogs/");
  dbConn.query(
      "CREATE TABLE mapd_tables (tableid integer primary key, name text unique, ncolumns integer, isview boolean, "
      "fragments text, frag_type integer, max_frag_rows integer, frag_page_size integer, max_rows bigint, partitions "
      "text)");
  dbConn.query(
      "CREATE TABLE mapd_columns (tableid integer references mapd_tables, columnid integer, name text, coltype "
      "integer, colsubtype integer, coldim integer, colscale integer, is_notnull boolean, compression integer, "
      "comp_param integer, size integer, chunks text, is_systemcol boolean, is_virtualcol boolean, virtual_expr text, "
      "primary key(tableid, columnid), unique(tableid, name))");
  dbConn.query(
      "CREATE TABLE mapd_views (tableid integer references mapd_tables, sql text, materialized boolean, storage int, "
      "refresh int)");
  dbConn.query(
      "CREATE TABLE mapd_frontend_views (viewid integer primary key, name text unique, userid integer references "
      "mapd_users, view_state text)");
  dbConn.query(
      "CREATE TABLE mapd_dictionaries (dictid integer primary key, name text unique, nbits int, is_shared boolean)");
}

void SysCatalog::dropDatabase(const int32_t dbid, const std::string& name) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query("DELETE FROM mapd_databases WHERE dbid = " + std::to_string(dbid));
  boost::filesystem::remove(basePath_ + "/mapd_catalogs/" + name);
  ChunkKey chunkKeyPrefix = {dbid};
  dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix);
  dataMgr_->checkpoint();
}

bool SysCatalog::getMetadataForUser(const string& name, UserMetadata& user) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query("SELECT userid, name, passwd, issuper FROM mapd_users WHERE name = '" + name + "'");
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
  sqliteConnector_.query("SELECT dbid, name, owner FROM mapd_databases WHERE name = '" + name + "'");
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
                 bool is_initdb)
    : basePath_(basePath), sqliteConnector_(dbname, basePath + "/mapd_catalogs/"), dataMgr_(dataMgr) {
  if (!is_initdb)
    buildMaps();
}

Catalog::Catalog(const string& basePath, const DBMetadata& curDB, std::shared_ptr<Data_Namespace::DataMgr> dataMgr)
    : basePath_(basePath),
      sqliteConnector_(curDB.dbName, basePath + "/mapd_catalogs/"),
      currentDB_(curDB),
      dataMgr_(dataMgr) {
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

void Catalog::buildMaps() {
  string dictQuery("SELECT dictid, name, nbits, is_shared from mapd_dictionaries");
  sqliteConnector_.query(dictQuery);
  size_t numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    int dictId = sqliteConnector_.getData<int>(r, 0);
    std::string dictName = sqliteConnector_.getData<string>(r, 1);
    int dictNBits = sqliteConnector_.getData<int>(r, 2);
    bool is_shared = sqliteConnector_.getData<bool>(r, 3);
    std::string fname = basePath_ + "/mapd_data/" + currentDB_.dbName + "_" + dictName;
    DictDescriptor* dd = new DictDescriptor(dictId, dictName, dictNBits, is_shared, fname);
    dictDescriptorMapById_[dd->dictId] = dd;
  }
  string tableQuery(
      "SELECT tableid, name, ncolumns, isview, fragments, frag_type, max_frag_rows, frag_page_size, max_rows, "
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
    td->fragPageSize = sqliteConnector_.getData<int>(r, 7);
    td->maxRows = sqliteConnector_.getData<int64_t>(r, 8);
    td->partitions = sqliteConnector_.getData<string>(r, 9);
    if (!td->isView) {
      // initialize view fields even though irrelevant
      td->isMaterialized = false;
      td->storageOption = kDISK;
      td->refreshOption = kMANUAL;
      td->checkOption = false;
      td->isReady = true;
      td->fragmenter = nullptr;
    }
    tableDescriptorMap_[td->tableName] = td;
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
    ColumnKey columnKey(cd->tableId, cd->columnName);
    columnDescriptorMap_[columnKey] = cd;
    ColumnIdKey columnIdKey(cd->tableId, cd->columnId);
    columnDescriptorMapById_[columnIdKey] = cd;
  }
  string viewQuery("SELECT tableid, sql, materialized, storage, refresh FROM mapd_views");
  sqliteConnector_.query(viewQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    int32_t tableId = sqliteConnector_.getData<int>(r, 0);
    TableDescriptor* td = tableDescriptorMapById_[tableId];
    td->viewSQL = sqliteConnector_.getData<string>(r, 1);
    td->isMaterialized = sqliteConnector_.getData<bool>(r, 2);
    td->storageOption = (StorageOption)sqliteConnector_.getData<int>(r, 3);
    td->refreshOption = (ViewRefreshOption)sqliteConnector_.getData<int>(r, 4);
    td->isReady = !td->isMaterialized;
    td->fragmenter = nullptr;
  }

  string frontendViewQuery("SELECT viewid, view_state, name FROM mapd_frontend_views");
  sqliteConnector_.query(frontendViewQuery);
  numRows = sqliteConnector_.getNumRows();
  for (size_t r = 0; r < numRows; ++r) {
    FrontendViewDescriptor* vd = new FrontendViewDescriptor();
    vd->viewId = sqliteConnector_.getData<int>(r, 0);
    vd->viewState = sqliteConnector_.getData<string>(r, 1);
    vd->viewName = sqliteConnector_.getData<string>(r, 2);
    frontendViewDescriptorMap_[vd->viewName] = vd;
    frontendViewDescriptorMapById_[vd->viewId] = vd;
  }
}

void Catalog::addTableToMap(TableDescriptor& td,
                            const list<ColumnDescriptor>& columns,
                            const list<DictDescriptor>& dicts) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  TableDescriptor* new_td = new TableDescriptor();
  *new_td = td;
  tableDescriptorMap_[td.tableName] = new_td;
  tableDescriptorMapById_[td.tableId] = new_td;
  for (auto cd : columns) {
    ColumnDescriptor* new_cd = new ColumnDescriptor();
    *new_cd = cd;
    ColumnKey columnKey(new_cd->tableId, new_cd->columnName);
    columnDescriptorMap_[columnKey] = new_cd;
    ColumnIdKey columnIdKey(new_cd->tableId, new_cd->columnId);
    columnDescriptorMapById_[columnIdKey] = new_cd;
  }
  for (auto dd : dicts) {
    DictDescriptor* new_dd = new DictDescriptor(dd);
    dictDescriptorMapById_[dd.dictId] = new_dd;
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
  tableDescriptorMap_.erase(tableName);
  if (td->fragmenter != nullptr)
    delete td->fragmenter;
  delete td;

  // delete all column descriptors for the table
  for (int i = 1; i <= ncolumns; i++) {
    ColumnIdKey cidKey(tableId, i);
    ColumnDescriptorMapById::iterator colDescIt = columnDescriptorMapById_.find(cidKey);
    ColumnDescriptor* cd = colDescIt->second;
    columnDescriptorMapById_.erase(colDescIt);
    ColumnKey cnameKey(tableId, cd->columnName);
    columnDescriptorMap_.erase(cnameKey);
    if (cd->columnType.get_compression() == kENCODING_DICT) {
      DictDescriptorMapById::iterator dictIt = dictDescriptorMapById_.find(cd->columnType.get_comp_param());
      DictDescriptor* dd = dictIt->second;
      dictDescriptorMapById_.erase(dictIt);
      if (dd->stringDict != nullptr)
        delete dd->stringDict;
      boost::filesystem::remove_all(dd->dictFolderPath);
      delete dd;
    }
    delete cd;
  }
}

void Catalog::addFrontendViewToMap(FrontendViewDescriptor& vd) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  FrontendViewDescriptor* new_vd = new FrontendViewDescriptor();
  *new_vd = vd;
  frontendViewDescriptorMap_[vd.viewName] = new_vd;
  frontendViewDescriptorMapById_[vd.viewId] = new_vd;
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
      chunkKeyPrefix, chunkVec, dataMgr_.get(), td->maxFragRows, td->fragPageSize, td->maxRows);
}

const TableDescriptor* Catalog::getMetadataForTable(const string& tableName) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto tableDescIt = tableDescriptorMap_.find(tableName);
  if (tableDescIt == tableDescriptorMap_.end()) {  // check to make sure table exists
    return nullptr;
  }
  TableDescriptor* td = tableDescIt->second;
  if (td->fragmenter == nullptr)
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
  if (td->fragmenter == nullptr)
    instantiateFragmenter(td);
  return td;  // returns pointer to table descriptor
}

const DictDescriptor* Catalog::getMetadataForDict(int dictId) const {
  auto dictDescIt = dictDescriptorMapById_.find(dictId);
  if (dictDescIt == dictDescriptorMapById_.end()) {  // check to make sure dictionary exists
    return nullptr;
  }
  DictDescriptor* dd = dictDescIt->second;
  {
    std::lock_guard<std::mutex> lock(cat_mutex_);
    if (dd->stringDict == nullptr)
      dd->stringDict = new StringDictionary(dd->dictFolderPath);
  }
  return dd;
}

const ColumnDescriptor* Catalog::getMetadataForColumn(int tableId, const string& columnName) const {
  ColumnKey columnKey(tableId, columnName);
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

const FrontendViewDescriptor* Catalog::getMetadataForFrontendView(const string& viewName) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto viewDescIt = frontendViewDescriptorMap_.find(viewName);
  if (viewDescIt == frontendViewDescriptorMap_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return viewDescIt->second;  // returns pointer to view descriptor
}

const FrontendViewDescriptor* Catalog::getMetadataForFrontendView(int viewId) const {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  auto frontendViewDescIt = frontendViewDescriptorMapById_.find(viewId);
  if (frontendViewDescIt == frontendViewDescriptorMapById_.end()) {  // check to make sure view exists
    return nullptr;
  }
  return frontendViewDescIt->second;
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
  for (auto p : frontendViewDescriptorMapById_)
    view_list.push_back(p.second);
  return view_list;
}

void Catalog::createTable(TableDescriptor& td, const list<ColumnDescriptor>& columns) {
  list<ColumnDescriptor> cds;
  list<DictDescriptor> dds;
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query(
        "INSERT INTO mapd_tables (name, ncolumns, isview, fragments, frag_type, max_frag_rows, frag_page_size, "
        "max_rows, partitions) VALUES ('" +
        td.tableName + "', " + std::to_string(columns.size()) + ", " + std::to_string(td.isView) + ", '', " +
        std::to_string(td.fragType) + ", " + std::to_string(td.maxFragRows) + ", " + std::to_string(td.fragPageSize) +
        "," + std::to_string(td.maxRows) + ", '')");
    // now get the auto generated tableid
    sqliteConnector_.query("SELECT tableid FROM mapd_tables WHERE name = '" + td.tableName + "'");
    td.tableId = sqliteConnector_.getData<int>(0, 0);
    int colId = 1;
    for (auto cd : columns) {
      if (cd.columnType.get_compression() == kENCODING_DICT) {
        std::string dictName = td.tableName + "_" + cd.columnName + "_dict";
        sqliteConnector_.query("INSERT INTO mapd_dictionaries (name, nbits, is_shared) VALUES ('" + dictName + "', " +
                               std::to_string(cd.columnType.get_comp_param()) + ", 0)");
        sqliteConnector_.query("SELECT dictid FROM mapd_dictionaries WHERE name = '" + dictName + "'");
        int dictId = sqliteConnector_.getData<int>(0, 0);
        std::string folderPath = basePath_ + "/mapd_data/" + currentDB_.dbName + "_" + dictName;
        DictDescriptor dd(dictId, dictName, cd.columnType.get_comp_param(), false, folderPath);
        dds.push_back(dd);
        if (!cd.columnType.is_array()) {
          cd.columnType.set_size(cd.columnType.get_comp_param() / 8);
        }
        cd.columnType.set_comp_param(dictId);
      }
      sqliteConnector_.query(
          "INSERT INTO mapd_columns (tableid, columnid, name, coltype, colsubtype, coldim, colscale, is_notnull, "
          "compression, comp_param, size, chunks, is_systemcol, is_virtualcol, virtual_expr) VALUES (" +
          std::to_string(td.tableId) + ", " + std::to_string(colId) + ", '" + cd.columnName + "', " +
          std::to_string(cd.columnType.get_type()) + ", " + std::to_string(cd.columnType.get_subtype()) + ", " +
          std::to_string(cd.columnType.get_dimension()) + ", " + std::to_string(cd.columnType.get_scale()) + ", " +
          std::to_string(cd.columnType.get_notnull()) + ", " + std::to_string(cd.columnType.get_compression()) + ", " +
          std::to_string(cd.columnType.get_comp_param()) + ", " + std::to_string(cd.columnType.get_size()) + ",''," +
          std::to_string(cd.isSystemCol) + "," + std::to_string(cd.isVirtualCol) + ",'" + cd.virtualExpr + "')");
      cd.tableId = td.tableId;
      cd.columnId = colId++;
      cds.push_back(cd);
    }
    if (td.isView) {
      sqliteConnector_.query_with_text_param(
          "INSERT INTO mapd_views (tableid, sql, materialized, storage, refresh) VALUES (" +
              std::to_string(td.tableId) + ", ?, " + std::to_string(td.isMaterialized) + ", " +
              std::to_string(td.storageOption) + ", " + std::to_string(td.refreshOption) + ")",
          td.viewSQL);
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
    sqliteConnector_.query("DELETE FROM mapd_tables WHERE tableid = " + std::to_string(td->tableId));
    sqliteConnector_.query(
        "DELETE FROM mapd_dictionaries WHERE dictid in (select comp_param from mapd_columns where compression = " +
        std::to_string(kENCODING_DICT) + " and tableid = " + std::to_string(td->tableId) + ")");
    sqliteConnector_.query("DELETE FROM mapd_columns WHERE tableid = " + std::to_string(td->tableId));
    if (td->isView)
      sqliteConnector_.query("DELETE FROM mapd_views WHERE tableid = " + std::to_string(td->tableId));
    // must destroy fragmenter before deleteChunks is called.
    if (td->fragmenter != nullptr) {
      auto tableDescIt = tableDescriptorMapById_.find(td->tableId);
      delete td->fragmenter;
      tableDescIt->second->fragmenter = nullptr;  // get around const-ness
    }
    ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId};
    // assuming deleteChunksWithPrefix is atomic
    dataMgr_->deleteChunksWithPrefix(chunkKeyPrefix);
    dataMgr_->checkpoint();
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  removeTableFromMap(td->tableName, td->tableId);
}

void Catalog::renameTable(const TableDescriptor* td, const string& newTableName) {
  std::lock_guard<std::mutex> lock(cat_mutex_);
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("UPDATE mapd_tables SET name = '" + newTableName + "' WHERE tableid = " +
                           std::to_string(td->tableId));
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.find(td->tableName);
  CHECK(tableDescIt != tableDescriptorMap_.end());
  // Get table descriptor to change it
  TableDescriptor* changeTd = tableDescIt->second;
  changeTd->tableName = newTableName;
  tableDescriptorMap_.erase(tableDescIt);  // erase entry under old name
  tableDescriptorMap_[newTableName] = changeTd;
}

void Catalog::renameColumn(const TableDescriptor* td, const ColumnDescriptor* cd, const string& newColumnName) {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    sqliteConnector_.query("UPDATE mapd_columns SET name = '" + newColumnName + "' WHERE tableid = " +
                           std::to_string(td->tableId) + " AND columnid = " + std::to_string(cd->columnId));
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.find(std::make_tuple(td->tableId, cd->columnName));
  CHECK(columnDescIt != columnDescriptorMap_.end());
  ColumnDescriptor* changeCd = columnDescIt->second;
  changeCd->columnName = newColumnName;
  columnDescriptorMap_.erase(columnDescIt);  // erase entry under old name
  columnDescriptorMap_[std::make_tuple(td->tableId, newColumnName)] = changeCd;
}

void Catalog::createFrontendView(FrontendViewDescriptor& vd) {
  sqliteConnector_.query("BEGIN TRANSACTION");
  try {
    // TODO(andrew): this should be an upsert
    sqliteConnector_.query("SELECT viewid FROM mapd_frontend_views WHERE name = '" + vd.viewName + "'");
    if (sqliteConnector_.getNumRows() > 0) {
      sqliteConnector_.query("UPDATE mapd_frontend_views SET view_state = '" + vd.viewState + "' where name = '" +
                             vd.viewName + "'");
    } else {
      sqliteConnector_.query("INSERT INTO mapd_frontend_views (name, view_state) VALUES ('" + vd.viewName + "', '" +
                             vd.viewState + "')");
    }
    // now get the auto generated viewid
    sqliteConnector_.query("SELECT viewid FROM mapd_frontend_views WHERE name = '" + vd.viewName + "'");
    vd.viewId = sqliteConnector_.getData<int>(0, 0);
  } catch (std::exception& e) {
    sqliteConnector_.query("ROLLBACK TRANSACTION");
    throw;
  }
  sqliteConnector_.query("END TRANSACTION");
  addFrontendViewToMap(vd);
}

}  // Catalog_Namespace
