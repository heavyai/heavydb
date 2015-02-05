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
using Fragmenter_Namespace::ColumnInfo;
using Fragmenter_Namespace::InsertOrderFragmenter;

namespace Catalog_Namespace {

void
SysCatalog::initDB()
{
	sqliteConnector_.query("CREATE TABLE mapd_users (userid integer primary key, name text unique, passwd text, issuper boolean)");
	sqliteConnector_.query(string("INSERT INTO mapd_users VALUES (") + MAPD_ROOT_USER_ID_STR + ", '" + MAPD_ROOT_USER + "', '" + MAPD_ROOT_PASSWD_DEFAULT + "', 1)");
	sqliteConnector_.query("CREATE TABLE mapd_databases (dbid integer primary key, name text unique, owner integer references mapd_users)");
	createDatabase("mapd", MAPD_ROOT_USER_ID);
};

void
SysCatalog::createUser(const string &name, const string &passwd, bool issuper)
{
	if (!currentUser_.isSuper)
		throw runtime_error("Only super user can create new users.");
	UserMetadata user;
	if (getMetadataForUser(name, user))
		throw runtime_error("User " + name + " already exists.");
	sqliteConnector_.query("INSERT INTO mapd_users (name, passwd, issuper) VALUES ('" + name + "', '" + passwd + "', " + boost::lexical_cast<string>(issuper) + ")");
}

void
SysCatalog::dropUser(const string &name)
{
	if (!currentUser_.isSuper)
		throw runtime_error("Only super user can drop users.");
	UserMetadata user;
	if (!getMetadataForUser(name, user))
		throw runtime_error("User " + name + " does not exist.");
	sqliteConnector_.query("DELETE FROM mapd_users WHERE userid = " + boost::lexical_cast<string>(user.userId));
}

void
SysCatalog::alterUser(const string &name, const string *passwd, bool *is_superp)
{
	UserMetadata user;
	if (!getMetadataForUser(name, user))
		throw runtime_error("User " + name + " does not exist.");
	if (!currentUser_.isSuper && currentUser_.userId != user.userId)
		throw runtime_error("Only user super can change another user's password.");
	if (passwd != nullptr && is_superp != nullptr)
		sqliteConnector_.query("UPDATE mapd_users SET passwd = '" + *passwd + "', issuper = " + boost::lexical_cast<std::string>(*is_superp) + " WHERE userid = " + boost::lexical_cast<string>(user.userId));
	else if (passwd != nullptr)
		sqliteConnector_.query("UPDATE mapd_users SET passwd = '" + *passwd + "' WHERE userid = " + boost::lexical_cast<string>(user.userId));
	else if (is_superp != nullptr)
		sqliteConnector_.query("UPDATE mapd_users SET issuper = " + boost::lexical_cast<std::string>(*is_superp) + " WHERE userid = " + boost::lexical_cast<string>(user.userId));
}

void
SysCatalog::createDatabase(const string &name, int owner)
{
	DBMetadata db;
	if (getMetadataForDB(name, db))
		throw runtime_error("Database " + name + " already exists.");
	sqliteConnector_.query("INSERT INTO mapd_databases (name, owner) VALUES ('" + name + "', " + boost::lexical_cast<string>(owner) + ")");
	SqliteConnector dbConn(name, basePath_+"/mapd_catalogs/");
	dbConn.query("CREATE TABLE mapd_tables (tableid integer primary key, name text unique, ncolumns integer, isview boolean, fragments text, frag_type integer, max_frag_rows integer, frag_page_size integer, partitions text)");
	dbConn.query("CREATE TABLE mapd_columns (tableid integer references mapd_tables, columnid integer, name text, coltype integer, coldim integer, colscale integer, is_notnull boolean, compression integer, comp_param integer, chunks text, primary key(tableid, columnid), unique(tableid, name))");
	dbConn.query("CREATE TABLE mapd_views (tableid integer references mapd_tables, sql text, materialized boolean, storage int, refresh int)");
}

void
SysCatalog::dropDatabase(const string &name)
{
	DBMetadata db;
	if (!getMetadataForDB(name, db))
		throw runtime_error("Database " + name + " does not exist.");
	if (!currentUser_.isSuper && currentUser_.userId != db.dbOwner)
		throw runtime_error("Only the super user or the owner can drop database.");
	sqliteConnector_.query("DELETE FROM mapd_databases WHERE name = '" + name + "'");
	boost::filesystem::remove(basePath_+"/mapd_catalogs/" + name);
	ChunkKey chunkKeyPrefix = {db.dbId};
	dataMgr_.deleteChunksWithPrefix(chunkKeyPrefix);
}

bool
SysCatalog::getMetadataForUser(const string &name, UserMetadata &user)
{
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

list<DBMetadata>
SysCatalog::getAllDBMetadata()
{
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

list<UserMetadata>
SysCatalog::getAllUserMetadata() 
{
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

bool
SysCatalog::getMetadataForDB(const string &name, DBMetadata &db)
{
	sqliteConnector_.query("SELECT dbid, name, owner FROM mapd_databases WHERE name = '" + name + "'");
	int numRows = sqliteConnector_.getNumRows();
	if (numRows == 0)
		return false;
	db.dbId = sqliteConnector_.getData<int>(0, 0);
	db.dbName = sqliteConnector_.getData<string>(0, 1);
	db.dbOwner = sqliteConnector_.getData<int>(0, 2);
	return true;
}

Catalog::Catalog(const string &basePath, const string &dbname, Data_Namespace::DataMgr &dataMgr, bool is_initdb): basePath_(basePath), sqliteConnector_(dbname, basePath + "/mapd_catalogs/"), dataMgr_(dataMgr)
{
		if (!is_initdb)
			buildMaps();
		else {
			currentUser_ = UserMetadata(MAPD_ROOT_USER_ID, MAPD_ROOT_USER, MAPD_ROOT_PASSWD_DEFAULT, true);
		}
}

Catalog::Catalog(const string &basePath, const UserMetadata &curUser, const DBMetadata &curDB, Data_Namespace::DataMgr &dataMgr): basePath_(basePath), sqliteConnector_(curDB.dbName, basePath + "/mapd_catalogs/"), currentUser_(curUser), currentDB_(curDB), dataMgr_(dataMgr) 
{
    buildMaps();
}

Catalog::~Catalog() {
    // must clean up heap-allocated TableDescriptor and ColumnDescriptor structs
    for (TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.begin(); tableDescIt != tableDescriptorMap_.end(); ++tableDescIt) {
				if (tableDescIt->second->fragmenter != nullptr)
					delete tableDescIt->second->fragmenter;
        delete tableDescIt -> second;
		}

		// TableDescriptorMapById points to the same descriptors.  No need to delete
		
    for (ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin(); columnDescIt != columnDescriptorMap_.end(); ++columnDescIt)
        delete columnDescIt -> second;

		// ColumnDescriptorMapById points to the same descriptors.  No need to delete
}

void Catalog::buildMaps() {
    string tableQuery("SELECT tableid, name, ncolumns, isview, fragments, frag_type, max_frag_rows, frag_page_size, partitions from mapd_tables");
    sqliteConnector_.query(tableQuery);
    size_t numRows = sqliteConnector_.getNumRows();
    for (int r = 0; r < numRows; ++r) {
				TableDescriptor *td = new TableDescriptor();
				td->tableId = sqliteConnector_.getData<int>(r,0);
        td->tableName = sqliteConnector_.getData<string>(r,1);
				td->nColumns = sqliteConnector_.getData<int>(r,2);
        td->isView = sqliteConnector_.getData<bool>(r, 3);
				td->fragments = sqliteConnector_.getData<string>(r, 4);
				td->fragType = (Fragmenter_Namespace::FragmenterType)sqliteConnector_.getData<int>(r, 5);
				td->maxFragRows = sqliteConnector_.getData<int>(r, 6);
				td->fragPageSize = sqliteConnector_.getData<int>(r, 7);
				td->partitions = sqliteConnector_.getData<string>(r, 8);
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
    string columnQuery("SELECT tableid, columnid, name, coltype, coldim, colscale, is_notnull, compression, comp_param, chunks from mapd_columns");
    sqliteConnector_.query(columnQuery);
    numRows = sqliteConnector_.getNumRows();
    for (int r = 0; r < numRows; ++r) {
				ColumnDescriptor *cd = new ColumnDescriptor();
				cd->tableId = sqliteConnector_.getData<int>(r,0);
				cd->columnId = sqliteConnector_.getData<int>(r,1);
        cd->columnName = sqliteConnector_.getData<string>(r,2);
				cd->columnType.type = (SQLTypes)sqliteConnector_.getData<int>(r,3);
				cd->columnType.dimension = sqliteConnector_.getData<int>(r,4);
				cd->columnType.scale = sqliteConnector_.getData<int>(r,5);
				cd->columnType.notnull = sqliteConnector_.getData<bool>(r,6);
				cd->compression = (EncodingType)sqliteConnector_.getData<int>(r,7);
				cd->comp_param = sqliteConnector_.getData<int>(r,8);
				cd->chunks = sqliteConnector_.getData<string>(r,9);
        ColumnKey columnKey(cd->tableId, cd->columnName);
        columnDescriptorMap_[columnKey] = cd;
				ColumnIdKey columnIdKey(cd->tableId, cd->columnId);
        columnDescriptorMapById_[columnIdKey] = cd;
    }
		string viewQuery("SELECT tableid, sql, materialized, storage, refresh FROM mapd_views");
		sqliteConnector_.query(viewQuery);
    numRows = sqliteConnector_.getNumRows();
    for (int r = 0; r < numRows; ++r) {
				int32_t tableId = sqliteConnector_.getData<int>(r,0);
				TableDescriptor *td = tableDescriptorMapById_[tableId];
				td->viewSQL = sqliteConnector_.getData<string>(r,1);
				td->isMaterialized = sqliteConnector_.getData<bool>(r,2);
				td->storageOption = (StorageOption)sqliteConnector_.getData<int>(r,3);
				td->refreshOption = (ViewRefreshOption)sqliteConnector_.getData<int>(r,4);
				td->isReady = !td->isMaterialized;
				td->fragmenter = nullptr;
    }
}

void
Catalog::addTableToMap(TableDescriptor &td, const list<ColumnDescriptor> &columns)
{
	TableDescriptor *new_td = new TableDescriptor();
	*new_td = td;
	tableDescriptorMap_[td.tableName] = new_td;
	tableDescriptorMapById_[td.tableId] = new_td;
	for (auto cd : columns) {
			ColumnDescriptor *new_cd = new ColumnDescriptor();
			*new_cd = cd;
			ColumnKey columnKey(new_cd->tableId, new_cd->columnName);
			columnDescriptorMap_[columnKey] = new_cd;
			ColumnIdKey columnIdKey(new_cd->tableId, new_cd->columnId);
			columnDescriptorMapById_[columnIdKey] = new_cd;
	}
}

void 
Catalog::removeTableFromMap(const string &tableName, int tableId) 
{
	TableDescriptorMapById::iterator tableDescIt = tableDescriptorMapById_.find(tableId);
	if (tableDescIt == tableDescriptorMapById_.end())
			throw runtime_error ("Table " + tableName + " does not exist.");
	TableDescriptor *td = tableDescIt->second;
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
		ColumnDescriptor *cd = colDescIt->second;
		columnDescriptorMapById_.erase(colDescIt);
		ColumnKey cnameKey(tableId, cd->columnName);
		columnDescriptorMap_.erase(cnameKey);
		delete cd;
	}
}

void
Catalog::instantiateFragmenter(TableDescriptor *td) const
{
	// instatiion table fragmenter upon first use
	// assume only insert order fragmenter is supported
	assert(td->fragType == Fragmenter_Namespace::FragmenterType::INSERT_ORDER);
	vector<ColumnInfo> columnInfoVec;
	list<const ColumnDescriptor *> columnDescs;
	getAllColumnMetadataForTable(td, columnDescs);
	ColumnInfo::translateColumnDescriptorsToColumnInfoVec(columnDescs , columnInfoVec);
	ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId};
	td->fragmenter = new InsertOrderFragmenter(chunkKeyPrefix, columnInfoVec, &dataMgr_, td->maxFragRows, td->fragPageSize);
}

const TableDescriptor * Catalog::getMetadataForTable (const string &tableName) const  {
    auto tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) { // check to make sure table exists
        return nullptr;
    }
		TableDescriptor *td = tableDescIt->second;
		if (td->fragmenter == nullptr)
			instantiateFragmenter(td);
    return td; // returns pointer to table descriptor
}

const TableDescriptor * Catalog::getMetadataForTable (int tableId) const  {
    auto tableDescIt = tableDescriptorMapById_.find(tableId);
    if (tableDescIt == tableDescriptorMapById_.end()) { // check to make sure table exists
        return nullptr;
    }
		TableDescriptor *td = tableDescIt->second;
		if (td->fragmenter == nullptr)
			instantiateFragmenter(td);
    return td; // returns pointer to table descriptor
}

const ColumnDescriptor * Catalog::getMetadataForColumn (int tableId, const string &columnName) const {
    ColumnKey columnKey (tableId, columnName);       
    auto colDescIt = columnDescriptorMap_.find(columnKey);
    if (colDescIt == columnDescriptorMap_.end()) { // need to check to make sure column exists for table
        return nullptr;
    }
    return colDescIt -> second;
}

const ColumnDescriptor * Catalog::getMetadataForColumn (int tableId, int columnId) const {
    ColumnIdKey columnIdKey (tableId, columnId);       
    auto colDescIt = columnDescriptorMapById_.find(columnIdKey);
    if (colDescIt == columnDescriptorMapById_.end()) { // need to check to make sure column exists for table
        return nullptr;
    }
    return colDescIt -> second;
}

void 
Catalog::getAllColumnMetadataForTable(const TableDescriptor *td, list<const ColumnDescriptor *> &columnDescriptors) const {
		for (int i = 1; i <= td->nColumns; i++) {
			const ColumnDescriptor *cd = getMetadataForColumn(td->tableId, i);
			assert(cd != nullptr);
			columnDescriptors.push_back(cd);
		}
}

list <const ColumnDescriptor *> Catalog::getAllColumnMetadataForTable(const int tableId) const {
    list <const ColumnDescriptor *> columnDescriptors;
		const TableDescriptor *td = getMetadataForTable(tableId);
		getAllColumnMetadataForTable(td, columnDescriptors);
    return columnDescriptors;
}

list<const TableDescriptor*>
Catalog::getAllTableMetadata() const
{
	list<const TableDescriptor*> table_list;
	for (auto p : tableDescriptorMapById_)
		table_list.push_back(p.second);
	return table_list;
}

void
Catalog::createTable(TableDescriptor &td, const list<ColumnDescriptor> &columns)
{
	list<ColumnDescriptor> cds;
	sqliteConnector_.query("BEGIN TRANSACTION");
	try {
		sqliteConnector_.query("INSERT INTO mapd_tables (name, ncolumns, isview, fragments, frag_type, max_frag_rows, frag_page_size, partitions) VALUES ('" + td.tableName + "', " + boost::lexical_cast<string>(columns.size()) + ", " + boost::lexical_cast<string>(td.isView) + ", '', " + boost::lexical_cast<string>(td.fragType) + ", " + boost::lexical_cast<string>(td.maxFragRows) + ", " + boost::lexical_cast<string>(td.fragPageSize) +  ", '')");
		// now get the auto generated tableid
		sqliteConnector_.query("SELECT tableid FROM mapd_tables WHERE name = '" + td.tableName + "'");
		td.tableId = sqliteConnector_.getData<int>(0, 0);
		int colId = 1;
		for (auto cd : columns) {
			sqliteConnector_.query("INSERT INTO mapd_columns (tableid, columnid, name, coltype, coldim, colscale, is_notnull, compression, comp_param, chunks) VALUES (" + boost::lexical_cast<string>(td.tableId) + ", " + boost::lexical_cast<string>(colId) + ", '" + cd.columnName + "', " + boost::lexical_cast<string>(cd.columnType.type) + ", " + boost::lexical_cast<string>(cd.columnType.dimension) + ", " + boost::lexical_cast<string>(cd.columnType.scale) + ", " + boost::lexical_cast<string>(cd.columnType.notnull) + ", " + boost::lexical_cast<string>(cd.compression) + ", " + boost::lexical_cast<string>(cd.comp_param) + ", '')");
			cd.tableId = td.tableId;
			cd.columnId = colId++;
			cds.push_back(cd);
		}
		if (td.isView) {
			sqliteConnector_.query_with_text_param("INSERT INTO mapd_views (tableid, sql, materialized, storage, refresh) VALUES (" + boost::lexical_cast<string>(td.tableId) + ", ?, " + boost::lexical_cast<string>(td.isMaterialized) + ", " + boost::lexical_cast<string>(td.storageOption) + ", " + boost::lexical_cast<string>(td.refreshOption) + ")", td.viewSQL);
		}
	}
	catch (std::exception &e) {
		sqliteConnector_.query("ROLLBACK TRANSACTION");
		throw;
	}
	sqliteConnector_.query("END TRANSACTION");
	addTableToMap(td, cds);
}

void
Catalog::dropTable(const TableDescriptor *td)
{
	sqliteConnector_.query("BEGIN TRANSACTION");
	try {
		sqliteConnector_.query("DELETE FROM mapd_tables WHERE tableid = " + boost::lexical_cast<string>(td->tableId));
		sqliteConnector_.query("DELETE FROM mapd_columns WHERE tableid = " + boost::lexical_cast<string>(td->tableId));
		if (td->isView)
			sqliteConnector_.query("DELETE FROM mapd_views WHERE tableid = " + boost::lexical_cast<string>(td->tableId));
		ChunkKey chunkKeyPrefix = {currentDB_.dbId, td->tableId};
		// assuming deleteChunksWithPrefix is atomic
		dataMgr_.deleteChunksWithPrefix(chunkKeyPrefix);
	}
	catch (std::exception &e) {
		sqliteConnector_.query("ROLLBACK TRANSACTION");
		throw;
	}
	sqliteConnector_.query("END TRANSACTION");
	removeTableFromMap(td->tableName, td->tableId);
}

} // Catalog_Namespace
