/**
 * @file		Catalog.cpp
 * @author	Todd Mostak <todd@map-d.com>, Wei Hong <wei@map-d.com>
 * @brief		Functions for System Catalogs
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <fstream>
#include <set>
#include <iostream>
#include <exception>
#include <cassert>
#include "Catalog.h"

using std::runtime_error;
using std::string;
using std::map;
using std::tuple;
using std::vector;
using std::set;
using std::ifstream;
using std::ofstream;
using std::pair;

namespace Catalog_Namespace {

void
SysCatalog::initDB()
{
	sqliteConnector_.query("CREATE TABLE mapd_users (userid integer primary key, name text unique, passwd text, boolean issuper)");
	sqliteConnector_.query("INSERT INTO mapd_users VALUES (0,'mapd', 'HyperInteractive', 1)");
	sqliteConnector_.query("CREATE TABLE mapd_databases (dbid integer primary key, name text unique, owner integer references mapd_users)");
};

void
SysCatalog::createUser(const string &name, const string &passwd, bool issuper, const UserMetadata &curUser)
{
	if (!curUser.isSuper)
		throw runtime_error("Only super user can create new users.");
	UserMetadata user;
	if (getMetadataForUser(name, user))
		throw runtime_error("User " + name + " already exists.");
	sqliteConnector_.query("INSERT INTO mapd_users (name, passwd, issuper) VALUES ('" + name + "', '" + passwd + "', " + boost::lexical_cast<string>(issuper) + ")");
}

void
SysCatalog::dropUser(const string &name, const UserMetadata &curUser)
{
	if (!curUser.isSuper)
		throw runtime_error("Only super user can drop users.");
	UserMetadata user;
	if (!getMetadataForUser(name, user))
		throw runtime_error("User " + name + " does not exist.");
	sqliteConnector_.query("DELETE FROM mapd_users WHERE userid = " + boost::lexical_cast<string>(user.userId));
}

void
SysCatalog::changeUserPasswd(const string &name, const string &passwd, const UserMetadata &curUser)
{
	UserMetadata user;
	if (!getMetadataForUser(name, user))
		throw runtime_error("User " + name + " does not exist.");
	if (!curUser.isSuper && curUser.userId != user.userId)
		throw runtime_error("Only user super can change another user's password.");
	sqliteConnector_.query("UPDATE mapd_users SET passwd = '" + passwd + "' WHERE userid = " + boost::lexical_cast<string>(user.userId));
}

void
SysCatalog::createDatabase(const string &name, const UserMetadata &curUser)
{
	DBMetadata db;
	if (getMetadataForDB(name, db))
		throw runtime_error("Database " + name + " already exists.");
	sqliteConnector_.query("INSERT INTO mapd_databases (name, owner) VALUES ('" + name + "', " + boost::lexical_cast<string>(curUser.userId) + ")");
	SqliteConnector dbConn(name, basePath_);
	dbConn.query("CREATE TABLE mapd_tables (tableid integer primgary key, name text unique, ncolumns integer, isview boolean, fragments text, partitions text)");
	dbConn.query("CREATE TABLE mapd_columns (tableid integer references mapd_tables, columnid integer, name text, coltype integer, coldim integer, colscale integer, notnull boolean, compression integer, chunks text, primary key(tableid, columnid), unique(tableid, name))");
	dbConn.query("CREATE TABLE mapd_views (tableid integer references mapd_tables, sql text, isGPU boolean)");
}

void
SysCatalog::dropDatabase(const string &name, const UserMetadata &curUser)
{
	DBMetadata db;
	if (!getMetadataForDB(name, db))
		throw runtime_error("Database " + name + " does not exist.");
	if (!curUser.isSuper && curUser.userId != db.dbOwner)
		throw runtime_error("Only the super user or the owner can drop database.");
	sqliteConnector_.query("INSERT INTO mapd_databases (name, owner) VALUES ('" + name + "', " + boost::lexical_cast<string>(curUser.userId) + ")");
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

Catalog::Catalog(const string &basePath, const UserMetadata &curUser, const DBMetadata &curDB): basePath_(basePath), sqliteConnector_(curDB.dbName,basePath), currentUser_(curUser), currentDB_(curDB) 
{
    buildMaps();
}

Catalog::~Catalog() {
    // must clean up heap-allocated TableDescriptor and ColumnDescriptor structs
    for (TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.begin(); tableDescIt != tableDescriptorMap_.end(); ++tableDescIt)
        delete tableDescIt -> second;

		// TableDescriptorMapById points to the same descriptors.  No need to delete
		
    for (ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin(); columnDescIt != columnDescriptorMap_.end(); ++columnDescIt)
        delete columnDescIt -> second;

		// ColumnDescriptorMapById points to the same descriptors.  No need to delete
}

void Catalog::buildMaps() {
    string tableQuery("SELECT tableid, name, ncolumns, isview, fragments, partitions from mapd_tables");
    sqliteConnector_.query(tableQuery);
    size_t numRows = sqliteConnector_.getNumRows();
    for (int r = 0; r < numRows; ++r) {
				TableDescriptor *td = new TableDescriptor();
				td->tableId = sqliteConnector_.getData<int>(r,0);
        td->tableName = sqliteConnector_.getData<string>(r,1);
				td->nColumns = sqliteConnector_.getData<int>(r,2);
        td->isView = sqliteConnector_.getData<bool>(r, 3);
				if (!td->isView) {
					td->fragments = sqliteConnector_.getData<string>(r, 4);
					td->partitions = sqliteConnector_.getData<string>(r, 5);
					td->isGPU = false;
				} 
				else {
					throw runtime_error("Views are not supported yet.");
				}
        tableDescriptorMap_[td->tableName] = td;
        tableDescriptorMapById_[td->tableId] = td;
    }
    string columnQuery("SELECT tableid, columnid, name, coltype, coldim, colscale, notnull, compression, chunks from mapd_columns");
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
				cd->compression = sqliteConnector_.getData<int>(r,7);
				cd->chunks = sqliteConnector_.getData<string>(r,8);
        ColumnKey columnKey(cd->tableId, cd->columnName);
        columnDescriptorMap_[columnKey] = cd;
				ColumnIdKey columnIdKey(cd->tableId, cd->columnId);
        columnDescriptorMapById_[columnIdKey] = cd;
    }
}

void
Catalog::addTableToMap(TableDescriptor *td, const vector<ColumnDescriptor *> &columns)
{
	tableDescriptorMap_[td->tableName] = td;
	tableDescriptorMapById_[td->tableId] = td;
	for (auto cd : columns) {
			ColumnKey columnKey(cd->tableId, cd->columnName);
			columnDescriptorMap_[columnKey] = cd;
			ColumnIdKey columnIdKey(cd->tableId, cd->columnId);
			columnDescriptorMapById_[columnIdKey] = cd;
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

const TableDescriptor * Catalog::getMetadataForTable (const string &tableName) const  {
    auto tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) { // check to make sure table exists
        return nullptr;
    }
    return tableDescIt -> second; // returns pointer to table descriptor
}

const TableDescriptor * Catalog::getMetadataForTable (int tableId) const  {
    auto tableDescIt = tableDescriptorMapById_.find(tableId);
    if (tableDescIt == tableDescriptorMapById_.end()) { // check to make sure table exists
        return nullptr;
    }
    return tableDescIt -> second; // returns pointer to table descriptor
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

// must return vector of ColumnDescriptor's in the order of column id
vector <const ColumnDescriptor *> Catalog::getAllColumnMetadataForTable(const int tableId) const {
    vector <const ColumnDescriptor *> columnDescriptors;
		const TableDescriptor *td = getMetadataForTable(tableId);
		for (int i = 1; i <= td->nColumns; i++) {
			const ColumnDescriptor *cd = getMetadataForColumn(tableId, i);
			assert(cd != nullptr);
			columnDescriptors.push_back(cd);
		}
    return columnDescriptors;
}

void
Catalog::createTable(const string &tableName, const vector<ColumnDescriptor *> &columns)
{
	sqliteConnector_.query("INSERT INTO mapd_tables (name, ncolumns, isview, fragments, partitions) VALUES ('" + tableName + "', " + boost::lexical_cast<string>(columns.size()) + ", 0, '', '')");
	// now get the auto generated tableid
	sqliteConnector_.query("SELECT tableid FROM mapd_tables WHERE name = '" + tableName + "'");
	int32_t tableId = sqliteConnector_.getData<int>(0, 0);
	int colId = 1;
	for (auto cd : columns) {
		sqliteConnector_.query("INSERT INTO mapd_columns (tableid, columnid, name, coltype, coldim, colscale, notnull, compression, chunks) VALUES (" + boost::lexical_cast<string>(tableId) + ", " + boost::lexical_cast<string>(colId) + ", '" + cd->columnName + "', " + boost::lexical_cast<string>(cd->columnType.type) + ", " + boost::lexical_cast<string>(cd->columnType.dimension) + ", " + boost::lexical_cast<string>(cd->columnType.scale) + ", " + boost::lexical_cast<string>(cd->columnType.notnull) + ", 0, '')");
		cd->tableId = tableId;
		cd->columnId = colId++;
	}
	TableDescriptor *td = new TableDescriptor();
	td->tableId = tableId;
	td->tableName = tableName;
	td->isView = false;
	td->isGPU = false;
	addTableToMap(td, columns);
}

void
Catalog::dropTable(const string &tableName)
{
	const TableDescriptor *td = getMetadataForTable(tableName);
	if (td == nullptr)
		throw runtime_error("Table " + tableName + " does not exist.");
	removeTableFromMap(tableName, td->tableId);
}

} // Catalog_Namespace
