#include "Catalog.h"
#include <fstream>
#include <set>
#include <iostream>
#include <exception>

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

Catalog::Catalog(const string &basePath): basePath_(basePath), maxTableId_(-1), maxColumnId_(-1), isDirty_(false), sqliteConnector_("mapd",basePath) {
    //readCatalogFromFile();
    createStateTableIfDne();
    readState();
}

Catalog::~Catalog() {
    writeState();
    // must clean up heap-allocated TableDescriptor and ColumnDescriptor structs
    for (TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.begin(); tableDescIt != tableDescriptorMap_.end(); ++tableDescIt)
        delete tableDescIt -> second;

    for (ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin(); columnDescIt != columnDescriptorMap_.end(); ++columnDescIt)
        delete columnDescIt -> second;
}

void Catalog::readCatalogFromFile() {
    string tableFileFullPath (basePath_ + "/tables.cat");
    ifstream tableFile (tableFileFullPath.c_str());
    // read in table file if it exists
    if (tableFile) {
        // file exists and is open for input
        // expects tab-delimited file
        string tempTableName;
        int tempTableId;
        while (tableFile >> tempTableName >> tempTableId) {
           tableDescriptorMap_[tempTableName]  = new TableDescriptor(tempTableName,tempTableId); // should be no need to check this for errors as we check for table entries of same name on insert
           if (tempTableId > maxTableId_)
               maxTableId_ = tempTableId;
        }
        tableFile.close();
        string columnFileFullPath (basePath_ + "/columns.cat");
        ifstream columnFile (columnFileFullPath.c_str());
        // read in column file if it exists
        if (columnFile) {
            // file exists and is open for input
            // expects tab-delimited file
            int tempTableId;
            string tempColumnName;
            int tempColumnId;
            int tempMapd_data_t;
            bool tempNotNull;
            //while (columnFile >> tempTableId >> tempColumnName >> tempColumnId >> tempmapd_data_t >> tempNotNull) { 
            while (columnFile >> tempTableId >> tempColumnName >> tempColumnId >> tempMapd_data_t >> tempNotNull) {
                ColumnKey columnKey (tempTableId, tempColumnName); // construct the tuple that will serve as key for this entry into the column map
                columnDescriptorMap_[columnKey] = new ColumnDescriptor(tempTableId, tempColumnName, tempColumnId, static_cast<mapd_data_t>(tempMapd_data_t), static_cast<bool> (tempNotNull));
                //columnDescriptorMap_[columnKey] = new ColumnDescriptor(tempTableId, tempColumnName, tempColumnId, tempMapd_data_t, tempNotNull);
               // If this column has an id higher than maxColumnId_, set maxColumnId_ to it
               if (tempColumnId > maxColumnId_)
                   maxColumnId_ = tempColumnId;
            }
            columnFile.close();
        }
    }
}

void Catalog::readState() {
    string tableQuery("SELECT table_name, table_id from tables ORDER BY table_id");
    sqliteConnector_.query(tableQuery);
    size_t numRows = sqliteConnector_.getNumRows();
    for (int r = 0; r < numRows; ++r) {
        string tableName = sqliteConnector_.getData<string>(r,0);
        int tableId = sqliteConnector_.getData<int>(r,1);
        tableDescriptorMap_[tableName] = new TableDescriptor(tableName,tableId);
        if (tableId > maxTableId_)
            maxTableId_ = tableId;
    }
    string columnQuery("SELECT table_id, column_name, column_id, column_type, not_null from columns ORDER BY table_id,column_id");
    sqliteConnector_.query(columnQuery);
    numRows = sqliteConnector_.getNumRows();
    for (int r = 0; r < numRows; ++r) {
        int tableId = sqliteConnector_.getData<int>(r,0);
        string columnName = sqliteConnector_.getData<string>(r,1);
        int columnId = sqliteConnector_.getData<int>(r,2);
        mapd_data_t columnType = getTypeFromString(sqliteConnector_.getData<string>(r,3));
        bool notNull = sqliteConnector_.getData<bool>(r,4);
        ColumnKey columnKey(tableId,columnName);
        columnDescriptorMap_[columnKey] = new ColumnDescriptor(tableId,columnName,columnId,columnType,notNull);
        if (columnId > maxColumnId_)
            maxColumnId_ = columnId;
    }
}


void Catalog::createStateTableIfDne() {
     sqliteConnector_.query("CREATE TABLE IF NOT EXISTS tables(table_name TEXT, table_id INT UNIQUE, PRIMARY KEY (table_name))");
     sqliteConnector_.query("CREATE TABLE IF NOT EXISTS columns(table_id INT, column_name TEXT, column_id INT, column_type TEXT, not_null INT, PRIMARY KEY (table_id, column_name))");
}

void Catalog::writeState() {
    if (isDirty_) {
        // we will just overwrite table and column files with all TableDescriptors and ColumnDescriptors
        string deleteTableQuery ("DELETE FROM tables");
        sqliteConnector_.query(deleteTableQuery);
        string deleteColumnQuery ("DELETE FROM columns");
        sqliteConnector_.query(deleteColumnQuery);
        for (TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.begin(); tableDescIt != tableDescriptorMap_.end(); ++tableDescIt) {
            TableDescriptor *tableDescriptor = tableDescIt -> second;
            string insertTableQuery("INSERT INTO tables (table_name, table_id) VALUES ('" + tableDescriptor -> tableName + "'," + boost::lexical_cast<string>(tableDescriptor -> tableId) + ")");
            sqliteConnector_.query(insertTableQuery);
        }
        for (ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin(); columnDescIt != columnDescriptorMap_.end(); ++columnDescIt) {
            ColumnDescriptor *columnDescriptor = columnDescIt -> second;
            string insertColumnQuery("INSERT INTO columns (table_id, column_name, column_id, column_type, not_null) VALUES (" + boost::lexical_cast<string>(columnDescriptor -> tableId)  + ",'" + columnDescriptor -> columnName + "'," + boost::lexical_cast<string>(columnDescriptor -> columnId) + ",'" + getTypeName(columnDescriptor -> columnType) + "'," + (columnDescriptor -> notNull == true ? "1" : "0") + ")" );
            sqliteConnector_.query(insertColumnQuery);
        }
    }
    isDirty_ = false;
}

void Catalog::writeCatalogToFile() {
    if (isDirty_) {
        // we will just overwrite table and column files with all TableDescriptors and ColumnDescriptors
        string tableFileFullPath (basePath_ + "/tables.cat");
        ofstream tableFile (tableFileFullPath.c_str());
        if (tableFile.is_open()) {
            for (TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.begin(); tableDescIt != tableDescriptorMap_.end(); ++tableDescIt) {
                TableDescriptor *tableRow = tableDescIt -> second;
                tableFile << tableRow -> tableName << "\t" << tableRow -> tableId << "\n";
            }
            tableFile.flush();
            tableFile.close();
            // we only try to write to the column file if we've succeeded at writing to the table file
            string columnFileFullPath (basePath_ + "/columns.cat");
            ofstream columnFile (columnFileFullPath.c_str());
            if (columnFile.is_open()) {
                for (ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin(); columnDescIt != columnDescriptorMap_.end(); ++columnDescIt) {
                    ColumnDescriptor *columnRow = columnDescIt -> second;
                    columnFile << columnRow -> tableId << "\t" << columnRow -> columnName << "\t" << columnRow -> columnId << "\t" << columnRow -> columnType << "\t" << columnRow -> notNull << "\n";
                }
            }
            else {
                throw runtime_error("Error: Cannot open catalog column file");
            }
        }
        else {
                throw runtime_error("Error: Cannot open catalog table file");
        }

        //finally set dirty flag to false so catalog doesn't need to be rewritten until next modification
        isDirty_ = false;
    }
}

// returns created table Id
//
int Catalog::addTable(const string &tableName) {
    // This really shouldn't ever be used
    TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt != tableDescriptorMap_.end()) {
        throw runtime_error ("MAPD_ERR_TABLE_AREADY_EXISTS");
    }
    // if here then table did not exist and we can insert it into tableDescriptorMap_
    // Create tableDescriptor and pre-increment nextTableId_ so next TableDescriptor gets id one higher than current max
    int tableId = ++maxTableId_;
    TableDescriptor *tableDescriptor = new TableDescriptor(tableName,tableId); 
    tableDescriptorMap_[tableName] = tableDescriptor;
    isDirty_ = true;
    return tableId;
}


int Catalog::addTableWithColumns(const string &tableName, const vector <ColumnDescriptor > & columns) { 
     //note that vector of ColumnDescriptors should not be populated with tableId and columnIds - the catalog fills these in.
     //
    // first need to check if insert would result in any errors
    TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt != tableDescriptorMap_.end()) {
        throw runtime_error ("MAPD_ERR_TABLE_AREADY_EXISTS");
        //throw runtime_error ("Catalog error: Table already exists");
    }
    set <string> columnNames;
    for (auto colIt = columns.begin(); colIt != columns.end(); ++colIt) {
    if (columnNames.insert(colIt -> columnName).second == false) // tests if we have already specified a column with the same name
        throw runtime_error ("MAPD_ERR_COLUMN_AREADY_EXISTS");
        //throw runtime_error ("Catalog error: Column already exists");
    }

    // if we reached this far then we know table insert will succeed
    int tableId = addTable(tableName); 
    for (auto colIt = columns.begin(); colIt != columns.end(); ++colIt) {
        ColumnDescriptor *columnDescriptor = new ColumnDescriptor (tableId,colIt->columnName,++maxColumnId_,colIt->columnType,colIt->notNull);
         ColumnKey columnKey (tableId, columnDescriptor -> columnName);       
         columnDescriptorMap_[columnKey] = columnDescriptor; // insertion of column
    }
    isDirty_ = true;
    return tableId;
}

void Catalog::addColumnToTable(const string &tableName, const ColumnDescriptor &column) {
     //note that columnDescriptor should not be populated with tableId and columnIds - the database fills these in.
    TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) {
        //throw runtime_error ("Catalog error: Table does not exist");
        throw runtime_error ("MAPD_ERR_TABLE_DOES_NOT_EXIST");
    }
    int tableId = tableDescIt -> second -> tableId;
    ColumnKey columnKey (tableId, column.columnName);       
    ColumnDescriptorMap::iterator colDescIt = columnDescriptorMap_.find(columnKey);
    if (colDescIt != columnDescriptorMap_.end()) {
        throw runtime_error ("Catalog error: Column already exists");
    }
    ColumnDescriptor *columnDescriptor = new ColumnDescriptor (tableId, column.columnName, ++maxColumnId_, column.columnType,column.notNull);

    columnDescriptorMap_[columnKey] = columnDescriptor; // insertion of column 
    isDirty_ = true;
}

void Catalog::removeTable(const string &tableName) {
    TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) {
        //throw runtime_error ("Catalog error: Table does not exist");
        throw runtime_error ("MAPD_ERR_TABLE_DOES_NOT_EXIST");
    }

    // if here then table does exist so we can remove it and its associated columns
    tableDescriptorMap_.erase(tableDescIt);

    // can be multiple columns for the same table, so must iterate and delete each column that belongs to the table
    ColumnDescriptorMap::iterator columnDescIt = columnDescriptorMap_.begin();    
    while (columnDescIt != columnDescriptorMap_.end()) {
        if (std::get<1>(columnDescIt -> first) == tableName)
            columnDescriptorMap_.erase(columnDescIt++);
        else
            ++columnDescIt;
    }
    isDirty_ = true;
}

void Catalog::removeColumnFromTable(const string &tableName, const string &columnName) {
    TableDescriptorMap::iterator tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) { // check to make sure table exists
        //throw runtime_error ("Catalog error: Table does not exist");
        throw runtime_error ("MAPD_ERR_TABLE_DOES_NOT_EXIST");
    }
    int tableId = tableDescIt -> second -> tableId;
    ColumnKey columnKey (tableId, columnName);       
    ColumnDescriptorMap::iterator colDescIt = columnDescriptorMap_.find(columnKey);
    if (colDescIt == columnDescriptorMap_.end()) {// need to check to make sure column exists for table
        //throw runtime_error ("Catalog error: Column does not exist");
        throw runtime_error ("MAPD_ERR_COLUMN_DOES_NOT_EXIST");
    }
    columnDescriptorMap_.erase(colDescIt);
    isDirty_ = true;
}

const TableDescriptor * Catalog::getMetadataForTable (const string &tableName) const  {
    auto tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) { // check to make sure table exists
        //throw runtime_error ("Catalog error: Table does not exist");
        throw runtime_error ("MAPD_ERR_TABLE_DOES_NOT_EXIST");
    }
    return tableDescIt -> second; // returns pointer to table descriptor
}

const ColumnDescriptor * Catalog::getMetadataForColumn (const string &tableName, const string &columnName) const {
    auto tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) { // check to make sure table exists
        //throw runtime_error ("Catalog error: Table does not exist");
        throw runtime_error ("MAPD_ERR_TABLE_DOES_NOT_EXIST");
    }
    int tableId = tableDescIt -> second -> tableId;
    ColumnKey columnKey (tableId, columnName);       
    auto colDescIt = columnDescriptorMap_.find(columnKey);
    if (colDescIt == columnDescriptorMap_.end()) { // need to check to make sure column exists for table
        //throw runtime_error ("Catalog error: Column does not exist");
        throw runtime_error ("MAPD_ERR_COLUMN_DOES_NOT_EXIST");
    }
    return colDescIt -> second;
}

vector <const ColumnDescriptor *> Catalog::getMetadataForColumns (const string &tableName, const vector<string> &columnNames) const {
    auto tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) { // check to make sure table exists
        //throw runtime_error ("Catalog error: Table does not exist");
        throw runtime_error ("MAPD_ERR_TABLE_DOES_NOT_EXIST");
    }
    int tableId = tableDescIt -> second -> tableId;
    vector <const ColumnDescriptor *> columnDescriptors;
    for (vector<string>::const_iterator colNameIt = columnNames.begin(); colNameIt != columnNames.end(); ++colNameIt) {
        ColumnKey columnKey (tableId, *colNameIt);
        auto colDescIt = columnDescriptorMap_.find(columnKey);
        if (colDescIt ==  columnDescriptorMap_.end()) {
            //throw runtime_error ("Catalog error: Column does not exist");
            throw runtime_error ("MAPD_ERR_COLUMN_DOES_NOT_EXIST");
        }
        columnDescriptors.push_back(colDescIt -> second);
    }
    return columnDescriptors;
}

vector <const ColumnDescriptor *> Catalog::getAllColumnMetadataForTable(const string &tableName) const {
    auto tableDescIt = tableDescriptorMap_.find(tableName);
    if (tableDescIt == tableDescriptorMap_.end()) { // check to make sure table exists
        //throw runtime_error ("Catalog error: Table does not exist");
        throw runtime_error ("MAPD_ERR_TABLE_DOES_NOT_EXIST");
    }
    int tableId = tableDescIt -> second -> tableId;
    return getAllColumnMetadataForTable(tableId);
}

vector <const ColumnDescriptor *> Catalog::getAllColumnMetadataForTable(const int tableId) const {
    vector <const ColumnDescriptor *> columnDescriptors;
    for (auto colDescIt = columnDescriptorMap_.begin(); colDescIt != columnDescriptorMap_.end(); ++colDescIt) {
        if (colDescIt -> second -> tableId == tableId) {
            columnDescriptors.push_back(colDescIt -> second);
        }
    }
    return columnDescriptors;
}

vector <const ColumnDescriptor *> Catalog::getMetadataForColumns(const vector <string>  &tableNames, const vector <pair <string, string> > &columnNames) const {

    vector <int> tableIds;
    for (auto tableNameIt = tableNames.begin(); tableNameIt != tableNames.end(); ++tableNameIt) {
        auto tableDescIt = tableDescriptorMap_.find(*tableNameIt);
        if (tableDescIt == tableDescriptorMap_.end()) {
            //throw runtime_error ("Catalog error: Table does not exist");
            throw runtime_error ("MAPD_ERR_TABLE_DOES_NOT_EXIST");
        }
        tableIds.push_back(tableDescIt -> second -> tableId);
    }

    vector <const ColumnDescriptor *> columnDescriptors;
    
    // If here then all tables exist
    for (auto colNameIt = columnNames.begin(); colNameIt != columnNames.end(); ++colNameIt) {
        string tableName (colNameIt -> first);
        auto colDescIt = columnDescriptorMap_.end(); // set this to end at first to signify column not found yet
        if (tableName.size() == 0) { // no explicit table reference
            for (auto tableIdIt = tableIds.begin(); tableIdIt != tableIds.end(); ++tableIdIt) {
                ColumnKey columnKey (*tableIdIt, colNameIt -> second);
                auto tempColDescIt = columnDescriptorMap_.find(columnKey);
                if (tempColDescIt !=  columnDescriptorMap_.end()) { 
                    if (colDescIt != columnDescriptorMap_.end()) { // if we've already found the column
                        //throw runtime_error ("Catalog error: Column reference is ambiguous");
                        throw runtime_error ("MAPD_ERR_COLUMN_IS_AMBIGUOUS");
                    }
                    colDescIt = tempColDescIt;
                }
            }
        }
        else { // we have table_name.column_name
            int tableNameFound = false;
            int tableIndex;
            for (tableIndex = 0; tableIndex != tableNames.size(); ++tableIndex) {
                if (tableName == tableNames[tableIndex]) {
                    tableNameFound = true;
                    break;
                }
            }
            if (!tableNameFound) { 
                //throw runtime_error ("Catalog error: Column table reference is not in table list");
                throw runtime_error ("MAPD_ERR_COL_TABLE_REF_NOT_IN_TABLE_LIST");
            }
            // Note that tableIndex should be the index of the table that was
            // found
            int tableId = tableIds[tableIndex];
            ColumnKey columnKey(tableId, colNameIt -> second);
            colDescIt = columnDescriptorMap_.find(columnKey); 
        }
        if (colDescIt == columnDescriptorMap_.end()) {
            //throw runtime_error ("Catalog error: Column does not exist");
            throw runtime_error ("MAPD_ERR_COLUMN_DOES_NOT_EXIST");
        }
        columnDescriptors.push_back(colDescIt -> second);
    } 
    return columnDescriptors;
}

} // Catalog_Namespace
