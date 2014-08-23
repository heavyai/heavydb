#include "Catalog.h"
#include <fstream>
#include <set>
#include <iostream>

using std::string;
using std::map;
using std::tuple;
using std::vector;
using std::set;
using std::ifstream;
using std::ofstream;
using std::pair;

namespace Metadata_Namespace {

Catalog::Catalog(const string &basePath): basePath_(basePath), maxTableId_(-1), maxColumnId_(-1), isDirty_(false), pgConnector_("mapd","mapd") {
    //readCatalogFromFile();
    createStateTableIfDne();
    readState();
}

Catalog::~Catalog() {
    // first flush in-memory representation of Catalog to file
    //writeCatalogToFile();
    writeState();

    // must clean up heap-allocated TableRow and ColumnRow structs
    for (TableRowMap::iterator tableRowIt = tableRowMap_.begin(); tableRowIt != tableRowMap_.end(); ++tableRowIt)
        delete tableRowIt -> second;

    for (ColumnRowMap::iterator columnRowIt = columnRowMap_.begin(); columnRowIt != columnRowMap_.end(); ++columnRowIt)
        delete columnRowIt -> second;
}

mapd_err_t Catalog::readCatalogFromFile() {
    string tableFileFullPath (basePath_ + "/tables.cat");
    ifstream tableFile (tableFileFullPath.c_str());
    // read in table file if it exists
    if (tableFile) {
        // file exists and is open for input
        // expects tab-delimited file
        string tempTableName;
        int tempTableId;
        while (tableFile >> tempTableName >> tempTableId) {
           tableRowMap_[tempTableName]  = new TableRow(tempTableName,tempTableId); // should be no need to check this for errors as we check for table entries of same name on insert
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
            int tempmapd_data_t;
            bool tempNotNull;
            //while (columnFile >> tempTableId >> tempColumnName >> tempColumnId >> tempmapd_data_t >> tempNotNull) { 
            while (columnFile >> tempTableId >> tempColumnName >> tempColumnId >> tempmapd_data_t >> tempNotNull) {
                ColumnKey columnKey (tempTableId, tempColumnName); // construct the tuple that will serve as key for this entry into the column map
                columnRowMap_[columnKey] = new ColumnRow(tempTableId, tempColumnName, tempColumnId, static_cast<mapd_data_t>(tempmapd_data_t), static_cast<bool> (tempNotNull));
                //columnRowMap_[columnKey] = new ColumnRow(tempTableId, tempColumnName, tempColumnId, tempmapd_data_t, tempNotNull);
               // If this column has an id higher than maxColumnId_, set maxColumnId_ to it
               if (tempColumnId > maxColumnId_)
                   maxColumnId_ = tempColumnId;
            }
            columnFile.close();
        }
    }

    return MAPD_SUCCESS;
}

mapd_err_t Catalog::readState() {
    string tableQuery("SELECT table_name, table_id from tables ORDER BY table_id");
    mapd_err_t status = pgConnector_.query(tableQuery);
    assert(status == MAPD_SUCCESS);
    size_t numRows = pgConnector_.getNumRows();
    for (int r = 0; r < numRows; ++r) {
        string tableName = pgConnector_.getData<string>(r,0);
        int tableId = pgConnector_.getData<int>(r,1);
        tableRowMap_[tableName] = new TableRow(tableName,tableId);
        if (tableId > maxTableId_)
            maxTableId_ = tableId;
    }
    string columnQuery("SELECT table_id, column_name, column_id, column_type, not_null from columns ORDER BY table_id,column_id");
    status = pgConnector_.query(columnQuery);
    numRows = pgConnector_.getNumRows();
    for (int r = 0; r < numRows; ++r) {
        int tableId = pgConnector_.getData<int>(r,0);
        string columnName = pgConnector_.getData<string>(r,1);
        int columnId = pgConnector_.getData<int>(r,2);
        mapd_data_t columnType = getTypeFromString(pgConnector_.getData<string>(r,3));
        bool notNull = pgConnector_.getData<bool>(r,4);
        ColumnKey columnKey(tableId,columnName);
        columnRowMap_[columnKey] = new ColumnRow(tableId,columnName,columnId,columnType,notNull);
        if (columnId > maxColumnId_)
            maxColumnId_ = columnId;
    }
    return MAPD_SUCCESS;
}


void Catalog::createStateTableIfDne() {
     mapd_err_t status = pgConnector_.query("CREATE TABLE IF NOT EXISTS tables(table_name TEXT, table_id INT UNIQUE, PRIMARY KEY (table_name))");
     assert(status == MAPD_SUCCESS);
     status = pgConnector_.query("CREATE TABLE IF NOT EXISTS columns(table_id INT, column_name TEXT, column_id INT, column_type TEXT, not_null boolean, PRIMARY KEY (table_id, column_name))");
     assert(status == MAPD_SUCCESS);
}

mapd_err_t Catalog::writeState() {
    if (isDirty_) {
        // we will just overwrite table and column files with all TableRows and ColumnRows
        string deleteTableQuery ("DELETE FROM tables");
        mapd_err_t status = pgConnector_.query(deleteTableQuery);
        assert(status == MAPD_SUCCESS);
        string deleteColumnQuery ("DELETE FROM columns");
        status = pgConnector_.query(deleteColumnQuery);
        assert(status == MAPD_SUCCESS);
        for (TableRowMap::iterator tableRowIt = tableRowMap_.begin(); tableRowIt != tableRowMap_.end(); ++tableRowIt) {
            TableRow *tableRow = tableRowIt -> second;
            string insertTableQuery("INSERT INTO tables (table_name, table_id) VALUES ('" + tableRow -> tableName + "'," + boost::lexical_cast<string>(tableRow -> tableId) + ")");
            status = pgConnector_.query(insertTableQuery);
            assert (status == MAPD_SUCCESS);
        }
        for (ColumnRowMap::iterator columnRowIt = columnRowMap_.begin(); columnRowIt != columnRowMap_.end(); ++columnRowIt) {
            ColumnRow *columnRow = columnRowIt -> second;
            string insertColumnQuery("INSERT INTO columns (table_id, column_name, column_id, column_type, not_null) VALUES (" + boost::lexical_cast<string>(columnRow -> tableId)  + ",'" + columnRow -> columnName + "'," + boost::lexical_cast<string>(columnRow -> columnId) + ",'" + getTypeName(columnRow -> columnType) + "'," + (columnRow -> notNull == true ? "true" : "false") + ")" );
            status = pgConnector_.query(insertColumnQuery);
            assert (status == MAPD_SUCCESS);
        }
    }
    isDirty_ = false;
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::writeCatalogToFile() {
    if (isDirty_) {
        // we will just overwrite table and column files with all TableRows and ColumnRows
        string tableFileFullPath (basePath_ + "/tables.cat");
        ofstream tableFile (tableFileFullPath.c_str());
        if (tableFile.is_open()) {
            for (TableRowMap::iterator tableRowIt = tableRowMap_.begin(); tableRowIt != tableRowMap_.end(); ++tableRowIt) {
                TableRow *tableRow = tableRowIt -> second;
                tableFile << tableRow -> tableName << "\t" << tableRow -> tableId << "\n";
            }
            tableFile.flush();
            tableFile.close();
            // we only try to write to the column file if we've succeeded at writing to the table file
            string columnFileFullPath (basePath_ + "/columns.cat");
            ofstream columnFile (columnFileFullPath.c_str());
            if (columnFile.is_open()) {
                for (ColumnRowMap::iterator columnRowIt = columnRowMap_.begin(); columnRowIt != columnRowMap_.end(); ++columnRowIt) {
                    ColumnRow *columnRow = columnRowIt -> second;
                    columnFile << columnRow -> tableId << "\t" << columnRow -> columnName << "\t" << columnRow -> columnId << "\t" << columnRow -> columnType << "\t" << columnRow -> notNull << "\n";
                }
            }
            else
                return MAPD_ERR_FILE_OPEN;
        }
        else 
            return MAPD_ERR_FILE_OPEN;

        //finally set dirty flag to false so catalog doesn't need to be rewritten until next modification
        isDirty_ = false;
    }
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::addTable(const string &tableName) {
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt != tableRowMap_.end())
        return MAPD_ERR_TABLE_ALREADY_EXISTS;

    // if here then table did not exist and we can insert it into tableRowMap_
    // Create tableRow and pre-increment nextTableId_ so next TableRow gets id one higher than current max
    TableRow *tableRow = new TableRow(tableName,++maxTableId_); 
    tableRowMap_[tableName] = tableRow;

    isDirty_ = true;
    return MAPD_SUCCESS;
}


mapd_err_t Catalog::addTableWithColumns(const string &tableName, vector <ColumnRow *> & columns) { 
     //note that vector of ColumnRows should not be populated with tableId and columnIds - the catalog fills these in.
     //
    // first need to check if insert would result in any errors
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt != tableRowMap_.end())
        return MAPD_ERR_TABLE_ALREADY_EXISTS;
    set <string> columnNames;
    for (vector <ColumnRow *>::iterator colIt = columns.begin(); colIt != columns.end(); ++colIt) {
    if (columnNames.insert((*colIt) -> columnName).second == false) // ntests if we have already specified a column with the same name
        return MAPD_ERR_COLUMN_ALREADY_EXISTS; 
    }

    // if we reached this far then we know table insert will succeed
    addTable(tableName); // would be nice to return tableId instead of error as this means we could get post-incremented columnId
    //if (status != MAPD_SUCCESS) // table insert failed because table with same name already existed
    //    return status; 
    int tableId = maxTableId_; // because tableId was pre-incremented on table insert
    for (vector <ColumnRow *>::const_iterator colIt = columns.begin(); colIt != columns.end(); ++colIt) {
        ColumnRow *columnRow = *colIt;
        columnRow -> tableId = tableId;
        columnRow -> columnId = ++maxColumnId_; // get next value of maxColumnId for columnId
         ColumnKey columnKey (tableId, columnRow -> columnName);       
         columnRowMap_[columnKey] = columnRow; // insertion of column
    isDirty_ = true;
    }
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::addColumnToTable(const string &tableName, ColumnRow * columnRow) {
     //note that columnRow should not be populated with tableId and columnIds - the database fills these in.
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt == tableRowMap_.end())
        return MAPD_ERR_TABLE_DOES_NOT_EXIST;

    int tableId = tableRowIt -> second -> tableId;
    ColumnKey columnKey (tableId, columnRow -> columnName);       
    ColumnRowMap::iterator colRowIt = columnRowMap_.find(columnKey);
    if (colRowIt != columnRowMap_.end())
        return MAPD_ERR_COLUMN_ALREADY_EXISTS;
    columnRow -> tableId = tableId;
    columnRow -> columnId = ++maxColumnId_; // get next value of maxColumnId for columnId
    columnRowMap_[columnKey] = columnRow; // insertion of column 
    isDirty_ = true;
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::removeTable(const string &tableName) {
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt == tableRowMap_.end())
        return MAPD_ERR_TABLE_DOES_NOT_EXIST;

    // if here then table does exist so we can remove it and its associated columns
    tableRowMap_.erase(tableRowIt);

    // can be multiple columns for the same table, so must iterate and delete each column that belongs to the table
    ColumnRowMap::iterator columnRowIt = columnRowMap_.begin();    
    while (columnRowIt != columnRowMap_.end()) {
        if (std::get<1>(columnRowIt -> first) == tableName)
            columnRowMap_.erase(columnRowIt++);
        else
            ++columnRowIt;
    }
    isDirty_ = true;
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::removeColumnFromTable(const string &tableName, const string &columnName) {
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt == tableRowMap_.end()) // check to make sure table exists
        return MAPD_ERR_TABLE_DOES_NOT_EXIST;
    int tableId = tableRowIt -> second -> tableId;
    ColumnKey columnKey (tableId, columnName);       
    ColumnRowMap::iterator colRowIt = columnRowMap_.find(columnKey);
    if (colRowIt == columnRowMap_.end()) // need to check to make sure column exists for table
        return MAPD_ERR_COLUMN_DOES_NOT_EXIST;
    columnRowMap_.erase(colRowIt);
    isDirty_ = true;
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::getMetadataForTable (const string &tableName, TableRow &tableRow) {
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt == tableRowMap_.end()) // check to make sure table exists
        return MAPD_ERR_TABLE_DOES_NOT_EXIST;
    tableRow = *(tableRowIt -> second); 
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::getMetadataForColumn (const string &tableName, const string &columnName,  ColumnRow &columnRow) {
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt == tableRowMap_.end()) // check to make sure table exists
        return MAPD_ERR_TABLE_DOES_NOT_EXIST;
    int tableId = tableRowIt -> second -> tableId;
    ColumnKey columnKey (tableId, columnName);       
    ColumnRowMap::iterator colRowIt = columnRowMap_.find(columnKey);
    if (colRowIt == columnRowMap_.end()) // need to check to make sure column exists for table
        return MAPD_ERR_COLUMN_DOES_NOT_EXIST;
    columnRow = *(colRowIt -> second); // will invoke implicit copy constructor - otherwise a pointer given to the caller might be subsequently invalidated by the Catalog in a multithreaded environment
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::getMetadataForColumns (const string &tableName, const vector<string> &columnNames,  vector <ColumnRow> &columnRows) {
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt == tableRowMap_.end()) // check to make sure table exists
        return MAPD_ERR_TABLE_DOES_NOT_EXIST;
    int tableId = tableRowIt -> second -> tableId;
    for (vector<string>::const_iterator colNameIt = columnNames.begin(); colNameIt != columnNames.end(); ++colNameIt) {
        ColumnKey columnKey (tableId, *colNameIt);
        ColumnRowMap::iterator colRowIt = columnRowMap_.find(columnKey);
        if (colRowIt ==  columnRowMap_.end()) 
            return MAPD_ERR_COLUMN_DOES_NOT_EXIST;
        columnRows.push_back(*(colRowIt -> second));
    }
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::getAllColumnMetadataForTable(const string &tableName, vector <ColumnRow> &columnRows) {
    auto tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt == tableRowMap_.end()) // check to make sure table exists
        return MAPD_ERR_TABLE_DOES_NOT_EXIST;
    int tableId = tableRowIt -> second -> tableId;
    getAllColumnMetadataForTable(tableId, columnRows);
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::getAllColumnMetadataForTable(const int tableId, vector <ColumnRow> &columnRows) {
    for (auto colRowIt = columnRowMap_.begin(); colRowIt != columnRowMap_.end(); ++colRowIt) {
        if (colRowIt -> second -> tableId == tableId) {
            columnRows.push_back(*(colRowIt -> second));
        }
    }
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::getMetadataForColumns(const vector <string>  &tableNames, const vector <pair <string, string> > &columnNames, vector <ColumnRow> &columnRows) {
    //map <string, int> tableIdMap;
    vector <int> tableIds;
    for (auto tableNameIt = tableNames.begin(); tableNameIt != tableNames.end(); ++tableNameIt) {
        auto tableRowIt = tableRowMap_.find(*tableNameIt);
        if (tableRowIt == tableRowMap_.end())
            return MAPD_ERR_TABLE_DOES_NOT_EXIST;
        tableIds.push_back(tableRowIt -> second -> tableId);
    }
    // size_t numTables = tableIds.size();
    
    // If here then all tables exist
    for (auto colNameIt = columnNames.begin(); colNameIt != columnNames.end(); ++colNameIt) {
        string tableName (colNameIt -> first);
        ColumnRowMap::iterator colRowIt = columnRowMap_.end(); // set this to end at first to signify column not found yet
        if (tableName.size() == 0) { // no explicit table reference
            for (auto tableIdIt = tableIds.begin(); tableIdIt != tableIds.end(); ++tableIdIt) {
                ColumnKey columnKey (*tableIdIt, colNameIt -> second);
                auto tempColRowIt = columnRowMap_.find(columnKey);
                if (tempColRowIt !=  columnRowMap_.end()) { 
                    if (colRowIt != columnRowMap_.end()) // if we've already found the column
                        return MAPD_ERR_COLUMN_IS_AMBIGUOUS;
                    colRowIt = tempColRowIt;
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
            if (!tableNameFound) 
                return MAPD_ERR_COL_TABLE_REF_NOT_IN_TABLE_LIST;
            // Note that tableIndex should be the index of the table that was
            // found
            int tableId = tableIds[tableIndex];
            ColumnKey columnKey(tableId, colNameIt -> second);
            colRowIt = columnRowMap_.find(columnKey); 
        }
        if (colRowIt == columnRowMap_.end())
            return MAPD_ERR_COLUMN_DOES_NOT_EXIST;
        columnRows.push_back(*(colRowIt -> second));
    }
    return MAPD_SUCCESS;
}

} // Metadata_Namespace

