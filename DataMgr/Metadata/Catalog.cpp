#include "Catalog.h"

using namespace std::string;
using namespace std::map;
using namespace std::tuple;
using namespace std::vector;

Catalog::Catalog(const string &basePath): basePath_(basePath), maxTableId_(-1), isDirty_(false) {
    readCatalogFromFile();
}

Catalog::~Catalog() {
    // first flush in-memory representation of Catalog to file
    writeCatalogToFile();

    // must clean up heap-allocated TableRow and ColumnRow structs
    for (TableRowMap::iterator tableRowIt = tableRowMap.begin(); tableRowIt != tableRowMap.end(); ++tableRowIt)
        delete tableRowIt -> second;

    for (ColumnRowMap::iterator columnRowIt = columnRowMap.begin(); columnRowIt != columnRowMap.end(); ++columnRowIt)
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
        while (tablefile >> tempTableName >> tempTableId) {
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
            ColumnType tempColumnType;
            bool tempNotNull;
            while (columnFile >> tempTableId >> tempColumnName >> tempColumnId >> tempColumnType >> tempNotNull) {
                ColumnKey columnKey (tempTableId, tempColumnName); // construct the tuple that will serve as key for this entry into the column map
                columnRowMap_[columnKey] = new ColumnRow(tempTableId, tempColumnName, tempColumnId, tempColumnType, tempNotNull);
               // If this column has an id higher than maxColumnId_, set maxColumnId_ to it
               if (tempColumnId > maxColumnId_)
                   maxColumnId_ = tempColumnId;
            }
            columnFile.close();
        }
    }

    return MAPD_SUCCESS;
}

mapd_err_t Catalog::writeCatalogToFile() {
    if (!isDirty_) {
        // we will just overwrite table and column files with all TableRows and ColumnRows
        string tableFileFullPath (basePath_ + "/tables.cat");
        ofstream tableFile (tableFileFullPath.c_str());
        if (tableFile.is_open()) {
            for (TableRowMap::iterator tableRowIt = tableRowMap.begin(); tableRowIt != tableRowMap.end(); ++tableRowIt) {
                TableRow *tableRow = tableRowIt -> second;
                tableFile << tableRow -> tableName << "\t" << tableRow -> tableId << "\n";
            }
            tableFile.close();
            // we only try to write to the column file if we've succeeded at writing to the table file
            string columnFileFullPath (basePath_ + "/columns.cat");
            ofstream columnFile (tableFileFullPath.c_str());
            if (columnFile.is_open()) {
                for (ColumnRowMap::iterator columnRowIt = columnRowMap.begin(); columnRowIt != columnRowMap.end(); ++columnRowIt) {
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
     //note that vector of ColumnRows should not be populated with tableId and columnIds - the database fills these in.
     //
    // first need to check if insert would result in any errors
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt != tableRowMap_.end())
        return MAPD_ERR_TABLE_ALREADY_EXISTS;
    set <string> columnNames;
    for (vector <ColumnRow *>::iterator colIt = columns.begin(); colIt != columns.end(); ++colIt) {
    if (columnNames.insert(colIt -> second -> columnName).second == false) // ntests if we have already specified a column with the same name
        return MAPD_ERR_COLUMN_ALREADY_EXISTS; 
    }

    // if we reached this far then we know table insert will succeed
    addTable(tableName); // would be nice to return tableId instead of error as this means we could get post-incremented columnId
    //if (status != MAPD_SUCCESS) // table insert failed because table with same name already existed
    //    return status; 
    int tableId = maxTableId_; // because tableId was pre-incremented on table insert
    for (vector <ColumnRow *>::const_iterator colIt = columns.begin(); colIt != columns.end(); ++colIt) {
        ColumnRow *columnRow = colIt -> second;
        columnRow -> tableId = tableId;
        columnRow -> columnId = ++maxColumnId_; // get next value of maxColumnId for columnId
         ColumnKey columnKey (tableId, colIt -> columnName);       
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
    ColumnKey columnKey (tableId, columnRow -> columnId);       
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
        if (std::get<0>(columnRowIt.first) == tableName)
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
    ColumnKey columnKey (tableId, columnRow -> columnName);       
    ColumnRowMap::iterator colRowIt = columnRowMap_.find(columnKey);
    if (colRowIt == columnRowMap_.end()) // need to check to make sure column exists for table
        return MAPD_ERR_COLUMN_DOES_NOT_EXIST;
    columnRowMap_.erase(colRowIt);
    isDirty_ = true;
    return MAPD_SUCCESS:
}

mapd_err_t Catalog::getMetadataforColumn (const string &tableName, const string &columnName,  ColumnRow &columnRow) {
    TableRowMap::iterator tableRowIt = tableRowMap_.find(tableName);
    if (tableRowIt == tableRowMap_.end()) // check to make sure table exists
        return MAPD_ERR_TABLE_DOES_NOT_EXIST;
    int tableId = tableRowIt -> second -> tableId;
    ColumnKey columnKey (tableId, columnRow -> columnName);       
    ColumnRowMap::iterator colRowIt = columnRowMap_.find(columnKey);
    if (colRowIt == columnRowMap_.end()) // need to check to make sure column exists for table
        return MAPD_ERR_COLUMN_DOES_NOT_EXIST;
    columnRow = *(colRowIt -> second); // will invoke implicit copy constructor - otherwise a pointer given to the caller might be subsequently invalidated by the Catalog in a multithreaded environment
    return MAPD_SUCCESS;
}

mapd_err_t Catalog::getMetadataforColumns (const string &tableName, const vector<string> &columnNames,  vector <ColumnRow> &columnRows) {
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


        


        




}



