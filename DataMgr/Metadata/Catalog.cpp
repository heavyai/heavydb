#include "Catalog.h"

using namespace std::string;
using namespace std::map;
using namespace std::tuple;

Catalog::Catalog(const string &basePath): basePath_(basePath), isDirty_(false) {
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

