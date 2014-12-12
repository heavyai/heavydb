/**
 * @file catalogTest.cpp
 * @author Todd Mostak <todd@map-d.com>
 *
 * This program is used to test the Catalog class.
 *
 * @see Catalog
 */

#include "Catalog.h"
#include <exception>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>
#include <stdio.h>

#include "../../Shared/errors.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"
#include "../../Shared/types.h"

using namespace Testing;
using namespace std;
using namespace Catalog_Namespace;

//bool test_Catalog() {

bool test_Catalog();
bool test_AddTable();
bool test_AddTableWithColumns();
bool test_GetMultiColumnMetadata();
bool test_removeTable();
bool test_GetAllColumnMetadataForTable();

//helper methods
void reset(); // deletes Catalog's files to reset state

int main(void ) {

    reset();
    test_Catalog() ?
        PPASS("Catalog()") : PFAIL("Catalog()");
    reset();
    test_AddTable() ?
        PPASS("Catalog::addTable(tableName)") : PFAIL("Catalog::addTable(tableName)");
    reset();
    test_AddTableWithColumns() ?
        PPASS("Catalog::addTableWithColumns(tableName, columns)") : PFAIL("Catalog::addTableWithColumns(tableName, columns)");
    reset();
    test_GetMultiColumnMetadata() ?
        PPASS("Catalog::getMetadataForColumns(tableNames, columnNames, columnRows)") : PFAIL("Catalog::getMetadataForColumns(tableNames, columnNames, columnRows)");
    reset();
    test_GetAllColumnMetadataForTable() ? 
        PPASS("Catalog::getAllColumnMetadataForTable(tableName,columnRows)"): PFAIL("Catalog::getAllColumnMetadataForTable(tableName, columnRows)");

    printTestSummary();

    return EXIT_SUCCESS;

}


bool test_Catalog() {
    try {
        Catalog *catalog = new Catalog(".");
        if (!catalog)
            return false;
        delete catalog;
    }
    catch (std::exception &e) {
        return false;
    }
    return true;
}



bool test_AddTable() {
    Catalog catalog(".");
    try {
        catalog.addTable("test");
    }
    catch (...) {
        return false;
    }
	Testing::pass++;
    try {
        catalog.addTable("test");
        return false; // b/c above should have thrown exception
    }
    catch (...) {
    }
	Testing::pass++;

    try {
        catalog.writeCatalogToFile();
    }
    catch (...) {
        return false;
    }
	Testing::pass++;

    ifstream tableFile("tables.cat");
    if (tableFile) {
        string tableName;
        int tableId;
        if (tableFile >> tableName >> tableId) {
            if (tableName != "test" || tableId != 0)
                return false;
        } 
        else
            return false;
        tableFile.close();
    }
    else
        return false;
	Testing::pass++;
    return true;
}

bool test_GetMultiColumnMetadata() {

        Catalog catalog(".");
        {
            vector <ColumnDescriptor> columnDescriptors;
            columnDescriptors.push_back(ColumnDescriptor("a", INT_TYPE, true));
            columnDescriptors.push_back(ColumnDescriptor("b", INT_TYPE, false));
            columnDescriptors.push_back(ColumnDescriptor("c", INT_TYPE, false));
            try {
                catalog.addTableWithColumns("test1", columnDescriptors);
            }
            catch (...) {
                return false;
            }
        }

        {
            vector <ColumnDescriptor> columnDescriptors;
            columnDescriptors.push_back(ColumnDescriptor("a", INT_TYPE, true));
            columnDescriptors.push_back(ColumnDescriptor("d", INT_TYPE, false));
            columnDescriptors.push_back(ColumnDescriptor("e", INT_TYPE, false));
            try {
                catalog.addTableWithColumns("test2", columnDescriptors);
            }
            catch (...) {
                return false;
            }
        }

        {
            vector <string> tableNames {"test1", "test2"};
            vector <pair<string,string> > columnNames {make_pair("", "a"), make_pair("", "d"), make_pair("" ,"e")};
            vector <const ColumnDescriptor *> columnDescriptors;
            try {
                columnDescriptors = catalog.getMetadataForColumns(tableNames, columnNames); 
                return false;
            }
            catch (runtime_error &error) {
                string errorString (error.what());
                if (errorString != "MAPD_ERR_COLUMN_IS_AMBIGUOUS") {
                    return false;
                }
            }
            Testing::pass++;
        }

        {
            vector <string> tableNames {"test1"};
            vector <pair<string,string> > columnNames {make_pair("", "a"), make_pair("test2", "d"), make_pair("test2" ,"e")};
            vector <const ColumnDescriptor * > columnDescriptors;
            try {
                columnDescriptors = catalog.getMetadataForColumns(tableNames, columnNames); 
                return false; // should have throw exception
            }
            catch (runtime_error &error) {
                string errorString (error.what());
                if (errorString != "MAPD_ERR_COL_TABLE_REF_NOT_IN_TABLE_LIST") {
                    return false;
                }
            }
            Testing::pass++;
        }

        {
            vector <string> tableNames {"test1", "test3"};
            vector <pair<string,string> > columnNames {make_pair("test1", "a"), make_pair("", "d"), make_pair("" ,"e")};
            vector <const ColumnDescriptor *> columnDescriptors;
            try {
                columnDescriptors = catalog.getMetadataForColumns(tableNames, columnNames); 
                return false;
            }
            catch (runtime_error &error) {
                string errorString (error.what());
                if (errorString != "MAPD_ERR_TABLE_DOES_NOT_EXIST") {
                    return false;
                }
            }
            Testing::pass++;
        }
        
        {
            vector <string> tableNames {"test1", "test2"};
            vector <pair<string,string> > columnNames {make_pair("test1", "a"), make_pair("test2", "f"), make_pair("" ,"e")};
            vector <const ColumnDescriptor *> columnDescriptors;
            try {
                columnDescriptors = catalog.getMetadataForColumns(tableNames, columnNames); 
                return false;
            }
            catch (runtime_error &error) {
                string errorString(error.what());
                if (errorString != "MAPD_ERR_COLUMN_DOES_NOT_EXIST") {
                    return false;
                }
            }
            Testing::pass++;
        }

        {
            vector <string> tableNames {"test1", "test2"};
            vector <pair<string,string> > columnNames {make_pair("test1", "a"), make_pair("", "d"), make_pair("" ,"e")};
            vector <const ColumnDescriptor *> columnDescriptors;
            try {
                columnDescriptors = catalog.getMetadataForColumns(tableNames, columnNames); 
            }
            catch ( ... ) {
                return false;
            }
            Testing::pass++;
        }

        {
            vector <string> tableNames {"test1", "test2"};
            vector <pair<string,string> > columnNames {make_pair("test1", "a"), make_pair("test2", "d"), make_pair("test2" ,"e")};
            vector <const ColumnDescriptor *> columnDescriptors;
            try {
                columnDescriptors = catalog.getMetadataForColumns(tableNames, columnNames); 
            }
            catch ( ... ) {
                return false;
            }
            Testing::pass++;
        }

        return true;
}

bool test_GetAllColumnMetadataForTable() {

    Catalog catalog(".");
    {
        vector <ColumnDescriptor> columnDescriptors;
        columnDescriptors.push_back(ColumnDescriptor("a", INT_TYPE, true));
        columnDescriptors.push_back(ColumnDescriptor("b", INT_TYPE, false));
        columnDescriptors.push_back(ColumnDescriptor("c", INT_TYPE, false));
        try {
            catalog.addTableWithColumns("test1", columnDescriptors);
        }
        catch ( ... ) {
            return false;
        }
    }

    {
        vector <const ColumnDescriptor *> columnDescriptors;
        try {
            columnDescriptors = catalog.getAllColumnMetadataForTable("test2");  
            return false;
        }
        catch (runtime_error &error) {
            string errorString(error.what());
            if (errorString != "MAPD_ERR_TABLE_DOES_NOT_EXIST") {
                return false;
            }
        }
        Testing::pass++;
    }

    {
        vector <const ColumnDescriptor *> columnDescriptors;
        try {
            columnDescriptors = catalog.getAllColumnMetadataForTable("test1");  
        }
        catch ( ... ) {
            return false;
        }

        if (columnDescriptors.size() != 3) {
            return false;
        }

        set <string> colNameSet; 
        set <int> tableIdSet;
        set <int> columnIdSet;
        for (int c = 0; c != 3; ++c) {
            colNameSet.insert(columnDescriptors[c]->columnName);
            tableIdSet.insert(columnDescriptors[c]->tableId);
            columnIdSet.insert(columnDescriptors[c]->columnId);
            if (columnDescriptors[c]->columnName == "a") {
                if (columnDescriptors[c]->columnType != INT_TYPE || columnDescriptors[c]->notNull == false)
                    return false;
            }
            else if (columnDescriptors[c]->columnName == "b") {
                if (columnDescriptors[c]->columnType != INT_TYPE || columnDescriptors[c]->notNull == true)
                    return false;
            }
            else if (columnDescriptors[c]->columnName == "c") {
                if (columnDescriptors[c]->columnType != INT_TYPE || columnDescriptors[c]->notNull == true)
                    return false;
            }
            else // column was not a, b, or c
                return false;
        }
        if (colNameSet.size() != 3 || tableIdSet.size() != 1 || columnIdSet.size() != 3)
            return false;
        Testing::pass++;
    }

    return true;
}

bool test_AddTableWithColumns() {
    {
        Catalog catalog(".");
        {
            vector <ColumnDescriptor> columnDescriptors;
            columnDescriptors.push_back(ColumnDescriptor("a", INT_TYPE, true));
            columnDescriptors.push_back(ColumnDescriptor("b", INT_TYPE, false));
            columnDescriptors.push_back(ColumnDescriptor("c", INT_TYPE, false));
            try {
                catalog.addTableWithColumns("test1", columnDescriptors);
            }
            catch ( ... ) {
                return false;
            }
            Testing::pass++;
        }

        {
            vector <ColumnDescriptor> columnDescriptors;
            columnDescriptors.push_back(ColumnDescriptor("a", INT_TYPE, true));
            columnDescriptors.push_back(ColumnDescriptor("b", INT_TYPE, false));
            columnDescriptors.push_back(ColumnDescriptor("c", INT_TYPE, false));
            try {
                catalog.addTableWithColumns("test2", columnDescriptors);
            }
            catch ( ... ) {
                return false;
            }
            Testing::pass++;
        }



        try { 
            catalog.addTable("test1");
            return false;
        }
        catch ( ... ) {
        }
        Testing::pass++;
    }

    //close catalog and open again to see if we stored data correctly
    Catalog catalog(".");

    vector <string> columnNames {"a","b","c"};
    vector <const ColumnDescriptor * > columnDescriptors;
    try {
        columnDescriptors = catalog.getMetadataForColumns("test1", columnNames); 
    }
    catch ( ... ) {
        return false;
    }

    if (columnDescriptors.size() != 3)
        return false;

    set <string> colNameSet; 
    set <int> tableIdSet;
    set <int> columnIdSet;
    for (int c = 0; c != 3; ++c) {
        colNameSet.insert(columnDescriptors[c]->columnName);
        tableIdSet.insert(columnDescriptors[c]->tableId);
        columnIdSet.insert(columnDescriptors[c]->columnId);
        if (columnDescriptors[c]->columnName == "a") {
            if (columnDescriptors[c]->columnType != INT_TYPE || columnDescriptors[c]->notNull == false)
                return false;
        }
        else if (columnDescriptors[c]->columnName == "b") {
            if (columnDescriptors[c]->columnType != INT_TYPE || columnDescriptors[c]->notNull == true)
                return false;
        }
        else if (columnDescriptors[c]->columnName == "c") {
            if (columnDescriptors[c]->columnType != INT_TYPE || columnDescriptors[c]->notNull == true)
                return false;
        }
        else // column was not a, b, or c
            return false;
    }
    if (colNameSet.size() != 3 || tableIdSet.size() != 1 || columnIdSet.size() != 3)
        return false;
    Testing::pass++;
    return true;
}

//utility/helper methods

void reset() {
    remove ("tables.cat");
    remove ("columns.cat");
    try {
        SqliteConnector sqliteConnector ("mapd");
        sqliteConnector.query("drop table tables");
        sqliteConnector.query("drop table columns");
    }
    catch ( ... ) {

    }

}


