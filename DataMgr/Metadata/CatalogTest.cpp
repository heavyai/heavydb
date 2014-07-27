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

//bool test_Catalog() {

bool test_Catalog();
bool test_AddTable();
bool test_AddTableWithColumns();
bool test_removeTable();

//helper methods
void reset(); // deletes Catalog's files to reset state

int main(void ) {

    reset();
    test_Catalog ?
        PPASS("Catalog()") : PFAIL("Catalog()");
    reset();
    test_AddTable() ?
        PPASS("Catalog::addTable(tableName)") : PFAIL("Catalog::addTable(tableName)");
    reset();
    test_AddTableWithColumns() ?
        PPASS("Catalog::addTableWithColumns(tableName, columns)") : PFAIL("Catalog::addTableWithColumns(tableName, columns)");
    //reset();
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
    mapd_err_t status = catalog.addTable("test");
    if (status != MAPD_SUCCESS) 
        return false;
	Testing::pass++;

    status = catalog.addTable("test");
    if (status == MAPD_SUCCESS) 
        return false;
	Testing::pass++;

    status = catalog.writeCatalogToFile();
    if (status != MAPD_SUCCESS) 
        return false;
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

bool test_AddTableWithColumns() {
    {
        Catalog catalog(".");
        {
            vector <ColumnRow *> columnRows;
            columnRows.push_back(new ColumnRow("a", INT_TYPE, true));
            columnRows.push_back(new ColumnRow("b", INT_TYPE, false));
            columnRows.push_back(new ColumnRow("c", INT_TYPE, false));
            mapd_err_t status = catalog.addTableWithColumns("test1", columnRows);
            if (status != MAPD_SUCCESS) 
                return false;
            Testing::pass++;
        }

        {
            vector <ColumnRow *> columnRows;
            columnRows.push_back(new ColumnRow("a", INT_TYPE, true));
            columnRows.push_back(new ColumnRow("b", INT_TYPE, false));
            columnRows.push_back(new ColumnRow("c", INT_TYPE, false));
            mapd_err_t status = catalog.addTableWithColumns("test2", columnRows);
            if (status != MAPD_SUCCESS) 
                return false;
            Testing::pass++;
        }


        mapd_err_t status = catalog.addTable("test1");
        if (status == MAPD_SUCCESS) 
            return false;
        Testing::pass++;
    }

    //close catalog and open again to see if we stored data correctly
    Catalog catalog(".");
    vector <string> columnNames {"a","b","c"};
    vector <ColumnRow> columnRows;
    mapd_err_t status = catalog.getMetadataForColumns("test1", columnNames, columnRows); 
    if (status != MAPD_SUCCESS)
        return false;
    if (columnRows.size() != 3)
        return false;
    set <string> colNameSet; 
    set <int> tableIdSet;
    set <int> columnIdSet;
    for (int c = 0; c != 3; ++c) {
        colNameSet.insert(columnRows[c].columnName);
        tableIdSet.insert(columnRows[c].tableId);
        columnIdSet.insert(columnRows[c].columnId);
        if (columnRows[c].columnName == "a") {
            if (columnRows[c].columnType != INT_TYPE || columnRows[c].notNull == false)
                return false;
        }
        else if (columnRows[c].columnName == "b") {
            if (columnRows[c].columnType != INT_TYPE || columnRows[c].notNull == true)
                return false;
        }
        else if (columnRows[c].columnName == "c") {
            if (columnRows[c].columnType != INT_TYPE || columnRows[c].notNull == true)
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
}


