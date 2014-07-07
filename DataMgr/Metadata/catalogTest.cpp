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
#include <stdio.h>

#include "../../Shared/errors.h"
#include "../../Shared/ansi.h"
#include "../../Shared/testing.h"

using namespace Testing;
using namespace std;

//bool test_Catalog() {

bool test_Catalog();
bool test_AddTable();

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
    printTestSummary();

    return EXIT_SUCCESS;

}

void reset() {
    remove ("tables.cat");
    remove ("columns.cat");
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
