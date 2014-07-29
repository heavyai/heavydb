/**
 * @file	insertWalkerTest.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Used for testing the insert walker. In order to test the walker,
 * enter INSERT statements at the parser's prompt.
 */
#include <iostream>
#include <string>
#include <vector>
#include "../../Shared/types.h"
#include "../Parse/SQL/parser.h"
 #include "InsertWalker.h"
#include "../../DataMgr/Metadata/Catalog.h"

using namespace std;
using Analysis_Namespace::InsertWalker;

int main(int argc, char ** argv) {
    // Add a table to the catalog
    Catalog c(".");
    
    std::vector<ColumnRow*> cols;
    cols.push_back(new ColumnRow("a", INT_TYPE, true));
    cols.push_back(new ColumnRow("b", INT_TYPE, true));

    c.addTableWithColumns("T1", cols);

    // Create a parser for SQL and... do stuff
    SQLParser parser;
    string sql;
    do {
        cout << "mapd> ";
        getline(cin,sql);
        if (sql == "q")
            break;
        else sql = sql + "\n";

        ASTNode *parseRoot = 0;
        string lastParsed;
        int numErrors = parser.parse(sql, parseRoot,lastParsed);
        if (numErrors > 0) {
            cout << "Error at: " << lastParsed << endl;
            continue;
        }
        if (numErrors > 0)
            cout << "# Errors: " << numErrors << endl;
        if (parseRoot == NULL) printf("parseRoot is NULL\n");
        InsertWalker iw(c);
        if (parseRoot != 0)
            parseRoot->accept(iw); 
    }
    while(true);
    cout << "Good-bye." << endl;
}
