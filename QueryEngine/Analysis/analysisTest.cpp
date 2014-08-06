/**
 * @file	analysisTest.h
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Used for interactively testing the semantic analysis modules.
 */
#include <iostream>
#include <string>
#include <vector>
#include "Analysis.h"
#include "../../Shared/types.h"
#include "../Parse/SQL/parser.h"
#include "../../DataMgr/Metadata/Catalog.h"
//#include "../../DataMgr/Partition/TablePartitionMgr.h"

using namespace std;
using Analysis_Namespace::TypeChecker;

int main(int argc, char ** argv) {
    Catalog c(".");
    
    std::vector<ColumnRow*> cols;
    cols.push_back(new ColumnRow("a", INT_TYPE, true));
    cols.push_back(new ColumnRow("b", FLOAT_TYPE, true));
    
    mapd_err_t err = c.addTableWithColumns("T1", cols);
    if (err != MAPD_SUCCESS) {
        printf("[%s:%d] Catalog::addTableWithColumns: err = %d\n", __FILE__, __LINE__, err);
        //exit(EXIT_FAILURE);
    }

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
        
        // Check INSERT statements
        //InsertData idata;
        /*std::pair<bool, std::string> insertErr = Analysis_Namespace::checkInsert(parseRoot, c, &i);
        if (insertErr.first == true)
            cout << "Error: " << insertErr.second << std::endl;

        // Check SQL statements
        std::pair<bool, std::string> sqlErr = Analysis_Namespace::checkSql(parseRoot);
        if (sqlErr.first == true)
            cout << "Error: " << sqlErr.second << std::endl;
        */
    }
    while(true);
    cout << "Good-bye." << endl;
}
