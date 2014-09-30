/**
 * @file    TranslatorTest.cpp
 * @author  Steven Stewart <steve@map-d.com>
 *
 * An interactive program for testing the translation of SQL to an RA query plan tree.
 */
#include <iostream>
#include <string>
#include "../Planner.h"
#include "../../../DataMgr/Metadata/Catalog.h"

using namespace std;
using namespace Plan_Namespace;
using namespace Metadata_Namespace;

int main() {
    // set up Catalog with table T1 and columns 'a' and 'b'
    Catalog catalog(".");
    
    std::vector<ColumnRow*> cols;
    cols.push_back(new ColumnRow("a", INT_TYPE, true));
    cols.push_back(new ColumnRow("b", FLOAT_TYPE, true));
    
    mapd_err_t err = catalog.addTableWithColumns("t1", cols);
    if (err != MAPD_SUCCESS)
        printf("[%s:%d] Catalog::addTableWithColumns: err = %d\n", __FILE__, __LINE__, err);

    Translator tr(catalog);
    Planner planner(tr);
    string sql;
    pair<int, string> error;
    
    do {
        // obtain user input
        cout << "mapd> ";
        getline(cin,sql);
        if (sql == "q")
            break;
        else sql = sql + "\n";
        
        // get query plan
        RelAlgNode *queryPlan = nullptr;
        QueryStmtType stmtType = UNKNOWN_STMT;
        error = planner.makePlan(sql, &queryPlan, stmtType);
        
        if (error.first > 0)
            cout << error.second << endl;

    } while (true);
}
