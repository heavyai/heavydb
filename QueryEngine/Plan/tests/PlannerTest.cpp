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
#include "../../Parse/RA/visitor/XMLTranslator.h"

using namespace std;
using namespace Plan_Namespace;
using namespace Metadata_Namespace;
using namespace Plan_Namespace;

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
        QueryStmtType stmtType = UNKNOWN_STMT;
        AbstractPlan *thePlan = planner.makePlan(sql, stmtType);
        
        std::pair<bool, std::string> error = planner.checkError();

        if (error.first)
            cout << error.second << endl;
        else {
            // print for debugging
            if (stmtType == QUERY_STMT) {
                XMLTranslator ra2xml;
                ((RelAlgNode*)((QueryPlan*)thePlan)->getPlan())->accept(ra2xml);
            }
            else if (stmtType == INSERT_STMT) {
                // execute the insert plan
                thePlan->execute();
            }
        }
        
    } while (true);
}
