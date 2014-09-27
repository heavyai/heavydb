/**
 * @file    TranslatorTest.cpp
 * @author  Steven Stewart <steve@map-d.com>
 *
 * An interactive program for testing the translation of SQL to an RA query plan tree.
 */
#include <iostream>
#include <string>
#include "../../Parse/SQL/parser.h"
#include "../../Parse/SQL/visitor/Visitor.h"
#include "../Translator.h"
#include "../../Parse/RA/visitor/XMLTranslator.h"
#include "../../../DataMgr/Metadata/Catalog.h"

using namespace std;
using namespace Plan_Namespace;
using namespace SQL_Namespace;

int main() {
    SQLParser parser;
    string sql;
    
    // set up Catalog with table T1 and columns 'a' and 'b'
    Catalog catalog(".");

    std::vector<ColumnRow*> cols;
    cols.push_back(new ColumnRow("a", INT_TYPE, true));
    cols.push_back(new ColumnRow("b", FLOAT_TYPE, true));

    mapd_err_t err = catalog.addTableWithColumns("t1", cols);
    if (err != MAPD_SUCCESS)
        printf("[%s:%d] Catalog::addTableWithColumns: err = %d\n", __FILE__, __LINE__, err);
    
    do {
        // obtain user input
        cout << "mapd> ";
        getline(cin,sql);
        if (sql == "q")
            break;
        else sql = sql + "\n";
        
        // parse user input
        ASTNode *parseRoot = 0;
        string lastParsed;
        int numErrors = parser.parse(sql, parseRoot,lastParsed);
        if (numErrors > 0) {
            cout << "Error at: " << lastParsed << endl;
            continue;
        }
        if (numErrors > 0)
            cout << "# Errors: " << numErrors << endl;
        
        // translate the SQL parse tree into an RA query plan
        Translator tr(catalog);
        RA_Namespace::RelAlgNode *queryPlanRoot = tr.translate(parseRoot);
        
        std::pair<bool, std::string> catalogErr = tr.checkError();
        if (catalogErr.first) {
            cout << catalogErr.second << endl;
            continue;
        }
        
        // assert(queryPlanRoot);
        
        // print out XML representation of the RA query plan
        RA_Namespace::XMLTranslator ra2xml;
        queryPlanRoot->accept(ra2xml);
    }
    
    while (true);
    cout << "Good-bye." << endl;

}
