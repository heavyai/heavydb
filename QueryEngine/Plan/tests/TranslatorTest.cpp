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

using namespace std;
using namespace Plan_Namespace;
using namespace SQL_Namespace;

int main() {
    SQLParser parser;
    string sql;
    
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
        Translator tr;
        RA_Namespace::RelAlgNode *queryPlanRoot = tr.translate(parseRoot);
        assert(queryPlanRoot);
        
        // print out XML representation of the RA query plan
        RA_Namespace::XMLTranslator ra2xml;
        queryPlanRoot->accept(ra2xml);
    }
    
    while (true);
    cout << "Good-bye." << endl;

}
