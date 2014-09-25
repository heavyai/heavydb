/**
 * @file	translateTest.cpp
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
#include "SQL_RA_Translator.h"
#include "../Parse/RA/visitor/XMLTranslator.h"
#include "../Parse/RA/visitor/QPTranslator.h"

using namespace std;
using Translate_Namespace::SQL_RA_Translator;

int main(int argc, char ** argv) {

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
        
        SQL_RA_Translator sql2ra;
        if (parseRoot != 0)
            parseRoot->accept(sql2ra);

        RA_Namespace::XMLTranslator ra_xml;
        printf("\nXML Output of translated Relational Algebra AST\n");
        printf("-----------------------------------------------\n");
        if (sql2ra.root != 0)
            sql2ra.root->accept(ra_xml);
        printf("\n\n");

        RA_Namespace::QPTranslator ra_qp;
        printf("\nQuery Plan output of translated Relational Algebra AST\n");
        printf("------------------------------------------------------\n");
        if (sql2ra.root != 0)
            sql2ra.root->accept(ra_qp);
        printf("\n\n");

    }
    while(true);
    cout << "Good-bye." << endl;
}
