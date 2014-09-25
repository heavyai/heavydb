/**
 * @file	translateTest.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 */
#include <iostream>
#include <string>
#include <vector>
#include "../../../Shared/types.h"
#include "../../Parse/SQL/parser.h"
#include "../../Translate/SQL_RA_Translator.h"
#include "../../Parse/RA/visitor/XMLTranslator.h"
#include "../../Parse/RA/visitor/QPTranslator.h"
#include "../../Analysis/TypeChecker.h"
#include "../../Analysis/NameWalker.h"
#include "../../../DataMgr/Metadata/Catalog.h"


using namespace std;
using Translate_Namespace::SQL_RA_Translator;
using Analysis_Namespace::NameWalker;
using Analysis_Namespace::TypeChecker;

int main(int argc, char ** argv) {
    
    // Create a parser for SQL and... do stuff
    SQLParser parser;
    string sql;
    do {
        
        // Add a table to the catalog
        Catalog c(".");
        
        std::vector<ColumnRow*> cols;
        cols.push_back(new ColumnRow("a", INT_TYPE, true));
        cols.push_back(new ColumnRow("b", FLOAT_TYPE, true));
        
        mapd_err_t err = c.addTableWithColumns("T1", cols);
        if (err != MAPD_SUCCESS) {
            printf("[%s:%d] Catalog::addTableWithColumns: err = %d\n", __FILE__, __LINE__, err);
            //exit(EXIT_FAILURE);
        }
        
        // Take input from client
        cout << "mapd> ";
        getline(cin,sql);
        if (sql == "q")
            break;
        else sql = sql + "\n";
        
        // Lexing and Parsing
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
        
        // Type Checking
        /*TypeChecker typeChecker(catalog);
        if (parseRoot != 0)
            parseRoot->accept(typeChecker);*/

        // Translation from SQL to RA
        SQL_RA_Translator sql2ra;
        if (parseRoot != 0)
            parseRoot->accept(sql2ra);
        
        // XML output of RA AST
        RA_Namespace::XMLTranslator ra_xml;
        printf("\nXML Output of translated Relational Algebra AST\n");
        printf("-----------------------------------------------\n");
        if (sql2ra.root != 0)
            sql2ra.root->accept(ra_xml);
        printf("\n\n");
        
        // Translation to Query Plan
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
