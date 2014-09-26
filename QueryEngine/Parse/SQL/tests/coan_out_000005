/**
 * The main function in this file is used for testing a visitor of an
 * SQL AST.
 *
 *
 */
#include "../parser.h"
#include "../visitor/Visitor.h"
#include "../visitor/XMLTranslator.h"
#include <iostream>
#include <string>

using namespace std;
using namespace SQL_Namespace;

int main(int argc, char ** argv) {
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
        
         XMLTranslator xml;
         if (parseRoot != 0)
             parseRoot->accept(xml);
    }
    while (true);
    cout << "Good-bye." << endl;
}
