#include "parser.h"
#include "visitor/Visitor.h"
#include "visitor/TypeChecker.h"
#include <iostream>
#include <string>

using namespace std;
using SQL_Namespace::TypeChecker;

int main(int argc, char ** argv) {
    Parser parser;
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

        TypeChecker tc;
        if (parseRoot != 0) {
            parseRoot->accept(tc); 
        }

    }
    while (1==1);
    cout << "Good-bye." << endl;
}
