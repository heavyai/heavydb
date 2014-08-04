#include <iostream>
#include <string>
#include "parser.h"
//#include "visitor/Visitor.h"
#include "translator/SQL_RA_Translator.h"

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
        
        SQL_RA_Translator tr;
        if (parseRoot != 0) {
            printf("before accept\n");
            parseRoot->accept(tr);
            printf("after accept\n");
        }
    }
    while (true);
    cout << "Good-bye." << endl;
}
