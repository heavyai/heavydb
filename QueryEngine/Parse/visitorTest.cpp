#include "parser.h"
#include "visitor/Visitor.h"
#include "visitor/XMLTranslator.h"
#include <iostream>
#include <string>

using namespace std;

ASTNode *parse_root = 0;

int main(int argc, char ** argv) {
	Parser parser;
    string sql;
    do {
        cout << "Enter sql statement: ";
        getline(cin,sql);
        if (sql == "q")
            break;
        ASTNode *parseRoot = 0;
        string lastParsed;
        int numErrors = parser.parse(sql, parseRoot,lastParsed);
        if (numErrors > 0) {
            cout << "Error at: " << lastParsed << endl;
        }
        //cout << "# Errors: " << numErrors << endl;
        XMLTranslator xml;
        if (parseRoot != 0)
            parseRoot->accept(xml); 

    }
    while (1==1);
    cout << "After parse" << endl;
}
