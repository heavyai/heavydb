#include "parser.h"
#include "visitor/Visitor.h"
#include "visitor/XMLTranslatorSQL.h"
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
        parser.parse(sql, parseRoot);
      
        XMLTranslator xml;
        if (parseRoot != 0)
            parseRoot->accept(xml); 

    }
    while (1==1);
    cout << "After parse" << endl;
}

