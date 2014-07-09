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
        parser.parse(sql);
        XMLTranslator xml;
        parse_root->accept(xml);
    }
    while (1==1);
    cout << "After parse" << endl;
}
