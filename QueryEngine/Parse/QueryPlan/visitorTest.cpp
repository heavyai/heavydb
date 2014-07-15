#include "SQL/parser.h"
#include "SQL/visitor/Visitor.h"
//#include "RA/RelAlgebraParser.h"
//#include "RA/visitor/Visitor.h"
#include "SQL/visitor/QPTranslator.h"
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
      
        QPTranslator qp;
        if (parseRoot != 0)
            parseRoot->accept(qp); 

    }
    while (1==1);
    cout << "After parse" << endl;
}
