#include "parser.h"
#include "visitor/Visitor.h"
#include "visitor/XMLTranslator.h"
#include <iostream>

using namespace std;

ASTNode *parse_root = 0;

int main(int argc, char ** argv) {
	Parser parser;
	parser.yyparse();
  	
  	XMLTranslator xml;
    parse_root->accept(xml);
}
