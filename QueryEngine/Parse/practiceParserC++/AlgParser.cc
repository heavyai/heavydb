#include <iostream>
using namespace std;
#include "AlgParser.h"

int AlgParser::yylex()
{
	yylloc.first_line = scanner.line;
	yylloc.first_column = scanner.column;
	int token = scanner.yylex(&yylval, &yylloc);
	yylloc.last_line = scanner.line;
	yylloc.last_column = scanner.column;
	yylloc.text = (char *)scanner.yytext;
	return token;
}

void AlgParser::yyerror(char * msg)
{
	cerr << yylloc.first_line << ":" << yylloc.first_column
		<<  ": " << msg << " : <" << yylloc.text << ">" <<  endl;
}

int AlgParser::parse()
{
	return yyparse();
}
