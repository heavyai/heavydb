#include <iostream>
using namespace std;

#include "UPNParser.h"

int UPNParser::yylex()
{
	yylloc.first_line = scanner.line;
	yylloc.first_column = scanner.column;
	int token = scanner.yylex(&yylval, &yylloc);
	yylloc.last_line = scanner.line;
	yylloc.last_column = scanner.column;
	yylloc.text = (char *)scanner.yytext;
	return token;
}

void UPNParser::yyerror(char * msg)
{
	cerr << yylloc.first_line << ":" << yylloc.first_column
		<<  ": " << msg << " : <" << yylloc.text << ">" << endl;
}

int UPNParser::parse()
{
	return yyparse();
}
