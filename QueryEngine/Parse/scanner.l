%option c++
%option noyywrap

%{
#include <iostream>
#include "parser.h"

extern YY_Parser_STYPE yylval;

// Variables used to share data with the parser
std::string strData[10];

using namespace std;
%}

%%

"drop"						return Parser::DROP;
"table"						return Parser::TABLE;

[0-9]+						return Parser::INTNUM;

[-+*/(),;]					return yytext[0];

[A-Za-z][A-Za-z0-9_]*		{ strData[0] = yytext; return Parser::NAME; }

[ \t\r]+					; /* ignore white space */

\n 							{ ++yylineno; }

.							return Parser::UNKNOWN;

<<EOF>>						{ yyterminate(); }
%%