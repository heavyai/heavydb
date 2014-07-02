%option c++
%option noyywrap

%{
#include <iostream>
#include "parser.h"

extern YY_Parser_STYPE yylval;

// Variables used to share data with the parser
std::string strData[10];
int intData;
using namespace std;
%}

%%

		/* literal keyword tokens */
"user"		return Parser::USER;
"select"	return Parser::SELECT;
"from"      return Parser::FROM;
"where"		return Parser::WHERE;
"having"	return Parser::HAVING;
"all"		return Parser::ALL;
"distinct"	return Parser::DISTINCT;

"update"	return Parser::UPDATE;
"of"		return Parser::OF;
"current"	return Parser::CURRENT;
"null"		return Parser::NULLX;
"set"		return Parser::SET;

"insert"	return Parser::INSERT;
"values"	return Parser::VALUES;
"into"		return Parser::INTO;

"create"	return Parser::CREATE;
"table"		return Parser::TABLE;
"not"		return Parser::NOT;
"unique"	return Parser::UNIQUE;
"primary"	return Parser::PRIMARY;
"key"		return Parser::KEY;
"default"	return Parser::DEFAULT;
"check"		return Parser::CHECK;
"references"	return Parser::REFERENCES;
"foreign"		return Parser::FOREIGN;

"varchar"			return Parser::VARCHAR;
"char"("acter")?	return Parser::CHARACTER;
"int"("eger")?		return Parser::INTEGER;
"smallint"			return Parser::SMALLINT;
"numeric"	return Parser::NUMERIC;
"decimal"	return Parser::DECIMAL;
"float"		return Parser::FLOAT;
"real"		return Parser::REAL;
"double"	return Parser::DOUBLE;
"precision"	return Parser::PRECISION;
"drop"		return Parser::DROP;

"avg"		return Parser::AVG;
"min"		return Parser::MIN;
"max"		return Parser::MAX;
"sum"		return Parser::SUM;
"count"		return Parser::COUNT;

"group"		return Parser::GROUP;
"order"		return Parser::ORDER;

[0-9]+						{ intData = atoi(yytext); return Parser::INTNUM; }

[-+*/(),;]					return yytext[0];

[A-Za-z][A-Za-z0-9_]*		{ strData[0] = yytext; return Parser::NAME; }

[ \t\r]+					; /* ignore white space */

\n 							{ ++yylineno; }

.							return Parser::UNKNOWN;

<<EOF>>						{ yyterminate(); }
%%