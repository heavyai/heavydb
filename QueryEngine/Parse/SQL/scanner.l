%option c++
%option noyywrap case-insensitive

%{
#undef yyFlexLexer
#include <iostream>
#include <vector>
#include "parser.h"

extern YY_Parser_STYPE yylval;

using namespace std;

// Variables used to share data with the parser
std::vector<std::string> strData;
std::vector<long int> intData;
std::vector<double> realData;


%}


%%

	/* keywords */
"add"			{ return SQLParser::ADD; }
"all"			{ return SQLParser::ALL; }
"alter"			{ return SQLParser::ALTER; }
"as"			{ return SQLParser::AS; }
"asc"			{ return SQLParser::ASC; }
"column"		{ return SQLParser::COLUMN; }
"create"		{ return SQLParser::CREATE; }
"desc"			{ return SQLParser::DESC; }
"distinct"		{ return SQLParser::DISTINCT; }
"drop"			{ return SQLParser::DROP; }
"from"			{ return SQLParser::FROM; }
"group by"		{ return SQLParser::GROUPBY; }
"having"		{ return SQLParser::HAVING; }
"insert"		{ return SQLParser::INSERT; }
"into"			{ return SQLParser::INTO; }
"key"			{ return SQLParser::KEY; }
"null"			{ return SQLParser::NULLX; }
"order by"		{ return SQLParser::ORDERBY; }
"primary"		{ return SQLParser::PRIMARY; }
"rename"		{ return SQLParser::RENAME; }
"select"		{ return SQLParser::SELECT; }
"table"			{ return SQLParser::TABLE; }
"values"		{ return SQLParser::VALUES; }
"where"			{ return SQLParser::WHERE; }

	/* data types */
"int"			{ return SQLParser::INTEGER; }
"integer"		{ return SQLParser::INTEGER; }
"float"			{ return SQLParser::FLOAT; }
"bool"			{ return SQLParser::BOOLEAN; }
"boolean"		{ return SQLParser::BOOLEAN; }

	/* aggregate functions */
"avg"			{ return SQLParser::AVG; }
"count"			{ return SQLParser::COUNT; }
"max"			{ return SQLParser::MAX; }
"min"			{ return SQLParser::MIN; }
"sum"			{ return SQLParser::SUM; }

	/* arithmetic ops */
"+"				{ return SQLParser::PLUS; }
"-"				{ return SQLParser::MINUS; }
"/"				{ return SQLParser::DIVIDE; }
"*"				{ return SQLParser::MULTIPLY; }
	
	/* comparison ops */
"<> | !="		{ return SQLParser::NEQ; }
"="				{ return SQLParser::EQ; }
">"				{ return SQLParser::GT; }
"<"				{ return SQLParser::LT; }
">="			{ return SQLParser::GTE; }
"<="			{ return SQLParser::LTE; }

	/* logic ops */
"or"			{ return SQLParser::OR; }
"and"			{ return SQLParser::AND; }
"not"			{ return SQLParser::NOT; }

	/* punctuation */
[:(),.;]	{ return yytext[0]; }

	/* names */
[A-Za-z][A-Za-z0-9]* 		{ strData.push_back(yytext); return SQLParser::NAME; }

	/* numbers */
-?[0-9]+					{ intData.push_back(strtol(yytext, NULL, 10)); return SQLParser::INTVAL; }

[0-9]+"."[0-9]* |
"."[0-9]+  					{ realData.push_back(atof(yytext)); return SQLParser::FLOATVAL; }

	/* white space */
\n
[ \t\r]+

.							{ /* ignore bad characters */ }

<<EOF>>						{ yyterminate(); }
%%


