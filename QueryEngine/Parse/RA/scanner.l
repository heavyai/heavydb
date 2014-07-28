%option c++
%option noyywrap case-insensitive

%{
#include <iostream>
#include <vector>
#include "parser.h"

extern YY_Parser_STYPE yylval;

//void /* count() */;
int column;

// Variables used to share data with the parser
std::vector<std::string> strData;
std::vector<long int> intData;
std::vector<double> realData;

using namespace std;
%}

%%

	/* relational ops: 1 table */
"select"					{ /* count() */; return RAParser::SELECT; }
"project"					{ /* count() */; return RAParser::PROJECT; }
"sort"						{ /* count() */; return RAParser::SORT; }
"rename"					{ /* count() */; return RAParser::RENAME; }
"extend"					{ /* count() */; return RAParser::EXTEND; }
"groupby"					{ /* count() */; return RAParser::GROUPBY; }

	/* relational ops: 2 tables */
"product"					{ /* count() */; return RAParser::PRODUCT; }
"join"						{ /* count() */; return RAParser::JOIN; }
"semijoin"					{ /* count() */; return RAParser::SEMIJOIN; }
"antijoin"					{ /* count() */; return RAParser::ANTIJOIN; }
"outerjoin"					{ /* count() */; return RAParser::OUTERJOIN; }
"union"						{ /* count() */; return RAParser::UNION; }
"diff"						{ /* count() */; return RAParser::DIFF; }
"intersect"					{ /* count() */; return RAParser::INTERSECTION; }

	/* aggregation functions */
"max"						{ /* count() */; return RAParser::MAX; }
"min"						{ /* count() */; return RAParser::MIN; }
"count"						{ /* count() */; return RAParser::COUNT; }
"sum"						{ /* count() */; return RAParser::SUM; }
"avg"						{ /* count() */; return RAParser::AVG; }
"max distinct"				{ /* count() */; return RAParser::MAX_DISTINCT; }
"min distinct"				{ /* count() */; return RAParser::MIN_DISTINCT; }
"count distinct"			{ /* count() */; return RAParser::COUNT_DISTINCT; }
"sum distinct"				{ /* count() */; return RAParser::SUM_DISTINCT; }
"avg distinct"				{ /* count() */; return RAParser::AVG_DISTINCT; }

	/* arithmetic ops */
"+"						{ return RAParser::PLUS; }
"-"						{ return RAParser::MINUS; }
"/"						{ return RAParser::DIVIDE; }
"*"						{ return RAParser::MULTIPLY; }
	
	/* comparison ops */
"!="						{ /* count() */; return RAParser::NEQ; }
"="							{ /* count() */; return RAParser::EQ; }
">"							{ /* count() */; return RAParser::GT; }
"<"							{ /* count() */; return RAParser::LT; }
">="						{ /* count() */; return RAParser::GTE; }
"<="						{ /* count() */; return RAParser::LTE; }

	/* logic ops */
"or"						{ /* count() */; return RAParser::OR; }
"and"						{ /* count() */; return RAParser::AND; }
"not"						{ /* count() */; return RAParser::NOT; }

	/* numbers */
-?[0-9]+                	{ intData.push_back(atoi(yytext)); return RAParser::INTVAL; } 

-?[0-9]+"."[0-9]* |
-?"."[0-9]+     |
-?[0-9]+E[-+]?[0-9]+    |
-?[0-9]+"."[0-9]*E[-+]?[0-9]+ |
-?"."[0-9]+E[-+]?[0-9]+ { realData.push_back(atof(yytext)); return RAParser::FLOATVAL; }

	/* booleans */
TRUE    { intData.push_back(1); return RAParser::INTVAL; }
UNKNOWN { intData.push_back(-1); return RAParser::INTVAL; }
FALSE   { intData.push_back(0); return RAParser::INTVAL; }



[-+*/{}\[\](),.]				{/* count() */; return yytext[0]; }

;							{ strData.clear(); intData.clear(); realData.clear(); return yytext[0]; }

	/* names */
[A-Za-z][A-Za-z0-9_]*		{ /* count() */; strData.push_back(yytext); return RAParser::NAME; }

	/* strings */
L?\"(\\.|[^\\"])*\"			{ /* count() */; strData.push_back(yytext); return RAParser::STRVAL; }

[ \t\v\n\f]					{ /* count() */; }

.							{ /* ignore bad characters */ }

<<EOF>>						{ yyterminate(); }
%%

void count(char * yytext)
{
	int i;

	for (i = 0; yytext[i] != '\0'; i++)
		if (yytext[i] == '\n')
			column = 0;
		else if (yytext[i] == '\t')
			column += 8 - (column % 8);
		else
			column++;

	//ECHO;
}