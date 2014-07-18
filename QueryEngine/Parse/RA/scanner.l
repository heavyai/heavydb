%option c++
%option noyywrap case-insensitive

D			[0-9]
L			[a-zA-Z_]
H			[a-fA-F0-9]
E			[Ee][+-]?{D}+
FS			(f|F|l|L)
IS			(u|U|l|L)*

%{
#include <iostream>
#include "parser.h"

extern YY_Parser_STYPE yylval;

//void /* count() */;
int column;

// Variables used to share data with the parser
std::string strData[10];
int dData[5];

using namespace std;
%}

%%

"!="						{ /* count() */; return Parser::NEQ; }
"="							{ /* count() */; return Parser::EQ; }
">"							{ /* count() */; return Parser::GT; }
"<"							{ /* count() */; return Parser::LT; }
">="						{ /* count() */; return Parser::GTE; }
"<="						{ /* count() */; return Parser::LTE; }

"select"					{ /* count() */; return Parser::SELECT; }
"project"					{ /* count() */; return Parser::PROJECT; }
"sort"						{ /* count() */; return Parser::SORT; }
"rename"					{ /* count() */; return Parser::RENAME; }
"extend"					{ /* count() */; return Parser::EXTEND; }
"groupby"					{ /* count() */; return Parser::GROUPBY; }

"product"					{ /* count() */; return Parser::PRODUCT; }
"join"						{ /* count() */; return Parser::JOIN; }
"semijoin"					{ /* count() */; return Parser::SEMIJOIN; }
"antijoin"					{ /* count() */; return Parser::ANTIJOIN; }
"outerjoin"					{ /* count() */; return Parser::OUTERJOIN; }
"union"						{ /* count() */; return Parser::UNION; }

"max"						{ /* count() */; return Parser::MAX; }
"min"						{ /* count() */; return Parser::MIN; }
"count"						{ /* count() */; return Parser::COUNT; }
"sum"						{ /* count() */; return Parser::SUM; }
"avg"						{ /* count() */; return Parser::AVG; }

"max distinct"				{ /* count() */; return Parser::MAX_DISTINCT; }
"min distinct"				{ /* count() */; return Parser::MIN_DISTINCT; }
"count distinct"			{ /* count() */; return Parser::COUNT_DISTINCT; }
"sum distinct"				{ /* count() */; return Parser::SUM_DISTINCT; }
"avg distinct"				{ /* count() */; return Parser::AVG_DISTINCT; }

"+"						{ return Parser::PLUS; }
"-"						{ return Parser::MINUS; }
"/"						{ return Parser::DIVIDE; }
"*"						{ return Parser::MULTIPLY; }

0[xX]{H}+{IS}?				{ /* count() */; return(Parser::CONSTANT); }
0{D}+{IS}?					{ /* count() */; return(Parser::CONSTANT); }
{D}+{IS}?					{ /* count() */; return(Parser::CONSTANT); }
L?'(\\.|[^\\'])+'			{ /* count() */; return(Parser::CONSTANT); }

{D}+{E}{FS}?				{ /* count() */; return(Parser::CONSTANT); }
{D}*"."{D}+({E})?{FS}?		{ /* count() */; return(Parser::CONSTANT); }
{D}+"."{D}*({E})?{FS}?		{ /* count() */; return(Parser::CONSTANT); }

[-+*/{}(),;]					{ /* count() */; return yytext[0]; }

[A-Za-z][A-Za-z0-9_]*		{ /* count() */; strData[0] = yytext; return Parser::NAME; }

L?\"(\\.|[^\\"])*\"			{ /* count() */; return Parser::STRVAL; }

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