%option c++
%option noyywrap

D			[0-9]
L			[a-zA-Z_]
H			[a-fA-F0-9]
E			[Ee][+-]?{D}+
FS			(f|F|l|L)
IS			(u|U|l|L)*

%{
#include <iostream>
#include "RelAlgebraParser.h"

void count();
extern YY_Parser_STYPE yylval;

// Variables used to share data with the parser
std::string strData[10];

using namespace std;
%}

%%

"!="						{ count(); return RelAlgebraParser::NEQ; }
"="							{ count(); return RelAlgebraParser::EQ; }
">"							{ count(); return RelAlgebraParser::GT; }
"<"							{ count(); return RelAlgebraParser::LT; }
">="						{ count(); return RelAlgebraParser::GTE; }
"<="						{ count(); return RelAlgebraParser::LTE; }

"select"					{ count(); return RelAlgebraParser::SELECT; }
"project"					{ count(); return RelAlgebraParser::PROJECT; }
"sort"						{ count(); return RelAlgebraParser::SORT; }
"rename"					{ count(); return RelAlgebraParser::RENAME; }
"extend"					{ count(); return RelAlgebraParser::EXTEND; }
"groupby"					{ count(); return RelAlgebraParser::GROUPBY; }

"product"					{ count(); return RelAlgebraParser::PRODUCT; }
"join"						{ count(); return RelAlgebraParser::JOIN; }
"semijoin"					{ count(); return RelAlgebraParser::SEMIJOIN; }
"antijoin"					{ count(); return RelAlgebraParser::ANTIJOIN; }
"outerjoin"					{ count(); return RelAlgebraParser::OUTERJOIN; }
"union"						{ count(); return RelAlgebraParser::UNION; }

"max"						{ count(); return RelAlgebraParser::MAX; }
"min"						{ count(); return RelAlgebraParser::MIN; }
"count"						{ count(); return RelAlgebraParser::COUNT; }
"sum"						{ count(); return RelAlgebraParser::SUM; }
"avg"						{ count(); return RelAlgebraParser::AVG; }

"max distinct"				{ count(); return RelAlgebraParser::MAX_DISTINCT; }
"min distinct"				{ count(); return RelAlgebraParser::MIN_DISTINCT; }
"count distinct"			{ count(); return RelAlgebraParser::COUNT_DISTINCT; }
"sum distinct"				{ count(); return RelAlgebraParser::SUM_DISTINCT; }
"avg distinct"				{ count(); return RelAlgebraParser::AVG_DISTINCT; }

0[xX]{H}+{IS}?				{ count(); return(CONSTANT); }
0{D}+{IS}?					{ count(); return(CONSTANT); }
{D}+{IS}?					{ count(); return(CONSTANT); }
L?'(\\.|[^\\'])+'			{ count(); return(CONSTANT); }

{D}+{E}{FS}?				{ count(); return(CONSTANT); }
{D}*"."{D}+({E})?{FS}?		{ count(); return(CONSTANT); }
{D}+"."{D}*({E})?{FS}?		{ count(); return(CONSTANT); }

[-+*/(),;]					{ count(); return yytext[0]; }

[A-Za-z][A-Za-z0-9_]*		{ count(); strData[0] = yytext; return Parser::NAME; }

L?\"(\\.|[^\\"])*\"			{ count(); return Parser::STRVAL; }

[ \t\v\n\f]					{ count(); }

.							{ /* ignore bad characters */ }

<<EOF>>						{ yyterminate(); }
%%

void count()
{
	int i;

	for (i = 0; yytext[i] != '\0'; i++)
		if (yytext[i] == '\n')
			column = 0;
		else if (yytext[i] == '\t')
			column += 8 - (column % 8);
		else
			column++;

	ECHO;
}