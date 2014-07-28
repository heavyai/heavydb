%option c++
%option noyywrap case-insensitive
%option yylineno

%{
#include <iostream>
#include "parser.h"
#include <cstring>
#include <vector>
#include <string>

extern YY_Parser_STYPE yylval;

#define TOK(name) { dotCount = 0; offsetFlag = 0; mycolno += strlen(yytext); return Parser::name; }

// Variables used to share data with the parser
std::vector<std::string> strData;
std::vector<long int> intData;
std::vector<double> realData;

int dotCount = 0;
int offsetFlag = 0;
int asFlag = 0;
int mylineno = 1;
int mycolno = 0;

using namespace std;
%}

%%

	/* literal keyword tokens */
"user"				TOK(USER);
"select"			TOK(SELECT);
"from"				TOK(FROM);
"where"				TOK(WHERE);
"having"			TOK(HAVING);
"all"				TOK(ALL);
"distinct"			TOK(DISTINCT);
"and"				TOK(AND);
"or"				TOK(OR);

"update"			TOK(UPDATE);
"of"				TOK(OF);
"current"			TOK(CURRENT);
"null"				TOK(NULLX);
"set"				TOK(SET);

"insert"			TOK(INSERT);
"values"			TOK(VALUES);
"into"				TOK(INTO);

"create"			TOK(CREATE);
"table"				TOK(TABLE);
"not"				TOK(NOT);
"unique"			TOK(UNIQUE);
"primary"			TOK(PRIMARY);
"key"				TOK(KEY);
"default"			TOK(DEFAULT);
"check"				TOK(CHECK);
"references"		TOK(REFERENCES);
"foreign"			TOK(FOREIGN);

"varchar"			TOK(VARCHAR);
"char"("acter")?	TOK(CHARACTER);
"int"("eger")?		TOK(INTEGER);
"smallint"			TOK(SMALLINT);
"numeric"			TOK(NUMERIC);
"decimal"			TOK(DECIMAL);
"float"				TOK(FLOAT);
"real"				TOK(REAL);
"double"			TOK(DOUBLE);
"precision"			TOK(PRECISION);
"drop"				TOK(DROP);

"avg"				TOK(AVG);
"min"				TOK(MIN);
"max"				TOK(MAX);
"sum"				TOK(SUM);
"count"				TOK(COUNT);

"group"				TOK(GROUP);
"by"				TOK(BY);
"order"				TOK(ORDER);

"like"				TOK(LIKE);
"escape"			TOK(ESCAPE);
"limit"				TOK(LIMIT);

"asc"				TOK(ASC);
"desc"				TOK(DESC);
"between"			TOK(BETWEEN);
"substr"("ing")?	TOK(FSUBSTRING)	

"as" {
	/* ensure that aliased name is printed fully. */
	asFlag = 1;
	TOK(AS);
}
	
"offset" {
	/* ensure that aliased name is printed fully. */
	offsetFlag = 1;
	mycolno += strlen(yytext);
	return Parser::OFFSET;
}

"is"				TOK(IS);
"in"				TOK(IN);
"any"				TOK(ANY);
"some"				TOK(SOME);
"exists"			TOK(EXISTS);

"="|"<>"|"<"|">"|"!="|"<="|">=" { 
	strData.push_back(yytext);
	TOK(COMPARISON);
}


[-+*/,();]			{ mycolno++; return yytext[0]; }

[.] {
	dotCount++;
	mycolno += strlen(yytext);
	return yytext[0];
}


	/* names */
[A-Za-z][A-Za-z0-9_]* {
	strData.push_back(yytext);
	mycolno += strlen(yytext);
	TOK(NAME);
}

\"[^\"\n]*$ { cerr << "Unterminated string" << endl; }

	/* numbers */
[0-9]+ {
	intData.push_back((int)strtol(yytext, NULL, 10));
	TOK(INTNUM);
}
[0-9]+"."[0-9]* |
"."[0-9]* {
	realData.push_back((int)strtol(yytext, NULL, 10));
	TOK(APPROXNUM);
}

[0-9]+[eE][+-]?[0-9]+	|
[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
"."[0-9]*[eE][+-]?[0-9]+ { 
	realData.push_back(strtof(yytext, NULL));
	TOK(APPROXNUM);
}

	/* strings */
'[^'\n]*' {
	
	TOK(STRING);
}

'[^'\n]*$ { cerr << "Unterminated string" << endl; }


\n							{ mylineno++; mycolno = 0; /*printf("[%s:%d] lineno=%d\n", __func__, __LINE__, mylineno);*/  }

[ \t\r]+		/* add one to colno per white space */ { mycolno++; }

"--".*	;	/* comment */

.							{ cerr << "Unknown character: char is " << yytext << endl; };

<<EOF>>						{ yyterminate(); }
%%

