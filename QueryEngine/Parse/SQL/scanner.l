%option c++
%option noyywrap case-insensitive
%option yylineno

%{
#include <iostream>
#include "parser.h"
#include <cstring>

extern YY_Parser_STYPE yylval;

#define TOK(name) { dotCount = 0; offsetFlag = 0; mycolno += strlen(yytext); return Parser::name; }

// Variables used to share data with the parser
std::string strData[10];

double dData[10];
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
	strData[5] = yytext;
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
	if ((dotCount == 1) || (asFlag == 1)) {
		strData[1] = yytext;
		/* Reset asflag. */
		asFlag = 0; 
	}
	else if (dotCount == 2) {
		strData[2] = yytext;
		dotCount = 0;
	}
	else
		strData[0] = yytext;
	offsetFlag = 0;
	//printf("lexer: %d\n", strlen(yytext));
	mycolno += strlen(yytext);
	return Parser::NAME;
}

	/* parameters
":"[A-Za-z][A-Za-z0-9_]* {
	yylval.sParam = yytext + 1;
	textLength = ((int) strlen(yytext)) - 1;
	TOK(PARAMETER);
} 
	*/

	/* numbers */

[0-9]+	|
[0-9]+"."[0-9]* |
"."[0-9]* { 
	if (offsetFlag == 1) {
		dData[1] = atof(yytext);
		/* Reset asflag. */
		offsetFlag = 0; 
	}
	else 
		dData[0] = atof(yytext);
	dotCount = 0;
	mycolno += strlen(yytext);
	return Parser::INTNUM;
}


[0-9]+[eE][+-]?[0-9]+	|
[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
"."[0-9]*[eE][+-]?[0-9]+ { 
	dData[0] = atof(yytext);
	TOK(APPROXNUM);
}

	/* strings */

'[^'\n]*' {
	std::string temp;
	int c = yyinput();
	temp = yytext;
	yyunput(c, yytext);	/* just peeking */
	if(c != '\'') {
		//printf("reading string: %s\n", yytext);
		strData[0] = temp.substr(1, temp.length()-2);
		// yytext;
		TOK(STRING);
	} else 
		yymore();
}
		
'[^'\n]*$					{ cerr << "Unterminated string" << endl; }

\n							{ mylineno++; mycolno = 0; /*printf("[%s:%d] lineno=%d\n", __func__, __LINE__, mylineno);*/  }

[ \t\r]+		/* add one to colno per white space */ { mycolno++; }

"--".*	;	/* comment */

.							{ cerr << "Unknown character: char is " << yytext << endl; };

<<EOF>>						{ yyterminate(); }
%%
