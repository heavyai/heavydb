%name sqlScanner1
%define IOSTREAM
%define LEX_PARAM YY_CalcParser_STYPE *val, YY_CalcParser_LTYPE *loc
%define MEMBERS public: int line, column;
%define CONSTRUCTOR_INIT : line(1), column(1)
 
%header{
#include<sstream>
#include "defines.h"
#include <string.h>
#include <iostream>
using namespace std;

#define TOK(name) { return name; }

int lineno = 1;
void yyerror(const char *s);

int textLength;
int textLength2;

int dotFlag = 0; /* If the most recent token was a dot- for NAME . NAME */
int asFlag = 0; /* If the most recent token was an AS- for NAME AS NAME */

int comparisonLength;
char *lastTokenTypeRead = "start"; /* corresponds to which string token (name, string, etc), was the last read, to keep track of
	whether to store into the array at the next available index or to return to the first. */

/* sql lexer, only for select */
%}



%%

		/* literal keyword tokens */
"user"		return USER;
"select"	return SELECT;
"from"      return FROM;
"where"		return WHERE;
"having"	return HAVING;
"all"		return ALL;
"distinct"	return DISTINCT;

"update"	return UPDATE;
"of"		return OF;
"current"	return CURRENT;
"null"		return NULLX;
"set"		return SET;

"insert"	return INSERT;
"values"	return VALUES;
"into"		return INTO;

"create"	return CREATE;
"table"		return TABLE;
"not"		return NOT;
"unique"	return UNIQUE;
"primary"	return PRIMARY;
"key"		return KEY;
"default"	return DEFAULT;
"check"		return CHECK;
"references"	return REFERENCES;
"foreign"	return FOREIGN;

"varchar"			return VARCHAR;
"char"("acter")?	return CHARACTER;
"int"("eger")?		return INTEGER;
"smallint"			return SMALLINT;
"numeric"	return NUMERIC;
"decimal"	return DECIMAL;
"float"		return FLOAT;
"real"		return REAL;
"double"	return DOUBLE;
"precision"	return PRECISION;
"drop"		return DROP;

"avg"		return AVG;
"min"		return MIN;
"max"		return MAX;
"sum"		return SUM;
"count"		return COUNT;

"group"		return GROUP;
"order"		return ORDER;
"by"		return BY;

"as"						{
								/* ensure that aliased name is printed fully. */
								asFlag = 1;
								return AS;
							}

"asc"		return ASC;
"desc"		return DESC;

"limit"		return LIMIT;
"offset"	return OFFSET;

"="	|
"<>" |
"<"	|
">"	|
"<="	|
">="			{ 
					comparisonLength = (int) strlen(yytext);
					yylval.sSubtok = yytext;
					return COMPARISON;
				}


[-+*/(),;]	TOK(yytext[0])

[.]			{
				dotFlag = 1;
				TOK(yytext[0]);
			}


	/* names */
[A-Za-z][A-Za-z0-9_]*				{
										//printf("dotFlag is: %d\n", dotFlag);
										/* Was the most recent token a dot or AS? */
										if ((dotFlag == 1) || (asFlag == 1)) {
											textLength2 = (int) strlen(yytext);
											dotFlag = 0; /* Reset the dotFlag and asflag. */
											asFlag = 0; 
										}
										else textLength = (int) strlen(yytext);

										

										/* printf("%d %d %s \n", (int) strlen(yytext), textLength, yytext); */

										yylval.sName = yytext;
										return NAME;
									}

	/* parameters
":"[A-Za-z][A-Za-z0-9_]*	{
									yylval.sParam = yytext + 1;
									textLength = ((int) strlen(yytext)) - 1;
									return PARAMETER;
							} 
		 */

	/* numbers */

[0-9]+	|
[0-9]+"."[0-9]* |
"."[0-9]*							{ 
										yylval.iValue = atof(yytext);
										return INTNUM;
									}


[0-9]+[eE][+-]?[0-9]+	|
[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
"."[0-9]*[eE][+-]?[0-9]+			{ 
										yylval.iValue = atof(yytext);
										return APPROXNUM;
									}

	/* strings */

'[^'\n]*'	{
		textLength = (int) strlen(yytext);
		int c = input();

		unput(c);	/* just peeking */
		if(c != '\'') {
			//printf("reading string: %s\n", yytext);
			yylval.sValue = yytext;
			return STRING;
		} else 
			yymore();
		
	}
		
'[^'\n]*$			{	yyerror("Unterminated string"); }

\n		{ lineno++; }

[ \t\r]+	;	/* white space */

"--".*	;	/* comment */

.						{
							yyerror("Unknown character");
							printf("%s\n", yytext);
						}

%%
/*
void yyerror(char *s) {
	fprintf(stderr, "line %d: %s\n", lineno, s);
}
*/
int yywrap(void) {
    return 1;
}

