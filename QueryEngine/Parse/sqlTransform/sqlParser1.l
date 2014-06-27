%{
#include "sqlParser1.h"
#include <string.h>
#include "y.tab.h"

#define TOK(name) { return name; dotCount = 0; }

int lineno = 1;
void yyerror(const char *s);

int textLength;
int textLength2;
int textLength3;

int dotCount = 0; /* How many '.'s in NAME.NAME or NAME.NAME.NAME */
int asFlag = 0; /* If the most recent token was an AS- for NAME AS NAME */

int comparisonLength;
char *lastTokenTypeRead = "start"; /* corresponds to which string token (name, string, etc), was the last read, to keep track of
	whether to store into the array at the next available index or to return to the first. */

/* sql lexer, only for select */
%}



%%

		/* literal keyword tokens */
"user"		TOK(USER);
"select"	TOK(SELECT);
"from"      TOK(FROM);
"where"		TOK(WHERE);
"having"	TOK(HAVING);
"all"		TOK(ALL);
"distinct"	TOK(DISTINCT);
"and"		TOK(AND);
"or"		TOK(OR);
"between"	TOK(BETWEEN);

"update"	TOK(UPDATE);
"of"		TOK(OF);
"current"	TOK(CURRENT);
"null"		TOK(NULLX);
"set"		TOK(SET);

"insert"	TOK(INSERT);
"values"	TOK(VALUES);
"into"		TOK(INTO);

"create"	TOK(CREATE);
"table"		TOK(TABLE);
"not"		TOK(NOT);
"unique"	TOK(UNIQUE);
"primary"	TOK(PRIMARY);
"key"		TOK(KEY);
"default"	TOK(DEFAULT);
"check"		TOK(CHECK);
"references"	TOK(REFERENCES);
"foreign"	TOK(FOREIGN);

"varchar"			TOK(VARCHAR);
"char"("acter")?	TOK(CHARACTER);
"int"("eger")?		TOK(INTEGER);
"smallint"			TOK(SMALLINT);
"numeric"	TOK(NUMERIC);
"decimal"	TOK(DECIMAL);
"float"		TOK(FLOAT);
"real"		TOK(REAL);
"double"	TOK(DOUBLE);
"precision"	TOK(PRECISION);
"drop"		TOK(DROP);

"avg"		TOK(AVG);
"min"		TOK(MIN);
"max"		TOK(MAX);
"sum"		TOK(SUM);
"count"		TOK(COUNT);

"group"		TOK(GROUP);
"order"		TOK(ORDER);
"by"		TOK(BY);

"as"						{
								/* ensure that aliased name is printed fully. */
								asFlag = 1;
								TOK(AS);
							}

"asc"		TOK(ASC);
"desc"		TOK(DESC);

"limit"		TOK(LIMIT);
"offset"	TOK(OFFSET);

"like"		TOK(LIKE);
"escape"	TOK(ESCAPE);

"="	|
"<>" |
"<"	|
">"	|
"<="	|
">="			{ 
					comparisonLength = (int) strlen(yytext);
					yylval.sSubtok = yytext;
					TOK(COMPARISON);
				}


[-+*/(),;]	TOK(yytext[0])

[.]			{
				dotCount++;
				return yytext[0];
			}


	/* names */
[A-Za-z][A-Za-z0-9_]*				{
										//printf("dotFlag is: %d\n", dotFlag);
										/* Was the most recent token a dot or AS? */
										if ((dotCount == 1) || (asFlag == 1)) {
											textLength2 = (int) strlen(yytext);

											//printf("dot: %d length1: %d length2: %d \n", dotCount, textLength, textLength2);
											/* Reset asflag. */
											asFlag = 0; 
										}
										else if (dotCount == 2) {
											textLength3 = (int) strlen(yytext);

											//printf("%d\n", textLength3);
											dotCount = 0;
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
									TOK(PARAMETER);
							} 
		 */

	/* numbers */

[0-9]+	|
[0-9]+"."[0-9]* |
"."[0-9]*							{ 
										yylval.dValue = atof(yytext);
										TOK(INTNUM);
									}


[0-9]+[eE][+-]?[0-9]+	|
[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
"."[0-9]*[eE][+-]?[0-9]+			{ 
										yylval.dValue = atof(yytext);
										TOK(APPROXNUM);
									}

	/* strings */

'[^'\n]*'	{
		textLength = (int) strlen(yytext);
		int c = input();

		unput(c);	/* just peeking */
		if(c != '\'') {
			//printf("reading string: %s\n", yytext);
			yylval.sValue = yytext;
			TOK(STRING);
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

