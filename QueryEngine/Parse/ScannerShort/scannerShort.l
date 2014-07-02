%option c++
%option noyywrap

%{
#include <string.h>
#include <iostream>
#include "types.h"
#include "parserShort.h"
using namespace std;

yy_Parser_stype yylval;

#define TOK(name) { return name; }

//void yyerror(const char *s);

int textLength;
int textLength2;

int dotFlag = 0; /* If the most recent token was a dot- for NAME . NAME */
int asFlag = 0; /* If the most recent token was an AS- for NAME AS NAME */

int comparisonLength;
char *lastTokenTypeRead = "start"; /* corresponds to which string token (name, string, etc), was the last read, to keep track of
	whether to store into the array at the next available index or to return to the first. */

%}



%%

"table"		return Parser::TABLE;
"drop"		return Parser::DROP;

"="	|
"<>" |
"<"	|
">"	|
"<="	|
">="			{ 
					comparisonLength = (int) strlen(yytext);
					yylval.sSubtok = yytext;
					return Parser::COMPARISON;
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
										return Parser::NAME;
									}

	/* parameters
":"[A-Za-z][A-Za-z0-9_]*	{
									yylval.sParam = yytext + 1;
									textLength = ((int) strlen(yytext)) - 1;
									return Parser::PARAMETER;
							} 
		 */

	/* numbers */

[0-9]+	|
[0-9]+"."[0-9]* |
"."[0-9]*							{ 
										yylval.iValue = atof(yytext);
										return Parser::INTNUM;
									}


[0-9]+[eE][+-]?[0-9]+	|
[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
"."[0-9]*[eE][+-]?[0-9]+			{ 
										yylval.iValue = atof(yytext);
										return Parser::APPROXNUM;
									}

	/* strings */

'[^'\n]*'	{
		textLength = (int) strlen(yytext);
		int c = yyinput();

		unput(c);	/* just peeking */
		if(c != '\'') {
			//printf("reading string: %s\n", yytext);
			yylval.sValue = yytext;
			return Parser::STRING;
		} else 
			yymore();
		
	}
		
'[^'\n]*$			{	/*yyerror("Unterminated string");*/ }

\n		{ yylineno++; }

[ \t\r]+	;	/* white space */

"--".*	;	/* comment */

.						{
							/*yyerror("Unknown character");*/
							printf("%s\n", yytext);
						}

%%
