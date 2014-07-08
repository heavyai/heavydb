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

"="	|
"<>" |
"<"	|
">"	|
"<="	|
">="			{ 
					strData[5] = yytext;
					return Parser::COMPARISON;
				}


[-+*/(),;]	return yytext[0];

[.]			{
				dotCount++;
				return yytext[0];
			}


	/* names */
[A-Za-z][A-Za-z0-9_]*				{
										//printf("dotFlag is: %d\n", dotFlag);
										/* Was the most recent token a dot or AS? */
										if ((dotCount == 1) || (asFlag == 1)) {
											//textLength2 = (int) strlen(yytext);

											strData[1] = yytext;
											//printf("dot: %d length1: %d length2: %d \n", dotCount, textLength, textLength2);
											/* Reset asflag. */
											asFlag = 0; 
										}
										else if (dotCount == 2) {
											/*
											textLength3 = (int) strlen(yytext);
										
											//printf("%d\n", textLength3);
											dotCount = 0;
											*/
											strData[2] = yytext;
										}
										else {
											//textLength = (int) strlen(yytext);
											strData[0] == yytext;
										}

										/* printf("%d %d %s \n", (int) strlen(yytext), textLength, yytext); */

										return Parser::NAME;
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
										dData = atof(yytext);
										return Parser::INTNUM;
									}


[0-9]+[eE][+-]?[0-9]+	|
[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
"."[0-9]*[eE][+-]?[0-9]+			{ 
										dData = atof(yytext);
										return Parser::APPROXNUM;
									}

	/* strings */

'[^'\n]*'	{
		textLength = (int) strlen(yytext);
		int c = input();

		unput(c);	/* just peeking */
		if(c != '\'') {
			//printf("reading string: %s\n", yytext);
			strData[0] = yytext;
			return Parser::STRING;
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
						};

<<EOF>>						{ yyterminate(); }
%%