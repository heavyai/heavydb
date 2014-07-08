%option c++
%option noyywrap

%{
#include <iostream>
#include "parser.h"

extern YY_Parser_STYPE yylval;

// Variables used to share data with the parser
std::string strData[10];

double dData;
int dotCount = 0;
int asFlag = 0;
int lineno = 0;

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
"by"		return Parser::BY;
"order"		return Parser::ORDER;

"like"		return Parser::LIKE;
"escape"		return Parser::ESCAPE;
"limit"		return Parser::LIMIT;

"asc"		return Parser::ASC;
"desc"		return Parser::DESC;

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
										strData[0] = yytext;
										strData[1] = "placeholder 1";
										strData[2] = "placeholder 2";
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
		int c = getchar();

		unput(c);	/* just peeking */
		if(c != '\'') {
			//printf("reading string: %s\n", yytext);
			strData[0] = yytext;
			return Parser::STRING;
		} else 
			yymore();
		
	}
		
'[^'\n]*$			{	cerr << "Unterminated string" << endl; }

\n		{ }

[ \t\r]+	;	/* white space */

"--".*	;	/* comment */

.						{
							cerr << "Unknown character" << endl;
						};

<<EOF>>						{ yyterminate(); }
%%