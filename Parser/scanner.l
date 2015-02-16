%option c++
%option noyywrap case-insensitive yylineno

%{
#undef yyFlexLexer
#include <cstdint>
#include <../Shared/sqldefs.h>
#include "parser.h"

#define yylval (dynamic_cast<SQLLexer*>(this)->yylval)

using namespace std;

	/* macro to save the text and return a token */
#define TOK(name) { return SQLParser::name; }

%}

%%

	/* literal keyword tokens */

ALL		{ yylval.qualval = kALL; TOK(ALL) }
ALTER TOK(ALTER)
AND		TOK(AND)
ANY		{ yylval.qualval = kANY; TOK(ANY) }
AS			TOK(AS)
ASC		TOK(ASC)
AUTHORIZATION	TOK(AUTHORIZATION)
BETWEEN		TOK(BETWEEN)
BIGINT		TOK(BIGINT)
BOOLEAN	TOK(BOOLEAN)
BY			TOK(BY)
CASE		TOK(CASE)
CAST		TOK(CAST)
CHAR(ACTER)?	TOK(CHARACTER)
CHECK		TOK(CHECK)
CLOSE		TOK(CLOSE)
COMMIT		TOK(COMMIT)
CONTINUE		TOK(CONTINUE)
CREATE		TOK(CREATE)
CURRENT		TOK(CURRENT)
CURSOR		TOK(CURSOR)
DATABASE	TOK(DATABASE)
DATE	TOK(DATE)
DECIMAL		TOK(DECIMAL)
DECLARE		TOK(DECLARE)
DEFAULT		TOK(DEFAULT)
DELETE		TOK(DELETE)
DESC		TOK(DESC)
DISTINCT		TOK(DISTINCT)
DOUBLE		TOK(DOUBLE)
DROP			TOK(DROP)
ELSE			TOK(ELSE)
END				TOK(END)
ESCAPE		TOK(ESCAPE)
EXISTS		TOK(EXISTS)
FETCH		TOK(FETCH)
FIRST		TOK(FIRST)
FLOAT		TOK(FLOAT)
FOR		TOK(FOR)
FOREIGN		TOK(FOREIGN)
FOUND		TOK(FOUND)
FROM		TOK(FROM)
GRANT		TOK(GRANT)
GROUP		TOK(GROUP)
HAVING		TOK(HAVING)
IF			TOK(IF)
IN			TOK(IN)
INSERT		TOK(INSERT)
INT(EGER)?		TOK(INTEGER)
INTO		TOK(INTO)
IS			TOK(IS)
KEY		TOK(KEY)
LANGUAGE		TOK(LANGUAGE)
LAST	TOK(LAST)
LIKE		TOK(LIKE)
LIMIT		TOK(LIMIT)
NOT		TOK(NOT)
NULL		TOK(NULLX)
NUMERIC		TOK(NUMERIC)
OF			TOK(OF)
OFFSET	TOK(OFFSET)
ON			TOK(ON)
OPEN		TOK(OPEN)
OPTION		TOK(OPTION)
OR			TOK(OR)
ORDER		TOK(ORDER)
PRECISION		TOK(PRECISION)
PRIMARY		TOK(PRIMARY)
PRIVILEGES		TOK(PRIVILEGES)
PROCEDURE		TOK(PROCEDURE)
PUBLIC		TOK(PUBLIC)
REAL		TOK(REAL)
REFERENCES		TOK(REFERENCES)
ROLLBACK		TOK(ROLLBACK)
SCHEMA		TOK(SCHEMA)
SELECT		TOK(SELECT)
SET		TOK(SET)
SMALLINT		TOK(SMALLINT)
SOME		{ yylval.qualval = kANY; TOK(SOME) } /* SOME = ANY */
TABLE		TOK(TABLE)
TEXT		TOK(TEXT)
THEN		TOK(THEN)
TIME		TOK(TIME)
TIMESTAMP	TOK(TIMESTAMP)
TO			TOK(TO)
UNION		TOK(UNION)
UNIQUE		TOK(UNIQUE)
UPDATE		TOK(UPDATE)
USER		TOK(USER)
VALUES		TOK(VALUES)
VARCHAR	TOK(CHARACTER)	/* XXX don't distinguish char and varchar for now */
VIEW		TOK(VIEW)
WHEN		TOK(WHEN)
WHERE		TOK(WHERE)
WITH		TOK(WITH)
WORK		TOK(WORK)

	/* punctuation */

"="	{ yylval.opval = kEQ; TOK(EQUAL); }
"<>" { yylval.opval = kNE; TOK(COMPARISON); }
"<"	 { yylval.opval = kLT; TOK(COMPARISON); }
">"	 { yylval.opval = kGT; TOK(COMPARISON); }
"<=" { yylval.opval = kLE; TOK(COMPARISON); }
">=" { yylval.opval = kGE; TOK(COMPARISON); }

[-+*/(),.;]	{ return yytext[0]; }


	/* names */
[A-Za-z][A-Za-z0-9_]*	{ yylval.stringval = new std::string(yytext); TOK(NAME) }

	/* parameters */
":"[A-Za-z][A-Za-z0-9_]*	TOK(PARAMETER)

	/* numbers */

-?[0-9]+ { yylval.intval = atoll(yytext); TOK(INTNUM) }
-?[0-9]+"."[0-9]* |
-?"."[0-9]*		{ 
		yylval.stringval = new std::string(yytext); 
		// numeric and decimal precisions cannot exceed 19
		if (yylval.stringval->length() < 20)
			return SQLParser::FIXEDNUM;
		// otherwise return as double
		yylval.doubleval = std::stod(*yylval.stringval);
		return SQLParser::DOUBLE;
	}

	/* TODO: handle FLOAT v.s. DOUBLE */
-?[0-9]+[eE][+-]?[0-9]+	|
-?[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
-?"."[0-9]*[eE][+-]?[0-9]+	{ yylval.doubleval = atof(yytext); TOK(DOUBLE) }

	/* strings */

'[^'\n]*'	{
		int c = yyFlexLexer::yyinput();

		unput(c);	/* just peeking */
		if(c != '\'') {
			/* remove quotes */
			yylval.stringval = new std::string(yytext+1, strlen(yytext)-3);
			return SQLParser::STRING;
		}
		else
			yymore();
	}
		
'[^'\n]*$	{	YY_FATAL_ERROR("Unterminated string"); }

	/*
\n		{ lineno++; }
	*/

[ \t\r]+	;	/* white space */

"--".*	;	/* comment */


.			/* ignore random non-SQL text */

<<EOF>>		{ yyterminate(); }
%%

/*
void
yyerror(char *s)
{
	printf("%d: %s at %s\n", lineno, s, yytext);
}

int main()
{
	if(!yyparse())
		fprintf(stderr, "Embedded SQL parse worked\n");
	else
		fprintf(stderr, "Embedded SQL parse failed\n");
	return 0;
}
*/
