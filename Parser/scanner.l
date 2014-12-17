%option c++
%option noyywrap case-insensitive

%{
#undef yyFlexLexer
#include <cstdint>
#include <../Shared/sqldefs.h>
#include "parser.h"

	/* macro to save the text and return a token */
#define TOK(name) { return Parser::name; }
%}

%%

	/* literal keyword tokens */

ALL		{ yylval.opval = kALL; TOK(ALL) }
AND		TOK(AND)
ANY		{ yylval.opval = kANY; TOK(ANY) }
AS			TOK(AS)
ASC		TOK(ASC)
AUTHORIZATION	TOK(AUTHORIZATION)
BETWEEN		TOK(BETWEEN)
BIGINT		TOK(BIGINT)
BOOLEAN	TOK(BOOLEAN)
BY			TOK(BY)
CHAR(ACTER)?	TOK(CHARACTER)
CHECK		TOK(CHECK)
CLOSE		TOK(CLOSE)
COMMIT		TOK(COMMIT)
CONTINUE		TOK(CONTINUE)
CREATE		TOK(CREATE)
CURRENT		TOK(CURRENT)
CURSOR		TOK(CURSOR)
DECIMAL		TOK(DECIMAL)
DECLARE		TOK(DECLARE)
DEFAULT		TOK(DEFAULT)
DELETE		TOK(DELETE)
DESC		TOK(DESC)
DISTINCT		TOK(DISTINCT)
DOUBLE		TOK(DOUBLE)
ESCAPE		TOK(ESCAPE)
EXISTS		TOK(EXISTS)
FETCH		TOK(FETCH)
FLOAT		TOK(FLOAT)
FOR		TOK(FOR)
FOREIGN		TOK(FOREIGN)
FOUND		TOK(FOUND)
FROM		TOK(FROM)
GRANT		TOK(GRANT)
GROUP		TOK(GROUP)
HAVING		TOK(HAVING)
IN			TOK(IN)
INSERT		TOK(INSERT)
INT(EGER)?		TOK(INTEGER)
INTO		TOK(INTO)
IS			TOK(IS)
KEY		TOK(KEY)
LANGUAGE		TOK(LANGUAGE)
LIKE		TOK(LIKE)
NOT		TOK(NOT)
NULL		TOK(NULLX)
NUMERIC		TOK(NUMERIC)
OF			TOK(OF)
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
SOME		{ yylval.opval = kANY; TOK(SOME) } /* SOME = ANY */
TABLE		TOK(TABLE)
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
WHERE		TOK(WHERE)
WITH		TOK(WITH)
WORK		TOK(WORK)

	/* punctuation */

"="	{ yylval.opval = kEQ; TOK(COMPARISON); }
"<>" { yylval.opval = kNE; TOK(COMPARISON); }
"<"	 { yylval.opval = kLT; TOK(COMPARISON); }
">"	 { yylval.opval = kGT; TOK(COMPARISON); }
"<=" { yylval.opval = kLE; TOK(COMPARISON); }
">=" { yylval.opval = kGE; TOK(COMPARISON); }

[-+*/(),.;]	TOK(yytext[0])


	/* names */
[A-Za-z][A-Za-z0-9_]*	TOK(NAME)

	/* parameters */
":"[A-Za-z][A-Za-z0-9_]*	TOK(PARAMETER)

	/* numbers */

/* TODO: handle BIGINT literals */
[0-9]+ { yylval.intval = atoi(yytext); TOK(INTNUM) }
[0-9]+"."[0-9]* |
"."[0-9]*		{ yylval.strval = yytext; TOK(FIXEDNUM) }

/* TODO: handle FLOAT v.s. DOUBLE */
[0-9]+[eE][+-]?[0-9]+	|
[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
"."[0-9]*[eE][+-]?[0-9]+	{ yylval.floatval = atof(yytext); TOK(FLOAT) }

	/* strings */

'[^'\n]*'	{
		int c = input();

		unput(c);	/* just peeking */
		if(c != '\'')
			TOK(STRING);
		else
			yymore();
	}
		
'[^'\n]*$	{	yyerror("Unterminated string"); }

\n		{ lineno++; }

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
} /* main */
*/
