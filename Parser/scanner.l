%option c++
%option noyywrap case-insensitive yylineno

%{
#undef yyFlexLexer
#include <cstdint>
#include <cstdio>
#include <boost/algorithm/string/trim.hpp>
#include "../Shared/sqldefs.h"
#include "parser.h"

#define yylval (dynamic_cast<SQLLexer*>(this)->yylval)

using namespace std;

	/* macro to save the text and return a token */
#define TOK(name) { return SQLParser::name; }

%}

%%

	/* literal keyword tokens */

ACCESS        TOK(ACCESS)
ALL		{ yylval.qualval = kALL; TOK(ALL) }
ALTER         TOK(ALTER)
ADD           TOK(ADD)
AND           TOK(AND)
ANY           { yylval.qualval = kANY; TOK(ANY) }
ARRAY         TOK(ARRAY)
AS            TOK(AS)
ASC           TOK(ASC)
AUTHORIZATION	TOK(AUTHORIZATION)
BETWEEN       TOK(BETWEEN)
BIGINT        TOK(BIGINT)
BOOLEAN       TOK(BOOLEAN)
BY            TOK(BY)
CASE          TOK(CASE)
CAST          TOK(CAST)
CHAR_LENGTH   TOK(CHAR_LENGTH)
CHAR(ACTER)?	TOK(CHARACTER)
CHECK         TOK(CHECK)
CLOSE         TOK(CLOSE)
COLUMN        TOK(COLUMN)
COMMIT        TOK(COMMIT)
CONTINUE      TOK(CONTINUE)
COPY          TOK(COPY);
CREATE        TOK(CREATE)
CURRENT       TOK(CURRENT)
CURSOR        TOK(CURSOR)
DASHBOARD     TOK(DASHBOARD)
DATABASE      TOK(DATABASE)
DATE          TOK(DATE)
DATETIME      TOK(DATETIME)
DATE_TRUNC    TOK(DATE_TRUNC)
DECIMAL       TOK(DECIMAL)
DECLARE       TOK(DECLARE)
DEFAULT       TOK(DEFAULT)
DELETE        TOK(DELETE)
DESC          TOK(DESC)
DICTIONARY    TOK(DICTIONARY)
DISTINCT      TOK(DISTINCT)
DOUBLE        TOK(DOUBLE)
DROP          TOK(DROP)
EDIT          TOK(EDIT)
EDITOR        TOK(EDITOR)
ELSE          TOK(ELSE)
END           TOK(END)
EXISTS        TOK(EXISTS)
EXPLAIN       TOK(EXPLAIN)
EXTRACT       TOK(EXTRACT)
FETCH         TOK(FETCH)
FIRST         TOK(FIRST)
FLOAT         TOK(FLOAT)
FOR           TOK(FOR)
FOREIGN       TOK(FOREIGN)
FOUND         TOK(FOUND)
FROM          TOK(FROM)
GEOGRAPHY     TOK(GEOGRAPHY)
GEOMETRY      TOK(GEOMETRY)
GRANT         TOK(GRANT)
GROUP         TOK(GROUP)
HAVING        TOK(HAVING)
IF            TOK(IF)
ILIKE         TOK(ILIKE)
IN            TOK(IN)
INSERT        TOK(INSERT)
INT(EGER)?		TOK(INTEGER)
INTO          TOK(INTO)
IS            TOK(IS)
LANGUAGE      TOK(LANGUAGE)
LAST          TOK(LAST)
LENGTH        TOK(LENGTH)
LIKE          TOK(LIKE)
LIMIT         TOK(LIMIT)
LINESTRING    TOK(LINESTRING)
MAX           TOK(MAX)
MIN           TOK(MIN)
MOD           TOK(MOD)
MULTIPOLYGON  TOK(MULTIPOLYGON)
NOT           TOK(NOT)
NOW           TOK(NOW)
NULL          TOK(NULLX)
NUMERIC       TOK(NUMERIC)
OF            TOK(OF)
OFFSET        TOK(OFFSET)
ON            TOK(ON)
OPEN          TOK(OPEN)
OPTION        TOK(OPTION)
OPTIMIZE      TOK(OPTIMIZE)
OR            TOK(OR)
ORDER         TOK(ORDER)
POINT         TOK(POINT)
POLYGON       TOK(POLYGON)
PRECISION     TOK(PRECISION)
PRIMARY       TOK(PRIMARY)
PRIVILEGES		TOK(PRIVILEGES)
PROCEDURE     TOK(PROCEDURE)
PUBLIC        TOK(PUBLIC)
REAL          TOK(REAL)
REFERENCES		TOK(REFERENCES)
RENAME        TOK(RENAME)
REVOKE        TOK(REVOKE)
ROLE          TOK(ROLE)
ROLLBACK      TOK(ROLLBACK)
SCHEMA        TOK(SCHEMA)
SELECT        TOK(SELECT)
SET           TOK(SET)
SHARD         TOK(SHARD)
SHARED        TOK(SHARED)
SHOW          TOK(SHOW)
SKYLINE       TOK(SKYLINE)
SMALLINT      TOK(SMALLINT)
SOME          { yylval.qualval = kANY; TOK(SOME) } /* SOME = ANY */
SQL           TOK(SQL)
TABLE         TOK(TABLE)
TEMPORARY     TOK(TEMPORARY)
TEXT          TOK(TEXT)
THEN          TOK(THEN)
TIME          TOK(TIME)
TIMESTAMP     TOK(TIMESTAMP)
TINYINT       TOK(TINYINT)
TO            TOK(TO)
TRUNCATE      TOK(TRUNCATE)
UNION         TOK(UNION)
UNIQUE        TOK(UNIQUE)
UPDATE        TOK(UPDATE)
USER          TOK(USER)
VALUES        TOK(VALUES)
VARCHAR       TOK(CHARACTER)	/* XXX don't distinguish char and varchar for now */
VIEW          TOK(VIEW)
WHEN          TOK(WHEN)
WHERE         TOK(WHERE)
WITH          TOK(WITH)
WORK          TOK(WORK)

	/* punctuation */

"="	{ yylval.opval = kEQ; TOK(EQUAL); }
"<>" { yylval.opval = kNE; TOK(COMPARISON); }
"<"	 { yylval.opval = kLT; TOK(COMPARISON); }
">"	 { yylval.opval = kGT; TOK(COMPARISON); }
"<=" { yylval.opval = kLE; TOK(COMPARISON); }
">=" { yylval.opval = kGE; TOK(COMPARISON); }

[-+*/(),.;]	{ return yytext[0]; }


	/* names */
[A-Za-z_][A-Za-z0-9\$_]*	{ yylval.stringval = new std::string(yytext); TOK(NAME) }

  /* names with dashes */
[A-Za-z_][A-Za-z0-9\$_\-]*	{ yylval.stringval = new std::string(yytext); TOK(DASHEDNAME) }

	/* emails */
([^[:space:]\"]+|\".+\")@[A-Za-z0-9][A-Za-z0-9\-\.]*\.[A-Za-z]+	{ yylval.stringval = new std::string(yytext); TOK(EMAIL) }

	/* parameters */
":"[A-Za-z][A-Za-z0-9_]*	TOK(PARAMETER)

	/* numbers */

-?[0-9]+ { yylval.intval = atoll(yytext); TOK(INTNUM) }
-?[0-9]+"."[0-9]* |
-?"."[0-9]*		{
		yylval.stringval = new std::string(yytext);
    // numeric and decimal precisions cannot exceed 19
    if (yylval.stringval->length() < 20) {
            return SQLParser::FIXEDNUM;
    }
    delete yylval.stringval;
    // otherwise return as double
    yylval.doubleval = std::stod(yytext);
    return SQLParser::DOUBLE;
	}

	/* TODO: handle FLOAT v.s. DOUBLE */
-?[0-9]+[eE][+-]?[0-9]+	|
-?[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
-?"."[0-9]*[eE][+-]?[0-9]+	{ yylval.doubleval = atof(yytext); TOK(DOUBLE) }

	/* strings */
#~#.*#~# {
		int len = yyleng - 6;
		if (len > 0) {
			yylval.stringval = new std::string(yytext + 3, len);
			boost::trim(*yylval.stringval);
		} else {
			yylval.stringval = new std::string("");
		}
		return SQLParser::FWDSTR;
	}

'[^'\n]*'	{
		int c = yyFlexLexer::yyinput();

		unput(c);	/* just peeking */
		if(c != '\'') {
			/* remove quotes */
      char escape_removed[yyleng - 1];
      char *p = yytext + 1;
      int j = 0;
      int len = yyleng - 2;
      for (int i = 0; i < len; i++) {
        if (p[i] == '\\') {
          if (p[i+1] == 't') {
            escape_removed[j++] = '\t';
            i++;
            continue;
          } else if (p[i+1] == 'n') {
            escape_removed[j++] = '\n';
            i++;
            continue;
          } else if (p[i+1] == '\\') {
            escape_removed[j++] = '\\';
            i++;
            continue;
          } else if (p[i+1] == 'x' && i + 3 < len) {
            char s[3];
            s[0] = p[i+2];
            s[1] = p[i+3];
            s[2] = '\0';
            int c;
            sscanf(s, "%x", &c);
            escape_removed[j++] = (char)c;
            i += 3;
            continue;
          }
        }
        if (p[i] != '\'' || p[i + 1] != '\'')
          escape_removed[j++] = p[i];
      }
      escape_removed[j] = '\0';

			yylval.stringval = new std::string(escape_removed);
			return SQLParser::STRING;
		}
		else
			yymore();
	}

'[^'\n]*$	{	throw std::runtime_error("Unterminated string"); }

	/*
\n		{ lineno++; }
	*/

[ \n\t\r]+	;	/* white space */

"--".*	;	/* comment */


.			{ return yytext[0]; } /* random non-SQL text.  cause error in parser */

<<EOF>>		{ yyterminate(); }
%%
