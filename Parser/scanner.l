%option c++
%option noyywrap case-insensitive yylineno

%Start STATE_CREATE STATE_CREATE_AS STATE_INSERT STATE_INSERT_INTO

%{
#undef yyFlexLexer
#include <cstdint>
#include <cstdio>
#include <boost/algorithm/string/trim.hpp>
#include "../Shared/sqldefs.h"
#include "parser.h"

#define yylval (dynamic_cast<SQLLexer*>(this)->yylval)
#define parsed_str_tokens_ (dynamic_cast<SQLLexer*>(this)->parsed_str_tokens_)

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
ARCHIVE       TOK(ARCHIVE)
ARRAY         TOK(ARRAY)
<STATE_CREATE>AS    { BEGIN STATE_CREATE_AS; return SQLParser::AS; }
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
CREATE        { BEGIN STATE_CREATE; return SQLParser::CREATE; }
CURRENT       TOK(CURRENT)
CURSOR        TOK(CURSOR)
DASHBOARD     TOK(DASHBOARD)
DATABASE      TOK(DATABASE)
DATAFRAME     TOK(DATAFRAME)
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
DUMP          TOK(DUMP)
EDIT          TOK(EDIT)
EDITOR        TOK(EDITOR)
ELSE          TOK(ELSE)
END           TOK(END)
EXISTS        TOK(EXISTS)
EXTRACT       TOK(EXTRACT)
FETCH         TOK(FETCH)
FIRST         TOK(FIRST)
FLOAT         TOK(FLOAT)
FOR           TOK(FOR)
FOREIGN       TOK(FOREIGN)
FOUND         TOK(FOUND)
FROM          TOK(FROM)
GEOMETRY      TOK(GEOMETRY)
GRANT         TOK(GRANT)
GROUP         TOK(GROUP)
HAVING        TOK(HAVING)
IF            TOK(IF)
ILIKE         TOK(ILIKE)
IN            TOK(IN)
INSERT        { BEGIN STATE_INSERT; return SQLParser::INSERT; }
INT(EGER)?		TOK(INTEGER)
<STATE_INSERT>INTO    { BEGIN STATE_INSERT_INTO; return SQLParser::INTO; }
INTO          TOK(INTO)
IS            TOK(IS)
LANGUAGE      TOK(LANGUAGE)
LAST          TOK(LAST)
LENGTH        TOK(LENGTH)
LIKE          TOK(LIKE)
LIMIT         TOK(LIMIT)
LINESTRING    TOK(LINESTRING)
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
RESTORE       TOK(RESTORE)
REVOKE        TOK(REVOKE)
ROLE          TOK(ROLE)
ROLLBACK      TOK(ROLLBACK)
SCHEMA        TOK(SCHEMA)
SELECT        TOK(SELECT)
SERVER        TOK(SERVER)
SET           TOK(SET)
SHARD         TOK(SHARD)
SHARED        TOK(SHARED)
SHOW          TOK(SHOW)
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
USAGE         TOK(USAGE)
USER          TOK(USER)
VALUES        { BEGIN 0; return SQLParser::VALUES; }
VALIDATE      TOK(VALIDATE)
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

[-+*/(),.]	{ return yytext[0]; }

[;] { BEGIN 0; return yytext[0]; }

  /* validate type */
"CLUSTER"	{ yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext); TOK(CLUSTER) }

	/* names */
[A-Za-z_][A-Za-z0-9\$_]*	{
  yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext);
  TOK(NAME)
}

  /* names with dashes */
[A-Za-z_][A-Za-z0-9\$_\-]*	{ yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext); TOK(DASHEDNAME) }

	/* emails */
([^[:space:]\"]+|\".+\")@[A-Za-z0-9][A-Za-z0-9\-\.]*\.[A-Za-z]+	{ yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext); TOK(EMAIL) }

	/* parameters */
":"[A-Za-z][A-Za-z0-9_]*	TOK(PARAMETER)

	/* numbers */

-?[0-9]+ { yylval.intval = atoll(yytext); TOK(INTNUM) }
-?[0-9]+"."[0-9]* |
-?"."[0-9]*		{
		yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext);
    // numeric and decimal precisions cannot exceed 19
    if (yylval.stringval->get()->length() < 20) {
            return SQLParser::FIXEDNUM;
    }
    delete yylval.stringval->release();
    // otherwise return as double
    yylval.doubleval = std::stod(yytext);
    return SQLParser::DOUBLE;
	}

	/* TODO: handle FLOAT v.s. DOUBLE */
-?[0-9]+[eE][+-]?[0-9]+	|
-?[0-9]+"."[0-9]*[eE][+-]?[0-9]+ |
-?"."[0-9]*[eE][+-]?[0-9]+	{ yylval.doubleval = atof(yytext); TOK(DOUBLE) }

	/* strings */
<STATE_CREATE_AS>(.|\n|\r)+[ \n\t\r]+/([Ww][Ii][Tt]) {
    int len = yyleng;
    yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext, len);
    boost::trim(*yylval.stringval->get());
    
    BEGIN 0;
    return SQLParser::SELECTSTRING;
  }

<STATE_CREATE_AS>(.|\n|\r)+/[;] {
    int len = yyleng;
    std::string tmp(yytext, len);
    boost::regex keyword{"\\swith", boost::regex::icase};
    boost::smatch what;
    if (boost::regex_search(tmp, what, keyword)) {
      REJECT;
    } else {
      yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext, len);
      boost::trim(*yylval.stringval->get());
      BEGIN 0;
      return SQLParser::SELECTSTRING;
    }
  }

<STATE_INSERT_INTO>[( ]*([Ss][Ee][Ll][Ee][Cc][Tt]).+/[;] {
    int len = yyleng;
    std::string tmp(yytext, len);
    boost::regex keyword{"\\svalues", boost::regex::icase};
    boost::smatch what;
    if (boost::regex_search(tmp, what, keyword)) {
      REJECT;
    } else {
      yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext, len);
      boost::trim(*yylval.stringval->get());
      BEGIN 0;
      return SQLParser::SELECTSTRING;
    }
  }

#~#.*#~# {
		int len = yyleng - 6;
		if (len > 0) {
			yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, yytext + 3, len);
			boost::trim(*yylval.stringval->get());
		} else {
			yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, "");
		}
		return SQLParser::FWDSTR;
	}

'[^'\n]*' {
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

      yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, std::string{escape_removed});
      return SQLParser::STRING;
    }
    else
      yymore();
  }

'[^'\n]*$	{	throw std::runtime_error("Unterminated string"); }

\"[^\"\n]*\" {
    int c = yyFlexLexer::yyinput();

    unput(c);	/* just peeking */
    if(c != '"') {
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
          } else if (p[i+1] == '"') {
            escape_removed[j++] = '"';
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
        escape_removed[j++] = p[i];
      }
      escape_removed[j] = '\0';

      yylval.stringval = TrackedPtr<std::string>::make(parsed_str_tokens_, std::string{escape_removed});
      return SQLParser::QUOTED_IDENTIFIER;
    }
    else
      yymore();
  }

\"[^\"\n]*$	{	throw std::runtime_error("Unterminated string"); }


	/*
\n		{ lineno++; }
	*/

[ \n\t\r]+	;	/* white space */

"--".*	;	/* comment */


.			{ return yytext[0]; } /* random non-SQL text.  cause error in parser */

<<EOF>>		{ yyterminate(); }
%%
