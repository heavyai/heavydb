#define YY_Parser_h_included

/* with Bison++ version bison++ version 1.21-45, adapted from GNU Bison by
 * coetmeur@icdc.fr
 */

#line 1 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
/* Skeleton output parser for bison,
   Copyright (C) 1984, 1989, 1990 Bob Corbett and Richard Stallman

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 1, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */

/* HEADER SECTION */
#if defined(_MSDOS) || defined(MSDOS) || defined(__MSDOS__)
#define __MSDOS_AND_ALIKE
#endif
#if defined(_WINDOWS) && defined(_MSC_VER)
#define __HAVE_NO_ALLOCA
#define __MSDOS_AND_ALIKE
#endif

#ifndef alloca
#if defined(__GNUC__)
#define alloca __builtin_alloca

#elif (!defined(__STDC__) && defined(sparc)) || defined(__sparc__) || \
    defined(__sparc) || defined(__sgi)
#include <alloca.h>

#elif defined(__MSDOS_AND_ALIKE)
#include <malloc.h>
#ifndef __TURBOC__
/* MS C runtime lib */
#define alloca _alloca
#endif

#elif defined(_AIX)
#include <malloc.h>
#pragma alloca

#elif defined(__hpux)
#ifdef __cplusplus
extern "C" {
void* alloca(unsigned int);
};
#else  /* not __cplusplus */
void* alloca();
#endif /* not __cplusplus */

#endif /* not _AIX  not MSDOS, or __TURBOC__ or _AIX, not sparc.  */
#endif /* alloca not defined.  */
#ifdef c_plusplus
#ifndef __cplusplus
#define __cplusplus
#endif
#endif
#ifdef __cplusplus
#ifndef YY_USE_CLASS
#define YY_USE_CLASS
#endif
#else
#ifndef __STDC__
#define const
#endif
#endif
#include <stdio.h>
#define YYBISON 1

/* #line 73 "/usr/local/mapd-deps/20210608/lib/bison.cc" */
#define YY_Parser_CLASS SQLParser
#define YY_Parser_LVAL yylval
#define YY_Parser_CONSTRUCTOR_INIT  : lexer(yylval)
#define YY_Parser_MEMBERS                                                                     \
  virtual ~SQLParser() {}                                                                     \
  int parse(const std::string& inputStrOrig,                                                  \
            std::list<std::unique_ptr<Stmt>>& parseTrees,                                     \
            std::string& lastParsed) {                                                        \
    auto inputStr = boost::algorithm::trim_right_copy_if(                                     \
                        inputStrOrig, boost::is_any_of(";") || boost::is_space()) +           \
                    ";";                                                                      \
    boost::regex create_view_expr{                                                            \
        R"(CREATE\s+VIEW\s+(IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9\$_]*)\s+AS\s+(.*);?)", \
        boost::regex::extended | boost::regex::icase};                                        \
    std::lock_guard<std::mutex> lock(mutex_);                                                 \
    boost::smatch what;                                                                       \
    const auto trimmed_input = boost::algorithm::trim_copy(inputStr);                         \
    if (boost::regex_match(                                                                   \
            trimmed_input.cbegin(), trimmed_input.cend(), what, create_view_expr)) {          \
      const bool if_not_exists = what[1].length() > 0;                                        \
      const auto view_name = what[2].str();                                                   \
      const auto select_query = what[3].str();                                                \
      parseTrees.emplace_back(                                                                \
          new CreateViewStmt(view_name, select_query, if_not_exists));                        \
      return 0;                                                                               \
    }                                                                                         \
    std::istringstream ss(inputStr);                                                          \
    lexer.switch_streams(&ss, 0);                                                             \
    yyparse(parseTrees);                                                                      \
    lastParsed = lexer.YYText();                                                              \
    if (!errors_.empty()) {                                                                   \
      throw std::runtime_error(errors_[0]);                                                   \
    }                                                                                         \
    return yynerrs;                                                                           \
  }                                                                                           \
                                                                                              \
 private:                                                                                     \
  SQLLexer lexer;                                                                             \
  std::mutex mutex_;                                                                          \
  std::vector<std::string> errors_;
#define YY_Parser_LEX_BODY \
  { return lexer.yylex(); }
#define YY_Parser_ERROR_BODY                                                            \
  {} /*{ std::cerr << "Syntax error on line " << lexer.lineno() << ". Last word parsed: \
        " << lexer.YYText() << std::endl; } */

#include <FlexLexer.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/regex.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include "ParserNode.h"
#include "ReservedKeywords.h"
#include "TrackedPtr.h"

#ifdef DELETE
#undef DELETE
#endif

#ifdef IN
#undef IN
#endif

using namespace Parser;
#define YY_Parser_PARSE_PARAM std::list<std::unique_ptr<Stmt>>& parseTrees

typedef union {
  bool boolval;
  int64_t intval;
  float floatval;
  double doubleval;
  TrackedPtr<std::string>* stringval;
  SQLOps opval;
  SQLQualifier qualval;
  TrackedListPtr<Node>* listval;
  TrackedListPtr<std::string>* slistval;
  TrackedPtr<Node>* nodeval;
} yy_Parser_stype;
#define YY_Parser_STYPE yy_Parser_stype

class SQLLexer : public yyFlexLexer {
 public:
  SQLLexer(YY_Parser_STYPE& lval) : yylval(lval){};
  YY_Parser_STYPE& yylval;
  std::vector<std::unique_ptr<TrackedPtr<std::string>>> parsed_str_tokens_{};
  std::vector<std::unique_ptr<TrackedListPtr<std::string>>> parsed_str_list_tokens_{};
  std::vector<std::unique_ptr<TrackedPtr<Node>>> parsed_node_tokens_{};
  std::vector<std::unique_ptr<TrackedListPtr<Node>>> parsed_node_list_tokens_{};
};

#line 73 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/* %{ and %header{ and %union, during decl */
#define YY_Parser_BISON 1
#ifndef YY_Parser_COMPATIBILITY
#ifndef YY_USE_CLASS
#define YY_Parser_COMPATIBILITY 1
#else
#define YY_Parser_COMPATIBILITY 0
#endif
#endif

#if YY_Parser_COMPATIBILITY != 0
/* backward compatibility */
#ifdef YYLTYPE
#ifndef YY_Parser_LTYPE
#define YY_Parser_LTYPE YYLTYPE
#endif
#endif
#ifdef YYSTYPE
#ifndef YY_Parser_STYPE
#define YY_Parser_STYPE YYSTYPE
#endif
#endif
#ifdef YYDEBUG
#ifndef YY_Parser_DEBUG
#define YY_Parser_DEBUG YYDEBUG
#endif
#endif
#ifdef YY_Parser_STYPE
#ifndef yystype
#define yystype YY_Parser_STYPE
#endif
#endif
/* use goto to be compatible */
#ifndef YY_Parser_USE_GOTO
#define YY_Parser_USE_GOTO 1
#endif
#endif

/* use no goto to be clean in C++ */
#ifndef YY_Parser_USE_GOTO
#define YY_Parser_USE_GOTO 0
#endif

#ifndef YY_Parser_PURE

/* #line 117 "/usr/local/mapd-deps/20210608/lib/bison.cc" */

#line 117 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/*  YY_Parser_PURE */
#endif

/* section apres lecture def, avant lecture grammaire S2 */

/* #line 121 "/usr/local/mapd-deps/20210608/lib/bison.cc" */

#line 121 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/* prefix */
#ifndef YY_Parser_DEBUG

/* #line 123 "/usr/local/mapd-deps/20210608/lib/bison.cc" */

#line 123 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/* YY_Parser_DEBUG */
#endif

#ifndef YY_Parser_LSP_NEEDED

/* #line 128 "/usr/local/mapd-deps/20210608/lib/bison.cc" */

#line 128 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/* YY_Parser_LSP_NEEDED*/
#endif

/* DEFAULT LTYPE*/
#ifdef YY_Parser_LSP_NEEDED
#ifndef YY_Parser_LTYPE
typedef struct yyltype {
  int timestamp;
  int first_line;
  int first_column;
  int last_line;
  int last_column;
  char* text;
} yyltype;

#define YY_Parser_LTYPE yyltype
#endif
#endif
/* DEFAULT STYPE*/
/* We used to use `unsigned long' as YY_Parser_STYPE on MSDOS,
   but it seems better to be consistent.
   Most programs should declare their own type anyway.  */

#ifndef YY_Parser_STYPE
#define YY_Parser_STYPE int
#endif
/* DEFAULT MISCELANEOUS */
#ifndef YY_Parser_PARSE
#define YY_Parser_PARSE yyparse
#endif
#ifndef YY_Parser_LEX
#define YY_Parser_LEX yylex
#endif
#ifndef YY_Parser_LVAL
#define YY_Parser_LVAL yylval
#endif
#ifndef YY_Parser_LLOC
#define YY_Parser_LLOC yylloc
#endif
#ifndef YY_Parser_CHAR
#define YY_Parser_CHAR yychar
#endif
#ifndef YY_Parser_NERRS
#define YY_Parser_NERRS yynerrs
#endif
#ifndef YY_Parser_DEBUG_FLAG
#define YY_Parser_DEBUG_FLAG yydebug
#endif
#ifndef YY_Parser_ERROR
#define YY_Parser_ERROR yyerror
#endif
#ifndef YY_Parser_PARSE_PARAM
#ifndef __STDC__
#ifndef __cplusplus
#ifndef YY_USE_CLASS
#define YY_Parser_PARSE_PARAM
#ifndef YY_Parser_PARSE_PARAM_DEF
#define YY_Parser_PARSE_PARAM_DEF
#endif
#endif
#endif
#endif
#ifndef YY_Parser_PARSE_PARAM
#define YY_Parser_PARSE_PARAM void
#endif
#endif
#if YY_Parser_COMPATIBILITY != 0
/* backward compatibility */
#ifdef YY_Parser_LTYPE
#ifndef YYLTYPE
#define YYLTYPE YY_Parser_LTYPE
#else
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
#endif
#endif
#ifndef YYSTYPE
#define YYSTYPE YY_Parser_STYPE
#else
/* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
#endif
#ifdef YY_Parser_PURE
#ifndef YYPURE
#define YYPURE YY_Parser_PURE
#endif
#endif
#ifdef YY_Parser_DEBUG
#ifndef YYDEBUG
#define YYDEBUG YY_Parser_DEBUG
#endif
#endif
#ifndef YY_Parser_ERROR_VERBOSE
#ifdef YYERROR_VERBOSE
#define YY_Parser_ERROR_VERBOSE YYERROR_VERBOSE
#endif
#endif
#ifndef YY_Parser_LSP_NEEDED
#ifdef YYLSP_NEEDED
#define YY_Parser_LSP_NEEDED YYLSP_NEEDED
#endif
#endif
#endif
#ifndef YY_USE_CLASS
/* TOKEN C */

/* #line 236 "/usr/local/mapd-deps/20210608/lib/bison.cc" */
#define NAME 258
#define DASHEDNAME 259
#define EMAIL 260
#define STRING 261
#define FWDSTR 262
#define SELECTSTRING 263
#define QUOTED_IDENTIFIER 264
#define INTNUM 265
#define FIXEDNUM 266
#define OR 267
#define AND 268
#define NOT 269
#define EQUAL 270
#define COMPARISON 271
#define UMINUS 272
#define ADD 273
#define ALL 274
#define ALTER 275
#define AMMSC 276
#define ANY 277
#define ARCHIVE 278
#define ARRAY 279
#define AS 280
#define ASC 281
#define AUTHORIZATION 282
#define BETWEEN 283
#define BIGINT 284
#define BOOLEAN 285
#define BY 286
#define CASE 287
#define CAST 288
#define CHAR_LENGTH 289
#define CHARACTER 290
#define CHECK 291
#define CLOSE 292
#define CLUSTER 293
#define COLUMN 294
#define COMMIT 295
#define CONTINUE 296
#define COPY 297
#define CREATE 298
#define CURRENT 299
#define CURSOR 300
#define DATABASE 301
#define DATAFRAME 302
#define DATE 303
#define DATETIME 304
#define DATE_TRUNC 305
#define DECIMAL 306
#define DECLARE 307
#define DEFAULT 308
#define DELETE 309
#define DESC 310
#define DICTIONARY 311
#define DISTINCT 312
#define DOUBLE 313
#define DROP 314
#define DUMP 315
#define ELSE 316
#define END 317
#define EXISTS 318
#define EXTRACT 319
#define FETCH 320
#define FIRST 321
#define FLOAT 322
#define FOR 323
#define FOREIGN 324
#define FOUND 325
#define FROM 326
#define GEOGRAPHY 327
#define GEOMETRY 328
#define GRANT 329
#define GROUP 330
#define HAVING 331
#define IF 332
#define ILIKE 333
#define IN 334
#define INSERT 335
#define INTEGER 336
#define INTO 337
#define IS 338
#define LANGUAGE 339
#define LAST 340
#define LENGTH 341
#define LIKE 342
#define LIMIT 343
#define LINESTRING 344
#define MOD 345
#define MULTIPOLYGON 346
#define NOW 347
#define NULLX 348
#define NUMERIC 349
#define OF 350
#define OFFSET 351
#define ON 352
#define OPEN 353
#define OPTIMIZE 354
#define OPTIMIZED 355
#define OPTION 356
#define ORDER 357
#define PARAMETER 358
#define POINT 359
#define POLYGON 360
#define PRECISION 361
#define PRIMARY 362
#define PRIVILEGES 363
#define PROCEDURE 364
#define SERVER 365
#define SMALLINT 366
#define SOME 367
#define TABLE 368
#define TEMPORARY 369
#define TEXT 370
#define THEN 371
#define TIME 372
#define TIMESTAMP 373
#define TINYINT 374
#define TO 375
#define TRUNCATE 376
#define UNION 377
#define USAGE 378
#define PUBLIC 379
#define REAL 380
#define REFERENCES 381
#define RENAME 382
#define RESTORE 383
#define REVOKE 384
#define ROLE 385
#define ROLLBACK 386
#define SCHEMA 387
#define SELECT 388
#define SET 389
#define SHARD 390
#define SHARED 391
#define SHOW 392
#define UNIQUE 393
#define UPDATE 394
#define USER 395
#define VALIDATE 396
#define VALUES 397
#define VIEW 398
#define WHEN 399
#define WHENEVER 400
#define WHERE 401
#define WITH 402
#define WORK 403
#define EDIT 404
#define ACCESS 405
#define DASHBOARD 406
#define SQL 407
#define EDITOR 408

#line 236 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/* #defines tokens */
#else
/* CLASS */
#ifndef YY_Parser_CLASS
#define YY_Parser_CLASS Parser
#endif
#ifndef YY_Parser_INHERIT
#define YY_Parser_INHERIT
#endif
#ifndef YY_Parser_MEMBERS
#define YY_Parser_MEMBERS
#endif
#ifndef YY_Parser_LEX_BODY
#define YY_Parser_LEX_BODY
#endif
#ifndef YY_Parser_ERROR_BODY
#define YY_Parser_ERROR_BODY
#endif
#ifndef YY_Parser_CONSTRUCTOR_PARAM
#define YY_Parser_CONSTRUCTOR_PARAM
#endif
#ifndef YY_Parser_CONSTRUCTOR_CODE
#define YY_Parser_CONSTRUCTOR_CODE
#endif
#ifndef YY_Parser_CONSTRUCTOR_INIT
#define YY_Parser_CONSTRUCTOR_INIT
#endif
/* choose between enum and const */
#ifndef YY_Parser_USE_CONST_TOKEN
#define YY_Parser_USE_CONST_TOKEN 0
/* yes enum is more compatible with flex,  */
/* so by default we use it */
#endif
#if YY_Parser_USE_CONST_TOKEN != 0
#ifndef YY_Parser_ENUM_TOKEN
#define YY_Parser_ENUM_TOKEN yy_Parser_enum_token
#endif
#endif

class YY_Parser_CLASS YY_Parser_INHERIT {
 public:
#if YY_Parser_USE_CONST_TOKEN != 0
  /* static const int token ... */

  /* #line 280 "/usr/local/mapd-deps/20210608/lib/bison.cc" */
  static const int NAME;
  static const int DASHEDNAME;
  static const int EMAIL;
  static const int STRING;
  static const int FWDSTR;
  static const int SELECTSTRING;
  static const int QUOTED_IDENTIFIER;
  static const int INTNUM;
  static const int FIXEDNUM;
  static const int OR;
  static const int AND;
  static const int NOT;
  static const int EQUAL;
  static const int COMPARISON;
  static const int UMINUS;
  static const int ADD;
  static const int ALL;
  static const int ALTER;
  static const int AMMSC;
  static const int ANY;
  static const int ARCHIVE;
  static const int ARRAY;
  static const int AS;
  static const int ASC;
  static const int AUTHORIZATION;
  static const int BETWEEN;
  static const int BIGINT;
  static const int BOOLEAN;
  static const int BY;
  static const int CASE;
  static const int CAST;
  static const int CHAR_LENGTH;
  static const int CHARACTER;
  static const int CHECK;
  static const int CLOSE;
  static const int CLUSTER;
  static const int COLUMN;
  static const int COMMIT;
  static const int CONTINUE;
  static const int COPY;
  static const int CREATE;
  static const int CURRENT;
  static const int CURSOR;
  static const int DATABASE;
  static const int DATAFRAME;
  static const int DATE;
  static const int DATETIME;
  static const int DATE_TRUNC;
  static const int DECIMAL;
  static const int DECLARE;
  static const int DEFAULT;
  static const int DELETE;
  static const int DESC;
  static const int DICTIONARY;
  static const int DISTINCT;
  static const int DOUBLE;
  static const int DROP;
  static const int DUMP;
  static const int ELSE;
  static const int END;
  static const int EXISTS;
  static const int EXTRACT;
  static const int FETCH;
  static const int FIRST;
  static const int FLOAT;
  static const int FOR;
  static const int FOREIGN;
  static const int FOUND;
  static const int FROM;
  static const int GEOGRAPHY;
  static const int GEOMETRY;
  static const int GRANT;
  static const int GROUP;
  static const int HAVING;
  static const int IF;
  static const int ILIKE;
  static const int IN;
  static const int INSERT;
  static const int INTEGER;
  static const int INTO;
  static const int IS;
  static const int LANGUAGE;
  static const int LAST;
  static const int LENGTH;
  static const int LIKE;
  static const int LIMIT;
  static const int LINESTRING;
  static const int MOD;
  static const int MULTIPOLYGON;
  static const int NOW;
  static const int NULLX;
  static const int NUMERIC;
  static const int OF;
  static const int OFFSET;
  static const int ON;
  static const int OPEN;
  static const int OPTIMIZE;
  static const int OPTIMIZED;
  static const int OPTION;
  static const int ORDER;
  static const int PARAMETER;
  static const int POINT;
  static const int POLYGON;
  static const int PRECISION;
  static const int PRIMARY;
  static const int PRIVILEGES;
  static const int PROCEDURE;
  static const int SERVER;
  static const int SMALLINT;
  static const int SOME;
  static const int TABLE;
  static const int TEMPORARY;
  static const int TEXT;
  static const int THEN;
  static const int TIME;
  static const int TIMESTAMP;
  static const int TINYINT;
  static const int TO;
  static const int TRUNCATE;
  static const int UNION;
  static const int USAGE;
  static const int PUBLIC;
  static const int REAL;
  static const int REFERENCES;
  static const int RENAME;
  static const int RESTORE;
  static const int REVOKE;
  static const int ROLE;
  static const int ROLLBACK;
  static const int SCHEMA;
  static const int SELECT;
  static const int SET;
  static const int SHARD;
  static const int SHARED;
  static const int SHOW;
  static const int UNIQUE;
  static const int UPDATE;
  static const int USER;
  static const int VALIDATE;
  static const int VALUES;
  static const int VIEW;
  static const int WHEN;
  static const int WHENEVER;
  static const int WHERE;
  static const int WITH;
  static const int WORK;
  static const int EDIT;
  static const int ACCESS;
  static const int DASHBOARD;
  static const int SQL;
  static const int EDITOR;

#line 280 "/usr/local/mapd-deps/20210608/lib/bison.cc"
  /* decl const */
#else
  enum YY_Parser_ENUM_TOKEN {
    YY_Parser_NULL_TOKEN = 0

    /* #line 283 "/usr/local/mapd-deps/20210608/lib/bison.cc" */
    ,
    NAME = 258,
    DASHEDNAME = 259,
    EMAIL = 260,
    STRING = 261,
    FWDSTR = 262,
    SELECTSTRING = 263,
    QUOTED_IDENTIFIER = 264,
    INTNUM = 265,
    FIXEDNUM = 266,
    OR = 267,
    AND = 268,
    NOT = 269,
    EQUAL = 270,
    COMPARISON = 271,
    UMINUS = 272,
    ADD = 273,
    ALL = 274,
    ALTER = 275,
    AMMSC = 276,
    ANY = 277,
    ARCHIVE = 278,
    ARRAY = 279,
    AS = 280,
    ASC = 281,
    AUTHORIZATION = 282,
    BETWEEN = 283,
    BIGINT = 284,
    BOOLEAN = 285,
    BY = 286,
    CASE = 287,
    CAST = 288,
    CHAR_LENGTH = 289,
    CHARACTER = 290,
    CHECK = 291,
    CLOSE = 292,
    CLUSTER = 293,
    COLUMN = 294,
    COMMIT = 295,
    CONTINUE = 296,
    COPY = 297,
    CREATE = 298,
    CURRENT = 299,
    CURSOR = 300,
    DATABASE = 301,
    DATAFRAME = 302,
    DATE = 303,
    DATETIME = 304,
    DATE_TRUNC = 305,
    DECIMAL = 306,
    DECLARE = 307,
    DEFAULT = 308,
    DELETE = 309,
    DESC = 310,
    DICTIONARY = 311,
    DISTINCT = 312,
    DOUBLE = 313,
    DROP = 314,
    DUMP = 315,
    ELSE = 316,
    END = 317,
    EXISTS = 318,
    EXTRACT = 319,
    FETCH = 320,
    FIRST = 321,
    FLOAT = 322,
    FOR = 323,
    FOREIGN = 324,
    FOUND = 325,
    FROM = 326,
    GEOGRAPHY = 327,
    GEOMETRY = 328,
    GRANT = 329,
    GROUP = 330,
    HAVING = 331,
    IF = 332,
    ILIKE = 333,
    IN = 334,
    INSERT = 335,
    INTEGER = 336,
    INTO = 337,
    IS = 338,
    LANGUAGE = 339,
    LAST = 340,
    LENGTH = 341,
    LIKE = 342,
    LIMIT = 343,
    LINESTRING = 344,
    MOD = 345,
    MULTIPOLYGON = 346,
    NOW = 347,
    NULLX = 348,
    NUMERIC = 349,
    OF = 350,
    OFFSET = 351,
    ON = 352,
    OPEN = 353,
    OPTIMIZE = 354,
    OPTIMIZED = 355,
    OPTION = 356,
    ORDER = 357,
    PARAMETER = 358,
    POINT = 359,
    POLYGON = 360,
    PRECISION = 361,
    PRIMARY = 362,
    PRIVILEGES = 363,
    PROCEDURE = 364,
    SERVER = 365,
    SMALLINT = 366,
    SOME = 367,
    TABLE = 368,
    TEMPORARY = 369,
    TEXT = 370,
    THEN = 371,
    TIME = 372,
    TIMESTAMP = 373,
    TINYINT = 374,
    TO = 375,
    TRUNCATE = 376,
    UNION = 377,
    USAGE = 378,
    PUBLIC = 379,
    REAL = 380,
    REFERENCES = 381,
    RENAME = 382,
    RESTORE = 383,
    REVOKE = 384,
    ROLE = 385,
    ROLLBACK = 386,
    SCHEMA = 387,
    SELECT = 388,
    SET = 389,
    SHARD = 390,
    SHARED = 391,
    SHOW = 392,
    UNIQUE = 393,
    UPDATE = 394,
    USER = 395,
    VALIDATE = 396,
    VALUES = 397,
    VIEW = 398,
    WHEN = 399,
    WHENEVER = 400,
    WHERE = 401,
    WITH = 402,
    WORK = 403,
    EDIT = 404,
    ACCESS = 405,
    DASHBOARD = 406,
    SQL = 407,
    EDITOR = 408

#line 283 "/usr/local/mapd-deps/20210608/lib/bison.cc"
    /* enum token */
  }; /* end of enum declaration */
#endif
 public:
  int YY_Parser_PARSE(YY_Parser_PARSE_PARAM);
  virtual void YY_Parser_ERROR(const char* msg) YY_Parser_ERROR_BODY;
#ifdef YY_Parser_PURE
#ifdef YY_Parser_LSP_NEEDED
  virtual int YY_Parser_LEX(YY_Parser_STYPE* YY_Parser_LVAL,
                            YY_Parser_LTYPE* YY_Parser_LLOC) YY_Parser_LEX_BODY;
#else
  virtual int YY_Parser_LEX(YY_Parser_STYPE* YY_Parser_LVAL) YY_Parser_LEX_BODY;
#endif
#else
  virtual int YY_Parser_LEX() YY_Parser_LEX_BODY;
  YY_Parser_STYPE YY_Parser_LVAL;
#ifdef YY_Parser_LSP_NEEDED
  YY_Parser_LTYPE YY_Parser_LLOC;
#endif
  int YY_Parser_NERRS;
  int YY_Parser_CHAR;
#endif
#if YY_Parser_DEBUG != 0
  int YY_Parser_DEBUG_FLAG; /*  nonzero means print parse trace     */
#endif
 public:
  YY_Parser_CLASS(YY_Parser_CONSTRUCTOR_PARAM);

 public:
  YY_Parser_MEMBERS
};
/* other declare folow */
#if YY_Parser_USE_CONST_TOKEN != 0

/* #line 314 "/usr/local/mapd-deps/20210608/lib/bison.cc" */
const int YY_Parser_CLASS::NAME = 258;
const int YY_Parser_CLASS::DASHEDNAME = 259;
const int YY_Parser_CLASS::EMAIL = 260;
const int YY_Parser_CLASS::STRING = 261;
const int YY_Parser_CLASS::FWDSTR = 262;
const int YY_Parser_CLASS::SELECTSTRING = 263;
const int YY_Parser_CLASS::QUOTED_IDENTIFIER = 264;
const int YY_Parser_CLASS::INTNUM = 265;
const int YY_Parser_CLASS::FIXEDNUM = 266;
const int YY_Parser_CLASS::OR = 267;
const int YY_Parser_CLASS::AND = 268;
const int YY_Parser_CLASS::NOT = 269;
const int YY_Parser_CLASS::EQUAL = 270;
const int YY_Parser_CLASS::COMPARISON = 271;
const int YY_Parser_CLASS::UMINUS = 272;
const int YY_Parser_CLASS::ADD = 273;
const int YY_Parser_CLASS::ALL = 274;
const int YY_Parser_CLASS::ALTER = 275;
const int YY_Parser_CLASS::AMMSC = 276;
const int YY_Parser_CLASS::ANY = 277;
const int YY_Parser_CLASS::ARCHIVE = 278;
const int YY_Parser_CLASS::ARRAY = 279;
const int YY_Parser_CLASS::AS = 280;
const int YY_Parser_CLASS::ASC = 281;
const int YY_Parser_CLASS::AUTHORIZATION = 282;
const int YY_Parser_CLASS::BETWEEN = 283;
const int YY_Parser_CLASS::BIGINT = 284;
const int YY_Parser_CLASS::BOOLEAN = 285;
const int YY_Parser_CLASS::BY = 286;
const int YY_Parser_CLASS::CASE = 287;
const int YY_Parser_CLASS::CAST = 288;
const int YY_Parser_CLASS::CHAR_LENGTH = 289;
const int YY_Parser_CLASS::CHARACTER = 290;
const int YY_Parser_CLASS::CHECK = 291;
const int YY_Parser_CLASS::CLOSE = 292;
const int YY_Parser_CLASS::CLUSTER = 293;
const int YY_Parser_CLASS::COLUMN = 294;
const int YY_Parser_CLASS::COMMIT = 295;
const int YY_Parser_CLASS::CONTINUE = 296;
const int YY_Parser_CLASS::COPY = 297;
const int YY_Parser_CLASS::CREATE = 298;
const int YY_Parser_CLASS::CURRENT = 299;
const int YY_Parser_CLASS::CURSOR = 300;
const int YY_Parser_CLASS::DATABASE = 301;
const int YY_Parser_CLASS::DATAFRAME = 302;
const int YY_Parser_CLASS::DATE = 303;
const int YY_Parser_CLASS::DATETIME = 304;
const int YY_Parser_CLASS::DATE_TRUNC = 305;
const int YY_Parser_CLASS::DECIMAL = 306;
const int YY_Parser_CLASS::DECLARE = 307;
const int YY_Parser_CLASS::DEFAULT = 308;
const int YY_Parser_CLASS::DELETE = 309;
const int YY_Parser_CLASS::DESC = 310;
const int YY_Parser_CLASS::DICTIONARY = 311;
const int YY_Parser_CLASS::DISTINCT = 312;
const int YY_Parser_CLASS::DOUBLE = 313;
const int YY_Parser_CLASS::DROP = 314;
const int YY_Parser_CLASS::DUMP = 315;
const int YY_Parser_CLASS::ELSE = 316;
const int YY_Parser_CLASS::END = 317;
const int YY_Parser_CLASS::EXISTS = 318;
const int YY_Parser_CLASS::EXTRACT = 319;
const int YY_Parser_CLASS::FETCH = 320;
const int YY_Parser_CLASS::FIRST = 321;
const int YY_Parser_CLASS::FLOAT = 322;
const int YY_Parser_CLASS::FOR = 323;
const int YY_Parser_CLASS::FOREIGN = 324;
const int YY_Parser_CLASS::FOUND = 325;
const int YY_Parser_CLASS::FROM = 326;
const int YY_Parser_CLASS::GEOGRAPHY = 327;
const int YY_Parser_CLASS::GEOMETRY = 328;
const int YY_Parser_CLASS::GRANT = 329;
const int YY_Parser_CLASS::GROUP = 330;
const int YY_Parser_CLASS::HAVING = 331;
const int YY_Parser_CLASS::IF = 332;
const int YY_Parser_CLASS::ILIKE = 333;
const int YY_Parser_CLASS::IN = 334;
const int YY_Parser_CLASS::INSERT = 335;
const int YY_Parser_CLASS::INTEGER = 336;
const int YY_Parser_CLASS::INTO = 337;
const int YY_Parser_CLASS::IS = 338;
const int YY_Parser_CLASS::LANGUAGE = 339;
const int YY_Parser_CLASS::LAST = 340;
const int YY_Parser_CLASS::LENGTH = 341;
const int YY_Parser_CLASS::LIKE = 342;
const int YY_Parser_CLASS::LIMIT = 343;
const int YY_Parser_CLASS::LINESTRING = 344;
const int YY_Parser_CLASS::MOD = 345;
const int YY_Parser_CLASS::MULTIPOLYGON = 346;
const int YY_Parser_CLASS::NOW = 347;
const int YY_Parser_CLASS::NULLX = 348;
const int YY_Parser_CLASS::NUMERIC = 349;
const int YY_Parser_CLASS::OF = 350;
const int YY_Parser_CLASS::OFFSET = 351;
const int YY_Parser_CLASS::ON = 352;
const int YY_Parser_CLASS::OPEN = 353;
const int YY_Parser_CLASS::OPTIMIZE = 354;
const int YY_Parser_CLASS::OPTIMIZED = 355;
const int YY_Parser_CLASS::OPTION = 356;
const int YY_Parser_CLASS::ORDER = 357;
const int YY_Parser_CLASS::PARAMETER = 358;
const int YY_Parser_CLASS::POINT = 359;
const int YY_Parser_CLASS::POLYGON = 360;
const int YY_Parser_CLASS::PRECISION = 361;
const int YY_Parser_CLASS::PRIMARY = 362;
const int YY_Parser_CLASS::PRIVILEGES = 363;
const int YY_Parser_CLASS::PROCEDURE = 364;
const int YY_Parser_CLASS::SERVER = 365;
const int YY_Parser_CLASS::SMALLINT = 366;
const int YY_Parser_CLASS::SOME = 367;
const int YY_Parser_CLASS::TABLE = 368;
const int YY_Parser_CLASS::TEMPORARY = 369;
const int YY_Parser_CLASS::TEXT = 370;
const int YY_Parser_CLASS::THEN = 371;
const int YY_Parser_CLASS::TIME = 372;
const int YY_Parser_CLASS::TIMESTAMP = 373;
const int YY_Parser_CLASS::TINYINT = 374;
const int YY_Parser_CLASS::TO = 375;
const int YY_Parser_CLASS::TRUNCATE = 376;
const int YY_Parser_CLASS::UNION = 377;
const int YY_Parser_CLASS::USAGE = 378;
const int YY_Parser_CLASS::PUBLIC = 379;
const int YY_Parser_CLASS::REAL = 380;
const int YY_Parser_CLASS::REFERENCES = 381;
const int YY_Parser_CLASS::RENAME = 382;
const int YY_Parser_CLASS::RESTORE = 383;
const int YY_Parser_CLASS::REVOKE = 384;
const int YY_Parser_CLASS::ROLE = 385;
const int YY_Parser_CLASS::ROLLBACK = 386;
const int YY_Parser_CLASS::SCHEMA = 387;
const int YY_Parser_CLASS::SELECT = 388;
const int YY_Parser_CLASS::SET = 389;
const int YY_Parser_CLASS::SHARD = 390;
const int YY_Parser_CLASS::SHARED = 391;
const int YY_Parser_CLASS::SHOW = 392;
const int YY_Parser_CLASS::UNIQUE = 393;
const int YY_Parser_CLASS::UPDATE = 394;
const int YY_Parser_CLASS::USER = 395;
const int YY_Parser_CLASS::VALIDATE = 396;
const int YY_Parser_CLASS::VALUES = 397;
const int YY_Parser_CLASS::VIEW = 398;
const int YY_Parser_CLASS::WHEN = 399;
const int YY_Parser_CLASS::WHENEVER = 400;
const int YY_Parser_CLASS::WHERE = 401;
const int YY_Parser_CLASS::WITH = 402;
const int YY_Parser_CLASS::WORK = 403;
const int YY_Parser_CLASS::EDIT = 404;
const int YY_Parser_CLASS::ACCESS = 405;
const int YY_Parser_CLASS::DASHBOARD = 406;
const int YY_Parser_CLASS::SQL = 407;
const int YY_Parser_CLASS::EDITOR = 408;

#line 314 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/* const YY_Parser_CLASS::token */
#endif
/*apres const  */
YY_Parser_CLASS::YY_Parser_CLASS(YY_Parser_CONSTRUCTOR_PARAM) YY_Parser_CONSTRUCTOR_INIT {
#if YY_Parser_DEBUG != 0
  YY_Parser_DEBUG_FLAG = 0;
#endif
  YY_Parser_CONSTRUCTOR_CODE;
};
#endif

/* #line 325 "/usr/local/mapd-deps/20210608/lib/bison.cc" */

#define YYFINAL 512
#define YYFLAG -32768
#define YYNTBASE 168

#define YYTRANSLATE(x) ((unsigned)(x) <= 408 ? yytranslate[x] : 258)

static const short yytranslate[] = {
    0,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   21,  2,   2,   161, 162, 19,  17,  160, 18,  167, 20,  2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   159, 2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   163, 2,   164, 2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   165, 2,   166, 2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
    39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
    56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
    73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
    90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
    158};

#if YY_Parser_DEBUG != 0
static const short yyprhs[] = {
    0,   0,    3,    7,    9,    11,   13,  15,  17,  19,  21,  23,  25,  29,  33,  37,
    38,  40,   41,   51,   62,   65,   66,  71,  78,  87,  88,  90,  92,  96,  103, 111,
    116, 118,  122,  126,  132,  134,  138, 140, 142, 146, 151, 154, 160, 161, 164, 168,
    173, 176,  179,  182,  187,  190,  196, 201, 207, 215, 226, 232, 243, 248, 250, 254,
    259, 260,  265,  266,  270,  271,  275, 277, 281, 285, 289, 290, 292, 294, 295, 298,
    301, 303,  305,  307,  309,  311,  316, 325, 326, 328, 330, 332, 336, 340, 346, 347,
    349, 352,  355,  356,  359,  363,  364, 369, 371, 375, 380, 382, 386, 394, 396, 398,
    401, 403,  407,  409,  412,  415,  416, 420, 422, 426, 427, 430, 434, 438, 441, 445,
    447, 449,  451,  453,  455,  457,  459, 461, 463, 467, 471, 478, 484, 490, 495, 501,
    506, 507,  510,  515,  519,  524,  528, 535, 541, 543, 547, 552, 557, 559, 561, 563,
    565, 567,  570,  574,  579,  585,  588, 589, 594, 603, 610, 615, 620, 625, 629, 633,
    637, 641,  645,  652,  655,  658,  660, 662, 664, 668, 675, 677, 679, 681, 682, 684,
    687, 691,  692,  694,  698,  700,  702, 707, 713, 719, 724, 726, 728, 732, 737, 739,
    741, 743,  746,  750,  755,  757,  759, 763, 764, 766, 768, 770, 771, 773, 775, 777,
    779, 781,  783,  787,  789,  791,  793, 797, 799, 801, 803, 807, 809, 812, 814, 816,
    818, 820,  822,  824,  826,  828,  830, 832, 834, 836, 839, 842, 845, 848, 851, 854,
    857, 860,  863,  866,  869,  872,  876, 878, 880, 882, 884, 886, 888, 890, 892, 896,
    900, 902,  904,  906,  908,  910,  915, 917, 922, 929, 931, 936, 943, 945, 947, 949,
    951, 953,  956,  958,  960,  962,  967, 969, 974, 976, 978, 982, 987, 989, 991, 993,
    995, 1000, 1007, 1012, 1019, 1021, 1023};

static const short yyrhs[] = {
    169, 159, 0,   168, 169, 159, 0,   174, 0,   194, 0,   176, 0,   177, 0,   178, 0,
    181, 0,   182, 0,   185, 0,   171, 0,   170, 160, 171, 0,   3,   15,  247, 0,   82,
    14,  68,  0,   0,   119, 0,   0,   48,  173, 118, 172, 250, 161, 186, 162, 193, 0,
    48,  48,  3,   118, 172, 250, 161, 186, 162, 193, 0,   82,  68,  0,   0,   64,  118,
    175, 250, 0,   25,  118, 250, 132, 125, 250, 0,   25,  118, 250, 132, 44,  256, 125,
    256, 0,   0,   44,  0,   188, 0,   180, 160, 188, 0,   25,  118, 250, 23,  179, 188,
    0,   25,  118, 250, 23,  161, 180, 162, 0,   25,  118, 250, 183, 0,   184, 0,   183,
    160, 184, 0,   64,  179, 256, 0,   25,  118, 250, 139, 171, 0,   187, 0,   186, 160,
    187, 0,   188, 0,   191, 0,   256, 253, 189, 0,   256, 253, 190, 189, 0,   3,   3,
    0,   3,   3,   161, 10,  162, 0,   0,   14,  98,  0,   14,  98,  143, 0,   14,  98,
    112, 3,   0,   58,  247, 0,   58,  98,  0,   58,  145, 0,   41,  161, 223, 162, 0,
    131, 250, 0,   131, 250, 161, 256, 162, 0,   143, 161, 192, 162, 0,   112, 3,   161,
    192, 162, 0,   74,  3,   161, 192, 162, 131, 250, 0,   74,  3,   161, 192, 162, 131,
    250, 161, 192, 162, 0,   140, 3,   161, 256, 162, 0,   141, 61,  161, 256, 162, 131,
    250, 161, 256, 162, 0,   41,  161, 223, 162, 0,   256, 0,   192, 160, 256, 0,   152,
    161, 170, 162, 0,   0,   64,  148, 175, 250, 0,   0,   161, 192, 162, 0,   0,   107,
    36,  197, 0,   198, 0,   197, 160, 198, 0,   10,  199, 200, 0,   251, 199, 200, 0,
    0,   31,  0,   60,  0,   0,   98,  71,  0,   98,  90,  0,   201, 0,   202, 0,   203,
    0,   211, 0,   207, 0,   59,  76,  250, 208, 0,   85,  87,  250, 195, 147, 161, 231,
    162, 0,   0,   24,  0,   62,  0,   206, 0,   205, 160, 206, 0,   256, 15,  223, 0,
    144, 250, 139, 205, 208, 0,   0,   219, 0,   93,  10,  0,   93,  24,  0,   0,   101,
    10,  0,   101, 10,  3,   0,   0,   212, 196, 209, 210, 0,   213, 0,   212, 127, 213,
    0,   212, 127, 24,  213, 0,   214, 0,   161, 212, 162, 0,   138, 204, 215, 216, 208,
    220, 222, 0,   244, 0,   19,  0,   76,  217, 0,   218, 0,   217, 160, 218, 0,   250,
    0,   250, 257, 0,   151, 223, 0,   0,   80,  36,  221, 0,   223, 0,   221, 160, 223,
    0,   0,   81,  223, 0,   223, 12,  223, 0,   223, 13,  223, 0,   14,  223, 0,   161,
    223, 162, 0,   224, 0,   225, 0,   226, 0,   227, 0,   229, 0,   230, 0,   232, 0,
    235, 0,   242, 0,   242, 233, 242, 0,   242, 233, 236, 0,   242, 14,  33,  242, 13,
    242, 0,   242, 33,  242, 13,  242, 0,   242, 14,  92,  245, 228, 0,   242, 92,  245,
    228, 0,   242, 14,  83,  245, 228, 0,   242, 83,  245, 228, 0,   0,   3,   245, 0,
    251, 88,  14,  98,  0,   251, 88,  98,  0,   242, 14,  84,  236, 0,   242, 84,  236,
    0,   242, 14,  84,  161, 231, 162, 0,   242, 84,  161, 231, 162, 0,   245, 0,   231,
    160, 245, 0,   242, 233, 234, 236, 0,   242, 233, 234, 242, 0,   15,  0,   16,  0,
    27,  0,   24,  0,   117, 0,   68,  236, 0,   161, 214, 162, 0,   149, 223, 121, 223,
    0,   237, 149, 223, 121, 223, 0,   66,  223, 0,   0,   37,  237, 238, 67,  0,   82,
    161, 223, 160, 223, 160, 223, 162, 0,   82,  161, 223, 160, 223, 162, 0,   39,  161,
    242, 162, 0,   91,  161, 242, 162, 0,   251, 163, 242, 164, 0,   242, 17,  242, 0,
    242, 18,  242, 0,   242, 19,  242, 0,   242, 20,  242, 0,   242, 21,  242, 0,   95,
    161, 242, 160, 242, 162, 0,   17,  242, 0,   18,  242, 0,   245, 0,   251, 0,   246,
    0,   161, 242, 162, 0,   38,  161, 223, 30,  253, 162, 0,   239, 0,   240, 0,   241,
    0,   0,   223, 0,   223, 3,   0,   223, 30,  3,   0,   0,   243, 0,   244, 160, 243,
    0,   247, 0,   145, 0,   3,   161, 19,  162, 0,   3,   161, 62,  223, 162, 0,   3,
    161, 24,  223, 162, 0,   3,   161, 223, 162, 0,   6,   0,   10,  0,   97,  161, 162,
    0,   54,  161, 223, 162, 0,   11,  0,   72,  0,   63,  0,   253, 6,   0,   165, 249,
    166, 0,   29,  163, 249, 164, 0,   98,  0,   247, 0,   248, 160, 247, 0,   0,   248,
    0,   3,   0,   9,   0,   0,   250, 0,   3,   0,   5,   0,   4,   0,   9,   0,   0,
    0,   0,   160, 0,   0,   3,   0,   4,   0,   0,   0,   0,   160, 0,   0,   0,   0,
    0,   0,   0,   0,   0,   160, 0,   0,   24,  0,   24,  113, 0,   48,  0,   138, 0,
    85,  0,   126, 0,   144, 0,   59,  0,   25,  0,   64,  0,   148, 0,   154, 0,   155,
    0,   128, 0,   115, 128, 0,   25,  115, 0,   48,  115, 0,   48,  118, 0,   48,  148,
    0,   138, 148, 0,   64,  148, 0,   64,  115, 0,   48,  156, 0,   154, 156, 0,   148,
    156, 0,   59,  156, 0,   148, 157, 158, 0,   51,  0,   118, 0,   156, 0,   148, 0,
    115, 0,   3,   0,   10,  0,   3,   0,   3,   167, 3,   0,   3,   167, 19,  0,   10,
    0,   34,  0,   120, 0,   35,  0,   40,  0,   40,  161, 252, 162, 0,   99,  0,   99,
    161, 252, 162, 0,   99,  161, 252, 160, 252, 162, 0,   56,  0,   56,  161, 252, 162,
    0,   56,  161, 252, 160, 252, 162, 0,   86,  0,   124, 0,   116, 0,   72,  0,   130,
    0,   63,  111, 0,   63,  0,   53,  0,   122, 0,   122, 161, 252, 162, 0,   123, 0,
    123, 161, 252, 162, 0,   254, 0,   255, 0,   253, 163, 164, 0,   253, 163, 252, 164,
    0,   109, 0,   94,  0,   110, 0,   96,  0,   77,  161, 254, 162, 0,   77,  161, 254,
    160, 10,  162, 0,   78,  161, 254, 162, 0,   78,  161, 254, 160, 10,  162, 0,   3,
    0,   9,   0,   3,   0};

#endif

#if YY_Parser_DEBUG != 0
static const short yyrline[] = {
    0,   120, 122,  130,  132,  133,  134, 135, 136, 137, 138, 141, 146, 152, 155, 157,
    160, 162, 165,  170,  176,  178,  181, 187, 194, 201, 202, 204, 206, 213, 218, 224,
    228, 230, 233,  236,  243,  245,  252, 254, 257, 260, 264, 272, 279, 282, 284, 285,
    291, 292, 293,  294,  295,  296,  299, 302, 308, 315, 321, 329, 336, 340, 342, 349,
    352, 356, 363,  365,  368,  370,  373, 375, 382, 385, 389, 391, 392, 395, 397, 398,
    403, 406, 412,  415,  417,  420,  425, 432, 434, 435, 438, 441, 448, 453, 458, 460,
    463, 465, 466,  468,  470,  477,  483, 490, 492, 494, 498, 500, 503, 515, 517, 520,
    524, 526, 533,  535,  538,  542,  544, 547, 549, 556, 558, 563, 566, 568, 570, 571,
    574, 576, 577,  578,  579,  580,  581, 582, 585, 588, 595, 598, 602, 605, 607, 609,
    613, 615, 627,  629,  632,  635,  647, 649, 653, 655, 662, 667, 673, 675, 678, 680,
    681, 684, 688,  692,  697,  703,  705, 708, 712, 717, 724, 726, 731, 739, 741, 742,
    743, 744, 745,  746,  747,  748,  749, 750, 751, 752, 754, 755, 756, 759, 761, 762,
    763, 766, 768,  769,  776,  778,  791, 793, 794, 795, 798, 800, 801, 802, 803, 804,
    805, 806, 807,  808,  809,  812,  814, 821, 823, 828, 838, 841, 843, 845, 846, 846,
    846, 849, 851,  858,  859,  862,  864, 871, 872, 875, 877, 884, 886, 887, 888, 889,
    890, 891, 892,  893,  894,  895,  896, 897, 898, 899, 900, 901, 902, 903, 904, 905,
    906, 907, 908,  909,  910,  911,  914, 916, 917, 918, 919, 922, 924, 928, 930, 931,
    935, 945, 947,  948,  949,  950,  951, 952, 953, 954, 955, 956, 957, 958, 959, 960,
    962, 963, 964,  965,  966,  967,  968, 969, 970, 972, 973, 980, 990, 991, 992, 993,
    996, 998, 1002, 1004, 1010, 1019, 1022};

static const char* const yytname[] = {"$",
                                      "error",
                                      "$illegal.",
                                      "NAME",
                                      "DASHEDNAME",
                                      "EMAIL",
                                      "STRING",
                                      "FWDSTR",
                                      "SELECTSTRING",
                                      "QUOTED_IDENTIFIER",
                                      "INTNUM",
                                      "FIXEDNUM",
                                      "OR",
                                      "AND",
                                      "NOT",
                                      "EQUAL",
                                      "COMPARISON",
                                      "'+'",
                                      "'-'",
                                      "'*'",
                                      "'/'",
                                      "'%'",
                                      "UMINUS",
                                      "ADD",
                                      "ALL",
                                      "ALTER",
                                      "AMMSC",
                                      "ANY",
                                      "ARCHIVE",
                                      "ARRAY",
                                      "AS",
                                      "ASC",
                                      "AUTHORIZATION",
                                      "BETWEEN",
                                      "BIGINT",
                                      "BOOLEAN",
                                      "BY",
                                      "CASE",
                                      "CAST",
                                      "CHAR_LENGTH",
                                      "CHARACTER",
                                      "CHECK",
                                      "CLOSE",
                                      "CLUSTER",
                                      "COLUMN",
                                      "COMMIT",
                                      "CONTINUE",
                                      "COPY",
                                      "CREATE",
                                      "CURRENT",
                                      "CURSOR",
                                      "DATABASE",
                                      "DATAFRAME",
                                      "DATE",
                                      "DATETIME",
                                      "DATE_TRUNC",
                                      "DECIMAL",
                                      "DECLARE",
                                      "DEFAULT",
                                      "DELETE",
                                      "DESC",
                                      "DICTIONARY",
                                      "DISTINCT",
                                      "DOUBLE",
                                      "DROP",
                                      "DUMP",
                                      "ELSE",
                                      "END",
                                      "EXISTS",
                                      "EXTRACT",
                                      "FETCH",
                                      "FIRST",
                                      "FLOAT",
                                      "FOR",
                                      "FOREIGN",
                                      "FOUND",
                                      "FROM",
                                      "GEOGRAPHY",
                                      "GEOMETRY",
                                      "GRANT",
                                      "GROUP",
                                      "HAVING",
                                      "IF",
                                      "ILIKE",
                                      "IN",
                                      "INSERT",
                                      "INTEGER",
                                      "INTO",
                                      "IS",
                                      "LANGUAGE",
                                      "LAST",
                                      "LENGTH",
                                      "LIKE",
                                      "LIMIT",
                                      "LINESTRING",
                                      "MOD",
                                      "MULTIPOLYGON",
                                      "NOW",
                                      "NULLX",
                                      "NUMERIC",
                                      "OF",
                                      "OFFSET",
                                      "ON",
                                      "OPEN",
                                      "OPTIMIZE",
                                      "OPTIMIZED",
                                      "OPTION",
                                      "ORDER",
                                      "PARAMETER",
                                      "POINT",
                                      "POLYGON",
                                      "PRECISION",
                                      "PRIMARY",
                                      "PRIVILEGES",
                                      "PROCEDURE",
                                      "SERVER",
                                      "SMALLINT",
                                      "SOME",
                                      "TABLE",
                                      "TEMPORARY",
                                      "TEXT",
                                      "THEN",
                                      "TIME",
                                      "TIMESTAMP",
                                      "TINYINT",
                                      "TO",
                                      "TRUNCATE",
                                      "UNION",
                                      "USAGE",
                                      "PUBLIC",
                                      "REAL",
                                      "REFERENCES",
                                      "RENAME",
                                      "RESTORE",
                                      "REVOKE",
                                      "ROLE",
                                      "ROLLBACK",
                                      "SCHEMA",
                                      "SELECT",
                                      "SET",
                                      "SHARD",
                                      "SHARED",
                                      "SHOW",
                                      "UNIQUE",
                                      "UPDATE",
                                      "USER",
                                      "VALIDATE",
                                      "VALUES",
                                      "VIEW",
                                      "WHEN",
                                      "WHENEVER",
                                      "WHERE",
                                      "WITH",
                                      "WORK",
                                      "EDIT",
                                      "ACCESS",
                                      "DASHBOARD",
                                      "SQL",
                                      "EDITOR",
                                      "';'",
                                      "','",
                                      "'('",
                                      "')'",
                                      "'['",
                                      "']'",
                                      "'{'",
                                      "'}'",
                                      "'.'",
                                      "sql_list",
                                      "sql",
                                      "name_eq_value_list",
                                      "name_eq_value",
                                      "opt_if_not_exists",
                                      "opt_temporary",
                                      "create_table_statement",
                                      "opt_if_exists",
                                      "drop_table_statement",
                                      "rename_table_statement",
                                      "rename_column_statement",
                                      "opt_column",
                                      "column_defs",
                                      "add_column_statement",
                                      "drop_column_statement",
                                      "drop_columns",
                                      "drop_column",
                                      "alter_table_param_statement",
                                      "base_table_element_commalist",
                                      "base_table_element",
                                      "column_def",
                                      "opt_compression",
                                      "column_constraint_def",
                                      "table_constraint_def",
                                      "column_commalist",
                                      "opt_with_option_list",
                                      "drop_view_statement",
                                      "opt_column_commalist",
                                      "opt_order_by_clause",
                                      "ordering_spec_commalist",
                                      "ordering_spec",
                                      "opt_asc_desc",
                                      "opt_null_order",
                                      "manipulative_statement",
                                      "delete_statement",
                                      "insert_statement",
                                      "opt_all_distinct",
                                      "assignment_commalist",
                                      "assignment",
                                      "update_statement",
                                      "opt_where_clause",
                                      "opt_limit_clause",
                                      "opt_offset_clause",
                                      "select_statement",
                                      "query_exp",
                                      "query_term",
                                      "query_spec",
                                      "selection",
                                      "from_clause",
                                      "table_ref_commalist",
                                      "table_ref",
                                      "where_clause",
                                      "opt_group_by_clause",
                                      "exp_commalist",
                                      "opt_having_clause",
                                      "general_exp",
                                      "predicate",
                                      "comparison_predicate",
                                      "between_predicate",
                                      "like_predicate",
                                      "opt_escape",
                                      "test_for_null",
                                      "in_predicate",
                                      "atom_commalist",
                                      "all_or_any_predicate",
                                      "comparison",
                                      "any_all_some",
                                      "existence_test",
                                      "subquery",
                                      "when_then_list",
                                      "opt_else_expr",
                                      "case_exp",
                                      "charlength_exp",
                                      "array_at_exp",
                                      "scalar_exp",
                                      "select_entry",
                                      "select_entry_commalist",
                                      "atom",
                                      "function_ref",
                                      "literal",
                                      "literal_commalist",
                                      "opt_literal_commalist",
                                      "table",
                                      "column_ref",
                                      "non_neg_int",
                                      "data_type",
                                      "geo_type",
                                      "geometry_type",
                                      "column",
                                      "range_variable",
                                      "\37777777740$\37777777626"};
#endif

static const short yyr1[] = {
    0,   168, 168, 169, 169, 169, 169, 169, 169, 169, 169, 170, 170, 171, 172, 172, 173,
    173, 174, 174, 175, 175, 176, 177, 178, 179, 179, 180, 180, 181, 181, 182, 183, 183,
    184, 185, 186, 186, 187, 187, 188, 188, 189, 189, 189, 190, 190, 190, 190, 190, 190,
    190, 190, 190, 191, 191, 191, 191, 191, 191, 191, 192, 192, 193, 193, 194, 195, 195,
    196, 196, 197, 197, 198, 198, 199, 199, 199, 200, 200, 200, 169, 201, 201, 201, 201,
    202, 203, 204, 204, 204, 205, 205, 206, 207, 208, 208, 209, 209, 209, 210, 210, 210,
    211, 212, 212, 212, 213, 213, 214, 215, 215, 216, 217, 217, 218, 218, 219, 220, 220,
    221, 221, 222, 222, 223, 223, 223, 223, 223, 224, 224, 224, 224, 224, 224, 224, 224,
    225, 225, 226, 226, 227, 227, 227, 227, 228, 228, 229, 229, 230, 230, 230, 230, 231,
    231, 232, 232, 233, 233, 234, 234, 234, 235, 236, 237, 237, 238, 238, 239, 239, 239,
    240, 240, 241, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242,
    242, 242, 243, 243, 243, 243, 244, 244, 244, 245, 245, 246, 246, 246, 246, 247, 247,
    247, 247, 247, 247, 247, 247, 247, 247, 247, 248, 248, 249, 249, 250, 250, -1,  -1,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  251, 251, 251,
    252, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
    253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254, 254, -1,  -1,
    255, 255, 256, 256, 257};

static const short yyr2[] = {
    0, 2, 3, 1, 1, 1,  1, 1,  1, 1, 1, 1, 3, 3, 3, 0, 1, 0, 9, 10, 2, 0, 4, 6, 8, 0,
    1, 1, 3, 6, 7, 4,  1, 3,  3, 5, 1, 3, 1, 1, 3, 4, 2, 5, 0, 2,  3, 4, 2, 2, 2, 4,
    2, 5, 4, 5, 7, 10, 5, 10, 4, 1, 3, 4, 0, 4, 0, 3, 0, 3, 1, 3,  3, 3, 0, 1, 1, 0,
    2, 2, 1, 1, 1, 1,  1, 4,  8, 0, 1, 1, 1, 3, 3, 5, 0, 1, 2, 2,  0, 2, 3, 0, 4, 1,
    3, 4, 1, 3, 7, 1,  1, 2,  1, 3, 1, 2, 2, 0, 3, 1, 3, 0, 2, 3,  3, 2, 3, 1, 1, 1,
    1, 1, 1, 1, 1, 1,  3, 3,  6, 5, 5, 4, 5, 4, 0, 2, 4, 3, 4, 3,  6, 5, 1, 3, 4, 4,
    1, 1, 1, 1, 1, 2,  3, 4,  5, 2, 0, 4, 8, 6, 4, 4, 4, 3, 3, 3,  3, 3, 6, 2, 2, 1,
    1, 1, 3, 6, 1, 1,  1, 0,  1, 2, 3, 0, 1, 3, 1, 1, 4, 5, 5, 4,  1, 1, 3, 4, 1, 1,
    1, 2, 3, 4, 1, 1,  3, 0,  1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3, 1,  1, 1, 3, 1, 1, 1,
    3, 1, 2, 1, 1, 1,  1, 1,  1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,  2, 2, 2, 2, 2, 2,
    2, 3, 1, 1, 1, 1,  1, 1,  1, 1, 3, 3, 1, 1, 1, 1, 1, 4, 1, 4,  6, 1, 4, 6, 1, 1,
    1, 1, 1, 2, 1, 1,  1, 4,  1, 4, 1, 1, 3, 4, 1, 1, 1, 1, 4, 6,  4, 6, 1, 1, 1};

static const short yydefact[] = {
    0,   0,   17,  0,   0,   0,   87,  0,   0,   0,   0,   3,   5,   6,   7,   8,   9,
    10,  4,   80,  81,  82,  84,  83,  68,  103, 106, 0,   0,   16,  0,   0,   21,  21,
    0,   88,  89,  189, 217, 218, 0,   0,   0,   1,   0,   0,   98,  0,   0,   15,  94,
    0,   0,   0,   66,  269, 202, 203, 206, 0,   0,   0,   110, 0,   273, 275, 0,   0,
    0,   276, 291, 0,   281, 208, 0,   207, 0,   0,   284, 0,   301, 0,   303, 0,   212,
    278, 300, 302, 286, 274, 292, 294, 285, 288, 197, 0,   215, 0,   190, 127, 128, 129,
    130, 131, 132, 133, 134, 186, 187, 188, 135, 194, 109, 181, 183, 196, 182, 0,   296,
    297, 0,   107, 2,   0,   0,   104, 0,   101, 25,  25,  0,   0,   31,  32,  15,  0,
    0,   0,   85,  95,  20,  22,  65,  0,   0,   0,   0,   125, 0,   179, 182, 180, 215,
    0,   166, 0,   0,   0,   0,   0,   289, 0,   161, 0,   0,   0,   0,   0,   0,   0,
    0,   0,   135, 213, 216, 0,   0,   94,  191, 0,   0,   0,   0,   156, 157, 0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   189, 0,   0,   209, 0,   308, 309, 94,  90,
    0,   269, 74,  69,  70,  74,  105, 96,  97,  0,   102, 26,  0,   0,   0,   0,   0,
    0,   35,  0,   0,   0,   0,   116, 0,   61,  0,   0,   0,   0,   0,   270, 271, 0,
    0,   0,   0,   0,   0,   0,   0,   272, 0,   0,   0,   0,   0,   0,   0,   0,   204,
    0,   0,   0,   126, 184, 0,   210, 111, 112, 114, 117, 123, 124, 192, 0,   0,   0,
    0,   173, 174, 175, 176, 177, 0,   144, 0,   149, 144, 159, 158, 160, 0,   0,   137,
    136, 195, 0,   147, 0,   298, 0,   0,   93,  0,   75,  76,  77,  0,   77,  99,  0,
    27,  0,   29,  34,  0,   23,  0,   33,  0,   14,  0,   0,   67,  0,   198, 0,   0,
    201, 211, 0,   165, 0,   167, 0,   170, 277, 205, 0,   282, 162, 0,   306, 0,   171,
    0,   0,   279, 293, 295, 214, 0,   310, 115, 0,   121, 0,   144, 0,   148, 144, 0,
    0,   143, 0,   152, 141, 154, 155, 146, 172, 299, 91,  92,  0,   72,  71,  73,  100,
    0,   30,  290, 287, 44,  0,   13,  0,   0,   0,   0,   0,   0,   0,   0,   36,  38,
    39,  62,  0,   200, 199, 163, 0,   0,   0,   0,   0,   0,   0,   113, 0,   0,   108,
    0,   142, 0,   140, 139, 145, 0,   151, 78,  79,  28,  0,   0,   0,   0,   0,   40,
    44,  24,  0,   0,   0,   0,   0,   0,   0,   0,   64,  86,  164, 185, 283, 307, 0,
    169, 178, 280, 118, 119, 122, 138, 150, 153, 42,  45,  0,   49,  50,  48,  52,  41,
    64,  0,   0,   0,   0,   0,   0,   37,  0,   18,  0,   0,   0,   0,   46,  0,   0,
    19,  60,  0,   0,   0,   0,   54,  0,   168, 120, 0,   47,  51,  0,   0,   55,  58,
    0,   0,   11,  43,  53,  0,   0,   0,   63,  56,  0,   12,  0,   0,   0,   0,   57,
    59,  0,   0};

static const short yydefgoto[] = {
    9,   10,  494, 222, 136, 30,  11,  52,  12,  13,  14,  217, 305, 15,  16,
    132, 133, 17,  388, 389, 390, 424, 425, 391, 228, 468, 18,  144, 46,  207,
    208, 301, 370, 19,  20,  21,  37,  202, 203, 22,  138, 127, 214, 23,  24,
    25,  26,  97,  177, 262, 263, 139, 350, 445, 407, 98,  99,  100, 101, 102,
    358, 103, 104, 359, 105, 194, 287, 106, 162, 154, 242, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 174, 175, 264, 116, 246, 117, 118, 119, 307, 348};

static const short yypact[] = {
    301,    -92,    30,     52,     -37,    32,     11,     202,    43,     59,
    -95,    -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
    -32768, -32768, -32768, -32768, 115,    -32768, -32768, 202,    121,    -32768,
    34,     202,    119,    119,    202,    -32768, -32768, 518,    -32768, -32768,
    85,     -88,    135,    -32768, 213,    23,     215,    82,     219,    251,
    169,    283,    202,    202,    198,    87,     -32768, -32768, -32768, 748,
    942,    942,    -32768, 194,    -32768, -32768, 223,    203,    214,    227,
    -32768, 237,    247,    2,      249,    -1,     250,    254,    -32768, 258,
    -32768, 262,    -32768, 265,    -32768, 267,    -32768, -32768, -32768, -32768,
    268,    270,    -32768, -32768, -32768, 748,    1351,   286,    73,     -32768,
    -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
    1437,   -32768, 208,    -32768, -32768, -32768, -67,    25,     -32768, -32768,
    269,    -32768, -32768, 289,    43,     -32768, 199,    275,    -28,    340,
    -32,    390,    244,    -32768, 251,    386,    202,    748,    -32768, -32768,
    -32768, -32768, -32768, 269,    278,    403,    231,    -32768, 942,    -32768,
    271,    -32768, 1351,   748,    -12,    748,    942,    420,    748,    420,
    -32768, 295,    -32768, 197,    748,    942,    942,    273,    420,    420,
    420,    28,     212,    -32768, 276,    280,    202,    169,    -32768, 748,
    748,    441,    206,    -32768, -32768, 942,    942,    942,    942,    942,
    942,    1195,   287,    1195,   633,    748,    -8,     942,    -32768, 0,
    -32768, -32768, 137,    -32768, 432,    282,    60,     290,    -32768, 60,
    -32768, -32768, -32768, 442,    -32768, -32768, 269,    269,    269,    269,
    202,    436,    -32768, 389,    202,    387,    293,    333,    75,     -32768,
    299,    296,    748,    748,    38,     -32768, -32768, 174,    300,    6,
    748,    748,    394,    58,     240,    -32768, 305,    40,     143,    306,
    151,    20,     245,    264,    -32768, 159,    307,    308,    -32768, -32768,
    1351,   -32768, 303,    -32768, 469,    393,    461,    -32768, -32768, 942,
    1195,   315,    1195,   281,    281,    -32768, -32768, -32768, 297,    474,
    1117,   -32768, 474,    -32768, -32768, -32768, 845,    1039,   -32768, 323,
    -32768, 380,    -32768, 122,    -32768, 316,    269,    -32768, 748,    -32768,
    -32768, 381,    289,    381,    479,    172,    -32768, 1312,   -32768, -32768,
    358,    -32768, 1351,   -32768, 325,    -32768, 238,    269,    -32768, 1195,
    -32768, 44,     48,     -32768, -32768, 748,    333,    86,     -32768, 1312,
    -32768, -32768, -32768, 420,    -32768, -32768, 477,    -32768, 748,    -32768,
    942,    420,    -32768, -32768, -32768, -32768, 202,    -32768, -32768, 448,
    407,    310,    474,    1117,   -32768, 474,    942,    1195,   -32768, 196,
    -32768, -32768, -32768, 323,    -32768, -32768, -32768, -32768, 333,    4,
    -32768, -32768, -32768, -32768, 269,    -32768, 379,    -32768, 14,     269,
    -32768, 238,    332,    492,    493,    500,    443,    344,    201,    -32768,
    -32768, -32768, -32768, 207,    -32768, -32768, 333,    748,    90,     345,
    346,    36,     256,    347,    -32768, 748,    748,    -32768, 942,    -32768,
    211,    -32768, 323,    -32768, 1195,   -32768, -32768, -32768, -32768, 503,
    412,    350,    1273,   202,    -32768, 511,    -32768, 220,    748,    354,
    355,    356,    357,    269,    238,    368,    -32768, 333,    -32768, -32768,
    -32768, 748,    -32768, -32768, -32768, 362,    333,    333,    323,    -32768,
    -32768, 369,    -54,    748,    -32768, -32768, -32768, 370,    -32768, 368,
    50,     269,    269,    269,    269,    225,    -32768, 373,    -32768, 54,
    748,    528,    536,    -32768, 56,     269,    -32768, -32768, 229,    232,
    378,    382,    -32768, 390,    -32768, 333,    383,    -32768, -32768, 384,
    410,    -32768, -32768, 411,    239,    -32768, -32768, -32768, 202,    202,
    390,    -32768, 388,    398,    -32768, 269,    269,    243,    392,    -32768,
    -32768, 543,    -32768};

static const short yypgoto[] = {
    -32768, 541,    -32768, -264,   417,    -32768, -32768, 527,    -32768, -32768,
    -32768, 433,    -32768, -32768, -32768, -32768, 338,    -32768, 182,    131,
    -214,   141,    -32768, -32768, -218,   108,    -32768, -32768, -32768, -32768,
    274,    360,    272,    -32768, -32768, -32768, -32768, -32768, 277,    -32768,
    -94,    -32768, -32768, -32768, 562,    -23,    -154,   -32768, -32768, -32768,
    233,    -32768, -32768, -32768, -32768, -58,    -32768, -32768, -32768, -32768,
    -202,   -32768, -32768, -237,   -32768, -32768, -32768, -32768, -156,   -32768,
    -32768, -32768, -32768, -32768, -18,    385,    -32768, -168,   -32768, -87,
    -32768, 425,    -7,     -31,    -155,   -193,   415,    -32768, -109,   -32768};

#define YYLAST 1529

static const short yytable[] = {
    40,  147, 306,  308, 248, -287, 291, 249, -290, 173, 245, 204, 219, 255, 256, 257,
    215, 419, 179,  180, 47,  196,  125, 279, 50,   282, 27,  54,  420, 150, 150, 198,
    179, 180, 229,  35,  281, 171,  288, 45,  179,  180, 149, 151, 295, 141, 142, 124,
    179, 180, 179,  180, 179, 180,  240, 421, 179,  180, 472, 511, 179, 180, 179, 180,
    43,  173, 179,  180, 179, 180,  179, 180, 422,  36,  121, 416, 178, 172, 28,  227,
    361, 32,  393,  265, 1,   179,  180, 234, 329,  473, 292, 299, 209, 220, 417, 239,
    197, 243, 179,  180, 247, 210,  352, 181, 355,  128, 251, 2,   297, 309, 310, 33,
    360, 160, 378,  354, 410, 150,  3,   34,  300,  266, 267, 4,   48,  150, 249, 325,
    31,  226, 237,  362, 249, 216,  150, 150, 398,  241, 244, 185, 186, 187, 188, 189,
    5,   423, 129,  252, 253, 29,   409, 360, 49,   411, 150, 150, 150, 150, 150, 150,
    418, 6,   -287, 150, 294, -290, 150, 273, 274,  275, 276, 277, 278, 345, 321, 322,
    289, 199, 399,  293, 338, 6,    326, 327, 8,    360, 403, 204, 199, 413, 258, 185,
    186, 187, 188,  189, 441, 6,    442, 249, 323,  51,  332, 7,   8,   38,  394, 397,
    392, 211, 395,  39,  477, 311,  130, 465, 484,  314, 488, 495, 8,   131, 44,  212,
    120, 380, 182,  183, 184, 185,  186, 187, 188,  189, 235, 317, 504, 318, 150, 269,
    368, 200, 45,   478, 479, 190,  450, 201, 145,  123, 236, 351, 438, 199, 146, 150,
    150, 185, 186,  187, 188, 189,  185, 186, 187,  188, 189, 396, 237, 363, 426, 209,
    200, 185, 186,  187, 188, 189,  201, 382, 401,  185, 186, 187, 188, 189, 365, 507,
    137, 270, 271,  80,  205, 82,   122, 191, 192,  296, 272, 206, 187, 188, 189, 333,
    193, 334, 86,   87,  126, 150,  356, 336, 383,  337, 185, 186, 187, 188, 189, 341,
    137, 342, 402,  408, 229, 150,  1,   185, 186,  187, 188, 189, 374, 135, 375, 456,
    259, 134, 412,  437, 185, 186,  187, 188, 189,  179, 180, 446, 447, 2,   384, 140,
    229, 229, 480,  481, 414, 152,  415, 143, 3,    434, 176, 435, 155, 4,   489, 414,
    195, 436, 460,  414, 153, 449,  259, 156, 213,  150, 385, 386, 434, 387, 459, 469,
    215, 317, 5,    482, 157, 317,  448, 490, 317,  221, 491, 474, 229, 508, 158, 500,
    225, 501, 330,  317, 223, 509,  55,  339, 159,  56,  161, 163, 485, 57,  58,  164,
    457, 59,  443,  165, 60,  61,   231, 166, 340,  230, 167, 232, 168, 169, 245, 170,
    63,  6,   197,  254, 260, 64,   65,  6,   66,   67,  68,  69,  268, 7,   261, 298,
    280, 146, 302,  312, 304, 129,  316, 315, 70,   71,  320, 72,  319, 328, 8,   346,
    324, 233, 73,   331, 335, 343,  344, 74,  347,  349, 180, 75,  353, 357, 364, 369,
    366, 76,  373,  379, 405, 77,   381, 400, 406,  78,  160, 502, 503, 428, 79,  429,
    430, 80,  81,   82,  83,  84,   85,  431, 432,  433, 451, 439, 440, 444, 452, 453,
    86,  87,  419,  461, 462, 463,  464, 88,  467,  55,  470, 89,  56,  90,  91,  92,
    57,  58,  471,  475, 59,  93,   483, 60,  61,   62,  486, 487, 492, 498, 499, 512,
    493, 496, 497,  63,  94,  505,  42,  224, 64,   65,  510, 66,  67,  68,  69,  506,
    53,  313, 218,  427, 95,  466,  458, 476, 96,   303, 41,  70,  71,  367, 72,  372,
    371, 238, 250,  404, 290, 73,   0,   0,   0,    0,   74,  0,   0,   0,   75,  0,
    0,   0,   0,    0,   76,  0,    0,   0,   77,   0,   0,   0,   78,  0,   0,   0,
    0,   79,  0,    0,   80,  81,   82,  83,  84,   85,  0,   0,   0,   0,   0,   0,
    0,   0,   0,    86,  87,  0,    0,   0,   0,    0,   88,  0,   55,  0,   89,  56,
    90,  91,  92,   57,  58,  0,    0,   0,   93,   0,   60,  61,  0,   0,   0,   0,
    0,   283, 0,    0,   284, 0,    63,  94,  0,    0,   0,   64,  65,  0,   66,  67,
    68,  69,  0,    0,   0,   0,    0,   95,  0,    0,   0,   96,  0,   0,   70,  71,
    0,   72,  0,    0,   0,   0,    0,   0,   73,   0,   0,   0,   0,   0,   0,   0,
    0,   75,  0,    0,   0,   0,    0,   76,  0,    0,   0,   77,  0,   0,   0,   78,
    0,   0,   0,    0,   79,  0,    0,   80,  81,   82,  83,  84,  85,  0,   0,   0,
    0,   0,   0,    0,   0,   0,    86,  87,  0,    0,   0,   0,   0,   88,  285, 55,
    0,   89,  56,   90,  91,  92,   57,  58,  0,    0,   59,  93,  0,   60,  61,  0,
    0,   0,   0,    0,   0,   0,    0,   0,   0,    63,  94,  0,   0,   0,   64,  65,
    0,   66,  67,   68,  69,  0,    0,   0,   0,    0,   286, 0,   0,   0,   96,  0,
    0,   70,  71,   0,   72,  0,    0,   0,   0,    0,   0,   73,  0,   0,   0,   0,
    74,  0,   0,    0,   75,  0,    0,   0,   0,    0,   76,  0,   0,   0,   77,  0,
    0,   0,   78,   0,   0,   0,    0,   79,  0,    0,   80,  81,  82,  83,  84,  85,
    55,  0,   0,    56,  0,   0,    0,   57,  58,   86,  87,  0,   0,   0,   60,  61,
    88,  0,   0,    0,   89,  0,    90,  91,  92,   0,   63,  0,   0,   0,   93,  64,
    65,  0,   66,   67,  68,  69,   0,   0,   0,    0,   0,   0,   0,   94,  0,   0,
    0,   0,   70,   71,  0,   72,   0,   0,   0,    0,   0,   0,   73,  95,  0,   0,
    0,   96,  0,    0,   0,   75,   0,   0,   0,    0,   0,   76,  0,   0,   0,   77,
    0,   0,   0,    78,  0,   0,    0,   0,   79,   0,   0,   80,  81,  82,  83,  84,
    85,  55,  0,    0,   56,  0,    0,   0,   57,   58,  86,  87,  0,   0,   0,   60,
    61,  88,  0,    0,   0,   89,   0,   90,  91,   92,  0,   63,  0,   0,   0,   93,
    64,  65,  0,    66,  67,  68,   69,  6,   0,    0,   0,   0,   0,   0,   94,  0,
    0,   0,   0,    70,  71,  0,    72,  0,   0,    0,   0,   0,   0,   73,  148, 0,
    0,   0,   96,   0,   0,   0,    75,  0,   0,    0,   0,   0,   76,  0,   0,   0,
    77,  0,   0,    0,   78,  0,    0,   0,   0,    79,  0,   0,   80,  81,  82,  83,
    84,  85,  55,   0,   0,   56,   0,   0,   0,    57,  58,  86,  87,  0,   0,   0,
    60,  61,  88,   0,   0,   0,    89,  0,   90,   91,  92,  0,   63,  0,   0,   0,
    93,  64,  65,   0,   66,  67,   68,  69,  0,    0,   0,   0,   0,   0,   0,   94,
    0,   0,   0,    0,   70,  71,   0,   72,  0,    0,   0,   0,   0,   0,   73,  148,
    0,   0,   0,    96,  0,   0,    0,   75,  0,    0,   0,   0,   0,   76,  0,   0,
    0,   77,  0,    56,  0,   78,   0,   57,  58,   0,   79,  0,   0,   80,  81,  82,
    83,  84,  85,   0,   0,   0,    0,   0,   0,    0,   63,  0,   86,  87,  0,   64,
    65,  0,   0,    88,  0,   69,   0,   89,  0,    90,  91,  92,  0,   0,   0,   0,
    0,   93,  70,   71,  0,   72,   0,   0,   0,    0,   0,   0,   73,  0,   0,   0,
    94,  0,   0,    0,   0,   75,   0,   0,   0,    0,   0,   76,  0,   0,   0,   0,
    286, 56,  0,    78,  96,  57,   58,  0,   0,    0,   0,   80,  0,   82,  83,  84,
    85,  0,   0,    0,   0,   0,    0,   0,   63,   0,   86,  87,  0,   64,  65,  0,
    0,   88,  0,    69,  0,   89,   0,   90,  91,   92,  0,   0,   0,   0,   0,   93,
    70,  71,  0,    72,  0,   0,    0,   6,   0,    0,   73,  0,   0,   0,   94,  0,
    0,   0,   0,    75,  0,   0,    0,   0,   0,    76,  0,   0,   0,   0,   0,   56,
    0,   78,  96,   57,  58,  0,    0,   0,   0,    80,  0,   82,  83,  84,  85,  0,
    0,   0,   0,    0,   0,   0,    63,  0,   86,   87,  0,   64,  65,  0,   0,   88,
    0,   69,  0,    89,  0,   90,   91,  92,  0,    0,   0,   0,   0,   93,  70,  71,
    0,   72,  0,    0,   0,   0,    0,   0,   73,   0,   0,   0,   94,  0,   0,   0,
    0,   75,  64,   65,  0,   0,    0,   76,  69,   0,   0,   0,   0,   56,  0,   78,
    96,  57,  58,   0,   0,   70,   0,   80,  72,   82,  83,  454, 85,  0,   0,   376,
    0,   0,   0,    0,   63,  0,    86,  87,  377,  64,  65,  0,   0,   88,  76,  69,
    0,   89,  0,    90,  91,  92,   78,  0,   0,    0,   0,   93,  70,  71,  80,  72,
    82,  0,   0,    85,  0,   0,    73,  0,   0,    0,   455, 0,   0,   86,  87,  75,
    0,   0,   0,    0,   88,  76,   0,   0,   89,   0,   90,  91,  92,  78,  96,  0,
    0,   0,   93,   0,   0,   80,   0,   82,  83,   84,  85,  182, 183, 184, 185, 186,
    187, 188, 189,  0,   86,  87,   0,   0,   0,    0,   0,   88,  0,   0,   190, 89,
    0,   90,  91,   92,  0,   0,    0,   0,   0,    93,  0,   0,   0,   0,   0,   0,
    0,   0,   0,    0,   0,   0,    0,   0,   0,    0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,    0,   0,   0,    0,   0,   0,    0,   0,   0,   96,  0,   0,   0,
    191, 192, 0,    0,   0,   0,    0,   0,   0,    193};

static const short yycheck[] = {
    7,   59,  216, 217, 159, 6,   14,  161, 6,   96,  10,  120, 44,  168, 169, 170, 44,
    3,   12,  13,  27,  88,  45,  191, 31,  193, 118, 34,  14,  60,  61,  6,   12,  13,
    143, 24,  192, 95,  194, 127, 12,  13,  60,  61,  199, 52,  53,  24,  12,  13,  12,
    13,  12,  13,  66,  41,  12,  13,  112, 0,   12,  13,  12,  13,  159, 152, 12,  13,
    12,  13,  12,  13,  58,  62,  162, 71,  3,   95,  48,  137, 282, 118, 319, 177, 25,
    12,  13,  145, 30,  143, 98,  31,  123, 125, 90,  153, 163, 155, 12,  13,  158, 124,
    270, 30,  272, 23,  164, 48,  202, 218, 219, 148, 280, 111, 307, 271, 353, 148, 59,
    87,  60,  179, 180, 64,  3,   156, 280, 121, 76,  136, 148, 287, 286, 161, 165, 166,
    329, 149, 156, 17,  18,  19,  20,  21,  85,  131, 64,  165, 166, 119, 352, 319, 118,
    355, 185, 186, 187, 188, 189, 190, 374, 138, 163, 194, 164, 163, 197, 185, 186, 187,
    188, 189, 190, 260, 232, 233, 194, 163, 333, 197, 160, 138, 240, 241, 161, 353, 341,
    296, 163, 357, 162, 17,  18,  19,  20,  21,  160, 138, 162, 353, 162, 82,  162, 144,
    161, 3,   162, 121, 317, 10,  162, 9,   162, 220, 132, 433, 162, 224, 162, 483, 161,
    139, 107, 24,  139, 312, 14,  15,  16,  17,  18,  19,  20,  21,  3,   160, 500, 162,
    269, 33,  298, 3,   127, 461, 462, 33,  414, 9,   161, 36,  19,  269, 162, 163, 167,
    286, 287, 17,  18,  19,  20,  21,  17,  18,  19,  20,  21,  325, 286, 287, 379, 302,
    3,   17,  18,  19,  20,  21,  9,   41,  338, 17,  18,  19,  20,  21,  164, 505, 151,
    83,  84,  94,  3,   96,  159, 83,  84,  160, 92,  10,  19,  20,  21,  160, 92,  162,
    109, 110, 93,  340, 13,  160, 74,  162, 17,  18,  19,  20,  21,  160, 151, 162, 340,
    13,  433, 356, 25,  17,  18,  19,  20,  21,  160, 82,  162, 422, 162, 118, 356, 397,
    17,  18,  19,  20,  21,  12,  13,  405, 406, 48,  112, 68,  461, 462, 463, 464, 160,
    163, 162, 161, 59,  160, 76,  162, 161, 64,  475, 160, 160, 162, 428, 160, 149, 162,
    162, 161, 101, 408, 140, 141, 160, 143, 162, 441, 44,  160, 85,  162, 161, 160, 408,
    162, 160, 3,   162, 453, 505, 506, 161, 160, 14,  162, 162, 160, 160, 162, 3,   162,
    161, 6,   161, 161, 470, 10,  11,  161, 423, 14,  162, 161, 17,  18,  19,  161, 160,
    147, 161, 24,  161, 161, 10,  161, 29,  138, 163, 162, 160, 34,  35,  138, 37,  38,
    39,  40,  3,   144, 166, 15,  161, 167, 160, 15,  10,  64,  161, 68,  53,  54,  162,
    56,  161, 67,  161, 160, 164, 62,  63,  162, 162, 162, 162, 68,  3,   80,  13,  72,
    161, 3,   98,  98,  164, 78,  3,   125, 36,  82,  161, 10,  81,  86,  111, 498, 499,
    161, 91,  3,   3,   94,  95,  96,  97,  98,  99,  3,   61,  161, 3,   162, 162, 162,
    98,  161, 109, 110, 3,   161, 161, 161, 161, 116, 152, 3,   160, 120, 6,   122, 123,
    124, 10,  11,  161, 161, 14,  130, 161, 17,  18,  19,  10,  3,   162, 131, 131, 0,
    162, 162, 162, 29,  145, 161, 9,   134, 34,  35,  162, 37,  38,  39,  40,  161, 33,
    223, 129, 381, 161, 434, 425, 459, 165, 209, 8,   53,  54,  296, 56,  303, 302, 152,
    163, 346, 195, 63,  -1,  -1,  -1,  -1,  68,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,
    -1,  78,  -1,  -1,  -1,  82,  -1,  -1,  -1,  86,  -1,  -1,  -1,  -1,  91,  -1,  -1,
    94,  95,  96,  97,  98,  99,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  109, 110,
    -1,  -1,  -1,  -1,  -1,  116, -1,  3,   -1,  120, 6,   122, 123, 124, 10,  11,  -1,
    -1,  -1,  130, -1,  17,  18,  -1,  -1,  -1,  -1,  -1,  24,  -1,  -1,  27,  -1,  29,
    145, -1,  -1,  -1,  34,  35,  -1,  37,  38,  39,  40,  -1,  -1,  -1,  -1,  -1,  161,
    -1,  -1,  -1,  165, -1,  -1,  53,  54,  -1,  56,  -1,  -1,  -1,  -1,  -1,  -1,  63,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,
    -1,  82,  -1,  -1,  -1,  86,  -1,  -1,  -1,  -1,  91,  -1,  -1,  94,  95,  96,  97,
    98,  99,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  109, 110, -1,  -1,  -1,  -1,
    -1,  116, 117, 3,   -1,  120, 6,   122, 123, 124, 10,  11,  -1,  -1,  14,  130, -1,
    17,  18,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  29,  145, -1,  -1,  -1,
    34,  35,  -1,  37,  38,  39,  40,  -1,  -1,  -1,  -1,  -1,  161, -1,  -1,  -1,  165,
    -1,  -1,  53,  54,  -1,  56,  -1,  -1,  -1,  -1,  -1,  -1,  63,  -1,  -1,  -1,  -1,
    68,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,  -1,  82,  -1,  -1,
    -1,  86,  -1,  -1,  -1,  -1,  91,  -1,  -1,  94,  95,  96,  97,  98,  99,  3,   -1,
    -1,  6,   -1,  -1,  -1,  10,  11,  109, 110, -1,  -1,  -1,  17,  18,  116, -1,  -1,
    -1,  120, -1,  122, 123, 124, -1,  29,  -1,  -1,  -1,  130, 34,  35,  -1,  37,  38,
    39,  40,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  145, -1,  -1,  -1,  -1,  53,  54,  -1,
    56,  -1,  -1,  -1,  -1,  -1,  -1,  63,  161, -1,  -1,  -1,  165, -1,  -1,  -1,  72,
    -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,  -1,  82,  -1,  -1,  -1,  86,  -1,  -1,  -1,
    -1,  91,  -1,  -1,  94,  95,  96,  97,  98,  99,  3,   -1,  -1,  6,   -1,  -1,  -1,
    10,  11,  109, 110, -1,  -1,  -1,  17,  18,  116, -1,  -1,  -1,  120, -1,  122, 123,
    124, -1,  29,  -1,  -1,  -1,  130, 34,  35,  -1,  37,  38,  39,  40,  138, -1,  -1,
    -1,  -1,  -1,  -1,  145, -1,  -1,  -1,  -1,  53,  54,  -1,  56,  -1,  -1,  -1,  -1,
    -1,  -1,  63,  161, -1,  -1,  -1,  165, -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,
    78,  -1,  -1,  -1,  82,  -1,  -1,  -1,  86,  -1,  -1,  -1,  -1,  91,  -1,  -1,  94,
    95,  96,  97,  98,  99,  3,   -1,  -1,  6,   -1,  -1,  -1,  10,  11,  109, 110, -1,
    -1,  -1,  17,  18,  116, -1,  -1,  -1,  120, -1,  122, 123, 124, -1,  29,  -1,  -1,
    -1,  130, 34,  35,  -1,  37,  38,  39,  40,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  145,
    -1,  -1,  -1,  -1,  53,  54,  -1,  56,  -1,  -1,  -1,  -1,  -1,  -1,  63,  161, -1,
    -1,  -1,  165, -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,  -1,  82,
    -1,  6,   -1,  86,  -1,  10,  11,  -1,  91,  -1,  -1,  94,  95,  96,  97,  98,  99,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  29,  -1,  109, 110, -1,  34,  35,  -1,  -1,  116,
    -1,  40,  -1,  120, -1,  122, 123, 124, -1,  -1,  -1,  -1,  -1,  130, 53,  54,  -1,
    56,  -1,  -1,  -1,  -1,  -1,  -1,  63,  -1,  -1,  -1,  145, -1,  -1,  -1,  -1,  72,
    -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,  -1,  -1,  161, 6,   -1,  86,  165, 10,  11,
    -1,  -1,  -1,  -1,  94,  -1,  96,  97,  98,  99,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    29,  -1,  109, 110, -1,  34,  35,  -1,  -1,  116, -1,  40,  -1,  120, -1,  122, 123,
    124, -1,  -1,  -1,  -1,  -1,  130, 53,  54,  -1,  56,  -1,  -1,  -1,  138, -1,  -1,
    63,  -1,  -1,  -1,  145, -1,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  -1,
    -1,  -1,  -1,  -1,  6,   -1,  86,  165, 10,  11,  -1,  -1,  -1,  -1,  94,  -1,  96,
    97,  98,  99,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  29,  -1,  109, 110, -1,  34,  35,
    -1,  -1,  116, -1,  40,  -1,  120, -1,  122, 123, 124, -1,  -1,  -1,  -1,  -1,  130,
    53,  54,  -1,  56,  -1,  -1,  -1,  -1,  -1,  -1,  63,  -1,  -1,  -1,  145, -1,  -1,
    -1,  -1,  72,  34,  35,  -1,  -1,  -1,  78,  40,  -1,  -1,  -1,  -1,  6,   -1,  86,
    165, 10,  11,  -1,  -1,  53,  -1,  94,  56,  96,  97,  98,  99,  -1,  -1,  63,  -1,
    -1,  -1,  -1,  29,  -1,  109, 110, 72,  34,  35,  -1,  -1,  116, 78,  40,  -1,  120,
    -1,  122, 123, 124, 86,  -1,  -1,  -1,  -1,  130, 53,  54,  94,  56,  96,  -1,  -1,
    99,  -1,  -1,  63,  -1,  -1,  -1,  145, -1,  -1,  109, 110, 72,  -1,  -1,  -1,  -1,
    116, 78,  -1,  -1,  120, -1,  122, 123, 124, 86,  165, -1,  -1,  -1,  130, -1,  -1,
    94,  -1,  96,  97,  98,  99,  14,  15,  16,  17,  18,  19,  20,  21,  -1,  109, 110,
    -1,  -1,  -1,  -1,  -1,  116, -1,  -1,  33,  120, -1,  122, 123, 124, -1,  -1,  -1,
    -1,  -1,  130, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    -1,  -1,  -1,  165, -1,  -1,  -1,  83,  84,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  92};

#line 325 "/usr/local/mapd-deps/20210608/lib/bison.cc"
/* fattrs + tables */

/* parser code folow  */

/* This is the parser code that is written into each bison parser
  when the %semantic_parser declaration is not specified in the grammar.
  It was written by Richard Stallman by simplifying the hairy parser
  used when %semantic_parser is specified.  */

/* Note: dollar marks section change
   the next  is replaced by the list of actions, each action
   as one case of the switch.  */

#if YY_Parser_USE_GOTO != 0
/*
 SUPRESSION OF GOTO : on some C++ compiler (sun c++)
  the goto is strictly forbidden if any constructor/destructor
  is used in the whole function (very stupid isn't it ?)
 so goto are to be replaced with a 'while/switch/case construct'
 here are the macro to keep some apparent compatibility
*/
#define YYGOTO(lb)     \
  {                    \
    yy_gotostate = lb; \
    continue;          \
  }
#define YYBEGINGOTO                          \
  enum yy_labels yy_gotostate = yygotostart; \
  for (;;)                                   \
    switch (yy_gotostate) {                  \
      case yygotostart: {
#define YYLABEL(lb) \
  }                 \
  case lb: {
#define YYENDGOTO \
  }               \
  }
#define YYBEGINDECLARELABEL enum yy_labels { yygotostart
#define YYDECLARELABEL(lb) , lb
#define YYENDDECLARELABEL \
  }                       \
  ;
#else
/* macro to keep goto */
#define YYGOTO(lb) goto lb
#define YYBEGINGOTO
#define YYLABEL(lb) \
  lb:
#define YYENDGOTO
#define YYBEGINDECLARELABEL
#define YYDECLARELABEL(lb)
#define YYENDDECLARELABEL
#endif
/* LABEL DECLARATION */
YYBEGINDECLARELABEL
YYDECLARELABEL(yynewstate)
YYDECLARELABEL(yybackup)
/* YYDECLARELABEL(yyresume) */
YYDECLARELABEL(yydefault)
YYDECLARELABEL(yyreduce)
YYDECLARELABEL(yyerrlab)  /* here on detecting error */
YYDECLARELABEL(yyerrlab1) /* here on error raised explicitly by an action */
YYDECLARELABEL(
    yyerrdefault) /* current state does not do anything special for the error token. */
YYDECLARELABEL(
    yyerrpop) /* pop the current state because it cannot handle the error token */
YYDECLARELABEL(yyerrhandle)
YYENDDECLARELABEL
/* ALLOCA SIMULATION */
/* __HAVE_NO_ALLOCA */
#ifdef __HAVE_NO_ALLOCA
int __alloca_free_ptr(char* ptr, char* ref) {
  if (ptr != ref)
    free(ptr);
  return 0;
}

#define __ALLOCA_alloca(size) malloc(size)
#define __ALLOCA_free(ptr, ref) __alloca_free_ptr((char*)ptr, (char*)ref)

#ifdef YY_Parser_LSP_NEEDED
#define __ALLOCA_return(num)                                        \
  return (__ALLOCA_free(yyss, yyssa) + __ALLOCA_free(yyvs, yyvsa) + \
          __ALLOCA_free(yyls, yylsa) + (num))
#else
#define __ALLOCA_return(num) \
  return (__ALLOCA_free(yyss, yyssa) + __ALLOCA_free(yyvs, yyvsa) + (num))
#endif
#else
#define __ALLOCA_return(num) return (num)
#define __ALLOCA_alloca(size) alloca(size)
#define __ALLOCA_free(ptr, ref)
#endif

/* ENDALLOCA SIMULATION */

#define yyerrok (yyerrstatus = 0)
#define yyclearin (YY_Parser_CHAR = YYEMPTY)
#define YYEMPTY -2
#define YYEOF 0
#define YYACCEPT __ALLOCA_return(0)
#define YYABORT __ALLOCA_return(1)
#define YYERROR YYGOTO(yyerrlab1)
/* Like YYERROR except do call yyerror.
   This remains here temporarily to ease the
   transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */
#define YYFAIL YYGOTO(yyerrlab)
#define YYRECOVERING() (!!yyerrstatus)
#define YYBACKUP(token, value)                            \
  do                                                      \
    if (YY_Parser_CHAR == YYEMPTY && yylen == 1) {        \
      YY_Parser_CHAR = (token), YY_Parser_LVAL = (value); \
      yychar1 = YYTRANSLATE(YY_Parser_CHAR);              \
      YYPOPSTACK;                                         \
      YYGOTO(yybackup);                                   \
    } else {                                              \
      YY_Parser_ERROR("syntax error: cannot back up");    \
      YYERROR;                                            \
    }                                                     \
  while (0)

#define YYTERROR 1
#define YYERRCODE 256

#ifndef YY_Parser_PURE
/* UNPURE */
#define YYLEX YY_Parser_LEX()
#ifndef YY_USE_CLASS
/* If nonreentrant, and not class , generate the variables here */
int YY_Parser_CHAR;             /*  the lookahead symbol        */
YY_Parser_STYPE YY_Parser_LVAL; /*  the semantic value of the */
                                /*  lookahead symbol    */
int YY_Parser_NERRS;            /*  number of parse errors so far */
#ifdef YY_Parser_LSP_NEEDED
YY_Parser_LTYPE YY_Parser_LLOC; /*  location data for the lookahead     */
                                /*  symbol                              */
#endif
#endif

#else
/* PURE */
#ifdef YY_Parser_LSP_NEEDED
#define YYLEX YY_Parser_LEX(&YY_Parser_LVAL, &YY_Parser_LLOC)
#else
#define YYLEX YY_Parser_LEX(&YY_Parser_LVAL)
#endif
#endif
#ifndef YY_USE_CLASS
#if YY_Parser_DEBUG != 0
int YY_Parser_DEBUG_FLAG; /*  nonzero means print parse trace     */
/* Since this is uninitialized, it does not stop multiple parsers
   from coexisting.  */
#endif
#endif

/*  YYINITDEPTH indicates the initial size of the parser's stacks       */

#ifndef YYINITDEPTH
#define YYINITDEPTH 200
#endif

/*  YYMAXDEPTH is the maximum size the stacks can grow to
    (effective only if the built-in stack extension method is used).  */

#if YYMAXDEPTH == 0
#undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

#if __GNUC__ > 1 /* GNU C and GNU C++ define this.  */
#define __yy_bcopy(FROM, TO, COUNT) __builtin_memcpy(TO, FROM, COUNT)
#else /* not GNU C or C++ */

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */

#ifdef __cplusplus
static void __yy_bcopy(char* from, char* to, int count)
#else
#ifdef __STDC__
static void __yy_bcopy(char* from, char* to, int count)
#else
static void __yy_bcopy(from, to, count) char* from;
char* to;
int count;
#endif
#endif
{
  char* f = from;
  char* t = to;
  int i = count;

  while (i-- > 0)
    *t++ = *f++;
}
#endif

int
#ifdef YY_USE_CLASS
 YY_Parser_CLASS::
#endif
     YY_Parser_PARSE(YY_Parser_PARSE_PARAM)
#ifndef __STDC__
#ifndef __cplusplus
#ifndef YY_USE_CLASS
/* parameter definition without protypes */
YY_Parser_PARSE_PARAM_DEF
#endif
#endif
#endif
{
  int yystate;
  int yyn;
  short* yyssp;
  YY_Parser_STYPE* yyvsp;
  int yyerrstatus; /*  number of tokens to shift before error messages enabled */
  int yychar1 = 0; /*  lookahead token as an internal (translated) token number */

  short yyssa[YYINITDEPTH];           /*  the state stack                     */
  YY_Parser_STYPE yyvsa[YYINITDEPTH]; /*  the semantic value stack            */

  short* yyss = yyssa;           /*  refer to the stacks thru separate pointers */
  YY_Parser_STYPE* yyvs = yyvsa; /*  to allow yyoverflow to reallocate them elsewhere */

#ifdef YY_Parser_LSP_NEEDED
  YY_Parser_LTYPE yylsa[YYINITDEPTH]; /*  the location stack                  */
  YY_Parser_LTYPE* yyls = yylsa;
  YY_Parser_LTYPE* yylsp;

#define YYPOPSTACK (yyvsp--, yyssp--, yylsp--)
#else
#define YYPOPSTACK (yyvsp--, yyssp--)
#endif

  int yystacksize = YYINITDEPTH;

#ifdef YY_Parser_PURE
  int YY_Parser_CHAR;
  YY_Parser_STYPE YY_Parser_LVAL;
  int YY_Parser_NERRS;
#ifdef YY_Parser_LSP_NEEDED
  YY_Parser_LTYPE YY_Parser_LLOC;
#endif
#endif

  YY_Parser_STYPE yyval; /*  the variable used to return         */
  /*  semantic values from the action     */
  /*  routines                            */

  int yylen;
  /* start loop, in which YYGOTO may be used. */
  YYBEGINGOTO

#if YY_Parser_DEBUG != 0
  if (YY_Parser_DEBUG_FLAG)
    fprintf(stderr, "Starting parse\n");
#endif
  yystate = 0;
  yyerrstatus = 0;
  YY_Parser_NERRS = 0;
  YY_Parser_CHAR = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss - 1;
  yyvsp = yyvs;
#ifdef YY_Parser_LSP_NEEDED
  yylsp = yyls;
#endif

  /* Push a new state, which is found in  yystate  .  */
  /* In all cases, when you get here, the value and location stacks
     have just been pushed. so pushing a state here evens the stacks.  */
  YYLABEL(yynewstate)

  *++yyssp = yystate;

  if (yyssp >= yyss + yystacksize - 1) {
    /* Give user a chance to reallocate the stack */
    /* Use copies of these so that the &'s don't force the real ones into memory. */
    YY_Parser_STYPE* yyvs1 = yyvs;
    short* yyss1 = yyss;
#ifdef YY_Parser_LSP_NEEDED
    YY_Parser_LTYPE* yyls1 = yyls;
#endif

    /* Get the current used size of the three stacks, in elements.  */
    int size = yyssp - yyss + 1;

#ifdef yyoverflow
    /* Each stack pointer address is followed by the size of
       the data in use in that stack, in bytes.  */
#ifdef YY_Parser_LSP_NEEDED
    /* This used to be a conditional around just the two extra args,
       but that might be undefined if yyoverflow is a macro.  */
    yyoverflow("parser stack overflow",
               &yyss1,
               size * sizeof(*yyssp),
               &yyvs1,
               size * sizeof(*yyvsp),
               &yyls1,
               size * sizeof(*yylsp),
               &yystacksize);
#else
    yyoverflow("parser stack overflow",
               &yyss1,
               size * sizeof(*yyssp),
               &yyvs1,
               size * sizeof(*yyvsp),
               &yystacksize);
#endif

    yyss = yyss1;
    yyvs = yyvs1;
#ifdef YY_Parser_LSP_NEEDED
    yyls = yyls1;
#endif
#else /* no yyoverflow */
    /* Extend the stack our own way.  */
    if (yystacksize >= YYMAXDEPTH) {
      YY_Parser_ERROR("parser stack overflow");
      __ALLOCA_return(2);
    }
    yystacksize *= 2;
    if (yystacksize > YYMAXDEPTH)
      yystacksize = YYMAXDEPTH;
    yyss = (short*)__ALLOCA_alloca(yystacksize * sizeof(*yyssp));
    __yy_bcopy((char*)yyss1, (char*)yyss, size * sizeof(*yyssp));
    __ALLOCA_free(yyss1, yyssa);
    yyvs = (YY_Parser_STYPE*)__ALLOCA_alloca(yystacksize * sizeof(*yyvsp));
    __yy_bcopy((char*)yyvs1, (char*)yyvs, size * sizeof(*yyvsp));
    __ALLOCA_free(yyvs1, yyvsa);
#ifdef YY_Parser_LSP_NEEDED
    yyls = (YY_Parser_LTYPE*)__ALLOCA_alloca(yystacksize * sizeof(*yylsp));
    __yy_bcopy((char*)yyls1, (char*)yyls, size * sizeof(*yylsp));
    __ALLOCA_free(yyls1, yylsa);
#endif
#endif /* no yyoverflow */

    yyssp = yyss + size - 1;
    yyvsp = yyvs + size - 1;
#ifdef YY_Parser_LSP_NEEDED
    yylsp = yyls + size - 1;
#endif

#if YY_Parser_DEBUG != 0
    if (YY_Parser_DEBUG_FLAG)
      fprintf(stderr, "Stack size increased to %d\n", yystacksize);
#endif

    if (yyssp >= yyss + yystacksize - 1)
      YYABORT;
  }

#if YY_Parser_DEBUG != 0
  if (YY_Parser_DEBUG_FLAG)
    fprintf(stderr, "Entering state %d\n", yystate);
#endif

  YYGOTO(yybackup);
  YYLABEL(yybackup)

  /* Do appropriate processing given the current state.  */
  /* Read a lookahead token if we need one and don't already have one.  */
  /* YYLABEL(yyresume) */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    YYGOTO(yydefault);

  /* Not known => get a lookahead token if don't already have one.  */

  /* yychar is either YYEMPTY or YYEOF
     or a valid token in external form.  */

  if (YY_Parser_CHAR == YYEMPTY) {
#if YY_Parser_DEBUG != 0
    if (YY_Parser_DEBUG_FLAG)
      fprintf(stderr, "Reading a token: ");
#endif
    YY_Parser_CHAR = YYLEX;
  }

  /* Convert token to internal form (in yychar1) for indexing tables with */

  if (YY_Parser_CHAR <= 0) /* This means end of input. */
  {
    yychar1 = 0;
    YY_Parser_CHAR = YYEOF; /* Don't call YYLEX any more */

#if YY_Parser_DEBUG != 0
    if (YY_Parser_DEBUG_FLAG)
      fprintf(stderr, "Now at end of input.\n");
#endif
  } else {
    yychar1 = YYTRANSLATE(YY_Parser_CHAR);

#if YY_Parser_DEBUG != 0
    if (YY_Parser_DEBUG_FLAG) {
      fprintf(stderr, "Next token is %d (%s", YY_Parser_CHAR, yytname[yychar1]);
      /* Give the individual parser a way to print the precise meaning
         of a token, for further debugging info.  */
#ifdef YYPRINT
      YYPRINT(stderr, YY_Parser_CHAR, YY_Parser_LVAL);
#endif
      fprintf(stderr, ")\n");
    }
#endif
  }

  yyn += yychar1;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != yychar1)
    YYGOTO(yydefault);

  yyn = yytable[yyn];

  /* yyn is what to do for this token type in this state.
     Negative => reduce, -yyn is rule number.
     Positive => shift, yyn is new state.
       New state is final state => don't bother to shift,
       just return success.
     0, or most negative number => error.  */

  if (yyn < 0) {
    if (yyn == YYFLAG)
      YYGOTO(yyerrlab);
    yyn = -yyn;
    YYGOTO(yyreduce);
  } else if (yyn == 0)
    YYGOTO(yyerrlab);

  if (yyn == YYFINAL)
    YYACCEPT;

    /* Shift the lookahead token.  */

#if YY_Parser_DEBUG != 0
  if (YY_Parser_DEBUG_FLAG)
    fprintf(stderr, "Shifting token %d (%s), ", YY_Parser_CHAR, yytname[yychar1]);
#endif

  /* Discard the token being shifted unless it is eof.  */
  if (YY_Parser_CHAR != YYEOF)
    YY_Parser_CHAR = YYEMPTY;

  *++yyvsp = YY_Parser_LVAL;
#ifdef YY_Parser_LSP_NEEDED
  *++yylsp = YY_Parser_LLOC;
#endif

  /* count tokens shifted since error; after three, turn off error status.  */
  if (yyerrstatus)
    yyerrstatus--;

  yystate = yyn;
  YYGOTO(yynewstate);

  /* Do the default action for the current state.  */
  YYLABEL(yydefault)

  yyn = yydefact[yystate];
  if (yyn == 0)
    YYGOTO(yyerrlab);

  /* Do a reduction.  yyn is the number of a rule to reduce with.  */
  YYLABEL(yyreduce)
  yylen = yyr2[yyn];
  if (yylen > 0)
    yyval = yyvsp[1 - yylen]; /* implement default value of the action */

#if YY_Parser_DEBUG != 0
  if (YY_Parser_DEBUG_FLAG) {
    int i;

    fprintf(stderr, "Reducing via rule %d (line %d), ", yyn, yyrline[yyn]);

    /* Print the symbols being reduced, and their result.  */
    for (i = yyprhs[yyn]; yyrhs[i] > 0; i++)
      fprintf(stderr, "%s ", yytname[yyrhs[i]]);
    fprintf(stderr, " -> %s\n", yytname[yyr1[yyn]]);
  }
#endif

  /* #line 811 "/usr/local/mapd-deps/20210608/lib/bison.cc" */

  switch (yyn) {
    case 1: {
      parseTrees.emplace_front(dynamic_cast<Stmt*>((yyvsp[-1].nodeval)->release()));
      ;
      break;
    }
    case 2: {
      parseTrees.emplace_front(dynamic_cast<Stmt*>((yyvsp[-1].nodeval)->release()));
      ;
      break;
    }
    case 3: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 4: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 5: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 6: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 7: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 8: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 9: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 10: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 11: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 12: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 13: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new NameValueAssign((yyvsp[-2].stringval)->release(),
                              dynamic_cast<Literal*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 14: {
      yyval.boolval = true;
      ;
      break;
    }
    case 15: {
      yyval.boolval = false;
      ;
      break;
    }
    case 16: {
      yyval.boolval = true;
      ;
      break;
    }
    case 17: {
      yyval.boolval = false;
      ;
      break;
    }
    case 18: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CreateTableStmt(
              (yyvsp[-4].stringval)->release(),
              nullptr,
              reinterpret_cast<std::list<TableElement*>*>((yyvsp[-2].listval)->release()),
              yyvsp[-7].boolval,
              yyvsp[-5].boolval,
              reinterpret_cast<std::list<NameValueAssign*>*>(
                  (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 19: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CreateTableStmt(
              (yyvsp[-5].stringval)->release(),
              (yyvsp[-8].stringval)->release(),
              reinterpret_cast<std::list<TableElement*>*>((yyvsp[-3].listval)->release()),
              false,
              yyvsp[-6].boolval,
              reinterpret_cast<std::list<NameValueAssign*>*>(
                  (yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 20: {
      yyval.boolval = true;
      ;
      break;
    }
    case 21: {
      yyval.boolval = false;
      ;
      break;
    }
    case 22: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new DropTableStmt((yyvsp[0].stringval)->release(), yyvsp[-1].boolval));
      ;
      break;
    }
    case 23: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new RenameTableStmt((yyvsp[-3].stringval)->release(),
                                                     (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 24: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new RenameColumnStmt((yyvsp[-5].stringval)->release(),
                                                      (yyvsp[-2].stringval)->release(),
                                                      (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 27: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 28: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 29: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new AddColumnStmt((yyvsp[-3].stringval)->release(),
                            dynamic_cast<ColumnDef*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 30: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new AddColumnStmt(
              (yyvsp[-4].stringval)->release(),
              reinterpret_cast<std::list<ColumnDef*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 31: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new DropColumnStmt((yyvsp[-1].stringval)->release(),
                                                    (yyvsp[0].slistval)->release()));
      ;
      break;
    }
    case 32: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 33: {
      (yyvsp[-2].listval)->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 34: {
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
    case 35: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new AlterTableParamStmt(
              (yyvsp[-2].stringval)->release(),
              reinterpret_cast<NameValueAssign*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 36: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 37: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 38: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 39: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 40: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnDef((yyvsp[-2].stringval)->release(),
                        dynamic_cast<SQLType*>((yyvsp[-1].nodeval)->release()),
                        dynamic_cast<CompressDef*>((yyvsp[0].nodeval)->release()),
                        nullptr));
      ;
      break;
    }
    case 41: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnDef(
              (yyvsp[-3].stringval)->release(),
              dynamic_cast<SQLType*>((yyvsp[-2].nodeval)->release()),
              dynamic_cast<CompressDef*>((yyvsp[0].nodeval)->release()),
              dynamic_cast<ColumnConstraintDef*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 42: {
      if (!boost::iequals(*(yyvsp[-1].stringval)->get(), "encoding"))
        throw std::runtime_error("Invalid identifier " + *(yyvsp[-1].stringval)->get() +
                                 " in column definition.");
      delete (yyvsp[-1].stringval)->release();
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new CompressDef((yyvsp[0].stringval)->release(), 0));
      ;
      break;
    }
    case 43: {
      if (!boost::iequals(*(yyvsp[-4].stringval)->get(), "encoding"))
        throw std::runtime_error("Invalid identifier " + *(yyvsp[-4].stringval)->get() +
                                 " in column definition.");
      delete (yyvsp[-4].stringval)->release();
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CompressDef((yyvsp[-3].stringval)->release(), (int)yyvsp[-1].intval));
      ;
      break;
    }
    case 44: {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 45: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new ColumnConstraintDef(true, false, false, nullptr));
      ;
      break;
    }
    case 46: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new ColumnConstraintDef(true, true, false, nullptr));
      ;
      break;
    }
    case 47: {
      if (!boost::iequals(*(yyvsp[0].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[0].stringval)->get());
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new ColumnConstraintDef(true, true, true, nullptr));
      ;
      break;
    }
    case 48: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef(false,
                                  false,
                                  false,
                                  dynamic_cast<Literal*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 49: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef(false, false, false, new NullLiteral()));
      ;
      break;
    }
    case 50: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef(false, false, false, new UserLiteral()));
      ;
      break;
    }
    case 51: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 52: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef((yyvsp[0].stringval)->release(), nullptr));
      ;
      break;
    }
    case 53: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef((yyvsp[-3].stringval)->release(),
                                  (yyvsp[-1].stringval)->release()));
      ;
      break;
    }
    case 54: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new UniqueDef(false, (yyvsp[-1].slistval)->release()));
      ;
      break;
    }
    case 55: {
      if (!boost::iequals(*(yyvsp[-3].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[-3].stringval)->get());
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new UniqueDef(true, (yyvsp[-1].slistval)->release()));
      ;
      break;
    }
    case 56: {
      if (!boost::iequals(*(yyvsp[-5].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[-5].stringval)->get());
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ForeignKeyDef(
              (yyvsp[-3].slistval)->release(), (yyvsp[0].stringval)->release(), nullptr));
      ;
      break;
    }
    case 57: {
      if (!boost::iequals(*(yyvsp[-8].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[-8].stringval)->get());
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new ForeignKeyDef((yyvsp[-6].slistval)->release(),
                                                   (yyvsp[-3].stringval)->release(),
                                                   (yyvsp[-1].slistval)->release()));
      ;
      break;
    }
    case 58: {
      if (!boost::iequals(*(yyvsp[-3].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[-3].stringval)->get());
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new ShardKeyDef(*(yyvsp[-1].stringval)->get()));
      delete (yyvsp[-3].stringval)->release();
      delete (yyvsp[-1].stringval)->release();
      ;
      break;
    }
    case 59: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SharedDictionaryDef(*(yyvsp[-6].stringval)->get(),
                                                         *(yyvsp[-3].stringval)->get(),
                                                         *(yyvsp[-1].stringval)->get()));
      delete (yyvsp[-6].stringval)->release();
      delete (yyvsp[-3].stringval)->release();
      delete (yyvsp[-1].stringval)->release();
      ;
      break;
    }
    case 60: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CheckDef(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 61: {
      yyval.slistval = TrackedListPtr<std::string>::make(
          lexer.parsed_str_list_tokens_, 1, yyvsp[0].stringval);
      ;
      break;
    }
    case 62: {
      yyval.slistval = yyvsp[-2].slistval;
      yyval.slistval->push_back(yyvsp[0].stringval);
      ;
      break;
    }
    case 63: {
      yyval.listval = yyvsp[-1].listval;
      ;
      break;
    }
    case 64: {
      yyval.listval = TrackedListPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 65: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new DropViewStmt((yyvsp[0].stringval)->release(), yyvsp[-1].boolval));
      ;
      break;
    }
    case 66: {
      yyval.slistval = TrackedListPtr<std::string>::makeEmpty();
      ;
      break;
    }
    case 67: {
      yyval.slistval = yyvsp[-1].slistval;
      ;
      break;
    }
    case 68: {
      yyval.listval = TrackedListPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 69: {
      yyval.listval = yyvsp[0].listval;
      ;
      break;
    }
    case 70: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 71: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 72: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OrderSpec(yyvsp[-2].intval, nullptr, yyvsp[-1].boolval, yyvsp[0].boolval));
      ;
      break;
    }
    case 73: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OrderSpec(0,
                        dynamic_cast<ColumnRef*>((yyvsp[-2].nodeval)->release()),
                        yyvsp[-1].boolval,
                        yyvsp[0].boolval));
      ;
      break;
    }
    case 74: {
      yyval.boolval = false; /* default is ASC */
      ;
      break;
    }
    case 75: {
      yyval.boolval = false;
      ;
      break;
    }
    case 76: {
      yyval.boolval = true;
      ;
      break;
    }
    case 77: {
      yyval.boolval = false; /* default is NULL LAST */
      ;
      break;
    }
    case 78: {
      yyval.boolval = true;
      ;
      break;
    }
    case 79: {
      yyval.boolval = false;
      ;
      break;
    }
    case 85: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new DeleteStmt((yyvsp[-1].stringval)->release(),
                         dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 86: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InsertValuesStmt(
              (yyvsp[-5].stringval)->release(),
              (yyvsp[-4].slistval)->release(),
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 87: {
      yyval.boolval = false;
      ;
      break;
    }
    case 88: {
      yyval.boolval = false;
      ;
      break;
    }
    case 89: {
      yyval.boolval = true;
      ;
      break;
    }
    case 90: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 91: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 92: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new Assignment((yyvsp[-2].stringval)->release(),
                         dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 93: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new UpdateStmt(
              (yyvsp[-3].stringval)->release(),
              reinterpret_cast<std::list<Assignment*>*>((yyvsp[-1].listval)->release()),
              dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 94: {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 95: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 96: {
      yyval.intval = yyvsp[0].intval;
      if (yyval.intval <= 0)
        throw std::runtime_error("LIMIT must be positive.");
      ;
      break;
    }
    case 97: {
      yyval.intval = 0; /* 0 means ALL */
      ;
      break;
    }
    case 98: {
      yyval.intval = 0; /* 0 means ALL */
      ;
      break;
    }
    case 99: {
      yyval.intval = yyvsp[0].intval;
      ;
      break;
    }
    case 100: {
      if (!boost::iequals(*(yyvsp[0].stringval)->get(), "row") &&
          !boost::iequals(*(yyvsp[0].stringval)->get(), "rows"))
        throw std::runtime_error("Invalid word in OFFSET clause " +
                                 *(yyvsp[0].stringval)->get());
      delete (yyvsp[0].stringval)->release();
      yyval.intval = yyvsp[-1].intval;
      ;
      break;
    }
    case 101: {
      yyval.intval = 0;
      ;
      break;
    }
    case 102: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SelectStmt(
              dynamic_cast<QueryExpr*>((yyvsp[-3].nodeval)->release()),
              reinterpret_cast<std::list<OrderSpec*>*>((yyvsp[-2].listval)->release()),
              yyvsp[-1].intval,
              yyvsp[0].intval));
      ;
      break;
    }
    case 103: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 104: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new UnionQuery(false,
                         dynamic_cast<QueryExpr*>((yyvsp[-2].nodeval)->release()),
                         dynamic_cast<QueryExpr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 105: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new UnionQuery(true,
                         dynamic_cast<QueryExpr*>((yyvsp[-3].nodeval)->release()),
                         dynamic_cast<QueryExpr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 106: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 107: {
      yyval.nodeval = yyvsp[-1].nodeval;
      ;
      break;
    }
    case 108: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new QuerySpec(
              yyvsp[-5].boolval,
              reinterpret_cast<std::list<SelectEntry*>*>((yyvsp[-4].listval)->release()),
              reinterpret_cast<std::list<TableRef*>*>((yyvsp[-3].listval)->release()),
              dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release()),
              dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 109: {
      yyval.listval = yyvsp[0].listval;
      ;
      break;
    }
    case 110: {
      yyval.listval = TrackedListPtr<Node>::makeEmpty(); /* nullptr means SELECT * */
      ;
      break;
    }
    case 111: {
      yyval.listval = yyvsp[0].listval;
      ;
      break;
    }
    case 112: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 113: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 114: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new TableRef((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 115: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new TableRef((yyvsp[-1].stringval)->release(),
                                              (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 116: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 117: {
      yyval.listval = TrackedListPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 118: {
      yyval.listval = yyvsp[0].listval;
      ;
      break;
    }
    case 119: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 120: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 121: {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 122: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 123: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kOR,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 124: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kAND,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 125: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(
              kNOT, dynamic_cast<Expr*>((yyvsp[0].nodeval)->release()), nullptr));
      ;
      break;
    }
    case 126: {
      yyval.nodeval = yyvsp[-1].nodeval;
      ;
      break;
    }
    case 127: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 128: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 129: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 130: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 131: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 132: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 133: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 134: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 135: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 136: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(yyvsp[-1].opval,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 137: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(yyvsp[-1].opval,
                       kONE,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      /* subquery can only return a single result */
      ;
      break;
    }
    case 138: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new BetweenExpr(true,
                          dynamic_cast<Expr*>((yyvsp[-5].nodeval)->release()),
                          dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                          dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 139: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new BetweenExpr(false,
                          dynamic_cast<Expr*>((yyvsp[-4].nodeval)->release()),
                          dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                          dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 140: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new LikeExpr(true,
                       false,
                       dynamic_cast<Expr*>((yyvsp[-4].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 141: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new LikeExpr(false,
                       false,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 142: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new LikeExpr(true,
                       true,
                       dynamic_cast<Expr*>((yyvsp[-4].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 143: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new LikeExpr(false,
                       true,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 144: {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 145: {
      std::string escape_tok = *(yyvsp[-1].stringval)->get();
      std::transform(escape_tok.begin(), escape_tok.end(), escape_tok.begin(), ::tolower);
      if (escape_tok != "escape") {
        throw std::runtime_error("Syntax error: wrong escape specifier");
      }
      delete (yyvsp[-1].stringval)->release();
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 146: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new IsNullExpr(true, dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release())));
      ;
      break;
    }
    case 147: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new IsNullExpr(false, dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release())));
      ;
      break;
    }
    case 148: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InSubquery(true,
                         dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                         dynamic_cast<SubqueryExpr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 149: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InSubquery(false,
                         dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                         dynamic_cast<SubqueryExpr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 150: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InValues(
              true,
              dynamic_cast<Expr*>((yyvsp[-5].nodeval)->release()),
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 151: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InValues(
              false,
              dynamic_cast<Expr*>((yyvsp[-4].nodeval)->release()),
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 152: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 153: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 154: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(yyvsp[-2].opval,
                       yyvsp[-1].qualval,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 155: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(yyvsp[-2].opval,
                       yyvsp[-1].qualval,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 156: {
      yyval.opval = yyvsp[0].opval;
      ;
      break;
    }
    case 157: {
      yyval.opval = yyvsp[0].opval;
      ;
      break;
    }
    case 161: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ExistsExpr(dynamic_cast<QuerySpec*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 162: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SubqueryExpr(dynamic_cast<QuerySpec*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 163: {
      yyval.listval = TrackedListPtr<Node>::make(
          lexer.parsed_node_list_tokens_,
          1,
          new ExprPair(dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 164: {
      yyval.listval = yyvsp[-4].listval;
      yyval.listval->push_back(
          new ExprPair(dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 165: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 166: {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 167: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CaseExpr(
              reinterpret_cast<std::list<ExprPair*>*>((yyvsp[-2].listval)->release()),
              dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 168: {
      std::list<ExprPair*>* when_then_list = new std::list<ExprPair*>(
          1,
          new ExprPair(dynamic_cast<Expr*>((yyvsp[-5].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release())));
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CaseExpr(when_then_list,
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 169: {
      std::list<ExprPair*>* when_then_list = new std::list<ExprPair*>(
          1,
          new ExprPair(dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new CaseExpr(when_then_list, nullptr));
      ;
      break;
    }
    case 170: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CharLengthExpr(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()), true));
      ;
      break;
    }
    case 171: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CharLengthExpr(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()), false));
      ;
      break;
    }
    case 172: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kARRAY_AT,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 173: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kPLUS,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 174: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kMINUS,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 175: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kMULTIPLY,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 176: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kDIVIDE,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 177: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kMODULO,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 178: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kMODULO,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 179: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 180: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(
              kUMINUS, dynamic_cast<Expr*>((yyvsp[0].nodeval)->release()), nullptr));
      ;
      break;
    }
    case 181: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 182: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 183: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 184: {
      yyval.nodeval = yyvsp[-1].nodeval;
      ;
      break;
    }
    case 185: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CastExpr(dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<SQLType*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 186: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 187: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 188: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 189: {
      throw std::runtime_error("Empty select entry");
      ;
      break;
    }
    case 190: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SelectEntry(dynamic_cast<Expr*>((yyvsp[0].nodeval)->release()), nullptr));
      ;
      break;
    }
    case 191: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SelectEntry(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()),
                          (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 192: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SelectEntry(dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                          (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 193: {
      throw std::runtime_error("Empty select entry list");
      ;
      break;
    }
    case 194: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 195: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 196: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 197: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new UserLiteral());
      ;
      break;
    }
    case 198: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new FunctionRef((yyvsp[-3].stringval)->release()));
      ;
      break;
    }
    case 199: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new FunctionRef((yyvsp[-4].stringval)->release(),
                          true,
                          dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 200: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new FunctionRef((yyvsp[-4].stringval)->release(),
                          dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 201: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new FunctionRef((yyvsp[-3].stringval)->release(),
                          dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 202: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new StringLiteral((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 203: {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new IntLiteral(yyvsp[0].intval));
      ;
      break;
    }
    case 204: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TimestampLiteral());
      ;
      break;
    }
    case 205: {
      delete dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release());
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TimestampLiteral());
      ;
      break;
    }
    case 206: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new FixedPtLiteral((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 207: {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new FloatLiteral(yyvsp[0].floatval));
      ;
      break;
    }
    case 208: {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new DoubleLiteral(yyvsp[0].doubleval));
      ;
      break;
    }
    case 209: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CastExpr(new StringLiteral((yyvsp[0].stringval)->release()),
                       dynamic_cast<SQLType*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 210: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ArrayLiteral(
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 211: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ArrayLiteral(
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 212: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new NullLiteral());
      ;
      break;
    }
    case 213: {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 214: {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 215: {
      yyval.listval = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 0);
      ;
      break;
    }
    case 217: {
      const auto uc_col_name =
          boost::to_upper_copy<std::string>(*(yyvsp[0].stringval)->get());
      if (reserved_keywords.find(uc_col_name) != reserved_keywords.end()) {
        errors_.push_back("Cannot use a reserved keyword as table name: " +
                          *(yyvsp[0].stringval)->get());
      }
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
    case 218: {
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
    case 219: {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 225: {
      yyval.slistval = TrackedListPtr<std::string>::make(
          lexer.parsed_str_list_tokens_, 1, yyvsp[0].stringval);
      ;
      break;
    }
    case 226: {
      yyval.slistval = yyvsp[-2].slistval;
      yyval.slistval->push_back(yyvsp[0].stringval);
      ;
      break;
    }
    case 229: {
      yyval.slistval = TrackedListPtr<std::string>::make(
          lexer.parsed_str_list_tokens_, 1, yyvsp[0].stringval);
      ;
      break;
    }
    case 230: {
      yyval.slistval = yyvsp[-2].slistval;
      yyval.slistval->push_back(yyvsp[0].stringval);
      ;
      break;
    }
    case 233: {
      yyval.slistval = TrackedListPtr<std::string>::make(
          lexer.parsed_str_list_tokens_, 1, yyvsp[0].stringval);
      ;
      break;
    }
    case 234: {
      yyval.slistval = yyvsp[-2].slistval;
      yyval.slistval->push_back(yyvsp[0].stringval);
      ;
      break;
    }
    case 235: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ALL");
      ;
      break;
    }
    case 236: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ALL");
      ;
      break;
    }
    case 237: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE");
      ;
      break;
    }
    case 238: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "SELECT");
      ;
      break;
    }
    case 239: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "INSERT");
      ;
      break;
    }
    case 240: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "TRUNCATE");
      ;
      break;
    }
    case 241: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "UPDATE");
      ;
      break;
    }
    case 242: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DELETE");
      ;
      break;
    }
    case 243: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ALTER");
      ;
      break;
    }
    case 244: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DROP");
      ;
      break;
    }
    case 245: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "VIEW");
      ;
      break;
    }
    case 246: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "EDIT");
      ;
      break;
    }
    case 247: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ACCESS");
      ;
      break;
    }
    case 248: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "USAGE");
      ;
      break;
    }
    case 249: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "SERVER USAGE");
      ;
      break;
    }
    case 250: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ALTER SERVER");
      ;
      break;
    }
    case 251: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE SERVER");
      ;
      break;
    }
    case 252: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE TABLE");
      ;
      break;
    }
    case 253: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE VIEW");
      ;
      break;
    }
    case 254: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "SELECT VIEW");
      ;
      break;
    }
    case 255: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DROP VIEW");
      ;
      break;
    }
    case 256: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DROP SERVER");
      ;
      break;
    }
    case 257: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE DASHBOARD");
      ;
      break;
    }
    case 258: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "EDIT DASHBOARD");
      ;
      break;
    }
    case 259: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "VIEW DASHBOARD");
      ;
      break;
    }
    case 260: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DELETE DASHBOARD");
      ;
      break;
    }
    case 261: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "VIEW SQL EDITOR");
      ;
      break;
    }
    case 262: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DATABASE");
      ;
      break;
    }
    case 263: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "TABLE");
      ;
      break;
    }
    case 264: {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DASHBOARD");
      ;
      break;
    }
    case 265: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "VIEW");
      ;
      break;
    }
    case 266: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "SERVER");
      ;
      break;
    }
    case 268: {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_,
                                                      std::to_string(yyvsp[0].intval));
      ;
      break;
    }
    case 269: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new ColumnRef((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 270: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new ColumnRef((yyvsp[-2].stringval)->release(),
                                               (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 271: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnRef((yyvsp[-2].stringval)->release(), nullptr));
      ;
      break;
    }
    case 272: {
      if (yyvsp[0].intval < 0)
        throw std::runtime_error("No negative number in type definition.");
      yyval.intval = yyvsp[0].intval;
      ;
      break;
    }
    case 273: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kBIGINT));
      ;
      break;
    }
    case 274: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTEXT));
      ;
      break;
    }
    case 275: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kBOOLEAN));
      ;
      break;
    }
    case 276: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kCHAR));
      ;
      break;
    }
    case 277: {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kCHAR, yyvsp[-1].intval));
      ;
      break;
    }
    case 278: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kNUMERIC));
      ;
      break;
    }
    case 279: {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kNUMERIC, yyvsp[-1].intval));
      ;
      break;
    }
    case 280: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SQLType(kNUMERIC, yyvsp[-3].intval, yyvsp[-1].intval, false));
      ;
      break;
    }
    case 281: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDECIMAL));
      ;
      break;
    }
    case 282: {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kDECIMAL, yyvsp[-1].intval));
      ;
      break;
    }
    case 283: {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SQLType(kDECIMAL, yyvsp[-3].intval, yyvsp[-1].intval, false));
      ;
      break;
    }
    case 284: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kINT));
      ;
      break;
    }
    case 285: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTINYINT));
      ;
      break;
    }
    case 286: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kSMALLINT));
      ;
      break;
    }
    case 287: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kFLOAT));
      ;
      break;
    }
    case 288: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kFLOAT));
      ;
      break;
    }
    case 289: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDOUBLE));
      ;
      break;
    }
    case 290: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDOUBLE));
      ;
      break;
    }
    case 291: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDATE));
      ;
      break;
    }
    case 292: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTIME));
      ;
      break;
    }
    case 293: {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kTIME, yyvsp[-1].intval));
      ;
      break;
    }
    case 294: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTIMESTAMP));
      ;
      break;
    }
    case 295: {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kTIMESTAMP, yyvsp[-1].intval));
      ;
      break;
    }
    case 296: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[0].intval),
                                             static_cast<int>(kGEOMETRY),
                                             0,
                                             false));
      ;
      break;
    }
    case 297: {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 298: {
      yyval.nodeval = yyvsp[-2].nodeval;
      if (dynamic_cast<SQLType*>((yyval.nodeval)->get())->get_is_array())
        throw std::runtime_error("array of array not supported.");
      dynamic_cast<SQLType*>((yyval.nodeval)->get())->set_is_array(true);
      ;
      break;
    }
    case 299: {
      yyval.nodeval = yyvsp[-3].nodeval;
      if (dynamic_cast<SQLType*>((yyval.nodeval)->get())->get_is_array())
        throw std::runtime_error("array of array not supported.");
      dynamic_cast<SQLType*>((yyval.nodeval)->get())->set_is_array(true);
      dynamic_cast<SQLType*>((yyval.nodeval)->get())->set_array_size(yyvsp[-1].intval);
      ;
      break;
    }
    case 300: {
      yyval.intval = kPOINT;
      ;
      break;
    }
    case 301: {
      yyval.intval = kLINESTRING;
      ;
      break;
    }
    case 302: {
      yyval.intval = kPOLYGON;
      ;
      break;
    }
    case 303: {
      yyval.intval = kMULTIPOLYGON;
      ;
      break;
    }
    case 304: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[-1].intval),
                                             static_cast<int>(kGEOGRAPHY),
                                             4326,
                                             false));
      ;
      break;
    }
    case 305: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[-3].intval),
                                             static_cast<int>(kGEOGRAPHY),
                                             yyvsp[-1].intval,
                                             false));
      ;
      break;
    }
    case 306: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[-1].intval),
                                             static_cast<int>(kGEOMETRY),
                                             0,
                                             false));
      ;
      break;
    }
    case 307: {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[-3].intval),
                                             static_cast<int>(kGEOMETRY),
                                             yyvsp[-1].intval,
                                             false));
      ;
      break;
    }
    case 308: {
      const auto uc_col_name =
          boost::to_upper_copy<std::string>(*(yyvsp[0].stringval)->get());
      if (reserved_keywords.find(uc_col_name) != reserved_keywords.end()) {
        errors_.push_back("Cannot use a reserved keyword as column name: " +
                          *(yyvsp[0].stringval)->get());
      }
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
    case 309: {
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
    case 310: {
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
  }

#line 811 "/usr/local/mapd-deps/20210608/lib/bison.cc"
  /* the action file gets copied in in place of this dollarsign  */
  yyvsp -= yylen;
  yyssp -= yylen;
#ifdef YY_Parser_LSP_NEEDED
  yylsp -= yylen;
#endif

#if YY_Parser_DEBUG != 0
  if (YY_Parser_DEBUG_FLAG) {
    short* ssp1 = yyss - 1;
    fprintf(stderr, "state stack now");
    while (ssp1 != yyssp)
      fprintf(stderr, " %d", *++ssp1);
    fprintf(stderr, "\n");
  }
#endif

  *++yyvsp = yyval;

#ifdef YY_Parser_LSP_NEEDED
  yylsp++;
  if (yylen == 0) {
    yylsp->first_line = YY_Parser_LLOC.first_line;
    yylsp->first_column = YY_Parser_LLOC.first_column;
    yylsp->last_line = (yylsp - 1)->last_line;
    yylsp->last_column = (yylsp - 1)->last_column;
    yylsp->text = 0;
  } else {
    yylsp->last_line = (yylsp + yylen - 1)->last_line;
    yylsp->last_column = (yylsp + yylen - 1)->last_column;
  }
#endif

  /* Now "shift" the result of the reduction.
     Determine what state that goes to,
     based on the state we popped back to
     and the rule number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTBASE] + *yyssp;
  if (yystate >= 0 && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTBASE];

  YYGOTO(yynewstate);

  YYLABEL(yyerrlab) /* here on detecting error */

  if (!yyerrstatus)
  /* If not already recovering from an error, report this error.  */
  {
    ++YY_Parser_NERRS;

#ifdef YY_Parser_ERROR_VERBOSE
    yyn = yypact[yystate];

    if (yyn > YYFLAG && yyn < YYLAST) {
      int size = 0;
      char* msg;
      int x, count;

      count = 0;
      /* Start X at -yyn if nec to avoid negative indexes in yycheck.  */
      for (x = (yyn < 0 ? -yyn : 0); x < (sizeof(yytname) / sizeof(char*)); x++)
        if (yycheck[x + yyn] == x)
          size += strlen(yytname[x]) + 15, count++;
      msg = (char*)malloc(size + 15);
      if (msg != 0) {
        strcpy(msg, "parse error");

        if (count < 5) {
          count = 0;
          for (x = (yyn < 0 ? -yyn : 0); x < (sizeof(yytname) / sizeof(char*)); x++)
            if (yycheck[x + yyn] == x) {
              strcat(msg, count == 0 ? ", expecting `" : " or `");
              strcat(msg, yytname[x]);
              strcat(msg, "'");
              count++;
            }
        }
        YY_Parser_ERROR(msg);
        free(msg);
      } else
        YY_Parser_ERROR("parse error; also virtual memory exceeded");
    } else
#endif /* YY_Parser_ERROR_VERBOSE */
      YY_Parser_ERROR("parse error");
  }

  YYGOTO(yyerrlab1);
  YYLABEL(yyerrlab1) /* here on error raised explicitly by an action */

  if (yyerrstatus == 3) {
    /* if just tried and failed to reuse lookahead token after an error, discard it.  */

    /* return failure if at end of input */
    if (YY_Parser_CHAR == YYEOF)
      YYABORT;

#if YY_Parser_DEBUG != 0
    if (YY_Parser_DEBUG_FLAG)
      fprintf(stderr, "Discarding token %d (%s).\n", YY_Parser_CHAR, yytname[yychar1]);
#endif

    YY_Parser_CHAR = YYEMPTY;
  }

  /* Else will try to reuse lookahead token
     after shifting the error token.  */

  yyerrstatus = 3; /* Each real token shifted decrements this */

  YYGOTO(yyerrhandle);

  YYLABEL(
      yyerrdefault) /* current state does not do anything special for the error token. */

#if 0
  /* This is wrong; only states that explicitly want error tokens
     should shift them.  */
  yyn = yydefact[yystate];  /* If its default is to accept any token, ok.  Otherwise pop it.*/
  if (yyn) YYGOTO(yydefault);
#endif

  YYLABEL(yyerrpop) /* pop the current state because it cannot handle the error token */

  if (yyssp == yyss)
    YYABORT;
  yyvsp--;
  yystate = *--yyssp;
#ifdef YY_Parser_LSP_NEEDED
  yylsp--;
#endif

#if YY_Parser_DEBUG != 0
  if (YY_Parser_DEBUG_FLAG) {
    short* ssp1 = yyss - 1;
    fprintf(stderr, "Error: state stack now");
    while (ssp1 != yyssp)
      fprintf(stderr, " %d", *++ssp1);
    fprintf(stderr, "\n");
  }
#endif

  YYLABEL(yyerrhandle)

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    YYGOTO(yyerrdefault);

  yyn += YYTERROR;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != YYTERROR)
    YYGOTO(yyerrdefault);

  yyn = yytable[yyn];
  if (yyn < 0) {
    if (yyn == YYFLAG)
      YYGOTO(yyerrpop);
    yyn = -yyn;
    YYGOTO(yyreduce);
  } else if (yyn == 0)
    YYGOTO(yyerrpop);

  if (yyn == YYFINAL)
    YYACCEPT;

#if YY_Parser_DEBUG != 0
  if (YY_Parser_DEBUG_FLAG)
    fprintf(stderr, "Shifting error token, ");
#endif

  *++yyvsp = YY_Parser_LVAL;
#ifdef YY_Parser_LSP_NEEDED
  *++yylsp = YY_Parser_LLOC;
#endif

  yystate = yyn;
  YYGOTO(yynewstate);
  /* end loop, in which YYGOTO may be used. */
  YYENDGOTO
}

/* END */

/* #line 1010 "/usr/local/mapd-deps/20210608/lib/bison.cc" */
