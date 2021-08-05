#define YY_Parser_h_included

/*  A Bison++ parser, made from Parser/parser.y  */

/* with Bison++ version bison++ version 1.21-45, adapted from GNU Bison by
 * coetmeur@icdc.fr
 */

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
#include "Parser/FlexLexer.h"
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

/*  YY_Parser_PURE */
#endif

/* section apres lecture def, avant lecture grammaire S2 */

/* prefix */
#ifndef YY_Parser_DEBUG

/* YY_Parser_DEBUG */
#endif

#ifndef YY_Parser_LSP_NEEDED

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
#define PUBLIC 378
#define REAL 379
#define REFERENCES 380
#define RENAME 381
#define RESTORE 382
#define REVOKE 383
#define ROLE 384
#define ROLLBACK 385
#define SCHEMA 386
#define SELECT 387
#define SET 388
#define SHARD 389
#define SHARED 390
#define SHOW 391
#define UNIQUE 392
#define UPDATE 393
#define USER 394
#define VALIDATE 395
#define VALUES 396
#define VIEW 397
#define WHEN 398
#define WHENEVER 399
#define WHERE 400
#define WITH 401
#define WORK 402
#define EDIT 403
#define ACCESS 404
#define DASHBOARD 405
#define SQL 406
#define EDITOR 407

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

  /* decl const */
#else
  enum YY_Parser_ENUM_TOKEN {
    YY_Parser_NULL_TOKEN = 0

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
    PUBLIC = 378,
    REAL = 379,
    REFERENCES = 380,
    RENAME = 381,
    RESTORE = 382,
    REVOKE = 383,
    ROLE = 384,
    ROLLBACK = 385,
    SCHEMA = 386,
    SELECT = 387,
    SET = 388,
    SHARD = 389,
    SHARED = 390,
    SHOW = 391,
    UNIQUE = 392,
    UPDATE = 393,
    USER = 394,
    VALIDATE = 395,
    VALUES = 396,
    VIEW = 397,
    WHEN = 398,
    WHENEVER = 399,
    WHERE = 400,
    WITH = 401,
    WORK = 402,
    EDIT = 403,
    ACCESS = 404,
    DASHBOARD = 405,
    SQL = 406,
    EDITOR = 407

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
const int YY_Parser_CLASS::PUBLIC = 378;
const int YY_Parser_CLASS::REAL = 379;
const int YY_Parser_CLASS::REFERENCES = 380;
const int YY_Parser_CLASS::RENAME = 381;
const int YY_Parser_CLASS::RESTORE = 382;
const int YY_Parser_CLASS::REVOKE = 383;
const int YY_Parser_CLASS::ROLE = 384;
const int YY_Parser_CLASS::ROLLBACK = 385;
const int YY_Parser_CLASS::SCHEMA = 386;
const int YY_Parser_CLASS::SELECT = 387;
const int YY_Parser_CLASS::SET = 388;
const int YY_Parser_CLASS::SHARD = 389;
const int YY_Parser_CLASS::SHARED = 390;
const int YY_Parser_CLASS::SHOW = 391;
const int YY_Parser_CLASS::UNIQUE = 392;
const int YY_Parser_CLASS::UPDATE = 393;
const int YY_Parser_CLASS::USER = 394;
const int YY_Parser_CLASS::VALIDATE = 395;
const int YY_Parser_CLASS::VALUES = 396;
const int YY_Parser_CLASS::VIEW = 397;
const int YY_Parser_CLASS::WHEN = 398;
const int YY_Parser_CLASS::WHENEVER = 399;
const int YY_Parser_CLASS::WHERE = 400;
const int YY_Parser_CLASS::WITH = 401;
const int YY_Parser_CLASS::WORK = 402;
const int YY_Parser_CLASS::EDIT = 403;
const int YY_Parser_CLASS::ACCESS = 404;
const int YY_Parser_CLASS::DASHBOARD = 405;
const int YY_Parser_CLASS::SQL = 406;
const int YY_Parser_CLASS::EDITOR = 407;

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

#define YYFINAL 686
#define YYFLAG -32768
#define YYNTBASE 167

#define YYTRANSLATE(x) ((unsigned)(x) <= 407 ? yytranslate[x] : 288)

static const short yytranslate[] = {
    0,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   21,  2,   2,   159, 160, 19,  17,  161, 18,  166, 20,  2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   158, 2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   162, 2,   163, 2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
    2,   2,   2,   2,   164, 2,   165, 2,   2,   2,   2,   2,   2,   2,   2,   2,   2,
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
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157};

#if YY_Parser_DEBUG != 0
static const short yyprhs[] = {
    0,    0,    3,    7,    9,    11,   13,   15,   17,   19,   21,   23,   25,   27,
    29,   31,   33,   35,   37,   39,   41,   43,   45,   47,   49,   51,   53,   55,
    57,   59,   61,   63,   68,   76,   81,   88,   95,   99,   106,  113,  115,  119,
    123,  127,  128,  130,  131,  140,  150,  160,  170,  175,  178,  179,  184,  188,
    195,  204,  205,  207,  209,  213,  220,  228,  233,  235,  239,  243,  249,  257,
    259,  261,  268,  275,  279,  283,  291,  299,  304,  309,  314,  318,  320,  324,
    326,  328,  332,  337,  340,  346,  347,  350,  354,  359,  362,  365,  368,  373,
    376,  382,  387,  393,  401,  412,  418,  429,  434,  436,  440,  445,  446,  451,
    452,  456,  457,  461,  463,  467,  471,  475,  476,  478,  480,  481,  484,  487,
    489,  491,  493,  495,  497,  502,  511,  517,  518,  520,  522,  524,  528,  532,
    538,  539,  541,  544,  547,  548,  551,  555,  556,  561,  563,  567,  572,  574,
    578,  586,  588,  590,  593,  595,  599,  601,  604,  607,  608,  612,  614,  618,
    619,  622,  626,  630,  633,  637,  639,  641,  643,  645,  647,  649,  651,  653,
    655,  659,  663,  670,  676,  682,  687,  693,  698,  699,  702,  707,  711,  716,
    720,  727,  733,  735,  739,  744,  749,  751,  753,  755,  757,  759,  762,  766,
    771,  777,  780,  781,  786,  795,  802,  807,  812,  817,  821,  825,  829,  833,
    837,  844,  847,  850,  852,  854,  856,  860,  867,  869,  871,  873,  874,  876,
    879,  883,  884,  886,  890,  892,  894,  899,  905,  911,  916,  918,  920,  924,
    929,  931,  933,  935,  938,  942,  947,  949,  951,  955,  956,  958,  960,  962,
    963,  965,  967,  969,  971,  973,  975,  979,  981,  983,  985,  989,  991,  993,
    995,  999,  1001, 1004, 1006, 1008, 1010, 1012, 1014, 1016, 1018, 1020, 1022, 1024,
    1026, 1029, 1032, 1035, 1038, 1041, 1044, 1047, 1050, 1053, 1056, 1059, 1063, 1065,
    1067, 1069, 1071, 1073, 1075, 1077, 1079, 1083, 1087, 1089, 1091, 1093, 1095, 1097,
    1102, 1104, 1109, 1116, 1118, 1123, 1130, 1132, 1134, 1136, 1138, 1140, 1143, 1145,
    1147, 1149, 1154, 1156, 1161, 1163, 1165, 1169, 1174, 1176, 1178, 1180, 1182, 1187,
    1194, 1199, 1206, 1208, 1210};

static const short yyrhs[] = {
    168, 158, 0,   167, 168, 158, 0,   179, 0,   180, 0,   181, 0,   182, 0,   214, 0,
    184, 0,   185, 0,   186, 0,   187, 0,   190, 0,   191, 0,   194, 0,   169, 0,   170,
    0,   171, 0,   172, 0,   173, 0,   174, 0,   198, 0,   199, 0,   200, 0,   201, 0,
    202, 0,   204, 0,   205, 0,   203, 0,   196, 0,   197, 0,   48,  51,  177, 3,   0,
    48,  51,  177, 3,   159, 175, 160, 0,   64,  51,  183, 3,   0,   25,  51,  3,   131,
    125, 3,   0,   48,  144, 272, 159, 175, 160, 0,   64,  144, 272, 0,   25,  144, 272,
    159, 175, 160, 0,   25,  144, 272, 131, 125, 272, 0,   176, 0,   175, 161, 176, 0,
    3,   15,  267, 0,   82,  14,  68,  0,   0,   119, 0,   0,   48,  178, 118, 177, 270,
    30,  8,   213, 0,   48,  178, 118, 177, 270, 159, 206, 160, 213, 0,   48,  3,   118,
    177, 270, 159, 206, 160, 213, 0,   48,  52,  270, 159, 206, 160, 76,  6,   213, 0,
    141, 48,  118, 270, 0,   82,  68,  0,   0,   64,  118, 183, 270, 0,   126, 118, 270,
    0,   25,  118, 270, 131, 125, 270, 0,   25,  118, 270, 131, 44,  286, 125, 286, 0,
    0,   44,  0,   208, 0,   189, 161, 208, 0,   25,  118, 270, 23,  188, 208, 0,   25,
    118, 270, 23,  159, 189, 160, 0,   25,  118, 270, 192, 0,   193, 0,   192, 161, 193,
    0,   64,  188, 286, 0,   47,  270, 76,  6,   213, 0,   47,  159, 7,   160, 125, 6,
    213, 0,   65,  0,   28,  0,   195, 118, 270, 125, 6,   213, 0,   132, 118, 270, 76,
    6,   213, 0,   48,  134, 274, 0,   64,  134, 274, 0,   79,  277, 102, 279, 280, 125,
    275, 0,   133, 277, 102, 279, 280, 76,  275, 0,   79,  273, 125, 275, 0,   133, 273,
    76,  275, 0,   104, 118, 271, 213, 0,   145, 43,  213, 0,   207, 0,   206, 161, 207,
    0,   208, 0,   211, 0,   286, 283, 209, 0,   286, 283, 210, 209, 0,   3,   3,   0,
    3,   3,   159, 10,  160, 0,   0,   14,  98,  0,   14,  98,  142, 0,   14,  98,  112,
    3,   0,   58,  267, 0,   58,  98,  0,   58,  144, 0,   41,  159, 243, 160, 0,   130,
    270, 0,   130, 270, 159, 286, 160, 0,   142, 159, 212, 160, 0,   112, 3,   159, 212,
    160, 0,   74,  3,   159, 212, 160, 130, 270, 0,   74,  3,   159, 212, 160, 130, 270,
    159, 212, 160, 0,   139, 3,   159, 286, 160, 0,   140, 61,  159, 286, 160, 130, 270,
    159, 286, 160, 0,   41,  159, 243, 160, 0,   286, 0,   212, 161, 286, 0,   151, 159,
    175, 160, 0,   0,   64,  147, 183, 270, 0,   0,   159, 212, 160, 0,   0,   107, 36,
    217, 0,   218, 0,   217, 161, 218, 0,   10,  219, 220, 0,   281, 219, 220, 0,   0,
    31,  0,   60,  0,   0,   98,  71,  0,   98,  90,  0,   221, 0,   222, 0,   223, 0,
    231, 0,   227, 0,   59,  76,  270, 228, 0,   85,  87,  270, 215, 146, 159, 251, 160,
    0,   85,  87,  270, 215, 8,   0,   0,   24,  0,   62,  0,   226, 0,   225, 161, 226,
    0,   286, 15,  243, 0,   143, 270, 138, 225, 228, 0,   0,   239, 0,   93,  10,  0,
    93,  24,  0,   0,   101, 10,  0,   101, 10,  3,   0,   0,   232, 216, 229, 230, 0,
    233, 0,   232, 127, 233, 0,   232, 127, 24,  233, 0,   234, 0,   159, 232, 160, 0,
    137, 224, 235, 236, 228, 240, 242, 0,   264, 0,   19,  0,   76,  237, 0,   238, 0,
    237, 161, 238, 0,   270, 0,   270, 287, 0,   150, 243, 0,   0,   80,  36,  241, 0,
    243, 0,   241, 161, 243, 0,   0,   81,  243, 0,   243, 12,  243, 0,   243, 13,  243,
    0,   14,  243, 0,   159, 243, 160, 0,   244, 0,   245, 0,   246, 0,   247, 0,   249,
    0,   250, 0,   252, 0,   255, 0,   262, 0,   262, 253, 262, 0,   262, 253, 256, 0,
    262, 14,  33,  262, 13,  262, 0,   262, 33,  262, 13,  262, 0,   262, 14,  92,  265,
    248, 0,   262, 92,  265, 248, 0,   262, 14,  83,  265, 248, 0,   262, 83,  265, 248,
    0,   0,   3,   265, 0,   281, 88,  14,  98,  0,   281, 88,  98,  0,   262, 14,  84,
    256, 0,   262, 84,  256, 0,   262, 14,  84,  159, 251, 160, 0,   262, 84,  159, 251,
    160, 0,   265, 0,   251, 161, 265, 0,   262, 253, 254, 256, 0,   262, 253, 254, 262,
    0,   15,  0,   16,  0,   27,  0,   24,  0,   117, 0,   68,  256, 0,   159, 234, 160,
    0,   148, 243, 121, 243, 0,   257, 148, 243, 121, 243, 0,   66,  243, 0,   0,   37,
    257, 258, 67,  0,   82,  159, 243, 161, 243, 161, 243, 160, 0,   82,  159, 243, 161,
    243, 160, 0,   39,  159, 262, 160, 0,   91,  159, 262, 160, 0,   281, 162, 262, 163,
    0,   262, 17,  262, 0,   262, 18,  262, 0,   262, 19,  262, 0,   262, 20,  262, 0,
    262, 21,  262, 0,   95,  159, 262, 161, 262, 160, 0,   17,  262, 0,   18,  262, 0,
    265, 0,   281, 0,   266, 0,   159, 262, 160, 0,   38,  159, 243, 30,  283, 160, 0,
    259, 0,   260, 0,   261, 0,   0,   243, 0,   243, 3,   0,   243, 30,  3,   0,   0,
    263, 0,   264, 161, 263, 0,   267, 0,   144, 0,   3,   159, 19,  160, 0,   3,   159,
    62,  243, 160, 0,   3,   159, 24,  243, 160, 0,   3,   159, 243, 160, 0,   6,   0,
    10,  0,   97,  159, 160, 0,   54,  159, 243, 160, 0,   11,  0,   72,  0,   63,  0,
    283, 6,   0,   164, 269, 165, 0,   29,  162, 269, 163, 0,   98,  0,   267, 0,   268,
    161, 267, 0,   0,   268, 0,   3,   0,   9,   0,   0,   270, 0,   3,   0,   5,   0,
    4,   0,   9,   0,   274, 0,   273, 161, 274, 0,   3,   0,   4,   0,   276, 0,   275,
    161, 276, 0,   272, 0,   274, 0,   278, 0,   277, 161, 278, 0,   24,  0,   24,  113,
    0,   48,  0,   137, 0,   85,  0,   126, 0,   143, 0,   59,  0,   25,  0,   64,  0,
    147, 0,   153, 0,   154, 0,   25,  115, 0,   48,  115, 0,   48,  118, 0,   48,  147,
    0,   137, 147, 0,   64,  147, 0,   64,  115, 0,   48,  155, 0,   153, 155, 0,   147,
    155, 0,   59,  155, 0,   147, 156, 157, 0,   51,  0,   118, 0,   155, 0,   147, 0,
    115, 0,   3,   0,   10,  0,   3,   0,   3,   166, 3,   0,   3,   166, 19,  0,   10,
    0,   34,  0,   120, 0,   35,  0,   40,  0,   40,  159, 282, 160, 0,   99,  0,   99,
    159, 282, 160, 0,   99,  159, 282, 161, 282, 160, 0,   56,  0,   56,  159, 282, 160,
    0,   56,  159, 282, 161, 282, 160, 0,   86,  0,   124, 0,   116, 0,   72,  0,   129,
    0,   63,  111, 0,   63,  0,   53,  0,   122, 0,   122, 159, 282, 160, 0,   123, 0,
    123, 159, 282, 160, 0,   284, 0,   285, 0,   283, 162, 163, 0,   283, 162, 282, 163,
    0,   109, 0,   94,  0,   110, 0,   96,  0,   77,  159, 284, 160, 0,   77,  159, 284,
    161, 10,  160, 0,   78,  159, 284, 160, 0,   78,  159, 284, 161, 10,  160, 0,   3,
    0,   9,   0,   3,   0};

#endif

#if YY_Parser_DEBUG != 0
static const short yyrline[] = {
    0,    120,  122,  130,  132,  133,  134,  136,  137,  138,  139,  140,  141,  142,
    143,  144,  145,  146,  147,  148,  149,  150,  151,  152,  153,  154,  155,  156,
    157,  158,  159,  185,  190,  195,  201,  208,  214,  220,  225,  232,  237,  243,
    246,  248,  251,  253,  256,  263,  268,  274,  281,  287,  289,  292,  298,  304,
    311,  318,  319,  321,  323,  330,  335,  341,  345,  347,  350,  353,  358,  368,
    369,  371,  378,  385,  391,  397,  403,  409,  415,  422,  429,  436,  438,  445,
    447,  450,  453,  457,  465,  472,  475,  477,  478,  484,  485,  486,  487,  488,
    489,  492,  495,  501,  508,  514,  522,  529,  533,  535,  542,  545,  549,  556,
    558,  613,  615,  618,  620,  627,  630,  634,  636,  637,  640,  642,  643,  648,
    651,  657,  660,  662,  683,  688,  693,  715,  717,  718,  728,  731,  738,  743,
    759,  761,  764,  766,  767,  769,  771,  778,  784,  791,  793,  795,  799,  801,
    804,  816,  818,  821,  825,  827,  834,  836,  839,  843,  845,  848,  850,  857,
    859,  864,  867,  869,  871,  872,  875,  877,  878,  879,  880,  881,  882,  883,
    886,  889,  896,  899,  903,  906,  908,  910,  914,  916,  928,  930,  933,  936,
    948,  950,  954,  956,  963,  968,  974,  976,  979,  981,  982,  985,  989,  993,
    998,  1004, 1006, 1009, 1013, 1018, 1025, 1027, 1032, 1040, 1042, 1043, 1044, 1045,
    1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1055, 1056, 1057, 1060, 1062, 1063,
    1064, 1067, 1069, 1070, 1077, 1079, 1092, 1094, 1095, 1096, 1099, 1101, 1102, 1103,
    1104, 1105, 1106, 1107, 1108, 1109, 1110, 1113, 1115, 1122, 1124, 1129, 1139, 1142,
    1144, 1146, 1147, 1147, 1147, 1150, 1152, 1159, 1160, 1163, 1165, 1172, 1173, 1176,
    1178, 1185, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198,
    1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1213, 1215,
    1216, 1217, 1218, 1221, 1223, 1227, 1229, 1230, 1234, 1243, 1245, 1246, 1247, 1248,
    1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1260, 1261, 1262, 1263,
    1264, 1265, 1266, 1267, 1268, 1270, 1271, 1278, 1288, 1289, 1290, 1291, 1294, 1296,
    1299, 1301, 1306, 1315, 1330};

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
                                      "'('",
                                      "')'",
                                      "','",
                                      "'['",
                                      "']'",
                                      "'{'",
                                      "'}'",
                                      "'.'",
                                      "sql_list",
                                      "sql",
                                      "create_database_statement",
                                      "drop_database_statement",
                                      "rename_database_statement",
                                      "create_user_statement",
                                      "drop_user_statement",
                                      "alter_user_statement",
                                      "name_eq_value_list",
                                      "name_eq_value",
                                      "opt_if_not_exists",
                                      "opt_temporary",
                                      "create_table_as_statement",
                                      "create_table_statement",
                                      "create_dataframe_statement",
                                      "show_table_schema",
                                      "opt_if_exists",
                                      "drop_table_statement",
                                      "truncate_table_statement",
                                      "rename_table_statement",
                                      "rename_column_statement",
                                      "opt_column",
                                      "column_defs",
                                      "add_column_statement",
                                      "drop_column_statement",
                                      "drop_columns",
                                      "drop_column",
                                      "copy_table_statement",
                                      "dump_or_archive",
                                      "dump_table_statement",
                                      "restore_table_statement",
                                      "create_role_statement",
                                      "drop_role_statement",
                                      "grant_privilege_statement",
                                      "revoke_privileges_statement",
                                      "grant_role_statement",
                                      "revoke_role_statement",
                                      "optimize_table_statement",
                                      "validate_system_statement",
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
                                      "opt_table",
                                      "username",
                                      "rolenames",
                                      "rolename",
                                      "grantees",
                                      "grantee",
                                      "privileges",
                                      "privilege",
                                      "privileges_target_type",
                                      "privileges_target",
                                      "column_ref",
                                      "non_neg_int",
                                      "data_type",
                                      "geo_type",
                                      "geometry_type",
                                      "column",
                                      "range_variable",
                                      "range_variable"};
#endif

static const short yyr1[] = {
    0,   167, 167, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168,
    168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 168, 169, 169, 170,
    171, 172, 173, 174, 174, 175, 175, 176, 177, 177, 178, 178, 179, 180, 180, 181, 182,
    183, 183, 184, 185, 186, 187, 188, 188, 189, 189, 190, 190, 191, 192, 192, 193, 194,
    194, 195, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 206, 207, 207,
    208, 208, 209, 209, 209, 210, 210, 210, 210, 210, 210, 210, 210, 210, 211, 211, 211,
    211, 211, 211, 211, 212, 212, 213, 213, 214, 215, 215, 216, 216, 217, 217, 218, 218,
    219, 219, 219, 220, 220, 220, 168, 221, 221, 221, 221, 222, 223, 223, 224, 224, 224,
    225, 225, 226, 227, 228, 228, 229, 229, 229, 230, 230, 230, 231, 232, 232, 232, 233,
    233, 234, 235, 235, 236, 237, 237, 238, 238, 239, 240, 240, 241, 241, 242, 242, 243,
    243, 243, 243, 243, 244, 244, 244, 244, 244, 244, 244, 244, 245, 245, 246, 246, 247,
    247, 247, 247, 248, 248, 249, 249, 250, 250, 250, 250, 251, 251, 252, 252, 253, 253,
    254, 254, 254, 255, 256, 257, 257, 258, 258, 259, 259, 259, 260, 260, 261, 262, 262,
    262, 262, 262, 262, 262, 262, 262, 262, 262, 262, 262, 262, 262, 262, 263, 263, 263,
    263, 264, 264, 264, 265, 265, 266, 266, 266, 266, 267, 267, 267, 267, 267, 267, 267,
    267, 267, 267, 267, 268, 268, 269, 269, 270, 270, 271, 271, 272, 272, 272, 272, 273,
    273, 274, 274, 275, 275, 276, 276, 277, 277, 278, 278, 278, 278, 278, 278, 278, 278,
    278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278,
    279, 279, 279, 279, 279, 280, 280, 281, 281, 281, 282, 283, 283, 283, 283, 283, 283,
    283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283,
    283, 283, 283, 283, 284, 284, 284, 284, -1,  -1,  285, 285, 286, 286, 287};

static const short yyr2[] = {
    0, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1,  1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 4, 7, 4, 6, 6, 3, 6, 6, 1, 3, 3, 3, 0, 1, 0, 8,  9, 9,  9, 4, 2, 0, 4, 3, 6,
    8, 0, 1, 1, 3, 6, 7, 4, 1, 3, 3, 5, 7, 1, 1, 6, 6, 3, 3,  7, 7,  4, 4, 4, 3, 1, 3, 1,
    1, 3, 4, 2, 5, 0, 2, 3, 4, 2, 2, 2, 4, 2, 5, 4, 5, 7, 10, 5, 10, 4, 1, 3, 4, 0, 4, 0,
    3, 0, 3, 1, 3, 3, 3, 0, 1, 1, 0, 2, 2, 1, 1, 1, 1, 1, 4,  8, 5,  0, 1, 1, 1, 3, 3, 5,
    0, 1, 2, 2, 0, 2, 3, 0, 4, 1, 3, 4, 1, 3, 7, 1, 1, 2, 1,  3, 1,  2, 2, 0, 3, 1, 3, 0,
    2, 3, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 6, 5, 5,  4, 5,  4, 0, 2, 4, 3, 4, 3,
    6, 5, 1, 3, 4, 4, 1, 1, 1, 1, 1, 2, 3, 4, 5, 2, 0, 4, 8,  6, 4,  4, 4, 3, 3, 3, 3, 3,
    6, 2, 2, 1, 1, 1, 3, 6, 1, 1, 1, 0, 1, 2, 3, 0, 1, 3, 1,  1, 4,  5, 5, 4, 1, 1, 3, 4,
    1, 1, 1, 2, 3, 4, 1, 1, 3, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1,  1, 3,  1, 1, 1, 3, 1, 1, 1,
    3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,  2, 2,  2, 2, 2, 2, 3, 1, 1,
    1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 4, 1, 4, 6, 1, 4,  6, 1,  1, 1, 1, 1, 2, 1, 1,
    1, 4, 1, 4, 1, 1, 3, 4, 1, 1, 1, 1, 4, 6, 4, 6, 1, 1, 1};

static const short yydefact[] = {
    0,   0,   70,  0,   45,  0,   0,   69,  0,   0,   0,   0,   0,   0,   133, 0,   0,
    0,   0,   0,   0,   15,  16,  17,  18,  19,  20,  3,   4,   5,   6,   8,   9,   10,
    11,  12,  13,  14,  0,   29,  30,  21,  22,  23,  24,  25,  28,  26,  27,  7,   125,
    126, 127, 129, 128, 113, 149, 152, 0,   0,   0,   263, 264, 0,   0,   0,   43,  0,
    44,  0,   0,   0,   0,   52,  52,  0,   0,   52,  273, 274, 281, 289, 283, 288, 290,
    285, 286, 284, 287, 291, 292, 293, 0,   271, 0,   279, 0,   265, 0,   0,   0,   0,
    134, 135, 235, 0,   0,   109, 0,   0,   1,   0,   0,   0,   144, 0,   0,   267, 269,
    268, 270, 0,   0,   0,   43,  0,   0,   0,   73,  0,   43,  140, 0,   0,   0,   74,
    36,  0,   282, 294, 295, 296, 297, 301, 304, 300, 299, 298, 303, 0,   302, 0,   0,
    0,   0,   111, 266, 109, 54,  0,   0,   0,   313, 248, 249, 252, 0,   0,   0,   156,
    0,   317, 319, 0,   0,   0,   320, 335, 0,   325, 254, 0,   253, 0,   0,   328, 0,
    345, 0,   347, 0,   258, 322, 344, 346, 330, 318, 336, 338, 329, 332, 243, 0,   261,
    0,   236, 173, 174, 175, 176, 177, 178, 179, 180, 232, 233, 234, 181, 240, 155, 227,
    229, 242, 228, 0,   340, 341, 0,   0,   0,   80,  153, 2,   0,   0,   0,   150, 0,
    147, 0,   57,  57,  0,   63,  64,  0,   0,   0,   109, 0,   0,   31,  0,   0,   0,
    0,   130, 141, 51,  33,  53,  110, 305, 267, 269, 277, 278, 77,  275, 272, 306, 310,
    307, 309, 308, 0,   280, 0,   0,   79,  0,   78,  0,   0,   0,   171, 0,   225, 228,
    226, 261, 0,   212, 0,   0,   0,   0,   0,   333, 0,   207, 0,   0,   0,   0,   0,
    0,   0,   0,   0,   181, 259, 262, 0,   0,   140, 237, 0,   0,   0,   0,   202, 203,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   235, 0,   0,   255, 0,   50,  352,
    353, 140, 136, 0,   0,   0,   313, 119, 114, 115, 119, 151, 142, 143, 0,   148, 0,
    58,  0,   0,   0,   0,   0,   0,   0,   0,   0,   39,  0,   67,  0,   42,  0,   0,
    0,   0,   0,   0,   0,   0,   81,  83,  84,  0,   0,   0,   162, 0,   311, 312, 0,
    0,   106, 132, 0,   109, 0,   0,   0,   0,   0,   314, 315, 0,   0,   0,   0,   0,
    0,   0,   0,   316, 0,   0,   0,   0,   0,   0,   0,   0,   250, 0,   0,   0,   172,
    230, 0,   256, 157, 158, 160, 163, 169, 170, 238, 0,   0,   0,   0,   219, 220, 221,
    222, 223, 0,   190, 0,   195, 190, 205, 204, 206, 0,   0,   183, 182, 241, 0,   193,
    0,   342, 0,   0,   139, 0,   0,   109, 120, 121, 122, 0,   122, 145, 34,  0,   59,
    61,  66,  0,   55,  65,  38,  0,   37,  0,   109, 0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   334, 331, 89,  35,  0,   0,   276, 0,   112, 0,   0,   72,  0,   244,
    0,   0,   247, 257, 0,   211, 0,   213, 0,   216, 321, 251, 326, 0,   208, 350, 0,
    0,   217, 0,   323, 0,   337, 339, 260, 0,   354, 161, 0,   167, 0,   190, 0,   194,
    190, 0,   0,   189, 0,   198, 187, 200, 201, 192, 218, 343, 137, 138, 108, 71,  0,
    117, 116, 118, 146, 62,  0,   0,   41,  40,  68,  0,   32,  0,   0,   0,   0,   0,
    0,   0,   82,  0,   0,   0,   0,   0,   85,  89,  109, 0,   75,  107, 0,   76,  246,
    245, 209, 0,   0,   0,   0,   0,   0,   0,   159, 0,   0,   154, 0,   188, 0,   186,
    185, 191, 197, 0,   123, 124, 60,  56,  109, 105, 0,   0,   0,   0,   99,  109, 87,
    90,  0,   94,  95,  93,  97,  86,  46,  109, 131, 210, 231, 327, 351, 215, 0,   224,
    324, 164, 165, 168, 184, 196, 199, 48,  0,   100, 103, 0,   49,  0,   0,   91,  0,
    0,   47,  0,   0,   0,   0,   0,   92,  96,  0,   214, 166, 101, 0,   88,  98,  0,
    0,   0,   0,   102, 104, 0,   0};

static const short yydefgoto[] = {
    19,  20,  21,  22,  23,  24,  25,  26,  366, 367, 126, 71,  27,  28,  29,  30,
    133, 31,  32,  33,  34,  359, 474, 35,  36,  243, 244, 37,  38,  39,  40,  41,
    42,  43,  44,  45,  46,  47,  48,  379, 380, 381, 586, 587, 382, 391, 230, 49,
    278, 114, 348, 349, 469, 561, 50,  51,  52,  104, 341, 342, 53,  256, 238, 355,
    54,  55,  56,  57,  204, 315, 428, 429, 257, 539, 647, 607, 205, 206, 207, 208,
    209, 547, 210, 211, 548, 212, 332, 453, 213, 300, 292, 408, 214, 215, 216, 217,
    218, 219, 220, 221, 222, 312, 313, 430, 157, 265, 92,  266, 267, 268, 94,  95,
    275, 390, 223, 412, 224, 225, 226, 383, 537};

static const short yypact[] = {
    1499,   167,    -32768, 13,     204,    -39,    160,    -32768, 277,    -7,
    18,     41,     103,    277,    28,     127,    317,    188,    139,    268,
    107,    -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
    -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, 172,    -32768,
    -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768,
    -32768, -32768, -32768, -32768, -32768, 192,    -32768, -32768, 300,    317,
    431,    -32768, -32768, 324,    280,    272,    262,    317,    -32768, 336,
    431,    274,    317,    288,    288,    336,    431,    288,    -32768, -32768,
    286,    289,    132,    271,    117,    -32768, -32768, 260,    -32768, -84,
    282,    -32768, -38,    -32768, 97,     -32768, 317,    317,    317,    317,
    -6,     123,    -32768, -32768, 604,    292,    322,    314,    77,     265,
    -32768, 317,    441,    58,     374,    347,    66,     -32768, -32768, -32768,
    -32768, 55,     325,    482,    262,    485,    493,    353,    -32768, 362,
    262,    378,    433,    528,    317,    -32768, -32768, 317,    -32768, -32768,
    -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, 390,
    -32768, 445,    336,    270,    752,    392,    -32768, 314,    -32768, 477,
    445,    270,    171,    -32768, -32768, -32768, 829,    1054,   1054,   -32768,
    393,    -32768, -32768, 404,    397,    398,    400,    -32768, 402,    405,
    11,     406,    20,     407,    409,    -32768, 410,    -32768, 417,    -32768,
    418,    -32768, 419,    -32768, -32768, -32768, -32768, 420,    421,    -32768,
    -32768, -32768, 829,    1463,   487,    305,    -32768, -32768, -32768, -32768,
    -32768, -32768, -32768, -32768, -32768, -32768, -32768, 633,    -32768, 422,
    -32768, -32768, -32768, -57,    21,     -32768, -32768, 317,    319,    425,
    -32768, -32768, -32768, 456,    342,    139,    -32768, 230,    488,    466,
    44,     549,    35,     434,    -32768, 469,    594,    478,    314,    317,
    533,    443,    448,    594,    317,    829,    -32768, -32768, -32768, -32768,
    -32768, -32768, -32768, -32768, -32768, -32768, -32768, 447,    -32768, -32768,
    -32768, -32768, -32768, -32768, -32768, 351,    -32768, 319,    45,     -32768,
    598,    447,    351,    476,    65,     -32768, 1054,   -32768, 444,    -32768,
    1463,   829,    75,     829,    1054,   599,    829,    599,    -32768, 474,
    -32768, 312,    829,    1054,   1054,   452,    599,    599,    599,    27,
    360,    -32768, 458,    451,    317,    378,    -32768, 829,    829,    610,
    183,    -32768, -32768, 1054,   1054,   1054,   1054,   1054,   1054,   1307,
    465,    1307,   732,    829,    112,    1054,   -32768, -2,     -32768, -32768,
    -32768, 136,    -32768, 602,    594,    619,    460,    68,     467,    -32768,
    68,     -32768, -32768, -32768, 617,    -32768, 626,    -32768, 319,    319,
    319,    319,    317,    566,    431,    616,    226,    -32768, 628,    -32768,
    473,    -32768, 594,    486,    634,    643,    652,    575,    497,    255,
    -32768, -32768, -32768, 1561,   258,    -15,    416,    445,    -32768, -32768,
    534,    278,    -32768, -32768, 502,    314,    586,    503,    829,    829,
    29,     -32768, -32768, 93,     505,    94,     829,    829,    603,    330,
    209,    -32768, 509,    34,     285,    511,    295,    23,     242,    83,
    -32768, 298,    513,    514,    -32768, -32768, 1463,   -32768, 517,    -32768,
    672,    597,    666,    -32768, -32768, 1054,   1307,   521,    1307,   376,
    376,    -32768, -32768, -32768, 455,    678,    1228,   -32768, 678,    -32768,
    -32768, -32768, 926,    1151,   -32768, 516,    -32768, 585,    -32768, 57,
    -32768, 522,    319,    -32768, 829,    302,    314,    -32768, -32768, 589,
    342,    589,    681,    -32768, 323,    -32768, -32768, -32768, 563,    -32768,
    -32768, -32768, 1463,   -32768, 594,    314,    448,    331,    829,    530,
    532,    535,    537,    319,    621,    448,    581,    -32768, 40,     -32768,
    685,    448,    -32768, 445,    -32768, 319,    1307,   -32768, 445,    -32768,
    36,     38,     -32768, -32768, 829,    416,    156,    -32768, 1561,   -32768,
    -32768, -32768, -32768, 599,    -32768, -32768, 694,    829,    -32768, 1054,
    -32768, 599,    -32768, -32768, -32768, 317,    -32768, -32768, 669,    625,
    506,    678,    1228,   -32768, 678,    1054,   1307,   -32768, 337,    -32768,
    -32768, -32768, 516,    -32768, -32768, -32768, -32768, 416,    -32768, -32768,
    50,     -32768, -32768, -32768, -32768, -32768, 319,    319,    -32768, -32768,
    -32768, 343,    -32768, 46,     319,    319,    319,    319,    346,    701,
    -32768, 705,    611,    551,    1384,   317,    -32768, 708,    314,    348,
    447,    -32768, 357,    447,    -32768, -32768, 416,    829,    211,    552,
    555,    32,     252,    558,    -32768, 829,    829,    -32768, 1054,   -32768,
    380,    -32768, 516,    -32768, -32768, 1307,   -32768, -32768, -32768, -32768,
    314,    -32768, 382,    385,    559,    561,    -32768, 314,    564,    170,
    829,    -32768, -32768, -32768, 570,    -32768, -32768, 314,    -32768, 416,
    -32768, -32768, -32768, -32768, 829,    -32768, -32768, 569,    416,    416,
    516,    -32768, -32768, -32768, 592,    -32768, -32768, 601,    -32768, 722,
    731,    -32768, 48,     319,    -32768, 53,     829,    317,    317,    576,
    -32768, -32768, 577,    -32768, 416,    580,    582,    -32768, -32768, 319,
    319,    389,    584,    -32768, -32768, 740,    -32768};

static const short yypgoto[] = {
    -32768, 726,    -32768, -32768, -32768, -32768, -32768, -32768, -224,   263,
    259,    -32768, -32768, -32768, -32768, -32768, 281,    -32768, -32768, -32768,
    -32768, 510,    -32768, -32768, -32768, -32768, 383,    -32768, -32768, -32768,
    -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768, -152,
    257,    -347,   166,    -32768, -32768, -489,   -156,   -32768, -32768, -32768,
    -32768, 284,    408,    291,    -32768, -32768, -32768, -32768, -32768, 293,
    -32768, -232,   -32768, -32768, -32768, 739,    -89,    -294,   -32768, -32768,
    -32768, 225,    -32768, -32768, -32768, -32768, -164,   -32768, -32768, -32768,
    -32768, -332,   -32768, -32768, -306,   -32768, -32768, -32768, -32768, -215,
    -32768, -32768, -32768, -32768, -32768, -147,   432,    -32768, -301,   -32768,
    -193,   -32768, 483,    -3,     -32768, -51,    751,    49,     -157,   387,
    762,    624,    618,    498,    -161,   -274,   -369,   480,    -32768, -210,
    -32768};

#define YYLAST 1690

static const short yytable[] = {
    64,  279,  285, 281, 578, 415, 288,  288, 411, 121, 311,  475, 476, 106,  498, 500,
    61,  -334, 343, 129, 287, 289, 62,   414, 236, 136, -331, 336, 445, 384,  448, 334,
    421, 422,  423, 317, 318, 72,  309,  317, 318, 317, 318,  581, 317, 318,  317, 318,
    317, 318,  317, 318, 102, 393, 582,  310, 116, 93,  317,  318, 317, 318,  93,  461,
    127, 317,  318, 392, 401, 131, 160,  148, 149, 350, 323,  324, 325, 326,  327, 361,
    96,  583,  235, 431, 402, 622, 623,  151, 357, 240, 103,  386, 369, 155,  156, 158,
    159, 311,  584, 467, 323, 324, 325,  326, 327, 335, 317,  318, 233, 463,  323, 324,
    325, 326,  327, 447, 550, 454, 128,  400, 465, 616, 298,  152, 135, 288,  457, 405,
    468, 409,  241, 260, 413, 288, 261,  541, 97,  544, 417,  403, 617, 406,  288, 288,
    501, 549,  351, 410, 487, 598, 477,  478, 415, 432, 433,  152, 418, 419,  415, 98,
    362, 460,  288, 288, 288, 288, 288,  288, 317, 318, 585,  288, 63,  -334, 288, 105,
    439, 440,  441, 442, 443, 444, -331, 337, 527, 455, 245,  424, 459, 512,  681, 394,
    643, 644,  521, 14,  594, 242, 595,  153, 592, 269, 337,  358, 113, 549,  621, 65,
    671, 609,  458, 73,  611, 673, 246,  514, 435, 18,  58,   618, 554, 99,   543, 407,
    338, 161,  323, 324, 325, 326, 327,  107, 145, 534, 510,  511, 610, 231,  551, 507,
    352, 549,  515, 516, 529, 613, 370,  140, 415, 599, 141,  385, 343, 425,  353, 66,
    67,  603,  154, 323, 324, 325, 326,  327, 146, 110, 436,  437, 685, 323,  324, 325,
    326, 327,  288, 438, 14,  597, 74,   142, 78,  79,  660,  392, 154, 59,   255, 143,
    540, 568,  111, 288, 288, 1,   75,   591, 2,   462, 18,   112, 557, 80,   81,  115,
    76,  403,  552, 77,  316, 350, 559,  60,  661, 481, 652,  3,   4,   317,  318, 113,
    61,  270,  339, 68,  573, 82,  62,   5,   340, 570, 283,  122, 6,   7,    571, 319,
    83,  284,  69,  78,  79,  84,  317,  318, 125, 346, 590,  8,   70,  589,  596, 593,
    347, 9,    388, 134, 123, 619, 137,  479, 518, 389, 85,   601, 392, 392,  624, 625,
    288, 519,  132, 640, 10,  337, 320,  321, 322, 323, 324,  325, 326, 327,  602, 249,
    288, 271,  483, 484, 272, 254, 124,  633, 130, 328, 11,   325, 326, 327,  612, 138,
    12,  13,   528, 86,  139, 14,  187,  147, 189, 15,  227,  16,  645, 17,   87,  494,
    495, 273,  499, 484, 88,  193, 194,  232, 89,  274, 144,  18,  317, 318,  90,  91,
    636, 639,  117, 118, 119, 150, 504,  505, 120, 648, 649,  329, 330, 522,  523, 288,
    263, 264,  119, 339, 331, 672, 120,  525, 526, 340, 530,  531, 228, 650,  558, 484,
    653, 229,  662, 237, 545, 392, 682,  658, 323, 324, 325,  326, 327, 234,  239, 162,
    665, 664,  163, 565, 566, 247, 164,  165, 248, 373, 166,  572, 484, 167,  168, 397,
    251, 614,  615, 250, 398, 258, 674,  620, 495, 170, 626,  505, 637, 495,  171, 172,
    252, 173,  174, 175, 176, 638, 615,  608, 425, 253, 374,  323, 324, 325,  326, 327,
    255, 177,  178, 259, 179, 323, 324,  325, 326, 327, 399,  180, 651, 615,  654, 505,
    181, 655,  505, 262, 182, 683, 505,  277, 291, 280, 183,  290, 293, 294,  184, 295,
    375, 296,  185, 314, 297, 299, 301,  186, 302, 303, 187,  188, 189, 190,  191, 192,
    304, 305,  306, 307, 308, 345, 634,  333, 344, 193, 194,  376, 377, 354,  378, 356,
    195, 357,  364, 363, 196, 365, 197,  198, 199, 371, 372,  368, 395, 200,  335, 162,
    387, 411,  163, 14,  420, 434, 164,  165, 427, 464, 166,  426, 201, 167,  168, 169,
    446, 466,  284, 472, 470, 473, 241,  482, 486, 170, 485,  202, 492, 489,  171, 172,
    203, 173,  174, 175, 176, 488, 490,  320, 321, 322, 323,  324, 325, 326,  327, 491,
    493, 177,  178, 503, 179, 506, 508,  509, 675, 676, 328,  180, 513, 520,  517, 524,
    181, 532,  533, 536, 182, 538, 535,  318, 542, 546, 183,  553, 564, 555,  184, 560,
    567, 574,  185, 575, 298, 588, 576,  186, 577, 579, 187,  188, 189, 190,  191, 192,
    600, 605,  606, 627, 628, 629, 630,  581, 641, 193, 194,  642, 329, 330,  646, 656,
    195, 657,  667, 659, 196, 331, 197,  198, 199, 663, 666,  668, 669, 200,  670, 162,
    677, 678,  163, 679, 686, 680, 164,  165, 684, 109, 480,  569, 201, 167,  168, 360,
    580, 635,  562, 556, 449, 108, 471,  450, 604, 170, 563,  202, 100, 456,  171, 172,
    203, 173,  174, 175, 176, 404, 502,  101, 80,  81,  276,  282, 396, 416,  0,   0,
    0,   177,  178, 0,   179, 0,   0,    0,   0,   0,   0,    180, 0,   0,    0,   0,
    82,  0,    0,   0,   182, 0,   0,    0,   0,   0,   183,  83,  0,   0,    184, 0,
    84,  0,    185, 0,   0,   0,   0,    186, 0,   0,   187,  188, 189, 190,  191, 192,
    162, 0,    0,   163, 0,   85,  0,    164, 165, 193, 194,  166, 0,   0,    167, 168,
    195, 451,  0,   0,   196, 0,   197,  198, 199, 0,   170,  0,   0,   200,  0,   171,
    172, 0,    173, 174, 175, 176, 0,    0,   0,   0,   0,    0,   201, 0,    86,  0,
    0,   0,    177, 178, 0,   179, 0,    0,   0,   87,  0,    452, 180, 0,    0,   88,
    203, 181,  0,   89,  0,   182, 0,    0,   0,   90,  91,   183, 0,   0,    0,   184,
    0,   0,    0,   185, 0,   0,   0,    0,   186, 0,   0,    187, 188, 189,  190, 191,
    192, 162,  0,   0,   163, 0,   0,    0,   164, 165, 193,  194, 0,   0,    0,   167,
    168, 195,  0,   0,   0,   196, 0,    197, 198, 199, 0,    170, 0,   0,    200, 0,
    171, 172,  0,   173, 174, 175, 176,  0,   0,   0,   0,    0,   0,   201,  0,   0,
    0,   0,    0,   177, 178, 0,   179,  0,   0,   0,   0,    0,   202, 180,  0,   0,
    0,   203,  0,   0,   0,   0,   182,  0,   0,   0,   0,    0,   183, 0,    0,   0,
    184, 0,    0,   0,   185, 0,   0,    0,   0,   186, 0,    0,   187, 188,  189, 190,
    191, 192,  0,   0,   0,   0,   0,    0,   0,   0,   0,    193, 194, 0,    0,   0,
    0,   0,    195, 0,   0,   0,   196,  0,   197, 198, 199,  0,   0,   0,    0,   200,
    0,   162,  0,   0,   163, 0,   0,    14,  164, 165, 0,    0,   0,   0,    201, 167,
    168, 0,    0,   0,   0,   0,   0,    0,   0,   0,   0,    170, 0,   286,  0,   0,
    171, 172,  203, 173, 174, 175, 176,  0,   0,   0,   0,    0,   0,   0,    0,   0,
    0,   0,    0,   177, 178, 0,   179,  0,   0,   0,   0,    0,   0,   180,  0,   0,
    0,   0,    0,   0,   0,   0,   182,  0,   0,   0,   0,    0,   183, 0,    0,   0,
    184, 0,    0,   0,   185, 0,   0,    0,   0,   186, 0,    0,   187, 188,  189, 190,
    191, 192,  162, 0,   0,   163, 0,    0,   0,   164, 165,  193, 194, 0,    0,   0,
    167, 168,  195, 0,   0,   0,   196,  0,   197, 198, 199,  0,   170, 0,    0,   200,
    0,   171,  172, 0,   173, 174, 175,  176, 0,   0,   0,    0,   0,   0,    201, 0,
    0,   0,    0,   0,   177, 178, 0,    179, 0,   0,   0,    0,   0,   286,  180, 0,
    0,   0,    203, 0,   0,   0,   0,    182, 0,   0,   0,    0,   0,   183,  0,   0,
    0,   184,  163, 0,   0,   185, 164,  165, 0,   0,   186,  0,   0,   187,  188, 189,
    190, 191,  192, 0,   0,   0,   0,    0,   0,   170, 0,    0,   193, 194,  171, 172,
    0,   0,    0,   195, 176, 0,   0,    196, 0,   197, 198,  199, 0,   0,    0,   0,
    200, 177,  178, 0,   179, 0,   0,    0,   0,   0,   0,    180, 0,   0,    0,   201,
    0,   0,    0,   0,   182, 0,   0,    0,   0,   0,   183,  0,   0,   0,    452, 0,
    0,   163,  185, 203, 0,   164, 165,  0,   0,   0,   187,  0,   189, 190,  191, 192,
    0,   0,    0,   0,   0,   0,   0,    0,   170, 193, 194,  0,   0,   171,  172, 0,
    195, 0,    0,   176, 196, 0,   197,  198, 199, 0,   0,    0,   0,   200,  0,   0,
    177, 178,  0,   179, 0,   14,  0,    0,   0,   0,   180,  0,   201, 0,    0,   0,
    0,   0,    0,   182, 0,   0,   0,    0,   0,   183, 0,    0,   0,   0,    163, 0,
    203, 185,  164, 165, 0,   0,   0,    0,   0,   187, 0,    189, 190, 191,  192, 0,
    0,   0,    0,   0,   0,   170, 0,    0,   193, 194, 171,  172, 0,   0,    0,   195,
    176, 0,    0,   196, 0,   197, 198,  199, 0,   0,   0,    0,   200, 177,  178, 0,
    179, 0,    0,   0,   0,   0,   0,    180, 0,   0,   0,    201, 0,   0,    0,   0,
    182, 0,    0,   0,   0,   0,   183,  0,   0,   0,   0,    0,   0,   163,  185, 203,
    0,   164,  165, 0,   0,   0,   187,  0,   189, 190, 631,  192, 0,   0,    0,   0,
    0,   0,    0,   0,   170, 193, 194,  0,   0,   171, 172,  0,   195, 0,    0,   176,
    196, 0,    197, 198, 199, 0,   0,    0,   0,   200, 0,    0,   177, 178,  0,   179,
    0,   0,    0,   0,   1,   0,   180,  2,   632, 0,   0,    0,   0,   0,    0,   182,
    0,   0,    0,   0,   0,   183, 0,    0,   0,   0,   3,    4,   203, 185,  0,   0,
    0,   0,    0,   0,   0,   187, 5,    189, 190, 191, 192,  6,   7,   0,    0,   0,
    0,   0,    0,   0,   193, 194, 0,    0,   0,   0,   8,    195, 0,   0,    0,   196,
    9,   197,  198, 199, 0,   0,   0,    0,   200, 0,   0,    171, 172, 0,    0,   0,
    0,   176,  0,   10,  0,   0,   0,    0,   0,   0,   0,    0,   0,   0,    177, 0,
    0,   179,  0,   0,   0,   0,   0,    0,   496, 11,  0,    203, 0,   0,    0,   12,
    13,  497,  0,   0,   14,  0,   0,    183, 15,  0,   16,   0,   17,  0,    0,   185,
    0,   0,    0,   0,   0,   0,   0,    187, 0,   189, 18,   0,   192, 0,    0,   0,
    0,   0,    0,   0,   0,   0,   193,  194, 0,   0,   0,    0,   0,   195,  0,   0,
    0,   196,  0,   197, 198, 199, 0,    0,   0,   0,   200};

static const short yycheck[] = {
    3,   157, 166, 160, 493, 299, 167, 168, 10,  60,  203, 358, 359, 16,  383, 30,  3,
    6,   228, 70,  167, 168, 9,   297, 113, 76,  6,   6,   329, 253, 331, 88,  306, 307,
    308, 12,  13,  76,  202, 12,  13,  12,  13,  3,   12,  13,  12,  13,  12,  13,  12,
    13,  24,  8,   14,  202, 59,  8,   12,  13,  12,  13,  13,  337, 67,  12,  13,  277,
    3,   72,  76,  155, 156, 234, 17,  18,  19,  20,  21,  44,  87,  41,  24,  315, 19,
    574, 575, 125, 44,  23,  62,  255, 248, 96,  97,  98,  99,  290, 58,  31,  17,  18,
    19,  20,  21,  162, 12,  13,  111, 341, 17,  18,  19,  20,  21,  330, 448, 332, 69,
    283, 344, 71,  111, 161, 75,  286, 14,  291, 60,  293, 64,  134, 296, 294, 137, 436,
    118, 438, 302, 286, 90,  66,  303, 304, 159, 446, 235, 294, 372, 518, 360, 361, 446,
    317, 318, 161, 303, 304, 452, 118, 125, 163, 323, 324, 325, 326, 327, 328, 12,  13,
    130, 332, 159, 162, 335, 48,  323, 324, 325, 326, 327, 328, 162, 162, 161, 332, 131,
    160, 335, 160, 679, 146, 160, 161, 160, 137, 160, 131, 160, 102, 506, 152, 162, 159,
    127, 506, 160, 3,   160, 541, 98,  51,  544, 160, 159, 121, 33,  159, 51,  566, 163,
    118, 437, 148, 227, 102, 17,  18,  19,  20,  21,  43,  115, 426, 398, 399, 542, 160,
    453, 395, 10,  542, 406, 407, 161, 546, 249, 115, 542, 523, 118, 254, 462, 160, 24,
    51,  52,  531, 161, 17,  18,  19,  20,  21,  147, 158, 83,  84,  0,   17,  18,  19,
    20,  21,  435, 92,  137, 121, 118, 147, 3,   4,   112, 493, 161, 118, 150, 155, 435,
    482, 118, 452, 453, 25,  134, 505, 28,  161, 159, 107, 464, 24,  25,  3,   144, 452,
    453, 147, 3,   470, 466, 144, 142, 364, 615, 47,  48,  12,  13,  127, 3,   51,  3,
    119, 488, 48,  9,   59,  9,   485, 159, 7,   64,  65,  486, 30,  59,  166, 134, 3,
    4,   64,  12,  13,  82,  3,   503, 79,  144, 501, 514, 508, 10,  85,  3,   74,  76,
    567, 77,  362, 30,  10,  85,  527, 574, 575, 576, 577, 529, 160, 82,  160, 104, 162,
    14,  15,  16,  17,  18,  19,  20,  21,  529, 124, 545, 115, 160, 161, 118, 130, 118,
    584, 118, 33,  126, 19,  20,  21,  545, 113, 132, 133, 160, 126, 115, 137, 94,  147,
    96,  141, 118, 143, 160, 145, 137, 160, 161, 147, 160, 161, 143, 109, 110, 158, 147,
    155, 155, 159, 12,  13,  153, 154, 588, 597, 3,   4,   5,   155, 160, 161, 9,   605,
    606, 83,  84,  160, 161, 608, 3,   4,   5,   3,   92,  663, 9,   160, 161, 9,   160,
    161, 138, 608, 160, 161, 620, 151, 630, 93,  13,  679, 680, 627, 17,  18,  19,  20,
    21,  36,  131, 3,   644, 637, 6,   160, 161, 160, 10,  11,  6,   41,  14,  160, 161,
    17,  18,  19,  3,   160, 161, 14,  24,  68,  666, 160, 161, 29,  160, 161, 160, 161,
    34,  35,  159, 37,  38,  39,  40,  160, 161, 13,  160, 159, 74,  17,  18,  19,  20,
    21,  150, 53,  54,  3,   56,  17,  18,  19,  20,  21,  62,  63,  160, 161, 160, 161,
    68,  160, 161, 157, 72,  160, 161, 159, 148, 76,  78,  162, 159, 159, 82,  159, 112,
    159, 86,  76,  159, 159, 159, 91,  159, 159, 94,  95,  96,  97,  98,  99,  159, 159,
    159, 159, 159, 125, 585, 161, 159, 109, 110, 139, 140, 101, 142, 125, 116, 44,  125,
    161, 120, 3,   122, 123, 124, 68,  159, 125, 6,   129, 162, 3,   161, 10,  6,   137,
    160, 3,   10,  11,  165, 15,  14,  161, 144, 17,  18,  19,  159, 6,   166, 10,  161,
    3,   64,  15,  159, 29,  6,   159, 61,  3,   34,  35,  164, 37,  38,  39,  40,  159,
    3,   14,  15,  16,  17,  18,  19,  20,  21,  3,   159, 53,  54,  125, 56,  159, 76,
    160, 667, 668, 33,  63,  163, 160, 67,  160, 68,  160, 160, 3,   72,  80,  161, 13,
    159, 3,   78,  98,  3,   163, 82,  98,  125, 159, 86,  159, 111, 8,   159, 91,  159,
    76,  94,  95,  96,  97,  98,  99,  10,  36,  81,  6,   3,   98,  159, 3,   160, 109,
    110, 160, 83,  84,  160, 160, 116, 160, 130, 159, 120, 92,  122, 123, 124, 159, 161,
    130, 10,  129, 3,   3,   160, 160, 6,   159, 0,   159, 10,  11,  160, 19,  363, 484,
    144, 17,  18,  241, 495, 587, 470, 462, 24,  18,  350, 27,  535, 29,  471, 159, 13,
    333, 34,  35,  164, 37,  38,  39,  40,  290, 387, 13,  24,  25,  154, 161, 282, 301,
    -1,  -1,  -1,  53,  54,  -1,  56,  -1,  -1,  -1,  -1,  -1,  -1,  63,  -1,  -1,  -1,
    -1,  48,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  59,  -1,  -1,  82,  -1,
    64,  -1,  86,  -1,  -1,  -1,  -1,  91,  -1,  -1,  94,  95,  96,  97,  98,  99,  3,
    -1,  -1,  6,   -1,  85,  -1,  10,  11,  109, 110, 14,  -1,  -1,  17,  18,  116, 117,
    -1,  -1,  120, -1,  122, 123, 124, -1,  29,  -1,  -1,  129, -1,  34,  35,  -1,  37,
    38,  39,  40,  -1,  -1,  -1,  -1,  -1,  -1,  144, -1,  126, -1,  -1,  -1,  53,  54,
    -1,  56,  -1,  -1,  -1,  137, -1,  159, 63,  -1,  -1,  143, 164, 68,  -1,  147, -1,
    72,  -1,  -1,  -1,  153, 154, 78,  -1,  -1,  -1,  82,  -1,  -1,  -1,  86,  -1,  -1,
    -1,  -1,  91,  -1,  -1,  94,  95,  96,  97,  98,  99,  3,   -1,  -1,  6,   -1,  -1,
    -1,  10,  11,  109, 110, -1,  -1,  -1,  17,  18,  116, -1,  -1,  -1,  120, -1,  122,
    123, 124, -1,  29,  -1,  -1,  129, -1,  34,  35,  -1,  37,  38,  39,  40,  -1,  -1,
    -1,  -1,  -1,  -1,  144, -1,  -1,  -1,  -1,  -1,  53,  54,  -1,  56,  -1,  -1,  -1,
    -1,  -1,  159, 63,  -1,  -1,  -1,  164, -1,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,
    -1,  78,  -1,  -1,  -1,  82,  -1,  -1,  -1,  86,  -1,  -1,  -1,  -1,  91,  -1,  -1,
    94,  95,  96,  97,  98,  99,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  109, 110,
    -1,  -1,  -1,  -1,  -1,  116, -1,  -1,  -1,  120, -1,  122, 123, 124, -1,  -1,  -1,
    -1,  129, -1,  3,   -1,  -1,  6,   -1,  -1,  137, 10,  11,  -1,  -1,  -1,  -1,  144,
    17,  18,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  29,  -1,  159, -1,  -1,
    34,  35,  164, 37,  38,  39,  40,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    -1,  -1,  53,  54,  -1,  56,  -1,  -1,  -1,  -1,  -1,  -1,  63,  -1,  -1,  -1,  -1,
    -1,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,  -1,  82,  -1,  -1,
    -1,  86,  -1,  -1,  -1,  -1,  91,  -1,  -1,  94,  95,  96,  97,  98,  99,  3,   -1,
    -1,  6,   -1,  -1,  -1,  10,  11,  109, 110, -1,  -1,  -1,  17,  18,  116, -1,  -1,
    -1,  120, -1,  122, 123, 124, -1,  29,  -1,  -1,  129, -1,  34,  35,  -1,  37,  38,
    39,  40,  -1,  -1,  -1,  -1,  -1,  -1,  144, -1,  -1,  -1,  -1,  -1,  53,  54,  -1,
    56,  -1,  -1,  -1,  -1,  -1,  159, 63,  -1,  -1,  -1,  164, -1,  -1,  -1,  -1,  72,
    -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,  -1,  82,  6,   -1,  -1,  86,  10,  11,  -1,
    -1,  91,  -1,  -1,  94,  95,  96,  97,  98,  99,  -1,  -1,  -1,  -1,  -1,  -1,  29,
    -1,  -1,  109, 110, 34,  35,  -1,  -1,  -1,  116, 40,  -1,  -1,  120, -1,  122, 123,
    124, -1,  -1,  -1,  -1,  129, 53,  54,  -1,  56,  -1,  -1,  -1,  -1,  -1,  -1,  63,
    -1,  -1,  -1,  144, -1,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,
    -1,  159, -1,  -1,  6,   86,  164, -1,  10,  11,  -1,  -1,  -1,  94,  -1,  96,  97,
    98,  99,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  29,  109, 110, -1,  -1,  34,  35,
    -1,  116, -1,  -1,  40,  120, -1,  122, 123, 124, -1,  -1,  -1,  -1,  129, -1,  -1,
    53,  54,  -1,  56,  -1,  137, -1,  -1,  -1,  -1,  63,  -1,  144, -1,  -1,  -1,  -1,
    -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,  -1,  -1,  6,   -1,  164, 86,
    10,  11,  -1,  -1,  -1,  -1,  -1,  94,  -1,  96,  97,  98,  99,  -1,  -1,  -1,  -1,
    -1,  -1,  29,  -1,  -1,  109, 110, 34,  35,  -1,  -1,  -1,  116, 40,  -1,  -1,  120,
    -1,  122, 123, 124, -1,  -1,  -1,  -1,  129, 53,  54,  -1,  56,  -1,  -1,  -1,  -1,
    -1,  -1,  63,  -1,  -1,  -1,  144, -1,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,
    78,  -1,  -1,  -1,  -1,  -1,  -1,  6,   86,  164, -1,  10,  11,  -1,  -1,  -1,  94,
    -1,  96,  97,  98,  99,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  29,  109, 110, -1,
    -1,  34,  35,  -1,  116, -1,  -1,  40,  120, -1,  122, 123, 124, -1,  -1,  -1,  -1,
    129, -1,  -1,  53,  54,  -1,  56,  -1,  -1,  -1,  -1,  25,  -1,  63,  28,  144, -1,
    -1,  -1,  -1,  -1,  -1,  72,  -1,  -1,  -1,  -1,  -1,  78,  -1,  -1,  -1,  -1,  47,
    48,  164, 86,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  94,  59,  96,  97,  98,  99,  64,
    65,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  109, 110, -1,  -1,  -1,  -1,  79,  116, -1,
    -1,  -1,  120, 85,  122, 123, 124, -1,  -1,  -1,  -1,  129, -1,  -1,  34,  35,  -1,
    -1,  -1,  -1,  40,  -1,  104, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  53,
    -1,  -1,  56,  -1,  -1,  -1,  -1,  -1,  -1,  63,  126, -1,  164, -1,  -1,  -1,  132,
    133, 72,  -1,  -1,  137, -1,  -1,  78,  141, -1,  143, -1,  145, -1,  -1,  86,  -1,
    -1,  -1,  -1,  -1,  -1,  -1,  94,  -1,  96,  159, -1,  99,  -1,  -1,  -1,  -1,  -1,
    -1,  -1,  -1,  -1,  109, 110, -1,  -1,  -1,  -1,  -1,  116, -1,  -1,  -1,  120, -1,
    122, 123, 124, -1,  -1,  -1,  -1,  129};

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

  switch (yyn) {
    case 1:

    {
      parseTrees.emplace_front(dynamic_cast<Stmt*>((yyvsp[-1].nodeval)->release()));
      ;
      break;
    }
    case 2:

    {
      parseTrees.emplace_front(dynamic_cast<Stmt*>((yyvsp[-1].nodeval)->release()));
      ;
      break;
    }
    case 3:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 4:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 5:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 6:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 7:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 8:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 9:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 10:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 11:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 12:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 13:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 14:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 15:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 16:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 17:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 18:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 19:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 20:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 21:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 22:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 23:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 24:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 25:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 26:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 27:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 28:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 29:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 30:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 31:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CreateDBStmt((yyvsp[0].stringval)->release(), nullptr, yyvsp[-1].boolval));
      ;
      break;
    }
    case 32:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CreateDBStmt((yyvsp[-3].stringval)->release(),
                           reinterpret_cast<std::list<NameValueAssign*>*>(
                               (yyvsp[-1].listval)->release()),
                           yyvsp[-4].boolval));
      ;
      break;
    }
    case 33:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new DropDBStmt((yyvsp[0].stringval)->release(), yyvsp[-1].boolval));
      ;
      break;
    }
    case 34:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new RenameDBStmt((yyvsp[-3].stringval)->release(),
                                                  (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 35:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CreateUserStmt((yyvsp[-3].stringval)->release(),
                             reinterpret_cast<std::list<NameValueAssign*>*>(
                                 (yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 36:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new DropUserStmt((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 37:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new AlterUserStmt((yyvsp[-3].stringval)->release(),
                            reinterpret_cast<std::list<NameValueAssign*>*>(
                                (yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 38:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new RenameUserStmt((yyvsp[-3].stringval)->release(),
                                                    (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 39:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 40:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 41:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new NameValueAssign((yyvsp[-2].stringval)->release(),
                              dynamic_cast<Literal*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 42:

    {
      yyval.boolval = true;
      ;
      break;
    }
    case 43:

    {
      yyval.boolval = false;
      ;
      break;
    }
    case 44:

    {
      yyval.boolval = true;
      ;
      break;
    }
    case 45:

    {
      yyval.boolval = false;
      ;
      break;
    }
    case 46:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CreateTableAsSelectStmt((yyvsp[-3].stringval)->release(),
                                      (yyvsp[-1].stringval)->release(),
                                      yyvsp[-6].boolval,
                                      yyvsp[-4].boolval,
                                      reinterpret_cast<std::list<NameValueAssign*>*>(
                                          (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 47:

    {
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
    case 48:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CreateTableStmt(
              (yyvsp[-4].stringval)->release(),
              (yyvsp[-7].stringval)->release(),
              reinterpret_cast<std::list<TableElement*>*>((yyvsp[-2].listval)->release()),
              false,
              yyvsp[-5].boolval,
              reinterpret_cast<std::list<NameValueAssign*>*>(
                  (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 49:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CreateDataframeStmt(
              (yyvsp[-6].stringval)->release(),
              reinterpret_cast<std::list<TableElement*>*>((yyvsp[-4].listval)->release()),
              (yyvsp[-1].stringval)->release(),
              reinterpret_cast<std::list<NameValueAssign*>*>(
                  (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 50:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ShowCreateTableStmt((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 51:

    {
      yyval.boolval = true;
      ;
      break;
    }
    case 52:

    {
      yyval.boolval = false;
      ;
      break;
    }
    case 53:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new DropTableStmt((yyvsp[0].stringval)->release(), yyvsp[-1].boolval));
      ;
      break;
    }
    case 54:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new TruncateTableStmt((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 55:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new RenameTableStmt((yyvsp[-3].stringval)->release(),
                                                     (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 56:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new RenameColumnStmt((yyvsp[-5].stringval)->release(),
                                                      (yyvsp[-2].stringval)->release(),
                                                      (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 59:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 60:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 61:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new AddColumnStmt((yyvsp[-3].stringval)->release(),
                            dynamic_cast<ColumnDef*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 62:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new AddColumnStmt(
              (yyvsp[-4].stringval)->release(),
              reinterpret_cast<std::list<ColumnDef*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 63:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new DropColumnStmt((yyvsp[-1].stringval)->release(),
                                                    (yyvsp[0].slistval)->release()));
      ;
      break;
    }
    case 64:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 65:

    {
      (yyvsp[-2].listval)->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 66:

    {
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
    case 67:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CopyTableStmt((yyvsp[-3].stringval)->release(),
                            (yyvsp[-1].stringval)->release(),
                            reinterpret_cast<std::list<NameValueAssign*>*>(
                                (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 68:

    {
      if (!boost::istarts_with(*(yyvsp[-4].stringval)->get(), "SELECT") &&
          !boost::istarts_with(*(yyvsp[-4].stringval)->get(), "WITH")) {
        throw std::runtime_error("SELECT or WITH statement expected");
      }
      *(yyvsp[-4].stringval)->get() += ";";
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ExportQueryStmt((yyvsp[-4].stringval)->release(),
                              (yyvsp[-1].stringval)->release(),
                              reinterpret_cast<std::list<NameValueAssign*>*>(
                                  (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 71:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new DumpTableStmt((yyvsp[-3].stringval)->release(),
                            (yyvsp[-1].stringval)->release(),
                            reinterpret_cast<std::list<NameValueAssign*>*>(
                                (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 72:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new RestoreTableStmt((yyvsp[-3].stringval)->release(),
                               (yyvsp[-1].stringval)->release(),
                               reinterpret_cast<std::list<NameValueAssign*>*>(
                                   (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 73:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new CreateRoleStmt((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 74:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new DropRoleStmt((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 75:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new GrantPrivilegesStmt((yyvsp[-5].slistval)->release(),
                                                         (yyvsp[-3].stringval)->release(),
                                                         (yyvsp[-2].stringval)->release(),
                                                         (yyvsp[0].slistval)->release()));
      ;
      break;
    }
    case 76:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new RevokePrivilegesStmt((yyvsp[-5].slistval)->release(),
                                   (yyvsp[-3].stringval)->release(),
                                   (yyvsp[-2].stringval)->release(),
                                   (yyvsp[0].slistval)->release()));
      ;
      break;
    }
    case 77:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new GrantRoleStmt((yyvsp[-2].slistval)->release(),
                                                   (yyvsp[0].slistval)->release()));
      ;
      break;
    }
    case 78:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new RevokeRoleStmt((yyvsp[-2].slistval)->release(),
                                                    (yyvsp[0].slistval)->release()));
      ;
      break;
    }
    case 79:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OptimizeTableStmt((yyvsp[-1].stringval)->release(),
                                reinterpret_cast<std::list<NameValueAssign*>*>(
                                    (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 80:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ValidateStmt((yyvsp[-1].stringval)->release(),
                           reinterpret_cast<std::list<NameValueAssign*>*>(
                               (yyvsp[0].listval)->release())));
      ;
      break;
    }
    case 81:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 82:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 83:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 84:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 85:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnDef((yyvsp[-2].stringval)->release(),
                        dynamic_cast<SQLType*>((yyvsp[-1].nodeval)->release()),
                        dynamic_cast<CompressDef*>((yyvsp[0].nodeval)->release()),
                        nullptr));
      ;
      break;
    }
    case 86:

    {
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
    case 87:

    {
      if (!boost::iequals(*(yyvsp[-1].stringval)->get(), "encoding"))
        throw std::runtime_error("Invalid identifier " + *(yyvsp[-1].stringval)->get() +
                                 " in column definition.");
      delete (yyvsp[-1].stringval)->release();
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new CompressDef((yyvsp[0].stringval)->release(), 0));
      ;
      break;
    }
    case 88:

    {
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
    case 89:

    {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 90:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new ColumnConstraintDef(true, false, false, nullptr));
      ;
      break;
    }
    case 91:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new ColumnConstraintDef(true, true, false, nullptr));
      ;
      break;
    }
    case 92:

    {
      if (!boost::iequals(*(yyvsp[0].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[0].stringval)->get());
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new ColumnConstraintDef(true, true, true, nullptr));
      ;
      break;
    }
    case 93:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef(false,
                                  false,
                                  false,
                                  dynamic_cast<Literal*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 94:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef(false, false, false, new NullLiteral()));
      ;
      break;
    }
    case 95:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef(false, false, false, new UserLiteral()));
      ;
      break;
    }
    case 96:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 97:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef((yyvsp[0].stringval)->release(), nullptr));
      ;
      break;
    }
    case 98:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnConstraintDef((yyvsp[-3].stringval)->release(),
                                  (yyvsp[-1].stringval)->release()));
      ;
      break;
    }
    case 99:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new UniqueDef(false, (yyvsp[-1].slistval)->release()));
      ;
      break;
    }
    case 100:

    {
      if (!boost::iequals(*(yyvsp[-3].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[-3].stringval)->get());
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new UniqueDef(true, (yyvsp[-1].slistval)->release()));
      ;
      break;
    }
    case 101:

    {
      if (!boost::iequals(*(yyvsp[-5].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[-5].stringval)->get());
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ForeignKeyDef(
              (yyvsp[-3].slistval)->release(), (yyvsp[0].stringval)->release(), nullptr));
      ;
      break;
    }
    case 102:

    {
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
    case 103:

    {
      if (!boost::iequals(*(yyvsp[-3].stringval)->get(), "key"))
        throw std::runtime_error("Syntax error at " + *(yyvsp[-3].stringval)->get());
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new ShardKeyDef(*(yyvsp[-1].stringval)->get()));
      delete (yyvsp[-3].stringval)->release();
      delete (yyvsp[-1].stringval)->release();
      ;
      break;
    }
    case 104:

    {
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
    case 105:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CheckDef(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 106:

    {
      yyval.slistval = TrackedListPtr<std::string>::make(
          lexer.parsed_str_list_tokens_, 1, yyvsp[0].stringval);
      ;
      break;
    }
    case 107:

    {
      yyval.slistval = yyvsp[-2].slistval;
      yyval.slistval->push_back(yyvsp[0].stringval);
      ;
      break;
    }
    case 108:

    {
      yyval.listval = yyvsp[-1].listval;
      ;
      break;
    }
    case 109:

    {
      yyval.listval = TrackedListPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 110:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new DropViewStmt((yyvsp[0].stringval)->release(), yyvsp[-1].boolval));
      ;
      break;
    }
    case 111:

    {
      yyval.slistval = TrackedListPtr<std::string>::makeEmpty();
      ;
      break;
    }
    case 112:

    {
      yyval.slistval = yyvsp[-1].slistval;
      ;
      break;
    }
    case 113:

    {
      yyval.listval = TrackedListPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 114:

    {
      yyval.listval = yyvsp[0].listval;
      ;
      break;
    }
    case 115:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 116:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 117:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OrderSpec(yyvsp[-2].intval, nullptr, yyvsp[-1].boolval, yyvsp[0].boolval));
      ;
      break;
    }
    case 118:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OrderSpec(0,
                        dynamic_cast<ColumnRef*>((yyvsp[-2].nodeval)->release()),
                        yyvsp[-1].boolval,
                        yyvsp[0].boolval));
      ;
      break;
    }
    case 119:

    {
      yyval.boolval = false; /* default is ASC */
      ;
      break;
    }
    case 120:

    {
      yyval.boolval = false;
      ;
      break;
    }
    case 121:

    {
      yyval.boolval = true;
      ;
      break;
    }
    case 122:

    {
      yyval.boolval = false; /* default is NULL LAST */
      ;
      break;
    }
    case 123:

    {
      yyval.boolval = true;
      ;
      break;
    }
    case 124:

    {
      yyval.boolval = false;
      ;
      break;
    }
    case 130:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new DeleteStmt((yyvsp[-1].stringval)->release(),
                         dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 131:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InsertValuesStmt(
              (yyvsp[-5].stringval)->release(),
              (yyvsp[-4].slistval)->release(),
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 132:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InsertIntoTableAsSelectStmt((yyvsp[-2].stringval)->release(),
                                          (yyvsp[0].stringval)->release(),
                                          (yyvsp[-1].slistval)->release()));
      ;
      break;
    }
    case 133:

    {
      yyval.boolval = false;
      ;
      break;
    }
    case 134:

    {
      yyval.boolval = false;
      ;
      break;
    }
    case 135:

    {
      yyval.boolval = true;
      ;
      break;
    }
    case 136:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 137:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 138:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new Assignment((yyvsp[-2].stringval)->release(),
                         dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 139:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new UpdateStmt(
              (yyvsp[-3].stringval)->release(),
              reinterpret_cast<std::list<Assignment*>*>((yyvsp[-1].listval)->release()),
              dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 140:

    {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 141:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 142:

    {
      yyval.intval = yyvsp[0].intval;
      if (yyval.intval <= 0)
        throw std::runtime_error("LIMIT must be positive.");
      ;
      break;
    }
    case 143:

    {
      yyval.intval = 0; /* 0 means ALL */
      ;
      break;
    }
    case 144:

    {
      yyval.intval = 0; /* 0 means ALL */
      ;
      break;
    }
    case 145:

    {
      yyval.intval = yyvsp[0].intval;
      ;
      break;
    }
    case 146:

    {
      if (!boost::iequals(*(yyvsp[0].stringval)->get(), "row") &&
          !boost::iequals(*(yyvsp[0].stringval)->get(), "rows"))
        throw std::runtime_error("Invalid word in OFFSET clause " +
                                 *(yyvsp[0].stringval)->get());
      delete (yyvsp[0].stringval)->release();
      yyval.intval = yyvsp[-1].intval;
      ;
      break;
    }
    case 147:

    {
      yyval.intval = 0;
      ;
      break;
    }
    case 148:

    {
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
    case 149:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 150:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new UnionQuery(false,
                         dynamic_cast<QueryExpr*>((yyvsp[-2].nodeval)->release()),
                         dynamic_cast<QueryExpr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 151:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new UnionQuery(true,
                         dynamic_cast<QueryExpr*>((yyvsp[-3].nodeval)->release()),
                         dynamic_cast<QueryExpr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 152:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 153:

    {
      yyval.nodeval = yyvsp[-1].nodeval;
      ;
      break;
    }
    case 154:

    {
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
    case 155:

    {
      yyval.listval = yyvsp[0].listval;
      ;
      break;
    }
    case 156:

    {
      yyval.listval = TrackedListPtr<Node>::makeEmpty(); /* nullptr means SELECT * */
      ;
      break;
    }
    case 157:

    {
      yyval.listval = yyvsp[0].listval;
      ;
      break;
    }
    case 158:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 159:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 160:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new TableRef((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 161:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new TableRef((yyvsp[-1].stringval)->release(),
                                              (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 162:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 163:

    {
      yyval.listval = TrackedListPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 164:

    {
      yyval.listval = yyvsp[0].listval;
      ;
      break;
    }
    case 165:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 166:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 167:

    {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 168:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 169:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kOR,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 170:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kAND,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 171:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(
              kNOT, dynamic_cast<Expr*>((yyvsp[0].nodeval)->release()), nullptr));
      ;
      break;
    }
    case 172:

    {
      yyval.nodeval = yyvsp[-1].nodeval;
      ;
      break;
    }
    case 173:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 174:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 175:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 176:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 177:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 178:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 179:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 180:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 181:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 182:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(yyvsp[-1].opval,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 183:

    {
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
    case 184:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new BetweenExpr(true,
                          dynamic_cast<Expr*>((yyvsp[-5].nodeval)->release()),
                          dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                          dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 185:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new BetweenExpr(false,
                          dynamic_cast<Expr*>((yyvsp[-4].nodeval)->release()),
                          dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                          dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 186:

    {
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
    case 187:

    {
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
    case 188:

    {
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
    case 189:

    {
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
    case 190:

    {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 191:

    {
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
    case 192:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new IsNullExpr(true, dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release())));
      ;
      break;
    }
    case 193:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new IsNullExpr(false, dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release())));
      ;
      break;
    }
    case 194:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InSubquery(true,
                         dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                         dynamic_cast<SubqueryExpr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 195:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InSubquery(false,
                         dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                         dynamic_cast<SubqueryExpr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 196:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InValues(
              true,
              dynamic_cast<Expr*>((yyvsp[-5].nodeval)->release()),
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 197:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new InValues(
              false,
              dynamic_cast<Expr*>((yyvsp[-4].nodeval)->release()),
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 198:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 199:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 200:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(yyvsp[-2].opval,
                       yyvsp[-1].qualval,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 201:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(yyvsp[-2].opval,
                       yyvsp[-1].qualval,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 202:

    {
      yyval.opval = yyvsp[0].opval;
      ;
      break;
    }
    case 203:

    {
      yyval.opval = yyvsp[0].opval;
      ;
      break;
    }
    case 207:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ExistsExpr(dynamic_cast<QuerySpec*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 208:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SubqueryExpr(dynamic_cast<QuerySpec*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 209:

    {
      yyval.listval = TrackedListPtr<Node>::make(
          lexer.parsed_node_list_tokens_,
          1,
          new ExprPair(dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 210:

    {
      yyval.listval = yyvsp[-4].listval;
      yyval.listval->push_back(
          new ExprPair(dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 211:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 212:

    {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 213:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CaseExpr(
              reinterpret_cast<std::list<ExprPair*>*>((yyvsp[-2].listval)->release()),
              dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 214:

    {
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
    case 215:

    {
      std::list<ExprPair*>* when_then_list = new std::list<ExprPair*>(
          1,
          new ExprPair(dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new CaseExpr(when_then_list, nullptr));
      ;
      break;
    }
    case 216:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CharLengthExpr(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()), true));
      ;
      break;
    }
    case 217:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CharLengthExpr(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()), false));
      ;
      break;
    }
    case 218:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kARRAY_AT,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 219:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kPLUS,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 220:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kMINUS,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 221:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kMULTIPLY,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 222:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kDIVIDE,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 223:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kMODULO,
                       dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[0].nodeval)->release())));
      ;
      break;
    }
    case 224:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(kMODULO,
                       dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 225:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 226:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new OperExpr(
              kUMINUS, dynamic_cast<Expr*>((yyvsp[0].nodeval)->release()), nullptr));
      ;
      break;
    }
    case 227:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 228:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 229:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 230:

    {
      yyval.nodeval = yyvsp[-1].nodeval;
      ;
      break;
    }
    case 231:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CastExpr(dynamic_cast<Expr*>((yyvsp[-3].nodeval)->release()),
                       dynamic_cast<SQLType*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 232:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 233:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 234:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 235:

    {
      throw std::runtime_error("Empty select entry");
      ;
      break;
    }
    case 236:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SelectEntry(dynamic_cast<Expr*>((yyvsp[0].nodeval)->release()), nullptr));
      ;
      break;
    }
    case 237:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SelectEntry(dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release()),
                          (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 238:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SelectEntry(dynamic_cast<Expr*>((yyvsp[-2].nodeval)->release()),
                          (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 239:

    {
      throw std::runtime_error("Empty select entry list");
      ;
      break;
    }
    case 240:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 241:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 242:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 243:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new UserLiteral());
      ;
      break;
    }
    case 244:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new FunctionRef((yyvsp[-3].stringval)->release()));
      ;
      break;
    }
    case 245:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new FunctionRef((yyvsp[-4].stringval)->release(),
                          true,
                          dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 246:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new FunctionRef((yyvsp[-4].stringval)->release(),
                          dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 247:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new FunctionRef((yyvsp[-3].stringval)->release(),
                          dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 248:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new StringLiteral((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 249:

    {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new IntLiteral(yyvsp[0].intval));
      ;
      break;
    }
    case 250:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TimestampLiteral());
      ;
      break;
    }
    case 251:

    {
      delete dynamic_cast<Expr*>((yyvsp[-1].nodeval)->release());
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TimestampLiteral());
      ;
      break;
    }
    case 252:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new FixedPtLiteral((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 253:

    {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new FloatLiteral(yyvsp[0].floatval));
      ;
      break;
    }
    case 254:

    {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new DoubleLiteral(yyvsp[0].doubleval));
      ;
      break;
    }
    case 255:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new CastExpr(new StringLiteral((yyvsp[0].stringval)->release()),
                       dynamic_cast<SQLType*>((yyvsp[-1].nodeval)->release())));
      ;
      break;
    }
    case 256:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ArrayLiteral(
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 257:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ArrayLiteral(
              reinterpret_cast<std::list<Expr*>*>((yyvsp[-1].listval)->release())));
      ;
      break;
    }
    case 258:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new NullLiteral());
      ;
      break;
    }
    case 259:

    {
      yyval.listval =
          TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, yyvsp[0].nodeval);
      ;
      break;
    }
    case 260:

    {
      yyval.listval = yyvsp[-2].listval;
      yyval.listval->push_back(yyvsp[0].nodeval);
      ;
      break;
    }
    case 261:

    {
      yyval.listval = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 0);
      ;
      break;
    }
    case 263:

    {
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
    case 264:

    {
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
    case 265:

    {
      yyval.nodeval = TrackedPtr<Node>::makeEmpty();
      ;
      break;
    }
    case 271:

    {
      yyval.slistval = TrackedListPtr<std::string>::make(
          lexer.parsed_str_list_tokens_, 1, yyvsp[0].stringval);
      ;
      break;
    }
    case 272:

    {
      yyval.slistval = yyvsp[-2].slistval;
      yyval.slistval->push_back(yyvsp[0].stringval);
      ;
      break;
    }
    case 275:

    {
      yyval.slistval = TrackedListPtr<std::string>::make(
          lexer.parsed_str_list_tokens_, 1, yyvsp[0].stringval);
      ;
      break;
    }
    case 276:

    {
      yyval.slistval = yyvsp[-2].slistval;
      yyval.slistval->push_back(yyvsp[0].stringval);
      ;
      break;
    }
    case 279:

    {
      yyval.slistval = TrackedListPtr<std::string>::make(
          lexer.parsed_str_list_tokens_, 1, yyvsp[0].stringval);
      ;
      break;
    }
    case 280:

    {
      yyval.slistval = yyvsp[-2].slistval;
      yyval.slistval->push_back(yyvsp[0].stringval);
      ;
      break;
    }
    case 281:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ALL");
      ;
      break;
    }
    case 282:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ALL");
      ;
      break;
    }
    case 283:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE");
      ;
      break;
    }
    case 284:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "SELECT");
      ;
      break;
    }
    case 285:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "INSERT");
      ;
      break;
    }
    case 286:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "TRUNCATE");
      ;
      break;
    }
    case 287:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "UPDATE");
      ;
      break;
    }
    case 288:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DELETE");
      ;
      break;
    }
    case 289:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ALTER");
      ;
      break;
    }
    case 290:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DROP");
      ;
      break;
    }
    case 291:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "VIEW");
      ;
      break;
    }
    case 292:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "EDIT");
      ;
      break;
    }
    case 293:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ACCESS");
      ;
      break;
    }
    case 294:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "ALTER SERVER");
      ;
      break;
    }
    case 295:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE SERVER");
      ;
      break;
    }
    case 296:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE TABLE");
      ;
      break;
    }
    case 297:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE VIEW");
      ;
      break;
    }
    case 298:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "SELECT VIEW");
      ;
      break;
    }
    case 299:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DROP VIEW");
      ;
      break;
    }
    case 300:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DROP SERVER");
      ;
      break;
    }
    case 301:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "CREATE DASHBOARD");
      ;
      break;
    }
    case 302:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "EDIT DASHBOARD");
      ;
      break;
    }
    case 303:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "VIEW DASHBOARD");
      ;
      break;
    }
    case 304:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DELETE DASHBOARD");
      ;
      break;
    }
    case 305:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "VIEW SQL EDITOR");
      ;
      break;
    }
    case 306:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DATABASE");
      ;
      break;
    }
    case 307:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "TABLE");
      ;
      break;
    }
    case 308:

    {
      yyval.stringval =
          TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "DASHBOARD");
      ;
      break;
    }
    case 309:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "VIEW");
      ;
      break;
    }
    case 310:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_, "SERVER");
      ;
      break;
    }
    case 312:

    {
      yyval.stringval = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_,
                                                      std::to_string(yyvsp[0].intval));
      ;
      break;
    }
    case 313:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_, new ColumnRef((yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 314:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new ColumnRef((yyvsp[-2].stringval)->release(),
                                               (yyvsp[0].stringval)->release()));
      ;
      break;
    }
    case 315:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new ColumnRef((yyvsp[-2].stringval)->release(), nullptr));
      ;
      break;
    }
    case 316:

    {
      if (yyvsp[0].intval < 0)
        throw std::runtime_error("No negative number in type definition.");
      yyval.intval = yyvsp[0].intval;
      ;
      break;
    }
    case 317:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kBIGINT));
      ;
      break;
    }
    case 318:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTEXT));
      ;
      break;
    }
    case 319:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kBOOLEAN));
      ;
      break;
    }
    case 320:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kCHAR));
      ;
      break;
    }
    case 321:

    {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kCHAR, yyvsp[-1].intval));
      ;
      break;
    }
    case 322:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kNUMERIC));
      ;
      break;
    }
    case 323:

    {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kNUMERIC, yyvsp[-1].intval));
      ;
      break;
    }
    case 324:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SQLType(kNUMERIC, yyvsp[-3].intval, yyvsp[-1].intval, false));
      ;
      break;
    }
    case 325:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDECIMAL));
      ;
      break;
    }
    case 326:

    {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kDECIMAL, yyvsp[-1].intval));
      ;
      break;
    }
    case 327:

    {
      yyval.nodeval = TrackedPtr<Node>::make(
          lexer.parsed_node_tokens_,
          new SQLType(kDECIMAL, yyvsp[-3].intval, yyvsp[-1].intval, false));
      ;
      break;
    }
    case 328:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kINT));
      ;
      break;
    }
    case 329:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTINYINT));
      ;
      break;
    }
    case 330:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kSMALLINT));
      ;
      break;
    }
    case 331:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kFLOAT));
      ;
      break;
    }
    case 332:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kFLOAT));
      ;
      break;
    }
    case 333:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDOUBLE));
      ;
      break;
    }
    case 334:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDOUBLE));
      ;
      break;
    }
    case 335:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDATE));
      ;
      break;
    }
    case 336:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTIME));
      ;
      break;
    }
    case 337:

    {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kTIME, yyvsp[-1].intval));
      ;
      break;
    }
    case 338:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTIMESTAMP));
      ;
      break;
    }
    case 339:

    {
      yyval.nodeval = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                             new SQLType(kTIMESTAMP, yyvsp[-1].intval));
      ;
      break;
    }
    case 340:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[0].intval),
                                             static_cast<int>(kGEOMETRY),
                                             0,
                                             false));
      ;
      break;
    }
    case 341:

    {
      yyval.nodeval = yyvsp[0].nodeval;
      ;
      break;
    }
    case 342:

    {
      yyval.nodeval = yyvsp[-2].nodeval;
      if (dynamic_cast<SQLType*>((yyval.nodeval)->get())->get_is_array())
        throw std::runtime_error("array of array not supported.");
      dynamic_cast<SQLType*>((yyval.nodeval)->get())->set_is_array(true);
      ;
      break;
    }
    case 343:

    {
      yyval.nodeval = yyvsp[-3].nodeval;
      if (dynamic_cast<SQLType*>((yyval.nodeval)->get())->get_is_array())
        throw std::runtime_error("array of array not supported.");
      dynamic_cast<SQLType*>((yyval.nodeval)->get())->set_is_array(true);
      dynamic_cast<SQLType*>((yyval.nodeval)->get())->set_array_size(yyvsp[-1].intval);
      ;
      break;
    }
    case 344:

    {
      yyval.intval = kPOINT;
      ;
      break;
    }
    case 345:

    {
      yyval.intval = kLINESTRING;
      ;
      break;
    }
    case 346:

    {
      yyval.intval = kPOLYGON;
      ;
      break;
    }
    case 347:

    {
      yyval.intval = kMULTIPOLYGON;
      ;
      break;
    }
    case 348:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[-1].intval),
                                             static_cast<int>(kGEOGRAPHY),
                                             4326,
                                             false));
      ;
      break;
    }
    case 349:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[-3].intval),
                                             static_cast<int>(kGEOGRAPHY),
                                             yyvsp[-1].intval,
                                             false));
      ;
      break;
    }
    case 350:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[-1].intval),
                                             static_cast<int>(kGEOMETRY),
                                             0,
                                             false));
      ;
      break;
    }
    case 351:

    {
      yyval.nodeval =
          TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                 new SQLType(static_cast<SQLTypes>(yyvsp[-3].intval),
                                             static_cast<int>(kGEOMETRY),
                                             yyvsp[-1].intval,
                                             false));
      ;
      break;
    }
    case 352:

    {
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
    case 353:

    {
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
    case 354:

    {
      yyval.stringval = yyvsp[0].stringval;
      ;
      break;
    }
  }

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
