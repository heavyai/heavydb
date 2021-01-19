#ifndef YY_Parser_h_included
#define YY_Parser_h_included

/* before anything */
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
#endif
#include <stdio.h>

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
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
/* use %define LTYPE */
#endif
#endif
#ifdef YYSTYPE
#ifndef YY_Parser_STYPE
#define YY_Parser_STYPE YYSTYPE
/* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
/* use %define STYPE */
#endif
#endif
#ifdef YYDEBUG
#ifndef YY_Parser_DEBUG
#define YY_Parser_DEBUG YYDEBUG
/* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
/* use %define DEBUG */
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

/* YY_Parser_PURE */
#endif

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

/* TOKEN C */
#ifndef YY_USE_CLASS

#ifndef YY_Parser_PURE
extern YY_Parser_STYPE YY_Parser_LVAL;
#endif

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

/* #defines token */
/* after #define tokens, before const tokens S5*/
#else
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
 public:
  int YY_Parser_DEBUG_FLAG; /*  nonzero means print parse trace	*/
#endif
 public:
  YY_Parser_CLASS(YY_Parser_CONSTRUCTOR_PARAM);

 public:
  YY_Parser_MEMBERS
};
/* other declare folow */
#endif

#if YY_Parser_COMPATIBILITY != 0
/* backward compatibility */
#ifndef YYSTYPE
#define YYSTYPE YY_Parser_STYPE
#endif

#ifndef YYLTYPE
#define YYLTYPE YY_Parser_LTYPE
#endif
#ifndef YYDEBUG
#ifdef YY_Parser_DEBUG
#define YYDEBUG YY_Parser_DEBUG
#endif
#endif

#endif
/* END */

#endif
