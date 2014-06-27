/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison interface for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     NAME = 258,
     STRING = 259,
     INTNUM = 260,
     APPROXNUM = 261,
     OR = 262,
     AND = 263,
     NOT = 264,
     COMPARISON = 265,
     UMINUS = 266,
     ALL = 267,
     BETWEEN = 268,
     BY = 269,
     DISTINCT = 270,
     FROM = 271,
     GROUP = 272,
     HAVING = 273,
     SELECT = 274,
     USER = 275,
     WHERE = 276,
     WITH = 277,
     EMPTY = 278,
     SELALL = 279,
     DOT = 280,
     UPDATE = 281,
     SET = 282,
     CURRENT = 283,
     OF = 284,
     NULLX = 285,
     ASSIGN = 286,
     INSERT = 287,
     INTO = 288,
     VALUES = 289,
     CREATE = 290,
     UNIQUE = 291,
     PRIMARY = 292,
     FOREIGN = 293,
     KEY = 294,
     CHECK = 295,
     REFERENCES = 296,
     DEFAULT = 297,
     DROP = 298,
     DATATYPE = 299,
     DECIMAL = 300,
     SMALLINT = 301,
     NUMERIC = 302,
     CHARACTER = 303,
     INTEGER = 304,
     REAL = 305,
     FLOAT = 306,
     DOUBLE = 307,
     PRECISION = 308,
     VARCHAR = 309,
     AVG = 310,
     MAX = 311,
     MIN = 312,
     SUM = 313,
     COUNT = 314,
     ALIAS = 315,
     INTORDER = 316,
     COLORDER = 317,
     AS = 318,
     ORDER = 319,
     ASC = 320,
     DESC = 321,
     LIMIT = 322,
     OFFSET = 323,
     DOTNAME = 324,
     SQL = 325,
     MANIPULATIVE_STATEMENT = 326,
     SELECT_STATEMENT = 327,
     SELECTION = 328,
     TABLE_EXP = 329,
     OPT_ALL_DISTINCT = 330,
     FROM_CLAUSE = 331,
     OPT_WHERE_CLAUSE = 332,
     OPT_HAVING_CLAUSE = 333,
     COLUMN_REF = 334,
     TABLE_REF_COMMALIST = 335,
     SCALAR_EXP_COMMALIST = 336,
     TABLE_REF = 337,
     TABLE = 338,
     SEARCH_CONDITION = 339,
     PREDICATE = 340,
     COMPARISON_PREDICATE = 341,
     BETWEEN_PREDICATE = 342,
     SCALAR_EXP = 343,
     LITERAL = 344,
     ATOM = 345,
     UPDATE_STATEMENT_POSITIONED = 346,
     UPDATE_STATEMENT_SEARCHED = 347,
     ASSIGNMENT = 348,
     ASSIGNMENT_COMMALIST = 349,
     COLUMN = 350,
     CURSOR = 351,
     INSERT_STATEMENT = 352,
     COLUMN_COMMALIST = 353,
     OPT_COLUMN_COMMALIST = 354,
     VALUES_OR_QUERY_SPEC = 355,
     INSERT_ATOM_COMMALIST = 356,
     INSERT_ATOM = 357,
     QUERY_SPEC = 358,
     SCHEMA = 359,
     BASE_TABLE_DEF = 360,
     BASE_TABLE_ELEMENT_COMMALIST = 361,
     BASE_TABLE_ELEMENT = 362,
     COLUMN_DEF = 363,
     TABLE_CONSTRAINT_DEF = 364,
     COLUMN_DEF_OPT = 365,
     COLUMN_DEF_OPT_LIST = 366,
     DATA_TYPE = 367,
     FUNCTION_REF = 368,
     AMMSC = 369,
     OPT_GROUP_BY_CLAUSE = 370,
     COLUMN_REF_COMMALIST = 371,
     OPT_ASC_DESC = 372,
     OPT_ORDER_BY_CLAUSE = 373,
     ORDERING_SPEC = 374,
     ORDERING_SPEC_COMMALIST = 375,
     OPT_LIMIT_CLAUSE = 376,
     PRODUCT = 377,
     PROJECT = 378
   };
#endif
/* Tokens.  */
#define NAME 258
#define STRING 259
#define INTNUM 260
#define APPROXNUM 261
#define OR 262
#define AND 263
#define NOT 264
#define COMPARISON 265
#define UMINUS 266
#define ALL 267
#define BETWEEN 268
#define BY 269
#define DISTINCT 270
#define FROM 271
#define GROUP 272
#define HAVING 273
#define SELECT 274
#define USER 275
#define WHERE 276
#define WITH 277
#define EMPTY 278
#define SELALL 279
#define DOT 280
#define UPDATE 281
#define SET 282
#define CURRENT 283
#define OF 284
#define NULLX 285
#define ASSIGN 286
#define INSERT 287
#define INTO 288
#define VALUES 289
#define CREATE 290
#define UNIQUE 291
#define PRIMARY 292
#define FOREIGN 293
#define KEY 294
#define CHECK 295
#define REFERENCES 296
#define DEFAULT 297
#define DROP 298
#define DATATYPE 299
#define DECIMAL 300
#define SMALLINT 301
#define NUMERIC 302
#define CHARACTER 303
#define INTEGER 304
#define REAL 305
#define FLOAT 306
#define DOUBLE 307
#define PRECISION 308
#define VARCHAR 309
#define AVG 310
#define MAX 311
#define MIN 312
#define SUM 313
#define COUNT 314
#define ALIAS 315
#define INTORDER 316
#define COLORDER 317
#define AS 318
#define ORDER 319
#define ASC 320
#define DESC 321
#define LIMIT 322
#define OFFSET 323
#define DOTNAME 324
#define SQL 325
#define MANIPULATIVE_STATEMENT 326
#define SELECT_STATEMENT 327
#define SELECTION 328
#define TABLE_EXP 329
#define OPT_ALL_DISTINCT 330
#define FROM_CLAUSE 331
#define OPT_WHERE_CLAUSE 332
#define OPT_HAVING_CLAUSE 333
#define COLUMN_REF 334
#define TABLE_REF_COMMALIST 335
#define SCALAR_EXP_COMMALIST 336
#define TABLE_REF 337
#define TABLE 338
#define SEARCH_CONDITION 339
#define PREDICATE 340
#define COMPARISON_PREDICATE 341
#define BETWEEN_PREDICATE 342
#define SCALAR_EXP 343
#define LITERAL 344
#define ATOM 345
#define UPDATE_STATEMENT_POSITIONED 346
#define UPDATE_STATEMENT_SEARCHED 347
#define ASSIGNMENT 348
#define ASSIGNMENT_COMMALIST 349
#define COLUMN 350
#define CURSOR 351
#define INSERT_STATEMENT 352
#define COLUMN_COMMALIST 353
#define OPT_COLUMN_COMMALIST 354
#define VALUES_OR_QUERY_SPEC 355
#define INSERT_ATOM_COMMALIST 356
#define INSERT_ATOM 357
#define QUERY_SPEC 358
#define SCHEMA 359
#define BASE_TABLE_DEF 360
#define BASE_TABLE_ELEMENT_COMMALIST 361
#define BASE_TABLE_ELEMENT 362
#define COLUMN_DEF 363
#define TABLE_CONSTRAINT_DEF 364
#define COLUMN_DEF_OPT 365
#define COLUMN_DEF_OPT_LIST 366
#define DATA_TYPE 367
#define FUNCTION_REF 368
#define AMMSC 369
#define OPT_GROUP_BY_CLAUSE 370
#define COLUMN_REF_COMMALIST 371
#define OPT_ASC_DESC 372
#define OPT_ORDER_BY_CLAUSE 373
#define ORDERING_SPEC 374
#define ORDERING_SPEC_COMMALIST 375
#define OPT_LIMIT_CLAUSE 376
#define PRODUCT 377
#define PROJECT 378




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 2068 of yacc.c  */
#line 24 "sqlParser1T.y"

    char *sValue;                /* string*/
    char *sName;
    char *sParam;
    nodeType *nPtr;             /* node pointer */
    double dValue;                 /* approximate number */
    int iValue;
    char* sSubtok;      /* comparator subtokens */
    int iLength;



/* Line 2068 of yacc.c  */
#line 309 "y.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


