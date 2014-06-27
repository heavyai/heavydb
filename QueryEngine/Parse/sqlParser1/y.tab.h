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
     TABLE = 291,
     UNIQUE = 292,
     PRIMARY = 293,
     FOREIGN = 294,
     KEY = 295,
     CHECK = 296,
     REFERENCES = 297,
     DEFAULT = 298,
     DROP = 299,
     DATATYPE = 300,
     DECIMAL = 301,
     SMALLINT = 302,
     NUMERIC = 303,
     CHARACTER = 304,
     INTEGER = 305,
     REAL = 306,
     FLOAT = 307,
     DOUBLE = 308,
     PRECISION = 309,
     VARCHAR = 310,
     AMMSC = 311,
     AVG = 312,
     MAX = 313,
     MIN = 314,
     SUM = 315,
     COUNT = 316,
     ALIAS = 317,
     INTORDER = 318,
     COLORDER = 319,
     AS = 320,
     ORDER = 321,
     ASC = 322,
     DESC = 323,
     LIMIT = 324,
     OFFSET = 325
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
#define TABLE 291
#define UNIQUE 292
#define PRIMARY 293
#define FOREIGN 294
#define KEY 295
#define CHECK 296
#define REFERENCES 297
#define DEFAULT 298
#define DROP 299
#define DATATYPE 300
#define DECIMAL 301
#define SMALLINT 302
#define NUMERIC 303
#define CHARACTER 304
#define INTEGER 305
#define REAL 306
#define FLOAT 307
#define DOUBLE 308
#define PRECISION 309
#define VARCHAR 310
#define AMMSC 311
#define AVG 312
#define MAX 313
#define MIN 314
#define SUM 315
#define COUNT 316
#define ALIAS 317
#define INTORDER 318
#define COLORDER 319
#define AS 320
#define ORDER 321
#define ASC 322
#define DESC 323
#define LIMIT 324
#define OFFSET 325




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 2068 of yacc.c  */
#line 33 "sqlParser1AlteredInput.y"

    char *sValue;                /* string*/
    char *sName;
    char *sParam;
    nodeType *nPtr;             /* node pointer */
    float fValue;                 /* approximate number */
    int iValue;
    char* sSubtok;      /* comparator subtokens */
    int iLength;



/* Line 2068 of yacc.c  */
#line 203 "y.tab.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;


