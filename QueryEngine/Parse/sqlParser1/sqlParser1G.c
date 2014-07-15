/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Copy the first part of user declarations.  */

/* Line 268 of yacc.c  */
#line 1 "sqlParser1G.y"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "sqlParser1.h" 

#define YYDEBUG 1

#ifdef DEBUG
#define TRACE printf("reduce at line %d\n", __LINE__);
#else
#define TRACE
#endif

/* prototypes */
nodeType *opr(int oper, int nops, ...);
nodeType *id(char *s);
nodeType *id2(char *s);
nodeType *text(char *s);
nodeType *comp(char *s);
nodeType *compAssgn(char *s);
nodeType *con(float value);
void freeNode(nodeType *p);
int ex(nodeType *p);
int yylex(void);

int yyerror(const char *s);


/* Line 268 of yacc.c  */
#line 102 "sqlParser1G.c"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 1
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


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
     SQL = 324,
     MANIPULATIVE_STATEMENT = 325,
     SELECT_STATEMENT = 326,
     SELECTION = 327,
     TABLE_EXP = 328,
     OPT_ALL_DISTINCT = 329,
     FROM_CLAUSE = 330,
     OPT_WHERE_CLAUSE = 331,
     OPT_HAVING_CLAUSE = 332,
     COLUMN_REF = 333,
     TABLE_REF_COMMALIST = 334,
     SCALAR_EXP_COMMALIST = 335,
     TABLE_REF = 336,
     TABLE = 337,
     SEARCH_CONDITION = 338,
     PREDICATE = 339,
     COMPARISON_PREDICATE = 340,
     BETWEEN_PREDICATE = 341,
     SCALAR_EXP = 342,
     LITERAL = 343,
     ATOM = 344,
     UPDATE_STATEMENT_POSITIONED = 345,
     UPDATE_STATEMENT_SEARCHED = 346,
     ASSIGNMENT = 347,
     ASSIGNMENT_COMMALIST = 348,
     COLUMN = 349,
     CURSOR = 350,
     INSERT_STATEMENT = 351,
     COLUMN_COMMALIST = 352,
     OPT_COLUMN_COMMALIST = 353,
     VALUES_OR_QUERY_SPEC = 354,
     INSERT_ATOM_COMMALIST = 355,
     INSERT_ATOM = 356,
     QUERY_SPEC = 357,
     SCHEMA = 358,
     BASE_TABLE_DEF = 359,
     BASE_TABLE_ELEMENT_COMMALIST = 360,
     BASE_TABLE_ELEMENT = 361,
     COLUMN_DEF = 362,
     TABLE_CONSTRAINT_DEF = 363,
     COLUMN_DEF_OPT = 364,
     COLUMN_DEF_OPT_LIST = 365,
     DATA_TYPE = 366,
     FUNCTION_REF = 367,
     AMMSC = 368,
     OPT_GROUP_BY_CLAUSE = 369,
     COLUMN_REF_COMMALIST = 370,
     OPT_ASC_DESC = 371,
     OPT_ORDER_BY_CLAUSE = 372,
     ORDERING_SPEC = 373,
     ORDERING_SPEC_COMMALIST = 374,
     OPT_LIMIT_CLAUSE = 375
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
#define SQL 324
#define MANIPULATIVE_STATEMENT 325
#define SELECT_STATEMENT 326
#define SELECTION 327
#define TABLE_EXP 328
#define OPT_ALL_DISTINCT 329
#define FROM_CLAUSE 330
#define OPT_WHERE_CLAUSE 331
#define OPT_HAVING_CLAUSE 332
#define COLUMN_REF 333
#define TABLE_REF_COMMALIST 334
#define SCALAR_EXP_COMMALIST 335
#define TABLE_REF 336
#define TABLE 337
#define SEARCH_CONDITION 338
#define PREDICATE 339
#define COMPARISON_PREDICATE 340
#define BETWEEN_PREDICATE 341
#define SCALAR_EXP 342
#define LITERAL 343
#define ATOM 344
#define UPDATE_STATEMENT_POSITIONED 345
#define UPDATE_STATEMENT_SEARCHED 346
#define ASSIGNMENT 347
#define ASSIGNMENT_COMMALIST 348
#define COLUMN 349
#define CURSOR 350
#define INSERT_STATEMENT 351
#define COLUMN_COMMALIST 352
#define OPT_COLUMN_COMMALIST 353
#define VALUES_OR_QUERY_SPEC 354
#define INSERT_ATOM_COMMALIST 355
#define INSERT_ATOM 356
#define QUERY_SPEC 357
#define SCHEMA 358
#define BASE_TABLE_DEF 359
#define BASE_TABLE_ELEMENT_COMMALIST 360
#define BASE_TABLE_ELEMENT 361
#define COLUMN_DEF 362
#define TABLE_CONSTRAINT_DEF 363
#define COLUMN_DEF_OPT 364
#define COLUMN_DEF_OPT_LIST 365
#define DATA_TYPE 366
#define FUNCTION_REF 367
#define AMMSC 368
#define OPT_GROUP_BY_CLAUSE 369
#define COLUMN_REF_COMMALIST 370
#define OPT_ASC_DESC 371
#define OPT_ORDER_BY_CLAUSE 372
#define ORDERING_SPEC 373
#define ORDERING_SPEC_COMMALIST 374
#define OPT_LIMIT_CLAUSE 375




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 293 of yacc.c  */
#line 31 "sqlParser1G.y"

    char *sValue;                /* string*/
    char *sName;
    char *sParam;
    nodeType *nPtr;             /* node pointer */
    float fValue;                 /* approximate number */
    int iValue;
    char* sSubtok;      /* comparator subtokens */
    int iLength;



/* Line 293 of yacc.c  */
#line 391 "sqlParser1G.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 403 "sqlParser1G.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  24
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   347

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  130
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  55
/* YYNRULES -- Number of rules.  */
#define YYNRULES  146
/* YYNRULES -- Number of states.  */
#define YYNSTATES  276

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   375

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     126,   127,    13,    11,   128,    12,   129,    14,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,   125,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,   102,   103,   104,   105,   106,   107,   108,
     109,   110,   111,   112,   113,   114,   115,   116,   117,   118,
     119,   120,   121,   122,   123,   124
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     8,    12,    13,    17,    19,    21,
      28,    32,    34,    38,    40,    42,    46,    47,    50,    53,
      57,    62,    65,    68,    71,    76,    79,    85,    90,    96,
     104,   115,   120,   122,   126,   127,   131,   133,   137,   140,
     143,   144,   146,   148,   150,   152,   154,   156,   158,   164,
     169,   171,   173,   177,   179,   181,   186,   188,   190,   191,
     200,   201,   203,   207,   211,   215,   221,   226,   228,   230,
     237,   240,   242,   246,   248,   251,   252,   253,   257,   259,
     263,   266,   267,   268,   271,   276,   281,   285,   289,   292,
     296,   298,   300,   302,   306,   313,   319,   321,   325,   329,
     333,   337,   341,   344,   347,   349,   351,   353,   357,   359,
     361,   366,   372,   378,   383,   385,   387,   389,   391,   395,
     399,   401,   406,   408,   413,   415,   420,   427,   429,   434,
     441,   443,   445,   447,   452,   454,   457,   459,   463,   469,
     473,   475,   477,   479,   481,   483,   485
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     131,     0,    -1,   132,    -1,   134,   125,    -1,   132,   134,
     125,    -1,    -1,   126,   143,   127,    -1,   135,    -1,   136,
      -1,    39,    86,   179,   126,   137,   127,    -1,    47,    86,
     179,    -1,   138,    -1,   137,   128,   138,    -1,   139,    -1,
     142,    -1,   182,   180,   140,    -1,    -1,   140,   141,    -1,
       9,    34,    -1,     9,    34,    40,    -1,     9,    34,    41,
      43,    -1,    46,   178,    -1,    46,    34,    -1,    46,    24,
      -1,    44,   126,   170,   127,    -1,    45,   179,    -1,    45,
     179,   126,   143,   127,    -1,    40,   126,   143,   127,    -1,
      41,    43,   126,   143,   127,    -1,    42,    43,   126,   143,
     127,    45,   179,    -1,    42,    43,   126,   143,   127,    45,
     179,   126,   143,   127,    -1,    44,   126,   170,   127,    -1,
     182,    -1,   143,   128,   182,    -1,    -1,    68,    18,   145,
      -1,   146,    -1,   145,   128,   146,    -1,     5,   147,    -1,
     181,   147,    -1,    -1,    69,    -1,    70,    -1,   148,    -1,
     153,    -1,   155,    -1,   158,    -1,   149,    -1,    36,    37,
     179,   133,   150,    -1,    38,   126,   151,   127,    -1,   159,
      -1,   152,    -1,   151,   128,   152,    -1,   176,    -1,    34,
      -1,    23,   154,   160,   161,    -1,    16,    -1,    19,    -1,
      -1,    30,   179,    31,   156,    25,    32,    33,   183,    -1,
      -1,   157,    -1,   156,   128,   157,    -1,   182,    10,   175,
      -1,   182,    10,    34,    -1,    30,   179,    31,   156,   165,
      -1,    23,   154,   160,   161,    -1,   174,    -1,    13,    -1,
     162,   165,   166,   168,   144,   169,    -1,    20,   163,    -1,
     164,    -1,   163,   128,   164,    -1,   179,    -1,    25,   170,
      -1,    -1,    -1,    21,    18,   167,    -1,   181,    -1,   167,
     128,   181,    -1,    22,   170,    -1,    -1,    -1,    71,     5,
      -1,    71,     5,   128,     5,    -1,    71,     5,    72,     5,
      -1,   170,     7,   170,    -1,   170,     8,   170,    -1,     9,
     170,    -1,   126,   170,   127,    -1,   171,    -1,   172,    -1,
     173,    -1,   175,    10,   175,    -1,   175,     9,    17,   175,
       8,   175,    -1,   175,    17,   175,     8,   175,    -1,   175,
      -1,   174,   128,   175,    -1,   175,    11,   175,    -1,   175,
      12,   175,    -1,   175,    13,   175,    -1,   175,    14,   175,
      -1,    11,   175,    -1,    12,   175,    -1,   176,    -1,   181,
      -1,   177,    -1,   126,   175,   127,    -1,   178,    -1,    24,
      -1,   184,   126,    13,   127,    -1,   184,   126,    19,   181,
     127,    -1,   184,   126,    16,   175,   127,    -1,   184,   126,
     175,   127,    -1,     4,    -1,     5,    -1,     6,    -1,     3,
      -1,     3,   129,     3,    -1,     3,    67,     3,    -1,    52,
      -1,    52,   126,     5,   127,    -1,    58,    -1,    58,   126,
       5,   127,    -1,    51,    -1,    51,   126,     5,   127,    -1,
      51,   126,     5,   128,     5,   127,    -1,    49,    -1,    49,
     126,     5,   127,    -1,    49,   126,     5,   128,     5,   127,
      -1,    53,    -1,    50,    -1,    55,    -1,    55,   126,     5,
     127,    -1,    54,    -1,    56,    57,    -1,     3,    -1,     3,
     129,     3,    -1,     3,   129,     3,   129,     3,    -1,     3,
      67,     3,    -1,     3,    -1,     3,    -1,    59,    -1,    61,
      -1,    60,    -1,    62,    -1,    63,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   119,   119,   123,   124,   128,   129,   133,   137,   141,
     142,   146,   147,   151,   152,   156,   160,   161,   165,   166,
     167,   168,   169,   170,   171,   172,   173,   177,   178,   179,
     181,   183,   187,   188,   192,   193,   197,   198,   202,   203,
     207,   208,   209,   215,   219,   220,   221,   222,   237,   241,
     242,   246,   247,   251,   252,   256,   263,   264,   265,   269,
     275,   276,   277,   281,   282,   286,   290,   294,   295,   299,
     308,   312,   313,   317,   322,   323,   327,   328,   332,   333,
     337,   338,   342,   343,   344,   345,   349,   350,   351,   352,
     353,   357,   358,   367,   372,   373,   377,   378,   382,   383,
     384,   385,   386,   387,   388,   389,   390,   391,   396,   397,
     408,   409,   410,   411,   415,   416,   417,   422,   423,   424,
     429,   430,   431,   432,   433,   434,   435,   436,   437,   438,
     439,   440,   441,   442,   443,   444,   447,   448,   449,   450,
     454,   457,   461,   462,   463,   464,   465
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "NAME", "STRING", "INTNUM", "\"!\"",
  "OR", "AND", "NOT", "COMPARISON", "'+'", "'-'", "'*'", "'/'", "UMINUS",
  "ALL", "BETWEEN", "BY", "DISTINCT", "FROM", "GROUP", "HAVING", "SELECT",
  "USER", "WHERE", "WITH", "EMPTY", "SELALL", "DOT", "UPDATE", "SET",
  "CURRENT", "OF", "NULLX", "ASSIGN", "INSERT", "INTO", "VALUES", "CREATE",
  "UNIQUE", "PRIMARY", "FOREIGN", "KEY", "CHECK", "REFERENCES", "DEFAULT",
  "DROP", "DATATYPE", "DECIMAL", "SMALLINT", "NUMERIC", "CHARACTER",
  "INTEGER", "REAL", "FLOAT", "DOUBLE", "PRECISION", "VARCHAR", "AVG",
  "MAX", "MIN", "SUM", "COUNT", "ALIAS", "INTORDER", "COLORDER", "AS",
  "ORDER", "ASC", "DESC", "LIMIT", "OFFSET", "SQL",
  "MANIPULATIVE_STATEMENT", "SELECT_STATEMENT", "SELECTION", "TABLE_EXP",
  "OPT_ALL_DISTINCT", "FROM_CLAUSE", "OPT_WHERE_CLAUSE",
  "OPT_HAVING_CLAUSE", "COLUMN_REF", "TABLE_REF_COMMALIST",
  "SCALAR_EXP_COMMALIST", "TABLE_REF", "TABLE", "SEARCH_CONDITION",
  "PREDICATE", "COMPARISON_PREDICATE", "BETWEEN_PREDICATE", "SCALAR_EXP",
  "LITERAL", "ATOM", "UPDATE_STATEMENT_POSITIONED",
  "UPDATE_STATEMENT_SEARCHED", "ASSIGNMENT", "ASSIGNMENT_COMMALIST",
  "COLUMN", "CURSOR", "INSERT_STATEMENT", "COLUMN_COMMALIST",
  "OPT_COLUMN_COMMALIST", "VALUES_OR_QUERY_SPEC", "INSERT_ATOM_COMMALIST",
  "INSERT_ATOM", "QUERY_SPEC", "SCHEMA", "BASE_TABLE_DEF",
  "BASE_TABLE_ELEMENT_COMMALIST", "BASE_TABLE_ELEMENT", "COLUMN_DEF",
  "TABLE_CONSTRAINT_DEF", "COLUMN_DEF_OPT", "COLUMN_DEF_OPT_LIST",
  "DATA_TYPE", "FUNCTION_REF", "AMMSC", "OPT_GROUP_BY_CLAUSE",
  "COLUMN_REF_COMMALIST", "OPT_ASC_DESC", "OPT_ORDER_BY_CLAUSE",
  "ORDERING_SPEC", "ORDERING_SPEC_COMMALIST", "OPT_LIMIT_CLAUSE", "';'",
  "'('", "')'", "','", "'.'", "$accept", "program", "sql_list",
  "opt_column_commalist", "sql", "schema", "base_table_def",
  "base_table_element_commalist", "base_table_element", "column_def",
  "column_def_opt_list", "column_def_opt", "table_constraint_def",
  "column_commalist", "opt_order_by_clause", "ordering_spec_commalist",
  "ordering_spec", "opt_asc_desc", "manipulative_statement",
  "insert_statement", "values_or_query_spec", "insert_atom_commalist",
  "insert_atom", "select_statement", "opt_all_distinct",
  "update_statement_positioned", "assignment_commalist", "assignment",
  "update_statement_searched", "query_spec", "selection", "table_exp",
  "from_clause", "table_ref_commalist", "table_ref", "opt_where_clause",
  "opt_group_by_clause", "column_ref_commalist", "opt_having_clause",
  "opt_limit_clause", "search_condition", "predicate",
  "comparison_predicate", "between_predicate", "scalar_exp_commalist",
  "scalar_exp", "atom", "function_ref", "literal", "table", "data_type",
  "column_ref", "column", "cursor", "ammsc", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,    43,    45,    42,    47,   266,   267,   268,   269,   270,
     271,   272,   273,   274,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   288,   289,   290,
     291,   292,   293,   294,   295,   296,   297,   298,   299,   300,
     301,   302,   303,   304,   305,   306,   307,   308,   309,   310,
     311,   312,   313,   314,   315,   316,   317,   318,   319,   320,
     321,   322,   323,   324,   325,   326,   327,   328,   329,   330,
     331,   332,   333,   334,   335,   336,   337,   338,   339,   340,
     341,   342,   343,   344,   345,   346,   347,   348,   349,   350,
     351,   352,   353,   354,   355,   356,   357,   358,   359,   360,
     361,   362,   363,   364,   365,   366,   367,   368,   369,   370,
     371,   372,   373,   374,   375,    59,    40,    41,    44,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   130,   131,   132,   132,   133,   133,   134,   135,   136,
     136,   137,   137,   138,   138,   139,   140,   140,   141,   141,
     141,   141,   141,   141,   141,   141,   141,   142,   142,   142,
     142,   142,   143,   143,   144,   144,   145,   145,   146,   146,
     147,   147,   147,   134,   148,   148,   148,   148,   149,   150,
     150,   151,   151,   152,   152,   153,   154,   154,   154,   155,
     156,   156,   156,   157,   157,   158,   159,   160,   160,   161,
     162,   163,   163,   164,   165,   165,   166,   166,   167,   167,
     168,   168,   169,   169,   169,   169,   170,   170,   170,   170,
     170,   171,   171,   172,   173,   173,   174,   174,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   176,   176,
     177,   177,   177,   177,   178,   178,   178,   179,   179,   179,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   181,   181,   181,   181,
     182,   183,   184,   184,   184,   184,   184
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     2,     3,     0,     3,     1,     1,     6,
       3,     1,     3,     1,     1,     3,     0,     2,     2,     3,
       4,     2,     2,     2,     4,     2,     5,     4,     5,     7,
      10,     4,     1,     3,     0,     3,     1,     3,     2,     2,
       0,     1,     1,     1,     1,     1,     1,     1,     5,     4,
       1,     1,     3,     1,     1,     4,     1,     1,     0,     8,
       0,     1,     3,     3,     3,     5,     4,     1,     1,     6,
       2,     1,     3,     1,     2,     0,     0,     3,     1,     3,
       2,     0,     0,     2,     4,     4,     3,     3,     2,     3,
       1,     1,     1,     3,     6,     5,     1,     3,     3,     3,
       3,     3,     2,     2,     1,     1,     1,     3,     1,     1,
       4,     5,     5,     4,     1,     1,     1,     1,     3,     3,
       1,     4,     1,     4,     1,     4,     6,     1,     4,     6,
       1,     1,     1,     4,     1,     2,     1,     3,     5,     3,
       1,     1,     1,     1,     1,     1,     1
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,    58,     0,     0,     0,     0,     0,     2,     0,     7,
       8,    43,    47,    44,    45,    46,    56,    57,     0,   117,
       0,     0,     0,     0,     1,     0,     3,   136,   114,   115,
     116,     0,     0,    68,   109,   142,   144,   143,   145,   146,
       0,     0,    67,    96,   104,   106,   108,   105,     0,     0,
       0,    60,     5,     0,    10,     4,     0,     0,   102,   103,
       0,     0,    55,    75,     0,     0,     0,     0,     0,     0,
     119,   118,   140,    75,    61,     0,     0,     0,     0,   139,
     137,   107,    70,    71,    73,     0,    76,    97,    98,    99,
     100,   101,     0,     0,     0,     0,     0,     0,    65,     0,
       0,    32,    58,     0,    48,    50,     0,     0,     0,     0,
       0,    11,    13,    14,     0,     0,     0,     0,     0,    74,
      90,    91,    92,     0,     0,    81,   110,     0,     0,   113,
       0,    62,    64,    63,     6,     0,     0,     0,     0,     0,
       0,     0,     9,     0,   127,   131,   124,   120,   130,   134,
     132,     0,   122,    16,   138,    72,    88,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    34,   112,   111,     0,
      33,     0,    54,     0,    51,    53,     0,     0,     0,     0,
      12,     0,     0,     0,     0,   135,     0,    15,    89,    86,
      87,     0,    93,     0,    77,    78,    80,     0,    82,   141,
      59,    66,    49,     0,    27,     0,     0,    31,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    17,     0,     0,
       0,     0,     0,    69,    52,    28,     0,   128,     0,   125,
       0,   121,   133,   123,    18,     0,    25,    23,    22,    21,
       0,    95,    79,    40,    35,    36,    40,    83,     0,     0,
       0,    19,     0,     0,     0,    94,    41,    42,    38,     0,
      39,     0,     0,    29,   129,   126,    20,    24,     0,    37,
      85,    84,     0,    26,     0,    30
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     6,     7,    77,     8,     9,    10,   110,   111,   112,
     187,   217,   113,   100,   198,   244,   245,   258,    11,    12,
     104,   173,   174,    13,    18,    14,    73,    74,    15,   105,
      41,    62,    63,    82,    83,    86,   125,   194,   166,   223,
     119,   120,   121,   122,    42,   123,    44,    45,    46,    84,
     153,    47,   101,   200,    48
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -135
static const yytype_int16 yypact[] =
{
     218,    48,    19,   -19,   -56,   -18,    71,   218,   -41,  -135,
    -135,  -135,  -135,  -135,  -135,  -135,  -135,  -135,    91,   -64,
      90,    19,    19,    19,  -135,   -11,  -135,   -59,  -135,  -135,
    -135,   200,   200,  -135,  -135,  -135,  -135,  -135,  -135,  -135,
     200,   107,    32,    79,  -135,  -135,  -135,  -135,    40,   128,
     184,   186,    58,    70,  -135,  -135,   196,   198,  -135,  -135,
      22,    19,  -135,   189,   200,   200,   200,   200,   200,    50,
    -135,  -135,  -135,   -20,  -135,   232,   186,    -7,   153,  -135,
      99,  -135,   115,  -135,  -135,   159,   223,    79,    24,    24,
    -135,  -135,   120,   200,   246,    28,    74,   186,  -135,   174,
     -82,  -135,    48,   124,  -135,  -135,   125,   209,   212,   130,
     -68,  -135,  -135,  -135,   217,   255,    19,   159,   159,   136,
    -135,  -135,  -135,   267,   256,   242,  -135,   111,   155,  -135,
     258,  -135,  -135,    79,  -135,   186,    91,    83,   186,   166,
     171,   159,  -135,   153,   172,  -135,   173,   176,  -135,  -135,
     177,   244,   178,  -135,  -135,  -135,  -135,    -1,    15,   159,
     159,   288,   200,   200,   246,   159,   238,  -135,  -135,   304,
    -135,   107,  -135,    20,  -135,  -135,    30,   186,   186,     3,
    -135,   303,   305,   306,   307,  -135,   308,   181,  -135,   301,
    -135,   200,    79,   275,   187,  -135,   136,   296,   245,  -135,
    -135,  -135,  -135,    83,  -135,    45,    54,  -135,    64,    82,
     190,   191,   192,   286,   195,    19,   135,  -135,   282,   200,
     246,   113,   317,  -135,  -135,  -135,   278,  -135,   319,  -135,
     320,  -135,  -135,  -135,   175,   159,   201,  -135,  -135,  -135,
     200,    79,  -135,   160,   202,  -135,   160,   -55,    19,   204,
     205,  -135,   285,     5,   186,    79,  -135,  -135,  -135,   113,
    -135,   324,   328,   208,  -135,  -135,  -135,  -135,   104,  -135,
    -135,  -135,   186,  -135,   112,  -135
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -135,  -135,  -135,  -135,   329,  -135,  -135,  -135,   194,  -135,
    -135,  -135,  -135,  -134,  -135,  -135,    76,    92,  -135,  -135,
    -135,  -135,   137,  -135,   237,  -135,  -135,   247,  -135,  -135,
     206,   170,  -135,  -135,   227,   272,  -135,  -135,  -135,  -135,
     -60,  -135,  -135,  -135,  -135,   -17,  -128,  -135,   131,    -2,
    -135,   -92,   110,  -135,  -135
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      20,    43,   128,    49,   176,    96,   159,   160,    56,   175,
     159,   160,   159,   160,    58,    59,   102,   261,    21,    52,
      53,    54,    19,    60,   161,   162,    65,    66,    67,    68,
      22,   103,   163,    65,    66,    67,    68,    67,    68,    65,
      66,    67,    68,   205,   206,   134,   135,    87,    88,    89,
      90,    91,    95,    27,    28,    29,    30,   156,   157,   142,
     143,    31,    32,    92,    16,    50,    93,    17,    23,    94,
      57,    24,   195,   262,    34,   175,   127,    27,    28,    29,
      30,   179,   133,   117,    26,    31,    32,    28,    29,    30,
      65,    66,    67,    68,    27,    28,    29,    30,    34,   189,
     190,   158,    31,    32,    33,   196,   130,    34,    97,    35,
      36,    37,    38,    39,    55,    34,    27,   172,   243,    43,
     268,    51,    65,    66,    67,    68,   188,    61,   242,   246,
     207,    70,   267,    35,    36,    37,    38,    39,   274,    28,
      29,    30,    81,   159,   160,   192,   193,   202,   203,    81,
      35,    36,    37,    38,    39,   129,    72,   204,   135,   237,
      64,    75,    27,    28,    29,    30,    69,   246,   117,   238,
      31,    32,   225,   135,   218,   253,    40,    27,    28,    29,
      30,   226,   135,    34,    76,    31,    32,    71,   114,    72,
     213,   227,   228,   106,   107,   108,    78,   109,    34,    79,
     118,    80,   241,    27,    28,    29,    30,    75,   132,   229,
     230,    31,    32,   236,    85,   251,   252,    40,    35,    36,
      37,    38,    39,   255,    34,   214,   215,   216,   115,   256,
     257,   273,   135,    35,    36,    37,    38,    39,   167,   275,
     135,     1,    99,   116,   124,   170,   263,   126,     2,    27,
     137,   138,   139,   114,     3,   140,   141,     4,   154,    35,
      36,    37,    38,    39,   165,     5,   144,   145,   146,   147,
     148,   149,   150,   151,   164,   152,   161,   162,    65,    66,
      67,    68,   168,   219,   163,   118,    65,    66,    67,    68,
     240,   169,   177,    65,    66,    67,    68,   178,   181,   182,
      40,   185,   183,   184,   186,   191,   197,   199,   208,   160,
     209,   210,   211,   212,   221,   220,   222,   231,   232,   233,
     234,   235,   247,   248,   249,   250,    40,   254,   266,   270,
     259,   264,   265,   271,   272,   269,    25,   180,   260,   136,
     224,   201,   171,   155,   131,    98,     0,   239
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-135))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       2,    18,    94,    67,   138,    25,     7,     8,    67,   137,
       7,     8,     7,     8,    31,    32,    23,    72,    37,    21,
      22,    23,     3,    40,     9,    10,    11,    12,    13,    14,
      86,    38,    17,    11,    12,    13,    14,    13,    14,    11,
      12,    13,    14,   177,   178,   127,   128,    64,    65,    66,
      67,    68,    69,     3,     4,     5,     6,   117,   118,   127,
     128,    11,    12,    13,    16,   129,    16,    19,    86,    19,
     129,     0,   164,   128,    24,   203,    93,     3,     4,     5,
       6,   141,    99,     9,   125,    11,    12,     4,     5,     6,
      11,    12,    13,    14,     3,     4,     5,     6,    24,   159,
     160,   118,    11,    12,    13,   165,    32,    24,   128,    59,
      60,    61,    62,    63,   125,    24,     3,    34,     5,   136,
     254,    31,    11,    12,    13,    14,   127,    20,   220,   221,
     127,     3,   127,    59,    60,    61,    62,    63,   272,     4,
       5,     6,   127,     7,     8,   162,   163,   127,   128,   127,
      59,    60,    61,    62,    63,   127,     3,   127,   128,    24,
     128,    51,     3,     4,     5,     6,   126,   259,     9,    34,
      11,    12,   127,   128,   191,   235,   126,     3,     4,     5,
       6,   127,   128,    24,   126,    11,    12,     3,    78,     3,
       9,   127,   128,    40,    41,    42,   126,    44,    24,     3,
     126,     3,   219,     3,     4,     5,     6,    97,    34,   127,
     128,    11,    12,   215,    25,    40,    41,   126,    59,    60,
      61,    62,    63,   240,    24,    44,    45,    46,   129,    69,
      70,   127,   128,    59,    60,    61,    62,    63,   127,   127,
     128,    23,    10,   128,    21,   135,   248,   127,    30,     3,
     126,   126,    43,   143,    36,    43,   126,    39,     3,    59,
      60,    61,    62,    63,    22,    47,    49,    50,    51,    52,
      53,    54,    55,    56,    18,    58,     9,    10,    11,    12,
      13,    14,   127,     8,    17,   126,    11,    12,    13,    14,
       8,    33,   126,    11,    12,    13,    14,   126,   126,   126,
     126,    57,   126,   126,   126,    17,    68,     3,     5,     8,
       5,     5,     5,     5,    18,   128,    71,   127,   127,   127,
      34,   126,     5,    45,     5,     5,   126,   126,    43,     5,
     128,   127,   127,     5,   126,   259,     7,   143,   246,   102,
     203,   171,   136,   116,    97,    73,    -1,   216
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    23,    30,    36,    39,    47,   131,   132,   134,   135,
     136,   148,   149,   153,   155,   158,    16,    19,   154,     3,
     179,    37,    86,    86,     0,   134,   125,     3,     4,     5,
       6,    11,    12,    13,    24,    59,    60,    61,    62,    63,
     126,   160,   174,   175,   176,   177,   178,   181,   184,    67,
     129,    31,   179,   179,   179,   125,    67,   129,   175,   175,
     175,    20,   161,   162,   128,    11,    12,    13,    14,   126,
       3,     3,     3,   156,   157,   182,   126,   133,   126,     3,
       3,   127,   163,   164,   179,    25,   165,   175,   175,   175,
     175,   175,    13,    16,    19,   175,    25,   128,   165,    10,
     143,   182,    23,    38,   150,   159,    40,    41,    42,    44,
     137,   138,   139,   142,   182,   129,   128,     9,   126,   170,
     171,   172,   173,   175,    21,   166,   127,   175,   181,   127,
      32,   157,    34,   175,   127,   128,   154,   126,   126,    43,
      43,   126,   127,   128,    49,    50,    51,    52,    53,    54,
      55,    56,    58,   180,     3,   164,   170,   170,   175,     7,
       8,     9,    10,    17,    18,    22,   168,   127,   127,    33,
     182,   160,    34,   151,   152,   176,   143,   126,   126,   170,
     138,   126,   126,   126,   126,    57,   126,   140,   127,   170,
     170,    17,   175,   175,   167,   181,   170,    68,   144,     3,
     183,   161,   127,   128,   127,   143,   143,   127,     5,     5,
       5,     5,     5,     9,    44,    45,    46,   141,   175,     8,
     128,    18,    71,   169,   152,   127,   127,   127,   128,   127,
     128,   127,   127,   127,    34,   126,   179,    24,    34,   178,
       8,   175,   181,     5,   145,   146,   181,     5,    45,     5,
       5,    40,    41,   170,   126,   175,    69,    70,   147,   128,
     147,    72,   128,   179,   127,   127,    43,   127,   143,   146,
       5,     5,   126,   127,   143,   127
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* This macro is provided for backward compatibility. */

#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
                if (! (yysize <= yysize1
                       && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                  return 2;
                yysize = yysize1;
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  yysize1 = yysize + yystrlen (yyformat);
  if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
    return 2;
  yysize = yysize1;

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:

/* Line 1806 of yacc.c  */
#line 119 "sqlParser1G.y"
    { exit(0);}
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 123 "sqlParser1G.y"
    { ex((yyvsp[(1) - (2)].nPtr)); freeNode((yyvsp[(1) - (2)].nPtr)); }
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 124 "sqlParser1G.y"
    { ex((yyvsp[(2) - (3)].nPtr)); freeNode((yyvsp[(2) - (3)].nPtr)); }
    break;

  case 5:

/* Line 1806 of yacc.c  */
#line 128 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_COLUMN_COMMALIST, 0); }
    break;

  case 6:

/* Line 1806 of yacc.c  */
#line 129 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_COLUMN_COMMALIST, 1, (yyvsp[(2) - (3)].nPtr)); }
    break;

  case 7:

/* Line 1806 of yacc.c  */
#line 133 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SQL, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 137 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCHEMA, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 141 "sqlParser1G.y"
    { (yyval.nPtr) = opr(BASE_TABLE_DEF, 4, opr(CREATE, 0), opr(TABLE, 0), (yyvsp[(3) - (6)].nPtr), (yyvsp[(5) - (6)].nPtr)); }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 142 "sqlParser1G.y"
    { (yyval.nPtr) = opr(BASE_TABLE_DEF, 3, opr(DROP, 0), opr(TABLE, 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 146 "sqlParser1G.y"
    { (yyval.nPtr) = opr(BASE_TABLE_ELEMENT_COMMALIST, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 147 "sqlParser1G.y"
    { (yyval.nPtr) = opr(BASE_TABLE_ELEMENT_COMMALIST, 3, (yyvsp[(1) - (3)].nPtr), opr(',', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 151 "sqlParser1G.y"
    { (yyval.nPtr) = opr(BASE_TABLE_ELEMENT, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 152 "sqlParser1G.y"
    { (yyval.nPtr) = opr(BASE_TABLE_ELEMENT, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 156 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF, 3, (yyvsp[(1) - (3)].nPtr), (yyvsp[(2) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 160 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT_LIST, 0); }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 161 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT_LIST, 2, (yyvsp[(1) - (2)].nPtr), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 165 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 2, opr(NOT, 0), opr(NULLX, 0)); }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 166 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 3, opr(NOT, 0), opr(NULLX, 0), opr(UNIQUE, 0)); }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 167 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 3, opr(NOT, 0), opr(NULLX, 0), opr(PRIMARY, 0), opr(KEY, 0)); }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 168 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 2, opr(DEFAULT, 0), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 169 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 2, opr(DEFAULT, 0), opr(NULLX, 0)); }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 170 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 2, opr(DEFAULT, 0), opr(USER, 0)); }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 171 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 2, opr(CHECK, 0), (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 172 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 2, opr(REFERENCES, 0), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 173 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_DEF_OPT, 3, opr(REFERENCES, 0), (yyvsp[(2) - (5)].nPtr), (yyvsp[(4) - (5)].nPtr)); }
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 177 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_CONSTRAINT_DEF, 2, opr(UNIQUE, 0), (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 178 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_CONSTRAINT_DEF, 3, opr(PRIMARY, 0), opr(KEY, 0), (yyvsp[(4) - (5)].nPtr)); }
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 180 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_CONSTRAINT_DEF, 5, opr(FOREIGN, 0), opr(KEY, 0), (yyvsp[(4) - (7)].nPtr), opr(REFERENCES, 0), (yyvsp[(7) - (7)].nPtr)); }
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 182 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_CONSTRAINT_DEF, 6, opr(FOREIGN, 0), opr(KEY, 0), (yyvsp[(4) - (10)].nPtr), opr(REFERENCES, 0), (yyvsp[(7) - (10)].nPtr), (yyvsp[(9) - (10)].nPtr)); }
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 183 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_CONSTRAINT_DEF, 2, opr(CHECK, 0), (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 187 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_COMMALIST, 1, (yyvsp[(1) - (1)].nPtr); }
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 188 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_COMMALIST, 3, (yyvsp[(1) - (3)].nPtr), opr(',', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 192 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_ORDER_BY_CLAUSE, 0); }
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 193 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_ORDER_BY_CLAUSE, 3, opr(ORDER, 0), opr(BY, 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 197 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ORDERING_SPEC_COMMALIST, 1, (yyvsp[(1) - (1)].nPtr); }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 198 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ORDERING_SPEC_COMMALIST, 3, (yyvsp[(1) - (3)].nPtr), ',', (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 202 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ORDERING_SPEC, 2, con((yyvsp[(1) - (2)].iValue)), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 203 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ORDERING_SPEC, 2, (yyvsp[(1) - (2)].nPtr), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 207 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_ASC_DESC, 0); }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 208 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_ASC_DESC, 1, opr(ASC)); }
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 209 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_ASC_DESC, 1, opr(DESC)); }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 215 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SQL, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 219 "sqlParser1G.y"
    { (yyval.nPtr) = opr(MANIPULATIVE_STATEMENT, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 220 "sqlParser1G.y"
    { (yyval.nPtr) = opr(MANIPULATIVE_STATEMENT, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 221 "sqlParser1G.y"
    { (yyval.nPtr) = opr(MANIPULATIVE_STATEMENT, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 222 "sqlParser1G.y"
    { (yyval.nPtr) = opr(MANIPULATIVE_STATEMENT, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 237 "sqlParser1G.y"
    { (yyval.nPtr) = opr(INSERT_STATEMENT, 5, opr(INSERT, 0), opr(INTO, 0), (yyvsp[(3) - (5)].nPtr), (yyvsp[(4) - (5)].nPtr), (yyvsp[(5) - (5)].nPtr)); }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 241 "sqlParser1G.y"
    { (yyval.nPtr) = opr(VALUES_OR_QUERY_SPEC, 2, opr(VALUES, 0), (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 242 "sqlParser1G.y"
    { (yyval.nPtr) = opr(VALUES_OR_QUERY_SPEC, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 246 "sqlParser1G.y"
    { (yyval.nPtr) = opr(INSERT_ATOM_COMMALIST, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 247 "sqlParser1G.y"
    { (yyval.nPtr) = opr(INSERT_ATOM_COMMALIST, 3, (yyvsp[(1) - (3)].nPtr), opr(',', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 251 "sqlParser1G.y"
    { (yyval.nPtr) = opr(INSERT_ATOM, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 252 "sqlParser1G.y"
    { (yyval.nPtr) = opr(INSERT_ATOM, 1, (opr(NULLX, 0)); }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 258 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SELECT_STATEMENT, 4, opr(SELECT, 0), (yyvsp[(2) - (4)].nPtr), (yyvsp[(3) - (4)].nPtr), (yyvsp[(4) - (4)].nPtr)); }
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 263 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_ALL_DISTINCT, 1, opr(ALL, 0)); }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 264 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_ALL_DISTINCT, 1, opr(DISTINCT, 0)); }
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 265 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_ALL_DISTINCT, 0); }
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 270 "sqlParser1G.y"
    { (yyval.nPtr) = opr(UPDATE_STATEMENT_POSITIONED, 8, opr(UPDATE, 0), (yyvsp[(2) - (8)].nPtr), opr(SET, 0), (yyvsp[(4) - (8)].nPtr),
                                                       opr(WHERE, 0), opr(CURRENT, 0), opr(OF, 0), (yyvsp[(8) - (8)].nPtr)); }
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 275 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ASSIGNMENT_COMMALIST, 0); }
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 276 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ASSIGNMENT, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 277 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ASSIGNMENT_COMMALIST, 3, (yyvsp[(1) - (3)].nPtr), opr(',', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 281 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ASSIGNMENT, 3, (yyvsp[(1) - (3)].nPtr), compAssgn((yyvsp[(2) - (3)].sSubtok)), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 282 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ASSIGNMENT, 3, (yyvsp[(1) - (3)].nPtr), compAssgn((yyvsp[(2) - (3)].sSubtok)), opr(NULLX, 0)); }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 286 "sqlParser1G.y"
    { (yyval.nPtr) = opr(UPDATE_STATEMENT_SEARCHED, 5, opr(UPDATE, 0), (yyvsp[(2) - (5)].nPtr), opr(SET, 0), (yyvsp[(4) - (5)].nPtr), (yyvsp[(5) - (5)].nPtr)); }
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 290 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SELECT_STATEMENT, 4, opr(SELECT, 0), (yyvsp[(2) - (4)].nPtr), (yyvsp[(3) - (4)].nPtr), (yyvsp[(4) - (4)].nPtr)); }
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 294 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SELECTION, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 295 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SELECTION, 1, opr(SELALL, 0)); }
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 304 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_EXP, 6, (yyvsp[(1) - (6)].nPtr), (yyvsp[(2) - (6)].nPtr), (yyvsp[(3) - (6)].nPtr), (yyvsp[(4) - (6)].nPtr), (yyvsp[(5) - (6)].nPtr), (yyvsp[(6) - (6)].nPtr)); }
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 308 "sqlParser1G.y"
    { (yyval.nPtr) = opr(FROM_CLAUSE, 2, opr(FROM, 0), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 312 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_REF_COMMALIST, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 313 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_REF_COMMALIST, 3 (yyvsp[(1) - (3)].nPtr), opr(',', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 317 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE_REF, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 322 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_WHERE_CLAUSE, 2, opr(WHERE, 0), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 75:

/* Line 1806 of yacc.c  */
#line 323 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_WHERE_CLAUSE, 0); }
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 327 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_GROUP_BY_CLAUSE, 0); }
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 328 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_GROUP_BY_CLAUSE, 3, opr(GROUP, 0), opr(BY, 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 332 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_REF_COMMALIST, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 333 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_REF_COMMALIST, 3, (yyvsp[(1) - (3)].nPtr), opr(',', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 337 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_HAVING_CLAUSE, 2, opr(HAVING, 0), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 338 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_HAVING_CLAUSE, 0); }
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 342 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_LIMIT_CLAUSE, 0); }
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 343 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_LIMIT_CLAUSE, 2, opr(LIMIT, 0), con((yyvsp[(2) - (2)].iValue))); }
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 344 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_LIMIT_CLAUSE, 3, opr(LIMIT, 0), opr(',', 0), con((yyvsp[(2) - (4)].iValue))); }
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 345 "sqlParser1G.y"
    { (yyval.nPtr) = opr(OPT_LIMIT_CLAUSE, 3, opr(LIMIT, 0), opr(OFFSET, 0), con((yyvsp[(2) - (4)].iValue))); }
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 349 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SEARCH_CONDITION, 3, (yyvsp[(1) - (3)].nPtr), opr(OR, 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 350 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SEARCH_CONDITION, 3, (yyvsp[(1) - (3)].nPtr), opr(AND, 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 351 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SEARCH_CONDITION, 2, opr(NOT, 0), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 352 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SEARCH_CONDITION, 1, (yyvsp[(2) - (3)].nPtr)); }
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 353 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SEARCH_CONDITION, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 357 "sqlParser1G.y"
    { (yyval.nPtr) = opr(PREDICATE, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 358 "sqlParser1G.y"
    { (yyval.nPtr) = opr(PREDICATE, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 367 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COMPARISON_PREDICATE, 3, (yyvsp[(1) - (3)].nPtr), comp((yyvsp[(2) - (3)].sSubtok)), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 372 "sqlParser1G.y"
    { (yyval.nPtr) = opr(BETWEEN_PREDICATE, 6, (yyvsp[(1) - (6)].nPtr), opr(NOT, 0), opr(BETWEEN, 0), (yyvsp[(4) - (6)].nPtr), opr(AND, 0), (yyvsp[(6) - (6)].nPtr)); }
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 373 "sqlParser1G.y"
    { (yyval.nPtr) = opr(BETWEEN_PREDICATE, 5, (yyvsp[(1) - (5)].nPtr), opr(BETWEEN, 0), (yyvsp[(3) - (5)].nPtr), opr(AND, 0), (yyvsp[(5) - (5)].nPtr)); }
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 377 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP_COMMALIST, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 378 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP_COMMALIST, 3, (yyvsp[(1) - (3)].nPtr), opr(',', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 382 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 3, (yyvsp[(1) - (3)].nPtr), opr('+', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 383 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 3, (yyvsp[(1) - (3)].nPtr), opr('-', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 384 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 3, (yyvsp[(1) - (3)].nPtr), opr('*', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 385 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 3, (yyvsp[(1) - (3)].nPtr), opr('/', 0), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 386 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 3, opr('+', 0), (yyvsp[(2) - (2)].nPtr), opr(UMINUS, 0)); }
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 387 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 3, opr('-', 0), (yyvsp[(2) - (2)].nPtr), opr(UMINUS, 0)); }
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 388 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 389 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 390 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 391 "sqlParser1G.y"
    { (yyval.nPtr) = opr(SCALAR_EXP, 1, (yyvsp[(2) - (3)].nPtr)); }
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 396 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ATOM, 1, (yyvsp[(1) - (1)].nPtr)); }
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 397 "sqlParser1G.y"
    { (yyval.nPtr) = opr(ATOM, 1, opr(USER, 0)); }
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 408 "sqlParser1G.y"
    { (yyval.nPtr) = opr(FUNCTION_REF, 2, (yyvsp[(1) - (4)].nPtr), opr(SELALL, 0));}
    break;

  case 111:

/* Line 1806 of yacc.c  */
#line 409 "sqlParser1G.y"
    { (yyval.nPtr) = opr(FUNCTION_REF, 3, (yyvsp[(1) - (5)].nPtr), opr(DISTINCT, 0), (yyvsp[(4) - (5)].nPtr)); }
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 410 "sqlParser1G.y"
    { (yyval.nPtr) = opr(FUNCTION_REF, 3, (yyvsp[(1) - (5)].nPtr), opr(ALL, 0), (yyvsp[(4) - (5)].nPtr)); }
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 411 "sqlParser1G.y"
    { (yyval.nPtr) = opr(FUNCTION_REF, 2, (yyvsp[(1) - (4)].nPtr), (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 114:

/* Line 1806 of yacc.c  */
#line 415 "sqlParser1G.y"
    { (yyval.nPtr) = opr(LITERAL, 1, text((yyvsp[(1) - (1)].sValue)); }
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 416 "sqlParser1G.y"
    { (yyval.nPtr) = opr(LITERAL, 1, con((yyvsp[(1) - (1)].iValue)); }
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 417 "sqlParser1G.y"
    { (yyval.nPtr) = opr(LITERAL, 1, con((yyvsp[(1) - (1)].fValue)); }
    break;

  case 117:

/* Line 1806 of yacc.c  */
#line 422 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE, 1, id((yyvsp[(1) - (1)].sName)); }
    break;

  case 118:

/* Line 1806 of yacc.c  */
#line 423 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE, 3, id((yyvsp[(1) - (3)].sName)), opr('.', 0), id2((yyvsp[(3) - (3)].sName)));}
    break;

  case 119:

/* Line 1806 of yacc.c  */
#line 424 "sqlParser1G.y"
    { (yyval.nPtr) = opr(TABLE, 3, id((yyvsp[(1) - (3)].sName)), opr(AS, 0), id2((yyvsp[(3) - (3)].sName)));  }
    break;

  case 120:

/* Line 1806 of yacc.c  */
#line 429 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 1, opr(CHARACTER, 0)); }
    break;

  case 121:

/* Line 1806 of yacc.c  */
#line 430 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 2, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 431 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 1, opr(VARCHAR, 0)); }
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 432 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 2, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 124:

/* Line 1806 of yacc.c  */
#line 433 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 1, opr(NUMERIC, 0)); }
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 434 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 2, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 126:

/* Line 1806 of yacc.c  */
#line 435 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 2, con((yyvsp[(3) - (6)].iValue))); }
    break;

  case 127:

/* Line 1806 of yacc.c  */
#line 436 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 1, opr(DECIMAL)); }
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 437 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 2, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 129:

/* Line 1806 of yacc.c  */
#line 438 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 2, con((yyvsp[(3) - (6)].iValue))); }
    break;

  case 130:

/* Line 1806 of yacc.c  */
#line 439 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 1, opr(INTEGER, 0)); }
    break;

  case 131:

/* Line 1806 of yacc.c  */
#line 440 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 1, opr(SMALLINT, 0)); }
    break;

  case 132:

/* Line 1806 of yacc.c  */
#line 441 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 1, opr(FLOAT, 0)); }
    break;

  case 133:

/* Line 1806 of yacc.c  */
#line 442 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 2, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 443 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 1, opr(REAL, 0)); }
    break;

  case 135:

/* Line 1806 of yacc.c  */
#line 444 "sqlParser1G.y"
    { (yyval.nPtr) = opr(DATATYPE, 2, opr(DOUBLE, 0), opr(PRECISION, 0)); }
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 447 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_REF, 1, id((yyvsp[(1) - (1)].sName))); }
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 448 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_REF, 3, id((yyvsp[(1) - (3)].sName)), opr('.', 0), id2((yyvsp[(3) - (3)].sName)));}
    break;

  case 138:

/* Line 1806 of yacc.c  */
#line 449 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_REF, 3, id((yyvsp[(1) - (5)].sName)), opr('.', 0), opr(COLUMN_REF, 3, id((yyvsp[(1) - (5)].sName)),  opr('.', 0), id2((yyvsp[(3) - (5)].sName))));}
    break;

  case 139:

/* Line 1806 of yacc.c  */
#line 450 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN_REF, 3, id((yyvsp[(1) - (3)].sName)), opr(AS, 0), id2((yyvsp[(3) - (3)].sName))); }
    break;

  case 140:

/* Line 1806 of yacc.c  */
#line 454 "sqlParser1G.y"
    { (yyval.nPtr) = opr(COLUMN, 1, id((yyvsp[(1) - (1)].sName))); }
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 457 "sqlParser1G.y"
    { (yyval.nPtr) = opr(CURSOR, 1, id((yyvsp[(1) - (1)].sName))); }
    break;

  case 142:

/* Line 1806 of yacc.c  */
#line 461 "sqlParser1G.y"
    { (yyval.nPtr) = opr(AMMSC, 1, opr(AVG, 0); }
    break;

  case 143:

/* Line 1806 of yacc.c  */
#line 462 "sqlParser1G.y"
    { (yyval.nPtr) = opr(AMMSC, 1, opr(MIN, 0); }
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 463 "sqlParser1G.y"
    { (yyval.nPtr) = opr(AMMSC, 1, opr(MAX, 0); }
    break;

  case 145:

/* Line 1806 of yacc.c  */
#line 464 "sqlParser1G.y"
    { (yyval.nPtr) = opr(AMMSC, 1, opr(SUM, 0); }
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 465 "sqlParser1G.y"
    { (yyval.nPtr) = opr(AMMSC, 1, opr(COUNT, 0); }
    break;



/* Line 1806 of yacc.c  */
#line 2954 "sqlParser1G.c"
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 2067 of yacc.c  */
#line 467 "sqlParser1G.y"


nodeType *id(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeId;
    p->id.s = s;
    p->id.iLength = textLength;
    //printf("text: %d %s \n", textLength, s);
    return p;
}

/* backup of id when two names are needed simultaneously, such as Name.Name */
nodeType *id2(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeId;
    p->id.s = s;
    p->id.iLength = textLength2;
    //Fprintf("text: %d %s \n", textLength2, s);
    return p;
}

/* Handles regular text, as copied by the STRING token. */
nodeType *text(char* s) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeText;
    p->id.s = s;
    p->id.iLength = textLength;
 //   printf("text: %d %s \n", textLength, s);
    return p;
}


nodeType *comp(char* s) {
    nodeType *p;

    /* comparators: =, >, etc. */
    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeComp;
    p->id.s = s;
    p->id.iLength = comparisonLength;
   // printf("comp: %d %s \n", comparisonLength, s);
    return p;
}

/* Treat assignment statements as a comparator. Cheesy, I know. */
nodeType *compAssgn(char* s) {
    nodeType *p;

    /* Treat all comparators: =, >, etc. that appear grammatically as an assignment "=" as that "=".

    To do: Find a workaround that allows the assignment rule to appear properly in the grammar. */
    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    if (!strcmp(s, "=")) {
        printf("wrong comparator\n");
        fflush(stdout);
        yyerror("wrong comparator");
    }
    /* copy information */
    p->type = typeAssgn;
    p->id.s = "=";
    p->id.iLength = strlen("=");
  //  printf("comp: %d %s \n", comparisonLength, s);
    return p;
}

nodeType *con(float value) {
    nodeType *p;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeCon;
    p->con.fValue = value;

    return p;
}

nodeType *opr(int oper, int nops, ...) {
    va_list ap;
    nodeType *p;
    int i;

    /* allocate node */
    if ((p = malloc(sizeof(nodeType))) == NULL)
        yyerror("out of memory");
    if ((p->opr.op = malloc(nops * sizeof(nodeType))) == NULL)
        yyerror("out of memory");

    /* copy information */
    p->type = typeOpr;
    p->opr.oper = oper;
    p->opr.nops = nops;
    va_start(ap, nops);
    for (i = 0; i < nops; i++)
        p->opr.op[i] = va_arg(ap, nodeType*);
    va_end(ap);
    return p;
}

void freeNode(nodeType *p) {
    int i;

    if (!p) return;
    if (p->type == typeOpr) {
        for (i = 0; i < p->opr.nops; i++)
            freeNode(p->opr.op[i]);
		free (p->opr.op);
    }
    free (p);
}

int yyerror(const char *s) {
    fprintf(stdout, "%s\n", s);
    /* should this return 1? */
    return 1;
}

int main(void) {
    int i = yyparse();
    //printf("success? %d\n", i);
    return i;
}
