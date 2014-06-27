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
#line 1 "sqlParser1AlteredInput.y"

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

extern int readInputForLexer(char* buffer,int *numBytesRead,int maxBytesToRead);


/* Line 268 of yacc.c  */
#line 104 "y.tab.c"

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

/* Line 293 of yacc.c  */
#line 33 "sqlParser1AlteredInput.y"

    char *sValue;                /* string*/
    char *sName;
    char *sParam;
    nodeType *nPtr;             /* node pointer */
    float fValue;                 /* approximate number */
    int iValue;
    char* sSubtok;      /* comparator subtokens */
    int iLength;



/* Line 293 of yacc.c  */
#line 293 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 305 "y.tab.c"

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
#define YYLAST   353

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  80
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  55
/* YYNRULES -- Number of rules.  */
#define YYNRULES  146
/* YYNRULES -- Number of states.  */
#define YYNSTATES  276

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   325

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      76,    77,    13,    11,    78,    12,    79,    14,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,    75,
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
      69,    70,    71,    72,    73,    74
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
      81,     0,    -1,    82,    -1,    84,    75,    -1,    82,    84,
      75,    -1,    -1,    76,    93,    77,    -1,    85,    -1,    86,
      -1,    39,    40,   129,    76,    87,    77,    -1,    48,    40,
     129,    -1,    88,    -1,    87,    78,    88,    -1,    89,    -1,
      92,    -1,   132,   130,    90,    -1,    -1,    90,    91,    -1,
       9,    34,    -1,     9,    34,    41,    -1,     9,    34,    42,
      44,    -1,    47,   128,    -1,    47,    34,    -1,    47,    24,
      -1,    45,    76,   120,    77,    -1,    46,   129,    -1,    46,
     129,    76,    93,    77,    -1,    41,    76,    93,    77,    -1,
      42,    44,    76,    93,    77,    -1,    43,    44,    76,    93,
      77,    46,   129,    -1,    43,    44,    76,    93,    77,    46,
     129,    76,    93,    77,    -1,    45,    76,   120,    77,    -1,
     132,    -1,    93,    78,   132,    -1,    -1,    70,    18,    95,
      -1,    96,    -1,    95,    78,    96,    -1,     5,    97,    -1,
     131,    97,    -1,    -1,    71,    -1,    72,    -1,    98,    -1,
     103,    -1,   105,    -1,   108,    -1,    99,    -1,    36,    37,
     129,    83,   100,    -1,    38,    76,   101,    77,    -1,   109,
      -1,   102,    -1,   101,    78,   102,    -1,   126,    -1,    34,
      -1,    23,   104,   110,   111,    -1,    16,    -1,    19,    -1,
      -1,    30,   129,    31,   106,    25,    32,    33,   133,    -1,
      -1,   107,    -1,   106,    78,   107,    -1,   132,    10,   125,
      -1,   132,    10,    34,    -1,    30,   129,    31,   106,   115,
      -1,    23,   104,   110,   111,    -1,   124,    -1,    13,    -1,
     112,   115,   116,   118,    94,   119,    -1,    20,   113,    -1,
     114,    -1,   113,    78,   114,    -1,   129,    -1,    25,   120,
      -1,    -1,    -1,    21,    18,   117,    -1,   131,    -1,   117,
      78,   131,    -1,    22,   120,    -1,    -1,    -1,    73,     5,
      -1,    73,     5,    78,     5,    -1,    73,     5,    74,     5,
      -1,   120,     7,   120,    -1,   120,     8,   120,    -1,     9,
     120,    -1,    76,   120,    77,    -1,   121,    -1,   122,    -1,
     123,    -1,   125,    10,   125,    -1,   125,     9,    17,   125,
       8,   125,    -1,   125,    17,   125,     8,   125,    -1,   125,
      -1,   124,    78,   125,    -1,   125,    11,   125,    -1,   125,
      12,   125,    -1,   125,    13,   125,    -1,   125,    14,   125,
      -1,    11,   125,    -1,    12,   125,    -1,   126,    -1,   131,
      -1,   127,    -1,    76,   125,    77,    -1,   128,    -1,    24,
      -1,   134,    76,    13,    77,    -1,   134,    76,    19,   131,
      77,    -1,   134,    76,    16,   125,    77,    -1,   134,    76,
     125,    77,    -1,     4,    -1,     5,    -1,     6,    -1,     3,
      -1,     3,    79,     3,    -1,     3,    69,     3,    -1,    53,
      -1,    53,    76,     5,    77,    -1,    59,    -1,    59,    76,
       5,    77,    -1,    52,    -1,    52,    76,     5,    77,    -1,
      52,    76,     5,    78,     5,    77,    -1,    50,    -1,    50,
      76,     5,    77,    -1,    50,    76,     5,    78,     5,    77,
      -1,    54,    -1,    51,    -1,    56,    -1,    56,    76,     5,
      77,    -1,    55,    -1,    57,    58,    -1,     3,    -1,     3,
      79,     3,    -1,     3,    79,     3,    79,     3,    -1,     3,
      69,     3,    -1,     3,    -1,     3,    -1,    61,    -1,    63,
      -1,    62,    -1,    64,    -1,    65,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   104,   104,   108,   109,   113,   114,   118,   122,   126,
     127,   131,   132,   136,   137,   141,   145,   146,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   162,   163,   164,
     166,   168,   172,   173,   177,   178,   182,   183,   187,   188,
     192,   193,   194,   200,   204,   205,   206,   207,   222,   226,
     227,   231,   232,   236,   237,   241,   248,   249,   250,   254,
     259,   260,   261,   265,   266,   270,   274,   278,   279,   283,
     292,   296,   297,   301,   306,   307,   311,   312,   316,   317,
     321,   322,   326,   327,   328,   329,   333,   334,   335,   336,
     337,   341,   342,   351,   356,   357,   361,   362,   366,   367,
     368,   369,   370,   371,   372,   373,   374,   375,   380,   381,
     392,   393,   394,   395,   399,   400,   401,   406,   407,   408,
     413,   414,   415,   416,   417,   418,   419,   420,   421,   422,
     423,   424,   425,   426,   427,   428,   432,   433,   434,   435,
     439,   442,   446,   447,   448,   449,   450
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
  "TABLE", "UNIQUE", "PRIMARY", "FOREIGN", "KEY", "CHECK", "REFERENCES",
  "DEFAULT", "DROP", "DATATYPE", "DECIMAL", "SMALLINT", "NUMERIC",
  "CHARACTER", "INTEGER", "REAL", "FLOAT", "DOUBLE", "PRECISION",
  "VARCHAR", "AMMSC", "AVG", "MAX", "MIN", "SUM", "COUNT", "ALIAS",
  "INTORDER", "COLORDER", "AS", "ORDER", "ASC", "DESC", "LIMIT", "OFFSET",
  "';'", "'('", "')'", "','", "'.'", "$accept", "program", "sql_list",
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
     321,   322,   323,   324,   325,    59,    40,    41,    44,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    80,    81,    82,    82,    83,    83,    84,    85,    86,
      86,    87,    87,    88,    88,    89,    90,    90,    91,    91,
      91,    91,    91,    91,    91,    91,    91,    92,    92,    92,
      92,    92,    93,    93,    94,    94,    95,    95,    96,    96,
      97,    97,    97,    84,    98,    98,    98,    98,    99,   100,
     100,   101,   101,   102,   102,   103,   104,   104,   104,   105,
     106,   106,   106,   107,   107,   108,   109,   110,   110,   111,
     112,   113,   113,   114,   115,   115,   116,   116,   117,   117,
     118,   118,   119,   119,   119,   119,   120,   120,   120,   120,
     120,   121,   121,   122,   123,   123,   124,   124,   125,   125,
     125,   125,   125,   125,   125,   125,   125,   125,   126,   126,
     127,   127,   127,   127,   128,   128,   128,   129,   129,   129,
     130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   130,   131,   131,   131,   131,
     132,   133,   134,   134,   134,   134,   134
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
      87,    73,     3,   -11,     6,    48,    65,    87,    18,  -135,
    -135,  -135,  -135,  -135,  -135,  -135,  -135,  -135,     5,   -41,
      78,     3,     3,     3,  -135,    39,  -135,    46,  -135,  -135,
    -135,   238,   238,  -135,  -135,  -135,  -135,  -135,  -135,  -135,
     238,    98,    43,   159,  -135,  -135,  -135,  -135,    75,   121,
     133,   198,   127,   155,  -135,  -135,   229,   235,  -135,  -135,
      45,     3,  -135,   184,   238,   238,   238,   238,   238,    92,
    -135,  -135,  -135,    -3,  -135,   237,   198,     4,    42,  -135,
     160,  -135,   167,  -135,  -135,   172,   233,   159,    60,    60,
    -135,  -135,   183,   238,   258,    50,   128,   198,  -135,   194,
      29,  -135,    73,   187,  -135,  -135,   207,   240,   241,   210,
      35,  -135,  -135,  -135,   223,   284,     3,   172,   172,   134,
    -135,  -135,  -135,   255,   270,   267,  -135,    66,   213,  -135,
     259,  -135,  -135,   159,  -135,   198,     5,   206,   198,   215,
     217,   172,  -135,    42,   218,  -135,   219,   220,  -135,  -135,
     221,   246,   222,  -135,  -135,  -135,  -135,    17,   152,   172,
     172,   288,   238,   238,   258,   172,   236,  -135,  -135,   304,
    -135,    98,  -135,    81,  -135,  -135,   102,   198,   198,    23,
    -135,   303,   305,   306,   307,  -135,   308,    -6,  -135,   301,
    -135,   238,   159,   136,   239,  -135,   134,   297,   243,  -135,
    -135,  -135,  -135,   206,  -135,   117,   130,  -135,   137,   139,
     242,   244,   245,   286,   248,     3,   247,  -135,   174,   238,
     258,    32,   313,  -135,  -135,  -135,   277,  -135,   320,  -135,
     321,  -135,  -135,  -135,   178,   172,   251,  -135,  -135,  -135,
     238,   159,  -135,   150,   250,  -135,   150,    12,     3,   252,
     253,  -135,   287,    25,   198,   159,  -135,  -135,  -135,    32,
    -135,   327,   328,   260,  -135,  -135,  -135,  -135,   147,  -135,
    -135,  -135,   198,  -135,   149,  -135
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -135,  -135,  -135,  -135,   330,  -135,  -135,  -135,   191,  -135,
    -135,  -135,  -135,  -134,  -135,  -135,    76,    93,  -135,  -135,
    -135,  -135,   135,  -135,   249,  -135,  -135,   256,  -135,  -135,
     204,   170,  -135,  -135,   226,   271,  -135,  -135,  -135,  -135,
    -105,  -135,  -135,  -135,  -135,   -17,  -132,  -135,   129,    -2,
    -135,   -92,   -44,  -135,  -135
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint16 yytable[] =
{
      20,    43,   128,   213,   176,   175,    19,    75,    27,    28,
      29,    30,   156,   157,    58,    59,    31,    32,    33,    52,
      53,    54,    96,    60,   159,   160,    21,   102,    49,    34,
     159,   160,   159,   160,   114,    27,   179,   243,    50,   214,
     215,   216,   103,   205,   206,    72,    22,    87,    88,    89,
      90,    91,    95,    75,   189,   190,    65,    66,    67,    68,
     196,    65,    66,    67,    68,    24,    35,    36,    37,    38,
      39,   175,   195,    67,    68,    97,   127,    65,    66,    67,
      68,    40,   133,   106,   107,   108,   261,   109,    23,    16,
     262,   170,    17,    26,   188,    27,    28,    29,    30,   114,
     207,   158,   267,    31,    32,    92,   134,   135,    93,    51,
       1,    94,   142,   143,    55,    56,    34,     2,    61,    43,
     268,    64,    81,     3,    70,    57,     4,   129,   242,   246,
     253,    27,    28,    29,    30,     5,    71,   117,   274,    31,
      32,   159,   160,   167,   219,   192,   193,    65,    66,    67,
      68,    69,    34,    35,    36,    37,    38,    39,   202,   203,
     130,   161,   162,    65,    66,    67,    68,   246,    40,   163,
      65,    66,    67,    68,   218,    27,    28,    29,    30,   204,
     135,   117,   240,    31,    32,    65,    66,    67,    68,    35,
      36,    37,    38,    39,   225,   135,    34,    27,    28,    29,
      30,    72,   241,    76,   118,    31,    32,   226,   135,    85,
      28,    29,    30,   236,   227,   228,   229,   230,    34,   251,
     252,   256,   257,   255,   273,   135,   275,   135,   132,    81,
      34,    78,    79,    35,    36,    37,    38,    39,    80,   115,
     172,    27,    28,    29,    30,   116,   263,    99,   118,    31,
      32,    28,    29,    30,   124,    35,    36,    37,    38,    39,
     126,    27,    34,   137,   161,   162,    65,    66,    67,    68,
      40,   237,   163,   144,   145,   146,   147,   148,   149,   150,
     151,   238,   152,   138,   139,   140,   141,   154,   164,   165,
     168,   177,   169,   178,   181,   182,   183,   184,   186,    35,
      36,    37,    38,    39,   185,   191,   197,   199,   208,   160,
     209,   210,   211,   212,    40,   221,   222,   220,   247,   231,
     234,   232,   233,   248,   235,   249,   250,   254,   259,   264,
     265,   266,   270,   271,   180,   269,   272,    25,   224,   260,
     171,   201,   155,     0,    98,   239,     0,     0,     0,     0,
       0,   136,     0,   131
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-135))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       2,    18,    94,     9,   138,   137,     3,    51,     3,     4,
       5,     6,   117,   118,    31,    32,    11,    12,    13,    21,
      22,    23,    25,    40,     7,     8,    37,    23,    69,    24,
       7,     8,     7,     8,    78,     3,   141,     5,    79,    45,
      46,    47,    38,   177,   178,     3,    40,    64,    65,    66,
      67,    68,    69,    97,   159,   160,    11,    12,    13,    14,
     165,    11,    12,    13,    14,     0,    61,    62,    63,    64,
      65,   203,   164,    13,    14,    78,    93,    11,    12,    13,
      14,    76,    99,    41,    42,    43,    74,    45,    40,    16,
      78,   135,    19,    75,    77,     3,     4,     5,     6,   143,
      77,   118,    77,    11,    12,    13,    77,    78,    16,    31,
      23,    19,    77,    78,    75,    69,    24,    30,    20,   136,
     254,    78,    77,    36,     3,    79,    39,    77,   220,   221,
     235,     3,     4,     5,     6,    48,     3,     9,   272,    11,
      12,     7,     8,    77,     8,   162,   163,    11,    12,    13,
      14,    76,    24,    61,    62,    63,    64,    65,    77,    78,
      32,     9,    10,    11,    12,    13,    14,   259,    76,    17,
      11,    12,    13,    14,   191,     3,     4,     5,     6,    77,
      78,     9,     8,    11,    12,    11,    12,    13,    14,    61,
      62,    63,    64,    65,    77,    78,    24,     3,     4,     5,
       6,     3,   219,    76,    76,    11,    12,    77,    78,    25,
       4,     5,     6,   215,    77,    78,    77,    78,    24,    41,
      42,    71,    72,   240,    77,    78,    77,    78,    34,    77,
      24,    76,     3,    61,    62,    63,    64,    65,     3,    79,
      34,     3,     4,     5,     6,    78,   248,    10,    76,    11,
      12,     4,     5,     6,    21,    61,    62,    63,    64,    65,
      77,     3,    24,    76,     9,    10,    11,    12,    13,    14,
      76,    24,    17,    50,    51,    52,    53,    54,    55,    56,
      57,    34,    59,    76,    44,    44,    76,     3,    18,    22,
      77,    76,    33,    76,    76,    76,    76,    76,    76,    61,
      62,    63,    64,    65,    58,    17,    70,     3,     5,     8,
       5,     5,     5,     5,    76,    18,    73,    78,     5,    77,
      34,    77,    77,    46,    76,     5,     5,    76,    78,    77,
      77,    44,     5,     5,   143,   259,    76,     7,   203,   246,
     136,   171,   116,    -1,    73,   216,    -1,    -1,    -1,    -1,
      -1,   102,    -1,    97
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    23,    30,    36,    39,    48,    81,    82,    84,    85,
      86,    98,    99,   103,   105,   108,    16,    19,   104,     3,
     129,    37,    40,    40,     0,    84,    75,     3,     4,     5,
       6,    11,    12,    13,    24,    61,    62,    63,    64,    65,
      76,   110,   124,   125,   126,   127,   128,   131,   134,    69,
      79,    31,   129,   129,   129,    75,    69,    79,   125,   125,
     125,    20,   111,   112,    78,    11,    12,    13,    14,    76,
       3,     3,     3,   106,   107,   132,    76,    83,    76,     3,
       3,    77,   113,   114,   129,    25,   115,   125,   125,   125,
     125,   125,    13,    16,    19,   125,    25,    78,   115,    10,
      93,   132,    23,    38,   100,   109,    41,    42,    43,    45,
      87,    88,    89,    92,   132,    79,    78,     9,    76,   120,
     121,   122,   123,   125,    21,   116,    77,   125,   131,    77,
      32,   107,    34,   125,    77,    78,   104,    76,    76,    44,
      44,    76,    77,    78,    50,    51,    52,    53,    54,    55,
      56,    57,    59,   130,     3,   114,   120,   120,   125,     7,
       8,     9,    10,    17,    18,    22,   118,    77,    77,    33,
     132,   110,    34,   101,   102,   126,    93,    76,    76,   120,
      88,    76,    76,    76,    76,    58,    76,    90,    77,   120,
     120,    17,   125,   125,   117,   131,   120,    70,    94,     3,
     133,   111,    77,    78,    77,    93,    93,    77,     5,     5,
       5,     5,     5,     9,    45,    46,    47,    91,   125,     8,
      78,    18,    73,   119,   102,    77,    77,    77,    78,    77,
      78,    77,    77,    77,    34,    76,   129,    24,    34,   128,
       8,   125,   131,     5,    95,    96,   131,     5,    46,     5,
       5,    41,    42,   120,    76,   125,    71,    72,    97,    78,
      97,    74,    78,   129,    77,    77,    44,    77,    93,    96,
       5,     5,    76,    77,    93,    77
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
#line 104 "sqlParser1AlteredInput.y"
    { exit(0);}
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 108 "sqlParser1AlteredInput.y"
    { freeNode((yyvsp[(1) - (2)].nPtr)); }
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 109 "sqlParser1AlteredInput.y"
    { freeNode((yyvsp[(2) - (3)].nPtr)); }
    break;

  case 5:

/* Line 1806 of yacc.c  */
#line 113 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 6:

/* Line 1806 of yacc.c  */
#line 114 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(2) - (3)].nPtr); }
    break;

  case 7:

/* Line 1806 of yacc.c  */
#line 118 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 122 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 126 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(CREATE, 2, opr(TABLE, 1, (yyvsp[(3) - (6)].nPtr)), (yyvsp[(5) - (6)].nPtr)); }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 127 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DROP, 1, opr(TABLE, 1, (yyvsp[(3) - (3)].nPtr))); }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 131 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 132 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 136 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 137 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 141 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DATATYPE, 3, (yyvsp[(1) - (3)].nPtr), (yyvsp[(2) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 145 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 146 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (2)].nPtr), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 150 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(NOT, 1, opr(NULLX, 0)); }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 151 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(UNIQUE, 1, opr(NOT, 1, opr(NULLX, 0))); }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 152 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(KEY, 1, opr(PRIMARY, 1, opr(NOT, 1, opr(NULLX, 0)))); }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 153 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DEFAULT, 1, (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 154 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DEFAULT, 1, opr(NULLX, 0)); }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 155 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DEFAULT, 1, opr(USER, 0)); }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 156 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(CHECK, 1, (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 157 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(REFERENCES, 1, (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 158 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(REFERENCES, 2, (yyvsp[(2) - (5)].nPtr), (yyvsp[(4) - (5)].nPtr)); }
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 162 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(UNIQUE, 1, (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 163 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(KEY, 1, opr(PRIMARY, 1, (yyvsp[(4) - (5)].nPtr))); }
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 165 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(KEY, 2, opr(FOREIGN, 1, (yyvsp[(4) - (7)].nPtr)), opr(REFERENCES, 1, (yyvsp[(7) - (7)].nPtr))); }
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 167 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(KEY, 2, opr(FOREIGN, 1, (yyvsp[(4) - (10)].nPtr)), opr(REFERENCES, 2, (yyvsp[(7) - (10)].nPtr), (yyvsp[(9) - (10)].nPtr))); }
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 168 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(CHECK, 1, (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 172 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 173 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 177 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 178 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(ORDER, 1, opr(BY, 1, (yyvsp[(3) - (3)].nPtr))); }
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 182 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 183 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 187 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(INTORDER, 2, con((yyvsp[(1) - (2)].iValue)), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 188 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(COLORDER, 2, (yyvsp[(1) - (2)].nPtr), (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 192 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 193 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(ASC, 0); }
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 194 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DESC, 0); }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 200 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 204 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 205 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 206 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 207 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 222 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(INSERT, 3, opr(INTO, 1, (yyvsp[(3) - (5)].nPtr)), (yyvsp[(4) - (5)].nPtr), (yyvsp[(5) - (5)].nPtr)); }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 226 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(VALUES, 1, (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 227 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 231 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 232 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 236 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 237 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = NULL; }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 243 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(SELECT, 3, (yyvsp[(2) - (4)].nPtr), (yyvsp[(3) - (4)].nPtr), (yyvsp[(4) - (4)].nPtr)); }
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 248 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(ALL, 0); }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 249 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DISTINCT, 0); }
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 250 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 255 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(UPDATE, 5, (yyvsp[(2) - (8)].nPtr), opr(SET, 1, (yyvsp[(4) - (8)].nPtr)), opr(WHERE, 0), opr(CURRENT, 0), opr(OF, 1, (yyvsp[(8) - (8)].nPtr))); }
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 259 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 260 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr);}
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 261 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 265 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(ASSIGN, 3, (yyvsp[(1) - (3)].nPtr), compAssgn((yyvsp[(2) - (3)].sSubtok)), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 266 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(ASSIGN, 3, (yyvsp[(1) - (3)].nPtr), compAssgn((yyvsp[(2) - (3)].sSubtok)), opr(NULLX, 0)); }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 270 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(UPDATE, 3, (yyvsp[(2) - (5)].nPtr), opr(SET, 1, (yyvsp[(4) - (5)].nPtr)), (yyvsp[(5) - (5)].nPtr)); }
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 274 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(SELECT, 3, (yyvsp[(2) - (4)].nPtr), (yyvsp[(3) - (4)].nPtr), (yyvsp[(4) - (4)].nPtr)); }
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 278 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 279 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(SELALL, 0); }
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 288 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(FROM, 6, (yyvsp[(1) - (6)].nPtr), (yyvsp[(2) - (6)].nPtr), (yyvsp[(3) - (6)].nPtr), (yyvsp[(4) - (6)].nPtr), (yyvsp[(5) - (6)].nPtr), (yyvsp[(6) - (6)].nPtr)); }
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 292 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(2) - (2)].nPtr); }
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 296 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 297 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 301 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 306 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(WHERE, 1, (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 75:

/* Line 1806 of yacc.c  */
#line 307 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 311 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 312 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(GROUP, 1, opr(BY, 1, (yyvsp[(3) - (3)].nPtr))); }
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 316 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 317 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 321 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(HAVING, 1, (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 322 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 326 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(EMPTY, 0); }
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 327 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(LIMIT, 1, con((yyvsp[(2) - (2)].iValue))); }
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 328 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(LIMIT, 2, con((yyvsp[(2) - (4)].iValue)), con((yyvsp[(4) - (4)].iValue))); }
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 329 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(LIMIT, 2, con((yyvsp[(2) - (4)].iValue)), con((yyvsp[(4) - (4)].iValue))); }
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 333 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(OR, 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 334 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(AND, 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 335 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(NOT, 1, (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 336 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(2) - (3)].nPtr); }
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 337 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 341 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 342 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 351 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(COMPARISON, 3, (yyvsp[(1) - (3)].nPtr), comp((yyvsp[(2) - (3)].sSubtok)), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 356 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(NOT, 1, opr(BETWEEN, 2, (yyvsp[(1) - (6)].nPtr), opr(AND, 2, (yyvsp[(4) - (6)].nPtr), (yyvsp[(6) - (6)].nPtr)))); }
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 357 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(BETWEEN, 2, (yyvsp[(1) - (5)].nPtr), opr(AND, 2, (yyvsp[(3) - (5)].nPtr), (yyvsp[(5) - (5)].nPtr))); }
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 361 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 362 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(',', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 366 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr('+', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 367 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr('-', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 368 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr('*', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 369 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr('/', 2, (yyvsp[(1) - (3)].nPtr), (yyvsp[(3) - (3)].nPtr)); }
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 370 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(2) - (2)].nPtr); }
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 371 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(UMINUS, 1, (yyvsp[(2) - (2)].nPtr)); }
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 372 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 373 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 374 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 375 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(2) - (3)].nPtr); }
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 380 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = (yyvsp[(1) - (1)].nPtr); }
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 381 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(USER, 0); }
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 392 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(AMMSC, 2, (yyvsp[(1) - (4)].nPtr), opr(SELALL, 0));}
    break;

  case 111:

/* Line 1806 of yacc.c  */
#line 393 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(AMMSC, 3, (yyvsp[(1) - (5)].nPtr), opr(DISTINCT, 0), (yyvsp[(4) - (5)].nPtr)); }
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 394 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(AMMSC, 3, (yyvsp[(1) - (5)].nPtr), opr(ALL, 0), (yyvsp[(4) - (5)].nPtr)); }
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 395 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(AMMSC, 2, (yyvsp[(1) - (4)].nPtr), (yyvsp[(3) - (4)].nPtr)); }
    break;

  case 114:

/* Line 1806 of yacc.c  */
#line 399 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = text((yyvsp[(1) - (1)].sValue)); }
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 400 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = con((yyvsp[(1) - (1)].iValue)); }
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 401 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = con((yyvsp[(1) - (1)].fValue)); }
    break;

  case 117:

/* Line 1806 of yacc.c  */
#line 406 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = id((yyvsp[(1) - (1)].sName)); }
    break;

  case 118:

/* Line 1806 of yacc.c  */
#line 407 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DOT, 2, id((yyvsp[(1) - (3)].sName)), id2((yyvsp[(3) - (3)].sName)));}
    break;

  case 119:

/* Line 1806 of yacc.c  */
#line 408 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(ALIAS, 2, id((yyvsp[(1) - (3)].sName)), id2((yyvsp[(3) - (3)].sName)));  }
    break;

  case 120:

/* Line 1806 of yacc.c  */
#line 413 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(CHARACTER, 0); }
    break;

  case 121:

/* Line 1806 of yacc.c  */
#line 414 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(CHARACTER, 1, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 415 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(VARCHAR, 0); }
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 416 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(VARCHAR, 1, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 124:

/* Line 1806 of yacc.c  */
#line 417 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(NUMERIC, 0); }
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 418 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(NUMERIC, 1, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 126:

/* Line 1806 of yacc.c  */
#line 419 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(NUMERIC, 1, opr(',', 2, (yyvsp[(3) - (6)].iValue), (yyvsp[(5) - (6)].iValue))); }
    break;

  case 127:

/* Line 1806 of yacc.c  */
#line 420 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DECIMAL, 0); }
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 421 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DECIMAL, 1, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 129:

/* Line 1806 of yacc.c  */
#line 422 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DECIMAL, 1, opr(',', 2, con((yyvsp[(3) - (6)].iValue)), con((yyvsp[(5) - (6)].iValue)))); }
    break;

  case 130:

/* Line 1806 of yacc.c  */
#line 423 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(INTEGER, 0); }
    break;

  case 131:

/* Line 1806 of yacc.c  */
#line 424 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(SMALLINT, 0); }
    break;

  case 132:

/* Line 1806 of yacc.c  */
#line 425 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(FLOAT, 0); }
    break;

  case 133:

/* Line 1806 of yacc.c  */
#line 426 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(FLOAT, 1, con((yyvsp[(3) - (4)].iValue))); }
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 427 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(REAL, 0); }
    break;

  case 135:

/* Line 1806 of yacc.c  */
#line 428 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DOUBLE, 1, opr(PRECISION, 0)); }
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 432 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = id((yyvsp[(1) - (1)].sName)); }
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 433 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DOT, 2, id((yyvsp[(1) - (3)].sName)), id2((yyvsp[(3) - (3)].sName)));}
    break;

  case 138:

/* Line 1806 of yacc.c  */
#line 434 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(DOT, 2, id((yyvsp[(1) - (5)].sName)),  opr(DOT, 2, id((yyvsp[(1) - (5)].sName)), id2((yyvsp[(3) - (5)].sName))));}
    break;

  case 139:

/* Line 1806 of yacc.c  */
#line 435 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(ALIAS, 2, id((yyvsp[(1) - (3)].sName)), id2((yyvsp[(3) - (3)].sName))); }
    break;

  case 140:

/* Line 1806 of yacc.c  */
#line 439 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = id((yyvsp[(1) - (1)].sName)); }
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 442 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = id((yyvsp[(1) - (1)].sName)); }
    break;

  case 142:

/* Line 1806 of yacc.c  */
#line 446 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(AVG, 0); }
    break;

  case 143:

/* Line 1806 of yacc.c  */
#line 447 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(MIN, 0); }
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 448 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(MAX, 0); }
    break;

  case 145:

/* Line 1806 of yacc.c  */
#line 449 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(SUM, 0); }
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 450 "sqlParser1AlteredInput.y"
    { (yyval.nPtr) = opr(COUNT, 0); }
    break;



/* Line 1806 of yacc.c  */
#line 2832 "y.tab.c"
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
#line 452 "sqlParser1AlteredInput.y"


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

/*

int main(void) {
    int i = yyparse();
    //printf("success? %d\n", i);
    return i;
}*/
