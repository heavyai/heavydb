#ifndef YY_parse_h_included
#define YY_parse_h_included
#define YY_USE_CLASS

#line 1 "/usr/share/bison++/bison.h"
/* before anything */
#ifdef c_plusplus
 #ifndef __cplusplus
  #define __cplusplus
 #endif
#endif


 #line 8 "/usr/share/bison++/bison.h"

#line 33 "sqlParser1AlteredInput.y"
typedef union {
    char *sValue;                /* string*/
    char *sName;
    char *sParam;
    nodeType *nPtr;             /* node pointer */
    float fValue;                 /* approximate number */
    int iValue;
    char* sSubtok;      /* comparator subtokens */
    int iLength;
} yy_parse_stype;
#define YY_parse_STYPE yy_parse_stype

#line 21 "/usr/share/bison++/bison.h"
 /* %{ and %header{ and %union, during decl */
#ifndef YY_parse_COMPATIBILITY
 #ifndef YY_USE_CLASS
  #define  YY_parse_COMPATIBILITY 1
 #else
  #define  YY_parse_COMPATIBILITY 0
 #endif
#endif

#if YY_parse_COMPATIBILITY != 0
/* backward compatibility */
 #ifdef YYLTYPE
  #ifndef YY_parse_LTYPE
   #define YY_parse_LTYPE YYLTYPE
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
/* use %define LTYPE */
  #endif
 #endif
/*#ifdef YYSTYPE*/
  #ifndef YY_parse_STYPE
   #define YY_parse_STYPE YYSTYPE
  /* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
   /* use %define STYPE */
  #endif
/*#endif*/
 #ifdef YYDEBUG
  #ifndef YY_parse_DEBUG
   #define  YY_parse_DEBUG YYDEBUG
   /* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
   /* use %define DEBUG */
  #endif
 #endif 
 /* use goto to be compatible */
 #ifndef YY_parse_USE_GOTO
  #define YY_parse_USE_GOTO 1
 #endif
#endif

/* use no goto to be clean in C++ */
#ifndef YY_parse_USE_GOTO
 #define YY_parse_USE_GOTO 0
#endif

#ifndef YY_parse_PURE

 #line 65 "/usr/share/bison++/bison.h"

#line 65 "/usr/share/bison++/bison.h"
/* YY_parse_PURE */
#endif


 #line 68 "/usr/share/bison++/bison.h"

#line 68 "/usr/share/bison++/bison.h"
/* prefix */

#ifndef YY_parse_DEBUG

 #line 71 "/usr/share/bison++/bison.h"

#line 71 "/usr/share/bison++/bison.h"
/* YY_parse_DEBUG */
#endif

#ifndef YY_parse_LSP_NEEDED

 #line 75 "/usr/share/bison++/bison.h"

#line 75 "/usr/share/bison++/bison.h"
 /* YY_parse_LSP_NEEDED*/
#endif

/* DEFAULT LTYPE*/
#ifdef YY_parse_LSP_NEEDED
 #ifndef YY_parse_LTYPE
  #ifndef BISON_YYLTYPE_ISDECLARED
   #define BISON_YYLTYPE_ISDECLARED
typedef
  struct yyltype
    {
      int timestamp;
      int first_line;
      int first_column;
      int last_line;
      int last_column;
      char *text;
   }
  yyltype;
  #endif

  #define YY_parse_LTYPE yyltype
 #endif
#endif

/* DEFAULT STYPE*/
#ifndef YY_parse_STYPE
 #define YY_parse_STYPE int
#endif

/* DEFAULT MISCELANEOUS */
#ifndef YY_parse_PARSE
 #define YY_parse_PARSE yyparse
#endif

#ifndef YY_parse_LEX
 #define YY_parse_LEX yylex
#endif

#ifndef YY_parse_LVAL
 #define YY_parse_LVAL yylval
#endif

#ifndef YY_parse_LLOC
 #define YY_parse_LLOC yylloc
#endif

#ifndef YY_parse_CHAR
 #define YY_parse_CHAR yychar
#endif

#ifndef YY_parse_NERRS
 #define YY_parse_NERRS yynerrs
#endif

#ifndef YY_parse_DEBUG_FLAG
 #define YY_parse_DEBUG_FLAG yydebug
#endif

#ifndef YY_parse_ERROR
 #define YY_parse_ERROR yyerror
#endif

#ifndef YY_parse_PARSE_PARAM
 #ifndef __STDC__
  #ifndef __cplusplus
   #ifndef YY_USE_CLASS
    #define YY_parse_PARSE_PARAM
    #ifndef YY_parse_PARSE_PARAM_DEF
     #define YY_parse_PARSE_PARAM_DEF
    #endif
   #endif
  #endif
 #endif
 #ifndef YY_parse_PARSE_PARAM
  #define YY_parse_PARSE_PARAM void
 #endif
#endif

/* TOKEN C */
#ifndef YY_USE_CLASS

 #ifndef YY_parse_PURE
  #ifndef yylval
   extern YY_parse_STYPE YY_parse_LVAL;
  #else
   #if yylval != YY_parse_LVAL
    extern YY_parse_STYPE YY_parse_LVAL;
   #else
    #warning "Namespace conflict, disabling some functionality (bison++ only)"
   #endif
  #endif
 #endif


 #line 169 "/usr/share/bison++/bison.h"
#define	NAME	258
#define	STRING	259
#define	INTNUM	260
#define	APPROXNUM	261
#define	OR	262
#define	AND	263
#define	NOT	264
#define	COMPARISON	265
#define	UMINUS	266
#define	ALL	267
#define	BETWEEN	268
#define	BY	269
#define	DISTINCT	270
#define	FROM	271
#define	GROUP	272
#define	HAVING	273
#define	SELECT	274
#define	USER	275
#define	WHERE	276
#define	WITH	277
#define	EMPTY	278
#define	SELALL	279
#define	DOT	280
#define	UPDATE	281
#define	SET	282
#define	CURRENT	283
#define	OF	284
#define	NULLX	285
#define	ASSIGN	286
#define	INSERT	287
#define	INTO	288
#define	VALUES	289
#define	CREATE	290
#define	TABLE	291
#define	UNIQUE	292
#define	PRIMARY	293
#define	FOREIGN	294
#define	KEY	295
#define	CHECK	296
#define	REFERENCES	297
#define	DEFAULT	298
#define	DROP	299
#define	DATATYPE	300
#define	DECIMAL	301
#define	SMALLINT	302
#define	NUMERIC	303
#define	CHARACTER	304
#define	INTEGER	305
#define	REAL	306
#define	FLOAT	307
#define	DOUBLE	308
#define	PRECISION	309
#define	VARCHAR	310
#define	AMMSC	311
#define	AVG	312
#define	MAX	313
#define	MIN	314
#define	SUM	315
#define	COUNT	316
#define	ALIAS	317
#define	INTORDER	318
#define	COLORDER	319
#define	AS	320
#define	ORDER	321
#define	ASC	322
#define	DESC	323
#define	LIMIT	324
#define	OFFSET	325


#line 169 "/usr/share/bison++/bison.h"
 /* #defines token */
/* after #define tokens, before const tokens S5*/
#else
 #ifndef YY_parse_CLASS
  #define YY_parse_CLASS parse
 #endif

 #ifndef YY_parse_INHERIT
  #define YY_parse_INHERIT
 #endif

 #ifndef YY_parse_MEMBERS
  #define YY_parse_MEMBERS 
 #endif

 #ifndef YY_parse_LEX_BODY
  #define YY_parse_LEX_BODY  
 #endif

 #ifndef YY_parse_ERROR_BODY
  #define YY_parse_ERROR_BODY  
 #endif

 #ifndef YY_parse_CONSTRUCTOR_PARAM
  #define YY_parse_CONSTRUCTOR_PARAM
 #endif
 /* choose between enum and const */
 #ifndef YY_parse_USE_CONST_TOKEN
  #define YY_parse_USE_CONST_TOKEN 0
  /* yes enum is more compatible with flex,  */
  /* so by default we use it */ 
 #endif
 #if YY_parse_USE_CONST_TOKEN != 0
  #ifndef YY_parse_ENUM_TOKEN
   #define YY_parse_ENUM_TOKEN yy_parse_enum_token
  #endif
 #endif

class YY_parse_CLASS YY_parse_INHERIT
{
public: 
 #if YY_parse_USE_CONST_TOKEN != 0
  /* static const int token ... */
  
 #line 212 "/usr/share/bison++/bison.h"
static const int NAME;
static const int STRING;
static const int INTNUM;
static const int APPROXNUM;
static const int OR;
static const int AND;
static const int NOT;
static const int COMPARISON;
static const int UMINUS;
static const int ALL;
static const int BETWEEN;
static const int BY;
static const int DISTINCT;
static const int FROM;
static const int GROUP;
static const int HAVING;
static const int SELECT;
static const int USER;
static const int WHERE;
static const int WITH;
static const int EMPTY;
static const int SELALL;
static const int DOT;
static const int UPDATE;
static const int SET;
static const int CURRENT;
static const int OF;
static const int NULLX;
static const int ASSIGN;
static const int INSERT;
static const int INTO;
static const int VALUES;
static const int CREATE;
static const int TABLE;
static const int UNIQUE;
static const int PRIMARY;
static const int FOREIGN;
static const int KEY;
static const int CHECK;
static const int REFERENCES;
static const int DEFAULT;
static const int DROP;
static const int DATATYPE;
static const int DECIMAL;
static const int SMALLINT;
static const int NUMERIC;
static const int CHARACTER;
static const int INTEGER;
static const int REAL;
static const int FLOAT;
static const int DOUBLE;
static const int PRECISION;
static const int VARCHAR;
static const int AMMSC;
static const int AVG;
static const int MAX;
static const int MIN;
static const int SUM;
static const int COUNT;
static const int ALIAS;
static const int INTORDER;
static const int COLORDER;
static const int AS;
static const int ORDER;
static const int ASC;
static const int DESC;
static const int LIMIT;
static const int OFFSET;


#line 212 "/usr/share/bison++/bison.h"
 /* decl const */
 #else
  enum YY_parse_ENUM_TOKEN { YY_parse_NULL_TOKEN=0
  
 #line 215 "/usr/share/bison++/bison.h"
	,NAME=258
	,STRING=259
	,INTNUM=260
	,APPROXNUM=261
	,OR=262
	,AND=263
	,NOT=264
	,COMPARISON=265
	,UMINUS=266
	,ALL=267
	,BETWEEN=268
	,BY=269
	,DISTINCT=270
	,FROM=271
	,GROUP=272
	,HAVING=273
	,SELECT=274
	,USER=275
	,WHERE=276
	,WITH=277
	,EMPTY=278
	,SELALL=279
	,DOT=280
	,UPDATE=281
	,SET=282
	,CURRENT=283
	,OF=284
	,NULLX=285
	,ASSIGN=286
	,INSERT=287
	,INTO=288
	,VALUES=289
	,CREATE=290
	,TABLE=291
	,UNIQUE=292
	,PRIMARY=293
	,FOREIGN=294
	,KEY=295
	,CHECK=296
	,REFERENCES=297
	,DEFAULT=298
	,DROP=299
	,DATATYPE=300
	,DECIMAL=301
	,SMALLINT=302
	,NUMERIC=303
	,CHARACTER=304
	,INTEGER=305
	,REAL=306
	,FLOAT=307
	,DOUBLE=308
	,PRECISION=309
	,VARCHAR=310
	,AMMSC=311
	,AVG=312
	,MAX=313
	,MIN=314
	,SUM=315
	,COUNT=316
	,ALIAS=317
	,INTORDER=318
	,COLORDER=319
	,AS=320
	,ORDER=321
	,ASC=322
	,DESC=323
	,LIMIT=324
	,OFFSET=325


#line 215 "/usr/share/bison++/bison.h"
 /* enum token */
     }; /* end of enum declaration */
 #endif
public:
 int YY_parse_PARSE(YY_parse_PARSE_PARAM);
 virtual void YY_parse_ERROR(char *msg) YY_parse_ERROR_BODY;
 #ifdef YY_parse_PURE
  #ifdef YY_parse_LSP_NEEDED
   virtual int  YY_parse_LEX(YY_parse_STYPE *YY_parse_LVAL,YY_parse_LTYPE *YY_parse_LLOC) YY_parse_LEX_BODY;
  #else
   virtual int  YY_parse_LEX(YY_parse_STYPE *YY_parse_LVAL) YY_parse_LEX_BODY;
  #endif
 #else
  virtual int YY_parse_LEX() YY_parse_LEX_BODY;
  YY_parse_STYPE YY_parse_LVAL;
  #ifdef YY_parse_LSP_NEEDED
   YY_parse_LTYPE YY_parse_LLOC;
  #endif
  int YY_parse_NERRS;
  int YY_parse_CHAR;
 #endif
 #if YY_parse_DEBUG != 0
  public:
   int YY_parse_DEBUG_FLAG;	/*  nonzero means print parse trace	*/
 #endif
public:
 YY_parse_CLASS(YY_parse_CONSTRUCTOR_PARAM);
public:
 YY_parse_MEMBERS 
};
/* other declare folow */
#endif


#if YY_parse_COMPATIBILITY != 0
 /* backward compatibility */
 /* Removed due to bison problems
 /#ifndef YYSTYPE
 / #define YYSTYPE YY_parse_STYPE
 /#endif*/

 #ifndef YYLTYPE
  #define YYLTYPE YY_parse_LTYPE
 #endif
 #ifndef YYDEBUG
  #ifdef YY_parse_DEBUG 
   #define YYDEBUG YY_parse_DEBUG
  #endif
 #endif

#endif
/* END */

 #line 267 "/usr/share/bison++/bison.h"
#endif
