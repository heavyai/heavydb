#ifndef YY_AlgCalcParser_h_included
#define YY_AlgCalcParser_h_included
#define YY_USE_CLASS

#line 1 "/usr/share/bison++/bison.h"
/* before anything */
#ifdef c_plusplus
 #ifndef __cplusplus
  #define __cplusplus
 #endif
#endif


 #line 8 "/usr/share/bison++/bison.h"
#define YY_AlgCalcParser_LSP_NEEDED 
#define YY_AlgCalcParser_ERROR_BODY  = 0
#define YY_AlgCalcParser_LEX_BODY  = 0
#define YY_AlgCalcParser_LTYPE  alg_ltype_t
#line 7 "AlgCalcParser.y"

#include <iostream>
#include <fstream>

#line 12 "AlgCalcParser.y"
typedef union {
	int itype;
	char ctype;
} yy_AlgCalcParser_stype;
#define YY_AlgCalcParser_STYPE yy_AlgCalcParser_stype
#line 16 "AlgCalcParser.y"

typedef struct
{
	int first_line;
	int last_line;
	int first_column;
	int last_column;
	char *text;
} alg_ltype_t;

#line 21 "/usr/share/bison++/bison.h"
 /* %{ and %header{ and %union, during decl */
#ifndef YY_AlgCalcParser_COMPATIBILITY
 #ifndef YY_USE_CLASS
  #define  YY_AlgCalcParser_COMPATIBILITY 1
 #else
  #define  YY_AlgCalcParser_COMPATIBILITY 0
 #endif
#endif

#if YY_AlgCalcParser_COMPATIBILITY != 0
/* backward compatibility */
 #ifdef YYLTYPE
  #ifndef YY_AlgCalcParser_LTYPE
   #define YY_AlgCalcParser_LTYPE YYLTYPE
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
/* use %define LTYPE */
  #endif
 #endif
/*#ifdef YYSTYPE*/
  #ifndef YY_AlgCalcParser_STYPE
   #define YY_AlgCalcParser_STYPE YYSTYPE
  /* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
   /* use %define STYPE */
  #endif
/*#endif*/
 #ifdef YYDEBUG
  #ifndef YY_AlgCalcParser_DEBUG
   #define  YY_AlgCalcParser_DEBUG YYDEBUG
   /* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
   /* use %define DEBUG */
  #endif
 #endif 
 /* use goto to be compatible */
 #ifndef YY_AlgCalcParser_USE_GOTO
  #define YY_AlgCalcParser_USE_GOTO 1
 #endif
#endif

/* use no goto to be clean in C++ */
#ifndef YY_AlgCalcParser_USE_GOTO
 #define YY_AlgCalcParser_USE_GOTO 0
#endif

#ifndef YY_AlgCalcParser_PURE

 #line 65 "/usr/share/bison++/bison.h"

#line 65 "/usr/share/bison++/bison.h"
/* YY_AlgCalcParser_PURE */
#endif


 #line 68 "/usr/share/bison++/bison.h"

#line 68 "/usr/share/bison++/bison.h"
/* prefix */

#ifndef YY_AlgCalcParser_DEBUG

 #line 71 "/usr/share/bison++/bison.h"

#line 71 "/usr/share/bison++/bison.h"
/* YY_AlgCalcParser_DEBUG */
#endif

#ifndef YY_AlgCalcParser_LSP_NEEDED

 #line 75 "/usr/share/bison++/bison.h"

#line 75 "/usr/share/bison++/bison.h"
 /* YY_AlgCalcParser_LSP_NEEDED*/
#endif

/* DEFAULT LTYPE*/
#ifdef YY_AlgCalcParser_LSP_NEEDED
 #ifndef YY_AlgCalcParser_LTYPE
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

  #define YY_AlgCalcParser_LTYPE yyltype
 #endif
#endif

/* DEFAULT STYPE*/
#ifndef YY_AlgCalcParser_STYPE
 #define YY_AlgCalcParser_STYPE int
#endif

/* DEFAULT MISCELANEOUS */
#ifndef YY_AlgCalcParser_PARSE
 #define YY_AlgCalcParser_PARSE yyparse
#endif

#ifndef YY_AlgCalcParser_LEX
 #define YY_AlgCalcParser_LEX yylex
#endif

#ifndef YY_AlgCalcParser_LVAL
 #define YY_AlgCalcParser_LVAL yylval
#endif

#ifndef YY_AlgCalcParser_LLOC
 #define YY_AlgCalcParser_LLOC yylloc
#endif

#ifndef YY_AlgCalcParser_CHAR
 #define YY_AlgCalcParser_CHAR yychar
#endif

#ifndef YY_AlgCalcParser_NERRS
 #define YY_AlgCalcParser_NERRS yynerrs
#endif

#ifndef YY_AlgCalcParser_DEBUG_FLAG
 #define YY_AlgCalcParser_DEBUG_FLAG yydebug
#endif

#ifndef YY_AlgCalcParser_ERROR
 #define YY_AlgCalcParser_ERROR yyerror
#endif

#ifndef YY_AlgCalcParser_PARSE_PARAM
 #ifndef __STDC__
  #ifndef __cplusplus
   #ifndef YY_USE_CLASS
    #define YY_AlgCalcParser_PARSE_PARAM
    #ifndef YY_AlgCalcParser_PARSE_PARAM_DEF
     #define YY_AlgCalcParser_PARSE_PARAM_DEF
    #endif
   #endif
  #endif
 #endif
 #ifndef YY_AlgCalcParser_PARSE_PARAM
  #define YY_AlgCalcParser_PARSE_PARAM void
 #endif
#endif

/* TOKEN C */
#ifndef YY_USE_CLASS

 #ifndef YY_AlgCalcParser_PURE
  #ifndef yylval
   extern YY_AlgCalcParser_STYPE YY_AlgCalcParser_LVAL;
  #else
   #if yylval != YY_AlgCalcParser_LVAL
    extern YY_AlgCalcParser_STYPE YY_AlgCalcParser_LVAL;
   #else
    #warning "Namespace conflict, disabling some functionality (bison++ only)"
   #endif
  #endif
 #endif


 #line 169 "/usr/share/bison++/bison.h"
#define	UNKNOWN	258
#define	PLUS	259
#define	MINUS	260
#define	EQUALS	261
#define	NUMBER	262


#line 169 "/usr/share/bison++/bison.h"
 /* #defines token */
/* after #define tokens, before const tokens S5*/
#else
 #ifndef YY_AlgCalcParser_CLASS
  #define YY_AlgCalcParser_CLASS AlgCalcParser
 #endif

 #ifndef YY_AlgCalcParser_INHERIT
  #define YY_AlgCalcParser_INHERIT
 #endif

 #ifndef YY_AlgCalcParser_MEMBERS
  #define YY_AlgCalcParser_MEMBERS 
 #endif

 #ifndef YY_AlgCalcParser_LEX_BODY
  #define YY_AlgCalcParser_LEX_BODY  
 #endif

 #ifndef YY_AlgCalcParser_ERROR_BODY
  #define YY_AlgCalcParser_ERROR_BODY  
 #endif

 #ifndef YY_AlgCalcParser_CONSTRUCTOR_PARAM
  #define YY_AlgCalcParser_CONSTRUCTOR_PARAM
 #endif
 /* choose between enum and const */
 #ifndef YY_AlgCalcParser_USE_CONST_TOKEN
  #define YY_AlgCalcParser_USE_CONST_TOKEN 0
  /* yes enum is more compatible with flex,  */
  /* so by default we use it */ 
 #endif
 #if YY_AlgCalcParser_USE_CONST_TOKEN != 0
  #ifndef YY_AlgCalcParser_ENUM_TOKEN
   #define YY_AlgCalcParser_ENUM_TOKEN yy_AlgCalcParser_enum_token
  #endif
 #endif

class YY_AlgCalcParser_CLASS YY_AlgCalcParser_INHERIT
{
public: 
 #if YY_AlgCalcParser_USE_CONST_TOKEN != 0
  /* static const int token ... */
  
 #line 212 "/usr/share/bison++/bison.h"
static const int UNKNOWN;
static const int PLUS;
static const int MINUS;
static const int EQUALS;
static const int NUMBER;


#line 212 "/usr/share/bison++/bison.h"
 /* decl const */
 #else
  enum YY_AlgCalcParser_ENUM_TOKEN { YY_AlgCalcParser_NULL_TOKEN=0
  
 #line 215 "/usr/share/bison++/bison.h"
	,UNKNOWN=258
	,PLUS=259
	,MINUS=260
	,EQUALS=261
	,NUMBER=262


#line 215 "/usr/share/bison++/bison.h"
 /* enum token */
     }; /* end of enum declaration */
 #endif
public:
 int YY_AlgCalcParser_PARSE(YY_AlgCalcParser_PARSE_PARAM);
 virtual void YY_AlgCalcParser_ERROR(char *msg) YY_AlgCalcParser_ERROR_BODY;
 #ifdef YY_AlgCalcParser_PURE
  #ifdef YY_AlgCalcParser_LSP_NEEDED
   virtual int  YY_AlgCalcParser_LEX(YY_AlgCalcParser_STYPE *YY_AlgCalcParser_LVAL,YY_AlgCalcParser_LTYPE *YY_AlgCalcParser_LLOC) YY_AlgCalcParser_LEX_BODY;
  #else
   virtual int  YY_AlgCalcParser_LEX(YY_AlgCalcParser_STYPE *YY_AlgCalcParser_LVAL) YY_AlgCalcParser_LEX_BODY;
  #endif
 #else
  virtual int YY_AlgCalcParser_LEX() YY_AlgCalcParser_LEX_BODY;
  YY_AlgCalcParser_STYPE YY_AlgCalcParser_LVAL;
  #ifdef YY_AlgCalcParser_LSP_NEEDED
   YY_AlgCalcParser_LTYPE YY_AlgCalcParser_LLOC;
  #endif
  int YY_AlgCalcParser_NERRS;
  int YY_AlgCalcParser_CHAR;
 #endif
 #if YY_AlgCalcParser_DEBUG != 0
  public:
   int YY_AlgCalcParser_DEBUG_FLAG;	/*  nonzero means print parse trace	*/
 #endif
public:
 YY_AlgCalcParser_CLASS(YY_AlgCalcParser_CONSTRUCTOR_PARAM);
public:
 YY_AlgCalcParser_MEMBERS 
};
/* other declare folow */
#endif


#if YY_AlgCalcParser_COMPATIBILITY != 0
 /* backward compatibility */
 /* Removed due to bison problems
 /#ifndef YYSTYPE
 / #define YYSTYPE YY_AlgCalcParser_STYPE
 /#endif*/

 #ifndef YYLTYPE
  #define YYLTYPE YY_AlgCalcParser_LTYPE
 #endif
 #ifndef YYDEBUG
  #ifdef YY_AlgCalcParser_DEBUG 
   #define YYDEBUG YY_AlgCalcParser_DEBUG
  #endif
 #endif

#endif
/* END */

 #line 267 "/usr/share/bison++/bison.h"
#endif
