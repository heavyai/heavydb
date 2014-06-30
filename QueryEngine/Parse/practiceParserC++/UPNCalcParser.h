#ifndef YY_UPNCalcParser_h_included
#define YY_UPNCalcParser_h_included
#define YY_USE_CLASS

#line 1 "/usr/share/bison++/bison.h"
/* before anything */
#ifdef c_plusplus
 #ifndef __cplusplus
  #define __cplusplus
 #endif
#endif


 #line 8 "/usr/share/bison++/bison.h"
#define YY_UPNCalcParser_LSP_NEEDED 
#define YY_UPNCalcParser_ERROR_BODY  = 0
#define YY_UPNCalcParser_LEX_BODY  = 0
#define YY_UPNCalcParser_LTYPE  upn_ltype_t
#line 7 "UPNCalcParser.y"

#include <iostream>
#include <fstream>

#line 12 "UPNCalcParser.y"
typedef union {
	int itype;
	char ctype;
} yy_UPNCalcParser_stype;
#define YY_UPNCalcParser_STYPE yy_UPNCalcParser_stype
#line 16 "UPNCalcParser.y"

typedef struct
{
	int first_line;
	int last_line;
	int first_column;
	int last_column;
	char *text;
} upn_ltype_t;

#line 21 "/usr/share/bison++/bison.h"
 /* %{ and %header{ and %union, during decl */
#ifndef YY_UPNCalcParser_COMPATIBILITY
 #ifndef YY_USE_CLASS
  #define  YY_UPNCalcParser_COMPATIBILITY 1
 #else
  #define  YY_UPNCalcParser_COMPATIBILITY 0
 #endif
#endif

#if YY_UPNCalcParser_COMPATIBILITY != 0
/* backward compatibility */
 #ifdef YYLTYPE
  #ifndef YY_UPNCalcParser_LTYPE
   #define YY_UPNCalcParser_LTYPE YYLTYPE
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
/* use %define LTYPE */
  #endif
 #endif
/*#ifdef YYSTYPE*/
  #ifndef YY_UPNCalcParser_STYPE
   #define YY_UPNCalcParser_STYPE YYSTYPE
  /* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
   /* use %define STYPE */
  #endif
/*#endif*/
 #ifdef YYDEBUG
  #ifndef YY_UPNCalcParser_DEBUG
   #define  YY_UPNCalcParser_DEBUG YYDEBUG
   /* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
   /* use %define DEBUG */
  #endif
 #endif 
 /* use goto to be compatible */
 #ifndef YY_UPNCalcParser_USE_GOTO
  #define YY_UPNCalcParser_USE_GOTO 1
 #endif
#endif

/* use no goto to be clean in C++ */
#ifndef YY_UPNCalcParser_USE_GOTO
 #define YY_UPNCalcParser_USE_GOTO 0
#endif

#ifndef YY_UPNCalcParser_PURE

 #line 65 "/usr/share/bison++/bison.h"

#line 65 "/usr/share/bison++/bison.h"
/* YY_UPNCalcParser_PURE */
#endif


 #line 68 "/usr/share/bison++/bison.h"

#line 68 "/usr/share/bison++/bison.h"
/* prefix */

#ifndef YY_UPNCalcParser_DEBUG

 #line 71 "/usr/share/bison++/bison.h"

#line 71 "/usr/share/bison++/bison.h"
/* YY_UPNCalcParser_DEBUG */
#endif

#ifndef YY_UPNCalcParser_LSP_NEEDED

 #line 75 "/usr/share/bison++/bison.h"

#line 75 "/usr/share/bison++/bison.h"
 /* YY_UPNCalcParser_LSP_NEEDED*/
#endif

/* DEFAULT LTYPE*/
#ifdef YY_UPNCalcParser_LSP_NEEDED
 #ifndef YY_UPNCalcParser_LTYPE
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

  #define YY_UPNCalcParser_LTYPE yyltype
 #endif
#endif

/* DEFAULT STYPE*/
#ifndef YY_UPNCalcParser_STYPE
 #define YY_UPNCalcParser_STYPE int
#endif

/* DEFAULT MISCELANEOUS */
#ifndef YY_UPNCalcParser_PARSE
 #define YY_UPNCalcParser_PARSE yyparse
#endif

#ifndef YY_UPNCalcParser_LEX
 #define YY_UPNCalcParser_LEX yylex
#endif

#ifndef YY_UPNCalcParser_LVAL
 #define YY_UPNCalcParser_LVAL yylval
#endif

#ifndef YY_UPNCalcParser_LLOC
 #define YY_UPNCalcParser_LLOC yylloc
#endif

#ifndef YY_UPNCalcParser_CHAR
 #define YY_UPNCalcParser_CHAR yychar
#endif

#ifndef YY_UPNCalcParser_NERRS
 #define YY_UPNCalcParser_NERRS yynerrs
#endif

#ifndef YY_UPNCalcParser_DEBUG_FLAG
 #define YY_UPNCalcParser_DEBUG_FLAG yydebug
#endif

#ifndef YY_UPNCalcParser_ERROR
 #define YY_UPNCalcParser_ERROR yyerror
#endif

#ifndef YY_UPNCalcParser_PARSE_PARAM
 #ifndef __STDC__
  #ifndef __cplusplus
   #ifndef YY_USE_CLASS
    #define YY_UPNCalcParser_PARSE_PARAM
    #ifndef YY_UPNCalcParser_PARSE_PARAM_DEF
     #define YY_UPNCalcParser_PARSE_PARAM_DEF
    #endif
   #endif
  #endif
 #endif
 #ifndef YY_UPNCalcParser_PARSE_PARAM
  #define YY_UPNCalcParser_PARSE_PARAM void
 #endif
#endif

/* TOKEN C */
#ifndef YY_USE_CLASS

 #ifndef YY_UPNCalcParser_PURE
  #ifndef yylval
   extern YY_UPNCalcParser_STYPE YY_UPNCalcParser_LVAL;
  #else
   #if yylval != YY_UPNCalcParser_LVAL
    extern YY_UPNCalcParser_STYPE YY_UPNCalcParser_LVAL;
   #else
    #warning "Namespace conflict, disabling some functionality (bison++ only)"
   #endif
  #endif
 #endif


 #line 169 "/usr/share/bison++/bison.h"
#define	UNKNOWN	258
#define	PLUS	259
#define	MINUS	260
#define	NUMBER	261


#line 169 "/usr/share/bison++/bison.h"
 /* #defines token */
/* after #define tokens, before const tokens S5*/
#else
 #ifndef YY_UPNCalcParser_CLASS
  #define YY_UPNCalcParser_CLASS UPNCalcParser
 #endif

 #ifndef YY_UPNCalcParser_INHERIT
  #define YY_UPNCalcParser_INHERIT
 #endif

 #ifndef YY_UPNCalcParser_MEMBERS
  #define YY_UPNCalcParser_MEMBERS 
 #endif

 #ifndef YY_UPNCalcParser_LEX_BODY
  #define YY_UPNCalcParser_LEX_BODY  
 #endif

 #ifndef YY_UPNCalcParser_ERROR_BODY
  #define YY_UPNCalcParser_ERROR_BODY  
 #endif

 #ifndef YY_UPNCalcParser_CONSTRUCTOR_PARAM
  #define YY_UPNCalcParser_CONSTRUCTOR_PARAM
 #endif
 /* choose between enum and const */
 #ifndef YY_UPNCalcParser_USE_CONST_TOKEN
  #define YY_UPNCalcParser_USE_CONST_TOKEN 0
  /* yes enum is more compatible with flex,  */
  /* so by default we use it */ 
 #endif
 #if YY_UPNCalcParser_USE_CONST_TOKEN != 0
  #ifndef YY_UPNCalcParser_ENUM_TOKEN
   #define YY_UPNCalcParser_ENUM_TOKEN yy_UPNCalcParser_enum_token
  #endif
 #endif

class YY_UPNCalcParser_CLASS YY_UPNCalcParser_INHERIT
{
public: 
 #if YY_UPNCalcParser_USE_CONST_TOKEN != 0
  /* static const int token ... */
  
 #line 212 "/usr/share/bison++/bison.h"
static const int UNKNOWN;
static const int PLUS;
static const int MINUS;
static const int NUMBER;


#line 212 "/usr/share/bison++/bison.h"
 /* decl const */
 #else
  enum YY_UPNCalcParser_ENUM_TOKEN { YY_UPNCalcParser_NULL_TOKEN=0
  
 #line 215 "/usr/share/bison++/bison.h"
	,UNKNOWN=258
	,PLUS=259
	,MINUS=260
	,NUMBER=261


#line 215 "/usr/share/bison++/bison.h"
 /* enum token */
     }; /* end of enum declaration */
 #endif
public:
 int YY_UPNCalcParser_PARSE(YY_UPNCalcParser_PARSE_PARAM);
 virtual void YY_UPNCalcParser_ERROR(char *msg) YY_UPNCalcParser_ERROR_BODY;
 #ifdef YY_UPNCalcParser_PURE
  #ifdef YY_UPNCalcParser_LSP_NEEDED
   virtual int  YY_UPNCalcParser_LEX(YY_UPNCalcParser_STYPE *YY_UPNCalcParser_LVAL,YY_UPNCalcParser_LTYPE *YY_UPNCalcParser_LLOC) YY_UPNCalcParser_LEX_BODY;
  #else
   virtual int  YY_UPNCalcParser_LEX(YY_UPNCalcParser_STYPE *YY_UPNCalcParser_LVAL) YY_UPNCalcParser_LEX_BODY;
  #endif
 #else
  virtual int YY_UPNCalcParser_LEX() YY_UPNCalcParser_LEX_BODY;
  YY_UPNCalcParser_STYPE YY_UPNCalcParser_LVAL;
  #ifdef YY_UPNCalcParser_LSP_NEEDED
   YY_UPNCalcParser_LTYPE YY_UPNCalcParser_LLOC;
  #endif
  int YY_UPNCalcParser_NERRS;
  int YY_UPNCalcParser_CHAR;
 #endif
 #if YY_UPNCalcParser_DEBUG != 0
  public:
   int YY_UPNCalcParser_DEBUG_FLAG;	/*  nonzero means print parse trace	*/
 #endif
public:
 YY_UPNCalcParser_CLASS(YY_UPNCalcParser_CONSTRUCTOR_PARAM);
public:
 YY_UPNCalcParser_MEMBERS 
};
/* other declare folow */
#endif


#if YY_UPNCalcParser_COMPATIBILITY != 0
 /* backward compatibility */
 /* Removed due to bison problems
 /#ifndef YYSTYPE
 / #define YYSTYPE YY_UPNCalcParser_STYPE
 /#endif*/

 #ifndef YYLTYPE
  #define YYLTYPE YY_UPNCalcParser_LTYPE
 #endif
 #ifndef YYDEBUG
  #ifdef YY_UPNCalcParser_DEBUG 
   #define YYDEBUG YY_UPNCalcParser_DEBUG
  #endif
 #endif

#endif
/* END */

 #line 267 "/usr/share/bison++/bison.h"
#endif
