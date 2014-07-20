#ifndef YY_Parser_h_included
#define YY_Parser_h_included
#define YY_USE_CLASS

#line 1 "/usr/share/bison++/bison.h"
/* before anything */
#ifdef c_plusplus
 #ifndef __cplusplus
  #define __cplusplus
 #endif
#endif


 #line 8 "/usr/share/bison++/bison.h"
#define YY_Parser_CLASS  RAParser
#define YY_Parser_LSP_NEEDED 
#define YY_Parser_MEMBERS                  \
    virtual ~RAParser()   {} \
    int parse(const string & inputStr, RelAlgNode *& parseRoot, string &lastParsed) { istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseRoot); lastParsed = lexer.YYText(); return yynerrs; } \
    private:                   \
       yyFlexLexer lexer;
#define YY_Parser_LEX_BODY  {return lexer.yylex();}
#define YY_Parser_ERROR_BODY  {cerr << "Syntax error on line " << lexer.lineno() << ". Last word parsed:" << lexer.YYText() << endl;}
#line 12 "parser.y"

#include <iostream>
#include <fstream>
#include <FlexLexer.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <cassert>

// RA Parse Tree Nodes
#include "ast/RelAlgNode.h"
#include "ast/UnaryOp.h"
#include "ast/BinaryOp.h"

#include "ast/AggrExpr.h"
#include "ast/AggrList.h"
#include "ast/AntijoinOp.h"
#include "ast/Attribute.h"
#include "ast/AttrList.h"
#include "ast/Comparison.h"
#include "ast/DiffOp.h"
#include "ast/Expr.h"
#include "ast/ExtendOp.h"
#include "ast/GroupbyOp.h"
#include "ast/JoinOp.h"
#include "ast/MathExpr.h"
#include "ast/OuterjoinOp.h"
#include "ast/Predicate.h"
#include "ast/ProductOp.h"
#include "ast/Program.h"
#include "ast/ProjectOp.h"
#include "ast/Relation.h"
#include "ast/RelExpr.h"
#include "ast/RelExprList.h"
#include "ast/RenameOp.h"
#include "ast/SelectOp.h"
#include "ast/SemijoinOp.h"
#include "ast/SortOp.h"
#include "ast/UnionOp.h"

using namespace std;
using namespace RA_Namespace;

extern RelAlgNode* parse_root;

// define stack element type to be a 
// pointer to an AST node	
#define YY_Parser_STYPE RelAlgNode*
#define YY_Parser_PARSE_PARAM RelAlgNode*& parseRoot

// Variables declared in RelAlgebraLexer.l
extern std::vector<std::string> strData;
extern std::vector<int> intData;
extern std::vector<float> floatData;


#line 21 "/usr/share/bison++/bison.h"
 /* %{ and %header{ and %union, during decl */
#ifndef YY_Parser_COMPATIBILITY
 #ifndef YY_USE_CLASS
  #define  YY_Parser_COMPATIBILITY 1
 #else
  #define  YY_Parser_COMPATIBILITY 0
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
/*#ifdef YYSTYPE*/
  #ifndef YY_Parser_STYPE
   #define YY_Parser_STYPE YYSTYPE
  /* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
   /* use %define STYPE */
  #endif
/*#endif*/
 #ifdef YYDEBUG
  #ifndef YY_Parser_DEBUG
   #define  YY_Parser_DEBUG YYDEBUG
   /* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
   /* use %define DEBUG */
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

 #line 65 "/usr/share/bison++/bison.h"

#line 65 "/usr/share/bison++/bison.h"
/* YY_Parser_PURE */
#endif


 #line 68 "/usr/share/bison++/bison.h"
#ifndef YY_USE_CLASS
# ifndef YYSTYPE
#  define YYSTYPE int
#  define YYSTYPE_IS_TRIVIAL 1
# endif
#endif

#line 68 "/usr/share/bison++/bison.h"
/* prefix */

#ifndef YY_Parser_DEBUG

 #line 71 "/usr/share/bison++/bison.h"

#line 71 "/usr/share/bison++/bison.h"
/* YY_Parser_DEBUG */
#endif

#ifndef YY_Parser_LSP_NEEDED

 #line 75 "/usr/share/bison++/bison.h"

#line 75 "/usr/share/bison++/bison.h"
 /* YY_Parser_LSP_NEEDED*/
#endif

/* DEFAULT LTYPE*/
#ifdef YY_Parser_LSP_NEEDED
 #ifndef YY_Parser_LTYPE
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
  #ifndef yylval
   extern YY_Parser_STYPE YY_Parser_LVAL;
  #else
   #if yylval != YY_Parser_LVAL
    extern YY_Parser_STYPE YY_Parser_LVAL;
   #else
    #warning "Namespace conflict, disabling some functionality (bison++ only)"
   #endif
  #endif
 #endif


 #line 169 "/usr/share/bison++/bison.h"
#define	PLUS	258
#define	MINUS	259
#define	MULTIPLY	260
#define	DIVIDE	261
#define	OR	262
#define	AND	263
#define	NOT	264
#define	NEQ	265
#define	EQ	266
#define	GT	267
#define	GTE	268
#define	LT	269
#define	LTE	270
#define	SELECT	271
#define	PROJECT	272
#define	SORT	273
#define	RENAME	274
#define	EXTEND	275
#define	GROUPBY	276
#define	PRODUCT	277
#define	JOIN	278
#define	SEMIJOIN	279
#define	ANTIJOIN	280
#define	OUTERJOIN	281
#define	UNION	282
#define	DIFF	283
#define	INTERSECTION	284
#define	MAX	285
#define	MIN	286
#define	COUNT	287
#define	SUM	288
#define	AVG	289
#define	MAX_DISTINCT	290
#define	MIN_DISTINCT	291
#define	COUNT_DISTINCT	292
#define	SUM_DISTINCT	293
#define	AVG_DISTINCT	294
#define	NAME	295
#define	INTVAL	296
#define	FLOATVAL	297
#define	STRVAL	298


#line 169 "/usr/share/bison++/bison.h"
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

class YY_Parser_CLASS YY_Parser_INHERIT
{
public: 
 #if YY_Parser_USE_CONST_TOKEN != 0
  /* static const int token ... */
  
 #line 212 "/usr/share/bison++/bison.h"
static const int PLUS;
static const int MINUS;
static const int MULTIPLY;
static const int DIVIDE;
static const int OR;
static const int AND;
static const int NOT;
static const int NEQ;
static const int EQ;
static const int GT;
static const int GTE;
static const int LT;
static const int LTE;
static const int SELECT;
static const int PROJECT;
static const int SORT;
static const int RENAME;
static const int EXTEND;
static const int GROUPBY;
static const int PRODUCT;
static const int JOIN;
static const int SEMIJOIN;
static const int ANTIJOIN;
static const int OUTERJOIN;
static const int UNION;
static const int DIFF;
static const int INTERSECTION;
static const int MAX;
static const int MIN;
static const int COUNT;
static const int SUM;
static const int AVG;
static const int MAX_DISTINCT;
static const int MIN_DISTINCT;
static const int COUNT_DISTINCT;
static const int SUM_DISTINCT;
static const int AVG_DISTINCT;
static const int NAME;
static const int INTVAL;
static const int FLOATVAL;
static const int STRVAL;


#line 212 "/usr/share/bison++/bison.h"
 /* decl const */
 #else
  enum YY_Parser_ENUM_TOKEN { YY_Parser_NULL_TOKEN=0
  
 #line 215 "/usr/share/bison++/bison.h"
	,PLUS=258
	,MINUS=259
	,MULTIPLY=260
	,DIVIDE=261
	,OR=262
	,AND=263
	,NOT=264
	,NEQ=265
	,EQ=266
	,GT=267
	,GTE=268
	,LT=269
	,LTE=270
	,SELECT=271
	,PROJECT=272
	,SORT=273
	,RENAME=274
	,EXTEND=275
	,GROUPBY=276
	,PRODUCT=277
	,JOIN=278
	,SEMIJOIN=279
	,ANTIJOIN=280
	,OUTERJOIN=281
	,UNION=282
	,DIFF=283
	,INTERSECTION=284
	,MAX=285
	,MIN=286
	,COUNT=287
	,SUM=288
	,AVG=289
	,MAX_DISTINCT=290
	,MIN_DISTINCT=291
	,COUNT_DISTINCT=292
	,SUM_DISTINCT=293
	,AVG_DISTINCT=294
	,NAME=295
	,INTVAL=296
	,FLOATVAL=297
	,STRVAL=298


#line 215 "/usr/share/bison++/bison.h"
 /* enum token */
     }; /* end of enum declaration */
 #endif
public:
 int YY_Parser_PARSE(YY_Parser_PARSE_PARAM);
 virtual void YY_Parser_ERROR(char *msg) YY_Parser_ERROR_BODY;
 #ifdef YY_Parser_PURE
  #ifdef YY_Parser_LSP_NEEDED
   virtual int  YY_Parser_LEX(YY_Parser_STYPE *YY_Parser_LVAL,YY_Parser_LTYPE *YY_Parser_LLOC) YY_Parser_LEX_BODY;
  #else
   virtual int  YY_Parser_LEX(YY_Parser_STYPE *YY_Parser_LVAL) YY_Parser_LEX_BODY;
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
   int YY_Parser_DEBUG_FLAG;	/*  nonzero means print parse trace	*/
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
 /* Removed due to bison problems
 /#ifndef YYSTYPE
 / #define YYSTYPE YY_Parser_STYPE
 /#endif*/

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

 #line 267 "/usr/share/bison++/bison.h"
#endif
