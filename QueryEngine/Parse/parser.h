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
#define YY_Parser_LSP_NEEDED 
#define YY_Parser_MEMBERS                  \
    virtual ~Parser()   {} \
    private:                   \
       yyFlexLexer lexer;
#define YY_Parser_LEX_BODY  {return lexer.yylex();}
#define YY_Parser_ERROR_BODY  {cerr << "error encountered at line: "<<lexer.lineno()<<" last word parsed:"<<lexer.YYText()<<"\n";}
#line 10 "parser.y"

#include <iostream>
#include <fstream>
#include <FlexLexer.h>
#include <cstdlib>

// AST nodes
#include "ast/ASTNode.h"
#include "ast/Program.h"
#include "ast/SQLList.h"
#include "ast/SQL.h"
#include "ast/Schema.h"
#include "ast/BaseTableDef.h"
#include "ast/Table.h"
#include "ast/ColumnDef.h"
#include "ast/ColumnCommalist.h"
#include "ast/TableConstraintDef.h"
#include "ast/BaseTableElementCommalist.h"
#include "ast/BaseTableElement.h"
#include "ast/ColumnDefOpt.h"
#include "ast/ColumnDefOptList.h"
#include "ast/Literal.h"
#include "ast/DataType.h"
#include "ast/Column.h"

#include "ast/ManipulativeStatement.h"
#include "ast/SelectStatement.h"
#include "ast/Selection.h"
#include "ast/OptAllDistinct.h"
#include "ast/TableExp.h"
#include "ast/FromClause.h"
#include "ast/TableRefCommalist.h"
#include "ast/TableRef.h"


// define stack element type to be a 
// pointer to an AST node
#define YY_Parser_STYPE ASTNode*

extern ASTNode* parse_root;

// Variables declared in scanner.l
extern std::string strData[10];
extern int intData;

using namespace std;


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
#define	AS	258
#define	DROP	259
#define	NAME	260
#define	TABLE	261
#define	CREATE	262
#define	INTNUM	263
#define	STRING	264
#define	UNKNOWN	265
#define	ALL	266
#define	BETWEEN	267
#define	BY	268
#define	DISTINCT	269
#define	FROM	270
#define	GROUP	271
#define	HAVING	272
#define	SELECT	273
#define	COMPARISON	274
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
#define	NOT	290
#define	UNIQUE	291
#define	PRIMARY	292
#define	FOREIGN	293
#define	KEY	294
#define	CHECK	295
#define	REFERENCES	296
#define	DEFAULT	297
#define	DATATYPE	298
#define	DECIMAL	299
#define	SMALLINT	300
#define	NUMERIC	301
#define	CHARACTER	302
#define	INTEGER	303
#define	REAL	304
#define	FLOAT	305
#define	DOUBLE	306
#define	PRECISION	307
#define	VARCHAR	308
#define	AVG	309
#define	MAX	310
#define	MIN	311
#define	SUM	312
#define	COUNT	313
#define	ALIAS	314
#define	INTORDER	315
#define	COLORDER	316
#define	ORDER	317
#define	ASC	318
#define	DESC	319
#define	LIMIT	320
#define	OFFSET	321
#define	DOTNAME	322


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
static const int AS;
static const int DROP;
static const int NAME;
static const int TABLE;
static const int CREATE;
static const int INTNUM;
static const int STRING;
static const int UNKNOWN;
static const int ALL;
static const int BETWEEN;
static const int BY;
static const int DISTINCT;
static const int FROM;
static const int GROUP;
static const int HAVING;
static const int SELECT;
static const int COMPARISON;
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
static const int NOT;
static const int UNIQUE;
static const int PRIMARY;
static const int FOREIGN;
static const int KEY;
static const int CHECK;
static const int REFERENCES;
static const int DEFAULT;
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
static const int AVG;
static const int MAX;
static const int MIN;
static const int SUM;
static const int COUNT;
static const int ALIAS;
static const int INTORDER;
static const int COLORDER;
static const int ORDER;
static const int ASC;
static const int DESC;
static const int LIMIT;
static const int OFFSET;
static const int DOTNAME;


#line 212 "/usr/share/bison++/bison.h"
 /* decl const */
 #else
  enum YY_Parser_ENUM_TOKEN { YY_Parser_NULL_TOKEN=0
  
 #line 215 "/usr/share/bison++/bison.h"
	,AS=258
	,DROP=259
	,NAME=260
	,TABLE=261
	,CREATE=262
	,INTNUM=263
	,STRING=264
	,UNKNOWN=265
	,ALL=266
	,BETWEEN=267
	,BY=268
	,DISTINCT=269
	,FROM=270
	,GROUP=271
	,HAVING=272
	,SELECT=273
	,COMPARISON=274
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
	,NOT=290
	,UNIQUE=291
	,PRIMARY=292
	,FOREIGN=293
	,KEY=294
	,CHECK=295
	,REFERENCES=296
	,DEFAULT=297
	,DATATYPE=298
	,DECIMAL=299
	,SMALLINT=300
	,NUMERIC=301
	,CHARACTER=302
	,INTEGER=303
	,REAL=304
	,FLOAT=305
	,DOUBLE=306
	,PRECISION=307
	,VARCHAR=308
	,AVG=309
	,MAX=310
	,MIN=311
	,SUM=312
	,COUNT=313
	,ALIAS=314
	,INTORDER=315
	,COLORDER=316
	,ORDER=317
	,ASC=318
	,DESC=319
	,LIMIT=320
	,OFFSET=321
	,DOTNAME=322


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
