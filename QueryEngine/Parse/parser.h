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
    void parse(const string & inputStr, ASTNode *& parseRoot) { istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseRoot); } \
    private:                   \
       yyFlexLexer lexer;
#define YY_Parser_LEX_BODY  {return lexer.yylex();}
#define YY_Parser_ERROR_BODY  {cerr << "error encountered at line: "<<lexer.lineno()<<" last word parsed:"<<lexer.YYText()<<"\n";}
#line 11 "parser.y"

#include <iostream>
#include <fstream>
#include <FlexLexer.h>
#include <cstdlib>
#include <string>
#include <sstream>

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

#include "ast/InsertStatement.h"
#include "ast/OptColumnCommalist.h"
#include "ast/ValuesOrQuerySpec.h"
#include "ast/QuerySpec.h"
#include "ast/InsertAtomCommalist.h"
#include "ast/InsertAtom.h"
#include "ast/Atom.h"

#include "ast/SearchCondition.h"
#include "ast/ScalarExpCommalist.h"
#include "ast/ScalarExp.h"
#include "ast/FunctionRef.h"
#include "ast/Ammsc.h"
#include "ast/Predicate.h"
#include "ast/ComparisonPredicate.h"
#include "ast/BetweenPredicate.h"
#include "ast/LikePredicate.h"
#include "ast/OptEscape.h"
#include "ast/ColumnRef.h"

#include "ast/ColumnRefCommalist.h"
#include "ast/OptWhereClause.h"
#include "ast/OptGroupByClause.h"
#include "ast/OptHavingClause.h"
#include "ast/OptLimitClause.h"
#include "ast/OptAscDesc.h"
#include "ast/OrderingSpecCommalist.h"
#include "ast/OrderingSpec.h"
#include "ast/OptOrderByClause.h"

#include "ast/UpdateStatementSearched.h"
#include "ast/UpdateStatementPositioned.h"
#include "ast/AssignmentCommalist.h"
#include "ast/Assignment.h"
#include "ast/Cursor.h"

#include "ast/TestForNull.h"
#include "ast/InPredicate.h"
#include "ast/ExistenceTest.h"
#include "ast/AllOrAnyPredicate.h"
#include "ast/AnyAllSome.h"
#include "ast/AtomCommalist.h"
#include "ast/Subquery.h"


// define stack element type to be a 
// pointer to an AST node
#define YY_Parser_STYPE ASTNode*
#define YY_Parser_PARSE_PARAM ASTNode*& parseRoot

extern ASTNode* parse_root;

// Variables declared in scanner.l
extern std::string strData[10];
extern double dData[10];

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
#define	OR	258
#define	AND	259
#define	NOT	260
#define	COMPARISON	261
#define	UMINUS	262
#define	AS	263
#define	DROP	264
#define	NAME	265
#define	TABLE	266
#define	CREATE	267
#define	INTNUM	268
#define	STRING	269
#define	APPROXNUM	270
#define	UNKNOWN	271
#define	ALL	272
#define	BETWEEN	273
#define	BY	274
#define	DISTINCT	275
#define	FROM	276
#define	GROUP	277
#define	HAVING	278
#define	SELECT	279
#define	USER	280
#define	WHERE	281
#define	WITH	282
#define	EMPTY	283
#define	SELALL	284
#define	DOT	285
#define	UPDATE	286
#define	SET	287
#define	CURRENT	288
#define	OF	289
#define	NULLX	290
#define	ASSIGN	291
#define	INSERT	292
#define	INTO	293
#define	VALUES	294
#define	UNIQUE	295
#define	PRIMARY	296
#define	FOREIGN	297
#define	KEY	298
#define	CHECK	299
#define	REFERENCES	300
#define	DEFAULT	301
#define	DATATYPE	302
#define	DECIMAL	303
#define	SMALLINT	304
#define	NUMERIC	305
#define	CHARACTER	306
#define	INTEGER	307
#define	REAL	308
#define	FLOAT	309
#define	DOUBLE	310
#define	PRECISION	311
#define	VARCHAR	312
#define	AVG	313
#define	MAX	314
#define	MIN	315
#define	SUM	316
#define	COUNT	317
#define	ALIAS	318
#define	INTORDER	319
#define	COLORDER	320
#define	ORDER	321
#define	ASC	322
#define	DESC	323
#define	LIMIT	324
#define	OFFSET	325
#define	DOTNAME	326
#define	ESCAPE	327
#define	LIKE	328
#define	IS	329
#define	IN	330
#define	ANY	331
#define	SOME	332
#define	EXISTS	333


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
static const int OR;
static const int AND;
static const int NOT;
static const int COMPARISON;
static const int UMINUS;
static const int AS;
static const int DROP;
static const int NAME;
static const int TABLE;
static const int CREATE;
static const int INTNUM;
static const int STRING;
static const int APPROXNUM;
static const int UNKNOWN;
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
static const int ESCAPE;
static const int LIKE;
static const int IS;
static const int IN;
static const int ANY;
static const int SOME;
static const int EXISTS;


#line 212 "/usr/share/bison++/bison.h"
 /* decl const */
 #else
  enum YY_Parser_ENUM_TOKEN { YY_Parser_NULL_TOKEN=0
  
 #line 215 "/usr/share/bison++/bison.h"
	,OR=258
	,AND=259
	,NOT=260
	,COMPARISON=261
	,UMINUS=262
	,AS=263
	,DROP=264
	,NAME=265
	,TABLE=266
	,CREATE=267
	,INTNUM=268
	,STRING=269
	,APPROXNUM=270
	,UNKNOWN=271
	,ALL=272
	,BETWEEN=273
	,BY=274
	,DISTINCT=275
	,FROM=276
	,GROUP=277
	,HAVING=278
	,SELECT=279
	,USER=280
	,WHERE=281
	,WITH=282
	,EMPTY=283
	,SELALL=284
	,DOT=285
	,UPDATE=286
	,SET=287
	,CURRENT=288
	,OF=289
	,NULLX=290
	,ASSIGN=291
	,INSERT=292
	,INTO=293
	,VALUES=294
	,UNIQUE=295
	,PRIMARY=296
	,FOREIGN=297
	,KEY=298
	,CHECK=299
	,REFERENCES=300
	,DEFAULT=301
	,DATATYPE=302
	,DECIMAL=303
	,SMALLINT=304
	,NUMERIC=305
	,CHARACTER=306
	,INTEGER=307
	,REAL=308
	,FLOAT=309
	,DOUBLE=310
	,PRECISION=311
	,VARCHAR=312
	,AVG=313
	,MAX=314
	,MIN=315
	,SUM=316
	,COUNT=317
	,ALIAS=318
	,INTORDER=319
	,COLORDER=320
	,ORDER=321
	,ASC=322
	,DESC=323
	,LIMIT=324
	,OFFSET=325
	,DOTNAME=326
	,ESCAPE=327
	,LIKE=328
	,IS=329
	,IN=330
	,ANY=331
	,SOME=332
	,EXISTS=333


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
