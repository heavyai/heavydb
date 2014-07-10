%name RelAlgebraParser
#define LSP_NEEDED
%define MEMBERS                 \
    virtual ~Parser()   {} \
    void parse(const string & inputStr, ASTNode *& parseRoot) { istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseRoot); } \
    private:                   \
       yyFlexLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {cerr << "error encountered at line: "<<lexer.lineno()<<" last word parsed:"<<lexer.YYText()<<"\n";}

%header{
#include <iostream>
#include <fstream>
#include <FlexLexer.h>
#include <cstdlib>

// AST nodes
#include "ast/ASTNode.h"

// define stack element type to be a 
// pointer to an AST node
#define YY_Parser_STYPE ASTNode*
#define YY_Parser_PARSE_PARAM ASTNode*& parseRoot

// Variables declared in RelAlgebraLexer.l
extern std::string strData[10];

using namespace std;

%}

%left PLUS MINUS
%left MULTIPLY DIVIDE

%token NEQ EQ GT GTE LT LTE
%token SELECT PROJECT SORT RENAME EXTEND GROUPBY
%token PRODUCT JOIN SEMIJOIN ANTIJOIN OUTERJOIN UNION
%token MAX MIN COUNT SUM AVG
%token MAX_DISTINCT MIN_DISTINCT COUNT_DISTINCT SUM_DISTINCT AVG_DISTINCT 
%token NAME CONSTANT STRVAL

%start S

%%

S:
	RelExprList
|		
;

RelExprList:
	RelExpr ';'
|	RelExprList RelExpr ';'
;

RelExpr:
	UnaryOp
|	BinaryOp
| 	'(' RelExpr ')'
| 	relation
;

UnaryOp:
|	SELECT 	'(' RelExpr ',' Predicate ')'
|	PROJECT '(' RelExpr ',' '{' AttrList '}' ')'
|	SORT	'(' RelExpr ',' AttrList ')'
|	RENAME	'(' RelExpr ',' attribute ',' NAME ')'
|	EXTEND	'(' RelExpr ',' MathExpr ',' NAME ')'
|	EXTEND	'(' RelExpr ',' data ',' NAME ')'
|	GROUPBY	'('	RelExpr ',' '{' AttrList '}' ',' AggrList ')'
|	GROUPBY	'('	RelExpr ',' '{' '}' ',' AggrList ')'

BinaryOp:
	PRODUCT '(' RelExpr ',' RelExpr ')'
|	JOIN	'(' RelExpr ',' RelExpr ',' Predicate ')'	/* 'join' is shorthand for 'selection of product' */
|	SEMIJOIN (' RelExpr ',' RelExpr ',' Predicate ')
|	ANTIJOIN '(' RelExpr ',' RelExpr ',' Predicate ')'
|	OUTERJOIN '(' RelExpr ',' RelExpr ',' Predicate ')'
|	UNION	'(' RelExpr ',' RelExpr ')'
;

MathExpr:
	MathExpr PLUS MathExpr
|	MathExpr MINUS MathExpr
|	MathExpr MULTIPLY MathExpr
|	MathExpr DIVIDE MathExpr
|	'(' MathExpr ')'
|	attribute

AggrList:
	AggrExpr
|	AggrExpr ',' AggrExpr
;

AggrExpr:
	MAX '(' attribute ')'	
|	MIN '(' attribute ')'
|	COUNT '(' attribute ')'
|	SUM '(' attribute ')'
|	AVG '(' attribute ')'
|	MAX_DISTINCT '(' attribute ')'	
|	MIN_DISTINCT '(' attribute ')'
|	COUNT_DISTINCT '(' attribute ')'
|	SUM_DISTINCT '(' attribute ')'
|	AVG_DISTINCT '(' attribute ')'
;

AttrList:
  attribute
| attribute ',' AttrList
;

attribute: 	NAME | fullname;

Predicate:
  Predicate OR Predicate
| Predicate AND Predicate
| NOT Predicate
| '(' Predicate ')'
| compared CompOp compared
;

CompOp:	NEQ | EQ | GT | GTE | LT | LTE;
compared: 	attribute | data;

fullname: 	NAME '.' NAME;
data: 		CONSTANT | STRVAL;

RelationList:
	relation
|	relation ',' relation
;

relation: 	NAME;

%%