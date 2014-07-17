%name Parser
%define LSP_NEEDED
%define MEMBERS                 \
    virtual ~Parser()   {} \
    void parse(const string & inputStr, RelAlgNode *& parseRoot) { istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseRoot); } \
    private:                   \
       yyFlexLexer lexer; 
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {cerr << "error encountered at line: "<<lexer.lineno()<<" last word parsed:"<<lexer.YYText()<<"\n";}

%header{
#include <iostream>
#include <fstream>
#include <FlexLexer.h>
#include <cstdlib>
#include <string>
#include <sstream>
	
// RA nodes
#include "relAlg/RelAlgNode.h"
#include "relAlg/RA_Program.h"
#include "relAlg/RelExprList.h"
#include "relAlg/RelExpr.h"
#include "relAlg/UnaryOp.h"
#include "relAlg/BinaryOp.h"
#include "relAlg/MathExpr.h"
#include "relAlg/SelectOp.h"
#include "relAlg/ProjectOp.h"
#include "relAlg/SortOp.h"
#include "relAlg/ExtendOp.h"
#include "relAlg/GroupByOp.h"
#include "relAlg/RenameOp.h"

#include "relAlg/JoinOp.h"
#include "relAlg/SemijoinOp.h"
#include "relAlg/ProductOp.h"
#include "relAlg/OuterjoinOp.h"
#include "relAlg/AntijoinOp.h"
#include "relAlg/UnionOp.h"
#include "relAlg/AggrExpr.h"
#include "relAlg/AggrList.h"
#include "relAlg/AttrList.h"
#include "relAlg/Attribute.h"
#include "relAlg/Relation.h"
#include "relAlg/Data.h"

#include "relAlg/RA_Predicate.h"
#include "relAlg/Comparison.h"
#include "relAlg/Compared.h"
#include "relAlg/CompOp.h"

// define stack element type to be a 
// pointer to an AST node
	
#define YY_Parser_STYPE RelAlgNode*
#define YY_Parser_PARSE_PARAM RelAlgNode*& parseRoot

extern RelAlgNode* parse_root;


// Variables declared in RelAlgebraLexer.l
extern std::string strData[10];
extern int dData[5];

using namespace std;
using namespace RA_Namespace;

%}

%left PLUS MINUS
%left MULTIPLY DIVIDE
%left OR
%left AND
%left NOT

%left NEQ EQ GT GTE LT LTE
%token SELECT PROJECT SORT RENAME EXTEND GROUPBY
%token PRODUCT JOIN SEMIJOIN ANTIJOIN OUTERJOIN UNION
%token MAX MIN COUNT SUM AVG
%token MAX_DISTINCT MIN_DISTINCT COUNT_DISTINCT SUM_DISTINCT AVG_DISTINCT 
%token NAME CONSTANT STRVAL

%start S

%%

S:
	RelExprList				{ $$ = new RA_Program((RelExprList*)$1); parseRoot = $$; }
|							{ $$ = 0; parseRoot = $$; }
;

RelExprList:
	RelExpr ';'					{ $$ = new RelExprList((RelExpr*)$1); }
|	RelExprList RelExpr ';'		{ $$ = new RelExprList((RelExprList*)$1, (RelExpr*)$3); }
;

RelExpr:
	UnaryOp						{ $$ = new RelExpr((UnaryOp*)$1); }
|	BinaryOp					{ $$ = new RelExpr((BinaryOp*)$1); }
| 	'(' RelExpr ')'				{ $$ = new RelExpr((RelExpr*)$2); }
| 	relation					{ $$ = new RelExpr((Relation*)$1); }
;

UnaryOp:
	SELECT 	'(' RelExpr ',' Predicate ')'						{ $$ = new SelectOp((RelExpr*)$3, (RA_Predicate*)$5); }
|	PROJECT '(' RelExpr ',' '{' AttrList '}' ')'				{ $$ = new ProjectOp((RelExpr*)$3, (AttrList*)$6); }
|	SORT	'(' RelExpr ',' AttrList ')'						{ $$ = new SortOp((RelExpr*)$3, (AttrList*)$5); }
|	RENAME	'(' RelExpr ',' attribute ',' NAME ')'				{ $$ = new RenameOp((RelExpr*)$3, (Attribute*)$5, strData[0]); }
|	EXTEND	'(' RelExpr ',' MathExpr ',' NAME ')'				{ $$ = new ExtendOp((RelExpr*)$3, (MathExpr*)$5, strData[0]); }
//|	EXTEND	'(' RelExpr ',' data ',' NAME ')'					{ $$ = new ExtendOp((RelExpr*)$3, (Data*)$5, strData[0]); }
|	GROUPBY	'('	RelExpr ',' '{' AttrList '}' ',' AggrList ')'	{ $$ = new GroupByOp((RelExpr*)$3, (AttrList*)$6, (AggrList*)$9); }
|	GROUPBY	'('	RelExpr ',' '{' '}' ',' AggrList ')'			{ $$ = new GroupByOp((RelExpr*)$3, (AggrList*)$8); }
;

BinaryOp:
	PRODUCT '(' RelExpr ',' RelExpr ')'						{ $$ = new ProductOp((RelExpr*)$3, (RelExpr*)$5); }
|	JOIN	'(' RelExpr ',' RelExpr ',' Predicate ')'		{ $$ = new JoinOp((RelExpr*)$3, (RelExpr*)$5, (RA_Predicate*)$7); }
|	SEMIJOIN '(' RelExpr ',' RelExpr ',' Predicate ')'		{ $$ = new SemijoinOp((RelExpr*)$3, (RelExpr*)$5, (RA_Predicate*)$7); }
|	OUTERJOIN '(' RelExpr ',' RelExpr ',' Predicate ')'		{ $$ = new OuterjoinOp((RelExpr*)$3, (RelExpr*)$5, (RA_Predicate*)$7); }
|	ANTIJOIN '(' RelExpr ',' RelExpr ',' Predicate ')'		{ $$ = new AntijoinOp((RelExpr*)$3, (RelExpr*)$5, (RA_Predicate*)$7); }
|	UNION	'(' RelExpr ',' RelExpr ')'						{ $$ = new UnionOp((RelExpr*)$3, (RelExpr*)$5); }
;

MathExpr:
	MathExpr PLUS MathExpr									{ $$ = new MathExpr(1, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr MINUS MathExpr									{ $$ = new MathExpr(2, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr MULTIPLY MathExpr								{ $$ = new MathExpr(3, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr DIVIDE MathExpr								{ $$ = new MathExpr(4, (MathExpr*)$1, (MathExpr*)$3); }
|	'(' MathExpr ')'										{ $$ = new MathExpr(0, (MathExpr*)$2); }
//|	'-' MathExpr %prec UMINUS								{ $$ = new MathExpr(5, (MathExpr*)$2); }
|	attribute												{ $$ = new MathExpr((Attribute*)$1); }
|	AggrExpr												{ $$ = new MathExpr((AggrExpr*)$1); }
|	data													{ $$ = new MathExpr((Data*)$1); }
;

AggrList:
	AggrExpr												{ $$ = new AggrList((AggrExpr*)$1); }
|	AggrList ',' AggrExpr									{ $$ = new AggrList((AggrList*)$1, (AggrExpr*)$3); }
;

AggrExpr:
	MAX '(' attribute ')'									{ $$ = new AggrExpr("MAX", "", (Attribute*)$3); }
|	MIN '(' attribute ')'									{ $$ = new AggrExpr("MIN", "", (Attribute*)$3); }
|	COUNT '(' attribute ')'									{ $$ = new AggrExpr("COUNT", "", (Attribute*)$3); }
|	SUM '(' attribute ')'									{ $$ = new AggrExpr("SUM", "", (Attribute*)$3); }
|	AVG '(' attribute ')'									{ $$ = new AggrExpr("AVG", "", (Attribute*)$3); }
|	MAX_DISTINCT '(' attribute ')'							{ $$ = new AggrExpr("MAX", "DISTINCT", (Attribute*)$3); }
|	MIN_DISTINCT '(' attribute ')'							{ $$ = new AggrExpr("MIN", "DISTINCT", (Attribute*)$3); }
|	COUNT_DISTINCT '(' attribute ')'						{ $$ = new AggrExpr("COUNT", "DISTINCT", (Attribute*)$3); }
|	SUM_DISTINCT '(' attribute ')'							{ $$ = new AggrExpr("SUM", "DISTINCT", (Attribute*)$3); }
|	AVG_DISTINCT '(' attribute ')'							{ $$ = new AggrExpr("AVG", "DISTINCT", (Attribute*)$3); }
;

AttrList:
  attribute													{ $$ = new AttrList((Attribute*)$1); }						
| AttrList ',' attribute									{ $$ = new AttrList((AttrList*)$1, (Attribute*)$3); }
;

attribute: 	NAME 							{ $$ = new Attribute(strData[0]); }
| NAME '.' NAME							{ $$ = new Attribute(strData[0], strData[1]); }
;

Predicate:
  Predicate OR Predicate					{ $$ = new RA_Predicate(0, (RA_Predicate*)$1, (RA_Predicate*)$3); }
| Predicate AND Predicate					{ $$ = new RA_Predicate(1, (RA_Predicate*)$1, (RA_Predicate*)$3); }
| NOT Predicate								{ $$ = new RA_Predicate(2, (RA_Predicate*)$2); }
| '(' Predicate ')'							{ $$ = new RA_Predicate(3, (RA_Predicate*)$2); }
| comparison								{ $$ = new RA_Predicate((Comparison*)$1); }
;

comparison:	compared CompOp compared		{ $$ = new Comparison((Compared*)$1, (CompOp*)$2, (Compared*)$3); }
;

CompOp:
NEQ 		{ $$ = new CompOp("NEQ"); }
| EQ 		{ $$ = new CompOp("EQ"); }
| GT		{ $$ = new CompOp("GT"); }
| GTE 		{ $$ = new CompOp("GTE"); }
| LT 		{ $$ = new CompOp("LT"); }
| LTE		{ $$ = new CompOp("LTE"); }
;

compared: 	
//attribute 	{ $$ = new Compared((Attribute*)$1); }
MathExpr		{ $$ = new Compared((MathExpr*)$1); }
;

data: 		
CONSTANT 	{ $$ = new Data(dData[0]); }
| STRVAL	{ $$ = new Data(strData[0]); }
;

/*
RelationList:
	relation
|	RelationList ',' relation
;
*/

relation: 	
NAME	{ $$ = new Relation(strData[0]); }
;

%%
/*
RelAlgNode *parse_root = 0;

int main() {

    string sql;
    cout << "Enter sql statement: ";
    getline(cin,sql);

    RelAlgNode *parseRoot = 0;
	Parser parser;
	parser.parse(sql, parseRoot);

	if (parseRoot != 0) cout << "parsed successfully, yo\n";
	QPTranslator qp;
    if (parseRoot != 0)
        parseRoot->accept(qp); 

	return 0;
}*/
