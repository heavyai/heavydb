%name Parser
%define CLASS RAParser
%define LSP_NEEDED
%define MEMBERS                 \
    virtual ~RAParser()   {} \
    int parse(const string & inputStr, RelAlgNode *& parseRoot, string &lastParsed) { istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseRoot); lastParsed = lexer.YYText(); return yynerrs; } \
    private:                   \
       yyFlexLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {cerr << "Syntax error on line " << lexer.lineno() << ". Last word parsed:" << lexer.YYText() << endl;}

%header{
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
#include "ast/ScanOp.h"
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
extern std::vector<long int> intData;
extern std::vector<double> realData;


%}

%left PLUS MINUS
%left MULTIPLY DIVIDE
%left OR
%left AND
%left NOT

%left NEQ EQ GT GTE LT LTE
%token SCAN SELECT PROJECT SORT RENAME EXTEND GROUPBY
%token PRODUCT JOIN SEMIJOIN ANTIJOIN OUTERJOIN UNION DIFF INTERSECTION
%token MAX MIN COUNT SUM AVG
%token MAX_DISTINCT MIN_DISTINCT COUNT_DISTINCT SUM_DISTINCT AVG_DISTINCT 
%token NAME INTVAL FLOATVAL STRVAL

%start Program

%%

Program:
	RelExprList				{ $$ = new Program((RelExprList*)$1); parseRoot = $$; }
|							{ $$ = 0; parseRoot = $$; }
;

RelExprList:
	RelExpr	';'				{ $$ = new RelExprList((RelExpr*)$1); }	
|	RelExprList RelExpr ';' { $$ = new RelExprList((RelExprList*)$1, (RelExpr*)$2); }	
;

RelExpr:
	UnaryOp					{ $$ = new RelExpr((UnaryOp*)$1); }
|	BinaryOp				{ $$ = new RelExpr((BinaryOp*)$1); }
| 	'(' RelExpr ')'			{ $$ = new RelExpr((RelExpr*)$1); }	
| 	Relation				{ $$ = new RelExpr((Relation*)$1); }
;

UnaryOp:
    SCAN    '(' RelExpr ',' '{' AttrList '}' ',' Predicate ')' { $$ = new ScanOp((RelExpr*)$3, (AttrList*)$6, (Predicate*)$9); }
|	SELECT 	'(' RelExpr ',' Predicate ')'		 	{ $$ = new SelectOp((RelExpr*)$3, (Predicate*)$5); }
|	PROJECT '(' RelExpr ',' '{' AttrList '}' ')' 	{ $$ = new ProjectOp((RelExpr*)$3, (AttrList*)$6); }
|	SORT	'(' RelExpr ',' '[' AttrList ']' ')' 	{ $$ = new SortOp((RelExpr*)$3, (AttrList*)$6); }
|	RENAME 	'(' RelExpr ',' NAME ',' NAME ')'		{ assert(strData.size() == 2); $$ = new RenameOp((RelExpr*)$3, strData.front(), strData.back()); strData.pop_back(); strData.pop_back();}
|	EXTEND	'(' RelExpr ',' Expr ',' NAME ')'		{ $$ = new ExtendOp((RelExpr*)$3, (Expr*)$5, strData.back()); strData.pop_back();}
|	GROUPBY	'('	RelExpr ',' '{' AttrList '}' ',' '{' AggrList '}' ')' { $$ = new GroupbyOp((RelExpr*)$3, (AttrList*)$6, (AggrList*)$10); }
|	GROUPBY	'('	RelExpr ',' '{' '}' ',' '{' AggrList '}' ')' { $$ = new GroupbyOp((RelExpr*)$3, (AggrList*)$9); }
|	GROUPBY	'('	RelExpr ',' '{' AttrList '}' ',' '{' '}' ')' { $$ = new GroupbyOp((RelExpr*)$3, (AttrList*)$6); }
;

BinaryOp:
	UNION	'(' RelExpr ',' RelExpr ')'					{ $$ = new UnionOp((RelExpr*)$3, (RelExpr*)$5); }
|	DIFF '(' RelExpr ',' RelExpr ')'					{ $$ = new DiffOp((RelExpr*)$3, (RelExpr*)$5); }
|	PRODUCT '(' RelExpr ',' RelExpr ')'					{ $$ = new ProductOp((RelExpr*)$3, (RelExpr*)$5); }
| 	INTERSECTION '(' RelExpr ',' RelExpr ')' 			{ $$ = new DiffOp((RelExpr*)$3, new DiffOp((RelExpr*)$3, (RelExpr*)$5)); }
|	JOIN	'(' RelExpr ',' RelExpr ',' Predicate ')'	{ $$ = new JoinOp((RelExpr*)$3, (RelExpr*)$5, (Predicate*)$7); }	
|	SEMIJOIN '(' RelExpr ',' RelExpr ',' Predicate ')'	{ $$ = new SemijoinOp((RelExpr*)$3, (RelExpr*)$5, (Predicate*)$7); }
|	OUTERJOIN '(' RelExpr ',' RelExpr ',' Predicate ')'	{ $$ = new OuterjoinOp((RelExpr*)$3, (RelExpr*)$5, (Predicate*)$7); }
|	ANTIJOIN '(' RelExpr ',' RelExpr ',' Predicate ')'	{ $$ = new AntijoinOp((RelExpr*)$3, (RelExpr*)$5, (Predicate*)$7); }
;

Expr:
	MathExpr		{ $$ = new Expr((MathExpr*)$1); }
|	Predicate		{ $$ = new Expr((Predicate*)$1); }
|	STRVAL			{ assert(strData.size() > 0); $$ = new Expr(strData.back()); strData.pop_back(); }
;

AggrList:
	AggrExpr					{ $$ = new AggrList((AggrExpr*)$1); }
|	AggrList ',' AggrExpr		{ $$ = new AggrList((AggrList*)$1, (AggrExpr*)$3); }
;

AggrExpr:
	MAX '(' Attribute ')'				{ $$ = new AggrExpr("MAX", (Attribute*)$3); }
|	MIN '(' Attribute ')'				{ $$ = new AggrExpr("MIN", (Attribute*)$3); }
|	COUNT '(' Attribute ')'				{ $$ = new AggrExpr("COUNT", (Attribute*)$3); }
|	SUM '(' Attribute ')'				{ $$ = new AggrExpr("SUM", (Attribute*)$3); }
|	AVG '(' Attribute ')'				{ $$ = new AggrExpr("AVG", (Attribute*)$3); }
|	MAX_DISTINCT '(' Attribute ')'		{ $$ = new AggrExpr("MAX_DISTINCT", (Attribute*)$3); }
|	MIN_DISTINCT '(' Attribute ')'		{ $$ = new AggrExpr("MIN_DISTINCT", (Attribute*)$3); }
|	COUNT_DISTINCT '(' Attribute ')'	{ $$ = new AggrExpr("COUNT_DISTINCT", (Attribute*)$3); }
|	SUM_DISTINCT '(' Attribute ')'		{ $$ = new AggrExpr("SUM_DISTINCT", (Attribute*)$3); }
|	AVG_DISTINCT '(' Attribute ')'		{ $$ = new AggrExpr("AVG_DISTINCT", (Attribute*)$3); }
;

AttrList:
  Attribute								{ $$ = new AttrList((Attribute*)$1); }
| AttrList ',' Attribute				{ $$ = new AttrList((AttrList*)$1, (Attribute*)$3); }
;

Attribute:
  NAME 									{ $$ = new Attribute(strData.back()); strData.pop_back(); }
| NAME '.' NAME							{ assert(strData.size() == 2); $$ = new Attribute(strData.front(), strData.back()); strData.pop_back(); strData.pop_back(); }
;

Predicate:
  Predicate OR Predicate				{ $$ = new Predicate(OP_OR, (Predicate*)$1, (Predicate*)$3); }
| Predicate AND Predicate				{ $$ = new Predicate(OP_AND, (Predicate*)$1, (Predicate*)$3); }
| NOT Predicate							{ $$ = new Predicate(OP_NOT, (Predicate*)$2); }
| '(' Predicate ')'						{ $$ = new Predicate((Predicate*)$2); }	
| Comparison							{ $$ = new Predicate((Comparison*)$1); }
;


MathExpr:
	MathExpr PLUS MathExpr				{ $$ = new MathExpr(OP_ADD, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr MINUS MathExpr				{ $$ = new MathExpr(OP_SUBTRACT, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr MULTIPLY MathExpr			{ $$ = new MathExpr(OP_MULTIPLY, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr DIVIDE MathExpr			{ $$ = new MathExpr(OP_DIVIDE, (MathExpr*)$1, (MathExpr*)$3); }
|	'(' MathExpr ')'					{ $$ = new MathExpr((MathExpr*)$2); }
|	Attribute							{ $$ = new MathExpr((Attribute*)$1); }
|	AggrExpr							{ $$ = new MathExpr((AggrExpr*)$1); }
|	INTVAL								{ $$ = new MathExpr((int)intData.back()); intData.pop_back(); }
|	FLOATVAL							{ $$ = new MathExpr((float)realData.back()); realData.pop_back(); }
;

Comparison:
	MathExpr NEQ MathExpr				{ $$ = new Comparison(OP_NEQ, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr EQ MathExpr				{ $$ = new Comparison(OP_EQ, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr GT MathExpr				{ $$ = new Comparison(OP_GT, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr GTE MathExpr				{ $$ = new Comparison(OP_GTE, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr LT MathExpr				{ $$ = new Comparison(OP_LT, (MathExpr*)$1, (MathExpr*)$3); }
|	MathExpr LTE MathExpr				{ $$ = new Comparison(OP_LTE, (MathExpr*)$1, (MathExpr*)$3); }
;

Relation: 	
	NAME								{ $$ = new Relation(strData.back()); strData.pop_back(); }
;

%%

/*
//RelAlgNode *parse_root = 0;

int main() {

	do {
    	string sql;
    	cout << "Enter RA statement: ";

    	getline(cin,sql);
    	if (sql == "q")
    		break;

    	RelAlgNode *parseRoot = 0;
    	string lastParsed;

		RAParser parser;
		parser.parse(sql, parseRoot, lastParsed);

		if (parseRoot != 0) cout << "ChunkKey Unicorns frolick!!\n";
		//QPTranslator qp;
    	//if (parseRoot != 0)
    	//    parseRoot->accept(qp); 
    } while (1);

	return 0;
}
*/

