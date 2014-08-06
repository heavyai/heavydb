%name Parser
%define CLASS SQLParser
%define LSP_NEEDED
%define MEMBERS                 \
    virtual ~SQLParser()   {} \
    int parse(const string & inputStr, ASTNode *& parseRoot, string &lastParsed) { istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseRoot); lastParsed = lexer.YYText(); return yynerrs; } \
    private:                   \
       yyFlexLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {cerr << "Syntax error on line " << lexer.lineno() << ". Last word parsed:" << lexer.YYText() << endl;}

%header{
#include <iostream>
#include <fstream>
#include <sstream>
#include <FlexLexer.h>
#include <vector>

// AST nodes
#include "ast/ASTNode.h"
#include "ast/Statement.h"
#include "ast/AggrExpr.h"
#include "ast/AlterStmt.h"
#include "ast/Column.h"
#include "ast/ColumnDef.h"
#include "ast/ColumnDefList.h"
#include "ast/ColumnList.h"
#include "ast/Comparison.h"
#include "ast/CreateStmt.h"
#include "ast/DdlStmt.h"
#include "ast/DmlStmt.h"
#include "ast/DropStmt.h"
#include "ast/FromClause.h"
#include "ast/InsertColumnList.h"
#include "ast/InsertStmt.h"
#include "ast/Literal.h"
#include "ast/LiteralList.h"
#include "ast/MathExpr.h"
#include "ast/MapdDataT.h"
#include "ast/OptAllDistinct.h"
#include "ast/OptGroupby.h"
#include "ast/OptHaving.h"
#include "ast/OptLimit.h"
#include "ast/OptOrderby.h"
#include "ast/OptWhere.h"
#include "ast/OrderbyColumn.h"
#include "ast/OrderbyColumnList.h"
#include "ast/Predicate.h"
#include "ast/RenameStmt.h"
#include "ast/ScalarExpr.h"
#include "ast/ScalarExprList.h"
#include "ast/SearchCondition.h"
#include "ast/Selection.h"
#include "ast/SelectStmt.h"
#include "ast/SqlStmt.h"
#include "ast/Table.h"
#include "ast/TableList.h"

using namespace std;
using namespace SQL_Namespace;

// define stack element type to be a 
// pointer to an AST node
#define YY_Parser_STYPE ASTNode*
#define YY_Parser_PARSE_PARAM ASTNode*& parseRoot

extern ASTNode* parse_root;

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
%token SELECT INSERT UPDATE DELETE
%token INTO VALUES
%token ALL DISTINCT
%token FROM WHERE GROUPBY HAVING ORDERBY LIMIT
%token AVG COUNT MAX MIN SUM
%token AS NULLX
%token NAME IF EXISTS
%token ASC DESC
%token CREATE ALTER DROP RENAME TABLE
%token INTVAL FLOATVAL STRING
%token ADD TO
%token PRIMARY KEY
%token COLUMN
%token INTEGER 	/* "integer" */
%token FLOAT 	/* "float" */
%token BOOLEAN 	/* "boolean" */

%start sql_stmt

%%

sql_stmt:
	dml_stmt ';'		{ $$ = new SqlStmt((DmlStmt*)$1); parseRoot = $$; }
|	ddl_stmt ';'		{ $$ = new SqlStmt((DdlStmt*)$1); parseRoot = $$; }
|						{ $$ = 0; parseRoot = $$; }
;

	/***** Data manipulation language *****/

dml_stmt:
	insert_stmt			{ $$ = new DmlStmt((InsertStmt*)$1); }
|	select_stmt			{ $$ = new DmlStmt((SelectStmt*)$1); }
;

insert_stmt:
	INSERT INTO table '(' insert_column_list ')'
		VALUES '(' literal_list ')' {
			$$ = new InsertStmt((Table*)$3, (InsertColumnList*)$5, (LiteralList*)$9);
		}
;

insert_column_list:
	NAME 						{ $$ = new InsertColumnList(strData.back()); strData.pop_back(); }
|	insert_column_list ',' NAME { $$ = new InsertColumnList((InsertColumnList*)$1, strData.back()); strData.pop_back(); }
;

literal_list:
	literal 					{ $$ = new LiteralList((Literal*)$1); }
|	literal_list ',' literal 	{ $$ = new LiteralList((LiteralList*)$1, (Literal*)$3); }
 /*| NULLX*/
;

literal:
	INTVAL 					{ $$ = new Literal(intData.back()); intData.pop_back(); }
|	FLOATVAL 				{ $$ = new Literal(realData.back()); realData.pop_back(); }
| 	STRING 					{ $$ = new Literal(strData.back()); strData.pop_back(); }
;

select_stmt:
	SELECT
		opt_all_distinct
		selection
		from_clause
		opt_where
		opt_groupby
		opt_having
		opt_orderby
		opt_limit {
			$$ = new SelectStmt((OptAllDistinct*)$2, (Selection*)$3, (FromClause*)$4, (OptWhere*)$5, (OptGroupby*)$6, 
				(OptHaving*)$7, (OptOrderby*)$8, (OptLimit*)$9);
		}
;

selection:
	scalar_expr_list	{ $$ = new Selection((ScalarExprList*)$1); }
|	MULTIPLY 			{ $$ = new Selection(true); }
;

scalar_expr_list:
	scalar_expr 						{ $$ = new ScalarExprList((ScalarExpr*)$1); }
|	scalar_expr_list ',' scalar_expr 	{ $$ = new ScalarExprList((ScalarExprList*)$1, (ScalarExpr*)$3); }
;

scalar_expr:
	scalar_expr PLUS scalar_expr 			{ $$ = new ScalarExpr("PLUS", (ScalarExpr*)$1, (ScalarExpr*)$3); }
|	scalar_expr MINUS scalar_expr 			{ $$ = new ScalarExpr("MINUS", (ScalarExpr*)$1, (ScalarExpr*)$3); }
|	scalar_expr MULTIPLY scalar_expr 		{ $$ = new ScalarExpr("MULTIPLY", (ScalarExpr*)$1, (ScalarExpr*)$3); }
|	scalar_expr DIVIDE scalar_expr 			{ $$ = new ScalarExpr("DIVIDE", (ScalarExpr*)$1, (ScalarExpr*)$3); }
|	literal 								{ $$ = new ScalarExpr((Literal*)$1); }
|	column 									{ $$ = new ScalarExpr((Column*)$1); }
|	'(' scalar_expr ')' 					{ $$ = new ScalarExpr((ScalarExpr*)$1); }
	/* 	|	'+' scalar_expr %prec UMINUS
		|	'-' scalar_expr %prec UMINUS */
;

opt_all_distinct:
	ALL 				{ $$ = new OptAllDistinct(); }
|	DISTINCT 			{ $$ = new OptAllDistinct(); }
|
;

from_clause:
	FROM table_list				{ $$ = new FromClause((TableList*)$2); }
|	FROM '(' select_stmt ')'	{ $$ = new FromClause((SelectStmt*)$3); }
;

opt_where:
	WHERE search_condition	{ $$ = new OptWhere((SearchCondition*)$2); }
|
;

opt_groupby:
	GROUPBY column_list		{ $$ = new OptGroupby((ColumnList*)$2); }
|
;

opt_having:
	HAVING  				{ $$ = new OptHaving(); }
|
;

opt_orderby:
	ORDERBY orderby_column_list	{ $$ = new OptOrderby((OrderbyColumnList*)$2); }
|
;

opt_limit:
	LIMIT 					{ $$ = new OptLimit(); }
|
;

table_list:
	table_list ',' table 	{ $$ = new TableList((TableList*)$1, (Table*)$3); }
|	table 					{ $$ = new TableList((Table*)$1); }
;

table:
	NAME 					{ $$ = new Table(strData.back()); strData.pop_back(); }
|	NAME '.' NAME { 
		std::string s1 = strData.back();
		strData.pop_back();
		std::string s2 = strData.back();
		strData.pop_back();
		$$ = new Table(s1, s2);
	}
|	NAME AS NAME {
		std::string s1 = strData.back();
		strData.pop_back();
		std::string s2 = strData.back();
		strData.pop_back();
		$$ = new Table(s1, s2);
	}
;

column_list:
	column 					{ $$ = new ColumnList((Column*)$1); }
|	column_list ',' column 	{ $$ = new ColumnList((ColumnList*)$1, (Column*)$3); }
;

column:
	NAME 					{ $$ = new Column(strData.back()); strData.pop_back(); }
|	NAME '.' NAME {
		std::string s1 = strData.back();
		strData.pop_back();
		std::string s2 = strData.back();
		strData.pop_back();
		$$ = new Column(s1, s2);
	}
	/* 	|   NAME AS NAME {
		std::string s1 = strData.back();
		strData.pop_back();
		std::string s2 = strData.back();
		strData.pop_back();
		$$ = new Column(s1, s2);
	} */
;

orderby_column_list:
	orderby_column								{ $$ = new OrderbyColumnList((OrderbyColumn*)$1); }
|	orderby_column_list ',' orderby_column		{ $$ = new OrderbyColumnList((OrderbyColumnList*)$1, (OrderbyColumn*)$3); }
;

orderby_column:
	column			{ $$ = new OrderbyColumn((Column*)$1); }
|	column ASC		{ $$ = new OrderbyColumn((Column*)$1, true);  /* true => ASC */}
|	column DESC		{ $$ = new OrderbyColumn((Column*)$1, false); /* false => DESC */ }
;

search_condition:
	predicate 							{ $$ = new SearchCondition((Predicate*)$1); }
;

predicate:
 	predicate OR predicate				{ $$ = new Predicate("OR", (Predicate*)$1, (Predicate*)$3); }
|	predicate AND predicate				{ $$ = new Predicate("AND", (Predicate*)$1, (Predicate*)$3); }
|	NOT predicate						{ $$ = new Predicate("NOT", (Predicate*)$2); }
|	'(' predicate ')'					{ $$ = new Predicate((Predicate*)$2); }	
|	comparison							{ $$ = new Predicate((Comparison*)$1); }
;

comparison:
	math_expr NEQ math_expr				{ $$ = new Comparison("NEQ", (MathExpr*)$1, (MathExpr*)$3); }
|	math_expr EQ math_expr				{ $$ = new Comparison("EQ", (MathExpr*)$1, (MathExpr*)$3); }
|	math_expr GT math_expr				{ $$ = new Comparison("GT", (MathExpr*)$1, (MathExpr*)$3); }
|	math_expr GTE math_expr				{ $$ = new Comparison("GTE", (MathExpr*)$1, (MathExpr*)$3); }
|	math_expr LT math_expr				{ $$ = new Comparison("LT", (MathExpr*)$1, (MathExpr*)$3); }
|	math_expr LTE math_expr				{ $$ = new Comparison("LTE", (MathExpr*)$1, (MathExpr*)$3); }
;

math_expr:
	math_expr PLUS math_expr			{ $$ = new MathExpr("PLUS", (MathExpr*)$1, (MathExpr*)$3); }
|	math_expr MINUS math_expr			{ $$ = new MathExpr("MINUS", (MathExpr*)$1, (MathExpr*)$3); }
|	math_expr MULTIPLY math_expr		{ $$ = new MathExpr("MULTIPLY", (MathExpr*)$1, (MathExpr*)$3); }
|	math_expr DIVIDE math_expr			{ $$ = new MathExpr("DIVIDE", (MathExpr*)$1, (MathExpr*)$3); }
|	'(' math_expr ')'					{ $$ = new MathExpr((MathExpr*)$2); }
|	column								{ $$ = new MathExpr((Column*)$1); }
|	aggr_expr							{ $$ = new MathExpr((AggrExpr*)$1); }
|	INTVAL								{ $$ = new MathExpr((int)intData.back()); intData.pop_back(); }
|	FLOATVAL							{ $$ = new MathExpr((float)realData.back()); realData.pop_back(); }
;

aggr_expr:
	MAX '(' column ')'					{ $$ = new AggrExpr("MAX", (Column*)$3); }
|	MIN '(' column ')'					{ $$ = new AggrExpr("MIN", (Column*)$3); }
|	COUNT '(' column ')'				{ $$ = new AggrExpr("COUNT", (Column*)$3); }
|	SUM '(' column ')'					{ $$ = new AggrExpr("SUM", (Column*)$3); }
|	AVG '(' column ')'					{ $$ = new AggrExpr("AVG", (Column*)$3); }
;


	/***** Data definition language *****/

ddl_stmt:
	create_stmt							{ $$ = new DdlStmt((CreateStmt*)$1); }
|	drop_stmt							{ $$ = new DdlStmt((DropStmt*)$1); }
|	alter_stmt							{ $$ = new DdlStmt((AlterStmt*)$1); }
|	rename_stmt							{ $$ = new DdlStmt((RenameStmt*)$1); }
;

create_stmt:
	CREATE TABLE table '(' column_def_list ')'	{ $$ = new CreateStmt((Table*)$3, (ColumnDefList*)$5); }
|	CREATE TABLE IF NOT EXISTS table '(' column_def_list ')' { $$ = new CreateStmt((Table*)$6, (ColumnDefList*)$8); }
;

column_def_list:
	column_def 							{ $$ = new ColumnDefList((ColumnDef*)$1); }
|	column_def_list ',' column_def 		{ $$ = new ColumnDefList((ColumnDefList*)$1, (ColumnDef*)$3); }
;

column_def:
	column mapd_data_t 					{ $$ = new ColumnDef((Column*)$1, (MapdDataT*)$2); }
|	column mapd_data_t PRIMARY KEY 		{ $$ = new ColumnDef((Column*)$1, (MapdDataT*)$2, "PRIMARY KEY"); }
|	column mapd_data_t NULLX 			{ $$ = new ColumnDef((Column*)$1, (MapdDataT*)$2, "NULL"); }
|	column mapd_data_t NOT NULLX 		{ $$ = new ColumnDef((Column*)$1, (MapdDataT*)$2, "NOT NULL"); }
;

drop_stmt:
	DROP TABLE table 				{ $$ = new DropStmt((Table*)$3); }
|	DROP TABLE IF EXISTS table 		{ $$ = new DropStmt((Table*)$5); }
;

alter_stmt:
	ALTER TABLE table ADD column mapd_data_t 	{ $$ = new AlterStmt((Table*)$3, (Column*)$5, (MapdDataT*)$6); }
|	ALTER TABLE table DROP COLUMN column 		{ $$ = new AlterStmt((Table*)$3, (Column*)$6); }
;

rename_stmt:
	RENAME TABLE table TO NAME 	{ $$ = new RenameStmt((Table*)$3, strData.back()); strData.pop_back(); }
;

mapd_data_t:
	INTEGER 	{ $$ = new MapdDataT(0); }
|	FLOAT 		{ $$ = new MapdDataT(1); }
|	BOOLEAN 	{ $$ = new MapdDataT(2); }
;

%%
/*
int main(int argc, char ** argv) {
	SQLParser parser;
    string sql;
    do {
        cout << "mapd> ";
        getline(cin,sql);
        if (sql == "q")
            break;
        else sql = sql + "\n";

        ASTNode *parseRoot = 0;
        string lastParsed;
        int numErrors = parser.parse(sql, parseRoot, lastParsed);
        if (numErrors > 0) {
            cout << "Error at: " << lastParsed << endl;
            continue;
        }
        if (numErrors > 0)
            cout << "# Errors: " << numErrors << endl;

    }
    while (true);
    cout << "Good-bye." << endl;
}
*/

