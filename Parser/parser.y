%name Parser
%define CLASS Parser
#define LSP_NEEDED
%define MEMBERS                 \
    virtual ~Parser()   {} \
    int parse(const string & inputStr, list<Stmt*>*& parseTrees, string &lastParsed) { istringstream ss(inputStr); lexer.switch_streams(&ss,0);  yyparse(parseTrees); lastParsed = lexer.YYText(); return yynerrs; } \
    private:                   \
       yyFlexLexer lexer;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY { /*cerr << "Syntax error on line " << lexer.lineno() << ". Last word parsed: " << lexer.YYText() << endl;*/ }

%union {
	bool boolval;
	int	intval;
	float floatval;
	string strval;
	SQLOps opval;
	list<Node*> *listval;
	list<string> *slistval;
	Node *nodeval;
}

%header {
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <FlexLexer.h>
#include "ParserNode.h"

using namespace std;
using namespace Parser;

#define YY_Parser_PARSE_PARAM list<Stmt>*& parseTrees
%}

	/* symbolic tokens */

%token NAME
%token STRING
%token INTNUM FIXEDNUM

	/* operators */

%left OR
%left AND
%left NOT
%left COMPARISON /* = <> < > <= >= */
%left '+' '-'
%left '*' '/'
%nonassoc UMINUS

	/* literal keyword tokens */

%token ALL AMMSC ANY AS ASC AUTHORIZATION BETWEEN BIGINT BOOLEAN BY
%token CHARACTER CHECK CLOSE COMMIT CONTINUE CREATE CURRENT
%token CURSOR DECIMAL DECLARE DEFAULT DELETE DESC DISTINCT DOUBLE
%token ESCAPE EXISTS FETCH FLOAT FOR FOREIGN FOUND FROM 
%token GRANT GROUP HAVING IN INSERT INTEGER INTO
%token IS KEY LANGUAGE LIKE NULLX NUMERIC OF ON OPEN OPTION
%token ORDER PARAMETER PRECISION PRIMARY PRIVILEGES PROCEDURE
%token PUBLIC REAL REFERENCES ROLLBACK SCHEMA SELECT SET
%token SMALLINT SOME TABLE TIME TIMESTAMP TO UNION
%token UNIQUE UPDATE USER VALUES VIEW WHENEVER WHERE WITH WORK

%%

sql_list:
		sql ';'	{ $<listval>$ = new list<Stmt*>(1, dynamic_cast<Stmt*>($<nodeval>1)); parseTrees = $<listval>$; }
	|	sql_list sql ';' { $<listval>$ = $<listval>1->append($<nodeval>2); parseTrees = $<listval>$; }
	;


	/* schema definition language */
sql:		/* schema {	$$ = $1; } */
	 base_table_def { $$ = $1; }
	| view_def { $$ = $1; }
	/* | prvililege_def { $$ = $1; } */
	;
	
/* NOT SUPPORTED
schema:
		CREATE SCHEMA AUTHORIZATION user opt_schema_element_list
	;

opt_schema_element_list:
		/* empty */	{	$$ = nullptr; }
	|	schema_element_list {	$$ = $1; }
	;

schema_element_list:
		schema_element 
	|	schema_element_list schema_element
	;

schema_element:
		base_table_def {	$$ = $1; }
	|	view_def {	$$ = $1; }
	|	privilege_def {	$$ = $1; }
	;
NOT SUPPORTED */

base_table_def:
		CREATE TABLE table '(' base_table_element_commalist ')'
		{
			$$ = new CreateTableStmt($<stringval>3, static_cast<list<TableElement*>*>($<listval>5));
		}
	;

base_table_element_commalist:
		base_table_element { $$ = new list<TableElement*>(1, dynamic_cast<TableElement*>($<nodeval>1)); }
	|	base_table_element_commalist ',' base_table_element
	{
		$$ = $1;
		$<listval>$->push_back(dynamic_cast<TableElement*>($<nodeval>3));
	}
	;

base_table_element:
		column_def { $$ = $1; }
	|	table_constraint_def { $$ = $1; }
	;

column_def:
		column data_type 
		{	$<nodeval>$ = new ColumnDef($<stringval>1, dynamic_cast<SQLType*>($<nodeval>2), nullptr); }
		column data_type column_constraint_def
		{ $<nodeval>$ = new ColumnDef($<stringval>1, dynamic_cast<SQLType*>($<nodeval>2), dynamic_cast<ColumnConstraintDef*>($3)); }
	;

column_constraint_def:
		NOT NULLX { $<nodeval>$ = new ColumnConstraintDef(true, false, false, nullptr); }
	|	NOT NULLX UNIQUE { $<nodeval>$ = new ColumnConstraintDef(true, true, false, nullptr); }
	|	NOT NULLX PRIMARY KEY { $<nodeval>$ = new ColumnConstraintDef(true, true, true, nullptr); }
	|	DEFAULT literal { $<nodeval>$ = new ColumnConstraintDef(false, false, false, dynamic_cast<Literal*>($<nodeval>2)); }
	|	DEFAULT NULLX { $<nodeval>$ = new ColumnConstraintDef(false, false, false, new NullLiteral()); }
	|	DEFAULT USER { $<nodeval>$ = new ColumnConstraintDef(false, false, false, new UserLiteral());
	|	CHECK '(' search_condition ')' { $<nodeval>$ = new ColumnConstraintDef(dynamic_cast<Expr*>($<nodeval>3)); }
	|	REFERENCES table { $<nodeval>$ = new ColumnConstraintDef($<stringval>2, ""); }
	|	REFERENCES table '(' column ')' { $<nodeval>$ = new ColumnConstraintDef($<stringval>2, $<stringval>4); }
	;

table_constraint_def:
		UNIQUE '(' column_commalist ')'
	{ $<nodeval>$ = new UniqueDef(false, $<slistval>3); }
	|	PRIMARY KEY '(' column_commalist ')'
	{ $<nodeval>$ = new UniqueDef(true, $<slistval>4); }
	|	FOREIGN KEY '(' column_commalist ')'
			REFERENCES table 
	{ $<nodeval>$ = new ForeignKeyDef($<slistval>4, $<stringval>7, nullptr); }
	|	FOREIGN KEY '(' column_commalist ')'
			REFERENCES table '(' column_commalist ')'
	{ $<nodeval>$ = new ForeignKeyDef($<slistval>4, $<stringval>7, $<slistval>9); }
	|	CHECK '(' search_condition ')'
	{ $<nodeval>$ = new CheckDef(dynamic_cast<Expr*>($<nodeval>3)); }
	;

column_commalist:
		column { $<slistval>$ = new list<string>(1, $<stringval>1); }
	|	column_commalist ',' column
	{
		$$ = $1;
		$<slistval>$->push_back($<stringval>3);
	}
	;

view_def:
		CREATE VIEW table opt_column_commalist
		AS query_spec opt_with_check_option
		{
			$<nodeval>$ = new CreateViewStmt($<string>3, $<slistval>4, dynamic_cast<QuerySpec*>($<nodeval>6), $<boolval>7);
		}
	;
	
opt_with_check_option:
		/* empty */	{	$<boolval>$ = false; }
	|	WITH CHECK OPTION { $<boolval>$ = true; }
	;

opt_column_commalist:
		/* empty */ { $<slistval>$ = nullptr; }
	|	'(' column_commalist ')' { $$ = $2; }
	;

/* NOT SUPPORTED
privilege_def:
		GRANT privileges ON table TO grantee_commalist
		opt_with_grant_option
	;

opt_with_grant_option:
		/* empty */
	|	WITH GRANT OPTION
	;

privileges:
		ALL PRIVILEGES
	|	ALL
	|	operation_commalist
	;

operation_commalist:
		operation
	|	operation_commalist ',' operation
	;

operation:
		SELECT
	|	INSERT
	|	DELETE
	|	UPDATE opt_column_commalist
	|	REFERENCES opt_column_commalist
	;


grantee_commalist:
		grantee
	|	grantee_commalist ',' grantee
	;

grantee:
		PUBLIC
	|	user
	;

	/* cursor definition */
sql:
		cursor_def
	;


cursor_def:
		DECLARE cursor CURSOR FOR query_exp opt_order_by_clause
	;

NOT SUPPORTED */

opt_order_by_clause:
		/* empty */ { $<listval>$ = nullptr; }
	|	ORDER BY ordering_spec_commalist { $$ = $3; }
	;

ordering_spec_commalist:
		ordering_spec { $<listval>$ = new list<OrderSpec*>(1, dynamic_cast<OrderSpec*>($<nodeval>1)); }
	|	ordering_spec_commalist ',' ordering_spec
	{
		$$ = $1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

ordering_spec:
		INTNUM opt_asc_desc { $<nodeval>$ = new OrderSpec($<intval>1, nullptr, $<boolval>2); }
	|	column_ref opt_asc_desc { $<nodeval>$ = new OrderSpec(0, dynamic_cast<OrderSpec*>($<nodeval>1), $<boolval>2); }
	;

opt_asc_desc:
		/* empty */ { $<boolval>$ = false; /* default is ASC */ }
	|	ASC { $<boolval>$ = false; }
	|	DESC { $<boolval>$ = true; }
	;

	/* manipulative statements */

sql:		manipulative_statement
	;

manipulative_statement:
		/* close_statement */
	/* |	commit_statement */
	/* |	delete_statement_positioned */
		delete_statement
	/* |	fetch_statement */
	|	insert_statement
	/* |	open_statement */
	/* |	rollback_statement */
	|	select_statement
	/* |	update_statement_positioned */
	|	update_statement
	;

/* NOT SUPPORTED
close_statement:
		CLOSE cursor
	;

commit_statement:
		COMMIT WORK
	;

delete_statement_positioned:
		DELETE FROM table WHERE CURRENT OF cursor
	;

fetch_statement:
		FETCH cursor INTO target_commalist
	;
NOT SUPPORTED */

delete_statement:
		DELETE FROM table opt_where_clause
		{ $<nodeval>$ = new DeleteStmt($<stringval>3, dynamic_cast<Expr*>($<nodeval>4)); }
	;

insert_statement:
		INSERT INTO table opt_column_commalist VALUES '(' atom_commalist ')'
		{
			$<nodeval>$ = new InsertQueryStmt($<stringval>3, $<slistval>4, static_cast<list<Expr*>*>($<listval>7));
		}
		| INSERT INTO table opt_column_commalist query_spec
		{
			$<nodeval>$ = new InsertQueryStmt($<stringval>3, $<slistval>4, dynamic_cast<QuerySpec*>($<nodeval>5));
		}
	;

/* NOT SUPPORTED
open_statement:
		OPEN cursor
	;

rollback_statement:
		ROLLBACK WORK
	;

select_statement:
		SELECT opt_all_distinct selection
		INTO target_commalist
		table_exp
	;
NOT SUPPORTED */

opt_all_distinct:
		/* empty */ { $<boolval>$ = false; }
	|	ALL { $<boolval>$ = false; }
	|	DISTINCT { $<boolval>$ = true; }
	;

/* NOT SUPPORTED
update_statement_positioned:
		UPDATE table SET assignment_commalist
		WHERE CURRENT OF cursor
	;
NOT SUPPORTED */

assignment_commalist:
		assignment
	{ $<listval>$ = new list<Assignment*>(1, dynamic_cast<Assignment*>($<nodeval>1)); }
	|	assignment_commalist ',' assignment
	{
		$$ = $1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

assignment:
		column '=' scalar_exp 
		{ $<nodeval>$ = new Assignment($<stringval>1, dynamic_cast<Expr*>($<nodeval>3)); }
	;

update_statement:
		UPDATE table SET assignment_commalist opt_where_clause
		{ $<nodeval>$ = new UpdateStmt($<stringval>2, static_cast<list<Assignment*>*>($<listval>4), dynamic_cast<Expr*>($<nodeval>5)); }
	;

/* NOT SUPPORTED
target_commalist:
		target
	|	target_commalist ',' target
	;

target:
		parameter_ref
	;
NOT SUPPORTED */

opt_where_clause:
		/* empty */ { $<nodeval>$ = nullptr; }
	|	where_clause { $$ = $1; }
	;

select_statement:
		query_exp opt_order_by_clause
		{ $<nodeval>$ = new SelectStmt(dynamic_cast<QueryExpr*>($<nodeval>1), static_cast<list<OrderSpec*>*>($<listval>2)); }
	;

	/* query expressions */

query_exp:
		query_term { $$ = $1; }
	|	query_exp UNION query_term 
	{ $<nodeval>$ = new UnionQuery(false, dynamic_cast<QueryExpr*>($<nodeval>1), dynamic_cast<QueryExpr*>($<nodeval>3)); }
	|	query_exp UNION ALL query_term
	{ $<nodeval>$ = new UnionQuery(true, dynamic_cast<QueryExpr*>($<nodeval>1), dynamic_cast<QueryExpr*>($<nodeval>4)); }
	;

query_term:
		query_spec { $$ = $1; }
	|	'(' query_exp ')' { $$ = $2; }
	;

query_spec:
		SELECT opt_all_distinct selection from_clause opt_where_clause opt_group_by_clause opt_having_clause
		{ $<nodeval>$ = new QuerySpec($<boolval>2,
																	static_cast<list<Expr*>*>($<listval>3),
																	static_cast<list<TableRef*>*>($<listval>4),
																	dynamic_cast<Expr*>($<nodeval>5),
																	static_cast<list<ColumnRef*>*>($<listval>6),
																	dynamic_cast<Expr*>($<nodeval>7));
		}
	;

selection:
		scalar_exp_commalist { $$ = $1; }
	|	'*' { $<listval>$ = nullptr; /* nullptr means SELECT * */ }
	;

from_clause:
		FROM table_ref_commalist { $$ = $2; }
	;

table_ref_commalist:
		table_ref { $<listval>$ = new list<TableRef*>(1, dynamic_cast<TableRef*>($<nodeval>1)); }
	|	table_ref_commalist ',' table_ref
	{
		$$ = $1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

table_ref:
		table { $<nodeval>$ = new TableRef($<stringval>1); }
	|	table range_variable { $<nodeval>$ = new TableRef($<stringval>1, $<stringval>2); }
	;

where_clause:
		WHERE search_condition { $$ = $2; }
	;

opt_group_by_clause:
		/* empty */ { $<listval>$ = nullptr; }
	|	GROUP BY column_ref_commalist { $$ = $3; }
	;

column_ref_commalist:
		column_ref { $<listval>$ = new list<ColumnRef*>(1, $<nodeval>1); }
	|	column_ref_commalist ',' column_ref
	{
		$$ = $1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

opt_having_clause:
		/* empty */ { $<nodeval>$ = nullptr; }
	|	HAVING search_condition { $$ = $2; }
	;

	/* search conditions */

search_condition:
	|	search_condition OR search_condition
	{ $<nodeval>$ = new OperExpr(kOR, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	search_condition AND search_condition
	{ $<nodeval>$ = new OperExpr(kAND, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	NOT search_condition
	{ $<nodeval>$ = new OperExpr(kNOT, dynamic_cast<Expr*>($<nodeval>2), nullptr); }
	|	'(' search_condition ')' { $$ = $2; }
	|	predicate { $$ = $1; }
	;

predicate:
		comparison_predicate { $$ = $1; }
	|	between_predicate { $$ = $1; }
	|	like_predicate { $$ = $1; }
	|	test_for_null { $$ = $1; }
	|	in_predicate { $$ = $1; }
	|	all_or_any_predicate { $$ = $1; }
	|	existence_test { $$ = $1; }
	;

comparison_predicate:
		scalar_exp COMPARISON scalar_exp
		{ $<nodeval>$ = new OperExpr($<opval>2, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp COMPARISON subquery
		{ 
			$<nodeval>$ = new OperExpr($<opval>2, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); 
			/* subquery can only return a single result */
			dynamic_cast<SubqueryExpr*>($<nodeval>3)->set_qualifer(kONE); 
		}
	;

between_predicate:
		scalar_exp NOT BETWEEN scalar_exp AND scalar_exp
		{ $<nodeval>$ = new BetweenExpr(true, dynaimic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4), dynamic_cast<Expr*>($<nodeval>6)); }
	|	scalar_exp BETWEEN scalar_exp AND scalar_exp
		{ $<nodeval>$ = new BetweenExpr(false, dynaimic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>5)); }
	;

like_predicate:
		scalar_exp NOT LIKE atom opt_escape
	{ $<nodeval>$ = new LikeExpr(true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4), dyanmic_cast<Expr*>($<nodeval>5)); }
	|	scalar_exp LIKE atom opt_escape
	{ $<nodeval>$ = new LikeExpr(false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3), dyanmic_cast<Expr*>($<nodeval>4)); }
	;

opt_escape:
		/* empty */ { $<nodeval>$ = nullptr; }
	|	ESCAPE atom { $$ = $2; }
	;

test_for_null:
		column_ref IS NOT NULLX { $<nodeval>$ = new IsNullExpr(true); }
	|	column_ref IS NULLX { $<nodeval>$ = new IsNullExpr(false); }
	;

in_predicate:
		scalar_exp NOT IN subquery
		{ $<nodeval>$ = new InSubquery(true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<SubqueryExpr*>($<nodeval>4)); }
	|	scalar_exp IN subquery
		{ $<nodeval>$ = new InSubquery(false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<SubqueryExpr*>($<nodeval>3)); }
	|	scalar_exp NOT IN '(' atom_commalist ')'
	{ $<nodeval>$ = new InValues(true, dynamic_cast<Expr*>($<nodeval>1), static_cast<list<Expr*>*>($<listval>5)); }
	|	scalar_exp IN '(' atom_commalist ')'
	{ $<nodeval>$ = new InValues(false, dynamic_cast<Expr*>($<nodeval>1), static_cast<list<Expr*>*>($<listval>4)); }
	;

atom_commalist:
		atom { $<listval>$ = new list<Expr*>(1, dynamic_cast<Expr*>($<nodeval>1)); }
	|	atom_commalist ',' atom
	{
		$$ = $1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

all_or_any_predicate:
		scalar_exp COMPARISON any_all_some subquery
		{
			$<nodeval>$ = new OperExpr($<opval>2, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4));
			dynamic_cast<SubqueryExpr*>($<nodeval>4)->set_qualifier($<opval>3);
		}
	;
			
any_all_some:
		ANY
	|	ALL
	|	SOME
	;

existence_test:
		EXISTS subquery { $<nodeval>$ = new ExistsExpr(dynamic_cast<QuerySpec*>($<nodeval>2)); }
	;

subquery:
		'(' query_spec ')' { $<nodeval>$ = new SubqueryExpr(dynamic_cast<QuerySpec*>($<nodeval>2)); }
	;

	/* scalar expressions */

scalar_exp:
		scalar_exp '+' scalar_exp { $<nodeval>$ = new OperExpr(kPLUS, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '-' scalar_exp { $<nodeval>$ = new OperExpr(kMINUS, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '*' scalar_exp { $<nodeval>$ = new OperExpr(kMULTIPLY, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '/' scalar_exp { new OperExpr(kDIVIDE, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	'+' scalar_exp %prec UMINUS { $$ = $2; }
	|	'-' scalar_exp %prec UMINUS { $<nodeval>$ = new OperExpr(kUMINUS, dynamic_cast<Expr*>($<nodeval>2), nullptr); }
	|	atom { $$ = $1; }
	|	column_ref { $$ = $1; }
	|	function_ref { $$ = $1; }
	|	'(' scalar_exp ')' { $$ = $2; }
	;

scalar_exp_commalist:
		scalar_exp { $<listval>$ = new list<Expr*>(1, $<nodeval>1); }
	|	scalar_exp_commalist ',' scalar_exp
	{
		$$ = $1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

atom:
		literal { $$ = $1; }
	|	USER { $<nodeval>$ = new UserLiteral(); }
	| NULLX { $<nodeval>$ = new NullLiteral(); }
	/* |	parameter_ref { $$ = $1; } */
	;

/* TODO: do Postgres style PARAM
parameter_ref:
		parameter
	|	parameter parameter
	|	parameter INDICATOR parameter
	;
*/

function_ref:
		NAME '(' '*' ')' { $<nodeval>$ = new FunctionRef($<strval>1); }
	|	NAME '(' DISTINCT column_ref ')' { $<nodeval>$ = new FunctionRef($<strval>1, true, dynamic_cast<Expr*>($<nodeval>4); }
	|	NAME '(' ALL scalar_exp ')' { $<nodeval>$ = new FunctionRef($<strval>1, dynamic_cast<Expr*>($<nodeval>4); }
	|	NAME '(' scalar_exp ')' { $<nodeval>$ = new FunctionRef($<strval>1, dynamic_cast<Expr*>($<nodeval>3); }
	;

literal:
		STRING { $<nodeval>$ = new StringLiteral($<stringval>1); }
	|	INTNUM { $<nodeval>$ = new IntLiteral($<intval>1); }
	|	FIXEDNUM { $<nodeval>$ = new FixedPtLiteral($<stringval>1); }
	| FLOAT { $<nodeval>$ = new FloatLiteral($<floatval>1); }
	| DOUBLE { $<nodeval>$ = new DoubleLiteral($<<doubleval>1); }
	;

	/* miscellaneous */

table:
		NAME { $$ = $1; }
	/* |	NAME '.' NAME { $$ = new TableRef($<strval>1, $<strval>3); } */
	;

column_ref:
		NAME { $<nodeval>$ = new ColumnRef($<strval>1); }
	|	NAME '.' NAME	{ $<nodeval>$ = new ColumnRef($<strval>1, $<strval>3); }
	/* |	NAME '.' NAME '.' NAME { $$ = new ColumnRef($<strval>1, $<strval>3, $<strval>5); } */
	;

		/* data types */

data_type:
		BIGINT { $<nodeval>$ = new SQLType(kBIGINT); }
	|	BOOLEAN { $<nodeval>$ = new SQLType(kBOOLEAN); }
	|	CHARACTER { $<nodeval>$ = new SQLType(kCHAR); }
	|	CHARACTER '(' INTNUM ')' { $<nodeval>$ = new SQLType(kCHAR, $<intval>3); }
	|	NUMERIC { $<nodeval>$ = new SQLType(kNUMERIC); }
	|	NUMERIC '(' INTNUM ')' { $<nodeval>$ = new SQLType(kNUMERIC, $<intval>3); }
	|	NUMERIC '(' INTNUM ',' INTNUM ')' { $<nodeval>$ = new SQLType(kNUMERIC, $<intval>3, $<intval>5); }
	|	DECIMAL { $<nodeval>$ = new SQLType(kDECIMAL); }
	|	DECIMAL '(' INTNUM ')' { $<nodeval>$ = new SQLType(kDECIMAL, $<intval>3); }
	|	DECIMAL '(' INTNUM ',' INTNUM ')' { $<nodeval>$ = new SQLType(kDECIMAL, $<intval>3, $<intval>5); }
	|	INTEGER { $<nodeval>$ = new SQLType(kINT); }
	|	SMALLINT { $<nodeval>$ = new SQLType(kSMALLINT); }
	|	FLOAT { $<nodeval>$ = new SQLType(kFLOAT); }
	|	FLOAT '(' INTNUM ')' { $<nodeval>$ = new SQLType(kFLOAT, $<intval>3); }
	|	REAL { $<nodeval>$ = new SQLType(kFLOAT); }
	|	DOUBLE PRECISION { $<nodeval>$ = new SQLType(kDOUBLE); }
	| TIME { $<nodeval>$ = new SQLType(kTIME); }
	| TIMESTAMP { $<nodeval>$ = new SQLType(kTIMESTAMP); }
	;

	/* the various things you can name */

column:		NAME { $<strval>$ = yytext; }
	;

/*
cursor:		NAME { $<strval>$ = yytext; }
	;
*/

/* TODO: do Postgres-styl PARAM
parameter:
		PARAMETER	/* :name handled in parser */
		{ $$ = new Parameter(yytext+1); }
	;
*/

range_variable:	NAME { $<strval>$ = yytext; }
	;

user:		NAME { $<strval>$ = yytext; }
	;

%%
