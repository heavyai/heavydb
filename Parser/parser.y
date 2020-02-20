%name Parser
%define CLASS SQLParser
%define LVAL yylval
%define CONSTRUCTOR_INIT : lexer(yylval)
%define MEMBERS                                                                                                         \
  virtual ~SQLParser() {}                                                                                               \
  int parse(const std::string & inputStrOrig, std::list<std::unique_ptr<Stmt>>& parseTrees, std::string &lastParsed) {  \
    auto inputStr = boost::algorithm::trim_right_copy_if(inputStrOrig, boost::is_any_of(";") || boost::is_space()) + ";"; \
    boost::regex create_view_expr{R"(CREATE\s+VIEW\s+(IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9\$_]*)\s+AS\s+(.*);?)", \
                                  boost::regex::extended | boost::regex::icase};                                        \
    std::lock_guard<std::mutex> lock(mutex_);                                                                           \
    boost::smatch what;                                                                                                 \
    const auto trimmed_input = boost::algorithm::trim_copy(inputStr);                                                   \
    if (boost::regex_match(trimmed_input.cbegin(), trimmed_input.cend(), what, create_view_expr)) {                     \
      const bool if_not_exists = what[1].length() > 0;                                                                  \
      const auto view_name = what[2].str();                                                                             \
      const auto select_query = what[3].str();                                                                          \
      parseTrees.emplace_back(new CreateViewStmt(view_name, select_query, if_not_exists));                              \
      return 0;                                                                                                         \
    }                                                                                                                   \
    std::istringstream ss(inputStr);                                                                                    \
    lexer.switch_streams(&ss,0);                                                                                        \
    yyparse(parseTrees);                                                                                                \
    lastParsed = lexer.YYText();                                                                                        \
    if (!errors_.empty()) {                                                                                             \
      throw std::runtime_error(errors_[0]);                                                                             \
    }                                                                                                                   \
    return yynerrs;                                                                                                     \
  }                                                                                                                     \
 private:                                                                                                               \
  SQLLexer lexer;                                                                                                       \
  std::mutex mutex_;                                                                                                    \
  std::vector<std::string> errors_;
%define LEX_BODY {return lexer.yylex();}
%define ERROR_BODY {} /*{ std::cerr << "Syntax error on line " << lexer.lineno() << ". Last word parsed: " << lexer.YYText() << std::endl; } */

%header{
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <mutex>
#include <string>
#include <utility>
#include <stdexcept>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/regex.hpp>
#include <FlexLexer.h>
#include "ParserNode.h"
#include "ReservedKeywords.h"

using namespace Parser;
#define YY_Parser_PARSE_PARAM std::list<std::unique_ptr<Stmt>>& parseTrees
%}

%union {
	bool boolval;
	int64_t	intval;
	float floatval;
	double doubleval;
	std::string *stringval;
	SQLOps opval;
	SQLQualifier qualval;
	std::list<Node*> *listval;
	std::list<std::string*> *slistval;
	Node *nodeval;
}

%header{
	class SQLLexer : public yyFlexLexer {
		public:
			SQLLexer(YY_Parser_STYPE &lval) : yylval(lval) {};
			YY_Parser_STYPE &yylval;
	};
%}

	/* symbolic tokens */

%token NAME
%token DASHEDNAME
%token EMAIL
%token STRING FWDSTR SELECTSTRING QUOTED_IDENTIFIER
%token INTNUM FIXEDNUM

	/* operators */

%left OR
%left AND
%left NOT
%left EQUAL COMPARISON /* = <> < > <= >= */
%left '+' '-'
%left '*' '/' '%'
%nonassoc UMINUS

	/* literal keyword tokens */

%token ADD ALL ALTER AMMSC ANY ARCHIVE ARRAY AS ASC AUTHORIZATION BETWEEN BIGINT BOOLEAN BY
%token CASE CAST CHAR_LENGTH CHARACTER CHECK CLOSE CLUSTER COLUMN COMMIT CONTINUE COPY CREATE CURRENT
%token CURSOR DATABASE DATE DATETIME DATE_TRUNC DECIMAL DECLARE DEFAULT DELETE DESC DICTIONARY DISTINCT DOUBLE DROP
%token DUMP ELSE END EXISTS EXTRACT FETCH FIRST FLOAT FOR FOREIGN FOUND FROM
%token GEOGRAPHY GEOMETRY GRANT GROUP HAVING IF ILIKE IN INSERT INTEGER INTO
%token IS LANGUAGE LAST LENGTH LIKE LIMIT LINESTRING MOD MULTIPOLYGON NOW NULLX NUMERIC OF OFFSET ON OPEN OPTIMIZE
%token OPTIMIZED OPTION ORDER PARAMETER POINT POLYGON PRECISION PRIMARY PRIVILEGES PROCEDURE
%token SMALLINT SOME TABLE TEMPORARY TEXT THEN TIME TIMESTAMP TINYINT TO TRUNCATE UNION
%token PUBLIC REAL REFERENCES RENAME RESTORE REVOKE ROLE ROLLBACK SCHEMA SELECT SET SHARD SHARED SHOW
%token UNIQUE UPDATE USER VALIDATE VALUES VIEW WHEN WHENEVER WHERE WITH WORK EDIT ACCESS DASHBOARD SQL EDITOR

%start sql_list

%%

sql_list:
		sql ';'	{ parseTrees.emplace_front(dynamic_cast<Stmt*>($<nodeval>1)); }
	|	sql_list sql ';'
	{
		parseTrees.emplace_front(dynamic_cast<Stmt*>($<nodeval>2));
	}
	;


	/* schema definition language */
sql:		/* schema {	$<nodeval>$ = $<nodeval>1; } */
  create_table_as_statement { $<nodeval>$ = $<nodeval>1; }
	| create_table_statement { $<nodeval>$ = $<nodeval>1; }
	| show_table_schema { $<nodeval>$ = $<nodeval>1; }
	/* | prvililege_def { $<nodeval>$ = $<nodeval>1; } */
	| drop_view_statement { $<nodeval>$ = $<nodeval>1; }
	| drop_table_statement { $<nodeval>$ = $<nodeval>1; }
	| truncate_table_statement { $<nodeval>$ = $<nodeval>1; }
	| rename_table_statement { $<nodeval>$ = $<nodeval>1; }
	| rename_column_statement { $<nodeval>$ = $<nodeval>1; }
	| add_column_statement { $<nodeval>$ = $<nodeval>1; }
	| drop_column_statement { $<nodeval>$ = $<nodeval>1; }
	| copy_table_statement { $<nodeval>$ = $<nodeval>1; }
	| create_database_statement { $<nodeval>$ = $<nodeval>1; }
	| drop_database_statement { $<nodeval>$ = $<nodeval>1; }
	| rename_database_statement { $<nodeval>$ = $<nodeval>1; }
	| create_user_statement { $<nodeval>$ = $<nodeval>1; }
	| drop_user_statement { $<nodeval>$ = $<nodeval>1; }
	| alter_user_statement { $<nodeval>$ = $<nodeval>1; }
	| create_role_statement { $<nodeval>$ = $<nodeval>1; }
	| drop_role_statement { $<nodeval>$ = $<nodeval>1; }
	| grant_privileges_statement { $<nodeval>$ = $<nodeval>1; }
	| revoke_privileges_statement { $<nodeval>$ = $<nodeval>1; }
	| grant_role_statement { $<nodeval>$ = $<nodeval>1; }
	| optimize_table_statement { $<nodeval>$ = $<nodeval>1; }
	| validate_system_statement { $<nodeval>$ = $<nodeval>1; }
	| revoke_role_statement { $<nodeval>$ = $<nodeval>1; }
	| dump_table_statement { $<nodeval>$ = $<nodeval>1; }
	| restore_table_statement { $<nodeval>$ = $<nodeval>1; }
	;

/* NOT SUPPORTED
schema:
		CREATE SCHEMA AUTHORIZATION user opt_schema_element_list
	;

opt_schema_element_list:
		{ $<listval>$ = nullptr; } // empty
	|	schema_element_list {	$<listval>$ = $<listval>1; }
	;

schema_element_list:
		schema_element
	|	schema_element_list schema_element
	;

schema_element:
    create_table_as_statement {  $$ = $1; }
	| create_table_statement {	$$ = $1; }
	|	create_view_statement {	$$ = $1; }
	|	privilege_def {	$$ = $1; }
	;
NOT SUPPORTED */

create_database_statement:
		CREATE DATABASE opt_if_not_exists NAME
		{
			$<nodeval>$ = new CreateDBStmt($<stringval>4, nullptr, $<boolval>3);
		}
		| CREATE DATABASE opt_if_not_exists NAME '(' name_eq_value_list ')'
		{
			$<nodeval>$ = new CreateDBStmt($<stringval>4, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>6), $<boolval>3);
		}
		;
drop_database_statement:
		DROP DATABASE opt_if_exists NAME
		{
			$<nodeval>$ = new DropDBStmt($<stringval>4, $<boolval>3);
		}
		;
rename_database_statement:
		ALTER DATABASE NAME RENAME TO NAME
		{
		   $<nodeval>$ = new RenameDatabaseStmt($<stringval>3, $<stringval>6);
		}
		;

create_user_statement:
		CREATE USER username '(' name_eq_value_list ')'
		{
			$<nodeval>$ = new CreateUserStmt($<stringval>3, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>5));
		}
		;
drop_user_statement:
		DROP USER username
		{
			$<nodeval>$ = new DropUserStmt($<stringval>3);
		}
		;
alter_user_statement:
		ALTER USER username '(' name_eq_value_list ')'
		{
			$<nodeval>$ = new AlterUserStmt($<stringval>3, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>5));
		}
		|
		ALTER USER username RENAME TO username
		{
		   $<nodeval>$ = new RenameUserStmt($<stringval>3, $<stringval>6);
		}
		;

name_eq_value_list:
		name_eq_value
		{
			$<listval>$ = new std::list<Node*>(1, $<nodeval>1);
		}
		| name_eq_value_list ',' name_eq_value
		{
			$<listval>$ = $<listval>1;
			$<listval>$->push_back($<nodeval>3);
		}
		;
name_eq_value: NAME EQUAL literal { $<nodeval>$ = new NameValueAssign($<stringval>1, dynamic_cast<Literal*>($<nodeval>3)); }
		;

opt_if_not_exists:
		IF NOT EXISTS { $<boolval>$ = true; }
		| /* empty */ { $<boolval>$ = false; }
		;

opt_temporary:
                TEMPORARY { $<boolval>$ = true; }
                | /* empty */ { $<boolval>$ = false; }
                ;

create_table_as_statement:
    CREATE opt_temporary TABLE opt_if_not_exists table AS SELECTSTRING opt_with_option_list
    {
      $<nodeval>$ = new CreateTableAsSelectStmt($<stringval>5, $<stringval>7, $<boolval>2, $<boolval>4, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>8));
    }
  ;

create_table_statement:
		CREATE opt_temporary TABLE opt_if_not_exists table '(' base_table_element_commalist ')' opt_with_option_list
		{
		  $<nodeval>$ = new CreateTableStmt($<stringval>5, nullptr, reinterpret_cast<std::list<TableElement*>*>($<listval>7), $<boolval>2,  $<boolval>4, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>9));
		}
		| CREATE NAME TABLE opt_if_not_exists table '(' base_table_element_commalist ')' opt_with_option_list
		{
		  $<nodeval>$ = new CreateTableStmt($<stringval>5, $<stringval>2, reinterpret_cast<std::list<TableElement*>*>($<listval>7), false,  $<boolval>4, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>9));
		}
	;

show_table_schema:
		SHOW CREATE TABLE table
		{
		  $<nodeval>$ = new ShowCreateTableStmt($<stringval>4);
		}

opt_if_exists:
		IF EXISTS { $<boolval>$ = true; }
		| /* empty */ { $<boolval>$ = false; }
		;

drop_table_statement:
		DROP TABLE opt_if_exists table
		{
		  $<nodeval>$ = new DropTableStmt($<stringval>4, $<boolval>3);
		}
		;
truncate_table_statement:
		TRUNCATE TABLE table
		{
		  $<nodeval>$ = new TruncateTableStmt($<stringval>3);
		}
		;
rename_table_statement:
		ALTER TABLE table RENAME TO table
		{
		   $<nodeval>$ = new RenameTableStmt($<stringval>3, $<stringval>6);
		}
		;

rename_column_statement:
		ALTER TABLE table RENAME COLUMN column TO column
		{
		   $<nodeval>$ = new RenameColumnStmt($<stringval>3, $<stringval>6, $<stringval>8);
		}
		;

opt_column:
		| COLUMN;

column_defs:
		 column_def	{ $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
		|column_defs ',' column_def
		{
			$<listval>$ = $<listval>1;
			$<listval>$->push_back($<nodeval>3);
		}
		;

add_column_statement:
		 ALTER TABLE table ADD opt_column column_def
		{
		   $<nodeval>$ = new AddColumnStmt($<stringval>3, dynamic_cast<ColumnDef*>($<nodeval>6));
		}
		|ALTER TABLE table ADD '(' column_defs ')' 
		{
		   $<nodeval>$ = new AddColumnStmt($<stringval>3, reinterpret_cast<std::list<ColumnDef*>*>($<nodeval>6));
		}
		;

drop_column_statement:
		ALTER TABLE table drop_columns { $<nodeval>$ = new DropColumnStmt($<stringval>3, $<slistval>4); }
		;

drop_columns:
		 drop_column { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
		|drop_columns ',' drop_column { ($<listval>1)->push_back($<nodeval>3); }
		;
		
drop_column:
		DROP opt_column column { $<stringval>$ = $<stringval>3; }

copy_table_statement:
	COPY table FROM STRING opt_with_option_list
	{
	    $<nodeval>$ = new CopyTableStmt($<stringval>2, $<stringval>4, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>5));
    }
	| COPY '(' FWDSTR ')' TO STRING opt_with_option_list
	{
	    if (!boost::istarts_with(*$<stringval>3, "SELECT")) {
	        throw std::runtime_error("Select statement expected");
	    }
	    *$<stringval>3 += ";";
	    $<nodeval>$ = new ExportQueryStmt($<stringval>3, $<stringval>6, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>7));
	}
	;

dump_or_archive:
	DUMP | ARCHIVE;

dump_table_statement:
	dump_or_archive TABLE table TO STRING opt_with_option_list
	{
	    $<nodeval>$ = new DumpTableStmt($<stringval>3, $<stringval>5, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>6));
    }
    ;

restore_table_statement:
	RESTORE TABLE table FROM STRING opt_with_option_list
	{
	    $<nodeval>$ = new RestoreTableStmt($<stringval>3, $<stringval>5, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>6));
    }
    ;

create_role_statement:
		CREATE ROLE rolename
		{
		    $<nodeval>$ = new CreateRoleStmt($<stringval>3);
		}
		;
drop_role_statement:
		DROP ROLE rolename
		{
		    $<nodeval>$ = new DropRoleStmt($<stringval>3);
		}
		;
grant_privileges_statement:
		GRANT privileges ON privileges_target_type privileges_target TO grantees
		{
		    $<nodeval>$ = new GrantPrivilegesStmt($<slistval>2, $<stringval>4, $<stringval>5, $<slistval>7);
		}
		;
revoke_privileges_statement:
		REVOKE privileges ON privileges_target_type privileges_target FROM grantees
		{
		    $<nodeval>$ = new RevokePrivilegesStmt($<slistval>2, $<stringval>4, $<stringval>5, $<slistval>7);
		}
		;
grant_role_statement:
		GRANT rolenames TO grantees
		{
		    $<nodeval>$ = new GrantRoleStmt($<slistval>2, $<slistval>4);
		}
		;
revoke_role_statement:
		REVOKE rolenames FROM grantees
		{
		    $<nodeval>$ = new RevokeRoleStmt($<slistval>2, $<slistval>4);
		}
		;

optimize_table_statement:
		OPTIMIZE TABLE opt_table opt_with_option_list
		{
			$<nodeval>$ = new OptimizeTableStmt($<stringval>3, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>4));
		}
		;

validate_system_statement:
		VALIDATE CLUSTER opt_with_option_list
		{
			$<nodeval>$ = new ValidateStmt($<stringval>2, reinterpret_cast<std::list<NameValueAssign*>*>($<listval>3));
		}
		;		

base_table_element_commalist:
		base_table_element { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	base_table_element_commalist ',' base_table_element
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

base_table_element:
		column_def { $<nodeval>$ = $<nodeval>1; }
	|	table_constraint_def { $<nodeval>$ = $<nodeval>1; }
	;

column_def:
		column data_type opt_compression
		{	$<nodeval>$ = new ColumnDef($<stringval>1, dynamic_cast<SQLType*>($<nodeval>2), dynamic_cast<CompressDef*>($<nodeval>3), nullptr); }
		| column data_type column_constraint_def opt_compression
		{ $<nodeval>$ = new ColumnDef($<stringval>1, dynamic_cast<SQLType*>($<nodeval>2), dynamic_cast<CompressDef*>($<nodeval>4), dynamic_cast<ColumnConstraintDef*>($<nodeval>3)); }
	;

opt_compression:
		 NAME NAME
		{
			if (!boost::iequals(*$<stringval>1, "encoding"))
				throw std::runtime_error("Invalid identifier " + *$<stringval>1 + " in column definition.");
			delete $<stringval>1;
			$<nodeval>$ = new CompressDef($<stringval>2, 0);
		}
		| NAME NAME '(' INTNUM ')'
		{
			if (!boost::iequals(*$<stringval>1, "encoding"))
				throw std::runtime_error("Invalid identifier " + *$<stringval>1 + " in column definition.");
			delete $<stringval>1;
			$<nodeval>$ = new CompressDef($<stringval>2, (int)$<intval>4);
		}
		| /* empty */ { $<nodeval>$ = nullptr; }
		;

column_constraint_def:
		NOT NULLX { $<nodeval>$ = new ColumnConstraintDef(true, false, false, nullptr); }
	|	NOT NULLX UNIQUE { $<nodeval>$ = new ColumnConstraintDef(true, true, false, nullptr); }
	|	NOT NULLX PRIMARY NAME
  {
    if (!boost::iequals(*$<stringval>4, "key"))
      throw std::runtime_error("Syntax error at " + *$<stringval>4);
    $<nodeval>$ = new ColumnConstraintDef(true, true, true, nullptr);
  }
	|	DEFAULT literal { $<nodeval>$ = new ColumnConstraintDef(false, false, false, dynamic_cast<Literal*>($<nodeval>2)); }
	|	DEFAULT NULLX { $<nodeval>$ = new ColumnConstraintDef(false, false, false, new NullLiteral()); }
	|	DEFAULT USER { $<nodeval>$ = new ColumnConstraintDef(false, false, false, new UserLiteral()); }
	|	CHECK '(' general_exp ')' { $<nodeval>$ = new ColumnConstraintDef(dynamic_cast<Expr*>($<nodeval>3)); }
	|	REFERENCES table { $<nodeval>$ = new ColumnConstraintDef($<stringval>2, nullptr); }
	|	REFERENCES table '(' column ')' { $<nodeval>$ = new ColumnConstraintDef($<stringval>2, $<stringval>4); }
	;

table_constraint_def:
		UNIQUE '(' column_commalist ')'
	{ $<nodeval>$ = new UniqueDef(false, $<slistval>3); }
	|	PRIMARY NAME '(' column_commalist ')'
	{
    if (!boost::iequals(*$<stringval>2, "key"))
      throw std::runtime_error("Syntax error at " + *$<stringval>2);
    $<nodeval>$ = new UniqueDef(true, $<slistval>4);
  }
	|	FOREIGN NAME '(' column_commalist ')'
			REFERENCES table
	{
    if (!boost::iequals(*$<stringval>2, "key"))
      throw std::runtime_error("Syntax error at " + *$<stringval>2);
    $<nodeval>$ = new ForeignKeyDef($<slistval>4, $<stringval>7, nullptr);
  }
	|	FOREIGN NAME '(' column_commalist ')'
			REFERENCES table '(' column_commalist ')'
	{
    if (!boost::iequals(*$<stringval>2, "key"))
      throw std::runtime_error("Syntax error at " + *$<stringval>2);
    $<nodeval>$ = new ForeignKeyDef($<slistval>4, $<stringval>7, $<slistval>9);   }
	|	SHARD NAME '(' column ')'
	{
	if (!boost::iequals(*$<stringval>2, "key"))
	  throw std::runtime_error("Syntax error at " + *$<stringval>2);
	$<nodeval>$ = new ShardKeyDef(*$<stringval>4);
	delete $<stringval>2;
	delete $<stringval>4;
	}
	|	SHARED DICTIONARY '(' column ')' REFERENCES table '(' column ')'
	{
		$<nodeval>$ = new SharedDictionaryDef(*$<stringval>4, *$<stringval>7, *$<stringval>9);
		delete $<stringval>4;
		delete $<stringval>7;
		delete $<stringval>9;
	}
	|	CHECK '(' general_exp ')'
	{ $<nodeval>$ = new CheckDef(dynamic_cast<Expr*>($<nodeval>3)); }
	;

column_commalist:
		column { $<slistval>$ = new std::list<std::string*>(1, $<stringval>1); }
	|	column_commalist ',' column
	{
		$<slistval>$ = $<slistval>1;
		$<slistval>$->push_back($<stringval>3);
	}
	;

opt_with_option_list:
		WITH '(' name_eq_value_list ')'
		{ $<listval>$ = $<listval>3; }
		| /* empty */
		{ $<listval>$ = nullptr; }
		;

drop_view_statement:
		DROP VIEW opt_if_exists table
		{
			$<nodeval>$ = new DropViewStmt($<stringval>4, $<boolval>3);
		}
		;

opt_column_commalist:
		/* empty */ { $<slistval>$ = nullptr; }
	|	'(' column_commalist ')' { $<slistval>$ = $<slistval>2; }
	;

/* NOT SUPPORTED
privilege_def:
		GRANT privileges ON table TO grantee_commalist
		opt_with_grant_option
	;

opt_with_grant_option:
		// empty
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

sql:
		cursor_def
	;


cursor_def:
		DECLARE cursor CURSOR FOR query_exp opt_order_by_clause
	;

NOT SUPPORTED */

opt_order_by_clause:
		/* empty */ { $<listval>$ = nullptr; }
	|	ORDER BY ordering_spec_commalist { $<listval>$ = $<listval>3; }
	;

ordering_spec_commalist:
		ordering_spec { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	ordering_spec_commalist ',' ordering_spec
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

ordering_spec:
		INTNUM opt_asc_desc opt_null_order
		{ $<nodeval>$ = new OrderSpec($<intval>1, nullptr, $<boolval>2, $<boolval>3); }
	|	column_ref opt_asc_desc opt_null_order
	{ $<nodeval>$ = new OrderSpec(0, dynamic_cast<ColumnRef*>($<nodeval>1), $<boolval>2, $<boolval>3); }
	;

opt_asc_desc:
		/* empty */ { $<boolval>$ = false; /* default is ASC */ }
	|	ASC { $<boolval>$ = false; }
	|	DESC { $<boolval>$ = true; }
	;

opt_null_order:
		/* empty */ { $<boolval>$ = false; /* default is NULL LAST */ }
	| NULLX FIRST { $<boolval>$ = true; }
	| NULLX LAST { $<boolval>$ = false; }
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
			$<nodeval>$ = new InsertValuesStmt($<stringval>3, $<slistval>4, reinterpret_cast<std::list<Expr*>*>($<listval>7));
		}
		| INSERT INTO table opt_column_commalist SELECTSTRING
		{
			$<nodeval>$ = new InsertIntoTableAsSelectStmt($<stringval>3, $<stringval>5, $<slistval>4);
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
	{ $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	assignment_commalist ',' assignment
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

assignment:
		column EQUAL general_exp
		{ $<nodeval>$ = new Assignment($<stringval>1, dynamic_cast<Expr*>($<nodeval>3)); }
	;

update_statement:
		UPDATE table SET assignment_commalist opt_where_clause
		{ $<nodeval>$ = new UpdateStmt($<stringval>2, reinterpret_cast<std::list<Assignment*>*>($<listval>4), dynamic_cast<Expr*>($<nodeval>5)); }
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
	|	where_clause { $<nodeval>$ = $<nodeval>1; }
	;

opt_limit_clause:
		LIMIT INTNUM { $<intval>$ = $<intval>2; if ($<intval>$ <= 0) throw std::runtime_error("LIMIT must be positive."); }
	| LIMIT ALL { $<intval>$ = 0; /* 0 means ALL */ }
	| /* empty */ { $<intval>$ = 0; /* 0 means ALL */ }
	;
opt_offset_clause:
		OFFSET INTNUM { $<intval>$ = $<intval>2; }
	| OFFSET INTNUM NAME
	{
		if (!boost::iequals(*$<stringval>3, "row") && !boost::iequals(*$<stringval>3, "rows"))
			throw std::runtime_error("Invalid word in OFFSET clause " + *$<stringval>3);
		delete $<stringval>3;
		$<intval>$ = $<intval>2;
	}
	| /* empty */
	{
		$<intval>$ = 0;
	}
	;

select_statement:
		query_exp opt_order_by_clause opt_limit_clause opt_offset_clause
		{ $<nodeval>$ = new SelectStmt(dynamic_cast<QueryExpr*>($<nodeval>1), reinterpret_cast<std::list<OrderSpec*>*>($<listval>2), $<intval>3, $<intval>4); }
	;

	/* query expressions */

query_exp:
		query_term { $<nodeval>$ = $<nodeval>1; }
	|	query_exp UNION query_term
	{ $<nodeval>$ = new UnionQuery(false, dynamic_cast<QueryExpr*>($<nodeval>1), dynamic_cast<QueryExpr*>($<nodeval>3)); }
	|	query_exp UNION ALL query_term
	{ $<nodeval>$ = new UnionQuery(true, dynamic_cast<QueryExpr*>($<nodeval>1), dynamic_cast<QueryExpr*>($<nodeval>4)); }
	;

query_term:
		query_spec { $<nodeval>$ = $<nodeval>1; }
	|	'(' query_exp ')' { $<nodeval>$ = $<nodeval>2; }
	;

query_spec:
		SELECT opt_all_distinct selection from_clause opt_where_clause opt_group_by_clause opt_having_clause
		{ $<nodeval>$ = new QuerySpec($<boolval>2,
																	reinterpret_cast<std::list<SelectEntry*>*>($<listval>3),
																	reinterpret_cast<std::list<TableRef*>*>($<listval>4),
																	dynamic_cast<Expr*>($<nodeval>5),
																	reinterpret_cast<std::list<Expr*>*>($<listval>6),
																	dynamic_cast<Expr*>($<nodeval>7));
		}
	;

selection:
		select_entry_commalist { $<listval>$ = $<listval>1; }
	|	'*' { $<listval>$ = nullptr; /* nullptr means SELECT * */ }
	;

from_clause:
		FROM table_ref_commalist { $<listval>$ = $<listval>2; }
	;

table_ref_commalist:
		table_ref { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	table_ref_commalist ',' table_ref
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

table_ref:
		table { $<nodeval>$ = new TableRef($<stringval>1); }
	|	table range_variable { $<nodeval>$ = new TableRef($<stringval>1, $<stringval>2); }
	;

where_clause:
		WHERE general_exp { $<nodeval>$ = $<nodeval>2; }
	;

opt_group_by_clause:
		/* empty */ { $<listval>$ = nullptr; }
	|	GROUP BY exp_commalist { $<listval>$ = $<listval>3; }
	;

exp_commalist:
		general_exp { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	exp_commalist ',' general_exp
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

opt_having_clause:
		/* empty */ { $<nodeval>$ = nullptr; }
	|	HAVING general_exp { $<nodeval>$ = $<nodeval>2; }
	;

	/* search conditions */

general_exp:
	general_exp OR general_exp
	{ $<nodeval>$ = new OperExpr(kOR, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	general_exp AND general_exp
	{ $<nodeval>$ = new OperExpr(kAND, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	NOT general_exp
	{ $<nodeval>$ = new OperExpr(kNOT, dynamic_cast<Expr*>($<nodeval>2), nullptr); }
	|	'(' general_exp ')' { $<nodeval>$ = $<nodeval>2; }
	|	predicate { $<nodeval>$ = $<nodeval>1; }
	;

predicate:
		comparison_predicate { $<nodeval>$ = $<nodeval>1; }
	|	between_predicate { $<nodeval>$ = $<nodeval>1; }
	|	like_predicate { $<nodeval>$ = $<nodeval>1; }
	|	test_for_null { $<nodeval>$ = $<nodeval>1; }
	|	in_predicate { $<nodeval>$ = $<nodeval>1; }
	|	all_or_any_predicate { $<nodeval>$ = $<nodeval>1; }
	|	existence_test { $<nodeval>$ = $<nodeval>1; }
  | scalar_exp { $<nodeval>$ = $<nodeval>1; }
	;

comparison_predicate:
		scalar_exp comparison scalar_exp
		{ $<nodeval>$ = new OperExpr($<opval>2, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp comparison subquery
		{
			$<nodeval>$ = new OperExpr($<opval>2, kONE, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3));
			/* subquery can only return a single result */
		}
	;

between_predicate:
		scalar_exp NOT BETWEEN scalar_exp AND scalar_exp
		{ $<nodeval>$ = new BetweenExpr(true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4), dynamic_cast<Expr*>($<nodeval>6)); }
	|	scalar_exp BETWEEN scalar_exp AND scalar_exp
		{ $<nodeval>$ = new BetweenExpr(false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>5)); }
	;

like_predicate:
		scalar_exp NOT LIKE atom opt_escape
	{ $<nodeval>$ = new LikeExpr(true, false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4), dynamic_cast<Expr*>($<nodeval>5)); }
	|	scalar_exp LIKE atom opt_escape
	{ $<nodeval>$ = new LikeExpr(false, false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>4)); }
	|	scalar_exp NOT ILIKE atom opt_escape
	{ $<nodeval>$ = new LikeExpr(true, true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4), dynamic_cast<Expr*>($<nodeval>5)); }
	|	scalar_exp ILIKE atom opt_escape
	{ $<nodeval>$ = new LikeExpr(false, true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>4)); }
	;

opt_escape:
		/* empty */ { $<nodeval>$ = nullptr; }
	|	NAME atom
    {
      std::string escape_tok = *$<stringval>1;
      std::transform(escape_tok.begin(), escape_tok.end(), escape_tok.begin(), ::tolower);
      if (escape_tok != "escape") {
        throw std::runtime_error("Syntax error: wrong escape specifier");
      }
      delete $<stringval>1;
      $<nodeval>$ = $<nodeval>2;
    }
	;

test_for_null:
		column_ref IS NOT NULLX { $<nodeval>$ = new IsNullExpr(true, dynamic_cast<Expr*>($<nodeval>1)); }
	|	column_ref IS NULLX { $<nodeval>$ = new IsNullExpr(false, dynamic_cast<Expr*>($<nodeval>1)); }
	;

in_predicate:
		scalar_exp NOT IN subquery
		{ $<nodeval>$ = new InSubquery(true, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<SubqueryExpr*>($<nodeval>4)); }
	|	scalar_exp IN subquery
		{ $<nodeval>$ = new InSubquery(false, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<SubqueryExpr*>($<nodeval>3)); }
  /* causes reduce/reduce conflict
  | scalar_exp NOT IN scalar_exp
  {
    $<nodeval>$ = new OperExpr(kNE, kALL, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4));
  }
  | scalar_exp IN scalar_exp
  {
    $<nodeval>$ = new OperExpr(kEQ, kANY, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4));
  }
  */
	|	scalar_exp NOT IN '(' atom_commalist ')'
	{ $<nodeval>$ = new InValues(true, dynamic_cast<Expr*>($<nodeval>1), reinterpret_cast<std::list<Expr*>*>($<listval>5)); }
	|	scalar_exp IN '(' atom_commalist ')'
	{ $<nodeval>$ = new InValues(false, dynamic_cast<Expr*>($<nodeval>1), reinterpret_cast<std::list<Expr*>*>($<listval>4)); }
	;

atom_commalist:
		atom { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	atom_commalist ',' atom
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

all_or_any_predicate:
		scalar_exp comparison any_all_some subquery
		{
			$<nodeval>$ = new OperExpr($<opval>2, $<qualval>3, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4));
		}
    | scalar_exp comparison any_all_some scalar_exp
    {
			$<nodeval>$ = new OperExpr($<opval>2, $<qualval>3, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>4));
    }
	;

comparison:
	EQUAL { $<opval>$ = $<opval>1; }
	| COMPARISON { $<opval>$ = $<opval>1; }
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

when_then_list:
		WHEN general_exp THEN general_exp
		{
			$<listval>$ = new std::list<Node*>(1, new ExprPair(dynamic_cast<Expr*>($<nodeval>2), dynamic_cast<Expr*>($<nodeval>4)));
		}
		| when_then_list WHEN general_exp THEN general_exp
		{
			$<listval>$ = $<listval>1;
			$<listval>$->push_back(new ExprPair(dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>5)));
		}
		;
opt_else_expr :
		ELSE general_exp { $<nodeval>$ = $<nodeval>2; }
		| /* empty */ { $<nodeval>$ = nullptr; }
		;

case_exp: CASE when_then_list opt_else_expr END
	{
		$<nodeval>$ = new CaseExpr(reinterpret_cast<std::list<ExprPair*>*>($<listval>2), dynamic_cast<Expr*>($<nodeval>3));
	}
  | IF '(' general_exp ',' general_exp ',' general_exp ')'
  {
    std::list<ExprPair*> *when_then_list = new std::list<ExprPair*>(1, new ExprPair(dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>5)));
    $<nodeval>$ = new CaseExpr(when_then_list, dynamic_cast<Expr*>($<nodeval>7));
  }
  | IF '(' general_exp ',' general_exp ')'
  {
    std::list<ExprPair*> *when_then_list = new std::list<ExprPair*>(1, new ExprPair(dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>5)));
    $<nodeval>$ = new CaseExpr(when_then_list, nullptr);
  }
	;

 charlength_exp:
	      CHAR_LENGTH '(' scalar_exp ')' { $<nodeval>$ = new CharLengthExpr(dynamic_cast<Expr*>($<nodeval>3),true); }
	    | LENGTH '(' scalar_exp ')'	{ $<nodeval>$ = new CharLengthExpr(dynamic_cast<Expr*>($<nodeval>3),false); }
	    ;


/* should be scaler_exp '[' scalar_exp ']' but it causes conflicts.  need to debug */
array_at_exp : column_ref '[' scalar_exp ']'
  {
    $<nodeval>$ = new OperExpr(kARRAY_AT, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3));
  }
  ;

	/* scalar expressions */

scalar_exp:
		scalar_exp '+' scalar_exp { $<nodeval>$ = new OperExpr(kPLUS, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '-' scalar_exp { $<nodeval>$ = new OperExpr(kMINUS, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '*' scalar_exp { $<nodeval>$ = new OperExpr(kMULTIPLY, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '/' scalar_exp { $<nodeval>$ = new OperExpr(kDIVIDE, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	scalar_exp '%' scalar_exp { $<nodeval>$ = new OperExpr(kMODULO, dynamic_cast<Expr*>($<nodeval>1), dynamic_cast<Expr*>($<nodeval>3)); }
	|	MOD '(' scalar_exp ',' scalar_exp ')' { $<nodeval>$ = new OperExpr(kMODULO, dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<Expr*>($<nodeval>5)); }
	|	'+' scalar_exp %prec UMINUS { $<nodeval>$ = $<nodeval>2; }
	|	'-' scalar_exp %prec UMINUS { $<nodeval>$ = new OperExpr(kUMINUS, dynamic_cast<Expr*>($<nodeval>2), nullptr); }
	|	atom { $<nodeval>$ = $<nodeval>1; }
	|	column_ref { $<nodeval>$ = $<nodeval>1; }
	|	function_ref { $<nodeval>$ = $<nodeval>1; }
	|	'(' scalar_exp ')' { $<nodeval>$ = $<nodeval>2; }
	| CAST '(' general_exp AS data_type ')'
	{ $<nodeval>$ = new CastExpr(dynamic_cast<Expr*>($<nodeval>3), dynamic_cast<SQLType*>($<nodeval>5)); }
	| case_exp { $<nodeval>$ = $<nodeval>1; }
  | charlength_exp { $<nodeval>$ = $<nodeval>1; }
  | array_at_exp { $<nodeval>$ = $<nodeval>1; }
	;

select_entry:
		{ throw std::runtime_error("Empty select entry"); }
	| general_exp { $<nodeval>$ = new SelectEntry(dynamic_cast<Expr*>($<nodeval>1), nullptr); }
	| general_exp NAME { $<nodeval>$ = new SelectEntry(dynamic_cast<Expr*>($<nodeval>1), $<stringval>2); }
	| general_exp AS NAME { $<nodeval>$ = new SelectEntry(dynamic_cast<Expr*>($<nodeval>1), $<stringval>3); }
	;

select_entry_commalist:
		{ throw std::runtime_error("Empty select entry list"); }
	|	select_entry { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	select_entry_commalist ',' select_entry
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

atom:
		literal { $<nodeval>$ = $<nodeval>1; }
	|	USER { $<nodeval>$ = new UserLiteral(); }
	/* |	NULLX { $<nodeval>$ = new NullLiteral(); } */
	/* |	parameter_ref { $<nodeval>$ = $<nodeval>1; } */
	;

/* TODO: do Postgres style PARAM
parameter_ref:
		parameter
	|	parameter parameter
	|	parameter INDICATOR parameter
	;
*/

function_ref:
		NAME '(' '*' ')' { $<nodeval>$ = new FunctionRef($<stringval>1); }
	|	NAME '(' DISTINCT general_exp ')' { $<nodeval>$ = new FunctionRef($<stringval>1, true, dynamic_cast<Expr*>($<nodeval>4)); }
	|	NAME '(' ALL general_exp ')' { $<nodeval>$ = new FunctionRef($<stringval>1, dynamic_cast<Expr*>($<nodeval>4)); }
	|	NAME '(' general_exp ')' { $<nodeval>$ = new FunctionRef($<stringval>1, dynamic_cast<Expr*>($<nodeval>3)); }
	;

literal:
		STRING { $<nodeval>$ = new StringLiteral($<stringval>1); }
	|	INTNUM { $<nodeval>$ = new IntLiteral($<intval>1); }
	|	NOW '(' ')' { $<nodeval>$ = new TimestampLiteral(); }
	|	DATETIME '(' general_exp ')' { delete dynamic_cast<Expr*>($<nodeval>3); $<nodeval>$ = new TimestampLiteral(); }
	|	FIXEDNUM { $<nodeval>$ = new FixedPtLiteral($<stringval>1); }
	|	FLOAT { $<nodeval>$ = new FloatLiteral($<floatval>1); }
	|	DOUBLE { $<nodeval>$ = new DoubleLiteral($<doubleval>1); }
	|	data_type STRING { $<nodeval>$ = new CastExpr(new StringLiteral($<stringval>2), dynamic_cast<SQLType*>($<nodeval>1)); }
	|	'{' opt_literal_commalist '}' { $<nodeval>$ = new ArrayLiteral(reinterpret_cast<std::list<Expr*>*>($<listval>2)); }
	|	ARRAY '[' opt_literal_commalist ']' { $<nodeval>$ = new ArrayLiteral(reinterpret_cast<std::list<Expr*>*>($<listval>3)); }
	|	NULLX { $<nodeval>$ = new NullLiteral(); }
	;

literal_commalist:
		literal { $<listval>$ = new std::list<Node*>(1, $<nodeval>1); }
	|	literal_commalist ',' literal
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

opt_literal_commalist:
    { $<listval>$ = new std::list<Node*>(0); }
  | literal_commalist
  ;

	/* miscellaneous */

table:
		NAME { $<stringval>$ = $<stringval>1; }
	/* |	NAME '.' NAME { $$ = new TableRef($<stringval>1, $<stringval>3); } */
	;

opt_table:
		{ $<nodeval>$ = nullptr; }
	|	table
	;
username:
        NAME | EMAIL | DASHEDNAME | QUOTED_IDENTIFIER
    ;

rolenames:
		rolename { $<slistval>$ = new std::list<std::string*>(1, $<stringval>1); }
	|	rolenames ',' rolename
	{
		$<slistval>$ = $<slistval>1;
		$<slistval>$->push_back($<stringval>3);
	}
	;

rolename:
        NAME | DASHEDNAME
    ;

grantees:
		grantee { $<slistval>$ = new std::list<std::string*>(1, $<stringval>1); }
	|	grantees ',' grantee
	{
		$<slistval>$ = $<slistval>1;
		$<slistval>$->push_back($<stringval>3);
	}
	;

grantee:
        username | rolename
    ;

privileges:
		privilege { $<slistval>$ = new std::list<std::string*>(1, $<stringval>1); }
	|	privileges ',' privilege
	{
		$<slistval>$ = $<slistval>1;
		$<slistval>$->push_back($<stringval>3);
	}
	;

privilege:
		ALL { $<stringval>$ = new std::string("ALL"); }
	|	CREATE { $<stringval>$ = new std::string("CREATE"); }
	|	SELECT { $<stringval>$ = new std::string("SELECT"); }
	|	INSERT { $<stringval>$ = new std::string("INSERT"); }
	|	TRUNCATE { $<stringval>$ = new std::string("TRUNCATE"); }
	|	UPDATE { $<stringval>$ = new std::string("UPDATE"); }
	|	DELETE { $<stringval>$ = new std::string("DELETE"); }
	| 	ALTER { $<stringval>$ = new std::string("ALTER"); }
	|	DROP { $<stringval>$ = new std::string("DROP"); }
	|	VIEW { $<stringval>$ = new std::string("VIEW"); }
	|	EDIT { $<stringval>$ = new std::string("EDIT"); }
	|	ACCESS { $<stringval>$ = new std::string("ACCESS"); }
	|	CREATE TABLE { $<stringval>$ = new std::string("CREATE TABLE"); }
	|	CREATE VIEW { $<stringval>$ = new std::string("CREATE VIEW"); }
	|	SELECT VIEW { $<stringval>$ = new std::string("SELECT VIEW"); }
	|	DROP VIEW { $<stringval>$ = new std::string("DROP VIEW"); }
	|	CREATE DASHBOARD { $<stringval>$ = new std::string("CREATE DASHBOARD"); }
	|	EDIT DASHBOARD { $<stringval>$ = new std::string("EDIT DASHBOARD"); }
	|	VIEW DASHBOARD { $<stringval>$ = new std::string("VIEW DASHBOARD"); }
	|	DELETE DASHBOARD { $<stringval>$ = new std::string("DELETE DASHBOARD"); }
	|	VIEW SQL EDITOR { $<stringval>$ = new std::string("VIEW SQL EDITOR"); }
	;

privileges_target_type:
		DATABASE { $<stringval>$ = new std::string("DATABASE"); }
	|	TABLE { $<stringval>$ = new std::string("TABLE"); }
	|	DASHBOARD { $<stringval>$ = new std::string("DASHBOARD"); }
	|	VIEW { $<stringval>$ = new std::string("VIEW"); }
	;

privileges_target:
		NAME
	|	INTNUM { $<stringval>$ = new std::string(std::to_string($<intval>1)); }
    ;


column_ref:
		NAME { $<nodeval>$ = new ColumnRef($<stringval>1); }
	|	NAME '.' NAME	{ $<nodeval>$ = new ColumnRef($<stringval>1, $<stringval>3); }
	| NAME '.' '*' { $<nodeval>$ = new ColumnRef($<stringval>1, nullptr); }
	/* |	NAME '.' NAME '.' NAME { $$ = new ColumnRef($<stringval>1, $<stringval>3, $<stringval>5); } */
	;

non_neg_int: INTNUM
		{
			if ($<intval>1 < 0)
				throw std::runtime_error("No negative number in type definition.");
			$<intval>$ = $<intval>1;
		}

		/* data types */

data_type:
		BIGINT { $<nodeval>$ = new SQLType(kBIGINT); }
	| TEXT { $<nodeval>$ = new SQLType(kTEXT); }
	|	BOOLEAN { $<nodeval>$ = new SQLType(kBOOLEAN); }
	|	CHARACTER { $<nodeval>$ = new SQLType(kCHAR); }
	|	CHARACTER '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kCHAR, $<intval>3); }
	|	NUMERIC { $<nodeval>$ = new SQLType(kNUMERIC); }
	|	NUMERIC '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kNUMERIC, $<intval>3); }
	|	NUMERIC '(' non_neg_int ',' non_neg_int ')' { $<nodeval>$ = new SQLType(kNUMERIC, $<intval>3, $<intval>5, false); }
	|	DECIMAL { $<nodeval>$ = new SQLType(kDECIMAL); }
	|	DECIMAL '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kDECIMAL, $<intval>3); }
	|	DECIMAL '(' non_neg_int ',' non_neg_int ')' { $<nodeval>$ = new SQLType(kDECIMAL, $<intval>3, $<intval>5, false); }
	|	INTEGER { $<nodeval>$ = new SQLType(kINT); }
	|	TINYINT { $<nodeval>$ = new SQLType(kTINYINT); }
	|	SMALLINT { $<nodeval>$ = new SQLType(kSMALLINT); }
	|	FLOAT { $<nodeval>$ = new SQLType(kFLOAT); }
	/* |	FLOAT '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kFLOAT, $<intval>3); } */
	|	REAL { $<nodeval>$ = new SQLType(kFLOAT); }
	|	DOUBLE PRECISION { $<nodeval>$ = new SQLType(kDOUBLE); }
	|	DOUBLE { $<nodeval>$ = new SQLType(kDOUBLE); }
	| DATE { $<nodeval>$ = new SQLType(kDATE); }
	| TIME { $<nodeval>$ = new SQLType(kTIME); }
	| TIME '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kTIME, $<intval>3); }
	| TIMESTAMP { $<nodeval>$ = new SQLType(kTIMESTAMP); }
	| TIMESTAMP '(' non_neg_int ')' { $<nodeval>$ = new SQLType(kTIMESTAMP, $<intval>3); }
	| geo_type { $<nodeval>$ = new SQLType(static_cast<SQLTypes>($<intval>1), static_cast<int>(kGEOMETRY), 0, false); }
        /* | geography_type { $<nodeval>$ = $<nodeval>1; } */
	| geometry_type { $<nodeval>$ = $<nodeval>1; }
	| data_type '[' ']'
	{
		$<nodeval>$ = $<nodeval>1;
		if (dynamic_cast<SQLType*>($<nodeval>$)->get_is_array())
		  throw std::runtime_error("array of array not supported.");
		dynamic_cast<SQLType*>($<nodeval>$)->set_is_array(true);
	}
	| data_type '[' non_neg_int ']'
	{
		$<nodeval>$ = $<nodeval>1;
		if (dynamic_cast<SQLType*>($<nodeval>$)->get_is_array())
		  throw std::runtime_error("array of array not supported.");
		dynamic_cast<SQLType*>($<nodeval>$)->set_is_array(true);
		dynamic_cast<SQLType*>($<nodeval>$)->set_array_size($<intval>3);
	}
	;

geo_type:	POINT { $<intval>$ = kPOINT; }
	|	LINESTRING { $<intval>$ = kLINESTRING; }
	|	POLYGON { $<intval>$ = kPOLYGON; }
	|	MULTIPOLYGON { $<intval>$ = kMULTIPOLYGON; }
	;

geography_type:	GEOGRAPHY '(' geo_type ')'
		{ $<nodeval>$ = new SQLType(static_cast<SQLTypes>($<intval>3), static_cast<int>(kGEOGRAPHY), 4326, false); }
	|	GEOGRAPHY '(' geo_type ',' INTNUM ')'
		{ $<nodeval>$ = new SQLType(static_cast<SQLTypes>($<intval>3), static_cast<int>(kGEOGRAPHY), $<intval>5, false); }

geometry_type:	GEOMETRY '(' geo_type ')'
		{ $<nodeval>$ = new SQLType(static_cast<SQLTypes>($<intval>3), static_cast<int>(kGEOMETRY), 0, false); }
	|	GEOMETRY '(' geo_type ',' INTNUM ')'
		{ $<nodeval>$ = new SQLType(static_cast<SQLTypes>($<intval>3), static_cast<int>(kGEOMETRY), $<intval>5, false); }

	/* the various things you can name */

column:
	NAME
	{
		const auto uc_col_name = boost::to_upper_copy<std::string>(*$<stringval>1);
		if (reserved_keywords.find(uc_col_name) != reserved_keywords.end()) {
			errors_.push_back("Cannot use a reserved keyword as column name: " + *$<stringval>1);
		}
		$<stringval>$ = $<stringval>1; 
	}
    | 	QUOTED_IDENTIFIER { $<stringval>$ = $<stringval>1; }
	;

/*
cursor:		NAME { $<stringval>$ = $<stringval>1; }
	;
*/

/* TODO: do Postgres-styl PARAM
parameter:
		PARAMETER	// :name handled in parser
		{ $$ = new Parameter(yytext+1); }
	;
*/

range_variable:	NAME { $<stringval>$ = $<stringval>1; }
	;

/*
user:		NAME { $<stringval>$ = $<stringval>1; }
	;
*/

%%
