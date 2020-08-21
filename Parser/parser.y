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
#include "TrackedPtr.h"

using namespace Parser;
#define YY_Parser_PARSE_PARAM std::list<std::unique_ptr<Stmt>>& parseTrees

%}

%union {
	bool boolval;
	int64_t	intval;
	float floatval;
	double doubleval;
	TrackedPtr<std::string> *stringval;
	SQLOps opval;
	SQLQualifier qualval;
	TrackedListPtr<Node> *listval;
	TrackedListPtr<std::string> *slistval;
	TrackedPtr<Node> *nodeval;
}

%header{
	class SQLLexer : public yyFlexLexer {
		public:
			SQLLexer(YY_Parser_STYPE &lval) : yylval(lval) {};
			YY_Parser_STYPE &yylval;
			std::vector<std::unique_ptr<TrackedPtr<std::string>>> parsed_str_tokens_{};
			std::vector<std::unique_ptr<TrackedListPtr<std::string>>> parsed_str_list_tokens_{};
			std::vector<std::unique_ptr<TrackedPtr<Node>>> parsed_node_tokens_{};
			std::vector<std::unique_ptr<TrackedListPtr<Node>>> parsed_node_list_tokens_{};
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
%token CURSOR DATABASE DATAFRAME DATE DATETIME DATE_TRUNC DECIMAL DECLARE DEFAULT DELETE DESC DICTIONARY DISTINCT DOUBLE DROP
%token DUMP ELSE END EXISTS EXTRACT FETCH FIRST FLOAT FOR FOREIGN FOUND FROM
%token GEOGRAPHY GEOMETRY GRANT GROUP HAVING IF ILIKE IN INSERT INTEGER INTO
%token IS LANGUAGE LAST LENGTH LIKE LIMIT LINESTRING MOD MULTIPOLYGON NOW NULLX NUMERIC OF OFFSET ON OPEN OPTIMIZE
%token OPTIMIZED OPTION ORDER PARAMETER POINT POLYGON PRECISION PRIMARY PRIVILEGES PROCEDURE
%token SERVER SMALLINT SOME TABLE TEMPORARY TEXT THEN TIME TIMESTAMP TINYINT TO TRUNCATE UNION
%token PUBLIC REAL REFERENCES RENAME RESTORE REVOKE ROLE ROLLBACK SCHEMA SELECT SET SHARD SHARED SHOW
%token UNIQUE UPDATE USER VALIDATE VALUES VIEW WHEN WHENEVER WHERE WITH WORK EDIT ACCESS DASHBOARD SQL EDITOR

%start sql_list

%%

sql_list:
		sql ';'	{ parseTrees.emplace_front(dynamic_cast<Stmt*>(($<nodeval>1)->release())); }
	|	sql_list sql ';'
	{
		parseTrees.emplace_front(dynamic_cast<Stmt*>(($<nodeval>2)->release()));
	}
	;


	/* schema definition language */
sql:		/* schema {	$<nodeval>$ = $<nodeval>1; } */
  create_table_as_statement { $<nodeval>$ = $<nodeval>1; }
	| create_table_statement { $<nodeval>$ = $<nodeval>1; }
	| create_dataframe_statement { $<nodeval>$ = $<nodeval>1; }
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
		{ $<listval>$ = TrackedListPtr<Node>::makeEmpty(); } // empty
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
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CreateDBStmt(($<stringval>4)->release(), nullptr, $<boolval>3));
		}
		| CREATE DATABASE opt_if_not_exists NAME '(' name_eq_value_list ')'
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CreateDBStmt(($<stringval>4)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>6)->release()), $<boolval>3));
		}
		;
drop_database_statement:
		DROP DATABASE opt_if_exists NAME
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DropDBStmt(($<stringval>4)->release(), $<boolval>3));
		}
		;
rename_database_statement:
		ALTER DATABASE NAME RENAME TO NAME
		{
		   $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new RenameDatabaseStmt(($<stringval>3)->release(), ($<stringval>6)->release()));
		}
		;

create_user_statement:
		CREATE USER username '(' name_eq_value_list ')'
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CreateUserStmt(($<stringval>3)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>5)->release())));
		}
		;
drop_user_statement:
		DROP USER username
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DropUserStmt(($<stringval>3)->release()));
		}
		;
alter_user_statement:
		ALTER USER username '(' name_eq_value_list ')'
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new AlterUserStmt(($<stringval>3)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>5)->release())));
		}
		|
		ALTER USER username RENAME TO username
		{
		   $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new RenameUserStmt(($<stringval>3)->release(), ($<stringval>6)->release()));
		}
		;

name_eq_value_list:
		name_eq_value
		{
			$<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1);
		}
		| name_eq_value_list ',' name_eq_value
		{
			$<listval>$ = $<listval>1;
			$<listval>$->push_back($<nodeval>3);
		}
		;
name_eq_value: NAME EQUAL literal { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new NameValueAssign(($<stringval>1)->release(), dynamic_cast<Literal*>(($<nodeval>3)->release()))); }
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
      $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CreateTableAsSelectStmt(($<stringval>5)->release(), ($<stringval>7)->release(), $<boolval>2, $<boolval>4, reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>8)->release())));
    }
  ;

create_table_statement:
		CREATE opt_temporary TABLE opt_if_not_exists table '(' base_table_element_commalist ')' opt_with_option_list
		{
		  $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CreateTableStmt(($<stringval>5)->release(), nullptr, reinterpret_cast<std::list<TableElement*>*>(($<listval>7)->release()), $<boolval>2,  $<boolval>4, reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>9)->release())));
		}
		| CREATE NAME TABLE opt_if_not_exists table '(' base_table_element_commalist ')' opt_with_option_list
		{
		  $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CreateTableStmt(($<stringval>5)->release(), ($<stringval>2)->release(), reinterpret_cast<std::list<TableElement*>*>(($<listval>7)->release()), false,  $<boolval>4, reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>9)->release())));
		}
	;

create_dataframe_statement:
		CREATE DATAFRAME table '(' base_table_element_commalist ')' FROM STRING opt_with_option_list
		{
		  $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CreateDataframeStmt(($<stringval>3)->release(), reinterpret_cast<std::list<TableElement*>*>(($<listval>5)->release()), ($<stringval>8)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>9)->release())));
		}
	;

show_table_schema:
		SHOW CREATE TABLE table
		{
		  $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ShowCreateTableStmt(($<stringval>4)->release()));
		}

opt_if_exists:
		IF EXISTS { $<boolval>$ = true; }
		| /* empty */ { $<boolval>$ = false; }
		;

drop_table_statement:
		DROP TABLE opt_if_exists table
		{
		  $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DropTableStmt(($<stringval>4)->release(), $<boolval>3));
		}
		;
truncate_table_statement:
		TRUNCATE TABLE table
		{
		  $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TruncateTableStmt(($<stringval>3)->release()));
		}
		;
rename_table_statement:
		ALTER TABLE table RENAME TO table
		{
		   $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new RenameTableStmt(($<stringval>3)->release(), ($<stringval>6)->release()));
		}
		;

rename_column_statement:
		ALTER TABLE table RENAME COLUMN column TO column
		{
		   $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new RenameColumnStmt(($<stringval>3)->release(), ($<stringval>6)->release(), ($<stringval>8)->release()));
		}
		;

opt_column:
		| COLUMN;

column_defs:
		 column_def	{ $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
		|column_defs ',' column_def
		{
			$<listval>$ = $<listval>1;
			$<listval>$->push_back($<nodeval>3);
		}
		;

add_column_statement:
		 ALTER TABLE table ADD opt_column column_def
		{
		   $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new AddColumnStmt(($<stringval>3)->release(), dynamic_cast<ColumnDef*>(($<nodeval>6)->release())));
		}
		|ALTER TABLE table ADD '(' column_defs ')' 
		{
		   $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new AddColumnStmt(($<stringval>3)->release(), reinterpret_cast<std::list<ColumnDef*>*>(($<listval>6)->release())));
		}
		;

drop_column_statement:
		ALTER TABLE table drop_columns { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DropColumnStmt(($<stringval>3)->release(), ($<slistval>4)->release())); }
		;

drop_columns:
		 drop_column { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
		|drop_columns ',' drop_column { ($<listval>1)->push_back($<nodeval>3); }
		;
		
drop_column:
		DROP opt_column column { $<stringval>$ = $<stringval>3; }

copy_table_statement:
	COPY table FROM STRING opt_with_option_list
	{
	    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CopyTableStmt(($<stringval>2)->release(), ($<stringval>4)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>5)->release())));
    }
	| COPY '(' FWDSTR ')' TO STRING opt_with_option_list
	{
	    if (!boost::istarts_with(*($<stringval>3)->get(), "SELECT") && !boost::istarts_with(*($<stringval>3)->get(), "WITH")) {
	        throw std::runtime_error("SELECT or WITH statement expected");
	    }
	    *($<stringval>3)->get() += ";";
	    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ExportQueryStmt(($<stringval>3)->release(), ($<stringval>6)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>7)->release())));
	}
	;

dump_or_archive:
	DUMP | ARCHIVE;

dump_table_statement:
	dump_or_archive TABLE table TO STRING opt_with_option_list
	{
	    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DumpTableStmt(($<stringval>3)->release(), ($<stringval>5)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>6)->release())));
    }
    ;

restore_table_statement:
	RESTORE TABLE table FROM STRING opt_with_option_list
	{
	    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new RestoreTableStmt(($<stringval>3)->release(), ($<stringval>5)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>6)->release())));
    }
    ;

create_role_statement:
		CREATE ROLE rolename
		{
		    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CreateRoleStmt(($<stringval>3)->release()));
		}
		;
drop_role_statement:
		DROP ROLE rolename
		{
		    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DropRoleStmt(($<stringval>3)->release()));
		}
		;
grant_privileges_statement:
		GRANT privileges ON privileges_target_type privileges_target TO grantees
		{
		    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new GrantPrivilegesStmt(($<slistval>2)->release(), ($<stringval>4)->release(), ($<stringval>5)->release(), ($<slistval>7)->release()));
		}
		;
revoke_privileges_statement:
		REVOKE privileges ON privileges_target_type privileges_target FROM grantees
		{
		    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new RevokePrivilegesStmt(($<slistval>2)->release(), ($<stringval>4)->release(), ($<stringval>5)->release(), ($<slistval>7)->release()));
		}
		;
grant_role_statement:
		GRANT rolenames TO grantees
		{
		    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new GrantRoleStmt(($<slistval>2)->release(), ($<slistval>4)->release()));
		}
		;
revoke_role_statement:
		REVOKE rolenames FROM grantees
		{
		    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new RevokeRoleStmt(($<slistval>2)->release(), ($<slistval>4)->release()));
		}
		;

optimize_table_statement:
		OPTIMIZE TABLE opt_table opt_with_option_list
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OptimizeTableStmt(($<stringval>3)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>4)->release())));
		}
		;

validate_system_statement:
		VALIDATE CLUSTER opt_with_option_list
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ValidateStmt(($<stringval>2)->release(), reinterpret_cast<std::list<NameValueAssign*>*>(($<listval>3)->release())));
		}
		;

base_table_element_commalist:
		base_table_element { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
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
		{	$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnDef(($<stringval>1)->release(), dynamic_cast<SQLType*>(($<nodeval>2)->release()), dynamic_cast<CompressDef*>(($<nodeval>3)->release()), nullptr)); }
		| column data_type column_constraint_def opt_compression
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnDef(($<stringval>1)->release(), dynamic_cast<SQLType*>(($<nodeval>2)->release()), dynamic_cast<CompressDef*>(($<nodeval>4)->release()), dynamic_cast<ColumnConstraintDef*>(($<nodeval>3)->release()))); }
	;

opt_compression:
		 NAME NAME
		{
			if (!boost::iequals(*($<stringval>1)->get(), "encoding"))
				throw std::runtime_error("Invalid identifier " + *($<stringval>1)->get() + " in column definition.");
			delete ($<stringval>1)->release();
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CompressDef(($<stringval>2)->release(), 0));
		}
		| NAME NAME '(' INTNUM ')'
		{
			if (!boost::iequals(*($<stringval>1)->get(), "encoding"))
				throw std::runtime_error("Invalid identifier " + *($<stringval>1)->get() + " in column definition.");
			delete ($<stringval>1)->release();
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CompressDef(($<stringval>2)->release(), (int)$<intval>4));
		}
		| /* empty */ { $<nodeval>$ = TrackedPtr<Node>::makeEmpty(); }
		;

column_constraint_def:
		NOT NULLX { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(true, false, false, nullptr)); }
	|	NOT NULLX UNIQUE { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(true, true, false, nullptr)); }
	|	NOT NULLX PRIMARY NAME
  {
    if (!boost::iequals(*($<stringval>4)->get(), "key"))
      throw std::runtime_error("Syntax error at " + *($<stringval>4)->get());
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(true, true, true, nullptr));
  }
	|	DEFAULT literal { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(false, false, false, dynamic_cast<Literal*>(($<nodeval>2)->release()))); }
	|	DEFAULT NULLX { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(false, false, false, new NullLiteral())); }
	|	DEFAULT USER { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(false, false, false, new UserLiteral())); }
	|	CHECK '(' general_exp ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	REFERENCES table { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(($<stringval>2)->release(), nullptr)); }
	|	REFERENCES table '(' column ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnConstraintDef(($<stringval>2)->release(), ($<stringval>4)->release())); }
	;

table_constraint_def:
		UNIQUE '(' column_commalist ')'
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new UniqueDef(false, ($<slistval>3)->release())); }
	|	PRIMARY NAME '(' column_commalist ')'
	{
    if (!boost::iequals(*($<stringval>2)->get(), "key"))
      throw std::runtime_error("Syntax error at " + *($<stringval>2)->get());
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new UniqueDef(true, ($<slistval>4)->release()));
  }
	|	FOREIGN NAME '(' column_commalist ')'
			REFERENCES table
	{
    if (!boost::iequals(*($<stringval>2)->get(), "key"))
      throw std::runtime_error("Syntax error at " + *($<stringval>2)->get());
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ForeignKeyDef(($<slistval>4)->release(), ($<stringval>7)->release(), nullptr));
  }
	|	FOREIGN NAME '(' column_commalist ')'
			REFERENCES table '(' column_commalist ')'
	{
    if (!boost::iequals(*($<stringval>2)->get(), "key"))
      throw std::runtime_error("Syntax error at " + *($<stringval>2)->get());
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ForeignKeyDef(($<slistval>4)->release(), ($<stringval>7)->release(), ($<slistval>9)->release()));   }
	|	SHARD NAME '(' column ')'
	{
	if (!boost::iequals(*($<stringval>2)->get(), "key"))
	  throw std::runtime_error("Syntax error at " + *($<stringval>2)->get());
	$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ShardKeyDef(*($<stringval>4)->get()));
	delete ($<stringval>2)->release();
	delete ($<stringval>4)->release();
	}
	|	SHARED DICTIONARY '(' column ')' REFERENCES table '(' column ')'
	{
		$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SharedDictionaryDef(*($<stringval>4)->get(), *($<stringval>7)->get(), *($<stringval>9)->get()));
		delete ($<stringval>4)->release();
		delete ($<stringval>7)->release();
		delete ($<stringval>9)->release();
	}
	|	CHECK '(' general_exp ')'
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CheckDef(dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	;

column_commalist:
		column { $<slistval>$ = TrackedListPtr<std::string>::make(lexer.parsed_str_list_tokens_, 1, $<stringval>1); }
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
		{ $<listval>$ = TrackedListPtr<Node>::makeEmpty(); }
		;

drop_view_statement:
		DROP VIEW opt_if_exists table
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DropViewStmt(($<stringval>4)->release(), $<boolval>3));
		}
		;

opt_column_commalist:
		/* empty */ { $<slistval>$ = TrackedListPtr<std::string>::makeEmpty(); }
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
		/* empty */ { $<listval>$ = TrackedListPtr<Node>::makeEmpty(); }
	|	ORDER BY ordering_spec_commalist { $<listval>$ = $<listval>3; }
	;

ordering_spec_commalist:
		ordering_spec { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
	|	ordering_spec_commalist ',' ordering_spec
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

ordering_spec:
		INTNUM opt_asc_desc opt_null_order
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OrderSpec($<intval>1, nullptr, $<boolval>2, $<boolval>3)); }
	|	column_ref opt_asc_desc opt_null_order
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OrderSpec(0, dynamic_cast<ColumnRef*>(($<nodeval>1)->release()), $<boolval>2, $<boolval>3)); }
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
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DeleteStmt(($<stringval>3)->release(), dynamic_cast<Expr*>(($<nodeval>4)->release()))); }
	;

insert_statement:
		INSERT INTO table opt_column_commalist VALUES '(' atom_commalist ')'
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new InsertValuesStmt(($<stringval>3)->release(), ($<slistval>4)->release(), reinterpret_cast<std::list<Expr*>*>(($<listval>7)->release())));
		}
		| INSERT INTO table opt_column_commalist SELECTSTRING
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new InsertIntoTableAsSelectStmt(($<stringval>3)->release(), ($<stringval>5)->release(), ($<slistval>4)->release()));
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
	{ $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
	|	assignment_commalist ',' assignment
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

assignment:
		column EQUAL general_exp
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new Assignment(($<stringval>1)->release(), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	;

update_statement:
		UPDATE table SET assignment_commalist opt_where_clause
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new UpdateStmt(($<stringval>2)->release(), reinterpret_cast<std::list<Assignment*>*>(($<listval>4)->release()), dynamic_cast<Expr*>(($<nodeval>5)->release()))); }
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
		/* empty */ { $<nodeval>$ = TrackedPtr<Node>::makeEmpty(); }
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
		if (!boost::iequals(*($<stringval>3)->get(), "row") && !boost::iequals(*($<stringval>3)->get(), "rows"))
			throw std::runtime_error("Invalid word in OFFSET clause " + *($<stringval>3)->get());
		delete ($<stringval>3)->release();
		$<intval>$ = $<intval>2;
	}
	| /* empty */
	{
		$<intval>$ = 0;
	}
	;

select_statement:
		query_exp opt_order_by_clause opt_limit_clause opt_offset_clause
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SelectStmt(dynamic_cast<QueryExpr*>(($<nodeval>1)->release()), reinterpret_cast<std::list<OrderSpec*>*>(($<listval>2)->release()), $<intval>3, $<intval>4)); }
	;

	/* query expressions */

query_exp:
		query_term { $<nodeval>$ = $<nodeval>1; }
	|	query_exp UNION query_term
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new UnionQuery(false, dynamic_cast<QueryExpr*>(($<nodeval>1)->release()), dynamic_cast<QueryExpr*>(($<nodeval>3)->release()))); }
	|	query_exp UNION ALL query_term
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new UnionQuery(true, dynamic_cast<QueryExpr*>(($<nodeval>1)->release()), dynamic_cast<QueryExpr*>(($<nodeval>4)->release()))); }
	;

query_term:
		query_spec { $<nodeval>$ = $<nodeval>1; }
	|	'(' query_exp ')' { $<nodeval>$ = $<nodeval>2; }
	;

query_spec:
		SELECT opt_all_distinct selection from_clause opt_where_clause opt_group_by_clause opt_having_clause
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_,
                                           new QuerySpec($<boolval>2,
                                                         reinterpret_cast<std::list<SelectEntry*>*>(($<listval>3)->release()),
                                                         reinterpret_cast<std::list<TableRef*>*>(($<listval>4)->release()),
                                                         dynamic_cast<Expr*>(($<nodeval>5)->release()),
                                                         reinterpret_cast<std::list<Expr*>*>(($<listval>6)->release()),
                                                         dynamic_cast<Expr*>(($<nodeval>7)->release())));
		}
	;

selection:
		select_entry_commalist { $<listval>$ = $<listval>1; }
	|	'*' { $<listval>$ = TrackedListPtr<Node>::makeEmpty(); /* nullptr means SELECT * */ }
	;

from_clause:
		FROM table_ref_commalist { $<listval>$ = $<listval>2; }
	;

table_ref_commalist:
		table_ref { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
	|	table_ref_commalist ',' table_ref
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

table_ref:
		table { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TableRef(($<stringval>1)->release())); }
	|	table range_variable { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TableRef(($<stringval>1)->release(), ($<stringval>2)->release())); }
	;

where_clause:
		WHERE general_exp { $<nodeval>$ = $<nodeval>2; }
	;

opt_group_by_clause:
		/* empty */ { $<listval>$ = TrackedListPtr<Node>::makeEmpty(); }
	|	GROUP BY exp_commalist { $<listval>$ = $<listval>3; }
	;

exp_commalist:
		general_exp { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
	|	exp_commalist ',' general_exp
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

opt_having_clause:
		/* empty */ { $<nodeval>$ = TrackedPtr<Node>::makeEmpty(); }
	|	HAVING general_exp { $<nodeval>$ = $<nodeval>2; }
	;

	/* search conditions */

general_exp:
	general_exp OR general_exp
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kOR, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	general_exp AND general_exp
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kAND, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	NOT general_exp
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kNOT, dynamic_cast<Expr*>(($<nodeval>2)->release()), nullptr)); }
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
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr($<opval>2, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	scalar_exp comparison subquery
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr($<opval>2, kONE, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release())));
			/* subquery can only return a single result */
		}
	;

between_predicate:
		scalar_exp NOT BETWEEN scalar_exp AND scalar_exp
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new BetweenExpr(true, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release()), dynamic_cast<Expr*>(($<nodeval>6)->release()))); }
	|	scalar_exp BETWEEN scalar_exp AND scalar_exp
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new BetweenExpr(false, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()), dynamic_cast<Expr*>(($<nodeval>5)->release()))); }
	;

like_predicate:
		scalar_exp NOT LIKE atom opt_escape
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new LikeExpr(true, false, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release()), dynamic_cast<Expr*>(($<nodeval>5)->release()))); }
	|	scalar_exp LIKE atom opt_escape
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new LikeExpr(false, false, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release()))); }
	|	scalar_exp NOT ILIKE atom opt_escape
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new LikeExpr(true, true, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release()), dynamic_cast<Expr*>(($<nodeval>5)->release()))); }
	|	scalar_exp ILIKE atom opt_escape
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new LikeExpr(false, true, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release()))); }
	;

opt_escape:
		/* empty */ { $<nodeval>$ = TrackedPtr<Node>::makeEmpty(); }
	|	NAME atom
    {
      std::string escape_tok = *($<stringval>1)->get();
      std::transform(escape_tok.begin(), escape_tok.end(), escape_tok.begin(), ::tolower);
      if (escape_tok != "escape") {
        throw std::runtime_error("Syntax error: wrong escape specifier");
      }
      delete ($<stringval>1)->release();
      $<nodeval>$ = $<nodeval>2;
    }
	;

test_for_null:
		column_ref IS NOT NULLX { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new IsNullExpr(true, dynamic_cast<Expr*>(($<nodeval>1)->release()))); }
	|	column_ref IS NULLX { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new IsNullExpr(false, dynamic_cast<Expr*>(($<nodeval>1)->release()))); }
	;

in_predicate:
		scalar_exp NOT IN subquery
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new InSubquery(true, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<SubqueryExpr*>(($<nodeval>4)->release()))); }
	|	scalar_exp IN subquery
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new InSubquery(false, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<SubqueryExpr*>(($<nodeval>3)->release()))); }
  /* causes reduce/reduce conflict
  | scalar_exp NOT IN scalar_exp
  {
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kNE, kALL, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release())));
  }
  | scalar_exp IN scalar_exp
  {
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kEQ, kANY, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release())));
  }
  */
	|	scalar_exp NOT IN '(' atom_commalist ')'
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new InValues(true, dynamic_cast<Expr*>(($<nodeval>1)->release()), reinterpret_cast<std::list<Expr*>*>(($<listval>5)->release()))); }
	|	scalar_exp IN '(' atom_commalist ')'
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new InValues(false, dynamic_cast<Expr*>(($<nodeval>1)->release()), reinterpret_cast<std::list<Expr*>*>(($<listval>4)->release()))); }
	;

atom_commalist:
		atom { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
	|	atom_commalist ',' atom
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

all_or_any_predicate:
		scalar_exp comparison any_all_some subquery
		{
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr($<opval>2, $<qualval>3, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release())));
		}
    | scalar_exp comparison any_all_some scalar_exp
    {
			$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr($<opval>2, $<qualval>3, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release())));
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
		EXISTS subquery { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ExistsExpr(dynamic_cast<QuerySpec*>(($<nodeval>2)->release()))); }
	;

subquery:
		'(' query_spec ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SubqueryExpr(dynamic_cast<QuerySpec*>(($<nodeval>2)->release()))); }
	;

when_then_list:
		WHEN general_exp THEN general_exp
		{
			$<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, new ExprPair(dynamic_cast<Expr*>(($<nodeval>2)->release()), dynamic_cast<Expr*>(($<nodeval>4)->release())));
		}
		| when_then_list WHEN general_exp THEN general_exp
		{
			$<listval>$ = $<listval>1;
			$<listval>$->push_back(new ExprPair(dynamic_cast<Expr*>(($<nodeval>3)->release()), dynamic_cast<Expr*>(($<nodeval>5)->release())));
		}
		;
opt_else_expr :
		ELSE general_exp { $<nodeval>$ = $<nodeval>2; }
		| /* empty */ { $<nodeval>$ = TrackedPtr<Node>::makeEmpty(); }
		;

case_exp: CASE when_then_list opt_else_expr END
	{
		$<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CaseExpr(reinterpret_cast<std::list<ExprPair*>*>(($<listval>2)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release())));
	}
  | IF '(' general_exp ',' general_exp ',' general_exp ')'
  {
    std::list<ExprPair*> *when_then_list = new std::list<ExprPair*>(1, new ExprPair(dynamic_cast<Expr*>(($<nodeval>3)->release()), dynamic_cast<Expr*>(($<nodeval>5)->release())));
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CaseExpr(when_then_list, dynamic_cast<Expr*>(($<nodeval>7)->release())));
  }
  | IF '(' general_exp ',' general_exp ')'
  {
    std::list<ExprPair*> *when_then_list = new std::list<ExprPair*>(1, new ExprPair(dynamic_cast<Expr*>(($<nodeval>3)->release()), dynamic_cast<Expr*>(($<nodeval>5)->release())));
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CaseExpr(when_then_list, nullptr));
  }
	;

 charlength_exp:
	      CHAR_LENGTH '(' scalar_exp ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CharLengthExpr(dynamic_cast<Expr*>(($<nodeval>3)->release()),true)); }
	    | LENGTH '(' scalar_exp ')'	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CharLengthExpr(dynamic_cast<Expr*>(($<nodeval>3)->release()),false)); }
	    ;


/* should be scaler_exp '[' scalar_exp ']' but it causes conflicts.  need to debug */
array_at_exp : column_ref '[' scalar_exp ']'
  {
    $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kARRAY_AT, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release())));
  }
  ;

	/* scalar expressions */

scalar_exp:
		scalar_exp '+' scalar_exp { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kPLUS, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	scalar_exp '-' scalar_exp { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kMINUS, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	scalar_exp '*' scalar_exp { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kMULTIPLY, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	scalar_exp '/' scalar_exp { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kDIVIDE, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	scalar_exp '%' scalar_exp { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kMODULO, dynamic_cast<Expr*>(($<nodeval>1)->release()), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	|	MOD '(' scalar_exp ',' scalar_exp ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kMODULO, dynamic_cast<Expr*>(($<nodeval>3)->release()), dynamic_cast<Expr*>(($<nodeval>5)->release()))); }
	|	'+' scalar_exp %prec UMINUS { $<nodeval>$ = $<nodeval>2; }
	|	'-' scalar_exp %prec UMINUS { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new OperExpr(kUMINUS, dynamic_cast<Expr*>(($<nodeval>2)->release()), nullptr)); }
	|	atom { $<nodeval>$ = $<nodeval>1; }
	|	column_ref { $<nodeval>$ = $<nodeval>1; }
	|	function_ref { $<nodeval>$ = $<nodeval>1; }
	|	'(' scalar_exp ')' { $<nodeval>$ = $<nodeval>2; }
	| CAST '(' general_exp AS data_type ')'
	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CastExpr(dynamic_cast<Expr*>(($<nodeval>3)->release()), dynamic_cast<SQLType*>(($<nodeval>5)->release()))); }
	| case_exp { $<nodeval>$ = $<nodeval>1; }
  | charlength_exp { $<nodeval>$ = $<nodeval>1; }
  | array_at_exp { $<nodeval>$ = $<nodeval>1; }
	;

select_entry:
		{ throw std::runtime_error("Empty select entry"); }
	| general_exp { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SelectEntry(dynamic_cast<Expr*>(($<nodeval>1)->release()), nullptr)); }
	| general_exp NAME { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SelectEntry(dynamic_cast<Expr*>(($<nodeval>1)->release()), ($<stringval>2)->release())); }
	| general_exp AS NAME { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SelectEntry(dynamic_cast<Expr*>(($<nodeval>1)->release()), ($<stringval>3)->release())); }
	;

select_entry_commalist:
		{ throw std::runtime_error("Empty select entry list"); }
	|	select_entry { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
	|	select_entry_commalist ',' select_entry
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

atom:
		literal { $<nodeval>$ = $<nodeval>1; }
	|	USER { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new UserLiteral()); }
	/* |	NULLX { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new NullLiteral()); } */
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
		NAME '(' '*' ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new FunctionRef(($<stringval>1)->release())); }
	|	NAME '(' DISTINCT general_exp ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new FunctionRef(($<stringval>1)->release(), true, dynamic_cast<Expr*>(($<nodeval>4)->release()))); }
	|	NAME '(' ALL general_exp ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new FunctionRef(($<stringval>1)->release(), dynamic_cast<Expr*>(($<nodeval>4)->release()))); }
	|	NAME '(' general_exp ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new FunctionRef(($<stringval>1)->release(), dynamic_cast<Expr*>(($<nodeval>3)->release()))); }
	;

literal:
		STRING { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new StringLiteral(($<stringval>1)->release())); }
	|	INTNUM { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new IntLiteral($<intval>1)); }
	|	NOW '(' ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TimestampLiteral()); }
	|	DATETIME '(' general_exp ')' { delete dynamic_cast<Expr*>(($<nodeval>3)->release()); $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new TimestampLiteral()); }
	|	FIXEDNUM { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new FixedPtLiteral(($<stringval>1)->release())); }
	|	FLOAT { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new FloatLiteral($<floatval>1)); }
	|	DOUBLE { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new DoubleLiteral($<doubleval>1)); }
	|	data_type STRING { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new CastExpr(new StringLiteral(($<stringval>2)->release()), dynamic_cast<SQLType*>(($<nodeval>1)->release()))); }
	|	'{' opt_literal_commalist '}' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ArrayLiteral(reinterpret_cast<std::list<Expr*>*>(($<listval>2)->release()))); }
	|	ARRAY '[' opt_literal_commalist ']' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ArrayLiteral(reinterpret_cast<std::list<Expr*>*>(($<listval>3)->release()))); }
	|	NULLX { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new NullLiteral()); }
	;

literal_commalist:
		literal { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 1, $<nodeval>1); }
	|	literal_commalist ',' literal
	{
		$<listval>$ = $<listval>1;
		$<listval>$->push_back($<nodeval>3);
	}
	;

opt_literal_commalist:
    { $<listval>$ = TrackedListPtr<Node>::make(lexer.parsed_node_list_tokens_, 0); }
  | literal_commalist
  ;

	/* miscellaneous */

table:
	NAME
	{
		const auto uc_col_name = boost::to_upper_copy<std::string>(*($<stringval>1)->get());
		if (reserved_keywords.find(uc_col_name) != reserved_keywords.end()) {
			errors_.push_back("Cannot use a reserved keyword as table name: " + *($<stringval>1)->get());
		}
		$<stringval>$ = $<stringval>1;
	}
	/* |	NAME '.' NAME { $$ = new TableRef(($<stringval>1)->release(), ($<stringval>3)->release()); } */
    | 	QUOTED_IDENTIFIER { $<stringval>$ = $<stringval>1; }
	;

opt_table:
		{ $<nodeval>$ = TrackedPtr<Node>::makeEmpty(); }
	|	table
	;
username:
        NAME | EMAIL | DASHEDNAME | QUOTED_IDENTIFIER
    ;

rolenames:
		rolename { $<slistval>$ = TrackedListPtr<std::string>::make(lexer.parsed_str_list_tokens_, 1, $<stringval>1); }
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
		grantee { $<slistval>$ = TrackedListPtr<std::string>::make(lexer.parsed_str_list_tokens_, 1, $<stringval>1); }
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
		privilege { $<slistval>$ = TrackedListPtr<std::string>::make(lexer.parsed_str_list_tokens_, 1, $<stringval>1); }
	|	privileges ',' privilege
	{
		$<slistval>$ = $<slistval>1;
		$<slistval>$->push_back($<stringval>3);
	}
	;

privilege:
		ALL { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "ALL"); }
	|	ALL PRIVILEGES { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "ALL"); }
	|	CREATE { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "CREATE"); }
	|	SELECT { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "SELECT"); }
	|	INSERT { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "INSERT"); }
	|	TRUNCATE { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "TRUNCATE"); }
	|	UPDATE { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "UPDATE"); }
	|	DELETE { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "DELETE"); }
	| 	ALTER { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "ALTER"); }
	|	DROP { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "DROP"); }
	|	VIEW { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "VIEW"); }
	|	EDIT { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "EDIT"); }
	|	ACCESS { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "ACCESS"); }
	|	CREATE SERVER { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "CREATE SERVER"); }
	|	CREATE TABLE { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "CREATE TABLE"); }
	|	CREATE VIEW { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "CREATE VIEW"); }
	|	SELECT VIEW { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "SELECT VIEW"); }
	|	DROP VIEW { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "DROP VIEW"); }
	|	DROP SERVER { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "DROP SERVER"); }
	|	CREATE DASHBOARD { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "CREATE DASHBOARD"); }
	|	EDIT DASHBOARD { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "EDIT DASHBOARD"); }
	|	VIEW DASHBOARD { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "VIEW DASHBOARD"); }
	|	DELETE DASHBOARD { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "DELETE DASHBOARD"); }
	|	VIEW SQL EDITOR { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "VIEW SQL EDITOR"); }
	;

privileges_target_type:
		DATABASE { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "DATABASE"); }
	|	TABLE { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "TABLE"); }
	|	DASHBOARD { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "DASHBOARD"); }
	|	VIEW { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "VIEW"); }
	|	SERVER { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , "SERVER"); }
	;

privileges_target:
		NAME
	|	INTNUM { $<stringval>$ = TrackedPtr<std::string>::make(lexer.parsed_str_tokens_ , std::to_string($<intval>1)); }
    ;


column_ref:
		NAME { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnRef(($<stringval>1)->release())); }
	|	NAME '.' NAME	{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnRef(($<stringval>1)->release(), ($<stringval>3)->release())); }
	| NAME '.' '*' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new ColumnRef(($<stringval>1)->release(), nullptr)); }
	/* |	NAME '.' NAME '.' NAME { $$ = new ColumnRef(($<stringval>1)->release(), ($<stringval>3)->release(), ($<stringval>5)->release()); } */
	;

non_neg_int: INTNUM
		{
			if ($<intval>1 < 0)
				throw std::runtime_error("No negative number in type definition.");
			$<intval>$ = $<intval>1;
		}

		/* data types */

data_type:
		BIGINT { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kBIGINT)); }
	|	TEXT { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTEXT)); }
	|	BOOLEAN { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kBOOLEAN)); }
	|	CHARACTER { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kCHAR)); }
	|	CHARACTER '(' non_neg_int ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kCHAR, $<intval>3)); }
	|	NUMERIC { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kNUMERIC)); }
	|	NUMERIC '(' non_neg_int ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kNUMERIC, $<intval>3)); }
	|	NUMERIC '(' non_neg_int ',' non_neg_int ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kNUMERIC, $<intval>3, $<intval>5, false)); }
	|	DECIMAL { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDECIMAL)); }
	|	DECIMAL '(' non_neg_int ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDECIMAL, $<intval>3)); }
	|	DECIMAL '(' non_neg_int ',' non_neg_int ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDECIMAL, $<intval>3, $<intval>5, false)); }
	|	INTEGER { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kINT)); }
	|	TINYINT { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTINYINT)); }
	|	SMALLINT { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kSMALLINT)); }
	|	FLOAT { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kFLOAT)); }
	/* |	FLOAT '(' non_neg_int ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kFLOAT, $<intval>3)); } */
	|	REAL { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kFLOAT)); }
	|	DOUBLE PRECISION { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDOUBLE)); }
	|	DOUBLE { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDOUBLE)); }
	| DATE { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kDATE)); }
	| TIME { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTIME)); }
	| TIME '(' non_neg_int ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTIME, $<intval>3)); }
	| TIMESTAMP { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTIMESTAMP)); }
	| TIMESTAMP '(' non_neg_int ')' { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(kTIMESTAMP, $<intval>3)); }
	| geo_type { $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(static_cast<SQLTypes>($<intval>1), static_cast<int>(kGEOMETRY), 0, false)); }
        /* | geography_type { $<nodeval>$ = $<nodeval>1; } */
	| geometry_type { $<nodeval>$ = $<nodeval>1; }
	| data_type '[' ']'
	{
		$<nodeval>$ = $<nodeval>1;
		if (dynamic_cast<SQLType*>(($<nodeval>$)->get())->get_is_array())
		  throw std::runtime_error("array of array not supported.");
		dynamic_cast<SQLType*>(($<nodeval>$)->get())->set_is_array(true);
	}
	| data_type '[' non_neg_int ']'
	{
		$<nodeval>$ = $<nodeval>1;
		if (dynamic_cast<SQLType*>(($<nodeval>$)->get())->get_is_array())
		  throw std::runtime_error("array of array not supported.");
		dynamic_cast<SQLType*>(($<nodeval>$)->get())->set_is_array(true);
		dynamic_cast<SQLType*>(($<nodeval>$)->get())->set_array_size($<intval>3);
	}
	;

geo_type:	POINT { $<intval>$ = kPOINT; }
	|	LINESTRING { $<intval>$ = kLINESTRING; }
	|	POLYGON { $<intval>$ = kPOLYGON; }
	|	MULTIPOLYGON { $<intval>$ = kMULTIPOLYGON; }
	;

geography_type:	GEOGRAPHY '(' geo_type ')'
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(static_cast<SQLTypes>($<intval>3), static_cast<int>(kGEOGRAPHY), 4326, false)); }
	|	GEOGRAPHY '(' geo_type ',' INTNUM ')'
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(static_cast<SQLTypes>($<intval>3), static_cast<int>(kGEOGRAPHY), $<intval>5, false)); }

geometry_type:	GEOMETRY '(' geo_type ')'
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(static_cast<SQLTypes>($<intval>3), static_cast<int>(kGEOMETRY), 0, false)); }
	|	GEOMETRY '(' geo_type ',' INTNUM ')'
		{ $<nodeval>$ = TrackedPtr<Node>::make(lexer.parsed_node_tokens_, new SQLType(static_cast<SQLTypes>($<intval>3), static_cast<int>(kGEOMETRY), $<intval>5, false)); }

	/* the various things you can name */

column:
	NAME
	{
		const auto uc_col_name = boost::to_upper_copy<std::string>(*($<stringval>1)->get());
		if (reserved_keywords.find(uc_col_name) != reserved_keywords.end()) {
			errors_.push_back("Cannot use a reserved keyword as column name: " + *($<stringval>1)->get());
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
