/**
 * @file		ParserNode.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Classes representing a parse tree
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef PARSER_NODE_H_
#define PARSER_NODE_H_

#include <list>
#include <string>
#include <cstring>
#include <cstdint>
#include "../Shared/sqltypes.h"
#include "../Shared/sqldefs.h"
#include "../Analyzer/Analyzer.h"
#include "../Catalog/Catalog.h"

namespace Parser {

	/*
	 * @type Node
	 * @brief root super class for all nodes in a pre-Analyzer
	 * parse tree.
	 */
	class Node {
	public:
		virtual ~Node() {}
		// for debugging only
		virtual std::string to_string() {
			return "NOT IMPLEMENTED!";
		}
	};

	/*
	 * @type Expr
	 * @brief root super class for all expression nodes
	 */
	class Expr : public Node {
		public:
			/*
			 * @brief Performs semantic analysis on the expression node
			 * @param catalog The catalog object for the current database
			 * @return An Analyzer::Expr object for the expression post semantic analysis
			 */
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const = 0;
	};

	/*
	 * @type Literal
	 * @brief root super class for all literals
	 */
	class Literal : public Expr {
		public:
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const = 0;
	};

	/*
	 * @type NullLiteral
	 * @brief the Literal NULL
	 */
	class NullLiteral : public Literal {
		public:
			NullLiteral() {}
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
	};

	/*
	 * @type StringLiteral
	 * @brief the literal for string constants
	 */
	class StringLiteral : public Literal {
		public:
			StringLiteral(std::string *s) : stringval(s) {}
			virtual ~StringLiteral() { delete stringval; }
			const std::string *get_stringval() { return stringval; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			std::string *stringval;
	};

	/*
	 * @type IntLiteral
	 * @brief the literal for integer constants
	 */
	class IntLiteral : public Literal {
		public:
			IntLiteral(int64_t i) : intval(i) {}
			int64_t get_intval() { return intval; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			int64_t intval;
	};

	/*
	 * @type FixedPtLiteral
	 * @brief the literal for DECIMAL and NUMERIC
	 */
	class FixedPtLiteral : public Literal {
		public:
			FixedPtLiteral(std::string *n) : fixedptval(n) {}
			virtual ~FixedPtLiteral() { delete fixedptval; }
			const std::string *get_fixedptval() { return fixedptval; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			std::string *fixedptval;
	};

	/*
	 * @type FloatLiteral
	 * @brief the literal for FLOAT or REAL
	 */
	class FloatLiteral : public Literal {
		public:
			FloatLiteral(float f) : floatval(f) {}
			float get_floatval() { return floatval; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			float floatval;
	};

	/*
	 * @type DoubleLiteral
	 * @brief the literal for DOUBLE PRECISION
	 */
	class DoubleLiteral : public Literal {
		public:
			DoubleLiteral(double d) : doubleval(d) {}
			double get_doubleval() { return doubleval; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			double doubleval;
	};

	/*
	 * @type UserLiteral
	 * @brief the literal for USER
	 */
	class UserLiteral : public Literal {
		public:
			UserLiteral() {}
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
	};
			
	/*
	 * @type OperExpr
	 * @brief all operator expressions
	 */
	class OperExpr : public Expr {
		public:
			OperExpr(SQLOps t, Expr *l, Expr *r) : optype(t), left(l), right(r) {}
			virtual ~OperExpr() { delete left; delete right; }
			SQLOps get_optype() { return optype; }
			const Expr *get_left() { return left; }
			const Expr *get_right() { return right; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			SQLOps	optype;
			Expr *left;
			Expr *right;
	};

	// forward reference of QuerySpec
	class QuerySpec;

	/*
	 * @type SubqueryExpr
	 * @brief expression for subquery
	 */
	class SubqueryExpr : public Expr {
		public:
			SubqueryExpr(QuerySpec *q) : qualifier(kANY), query(q) {}
			virtual ~SubqueryExpr();
			const QuerySpec *get_query() { return query; }
			SQLQualifier get_qualifier() { return qualifier; }
			void set_qualifier(SQLQualifier ql) { qualifier = ql; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			SQLQualifier qualifier;
			QuerySpec *query;
	};

	class IsNullExpr : public Expr {
		public:
			IsNullExpr(bool n, Expr *a) : is_not(n), arg(a) {}
			virtual ~IsNullExpr() { delete arg; }
			bool get_is_not() { return is_not; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			bool is_not;
			Expr *arg;
	};

	/*
	 * @type InExpr
	 * @brief expression for the IS NULL predicate
	 */
	class InExpr : public Expr {
		public:
			InExpr(bool n, Expr *a) : is_not(n), arg(a) {}
			virtual ~InExpr() { delete arg; }
			bool get_is_not() { return is_not; }
			const Expr *get_arg() { return arg; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const = 0;
		protected:
			bool is_not;
			Expr *arg;
	};

	/*
	 * @type InSubquery
	 * @brief expression for the IN (subquery) predicate
	 */
	class InSubquery : public InExpr {
		public:
			InSubquery(bool n, Expr *a, SubqueryExpr *q) : InExpr(n, a), subquery(q) {}
			virtual ~InSubquery() { InExpr::~InExpr(); delete subquery; }
			const SubqueryExpr *get_subquery() { return subquery; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			SubqueryExpr *subquery;
	};

	/*
	 * @type InValues
	 * @brief expression for IN (val1, val2, ...)
	 */
	class InValues : public InExpr {
		public:
			InValues(bool n, Expr *a, std::list<Expr*> *v) : InExpr(n, a), value_list(v) {}
			virtual ~InValues();
			const std::list<Expr*> *get_value_list() { return value_list; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			std::list<Expr*> *value_list;
	};

	/*
	 * @type BetweenExpr
	 * @brief expression for BETWEEN lower AND upper
	 */
	class BetweenExpr : public Expr {
		public:
			BetweenExpr(bool n, Expr *a, Expr *l, Expr *u) : is_not(n), arg(a), lower(l), upper(u) {}
			virtual ~BetweenExpr();
			bool get_is_not() { return is_not; }
			const Expr *get_arg() { return arg; }
			const Expr *get_lower() { return lower; }
			const Expr *get_upper() { return upper; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			bool is_not;
			Expr *arg;
			Expr *lower;
			Expr *upper;
	};

	/*
	 * @type LikeExpr
	 * @brief expression for the LIKE predicate
	 */
	class LikeExpr : public Expr {
		public:
			LikeExpr(bool n, Expr *a, Expr *l, Expr *e) : is_not(n), arg(a), like_string(l), escape_string(e) {}
			virtual ~LikeExpr();
			bool get_is_not() { return is_not; }
			const Expr *get_arg() { return arg; }
			const Expr *get_like_string() { return like_string; }
			const Expr *get_escape_string() { return escape_string; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			bool is_not;
			Expr *arg;
			Expr *like_string;
			Expr *escape_string;
	};

	/*
	 * @type ExistsExpr
	 * @brief expression for EXISTS (subquery)
	 */
	class ExistsExpr : public Expr {
		public:
			ExistsExpr(QuerySpec *q) : query(q) {}
			virtual ~ExistsExpr();
			const QuerySpec *get_query() { return query; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			QuerySpec *query;
	};

	/*
	 * @type ColumnRef
	 * @brief expression for a column reference
	 */
	class ColumnRef : public Expr {
		public:
			ColumnRef(std::string *n1) : table(nullptr), column(n1) {}
			ColumnRef(std::string *n1, std::string *n2) : table(n1), column(n2) {}
			virtual ~ColumnRef();
			const std::string *get_table() const { return table; }
			const std::string *get_column() const { return column; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			std::string *table;
			std::string *column; // can be nullptr in the t.* case
	};

	/*
	 * @type FunctionRef
	 * @brief expression for a function call
	 */
	class FunctionRef : public Expr {
		public:
			FunctionRef(std::string *n) : name(n), distinct(false), arg(nullptr) {}
			FunctionRef(std::string *n, Expr *a) : name(n), distinct(false), arg(a) {}
			FunctionRef(std::string *n, bool d, Expr *a) : name(n), distinct(d), arg(a) {} 
			virtual ~FunctionRef();
			const std::string *get_name() { return name; }
			bool get_distinct() { return distinct; }
			Expr *get_arg() { return arg; }
			virtual Analyzer::Expr *analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			std::string *name;
			bool distinct; // only true for COUNT(DISTINCT x)
			Expr *arg; // for COUNT, nullptr means '*'
	};

	/*
	 * @type TableRef
	 * @brief table reference in FROM clause
	 */
	class TableRef : public Node {
		public:
			TableRef(std::string *t) : table_name(t), range_var(nullptr) {}
			TableRef(std::string *t, std::string *r) : table_name(t), range_var(r) {}
			virtual ~TableRef();
			const std::string *get_table_name() { return table_name; }
			const std::string *get_range_var() { return range_var; }
		private:
			std::string *table_name;
			std::string *range_var;
	};

	/*
	 * @type Stmt
	 * @brief root super class for all SQL statements
	 */
	class Stmt : public Node {
		// intentionally empty
	};

	/*
	 * @type DMLStmt
	 * @brief DML Statements
	 */
	class DMLStmt : public Stmt {
		public:
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const = 0;
	};

	/*
	 * @type DDLStmt
	 * @brief DDL Statements
	 */
	class DDLStmt : public Stmt {
		public:
			virtual void execute(Catalog_Namespace::Catalog &catalog) = 0;
	};

	/*
	 * @type TableElement
	 * @brief elements in table definition
	 */
	class TableElement : public Node {
		// intentionally empty
	};

	/*
	 * @type SQLType
	 * @brief class that captures type, predication and scale.
	 */
	class SQLType : public Node {
		public:
			SQLType(SQLTypes t) : type(t), param1(0), param2(0) {}
			SQLType(SQLTypes t, int p1) : type(t), param1(p1), param2(0) {}
			SQLType(SQLTypes t, int p1, int p2) : type(t), param1(p1), param2(p2) {}
			SQLTypes get_type() const { return type; }
			int get_param1() const { return param1; }
			int get_param2() const { return param2; }
		private:
			SQLTypes	type;
			int param1; // e.g. for NUMERIC(10).  0 means unspecified.
			int param2; // e.g. for NUMERIC(10,3). 0 means unspecified.
	};
			
	/*
	 * @type ColumnConstraintDef
	 * @brief integrity constraint on a column
	 */
	class ColumnConstraintDef : public Node {
		public:
			ColumnConstraintDef(bool n, bool u, bool p, Literal *d) :
				notnull(n), unique(u), is_primarykey(p), defaultval(d), check_condition(nullptr), foreign_table(nullptr), foreign_column(nullptr) {}
			ColumnConstraintDef(Expr *c) : notnull(false), unique(false), is_primarykey(false), defaultval(nullptr), check_condition(c), foreign_table(nullptr), foreign_column(nullptr) {}
			ColumnConstraintDef(std::string *t, std::string *c) : notnull(false), unique(false), is_primarykey(false), defaultval(nullptr), check_condition(nullptr), foreign_table(t), foreign_column(c) {}
			virtual ~ColumnConstraintDef();
			bool get_notnull() const { return notnull; }
			bool get_unique() const { return unique; }
			bool get_is_primarykey() const { return is_primarykey; }
			const Literal *get_defaultval() const { return defaultval; }
			const Expr *get_check_condition() const { return check_condition; }
			const std::string *get_foreign_table() const { return foreign_table; }
			const std::string *get_foreign_column() const { return foreign_column; }
		private:
			bool notnull;
			bool unique;
			bool is_primarykey;
			Literal *defaultval;
			Expr *check_condition;
			std::string *foreign_table;
			std::string *foreign_column;
	};

	/*
	 * @type ColumnDef
	 * @brief Column definition
	 */
	class ColumnDef : public TableElement {
		public:
			ColumnDef(std::string *c, SQLType *t, ColumnConstraintDef *cc) : column_name(c), column_type(t), column_constraint(cc) {}
			virtual ~ColumnDef();
			const std::string *get_column_name() { return column_name; }
			const SQLType *get_column_type() { return column_type; }
			const ColumnConstraintDef *get_column_constraint() { return column_constraint; }
		private:
			std::string *column_name;
			SQLType *column_type;
			ColumnConstraintDef *column_constraint;
	};

	/*
	 * @type TableConstraintDef
	 * @brief integrity constraint for table
	 */
	class TableConstraintDef : public TableElement {
		// intentionally empty
	};

	/*
	 * @type UniqueDef
	 * @brief uniqueness constraint
	 */
	class UniqueDef : public TableConstraintDef {
		public:
			UniqueDef(bool p, std::list<std::string*> *cl) : is_primarykey(p), column_list(cl) {}
			virtual ~UniqueDef();
			bool get_is_primarykey() { return is_primarykey; }
			std::list<std::string*> *get_column_list() { return column_list; }
		private:
			bool is_primarykey;
			std::list<std::string*> *column_list;
	};
	
	/*
	 * @type ForeignKeyDef
	 * @brief foreign key constraint
	 */
	class ForeignKeyDef : public TableConstraintDef {
		public:
			ForeignKeyDef(std::list<std::string*> *cl, std::string *t, std::list<std::string*> *fcl) : column_list(cl), foreign_table(t), foreign_column_list(fcl) {}
			virtual ~ForeignKeyDef();
			const std::list<std::string*> *get_column_list() { return column_list; }
			const std::string *get_foreign_table() { return foreign_table; }
			const std::list<std::string*> *get_foreign_column_list() { return foreign_column_list; }
		private:
			std::list<std::string*> *column_list;
			std::string *foreign_table;
			std::list<std::string*> *foreign_column_list;
	};

	/*
	 * @type CheckDef
	 * @brief Check constraint
	 */
	class CheckDef : public TableConstraintDef {
		public:
			CheckDef(Expr *c): check_condition(c) {}
			virtual ~CheckDef() { TableConstraintDef::~TableConstraintDef(); delete check_condition; }
			const Expr *get_check_condition() { return check_condition; }
		private:
			Expr *check_condition;
	};

	/*
	 * @type CreateTableStmt
	 * @brief CREATE TABLE statement
	 */
	class CreateTableStmt : public DDLStmt {
		public:
			CreateTableStmt(std::string *tab, std::list<TableElement*> * table_elems) :
			table(tab), table_element_list(table_elems) {}
			virtual ~CreateTableStmt();
			const std::string *get_table() { return table; }
			const std::list<TableElement*> *get_table_element_list() { return table_element_list; }
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *table;
			std::list<TableElement*> *table_element_list;
	};

	/*
	 * @type DropTableStmt
	 * @brief DROP TABLE statement
	 */
	class DropTableStmt : public DDLStmt {
		public:
			DropTableStmt(std::string *tab) : table(tab) {}
			virtual ~DropTableStmt() { delete table; }
			const std::string *get_table() { return table; }
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *table;
	};

	/*
	 * @type QueryExpr
	 * @brief query expression
	 */
	class QueryExpr : public Node {
	public:
		virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const = 0;
	};

	/*
	 * @type UnionQuery
	 * @brief UNION or UNION ALL queries
	 */
	class UnionQuery : public QueryExpr {
		public:
			UnionQuery(bool u, QueryExpr *l, QueryExpr *r) : is_unionall(u), left(l), right(r) {}
			virtual ~UnionQuery() { delete left; delete right; }
			bool get_is_unionall() { return is_unionall; }
			const QueryExpr *get_left() { return left; }
			const QueryExpr *get_right() { return right; }
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			bool is_unionall;
			QueryExpr *left;
			QueryExpr *right;
	};

	class SelectEntry : public Node {
		public:
			SelectEntry(Expr *e, std::string *r) : select_expr(e), alias(r) {}
			virtual ~SelectEntry();
			const Expr *get_select_expr() { return select_expr; }
			const std::string *get_alias() { return alias; }
		private:
			Expr *select_expr;
			std::string *alias;
	};

	/*
	 * @type QuerySpec
	 * @brief a simple query
	 */
	class QuerySpec : public QueryExpr {
		public:
			QuerySpec(bool d, std::list<SelectEntry*> *s, std::list<TableRef*> *f, Expr *w, std::list<ColumnRef*> *g, Expr *h) : is_distinct(d), select_clause(s), from_clause(f), where_clause(w), groupby_clause(g), having_clause(h) {}
			virtual ~QuerySpec();
			bool get_is_distinct() { return is_distinct; }
			const std::list<SelectEntry*> *get_select_clause() { return select_clause; }
			const std::list<TableRef*> *get_from_clause() { return from_clause; }
			const Expr *get_where_clause() { return where_clause; }
			const std::list<ColumnRef*> *get_groupby_clause() { return groupby_clause; }
			const Expr *get_having_clause() { return having_clause; }
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			bool is_distinct;
			std::list<SelectEntry*> *select_clause; /* nullptr means SELECT * */
			std::list<TableRef*> *from_clause;
			Expr *where_clause;
			std::list<ColumnRef*> *groupby_clause;
			Expr *having_clause;
			void analyze_from_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
			void analyze_select_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
			void analyze_where_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
			void analyze_group_by(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
			void analyze_having_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
	};

	/*
	 * @type OrderSpec
	 * @brief order spec for a column in ORDER BY clause
	 */
	class OrderSpec : public Node {
		public:
			OrderSpec(int n, ColumnRef *c, bool d, bool f) : colno(n), column(c), is_desc(d), nulls_first(f) {}
			virtual ~OrderSpec() { if (column != nullptr) delete column; }
			int get_colno() { return colno; }
			const ColumnRef *get_column() { return column; }
			bool get_is_desc() { return is_desc; }
			bool get_nulls_first() { return nulls_first; }
		private:
			int colno; /* 0 means use column name */
			ColumnRef *column;
			bool is_desc;
			bool nulls_first;
	};

	/*
	 * @type SelectStmt
	 * @brief SELECT statement
	 */
	class SelectStmt : public DMLStmt {
		public:
			SelectStmt(QueryExpr *q, std::list<OrderSpec*> *o) : query_expr(q), orderby_clause(o) {}
			virtual ~SelectStmt();
			const QueryExpr *get_query_expr() { return query_expr; }
			const std::list<OrderSpec*> *get_orderby_clause() { return orderby_clause; }
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			QueryExpr *query_expr;
			std::list<OrderSpec*> *orderby_clause;
	};

	/*
	 * @type CreateViewStmt
	 * @brief CREATE VIEW statement
	 */
	class CreateViewStmt : public DDLStmt {
		public:
			CreateViewStmt(std::string *v, std::list<std::string*> *c, QuerySpec *q, bool ck) : view_name(v), column_list(c), query(q), checkoption(ck) {}
			virtual ~CreateViewStmt();
			const std::string *get_view_name() { return view_name; }
			const std::list<std::string*> *get_column_list() { return column_list; }
			const QuerySpec *get_query() { return query; }
			bool get_checkoption() { return checkoption; }
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *view_name;
			std::list<std::string*> *column_list;
			QuerySpec *query;
			bool checkoption;
	};

	/*
	 * @type DropViewStmt
	 * @brief DROP VIEW statement
	 */
	class DropViewStmt : public DDLStmt {
		public:
			DropViewStmt(std::string *v) : view_name(v) {}
			virtual ~DropViewStmt() { delete view_name; };
			const std::string *get_view_name() { return view_name; }
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *view_name;
	};

	/*
	 * @type CreateDBStmt
	 * @brief CREATE DATABASE statement
	 */
	class CreateDBStmt : public DDLStmt {
		public:
			CreateDBStmt(std::string *n) : db_name(n) {}
			virtual ~CreateDBStmt() { delete db_name; }
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *db_name;
	};

	/*
	 * @type DropDBStmt
	 * @brief DROP DATABASE statement
	 */
	class DropDBStmt : public DDLStmt {
		public:
			DropDBStmt(std::string *n) : db_name(n) {}
			virtual ~DropDBStmt() { delete db_name; }
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *db_name;
	};

	/*
	 * @type CreateUserStmt
	 * @brief CREATE USER statement
	 */
	class CreateUserStmt : public DDLStmt {
		public:
			CreateUserStmt(std::string *n, std::string *o1, std::string *p, std::string *o2) : user_name(n), option1(o1), passwd(p), option2(o2) {}
			virtual ~CreateUserStmt();
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *user_name;
			std::string *option1;
			std::string *passwd;
			std::string *option2;
	};

	/*
	 * @type AlterUserStmt
	 * @brief ALTER USER statement
	 */
	class AlterUserStmt : public DDLStmt {
		public:
			AlterUserStmt(std::string *n, std::string *o1, std::string *p, std::string *o2) : user_name(n), option1(o1), passwd(p), option2(o2) {}
			virtual ~AlterUserStmt();
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *user_name;
			std::string *option1;
			std::string *passwd;
			std::string *option2;
	};

	/*
	 * @type DropUserStmt
	 * @brief DROP USER statement
	 */
	class DropUserStmt : public DDLStmt {
		public:
			DropUserStmt(std::string *n) : user_name(n) {}
			virtual ~DropUserStmt() { delete user_name; }
			virtual void execute(Catalog_Namespace::Catalog &catalog);
		private:
			std::string *user_name;
	};

	/*
	 * @type InsertStmt
	 * @brief super class for INSERT statements
	 */
	class InsertStmt : public DMLStmt {
		public:
			InsertStmt(std::string *t, std::list<std::string*> *c) : table(t), column_list(c) {}
			virtual ~InsertStmt();
			const std::string *get_table() { return table; }
			const std::list<std::string*> *get_column_list() { return column_list; }
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const = 0;
		private:
			std::string *table;
			std::list<std::string*> *column_list;
	};

	/*
	 * @type InsertValuesStmt
	 * @brief INSERT INTO ... VALUES ...
	 */
	class InsertValuesStmt : public InsertStmt {
		public:
			InsertValuesStmt(std::string *t, std::list<std::string*> *c, std::list<Expr*> *v) : InsertStmt(t, c), value_list(v) {}
			virtual ~InsertValuesStmt();
			const std::list<Expr*> *get_value_list() { return value_list; }
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			std::list<Expr*> *value_list;
	};

	/*
	 * @type InsertQueryStmt
	 * @brief INSERT INTO ... SELECT ...
	 */
	class InsertQueryStmt : public InsertStmt {
		public:
			InsertQueryStmt(std::string *t, std::list<std::string*> *c, QuerySpec *q) : InsertStmt(t, c), query(q) {}
			virtual ~InsertQueryStmt() { InsertStmt::~InsertStmt(); delete query; }
			const QuerySpec *get_query() { return query; }
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			QuerySpec *query;
	};

	/*
	 * @type Assignment
	 * @brief assignment in UPDATE statement
	 */
	class Assignment : public Node {
		public:
			Assignment(std::string *c, Expr *a) : column(c), assignment(a) {}
			virtual ~Assignment() { delete column; delete assignment; }
			const std::string *get_column() { return column; }
			const Expr *get_assignment() { return assignment; }
		private:
			std::string *column;
			Expr *assignment;
	};

	/*
	 * @type UpdateStmt
	 * @brief UPDATE statement
	 */
	class UpdateStmt : public DMLStmt {
		public:
			UpdateStmt(std::string *t, std::list<Assignment*> *a, Expr *w) : table(t), assignment_list(a), where_clause(w) {}
			virtual ~UpdateStmt();
			const std::string *get_table() { return table; }
			const std::list<Assignment*> *get_assignment_list() { return assignment_list; }
			const Expr *get_where_clause() { return where_clause; }
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			std::string *table;
			std::list<Assignment*> *assignment_list;
			Expr *where_clause;
	};

	/*
	 * @type DeleteStmt
	 * @brief DELETE statement
	 */
	class DeleteStmt : public DMLStmt {
		public:
			DeleteStmt(std::string *t, Expr *w) : table(t), where_clause(w) {}
			virtual ~DeleteStmt();
			const std::string *get_table() { return table; }
			const Expr *get_where_clause() { return where_clause; }
			virtual void analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const;
		private:
			std::string *table;
			Expr *where_clause;
	};
}
#endif // PARSERNODE_H_
