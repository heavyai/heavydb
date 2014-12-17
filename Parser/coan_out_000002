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
#include <cstdint>
#include "../Shared/sqltypes.h"
#include "../Shared/sqldefs.h"

namespace Parser {
	class Node {
	public:
		// for debugging only
		virtual string to_string() {
			return "NOT IMPLEMENTED!";
		}
	}

	class Expr : public Node {
		// intentionally empty
	}

	class Literal : public Expr {
		/* intentionally empty */
	}

	class NullLiteral : public Literal {
		/* intentionally empty */
	}

	class StringLiteral : public Literal {
		public:
			StringLiteral(const string &s) : stringval(s) {}
			const string &get_stringval() { return stringval; }
		private:
			string stringval;
	}

	class IntLiteral : public Literal {
		public:
			IntLiteral(int32_t i) : intval(i) {}
			int32_t get_intval() { return intval; }
		private:
			int32_t intval;
	}

	class FixedPtLiteral : public Literal {
		public:
			FixedPtLiteral(const string &n) : fixedptval(n) {}
			const string &get_fixedptval() { return fixedptval; }
		private:
			string fixedptval;
	}

	class FloatLiteral : public Literal {
		public:
			FloatLiteral(float f) : floatval(f) {}
			float get_floatval() { return floatval; }
		private:
			float floatval;
	}

	class DoubleLiteral : public Literal {
		public:
			DoubleLiteral(double d) : doubleval(d) {}
			double get_doubleval() { return doubleval; }
		private:
			double doubleval;
	}

	class UserLiteral : public Literal {
		// intentionally empty
	}
			
	class OperExpr : public Expr {
		public:
			OperExpr(SQLOps t, Expr *l, Expr *r) : optype(t), left(l), right(r) {}
			Expr *get_left() { return left; }
			Expr *get_right() { return right; }
		private:
			SQLOps	optype;
			Expr *left;
			Expr *right;
	}

	class SubqueryExpr : public Expr {
		public:
			SubqueryExpr(QuerySpec *q) : qualifer(kANY), query(q) {}
			QuerySpec *get_query() { return query; }
			SQLQualifier get_qualifier() { return qualifier; }
			void set_qualifer(SQLOps ql) { qualifer = ql; }
		private:
			SQLQualifier qualifier;
			QuerySpec *query;
	}

	class IsNullExpr : public Expr {
		public:
			IsNullExpr(bool n) : is_not(n) {}
			bool get_is_not() { return is_not; }
		private:
			bool is_not;
	}

	class InExpr : public Expr {
		public:
			bool get_is_not() { return is_not; }
			Expr *get_arg() { return arg; }
		private:
			bool is_not;
			Expr *arg;
	}

	class InSubquery : public InExpr {
		public:
			InSubquery(bool n, Expr *a, SubqueryExpr *q) : is_not(n), arg(a), subquery(q) {}
			SubqueryExpr *get_subquery() { return subquery; }
		private:
			SubqueryExpr *subquery;
	}

	class InValues : public InExpr {
		public:
			InValues(bool n, Expr *a, list<Expr*> *v) : is_not(n), arg(a), value_list(v) {}
			list<Expr*> *get_value_list() { return value_list; }
		private:
			list<Expr*> *value_list;
	}

	class BetweenExpr : public Expr {
		public:
			BetweenExpr(bool n, Expr *a, Expr *l, Expr *u) : is_not(n), arg(a), lower(l), upper(u) {}
			bool get_is_not() { return is_not; }
			Expr *get_arg() { return arg; }
			Expr *get_lower() { return lower; }
			Expr *get_upper() { return upper; }
		private:
			bool is_not;
			Expr *arg;
			Expr *lower;
			Expr *upper;
	}

	class LikeExpr : public Expr {
		public:
			LikeExpr(bool n, Expr *a, Expr *l, Expr *e) : is_not(n), arg(a), like_string(l), escape_string(e) {}
			bool get_is_not() { return is_not; }
			Expr *get_arg() { return arg; }
			Expr *get_like_string() { return like_string; }
			Expr *get_escape_string() { return escape_string; }
		private:
			bool is_not;
			Expr *arg;
			Expr *like_string;
			Expr *escape_string;
	}

	class ExistsExpr : public Expr {
		public:
			ExistsExpr(QuerySpec *q) : query(q) {}
			QuerySpec *get_query() { return query; }
		private:
			QuerySpec *query;
	}

	class ColumnRef : public Expr {
		public:
			ColumnRef(const string &n1) : table(""), column(n1) {}
			ColumnRef(const string &n1, const string &n2) : table(n1), column(n2) {}
			const string &get_table() { return table; }
			const string &get_column() { return column; }
		private:
			string table;
			string column;
	}

	class TableRef : public Node {
		public:
			TableRef(const string &t) : table_name(t), range_var("") {}
			TableRef(const string &t, const string &r) : table_name(t), range_var(r) {}
			const string &get_table_name() { return table_name; }
			const string &get_range_var() { return range_var; }
		private:
			string table_name;
			string range_var;
	}

	class Stmt : public Node {
		// intentionally empty
	}

	class TableElement : public Node {
		// intentionally empty
	}

	class SQLType : public Node {
		public:
			SQLType(SQLTypes t) : type(t), param1(0), param2(0) {}
			SQLType(SQLTypes t, int p1) : type(t), param1(p1), param2(0) {}
			SQLType(SQLTypes t, int p1, int p2) : type(t), param1(p1), param2(p2) {}
			SQLTypes get_type() { return type; }
			int get_param1() { return param1; }
			int get_param2() { return param2; }
		private:
			SQLTypes	type;
			int param1; // e.g. for NUMERIC(10).  0 means unspecified.
			int param2; // e.g. for NUMERIC(10,3). 0 means unspecified.
	}
			
	class ColumnConstraintDef : public Node {
		public:
			ColumnConstraintDef(bool n, bool u, bool p, Literal *d) :
				notnull(n), unique(u), is_primarykey(p), default(d), check_condition(nullptr), foreign_table(nullptr), foreign_column("") {}
			ColumnConstraintDef(Expr *c) : notnull(false), unique(false), is_primarykey(false), default(nullptr), check_condition(c), foreign_table(nullptr), foreign_column("") {}
			ColumnConstraintDef(const string &t, const string &c) : notnull(false), unique(false), is_primarykey(false), default(nullptr), check_condition(c), foreign_table(t), foreign_column(c) {}
			bool get_notnull() { return notnull; }
			bool get_unique() { return unique; }
			bool get_is_primarykey() { return is_primarykey; }
			Literal *get_default() { return default; }
			Expr *get_check_condition() { return check_condition; }
			const string &get_foreign_table() { return foreign_table; }
			const string &get_foreign_column() { return foreign_column; }
		private:
			bool notnull;
			bool unique;
			bool is_primarykey;
			Literal *default;
			Expr *check_condition;
			string foreign_table;
			string foreign_column;
	}

	class ColumnDef : public TableElement {
		public:
			ColumnDef(const string &c, SQLType *t, ColumnConstraintDef *cc) : column_name(c), column_type(t), column_constraint(cc) {}
			const string &get_column_name() { return column_name; }
			SQLType *get_column_type() { return column_type; }
			ColumnConstraintDef *get_column_constraint() { return column_constraint; }
		private:
			string column_name;
			SQLType *column_type;
			ColumnConstraintDef *column_constraint;
	}

	class TableConstraintDef : public TableElement {
		// intentionally empty
	}

	class UniqueDef : public TableConstraintDef {
		public:
			UniqueDef(bool p, list<string> *cl) : is_primarykey(p), column_list(cl) {}
			bool get_is_primarykey() { return is_primarykey; }
			list<string> *get_column_list() { return column_list; }
		private:
			bool is_primarykey;
			list<string> *column_list;
	}
	
	class ForeignKeyDef : public TableConstraintDef {
		public:
			ForeignKeyDef(list<string> *cl, const string &t, list<string> *fcl) : column_list(cl), foreign_table(t), foreign_column_list(fcl) {}
			list<string> *get_column_list() { return column_list; }
			const string &get_foreign_table() { return foreign_table; }
			list<string> *get_foreign_column_list() { return foreign_column_list; }
		private:
			list<string> *column_list;
			string foreign_table;
			list<string> *foreign_column_list;
	}

	class CheckDef : public TableConstraintDef {
		public:
			CheckDef(Expr *c): check_condition(c) {}
			Expr *get_check_condition() { return check_condition; }
		private:
			Expr *check_condition;
	}

	class FunctionRef : public Expr {
		public:
			FunctionRef(const string &n) : name(n), distinct(false), arg(nullptr) {}
			FunctionRef(const string &n, Expr *a) : name(n), distinct(false), arg(a) {}
			FunctionRef(const string &n, bool d, Expr *a) : name(n), distinct(d), arg(a) {} 
			const string &get_name() { return name; }
			bool get_distinct() { return distinct; }
			Expr *get_arg() { return arg; }
		private:
			string name;
			bool distinct; // only true for COUNT(DISTINCT x)
			Expr *arg; // for COUNT, nullptr means '*'
	}

	class CreateTableStmt : public Stmt {
		public:
			CreateTableStmt(tab, list<TableElement*> * table_elems) :
			table(tab), table_element_list(table_elems) {}
			const string &get_table() { return table; }
			list<TableElement*> *get_table_element_list() { return table_element_list; }
		private:
			Table *table;
			list<TableElement*> *table_element_list;
	}

	class QueryExpr : public Node {
		// intentionally empty
	}

	class UnionQuery : public QueryExpr {
		public:
			UnionQuery(bool u, QueryExpr *l, QueryExpr *r) : is_unionall(u), left(l), right(r) {}
			bool get_is_unionall() { return is_unionall; }
			QueryExpr *get_left() { return left; }
			QueryExpr *get_right() { return right; }
		private:
			bool is_unionall;
			QueryExpr *left;
			QueryExpr *right;
	}

	class QuerySpec : public QueryExpr {
		public:
			QuerySpec(bool d, list<Expr*> *s, list<TableRef*> *f, Expr *w, list<ColumnRef*> *g, Expr *h) : is_distinct(d), select_clause(s), from_clause(f), where_clause(w), groupby_clause(g), having_clause(h) {}
			bool get_is_distinct() { return is_distinct; }
			list<Expr*> *get_select_clause() { return select_list; }
			list<TableRef*> *get_from_clause() { return from_clause; }
			Expr *get_where_clause() { return where_clause; }
			list<ColumnRef*> *get_groupby_clause() { return groupby_clause; }
			Expr *get_having_clause() { return having_clause; }
		private:
			bool is_distinct;
			list<Expr*> *select_clause; /* nullptr means SELECT * */
			list<TableRef*> *from_clause;
			Expr *where_clause;
			list<ColumnRef*> *groupby_clause;
			Expr *having_clause;
	}

	class SelectStmt : public Stmt {
		public:
			SelectStmt(QueryExpr *q, list<OrderSpec*> *o) : query_expr(q), orderby_clause(o) {}
			QueryExpr *get_query_expr() { return query_expr; }
			list<OrderSpec*> *get_orderby_clause() { return orderby_clause; }
		private:
			QueryExpr *query_expr;
			list<OrderSpec*> *orderby_clause;
	}

	class OrderSpec : public Node {
		public:
			OrderSpec(int n, ColumnRef *c, bool d) : colno(n), column(c), is_desc(d) {}
			int get_colno() { return colno; }
			ColumnRef *get_column { return column; }
			bool get_is_desc() { return is_desc; }
		private:
			int colno; /* 0 means use column name */
			ColumnRef *column;
			bool is_desc;
	}

	class CreateViewStmt : public Stmt {
		public:
			CreateViewStmt(const string &v, list<string> *c, QuerySpec *q, bool ck) : view_name(v), column_list(c), query(q), checkoption(ck) {}
			const string &get_view_name() { return view_name; }
			list<string> *get_column_list() { return column_list; }
			QuerySpec *get_query() { return query; }
			bool get_checkoption() { return checkoption; }
		private:
			string view_name;
			list<string> *column_list;
			QuerySpec *query;
			bool checkoption;
	}

	class InsertStmt : public Stmt {
		public:
			const string &get_table() { return table; }
			list<string> *get_column_list { return column_list; }
		private:
			string table;
			list<string> *column_list;
	}

	class InsertValuesStmt : public InsertStmt {
		public:
			InsertValueStmt(const string &t, list<string> *c, list<Expr*> *v) : table(t), column_list(c), value_list(v) {}
			list<Expr*> *get_value_list() { return value_list; }
		private:
			list<Expr*> *value_list;
	}

	class InsertQueryStmt : public InsertStmt {
		public:
			InsertQueryStmt(const string &t, list<string> *c, QuerySpec *q) : table(t), column_list(c), query(q) {}
			QuerySpec *get_query() { return query; }
		private:
			QuerySpec *query;
	}

	class Assignment : public Node {
		public:
			Assignment(const string &c, Expr *a) : column(c), assignment(a) {}
			const string &get_column() { return column; }
			Expr *get_assignment() { return assignment; }
		private:
			string column;
			Expr *assignment;
	}

	class UpdateStmt : public Stmt {
		public:
			UpdateStmt(const string &t, list<Assignment*> *a, Expr *w) : table(t), assignment_list(a), where_clause(w) {}
			const string &get_table() { return table; }
			list<Assignment*> *get_assignment_list() { return assignment_list; }
			Expr *get_where_clause() { return where_clause; }
		private:
			string table;
			list<Assignment*> *assignment_list;
			Expr *where_clause;
	}

	class DeleteStmt : public Stmt {
		public:
			DeleteStmt(const string &t, Expr *w) : table(t), where_clause(w) {}
			const string &get_table() { return table; }
			Expr *get_where_clause() { return where_clause; }
		private:
			string table;
			Expr *where_clause;
	}
}
