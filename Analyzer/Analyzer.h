/**
 * @file		Analyzer.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Defines data structures for the semantic analysis phase of query processing
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef ANALYZER_H
#define ANALYZER_H

#include <cstdint>
#include <string>
#include <vector>
#include <list>
#include "../Shared/sqltypes.h"
#include "../Shared/sqldefs.h"
#include "../Catalog/Catalog.h"

namespace Analyzer {

	/*
	 * @type Expr
	 * @brief super class for all expressions in parse trees and in query plans
	 */
	class Expr {
		public:
			Expr(SQLTypes t) { type_info.type = t; type_info.dimension = 0; type_info.scale = 0; }
			Expr(SQLTypes t, int d) { type_info.type = t; type_info.dimension = d; type_info.scale = 0; }
			Expr(SQLTypes t, int d, int s) { type_info.type = t; type_info.dimension = d; type_info.scale = s; }
			Expr(const SQLTypeInfo &ti) : type_info(ti) {}
			virtual ~Expr() {}
			const SQLTypeInfo &get_type_info() { return type_info; }
			virtual Expr *add_cast(const SQLTypeInfo &new_type_info);
			virtual void check_group_by(const std::list<Expr*> *groupby) const {};
		private:
			SQLTypeInfo type_info; // SQLTypeInfo of the return result of this expression
	};

	/*
	 * @type ColumnVar
	 * @brief expression that evaluates to the value of a column in a given row from a base table.
	 * It is used in parse trees and is only used in Scan nodes in a query plan 
	 * for scanning a table while Var nodes are used for all other plans.
	 */
	class ColumnVar : public Expr {
		public:
			ColumnVar(const SQLTypeInfo &ti, int r, int c) : Expr(ti), table_id(r), column_id(c) {}
			int get_table_id() const { return table_id; }
			int get_column_id() const { return column_id; }
			virtual void check_group_by(const std::list<Expr*> *groupby) const;
		private:
			int table_id; // index of the range table entry. 0 based
			int column_id; // index into the vector of columns. 0 based
	};

	/*
	 * @type Var
	 * @brief expression that evaluates to the value of a column in a given row generated
	 * from a query plan node.  It is only used in plan nodes above Scan nodes.
	 * The row can be produced by either the inner or the outer plan in case of a join.
	 * It inherits from ColumnVar to keep track of the lineage through the plan nodes.
	 * The rte_no will be set to -1 if the Var does not correspond to an original column value.
	 */
	class Var : public ColumnVar {
		public:
			Var(const SQLTypeInfo &ti, int r, int c, bool o, int v) : ColumnVar(ti, r, c), is_inner(o), varno(v) {}
			bool get_is_inner() { return is_inner; }
			int get_varno() { return varno; }
		private:
			bool is_inner; // indicate the row is produced by the inner plan
			int varno; // the column number in the row.
	};

	/*
	 * @type Constant
	 * @brief expression for a constant value
	 */
	class Constant : public Expr {
		public:
			Constant(SQLTypes t, bool n) : Expr (t), is_null(n) {}
			Constant(SQLTypes t, bool n, Datum v) : Expr (t), is_null(n), constval(v) {}
			Constant(const SQLTypeInfo &ti, bool n) : Expr(ti), is_null(n) {}
			Constant(const SQLTypeInfo &ti, bool n, Datum v) : Expr(ti), is_null(n), constval(v) {}
			bool get_is_null() { return is_null; }
			Datum get_constval() { return constval; }
			void set_constval(Datum d) { constval = d; }
		private:
			bool is_null; // constant is NULL
			Datum constval; // the constant value
	};

	/*
	 * @type UOper
	 * @brief represents unary operator expressions.  operator types include
	 * kUMINUS, kISNULL, kEXISTS, kCAST
	 */
	class UOper : public Expr {
		public:
			UOper(const SQLTypeInfo &ti, SQLOps o, Expr *p) : Expr(ti), optype(o), operand(p) {}
			UOper(SQLTypes t, SQLOps o, Expr *p) : Expr(t), optype(o), operand(p) {}
			virtual ~UOper() { delete operand; }
			SQLOps get_optype() { return optype; }
			const Expr *get_operand() { return operand; }
			virtual void check_group_by(const std::list<Expr*> *groupby) const;
		private:
			SQLOps optype; // operator type, e.g., kUMINUS, kISNULL, kEXISTS
			Expr *operand; // operand expression
	};

	/*
	 * @type BinOper
	 * @brief represents binary operator expressions.  it includes all
	 * comparison, arithmetic and boolean binary operators.  it handles ANY/ALL qualifiers
	 * in case the right_operand is a subquery.
	 */
	class BinOper : public Expr {
		public:
			BinOper(const SQLTypeInfo &ti, SQLOps o, SQLQualifier q, Expr *l, Expr *r) : Expr(ti), optype(o), qualifier(q), left_operand(l), right_operand(r) {}
			BinOper(SQLTypes t, SQLOps o, SQLQualifier q, Expr *l, Expr *r) : Expr(t), optype(o), qualifier(q), left_operand(l), right_operand(r) {}
			virtual ~BinOper() { delete left_operand; delete right_operand; }
			SQLOps get_optype() { return optype; }
			SQLQualifier get_qualifier() { return qualifier; }
			const Expr *get_left_operand() { return left_operand; }
			const Expr *get_right_operand() { return right_operand; }
			static SQLTypeInfo analyze_type_info(SQLOps op, const SQLTypeInfo &left_type, const SQLTypeInfo &right_type, SQLTypeInfo *new_left_type, SQLTypeInfo *new_right_type);
			static SQLTypeInfo common_numeric_type(const SQLTypeInfo &type1, const SQLTypeInfo &type2);
			virtual void check_group_by(const std::list<Expr*> *groupby) const;
		private:
			SQLOps optype; // operator type, e.g., kLT, kAND, kPLUS, etc.
			SQLQualifier qualifier; // qualifier kANY, kALL or kONE.  Only relevant with right_operand is Subquery
			Expr *left_operand; // the left operand expression
			Expr *right_operand; // the right operand expression
	};

	class Query;

	/*
	 * @type Subquery
	 * @brief subquery expression.  Note that the type of the expression is the type of the
	 * TargetEntry in the subquery instead of the set.
	 */
	class Subquery : public Expr {
		public:
			Subquery(const SQLTypeInfo &ti, Query *q) : Expr(ti), parsetree(q) /*, plan(nullptr)*/ {}
			virtual ~Subquery();
			const Query *get_parsetree() { return parsetree; }
			// const Plan *get_plan() { return plan; }
			// void set_plan(Plan *p) { plan = p; } // subquery plan is set by the optimizer
			virtual Expr *add_cast(const SQLTypeInfo &new_type_info);
		private:
			Query *parsetree; // parse tree of the subquery
			// Plan *plan; // query plan for the subquery.  to be filled in lazily.
	};

	/*
	 * @type InValues
	 * @brief represents predicate expr IN (v1, v2, ...)
	 * v1, v2, ... are can be either Constant or Parameter.
	 */
	class InValues : public Expr {
		public:
			InValues(Expr *a, std::list<Expr*> *l) : Expr(kBOOLEAN), arg(a), value_list(l) {}
			virtual ~InValues();
			const Expr *get_arg() { return arg; }
			const std::list<Expr*> *get_value_list() { return value_list; }
		private:
			Expr *arg; // the argument left of IN
			std::list<Expr*> *value_list; // the list of values right of IN
	};

	/*
	 * @type LikeExpr
	 * @brief expression for the LIKE predicate.
	 * arg must evaluate to char, varchar or text.
	 */
	class LikeExpr : public Expr {
		public:
			LikeExpr(Expr *a, Expr *l, Expr *e) : Expr(kBOOLEAN), arg(a), like_expr(l), escape_expr(e) {}
			virtual ~LikeExpr() { delete arg; }
			const Expr *get_arg() { return arg; }
			const Expr *get_like_expr() { return like_expr; }
			const Expr *get_escape_expr() { return escape_expr; }
		private:
			Expr *arg; // the argument right of LIKE
			Expr *like_expr; // expression that evaluates to like string
			Expr *escape_expr; // expression that evaluates to escape string, can be nullptr
	};

	/*
	 * @type AggExpr
	 * @brief expression for builtin SQL aggregates.
	 */
	class AggExpr : public Expr {
		public:
			AggExpr(const SQLTypeInfo &ti, SQLAgg a, Expr *g, bool d) : Expr(ti), aggtype(a), arg(g), is_distinct(d) {}
			AggExpr(SQLTypes t, SQLAgg a, Expr *g, bool d) : Expr(t), aggtype(a), arg(g), is_distinct(d) {}
			virtual ~AggExpr() { delete arg; }
			SQLAgg get_aggtype() { return aggtype; }
			const Expr *get_arg() { return arg; }
			bool get_is_distinct() { return is_distinct; }
		private:
			SQLAgg aggtype; // aggregate type: kAVG, kMIN, kMAX, kSUM, kCOUNT
			Expr *arg; // argument to aggregate
			bool is_distinct; // true only if it is for COUNT(DISTINCT x).
	};

	/*
	 * @type TargetEntry
	 * @brief Target list defines a relational projection.  It is a list of TargetEntry's.
	 */
	class TargetEntry {
		public:
			TargetEntry(int r, const std::string &n, Expr *e) : resno(r), resname(n), expr(e) {}
			virtual ~TargetEntry() { delete expr; }
			int get_resno() { return resno; }
			const std::string &get_resname() { return resname; }
			Expr *get_expr() { return expr; }
		private:
			int resno; // position in TargetList for SELECT, attribute number for UPDATE
			std::string resname; // alias name, e.g., SELECT salary + bonus AS compensation, 
			Expr *expr; // expression to evaluate for the value
	};

	/*
	 * @type RangeTblEntry 
	 * @brief Range table contains all the information about the tables/views
	 * and columns referenced in a query.  It is a vector of RangeTblEntry's.
	 */
	class RangeTblEntry {
		public:
			RangeTblEntry(const std::string &r, int32_t id, const std::string &t, Query *v) : rangevar(r), table_id(id), table_name(t), view_query(v) {}
			virtual ~RangeTblEntry();
			/* @brief get_column_no tries to find the column in column_descs and returns the index into the column_desc if found.
			 * otherwise, look up the column from Catalog, add the descriptor to column_descs and
			 * return the index.  return -1 if not found
			 * @param name name of column to look up
			 */
			const ColumnDescriptor *get_column_desc(int col_no) { return column_descs[col_no]; }
			const ColumnDescriptor *get_column_desc(const Catalog_Namespace::Catalog &catalog, const std::string &name);
			const std::vector<const ColumnDescriptor *> &get_column_descs() { return column_descs; }
			const std::string &get_rangevar() { return rangevar; }
			int32_t get_table_id() { return table_id; }
			const std::string &get_table_name() { return table_name; }
			const Query *get_view_query() { return view_query; }
			void expand_star_in_targetlist(const Catalog_Namespace::Catalog &catalog, int rte_no, std::vector<TargetEntry*> &tlist);
		private:
			std::string rangevar; // range variable name, e.g., FROM emp e, dept d
			int32_t table_id; // table id
			std::string table_name; // table name
			std::vector<const ColumnDescriptor *> column_descs; // column descriptors for all columns referenced in this query
			Query *view_query; // parse tree for the view query
	};

	/*
	 * @type OrderEntry
	 * @brief represents an entry in ORDER BY clause.
	 */
	struct OrderEntry {
		OrderEntry(int t, bool d, bool nf) : tle_no(t), is_desc(d), nulls_first(nf) {};
		~OrderEntry() {}
		int tle_no; /* targetlist entry number */
		bool is_desc; /* true if order is DESC */
		bool nulls_first; /* true if nulls are ordered first.  otherwise last. */
	};

	/*
	 * @type Query
	 * @brief parse tree for a query
	 */
	class Query {
		public:
			Query() : is_distinct(false), targetlist(nullptr), rangetable(nullptr), where_predicate(nullptr), group_by(nullptr), having_predicate(nullptr), order_by(nullptr), next_query(nullptr), is_unionall(false) {}
			Query(bool d, std::vector<TargetEntry*> *tl, std::vector<RangeTblEntry*> *rt, Expr *w, std::list<Expr*> *g, Expr *h, std::list<OrderEntry> *ord) : is_distinct(d), targetlist(tl), rangetable(rt), where_predicate(w), group_by(g), having_predicate(h), order_by(ord) {}
			virtual ~Query();
			bool get_is_distinct() { return is_distinct; }
			const std::vector<TargetEntry*> *get_targetlist() { return targetlist; }
			const std::vector<RangeTblEntry*> *get_rangetable() { return rangetable; }
			const Expr *get_where_predicate() { return where_predicate; }
			const std::list<Expr*> *get_group_by() { return group_by; };
			const Expr *get_having_predicate() { return having_predicate; }
			const std::list<OrderEntry> *get_order_by() { return order_by; }
			void set_is_distinct(bool d) { is_distinct = d; }
			void set_rangetable(std::vector<RangeTblEntry*> *rtbl) { rangetable = rtbl; }
			void set_targetlist(std::vector<TargetEntry*> *tlist) { targetlist = tlist; }
			void set_where_predicate(Expr *p) { where_predicate = p; }
			void set_group_by(std::list<Expr*> *g) { group_by = g; }
			void set_having_predicate(Expr *p) { having_predicate = p; }
			void set_order_by(std::list<OrderEntry> *o) { order_by = o; }
			void set_next_query(Query *q) { next_query = q; }
			void set_is_unionall(bool u) { is_unionall = u; }
			RangeTblEntry *get_rte(const std::string &range_var_name);
		private:
			bool is_distinct; // true only if SELECT DISTINCT
			std::vector<TargetEntry*> *targetlist; // represents the SELECT clause
			std::vector<RangeTblEntry*> *rangetable; // represents the FROM clause
			Expr *where_predicate; // represents the WHERE clause
			std::list<Expr*> *group_by; // represents the GROUP BY clause
			Expr *having_predicate; // represents the HAVING clause
			std::list<OrderEntry> *order_by; // represents the ORDER BY clause
			Query *next_query; // the next query to UNION
			bool is_unionall; // true only if it is UNION ALL
	};
}

#endif // ANALYZER_H
