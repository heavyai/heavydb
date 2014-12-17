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
#include "../Shared/sqldefs.h"

namespace Analyzer {

	/*
	 * @type Expr
	 * @brief super class for all expressions in parse trees and in query plans
	 */
	class Expr {
		public:
			Expr(SQLTypes t) : type(t) {}
			SQLTypes get_type() { return type; }
		private:
			SQLTypes type; // data type of the return result of this expression
	}

	/*
	 * @type ColumnVar
	 * @brief expression that evaluates to the value of a column in a given row from a base table.
	 * It is used in parse trees and is only used in Scan nodes in a query plan 
	 * for scanning a table while Var nodes are used for all other plans.
	 */
	class ColumnVar : public Expr {
		public:
			ColumnVar(SQLTypes t, int r, int c) : Expr(t), rte_no(r), col_no(c) {}
			~ColumnVar();
			int get_rte_no() { return ret_no; }
			int get_col_no() { return col_no; }
		private:
			int rte_no; // index of the range table entry. 0 based
			int col_no; // index into the vector of columns
	}

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
			Var(SQLTypes t, int r, int c, bool o, int v) : ColumnVar(t, r, c), is_inner(o), varno(v) {}
			~Var();
			bool get_is_inner() { return is_inner; }
			int get_varno() { return varno; }
		private:
			bool is_inner; // indicate the row is produced by the inner plan
			int varno; // the column number in the row.
	}

	/*
	 * @type Constant
	 * @brief expression for a constant value
	 */
	class Constant : public Expr {
		public:
			Constant(SQLTypes t, bool n, Datum c) : Expr(t), is_null(n), constval(c) {}
			~Constant();
			bool get_is_null() { return is_null; }
			Datum get_constval() { return constval; }
		private:
			bool is_null; // constant is NULL
			Datum constval; // the constant value
	}

	/*
	 * @type UOper
	 * @brief represents unary operator expressions.  operator types include
	 * kUMINUS, kISNULL, kEXISTS
	 */
	class UOper : public Expr {
		public:
			UOper(SQLTypes t, SQLOps o, Expr *p) : Expr(t), optype(o), operand(p) {}
			~UOper();
			SQLOps get_optype() { return optype; }
			const Expr *get_operand() { return operand; }
		private:
			SQLOps optype; // operator type, e.g., kUMINUS, kISNULL, kEXISTS
			Expr *operand; // operand expression
	}

	/*
	 * @type BinOper
	 * @brief represents binary operator expressions.  it includes all
	 * comparison, arithmetic and boolean binary operators.  it handles ANY/ALL qualifiers
	 * in case the right_operand is a subquery.
	 */
	class BinOper : public Expr {
		public:
			BinOper(SQLTypes t, SQLOps o, SQLQualifier q, Expr *l, Expr *r) : Expr(t), optype(o), qualifier(q), left_operand(l), right_operand(r) {}
			~BinOper();
			SQLOps get_optype() { return optype; }
			SQLQualifier get_qualifier() { return qualifier; }
			const Expr *get_left_operand() { return left_operand; }
			const Expr *get_right_operand() { return right_operand; }
		private:
			SQLOps optype; // operator type, e.g., kLT, kAND, kPLUS, etc.
			SQLQualifier qualifier; // qualifier kANY, kALL or kONE.  Only relevant with right_operand is Subquery
			Expr *left_operand; // the left operand expression
			Expr *right_operand; // the right operand expression
	}

	/*
	 * @type Subquery
	 * @brief subquery expression.  Note that the type of the expression is the type of the
	 * TargetEntry in the subquery instead of the set.
	 */
	class Subquery : public Expr {
		public:
			Subquery(SQLTypes t, Query *q) : Expr(t), parsetree(q), plan(nullptr) {}
			~Subquery();
			const Query *get_parsetree() { return parsetree; }
			const Plan *get_plan() { return plan; }
			void set_plan(Plan *p) { plan = p; } // subquery plan is set by the optimizer
		private:
			Query *parsetree; // parse tree of the subquery
			Plan *plan; // query plan for the subquery.  to be filled in lazily.
	}

	/*
	 * @type InValues
	 * @brief represents predicate expr IN (v1, v2, ...)
	 * v1, v2, ... are can be either Constant or Parameter.
	 */
	class InValues : public Expr {
		public:
			InValues(Expr *a, std::list<Expr*> *l) : Expr(kBOOLEAN), arg(a), value_list(l) {}
			~InValues();
			const Expr *get_arg() { return arg; }
			const std::list<Expr*> *get_value_list() { return value_list; }
		private:
			Expr *arg; // the argument left of IN
			std::list<Expr*> *value_list; // the list of values right of IN
	}

	/*
	 * @type LikeExpr
	 * @brief expression for the LIKE predicate.
	 * arg must evaluate to char, varchar or text.
	 */
	class LikeExpr : public Expr {
		public:
			LikeExpr(Expr *a, const std::string &l, const std::string &e) : Expr(kBOOLEAN), arg(a), like_str(l), escape_str(e) {}
			~LikeExpr();
			const Expr *get_arg() { return arg; }
			const std::string &get_like_str() { return like_str; }
			const std::string &get_escape_str() { return escape_str; }
		private:
			Expr *arg; // the argument right of LIKE
			std::string like_str; // the like string
			std::string escape_str; // the escape string
	}

	/*
	 * @type AggExpr
	 * @brief expression for builtin SQL aggregates.
	 */
	class AggExpr : public Expr {
		public:
			AggExpr(SQLTypes t, SQLAgg a, Expr *g, bool d) : Expr(t), aggtype(a), arg(g), is_distinct(d) {}
			~AggExpr();
			SQLAgg get_aggtype() { return aggtype; }
			const Expr *get_arg() { return arg; }
			bool get_is_distinct() { return is_distinct; }
		private:
			SQLAgg aggtype; // aggregate type: kAVG, kMIN, kMAX, kSUM, kCOUNT
			Expr *arg; // argument to aggregate
			bool is_distinct; // true only if it is for COUNT(DISTINCT x).
	}

	/*
	 * @type TargetEntry
	 * @brief Target list defines a relational projection.  It is a list of TargetEntry's.
	 */
	class TargetEntry {
		public:
			TargetEntry(int r, const std::string &n, Expr *e) : resno(r), resname(n), expr(e) {}
			~TargetEntry();
			int get_resno() { return resno; }
			const std::string &get_resname() { return resname; }
			Expr *get_expr() { return expr; }
		private:
			int resno; // position in TargetList for SELECT, attribute number for UPDATE
			std::string resname; // alias name, e.g., SELECT salary + bonus AS compensation, 
			Expr *expr; // expression to evaluate for the value
	}

	/*
	 * @type RangeTblEntry 
	 * @brief Range table contains all the information about the tables/views
	 * and columns referenced in a query.  It is a vector of RangeTblEntry's.
	 */
	class RangeTblEntry {
		public:
			RangeTblEntry(const std::string &r, int32_t id, const std::string &t, Query *v) : rangevar(r), table_id(id), table_name(t), view_query(v) {}
			~RangeTblEntry();
			/* @brief get_column tries to find the column in column_descs and returns the descriptor if found.
			 * otherwise, look up the column from Catalog, add the descriptor to column_descs and
			 * return it.  return nullptr if not found
			 * @param name name of column to look up
			 */
			const ColumnDescriptor *get_column(const std::string &name);
			const std::string &get_rangevar() { return rangevar; }
			int32_t get_table_id() { return table_id; }
			const std::string &get_table_name() { return table_name; }
			const Query *get_view_query() { return view_query; }
		private:
			std::string rangevar; // range variable name, e.g., FROM emp e, dept d
			int32_t table_id; // table id
			std::string table_name; // table name
			std::vector<const ColumnDescriptor *> column_descs; // column descriptors for all columns referenced in this query
			Query *view_query; // parse tree for the view query
	}

	/*
	 * @type OrderEntry
	 * @brief represents an entry in ORDER BY clause.
	 */
	struct OrderEntry {
		int tle_no; /* targetlist entry number */
		bool is_desc; /* true if order is DESC */
		bool nulls_first; /* true if nulls are ordered first.  otherwise last.
	}

	/*
	 * @type Query
	 * @brief parse tree for a query
	 */
	class Query {
		public:
			Query(bool d, std::list<TargetEntry*> tl, std::vector<RangeTblEntry*> *rt, Expr *w, std::vector<Expr*> *g, Expr *h, list<OrderEntry*> *ord) : is_distinct(d), targetlist(tl), rangetable(rt), where_predicate(w), group_by(g), having_predicate(h), order_by(ord) {}
			~Query();
			bool get_is_distinct() { return is_distinct; }
			const std::vector<TargetEntry*> *get_targetlist() { return targetlist; }
			const std::vector<RangeTblEntry*> *get_rangetable() { return rangetable; }
			const Expr *get_where_predicate() { return where_predicate; }
			const list<Expr*> *get_group_by() { return group_by; };
			const Expr *get_having_predicate() { return having_predicate; }
			const list<OrderEntry*> *get_order_by() { return order_by; }
		private:
			bool is_distinct; // true only if SELECT DISTINCT
			std::vector<TargetEntry*> *targetlist; // represents the SELECT clause
			std::vector<RangeTblEntry*> *rangetable; // represents the FROM clause
			Expr *where_predicate; // represents the WHERE clause
			list<Expr*> *group_by; // represents the GROUP BY clause
			Expr *having_predicate; // represents the HAVING clause
			list<OrderEntry> *order_by; // represents the ORDER BY clause
			Query *next_query; // the next query to UNION
			bool is_unionall; // true only if it is UNION ALL
	}
}

#endif // ANALYZER_H
