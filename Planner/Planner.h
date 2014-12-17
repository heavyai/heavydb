/**
 * @file		Planner.h
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Defines data structures for query plan nodes.
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/
#ifndef PLANNER_H
#define PLANNER_H

#include <cstdint>
#include <string>
#include <vector>
#include <list>
#include "../Analyzer/Analyzer.h"

namespace Planner {
	/*
	 * @type Plan
	 * @brief super class for all plan nodes
	 */
	class Plan {
		public:
			Plan(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p) : targetlist(t), quals(q), cost(c), child_plan(p) {}
			const std::vector<Analyzer::TargetEntry*> *get_targetlist() { return targetlist; }
			const std::list<Analyzer::Expr*> *get_quals() { return quals; }
			double get_cost() { return cost; }
			const Plan *get_child_plan() { return child_plan; }
		private:
			std::vector<Analyzer::TargetEntry*> *targetlist; // projection of this plan node
			std::list<Analyzer::Expr*> *quals; // list of boolean expressions, implicitly conjunctive
			double cost; // Planner assigned cost for optimization purpose
			Plan *child_plan; // most plan nodes have at least one child, therefore keep it in super class
	}

	/*
	 * @type Result
	 * @brief Result node is for evaluating constant predicates, e.g., 1 < 2 or $1 = 'foo'
	 * It is also used to perform further projections from the child_plan to eliminate
	 * columns that are required by the child_plan but not in the final targetlist, e.g.,
	 * group by columns that are not selected.
	 */
	class Result : public Plan {
		public:
			Result(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p, std::list<Analyzer::Expr*> *cq) : Plan(t, q, c, p), const_quals(cq) {}
			~Result();
			const std::list<Analyzer::Expr*> *get_constquals() { return const_quals; }
		private:
			std::list<Analyzer::Expr*> *const_quals; // constant quals to evaluate only once
	}

	/*
	 * @type Scan
	 * @brief Scan node is for scanning a table or rowset.
	 */
	class Scan : public Plan {
		public:
			Scan(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p, std::list<Analyzer::Expr*> *sq, int r, std::list<int> *cl) : Plan(t, q, c, p), simple_quals(sq), rte_no(r), col_list(cl) {}
			~Scan();
			const std::list<Analyzer::Expr*> *get_simple_quals() { return simple_quals; };
			int get_rte_no() { return rte_no; }
			const std::list<int> *get_col_list() { return col_list; }
		private:
			// simple_quals consists of predicates of the form 'ColumnVar BinOper Constant'
			// it can be used for eliminating fragments and/or partitions from the scan.
			std::list<Analyzer::Expr*> *simple_quals;
			int rte_no; // rangetable entry index for the table to scan
			std::list<int> *col_list; // list of column indexes if we only scan subset of columns. nullptr if we can all
	}

	/*
	 * @type ValuesScan
	 * @brief ValuesScan returns a row from a list of values.
	 * It is used for processing INSERT INTO tab VALUES (...)
	 */
	class ValuesScan : public Plan {
		public:
			ValuesScan(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p, std::list<Analyzer::Expr*> *vl) : Plan(t, q, c, p), value_list(vl) {}
			~ValuesScan();
			const std::list<Analyzer::Expr*> *get_value_list() { return value_list; }
		private:
			std::list<Analyzer::Expr*> *value_list; // values list, should be Constant or Parameter nodes
	}

	/*
	 * @type Join
	 * @brief super class for all join nodes.
	 */
	class Join : public Plan {
		public:
			Join(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p, Plan *cp2) : Plan(t, q, c, p), child_plan2(cp2) {}
			~Join();
			const plan *get_outerplan() { return child_plan; }
			const plan *get_innerplan() { return child_plan2; }
		private:
			Plan *child_plan2;
	}

	/*
	 * @type AggPlan
	 * @brief AggPlan handles aggregate functions and group by.
	 * Having clause is in the inherited quals
	 */
	class AggPlan : public Plan {
		public:
			AggPlan(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p, std::list<int> *gl) : Plan(t, q, c, p), groupby_list(gl) {}
			~AggPlan();
			const std::list<int> *get_groupby_list() { return groupby_list; }
		private:
			std::list<int> *groupby_list; // list of indexes into the targetlist for group by columns
	}

	/*
	 * @type Append
	 * @brief Append evaluates a list of query plans one by one and simply appends all rows
	 * to result set.  It is for processing UNION ALL queries.
	 */
	class Append : public Plan {
		public:
			Append(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p, std::list<Plan*> *pl) : Plan(t, q, c, p), plan_list(pl) {}
			~Append();
			const std::list<Plan*> *get_plan_list() { return plan_list; }
		private:
			std::list<Plan*> *plan_list; // list of plans to union all
	}

	/*
	 * @type MergeAppend
	 * @brief MergeAppend merges sorted streams of rows and eliminate duplicates.
	 * It is for processing UNION queries.
	 */
	class MergeAppend : public Plan {
		public:
			MergeAppend(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p, std::list<Plan*> *pl, std::list<Analyzer::OrderEntry> *oe) : Plan(t, q, c, p), mergeplan_list(pl), order_entries(oe) {}
			~MergeAppend();
			const std::list<Plan*> *get_mergeplan_list() { return mergeplan_list; }
			const std::list<Analyzer::OrderEntry> *get_order_entries() { return order_entries; }
		private:
			std::list<Plan*> *mergeplan_list; // list of plans to merge
			std::list<OrderEntry> *order_entries; // defines how the mergeplans are sorted
	}

	/*
	 * @type Sort
	 * @brief Sort node returns rows from the child_plan sorted by selected columns.
	 * It handles the Order By clause and is used for mergejoin and for remove duplicates.
	 */
	class Sort : public Plan {
		public:
			Sort(std::vector<Analyzer::TargetEntry*> *t, std::list<Analyzer::Expr*> *q, double c, Plan *p, std::list<Analyzer::OrderEntry> *oe) : Plan(t, q, c, p), order_entries(oe) {}
			~Sort();
			const std::list<Analyzer::OrderEntry> *get_order_entries() { return order_entries; }
		private:
			std::list<Analyzer::OrderEntry> *order_entries; // defines columns to sort on and in what order
	}

	/*
	 * @type PlannedStmt
	 * @brief PlannedStmt is the end result produced by the Planner for the Executor to execute.
	 * It captures all the information required for executing the query.
	 * @TODO add support for parametrized queries.
	 */
	class PlannedStmt {
		public:
			PlannedStmt(Plan *p, SQLStmtType t, int r, std::vector<Analyzer::RangeTblEntry*> *rt) : plan(p), stmt_type(t), result_rte(r), rangetable(rt) {}
			~PlannedStmt();
			const Plan *get_plan() { return plan; }
			SQLStmtType get_stmt_type() { return stmt_type; }
			int get_result_rte() { return result_rte; }
			const std::vector<Analyzer::RangeTblEntry*> *get_rangetable() { return rangetable; }
		private:
			Plan *plan; // query plan
			SQLStmtType stmt_type; // SELECT, UPDATE, DELETE or INSERT
			int result_rte; // rangetable entry index for the result table.  For UPDATE, DELETE or INSERT only.
			std::vector<Analyzer::RangeTblEntry*> *rangetable; // rangetable
	}
}

#endif // PLANNER_H
