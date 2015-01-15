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
			Plan(const std::list<Analyzer::TargetEntry*> &t, const std::list<Analyzer::Expr*> &q, double c, Plan *p) : targetlist(t), quals(q), cost(c), child_plan(p) {}
			Plan(const std::list<Analyzer::TargetEntry*> &t, double c, Plan *p) : targetlist(t), cost(c), child_plan(p) {}
			Plan() : cost(0.0), child_plan(nullptr) {}
			Plan(const std::list<Analyzer::TargetEntry*> &t) : targetlist(t), cost(0.0), child_plan(nullptr) {}
			virtual ~Plan();
			const std::list<Analyzer::TargetEntry*> &get_targetlist() const { return targetlist; }
			const std::list<Analyzer::Expr*> &get_quals() const { return quals; }
			double get_cost() const { return cost; }
			const Plan *get_child_plan() const { return child_plan; }
			void add_tle(Analyzer::TargetEntry *tle) { targetlist.push_back(tle); }
			void set_targetlist(const std::list<Analyzer::TargetEntry*> &t) { targetlist = t; }
			virtual void print() const;
		protected:
			std::list<Analyzer::TargetEntry*> targetlist; // projection of this plan node
			std::list<Analyzer::Expr*> quals; // list of boolean expressions, implicitly conjunctive
			double cost; // Planner assigned cost for optimization purpose
			Plan *child_plan; // most plan nodes have at least one child, therefore keep it in super class
	};

	/*
	 * @type Result
	 * @brief Result node is for evaluating constant predicates, e.g., 1 < 2 or $1 = 'foo'
	 * It is also used to perform further projections and qualifications 
	 * from the child_plan.  One use case is  to eliminate columns that 
	 * are required by the child_plan but not in the final targetlist
	 * , e.g., group by columns that are not selected.  Another use case is
	 * to evaluate expressions over group by columns and aggregates in the targetlist.
	 * A 3rd use case is to process the HAVING clause for filtering 
	 * rows produced by AggPlan.
	 */
	class Result : public Plan {
		public:
			Result(std::list<Analyzer::TargetEntry*> &t, const std::list<Analyzer::Expr*> &q, double c, Plan *p, const std::list<Analyzer::Expr*> &cq) : Plan(t, q, c, p), const_quals(cq) {}
			virtual ~Result();
			const std::list<Analyzer::Expr*> &get_constquals() const { return const_quals; }
			virtual void print() const;
		private:
			std::list<Analyzer::Expr*> const_quals; // constant quals to evaluate only once
	};

	/*
	 * @type Scan
	 * @brief Scan node is for scanning a table or rowset.
	 */
	class Scan : public Plan {
		public:
			Scan(const std::list<Analyzer::TargetEntry*> &t, const std::list<Analyzer::Expr*> &q, double c, Plan *p, const std::list<Analyzer::Expr*> &sq, int r, const std::list<int> &cl) : Plan(t, q, c, p), simple_quals(sq), table_id(r), col_list(cl) {}
			Scan(const Analyzer::RangeTblEntry &rte);
			virtual ~Scan();
			const std::list<Analyzer::Expr*> &get_simple_quals() const { return simple_quals; };
			int get_table_id() const { return table_id; }
			const std::list<int> &get_col_list() const { return col_list; }
			void add_predicate(Analyzer::Expr *pred) { quals.push_back(pred); }
			void add_simple_predicate(Analyzer::Expr *pred) { simple_quals.push_back(pred); }
			virtual void print() const;
		private:
			// simple_quals consists of predicates of the form 'ColumnVar BinOper Constant'
			// it can be used for eliminating fragments and/or partitions from the scan.
			std::list<Analyzer::Expr*> simple_quals;
			int table_id; // rangetable entry index for the table to scan
			std::list<int> col_list; // list of column ids to scan
	};

	/*
	 * @type ValuesScan
	 * @brief ValuesScan returns a row from a list of values.
	 * It is used for processing INSERT INTO tab VALUES (...)
	 */
	class ValuesScan : public Plan {
		public:
			ValuesScan(const std::list<Analyzer::TargetEntry*> &t) : Plan(t) {}
			virtual ~ValuesScan() {};
			virtual void print() const;
	};

	/*
	 * @type Join
	 * @brief super class for all join nodes.
	 */
	class Join : public Plan {
		public:
			Join(const std::list<Analyzer::TargetEntry*> &t, const std::list<Analyzer::Expr*> &q, double c, Plan *p, Plan *cp2) : Plan(t, q, c, p), child_plan2(cp2) {}
			virtual ~Join();
			virtual void print() const;
			const Plan *get_outerplan() const { return child_plan; }
			const Plan *get_innerplan() const { return child_plan2; }
		private:
			Plan *child_plan2;
	};

	/*
	 * @type AggPlan
	 * @brief AggPlan handles aggregate functions and group by.
	 */
	class AggPlan : public Plan {
		public:
			AggPlan(const std::list<Analyzer::TargetEntry*> &t, double c, Plan *p, const std::list<Analyzer::Expr*> &gl) : Plan(t, c, p), groupby_list(gl) {}
			virtual ~AggPlan();
			const std::list<Analyzer::Expr*> &get_groupby_list() const { return groupby_list; }
			virtual void print() const;
		private:
			std::list<Analyzer::Expr*> groupby_list; // list of expressions for group by.  only Var nodes are allow now.
	};

	/*
	 * @type Append
	 * @brief Append evaluates a list of query plans one by one and simply appends all rows
	 * to result set.  It is for processing UNION ALL queries.
	 */
	class Append : public Plan {
		public:
			Append(const std::list<Analyzer::TargetEntry*> &t, const std::list<Analyzer::Expr*> &q, double c, Plan *p, const std::list<Plan*> &pl) : Plan(t, q, c, p), plan_list(pl) {}
			virtual ~Append();
			const std::list<Plan*> &get_plan_list() const { return plan_list; }
			virtual void print() const;
		private:
			std::list<Plan*> plan_list; // list of plans to union all
	};

	/*
	 * @type MergeAppend
	 * @brief MergeAppend merges sorted streams of rows and eliminate duplicates.
	 * It is for processing UNION queries.
	 */
	class MergeAppend : public Plan {
		public:
			MergeAppend(const std::list<Analyzer::TargetEntry*> &t, const std::list<Analyzer::Expr*> &q, double c, Plan *p, const std::list<Plan*> &pl, const std::list<Analyzer::OrderEntry> &oe) : Plan(t, q, c, p), mergeplan_list(pl), order_entries(oe) {}
			virtual ~MergeAppend();
			const std::list<Plan*> &get_mergeplan_list() const { return mergeplan_list; }
			const std::list<Analyzer::OrderEntry> &get_order_entries() const { return order_entries; }
			virtual void print() const;
		private:
			std::list<Plan*> mergeplan_list; // list of plans to merge
			std::list<Analyzer::OrderEntry> order_entries; // defines how the mergeplans are sorted
	};

	/*
	 * @type Sort
	 * @brief Sort node returns rows from the child_plan sorted by selected columns.
	 * It handles the Order By clause and is used for mergejoin and for remove duplicates.
	 */
	class Sort : public Plan {
		public:
			Sort(const std::list<Analyzer::TargetEntry*> &t, const std::list<Analyzer::Expr*> &q, double c, Plan *p, const std::list<Analyzer::OrderEntry> &oe) : Plan(t, q, c, p), order_entries(oe) {}
			virtual ~Sort();
			const std::list<Analyzer::OrderEntry> &get_order_entries() const { return order_entries; }
			virtual void print() const;
		private:
			std::list<Analyzer::OrderEntry> order_entries; // defines columns to sort on and in what order
	};

	/*
	 * @type RootPlan
	 * @brief RootPlan is the end result produced by the Planner for the Executor to execute.
	 * @TODO add support for parametrized queries.
	 */
	class RootPlan {
		public:
			RootPlan(Plan *p, SQLStmtType t, int r, const std::list<int> &c) : plan(p), stmt_type(t), result_table_id(r), result_col_list(c) {}
			~RootPlan();
			const Plan *get_plan() const { return plan; }
			SQLStmtType get_stmt_type() const { return stmt_type; }
			int get_result_table_id() const { return result_table_id; }
			const std::list<int> &get_result_col_list() const { return result_col_list; }
			virtual void print() const;
		private:
			Plan *plan; // query plan
			SQLStmtType stmt_type; // SELECT, UPDATE, DELETE or INSERT
			int result_table_id; // For UPDATE, DELETE or INSERT only: table id for the result table
			std::list<int> result_col_list; // For UPDATE and INSERT only: list of result column ids.
	};

	/*
	 * @type Optimizer
	 * @brief This is the class for performing query optimizations.
	 */
	class Optimizer {
		public:
			Optimizer(const Analyzer::Query &q) : cur_query(nullptr), query(q) {}
			~Optimizer() {}
			/*
			 * @brief optimize optimize an entire SQL DML statement
			 */
			RootPlan *optimize();
		private:
			/*
			 * @brief optimize_query optimize the query portion of the statement.  can be a union
			 * query
			 */
			Plan *optimize_query();
			/*
			 * @brief optimize_current_query optimize cur_query and output plan to cur_plan.  
			 * must be a non-union query.
			 */
			void optimize_current_query();
			void optimize_scans();
			void optimize_joins();
			void optimize_aggs();
			void optimize_orderby();
			void process_targetlist();
			std::vector<Scan*> base_scans;
			std::list<const Analyzer::Expr*> join_predicates;
			std::list<const Analyzer::Expr*> const_predicates;
			const Analyzer::Query *cur_query;
			Plan *cur_plan;
			const Analyzer::Query &query;
	};
}

#endif // PLANNER_H
