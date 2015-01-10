/**
 * @file		Planner.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Functions for query plan nodes
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cassert>
#include <stdexcept>
#include "../Analyzer/Analyzer.h"
#include "Planner.h"

namespace Planner {
	Plan::~Plan()
	{
		for (auto p : targetlist)
			delete p;
		for (auto p : quals)
			delete p;
		if (child_plan != nullptr)
			delete child_plan;
	}

	Result::~Result()
	{
		Plan::~Plan();
		for (auto p : const_quals)
			delete p;
	}

	Scan::~Scan()
	{
		Plan::~Plan();
		for (auto p : simple_quals)
			delete p;
	}

	ValuesScan::~ValuesScan()
	{
		Plan::~Plan();
		for (auto p : value_list)
			delete p;
	}
	
	Join::~Join()
	{
		Plan::~Plan();
		delete child_plan2;
	}

	AggPlan::~AggPlan()
	{
		Plan::~Plan();
	}

	Append::~Append()
	{
		Plan::~Plan();
		for (auto p : plan_list)
			delete p;
	}

	MergeAppend::~MergeAppend()
	{
		Plan::~Plan();
		for (auto p : mergeplan_list)
			delete p;
	}

	Sort::~Sort()
	{
		Plan::~Plan();
	}

	RootPlan::~RootPlan()
	{
		delete plan;
	}

	Scan::Scan(const Analyzer::RangeTblEntry &rte) : Plan()
	{
		table_id = rte.get_table_id();
		for (auto cd : rte.get_column_descs())
			col_list.push_back(cd->columnId);
	}

	RootPlan *
	Optimizer::optimize()
	{
		Plan *plan;
		SQLStmtType stmt_type = query.get_stmt_type();
		int result_table_id = 0;
		std::list<int> result_col_list;
		Analyzer::RangeTblEntry *result_rte;
		switch (stmt_type) {
			case kSELECT:
				// nothing to do for SELECT for now
				break;
			case kINSERT:
				result_rte = query.get_rangetable().front(); // the first entry is the result table
				result_table_id = result_rte->get_table_id();
				for (auto cd : result_rte->get_column_descs())
					result_col_list.push_back(cd->columnId);
				break;
			case kUPDATE:
			case kDELETE:
				// should have been rejected by the Analyzer for now
				assert(false);
				break;
			default:
				assert(false);
		}
		plan = optimize_query();
		return new RootPlan(plan, stmt_type, result_table_id, result_col_list);
	}

	Plan *
	Optimizer::optimize_query()
	{
		//@TODO add support for union queries
		if (query.get_next_query() != nullptr)
			throw std::runtime_error("UNION queries are not supported yet.");
		cur_query = &query;
		optimize_current_query();
		if (query.get_order_by() != nullptr)
			optimize_orderby();
		return cur_plan;
	}

	void
	Optimizer::optimize_current_query()
	{
		optimize_scans();
		optimize_joins();
		optimize_aggs();
		process_targetlist();
	}

	void
	Optimizer::optimize_scans()
	{
		const std::vector<Analyzer::RangeTblEntry*> &rt = cur_query->get_rangetable();
		for (auto rte : rt)
			base_scans.push_back(new Scan(*rte));
		const Analyzer::Expr *where_pred = query.get_where_predicate();
		std::list<const Analyzer::Expr*> scan_predicates;
		if (where_pred != nullptr)
			where_pred->group_predicates(scan_predicates, join_predicates, const_predicates);
		for (auto p : scan_predicates) {
			int rte_idx;
			Analyzer::Expr *simple_pred = p->normalize_simple_predicate(rte_idx);
			if (simple_pred != nullptr)
				base_scans[rte_idx]->add_simple_predicate(simple_pred);
			else
				base_scans[rte_idx]->add_predicate(p->deep_copy());
		}
		const std::list<Analyzer::TargetEntry*> &tlist = cur_query->get_targetlist();
		bool(*fn_pt)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*) = Analyzer::ColumnVar::colvar_comp;
		std::set<const Analyzer::ColumnVar*, bool(*)(const Analyzer::ColumnVar*, const Analyzer::ColumnVar*)> colvar_set(fn_pt);
		for (auto tle : tlist)
			tle->get_expr()->collect_column_var(colvar_set);
		for (auto p : join_predicates)
			p->collect_column_var(colvar_set);
		for (auto colvar : colvar_set) {
			Analyzer::TargetEntry *tle = new Analyzer::TargetEntry("", colvar->deep_copy());
			base_scans[colvar->get_rte_idx()]->add_tle(tle);
		}
	}

	void
	Optimizer::optimize_joins()
	{
		if (base_scans.size() == 1)
			cur_plan = base_scans[0];
		else
			throw std::runtime_error("joins are not supported yet.");
	}

	void
	Optimizer::optimize_aggs()
	{
		if (cur_query->get_num_aggs() > 0 || cur_query->get_having_predicate() != nullptr)
			throw std::runtime_error("aggregates not supported yet.");
	}
	
	void
	Optimizer::optimize_orderby()
	{
		if (query.get_order_by() != nullptr)
			throw std::runtime_error("order by not supported yet.");
	}

	void
	Optimizer::process_targetlist()
	{
		std::list<Analyzer::TargetEntry*> final_tlist;
		for (auto tle : query.get_targetlist()) {
			Analyzer::TargetEntry *new_tle = new Analyzer::TargetEntry(tle->get_resname(), tle->get_expr()->rewrite_with_targetlist(cur_plan->get_targetlist()));
			final_tlist.push_back(new_tle);
		}
		//delete the old TargetEntry's
		for (auto tle : cur_plan->get_targetlist())
			delete tle;
		cur_plan->set_targetlist(final_tlist);
	}
}
