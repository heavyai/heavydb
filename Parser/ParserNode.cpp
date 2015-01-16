/**
 * @file		ParserNode.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Functions for ParserNode classes
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <cassert>
#include <stdexcept>
#include <typeinfo>
#include <boost/algorithm/string/predicate.hpp>
#include "../Catalog/Catalog.h"
#include "ParserNode.h"

namespace Parser {
	SubqueryExpr::~SubqueryExpr() 
	{
		delete query;
	}

	ExistsExpr::~ExistsExpr() 
	{
		delete query;
	}

	InValues::~InValues()
	{
		for (auto p : *value_list)
			delete p;
		delete value_list;
	}

	BetweenExpr::~BetweenExpr() 
	{
		delete arg;
		delete lower;
		delete upper;
	}

	LikeExpr::~LikeExpr() 
	{
		delete arg;
		delete like_string;
		if (escape_string != nullptr)
			delete escape_string;
	}

	ColumnRef::~ColumnRef() 
	{
		if (table != nullptr)
			delete table;
		if (column != nullptr)
			delete column;
	}

	FunctionRef::~FunctionRef() 
	{
		delete name;
		if (arg != nullptr)
			delete arg;
	}
	
	TableRef::~TableRef() 
	{
		delete table_name;
		if (range_var != nullptr)
			delete range_var;
	}

	ColumnConstraintDef::~ColumnConstraintDef() 
	{
		if (defaultval != nullptr)
			delete defaultval;
		if (check_condition != nullptr)
			delete check_condition;
		if (foreign_table != nullptr)
			delete foreign_table;
		if (foreign_column != nullptr)
			delete foreign_column;
	}

	ColumnDef::~ColumnDef() 
	{
		delete column_name;
		delete column_type;
		if (compression != nullptr)
			delete compression;
		if (column_constraint != nullptr)
			delete column_constraint;
	}

	UniqueDef::~UniqueDef() 
	{
		for (auto p : *column_list)
			delete p;
		delete column_list;
	}

	ForeignKeyDef::~ForeignKeyDef() 
	{
		for (auto p : *column_list)
			delete p;
		delete column_list;
		delete foreign_table;
		if (foreign_column_list != nullptr) {
			for (auto p : *foreign_column_list)
				delete p;
			delete foreign_column_list;
		}
	}

	CreateTableStmt::~CreateTableStmt() 
	{
		delete table;
		for (auto p : *table_element_list)
			delete p;
		delete table_element_list;
	}

	SelectEntry::~SelectEntry()
	{
		delete select_expr;
		if (alias != nullptr)
			delete alias;
	}

	QuerySpec::~QuerySpec() 
	{
		if (select_clause != nullptr) {
			for (auto p : *select_clause)
				delete p;
			delete select_clause;
		}
		for (auto p : *from_clause)
			delete p;
		delete from_clause;
		if (where_clause != nullptr)
			delete where_clause;
		if (groupby_clause != nullptr)
			delete groupby_clause;
		if (having_clause != nullptr)
			delete having_clause;
	}

	SelectStmt::~SelectStmt()
	{
		delete query_expr;
		if (orderby_clause != nullptr) {
			for (auto p : *orderby_clause)
				delete p;
			delete orderby_clause;
		}
	}

	CreateViewStmt::~CreateViewStmt() 
	{
		delete view_name;
		if (column_list != nullptr) {
			for (auto p : *column_list)
				delete p;
			delete column_list;
		}
		delete query;
	}

	InsertStmt::~InsertStmt()
	{
		delete table;
		if (column_list != nullptr) {
			for (auto p : *column_list)
				delete p;
			delete column_list;
		}
	}

	InsertValuesStmt::~InsertValuesStmt()
	{
		for (auto p : *value_list)
			delete p;
		delete value_list;
	}

	UpdateStmt::~UpdateStmt()
	{
		delete table;
		for (auto p : *assignment_list)
			delete p;
		delete assignment_list;
		if (where_clause != nullptr)
			delete where_clause;
	}

	DeleteStmt::~DeleteStmt()
	{
		delete table;
		if (where_clause != nullptr)
			delete where_clause;
	}
	
	Analyzer::Expr *
	NullLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Constant *c = new Analyzer::Constant(kNULLT, true);
		Datum d;
		d.pointerval = nullptr;
		c->set_constval(d);
		return c;
	}
	
	Analyzer::Expr *
	StringLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		SQLTypeInfo ti;
		ti.type = kVARCHAR;
		ti.dimension = stringval->length();
		ti.scale = 0;
		char *s = new char[stringval->length() + 1];
		strcpy(s, stringval->c_str());
		Datum d;
		d.pointerval = (void*)s;
		return new Analyzer::Constant(ti, false, d);
	}

	Analyzer::Expr *
	IntLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		SQLTypes t;
		Datum d;
		if (intval >= INT16_MIN && intval <= INT16_MAX) {
			t = kSMALLINT;
			d.smallintval = (int16_t)intval;
		} else if (intval >= INT32_MIN && intval <= INT32_MAX) {
			t = kINT;
			d.intval = (int32_t)intval;
		} else {
			t = kBIGINT;
			d.bigintval = intval;
		}
		Analyzer::Constant *c = new Analyzer::Constant(t, false, d);
		return c;
	}

	Analyzer::Expr *
	FixedPtLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		assert(fixedptval->length() <= 20);
		size_t dot = fixedptval->find_first_of('.', 0);
		assert(dot != std::string::npos);
		std::string before_dot = fixedptval->substr(0, dot);
		std::string after_dot = fixedptval->substr(dot+1);
		Datum d;
		d.bigintval = std::stoll(before_dot);
		int64_t fraction = std::stoll(after_dot);
		SQLTypeInfo ti;
		ti.type = kNUMERIC;
		ti.scale = after_dot.length();
		ti.dimension = before_dot.length() + ti.scale;
		// the following loop can be made more efficient if needed
		for (int i = 0; i < ti.scale; i++)
			d.bigintval *= 10;
		d.bigintval += fraction;
		return new Analyzer::Constant(ti, false, d);
	}

	Analyzer::Expr *
	FloatLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Datum d;
		d.floatval = floatval;
		return new Analyzer::Constant(kFLOAT, false, d);
	}
	
	Analyzer::Expr *
	DoubleLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Datum d;
		d.doubleval = doubleval;
		return new Analyzer::Constant(kDOUBLE, false, d);
	}

	Analyzer::Expr *
	UserLiteral::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		throw std::runtime_error("USER literal not supported yet.");
		return nullptr;
	}

	Analyzer::Expr *
	OperExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		SQLTypeInfo result_type, left_type, right_type;
		SQLTypeInfo new_left_type, new_right_type;
		Analyzer::Expr *left_expr, *right_expr;
		SQLQualifier qual = kONE;
		if (typeid(*right) == typeid(SubqueryExpr))
			qual = dynamic_cast<SubqueryExpr*>(right)->get_qualifier();
		left_expr = left->analyze(catalog, query);
		right_expr = right->analyze(catalog, query);
		left_type = left_expr->get_type_info();
		right_type = right_expr->get_type_info();
		result_type = Analyzer::BinOper::analyze_type_info(optype, left_type, right_type, &new_left_type, &new_right_type);
		if (left_type != new_left_type)
			left_expr = left_expr->add_cast(new_left_type);
		if (right_type != new_right_type)
			right_expr = right_expr->add_cast(new_right_type);
		return new Analyzer::BinOper(result_type, optype, qual, left_expr, right_expr);
	}

	Analyzer::Expr *
	SubqueryExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		throw std::runtime_error("Subqueries are not supported yet.");
		return nullptr;
	}

	Analyzer::Expr *
	IsNullExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Expr *arg_expr = arg->analyze(catalog, query);
		Analyzer::Expr *result = new Analyzer::UOper(kBOOLEAN, kISNULL, arg_expr);
		if (is_not)
			result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
		return result;
	}

	Analyzer::Expr *
	InSubquery::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		throw std::runtime_error("Subqueries are not supported yet.");
		return nullptr;
	}

	Analyzer::Expr *
	InValues::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Expr *arg_expr = arg->analyze(catalog, query);
		std::list<Analyzer::Expr*> *value_exprs = new std::list<Analyzer::Expr*>();
		for (auto p : *value_list) {
			Analyzer::Expr *e = p->analyze(catalog, query);
			value_exprs->push_back(e->add_cast(arg_expr->get_type_info()));
		}
		Analyzer::Expr *result = new Analyzer::InValues(arg_expr, value_exprs);
		if (is_not)
			result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
		return result;
	}

	Analyzer::Expr *
	BetweenExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Expr *arg_expr = arg->analyze(catalog, query);
		Analyzer::Expr *lower_expr = lower->analyze(catalog, query);
		Analyzer::Expr *upper_expr = upper->analyze(catalog, query);
		SQLTypeInfo new_left_type, new_right_type;
		(void)Analyzer::BinOper::analyze_type_info(kGE, arg_expr->get_type_info(), lower_expr->get_type_info(), &new_left_type, &new_right_type);
		Analyzer::BinOper *lower_pred = new Analyzer::BinOper(kBOOLEAN, kGE, kONE, arg_expr->add_cast(new_left_type), lower_expr->add_cast(new_right_type));
		(void)Analyzer::BinOper::analyze_type_info(kLE, arg_expr->get_type_info(), lower_expr->get_type_info(), &new_left_type, &new_right_type);
		Analyzer::BinOper *upper_pred = new Analyzer::BinOper(kBOOLEAN, kLE, kONE, arg_expr->deep_copy()->add_cast(new_left_type), upper_expr->add_cast(new_right_type));
		Analyzer::Expr *result = new Analyzer::BinOper(kBOOLEAN, kAND, kONE, lower_pred, upper_pred);
		if (is_not)
			result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
		return result;
	}

	Analyzer::Expr *
	LikeExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		Analyzer::Expr *arg_expr = arg->analyze(catalog, query);
		Analyzer::Expr *like_expr = like_string->analyze(catalog, query);
		Analyzer::Expr *escape_expr = escape_string == nullptr ? nullptr: escape_string->analyze(catalog, query);
		if (!IS_STRING(arg_expr->get_type_info().type))
			throw std::runtime_error("expression before LIKE must be of a string type.");
		if (!IS_STRING(like_expr->get_type_info().type))
			throw std::runtime_error("expression after LIKE must be of a string type.");
		if (escape_expr != nullptr && !IS_STRING(escape_expr->get_type_info().type))
			throw std::runtime_error("expression after ESCAPE must be of a string type.");
		Analyzer::Expr *result = new Analyzer::LikeExpr(arg_expr, like_expr, escape_expr);
		if (is_not)
			result = new Analyzer::UOper(kBOOLEAN, kNOT, result);
		return result;
	}

	Analyzer::Expr *
	ExistsExpr::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		throw std::runtime_error("Subqueries are not supported yet.");
		return nullptr;
	}

	Analyzer::Expr *
	ColumnRef::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		int table_id;
		int rte_idx;
		const ColumnDescriptor *cd;
		if (column == nullptr)
			throw std::runtime_error("invalid column name *.");
		if (table != nullptr) {
			rte_idx = query.get_rte_idx(*table);
			if (rte_idx < 0)
				throw std::runtime_error("range variable or table name " + *table + " does not exist.");
			Analyzer::RangeTblEntry *rte = query.get_rte(rte_idx);
			cd = rte->get_column_desc(catalog, *column);
			if (cd == nullptr)
				throw std::runtime_error("Column name " + *column + " does not exist.");
			table_id = rte->get_table_id();
		} else {
			bool found = false;
			int i = 0;
			for (auto rte : query.get_rangetable()) {
				cd = rte->get_column_desc(catalog, *column);
				if (cd != nullptr && !found) {
					found = true;
					rte_idx = i;
					table_id = rte->get_table_id();
				} else if (cd != nullptr && found)
					throw std::runtime_error("Column name " + *column + " is ambiguous.");
				i++;
			}
			if (cd == nullptr)
				throw std::runtime_error("Column name " + *column + " does not exist.");
		}
		return new Analyzer::ColumnVar(cd->columnType, table_id, cd->columnId, rte_idx, cd->compression, cd->comp_param);
	}

	Analyzer::Expr *
	FunctionRef::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const 
	{
		SQLTypeInfo result_type;
		SQLAgg agg_type;
		Analyzer::Expr *arg_expr;
		bool is_distinct = false;
		if (boost::iequals(*name, "count")) {
			result_type.type = kINT;
			agg_type = kCOUNT;
			if (arg == nullptr)
				arg_expr = nullptr;
			else
				arg_expr = arg->analyze(catalog, query);
			is_distinct = distinct;
		}
		else if (boost::iequals(*name, "min")) {
			agg_type = kMIN;
			arg_expr = arg->analyze(catalog, query);
			result_type = arg_expr->get_type_info();
		}
		else if (boost::iequals(*name, "max")) {
			agg_type = kMAX;
			arg_expr = arg->analyze(catalog, query);
			result_type = arg_expr->get_type_info();
		}
		else if (boost::iequals(*name, "avg")) {
			agg_type = kAVG;
			arg_expr = arg->analyze(catalog, query);
			result_type = arg_expr->get_type_info();
		}
		else if (boost::iequals(*name, "sum")) {
			agg_type = kSUM;
			arg_expr = arg->analyze(catalog, query);
			result_type = arg_expr->get_type_info();
		}
		else
			throw std::runtime_error("invalid function name: " + *name);
		int naggs = query.get_num_aggs();
		query.set_num_aggs(naggs+1);
		return new Analyzer::AggExpr(result_type, agg_type, arg_expr, is_distinct);
	}

	void
	UnionQuery::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		left->analyze(catalog, query);
		Analyzer::Query *right_query = new Analyzer::Query();
		right->analyze(catalog, *right_query);
		query.set_next_query(right_query);
		query.set_is_unionall(is_unionall);
	}

	void
	QuerySpec::analyze_having_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		if (having_clause == nullptr) {
			query.set_having_predicate(nullptr);
			return;
		}
		Analyzer::Expr *p = having_clause->analyze(catalog, query);
		if (p->get_type_info().type != kBOOLEAN)
			throw std::runtime_error("Only boolean expressions can be in HAVING clause.");
		p->check_group_by(query.get_group_by());
		query.set_having_predicate(p);
	}

	void
	QuerySpec::analyze_group_by(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		if (groupby_clause == nullptr) {
			query.set_group_by(nullptr);
			return;
		}
		std::list<Analyzer::Expr*> *groupby = new std::list<Analyzer::Expr*>();
		for (auto c : *groupby_clause) {
			Analyzer::Expr *e = c->analyze(catalog, query);
			groupby->push_back(e);
		}
		for (auto t : query.get_targetlist()) {
			Analyzer::Expr *e = t->get_expr();
			if (typeid(*e) != typeid(Analyzer::AggExpr))
				e->check_group_by(groupby);
		}
		query.set_group_by(groupby);
	}

	void
	QuerySpec::analyze_where_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		if (where_clause == nullptr) {
			query.set_where_predicate(nullptr);
			return;
		}
		Analyzer::Expr *p = where_clause->analyze(catalog, query);
		if (p->get_type_info().type != kBOOLEAN)
			throw std::runtime_error("Only boolean expressions can be in WHERE clause.");
		query.set_where_predicate(p);
	}

	void
	QuerySpec::analyze_select_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		std::list<Analyzer::TargetEntry*> &tlist = query.get_targetlist_nonconst();
		if (select_clause == nullptr) {
			// this means SELECT *
			int rte_idx = 0;
			for (auto rte : query.get_rangetable()) {
				rte->expand_star_in_targetlist(catalog, tlist, rte_idx++);
			}
		}
		else {
			for (auto p : *select_clause) {
				const Parser::Expr *select_expr = p->get_select_expr();
				// look for the case of range_var.*
				if (typeid(*select_expr) == typeid(ColumnRef) &&
						dynamic_cast<const ColumnRef*>(select_expr)->get_column() == nullptr) {
						const std::string *range_var_name = dynamic_cast<const ColumnRef*>(select_expr)->get_table();
						int rte_idx = query.get_rte_idx(*range_var_name);
						if (rte_idx < 0)
							throw std::runtime_error("invalid range variable name: " + *range_var_name);
						Analyzer::RangeTblEntry *rte = query.get_rte(rte_idx);
						rte->expand_star_in_targetlist(catalog, tlist, rte_idx);
				}
				else {
					Analyzer::Expr *e = select_expr->analyze(catalog, query);
					std::string resname;

					if (p->get_alias() != nullptr)
						resname = *p->get_alias();
					else if (typeid(*e) == typeid(Analyzer::ColumnVar)) {
						Analyzer::ColumnVar *colvar = dynamic_cast<Analyzer::ColumnVar*>(e);
						const ColumnDescriptor *col_desc = catalog.getMetadataForColumn(colvar->get_table_id(), colvar->get_column_id());
						resname = col_desc->columnName;
					}
					Analyzer::TargetEntry *tle = new Analyzer::TargetEntry(resname, e);
					tlist.push_back(tle);
				}
			}
		}
	}

	void
	QuerySpec::analyze_from_clause(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		Analyzer::RangeTblEntry *rte;
		for (auto p : *from_clause) {
			const TableDescriptor *table_desc;
			table_desc = catalog.getMetadataForTable(*p->get_table_name());
			if (table_desc == nullptr)
				throw std::runtime_error("Table " + *p->get_table_name() + " does not exist." );
			std::string range_var;
			if (p->get_range_var() == nullptr)
				range_var = *p->get_table_name();
			else
				range_var = *p->get_range_var();
			rte = new Analyzer::RangeTblEntry(range_var, table_desc, nullptr);
			query.add_rte(rte);
		}
	}

	void
	QuerySpec::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		query.set_is_distinct(is_distinct);
		analyze_from_clause(catalog, query);
		analyze_select_clause(catalog, query);
		analyze_where_clause(catalog, query);
		analyze_group_by(catalog, query);
		analyze_having_clause(catalog, query);
	}

	void
	SelectStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		query.set_stmt_type(kSELECT);
		query_expr->analyze(catalog, query);
		if (orderby_clause == nullptr) {
			query.set_order_by(nullptr);
			return;
		}
		const std::list<Analyzer::TargetEntry*> &tlist = query.get_targetlist();
		std::list<Analyzer::OrderEntry> *order_by = new std::list<Analyzer::OrderEntry>();
		for (auto p : *orderby_clause) {
			int tle_no = p->get_colno();
			if (tle_no == 0) {
				// use column name
				// search through targetlist for matching name
				const std::string *name = p->get_column()->get_column();
				tle_no = 1;
				bool found = false;
				for (auto tle : tlist) {
					if (tle->get_resname() == *name) {
						found = true;
						break;
					}
					tle_no++;
				}
				if (!found)
					throw std::runtime_error("invalid name in order by: " + *name);
			}
			order_by->push_back(Analyzer::OrderEntry(tle_no, p->get_is_desc(), p->get_nulls_first()));
		}
		query.set_order_by(order_by);
	}

	void
	InsertStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		query.set_stmt_type(kINSERT);
		const TableDescriptor *td = catalog.getMetadataForTable(*table);
		if (td == nullptr)
			throw std::runtime_error("Table " + *table + " does not exist.");
		if (td->isView)
			throw std::runtime_error("Insert to views is not supported yet.");
		Analyzer::RangeTblEntry *rte = new Analyzer::RangeTblEntry(*table, td, nullptr);
		query.set_result_table_id(td->tableId);
		std::list<int> result_col_list;
		if (column_list == nullptr) {
			const std::vector<const ColumnDescriptor *> all_cols = catalog.getAllColumnMetadataForTable(td->tableId);
			for (auto cd : all_cols)
				result_col_list.push_back(cd->columnId);
		} else {
			for (auto c : *column_list) {
				const ColumnDescriptor *cd = catalog.getMetadataForColumn(td->tableId, *c);
				if (cd == nullptr)
					throw std::runtime_error("Column " + *c + " does not exist.");
				result_col_list.push_back(cd->columnId);
			}
		}
		query.set_result_col_list(result_col_list);
	}

	void
	InsertValuesStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		InsertStmt::analyze(catalog, query);
		std::list<Analyzer::TargetEntry*> &tlist = query.get_targetlist_nonconst();
		for (auto v : *value_list) {
			Analyzer::Expr *e = v->analyze(catalog, query);
			tlist.push_back(new Analyzer::TargetEntry("", e));
		}
	}

	void
	InsertQueryStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &insert_query) const
	{
		InsertStmt::analyze(catalog, insert_query);
		query->analyze(catalog, insert_query);
	}

	void
	UpdateStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		throw std::runtime_error("UPDATE statement not supported yet.");
	}

	void
	DeleteStmt::analyze(const Catalog_Namespace::Catalog &catalog, Analyzer::Query &query) const
	{
		throw std::runtime_error("DELETE statement not supported yet.");
	}

	void
	CreateTableStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		if (catalog.getMetadataForTable(*table) != nullptr)
			throw std::runtime_error("Table " + *table + " already exits.");
		std::vector<ColumnDescriptor *> columns;
		for (auto e : *table_element_list) {
			if (typeid(*e) != typeid(ColumnDef))
				throw std::runtime_error("Table constraints are not supported yet.");
			ColumnDef *coldef = dynamic_cast<ColumnDef*>(e);
			ColumnDescriptor *cd = new ColumnDescriptor();
			cd->columnName = *coldef->get_column_name();
			const SQLType *t = coldef->get_column_type();
			cd->columnType.type = t->get_type();
			cd->columnType.dimension = t->get_param1();
			cd->columnType.scale = t->get_param2();
			const ColumnConstraintDef *cc = coldef->get_column_constraint();
			if (cc == nullptr)
				cd->columnType.notnull = false;
			else {
				cd->columnType.notnull = cc->get_notnull();
			}
			const CompressDef *compression = coldef->get_compression();
			if (compression == nullptr) {
				cd->compression = kENCODING_NONE;
				cd->comp_param = 0;
			} else {
				const std::string &comp = *compression->get_encoding_name();
				if (boost::iequals(comp, "fixed")) {
					// fixed-bits encoding
					if (compression->get_encoding_param() == 0 || compression->get_encoding_param() % 8 != 0 || compression->get_encoding_param() > 48)
						throw std::runtime_error("Must specify number of bits as 8, 16, 24, 32 or 48 as the parameter to fixed-bits encoding.");
					cd->compression = kENCODING_FIXED;
					cd->comp_param = compression->get_encoding_param();
				} else if (boost::iequals(comp, "rl")) {
					// run length encoding
					cd->compression = kENCODING_RL;
					cd->comp_param = 0;
				} else if (boost::iequals(comp, "diff")) {
					// differential encoding
					cd->compression = kENCODING_DIFF;
					cd->comp_param = 0;
				} else if (boost::iequals(comp, "dict")) {
					// diciontary encoding
					cd->compression = kENCODING_DICT;
					cd->comp_param = 0;
				} else if (boost::iequals(comp, "sparse")) {
					// sparse column encoding with mostly NULL values
					if (cd->columnType.notnull)
						throw std::runtime_error("Cannot do sparse column encoding on a NOT NULL column.");
					if (compression->get_encoding_param() == 0 || compression->get_encoding_param() % 8 != 0 || compression->get_encoding_param() > 48)
						throw std::runtime_error("Must specify number of bits as 8, 16, 24, 32 or 48 as the parameter to sparse-column encoding.");
					cd->compression = kENCODING_SPARSE;
					cd->comp_param = compression->get_encoding_param();
				} else
					throw std::runtime_error("Invalid column compression scheme " + comp);
			}
			columns.push_back(cd);
		}
		catalog.createTable(*table, columns);
	}

	void
	DropTableStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		catalog.dropTable(*table);
	}

	void
	CreateViewStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		throw std::runtime_error("CREATE VIEW not supported yet.");
	}

	void
	DropViewStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		throw std::runtime_error("DROP VIEW not supported yet.");
	}

	CreateUserStmt::~CreateUserStmt()
	{
		delete user_name;
		if (name_value_list != nullptr) {
			for (auto p : *name_value_list)
				delete p;
			delete name_value_list;
		}
	}

	CreateDBStmt::~CreateDBStmt()
	{
		delete db_name;
		if (name_value_list != nullptr) {
			for (auto p : *name_value_list)
				delete p;
			delete name_value_list;
		}
	}

	void
	CreateDBStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
			throw std::runtime_error("Must be in the system database to create databases.");
		Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
		int ownerId = catalog.get_currentUser().userId;
		if (name_value_list != nullptr) {
			for (auto p : *name_value_list) {
				if (boost::iequals(*p->get_name(), "owner")) {
					Catalog_Namespace::UserMetadata user;
					if (!syscat.getMetadataForUser(*p->get_value(), user))
						throw std::runtime_error("User " + *p->get_value() + " does not exist.");
					ownerId = user.userId;
				}
				else
					throw std::runtime_error("Invalid CREATE DATABASE option " + *p->get_name() + ". Only OWNER supported.");
			}
		}
		syscat.createDatabase(*db_name, ownerId);
	}

	void
	DropDBStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
			throw std::runtime_error("Must be in the system database to drop databases.");
		Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
		syscat.dropDatabase(*db_name);
	}

	void
	CreateUserStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		std::string passwd;
		bool is_super = false;
		for (auto p : *name_value_list) {
			if (boost::iequals(*p->get_name(), "password"))
				passwd = *p->get_value();
			else if (boost::iequals(*p->get_name(), "is_super")) {
				if (boost::iequals(*p->get_value(), "true"))
					is_super = true;
				else if (boost::iequals(*p->get_value(), "false"))
					is_super = false;
				else
					throw std::runtime_error("Value to IS_SUPER must be TRUE or FALSE.");
			} else
				throw std::runtime_error("Invalid CREATE USER option " + *p->get_name() + ".  Should be PASSWORD or IS_SUPER.");
		}
		if (passwd.empty())
			throw std::runtime_error("Must have a password for CREATE USER.");
		if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
			throw std::runtime_error("Must be in the system database to create users.");
		Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
		syscat.createUser(*user_name, passwd, is_super);
	}

	AlterUserStmt::~AlterUserStmt()
	{
		delete user_name;
		if (name_value_list != nullptr) {
			for (auto p : *name_value_list)
				delete p;
			delete name_value_list;
		}
	}

	void
	AlterUserStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		const std::string *passwd = nullptr;
		bool is_super = false;
		bool *is_superp = nullptr;
		for (auto p : *name_value_list) {
			if (boost::iequals(*p->get_name(), "password"))
				passwd = p->get_value();
			else if (boost::iequals(*p->get_name(), "is_super")) {
				if (boost::iequals(*p->get_value(), "true")) {
					is_super = true;
					is_superp = &is_super;
				} else if (boost::iequals(*p->get_value(), "false")) {
					is_super = false;
					is_superp = &is_super;
				} else
					throw std::runtime_error("Value to IS_SUPER must be TRUE or FALSE.");
			} else
				throw std::runtime_error("Invalid CREATE USER option " + *p->get_name() + ".  Should be PASSWORD or IS_SUPER.");
		}
		Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
		syscat.alterUser(*user_name, passwd, is_superp);
	}

	void
	DropUserStmt::execute(Catalog_Namespace::Catalog &catalog)
	{
		if (catalog.get_currentDB().dbName != MAPD_SYSTEM_DB)
			throw std::runtime_error("Must be in the system database to drop users.");
		Catalog_Namespace::SysCatalog &syscat = static_cast<Catalog_Namespace::SysCatalog&>(catalog);
		syscat.dropUser(*user_name);
	}

}
