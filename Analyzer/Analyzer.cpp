/**
 * @file		Analyzer.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Analyzer functions
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include <algorithm>
#include <stdexcept>
#include "../Catalog/Catalog.h"
#include "Analyzer.h"

namespace Analyzer {

	Subquery::~Subquery() {
		delete parsetree;
		/*
		if (plan != nullptr)
			delete plan;
			*/
	}

	InValues::~InValues() {
		delete arg;
		for (auto p : *value_list)
			delete p;
		delete value_list;
	}

	RangeTblEntry::~RangeTblEntry() {
		if (view_query != nullptr)
			delete view_query;
	}

	Query::~Query() {
		for (auto p : *targetlist)
			delete p;
		delete targetlist;
		for (auto p : *rangetable)
			delete p;
		delete rangetable;
		if (where_predicate != nullptr)
			delete where_predicate;
		if (group_by != nullptr) {
			for (auto p : *group_by)
				delete p;
			delete group_by;
		}
		if (having_predicate != nullptr)
			delete having_predicate;
		if (order_by != nullptr) {
			delete order_by;
		}
		if (next_query != nullptr)
			delete next_query;
	}

	SQLTypeInfo
	BinOper::analyze_type_info(SQLOps op, const SQLTypeInfo &left_type, const SQLTypeInfo &right_type, SQLTypeInfo *new_left_type, SQLTypeInfo *new_right_type)
	{
		SQLTypeInfo result_type;
		SQLTypeInfo common_type;
		*new_left_type = left_type;
		*new_right_type = right_type;
		if (IS_LOGIC(op)) {
			if (left_type.type != kBOOLEAN || right_type.type != kBOOLEAN)
				throw std::runtime_error("non-boolean operands cannot be used in logic operations.");
			result_type.type = kBOOLEAN;
		} else if (IS_COMPARISON(op)) {
			if ((IS_STRING(left_type.type) && !IS_STRING(right_type.type))
					||
					(!IS_STRING(left_type.type) && IS_STRING(right_type.type)))
				throw std::runtime_error("cannot compare between string and non-string types.");
			if ((IS_NUMBER(left_type.type) && !IS_NUMBER(right_type.type))
					||
					(!IS_NUMBER(left_type.type) && IS_NUMBER(right_type.type)))
				throw std::runtime_error("cannot compare between numeric and non-numeric types.");
			if (IS_NUMBER(left_type.type) && IS_NUMBER(right_type.type)) {
				common_type = common_numeric_type(left_type, right_type);
				*new_left_type = common_type;
				*new_right_type = common_type;
			}
			result_type.type = kBOOLEAN;
		} else if (IS_ARITHMETIC(op)) {
			if (!IS_NUMBER(left_type.type) || !IS_NUMBER(right_type.type))
				throw std::runtime_error("non-numeric operands in arithmetic operations.");
			common_type = common_numeric_type(left_type, right_type);
			*new_left_type = common_type;
			*new_right_type = common_type;
			result_type = common_type;
		} else {
			throw std::runtime_error("invalid binary operator type.");
		}
		return result_type;
	}

	SQLTypeInfo
	common_numeric_type(const SQLTypeInfo &type1, const SQLTypeInfo &type2)
	{
		SQLTypeInfo common_type;
		assert(IS_NUMBER(type1.type) && IS_NUMBER(type2.type));
		if (type1.type == type2.type) {
			common_type.type = type1.type;
			common_type.dimension = std::max(type1.dimension, type2.dimension);
			common_type.scale = std::max(type1.scale, type2.scale);
			return common_type;
		}
		switch (type1.type) {
			case kSMALLINT:
				switch (type2.type) {
				case kINT:
					common_type.type = kINT;
					break;
				case kBIGINT:
					common_type.type = kBIGINT;
					break;
				case kFLOAT:
					common_type.type = kFLOAT;
					break;
				case kDOUBLE:
					common_type.type = kDOUBLE;
					break;
				case kNUMERIC:
				case kDECIMAL:
					common_type.type = kNUMERIC;
					common_type.dimension = std::max(5+type2.scale, type2.dimension);
					common_type.scale = type2.scale;
					break;
				default:
					assert(false);
				}
				break;
			case kINT:
				switch (type2.type) {
					case kSMALLINT:
						common_type.type = kINT;
						break;
					case kBIGINT:
						common_type.type = kBIGINT;
						break;
					case kFLOAT:
						common_type.type = kFLOAT;
						break;
					case kDOUBLE:
						common_type.type = kDOUBLE;
						break;
					case kNUMERIC:
					case kDECIMAL:
						common_type.type = kNUMERIC;
						common_type.dimension = std::max(std::min(19, 10+type2.scale), type2.dimension);
						common_type.scale = type2.scale;
						break;
					default:
						assert(false);
				}
				break;
			case kBIGINT:
				switch (type2.type) {
					case kSMALLINT:
						common_type.type = kBIGINT;
						break;
					case kINT:
						common_type.type = kBIGINT;
						break;
					case kFLOAT:
						common_type.type = kDOUBLE;
						break;
					case kDOUBLE:
						common_type.type = kDOUBLE;
						break;
					case kNUMERIC:
					case kDECIMAL:
						common_type.type = kNUMERIC;
						common_type.dimension = 19; // maximum precision of BIGINT
						common_type.scale = type2.scale;
						break;
					default:
						assert(false);
				}
				break;
			case kFLOAT:
				switch (type2.type) {
					case kSMALLINT:
						common_type.type = kFLOAT;
						break;
					case kINT:
						common_type.type = kFLOAT;
						break;
					case kBIGINT:
						common_type.type = kDOUBLE;
						break;
					case kDOUBLE:
						common_type.type = kDOUBLE;
						break;
					case kNUMERIC:
					case kDECIMAL:
						common_type.type = kDOUBLE;
						break;
					default:
						assert(false);
				}
				break;
			case kDOUBLE:
				switch (type2.type) {
					case kSMALLINT:
					case kINT:
					case kBIGINT:
					case kFLOAT:
					case kNUMERIC:
					case kDECIMAL:
						common_type.type = kDOUBLE;
						break;
					default:
						assert(false);
				}
				break;
			case kNUMERIC:
			case kDECIMAL:
				switch (type2.type) {
					case kSMALLINT:
						common_type.type = kNUMERIC;
						common_type.dimension = std::max(5+type2.scale, type2.dimension);
						common_type.scale = type2.scale;
						break;
					case kINT:
						common_type.type = kNUMERIC;
						common_type.dimension = std::max(std::min(19, 10+type2.scale), type2.dimension);
						common_type.scale = type2.scale;
						break;
					case kBIGINT:
						common_type.type = kNUMERIC;
						common_type.dimension = 19; // maximum precision of BIGINT
						common_type.scale = type2.scale;
						break;
					case kFLOAT:
						common_type.type = kDOUBLE; // promote to DOUBLE
						break;
					case kDOUBLE:
						common_type.type = kDOUBLE;
						break;
					case kNUMERIC:
					case kDECIMAL:
						common_type.type = kNUMERIC;
						common_type.scale = std::max(type1.scale, type2.scale);
						common_type.dimension = std::max(type1.dimension - type1.scale, type2.dimension - type2.scale) + common_type.scale;
						break;
					default:
						assert(false);
				}
				break;
				default:
					assert(false);
		}
		return common_type;
	}

	Expr *
	Expr::add_cast(const SQLTypeInfo &new_type_info)
	{
		if (new_type_info == type_info)
			return this;
		return new UOper(new_type_info, kCAST, this);
	}

	Expr *
	Subquery::add_cast(const SQLTypeInfo &new_type_info)
	{
		// not supported yet.
		return nullptr;
	}

	void
	RangeTblEntry::expand_star_in_targetlist(const Catalog_Namespace::Catalog &catalog, int rte_no, std::vector<TargetEntry*> &tlist)
	{
		column_descs = catalog.getAllColumnMetadataForTable(table_id);
		int col_no = 0;
		int resno = tlist.size() + 1;
		for (auto col_desc : column_descs) {
			ColumnVar *cv = new ColumnVar(col_desc->columnType, rte_no, col_no);
			TargetEntry *tle = new TargetEntry(resno, col_desc->columnName, cv);
			tlist.push_back(tle);
			resno++;
			col_no++;
		}
	}

	int
	RangeTblEntry::get_column_no(const Catalog_Namespace::Catalog &catalog, const std::string &name)
	{
		int col_no = 0;
		for (auto td : column_descs) {
			if (td->columnName == name)
				return col_no;
			col_no++;
		}
		const ColumnDescriptor *td = catalog.getMetadataForColumn(table_id, name);
		if (td == nullptr)
			return -1;
		column_descs.push_back(td);
		return column_descs.size() - 1;
	}

	int
	Query::get_rte_no(const std::string &name)
	{
		int rte_no = 0;
		for (auto rte : *rangetable) {
			if (rte->get_rangevar() == name)
				return rte_no;
			rte_no++;
		}
		return -1;
	}

	void
	ColumnVar::check_group_by(const std::list<Expr*> *groupby) const
	{
		if (groupby != nullptr) {
			for (auto e : *groupby) {
				ColumnVar *c = dynamic_cast<ColumnVar*>(e);
				if (rte_no == c->get_rte_no() && col_no == c->get_col_no())
					return;
			}
		}
		throw std::runtime_error("expressions in the SELECT or HAVING clause must be an aggregate function or an expression over GROUP BY columns.");
	}

	void
	UOper::check_group_by(const std::list<Expr*> *groupby) const
	{
		operand->check_group_by(groupby);
	}

	void
	BinOper::check_group_by(const std::list<Expr*> *groupby) const
	{
		left_operand->check_group_by(groupby);
		right_operand->check_group_by(groupby);
	}

}
