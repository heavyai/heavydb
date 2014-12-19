/**
 * @file		ParserNode.cpp
 * @author	Wei Hong <wei@map-d.com>
 * @brief		Functions for ParserNode classes
 * 
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 **/

#include "ParserNode.h"

namespace Parser {
	~InValues() {
		for (auto p : *value_list)
			delete p;
	}

	~BetweenExpr() {
		delete arg;
		delete lower;
		delete upper;
	}

	~LikeExpr() {
		delete arg;
		delete like_string;
		if (escape_string != nullptr)
			delete escape_string;
	}

	~ColumnRef() {
		if (table != nullptr)
			delete table;
		delete column;
	}
	
	~TableRef() {
		delete table_name;
		if (range_var != nullptr)
			delete range_var;
	}

	~ColumnConstraintDef() {
		if (defaultval != nullptr)
			delete defaultval;
		if (check_condition != nullptr)
			delete check_condition;
		if (foreign_table != nullptr)
			delete foreign_table;
		if (foreign_column != nullptr)
			delete foreign_column;
	}

	~ColumnDef() {
		delete column_name;
		delete column_type;
		if (column_constraint != nullptr)
			delete column_constraint;
	}

	~UniqueDef() {
		for (auto p : *column_list)
			delete p;
	}

	~ForeignKeyDef() {
		for (auto p : *column_list)
			delete p;
		delete foreign_table;
		if (foreign_column_list != nullptr) {
			for (auto p : *foreign_column_list)
				delete p;
		}
	}

	~FunctionRef() {
		delete name;
		if (arg != nullptr)
			delete arg;
	}

	~QuerySpec() {
		delete select_clause;
		delete from_clause;
		if (where_clause != nullptr)
			delete where_clause;
		if (groupby_clause != nullptr)
			delete groupby_clause;
		if (having_clause != nullptr)
			delete having_clause;
	}

}
