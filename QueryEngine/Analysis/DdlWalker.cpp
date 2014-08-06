/**
 * @file	DdlWalker.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Executes DDL statements by interaction with Catalog.
 *
 * @see DdlWalker.h for type checking insert statements.
 */
#include <utility>
#include <vector>
#include "DdlWalker.h"

#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/ColumnDef.h"
#include "../Parse/SQL/ast/ColumnDefList.h"
#include "../Parse/SQL/ast/CreateStmt.h"
#include "../Parse/SQL/ast/DdlStmt.h"
#include "../Parse/SQL/ast/DropStmt.h"
#include "../Parse/SQL/ast/MapdDataT.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/Table.h"

namespace Analysis_Namespace {

void DdlWalker::visit(Column *v) {
	printf("<Column>\n");
	colNames_.push_back(v->name);
}

void DdlWalker::visit(ColumnDef *v) {
	printf("<ColumnDef>\n");
	if (v->n1) v->n1->accept(*this); // Column
	if (v->n2) v->n2->accept(*this); // MapdDataT
}

void DdlWalker::visit(ColumnDefList *v) {
	printf("<ColumnDefList>\n");
	if (v->n1) v->n1->accept(*this); // ColumnDefList
	if (v->n2) v->n2->accept(*this); // ColumnDef
}

void DdlWalker::visit(CreateStmt *v) {
	printf("<CreateStmt>\n");
	if (v->n1) v->n1->accept(*this); // Table
	if (v->n2) v->n2->accept(*this); // ColumnDefList

	// print out gathered information
	printf("tblName_ = %s\n", tblName_.c_str());
	for (int i = 0; i < colNames_.size(); ++i)
		printf("\t%s, %d\n", colNames_[i].second.c_str(), colTypes_[i]);

	// Prepare request to Catalog
	assert(colNames_.size() == colTypes_.size());
	std::vector<ColumnRow*> colObjs;
	for (int i = 0; i < colNames_.size(); ++i) {
		colObjs.push_back(new ColumnRow(colNames_[i].second, colTypes_[i], true));
	}

	// mapd_err_t addTableWithColumns(const std::string &tableName, std::vector <ColumnRow *> & columns);
	mapd_err_t err = c_.addTableWithColumns(tblName_, colObjs);

	if (err != MAPD_SUCCESS) {
		if (err == MAPD_ERR_TABLE_ALREADY_EXISTS) {
			errFlag_ = true;
			errMsg_ = "table with name " + tblName_ + " already exists";
		}
		else if (err == MAPD_ERR_COLUMN_ALREADY_EXISTS) {
			errFlag_ = true;
			errMsg_ = "duplicate column name specified";
		}
		else {
			errFlag_ = true;
			errMsg_ = "table creation failed";
		}
	}
 }

void DdlWalker::visit(DdlStmt *v) {
	printf("<DdlStmt>\n");
	if (v->n1) v->n1->accept(*this); // CreateStmt
	if (v->n2) v->n2->accept(*this); // DropStmt
	// if (v->n3) v->n3->accept(*this); // AlterStmt
	// if (v->n4) v->n4->accept(*this); // RenameStmt
}

void DdlWalker::visit(DropStmt *v) {
	printf("<DropStmt>\n");
	
	// Get name of table to be dropped
	if (v->n1)
		tblName_ = v->n1->name.second;

	// mapd_err_t removeTable(const std::string &tableName);
	assert(tblName_ != "");

	// Ask Catalog to remove the table
	mapd_err_t err = c_.removeTable(tblName_);

	// Check for error
	if (err != MAPD_SUCCESS) {
		if (err == MAPD_ERR_TABLE_DOES_NOT_EXIST) {
			errFlag_ = true;
			errMsg_ = "table with name " + tblName_ + " does not exist";
		}
		else {
			errFlag_ = true;
			errMsg_ = "drop table failed";
		}
	}
}

void DdlWalker::visit(MapdDataT *v) {
	printf("<MapdDataT>\n");
	colTypes_.push_back(v->type);
}

void DdlWalker::visit(SqlStmt *v) {
	printf("<SqlStmt>\n");
	if (v->n2) {
		tblName_ = "";
		colNames_.clear();
		colTypes_.clear();
		errFlag_ = false;
		errMsg_ = "";
		v->n2->accept(*this); // DdlStmt
	}
}

void DdlWalker::visit(Table *v) {
	printf("<Table>\n");
	tblName_ = v->name.second;
}

} // Analysis_Namespace
