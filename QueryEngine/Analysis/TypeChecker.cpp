/**
 * @file	TypeChecker.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Type checks SQL statements (except for INSERT).
 *
 * @see TypeChecker.h for type checking insert statements.
 */
#include "TypeChecker.h"

#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/DmlStmt.h"
#include "../Parse/SQL/ast/FromClause.h"
#include "../Parse/SQL/ast/Literal.h"
#include "../Parse/SQL/ast/ScalarExpr.h"
#include "../Parse/SQL/ast/ScalarExprList.h"
#include "../Parse/SQL/ast/SelectStmt.h"
#include "../Parse/SQL/ast/Selection.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/TableList.h"
#include "../Parse/SQL/ast/Table.h"


namespace Analysis_Namespace {

void TypeChecker::visit(Column *v) {
	printf("<Column>\n");
	colNames_.push_back(v->name);	
}

void TypeChecker::visit(DmlStmt *v) {
	printf("<DmlStmt>\n");
	if (v->n2) v->n2->accept(*this); // SelectStmt
}

void TypeChecker::visit(Literal *v) {
	printf("<Literal>\n");
}

void TypeChecker::visit(ScalarExprList *v) {
	printf("<ScalarExprList>\n");
	if (v->n1) v->n1->accept(*this); // ScalarExpr
	if (v->n2) v->n2->accept(*this); // ScalarExprList	
}

void TypeChecker::visit(ScalarExpr *v) {
	printf("<ScalarExpr>\n");
	
	// case: ScalarExpr OP ScalarExpr
	if (v->n1 && v->n2) {
		v->n1->accept(*this);
		v->n2->accept(*this);

		if (v->n1->type == FLOAT_TYPE || v->n2->type == FLOAT_TYPE)
			v->type = FLOAT_TYPE;
		else
			v->type = INT_TYPE;
	}

	// case: ( ScalarExpr )
	else if (v->n1 && !v->n2) {
		v->n1->accept(*this);
		v->type = v->n1->type;
	}

	// case: Literal
	else if (v->n3) {
		v->n3->accept(*this); // Literal
		v->type = v->n3->type;
	}

	// case: Column
	else if (v->n4) {
		v->n4->accept(*this); // Column
		v->type = v->n4->column_type;
	}

	printf("v->type=%d\n", v->type);
}

void TypeChecker::visit(SelectStmt *v) {
	printf("<SelectStmt>\n");

	if (v->n2) v->n2->accept(*this); // Selection
	for (int i = 0; i < colNames_.size(); ++i)
		printf("%s%s%s", i == 0 ? "columns: ":"", colNames_[i].second.c_str(), (i+1) == colNames_.size() ? "\n" : ", ");

	if (v->n3) v->n3->accept(*this); // FromClause
	for (int i = 0; i < tblNames_.size(); ++i)
		printf("%s%s%s", i == 0 ? "tables:  ":"", tblNames_[i].c_str(), (i+1) == tblNames_.size() ? "\n" : ", ");

	// Request metadata for the columns
	/*std::vector<ColumnRow> colMetadata;
	mapd_err_t err = c_.getMetadataForColumns(v->n1->name1, colNames_, colMetadata);

	// Check for error (table or column does not exists)
	if (err != MAPD_SUCCESS) {
		errFlag_ = true;
		if (err == MAPD_ERR_TABLE_DOES_NOT_EXIST)
			errMsg_ = "Table \"" + v->n1->name1 + "\" does not exist";
		else if (err == MAPD_ERR_COLUMN_DOES_NOT_EXIST)
			errMsg_ = "Column \"" + colNames_[colMetadata.size()] + "\" does not exist";
		else if (err != MAPD_SUCCESS)
			errMsg_ = "Catalog returned an error.";
		
		colNames_.clear();
		colMetadata.clear();
		return;
	}*/

	// Check for ambiguity
	
}

void TypeChecker::visit(Selection *v) {
	printf("<Selection>\n");
	if (v->n1) v->n1->accept(*this); // ScalarExprList
	// @todo select * ...
}

void TypeChecker::visit(SqlStmt *v) {
	printf("<SqlStmt>\n");
	if (v->n1) v->n1->accept(*this); // DmlStmt
}

void TypeChecker::visit(FromClause *v) {
	printf("<FromClause>\n");
	if (v->n1) v->n1->accept(*this); // TableList
	// @todo Nested select statement within from clause
}

void TypeChecker::visit(TableList *v) {
	printf("<TableList>\n");
	if (v->n1) v->n1->accept(*this); // TableList
	if (v->n2) v->n2->accept(*this); // Table
}

void TypeChecker::visit(Table *v) {
	printf("<Table>\n");
	tblNames_.push_back(v->name.second);
}

} // Analysis_Namespace