/**
 * @file	ExprWalker.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Steps through insert statements in the SQL AST.
 *
 */
#include <sstream>
#include "ExprWalker.h"
#include "../../Shared/types.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/DmlStmt.h"
#include "../Parse/SQL/ast/SelectStmt.h"
#include "../Parse/SQL/ast/FromClause.h"
#include "../Parse/SQL/ast/TableList.h" 
#include "../Parse/SQL/ast/Table.h"
 #include "../Parse/SQL/ast/Literal.h"
#include "../Parse/SQL/ast/Selection.h"
#include "../Parse/SQL/ast/ScalarExprList.h"
#include "../Parse/SQL/ast/ScalarExpr.h"
#include "../Parse/SQL/ast/Column.h"
 
namespace Analysis_Namespace {

void ExprWalker::visit(SqlStmt *v) {
	if (v->n1) v->n1->accept(*this); // DmlStmt
}

void ExprWalker::visit(DmlStmt *v) {
	if (v->n2) v->n2->accept(*this); // SelectStmt
}

void ExprWalker::visit(SelectStmt *v) {
	
	// visit FromClause to gather Table nodes
	if (v->n3) v->n3->accept(*this); // FromClause

	// visit Selection to gather Column nodes
	if (v->n2) v->n2->accept(*this); // Selection

	// Check for column name ambiguity, which occurs when a column name
	// belongs to more than one table.
	{
		mapd_err_t err = MAPD_SUCCESS;
		bool ambiguous = false;
		for (int i = 0; i < colNodes_.size(); ++i) {
			for (int j = 0; j < tblNodes_.size(); ++j) {
				// @todo Actually do the check instead of printing out a message
				printf("Check table \"%s\" for column \"%s\"\n", tblNodes_[j]->name1.c_str(), colNodes_[i]->colName.c_str());
			}
		}
	}

	// if (v->n4 && v->n4->n1) v->n4->n1->accept(*this); // OptWhere -> SearchCondition
	// if (v->n5 && v->n5->n1) v->n5->n1->accept(*this); // OptGroupby -> ColumnList
	// if (v->n6 && v->n6->n1) v->n6->n1->accept(*this); // OptHaving -> ??
	// if (v->n7 && v->n7->n1) v->n7->n1->accept(*this); // OptOrderby -> OrderbyColumnList
	// if (v->n8 && v->n8->n1) v->n8->n1->accept(*this); // OptLimit -> ??
}

void ExprWalker::visit(FromClause *v) {
	if (v->n1) v->n1->accept(*this); // TableList
	if (v->n2) v->n2->accept(*this); // SelectStmt
}

void ExprWalker::visit(TableList *v) {
	if (v->n1) v->n1->accept(*this); // TableList
	if (v->n2) v->n2->accept(*this); // Table	
}

void ExprWalker::visit(Table *v) {
	tblNodes_.push_back(v);
}

void ExprWalker::visit(Selection *v) {
	// does not visit if "all columns" flag is set;
	// i.e., SELECT * ...
	// @todo ExprWalker should type check when user specifies SELECT *, but this is not currently done
	if (v->n1 && !v->all)
		v->n1->accept(*this); // ScalarExprList
}

void ExprWalker::visit(ScalarExprList *v) {
	if (v->n1) v->n1->accept(*this); // ScalarExprList
	if (v->n2) v->n2->accept(*this); // ScalarExpr
}

void ExprWalker::visit(ScalarExpr *v) {
	if (v->n1 && !v->n2) { // '(' ScalarExpr ')'
		v->n1->accept(*this);
	}
	else if (v->n1 && v->n2) { // ScalarExpr OP ScalarExpr
		v->n1->accept(*this);
		v->n2->accept(*this);
	}
	else if (v->n3) // Literal
		v->n3->accept(*this);
	else if (v->n4) // Column
		v->n4->accept(*this);
}

void ExprWalker::visit(Column *v) {
	colNodes_.push_back(v);
}

} // Analysis_Namespace
