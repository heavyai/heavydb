/**
 * @file 	SQL_RA_Translator.cpp
 * @author	Steven Stewart <steve@map-d.com>
 */
#include "SQL_RA_Translator.h"

#include "../Parse/SQL/ast/Column.h"
#include "../Parse/SQL/ast/DdlStmt.h"
#include "../Parse/SQL/ast/DmlStmt.h"
#include "../Parse/SQL/ast/FromClause.h"
#include "../Parse/SQL/ast/InsertStmt.h"
#include "../Parse/SQL/ast/ScalarExpr.h" 
#include "../Parse/SQL/ast/ScalarExprList.h"
#include "../Parse/SQL/ast/Selection.h" 
#include "../Parse/SQL/ast/SelectStmt.h"
#include "../Parse/SQL/ast/SqlStmt.h"
#include "../Parse/SQL/ast/Table.h"
#include "../Parse/SQL/ast/TableList.h"

#include "../Parse/RA/ast/AttrList.h"
#include "../Parse/RA/ast/Attribute.h"
#include "../Parse/RA/ast/ProductOp.h"
#include "../Parse/RA/ast/Program.h"
#include "../Parse/RA/ast/ProjectOp.h"
#include "../Parse/RA/ast/Relation.h"
#include "../Parse/RA/ast/RelExpr.h"
#include "../Parse/RA/ast/RelExprList.h"

using RA_Namespace::AttrList;
using RA_Namespace::Attribute;
using RA_Namespace::ProductOp;
using RA_Namespace::Program;
using RA_Namespace::ProjectOp;
using RA_Namespace::Relation;
using RA_Namespace::RelExpr;
using RA_Namespace::RelExprList;

using std::to_string;

namespace Translate_Namespace {

SQL_RA_Translator::SQL_RA_Translator() {
	
}

void SQL_RA_Translator::visit(DdlStmt *v) {
	printf("<DdlStmt>\n");
}

void SQL_RA_Translator::visit(DmlStmt *v) {
	printf("<DmlStmt>\n");
	if (v->n1) v->n1->accept(*this); // InsertStmt
	if (v->n2) v->n2->accept(*this); // SelectStmt

    assert(nodeSqlStmt_);
	root = new Program(new RelExprList(nodeSqlStmt_));
}

void SQL_RA_Translator::visit(FromClause *v) {
	printf("<FromClause>\n");
	if (v->n1) v->n1->accept(*this); // TableList
	// if (v->n2) v->n1->accept(*this); // SelectStmt (nested query)
	int numTbls = tableIds_.size();

	// Print out collected table ids (for debugging)
	printf("numTbls = %d\n", numTbls);
	for (unsigned i = 0; i < tableIds_.size(); ++i)
		printf("%d ", tableIds_[i]);
	printf("\n");

	// case: 1 table
	if (numTbls == 1)
		nodeFromClause_ = new RelExpr(new Relation(to_string(tableIds_[0])));

	// case: > 1 table
	else if (numTbls > 1) {
		std::string tbl1 = to_string(tableIds_.back());
		tableIds_.pop_back();
		std::string tbl2 = to_string(tableIds_.back());
		tableIds_.pop_back();

		RelExpr *rexp1 = new RelExpr(new Relation(tbl1));
		RelExpr *rexp2 = new RelExpr(new Relation(tbl2));
		ProductOp *p = new ProductOp(rexp1, rexp2);
		ProductOp *n = new ProductOp(new RelExpr(new Relation(tbl1)), new RelExpr(new Relation(tbl2)));
		while (tableIds_.size() > 0) {
			n = new ProductOp(new RelExpr(new Relation(to_string(tableIds_.back()))), new RelExpr(n));
			tableIds_.pop_back();
		}
		nodeFromClause_ = new RelExpr(p);
	}

	// case: 0 tables, do nothing
    else
        nodeFromClause_ = NULL;
}

void SQL_RA_Translator::visit(Column *v) {
	printf("<Column>\n");
	columnIds_.push_back(v->column_id);
}

void SQL_RA_Translator::visit(InsertStmt *v) {
	printf("<InsertStmt>\n");
}

void SQL_RA_Translator::visit(ScalarExpr *v) {
	printf("<ScalarExpr>\n");
	if (v->n4) v->n4->accept(*this); // Column
}

void SQL_RA_Translator::visit(ScalarExprList *v) {
	printf("<ScalarExprList>\n");
	if (v->n1) v->n1->accept(*this); // ScalarExprList
	if (v->n2) v->n2->accept(*this); // ScalarExpr
}

void SQL_RA_Translator::visit(Selection *v) {
	printf("<Selection>\n");
	if (v->n1) v->n1->accept(*this); // ScalarExprList
	int numCols = columnIds_.size();
	
	// Print out collected columns ids (for debugging)
	printf("numCols = %d\n", numCols);
	for (unsigned i = 0; i < columnIds_.size(); ++i)
		printf("%d ", columnIds_[i]);
	printf("\n");

	// case: 1 column
	AttrList *n = NULL;
	if (numCols == 1) {
		n = new AttrList(new Attribute(to_string(columnIds_[0])));
	}

	// case: > 1 column
	else if (numCols > 1) {
		n = new AttrList(new Attribute(to_string(columnIds_[0])));
		while (columnIds_.size() > 0) {
			n = new AttrList(n, new Attribute(to_string(columnIds_.back())));
			columnIds_.pop_back();
		}
	}
    nodeSelection_ = n;
}

void SQL_RA_Translator::visit(SelectStmt *v) {
	printf("<SelectStmt>\n");
	if (v->n2) v->n2->accept(*this); // Selection
	if (v->n3) v->n3->accept(*this); // FromClause

    assert(nodeSelection_ && nodeFromClause_);
	nodeSelectStmt_ =  new ProjectOp((RelExpr*)nodeSelection_, (AttrList*)nodeFromClause_);
}

void SQL_RA_Translator::visit(SqlStmt *v) {
	printf("<SqlStmt>\n");
	if (v->n1) v->n1->accept(*this); // DmlStmt
	if (v->n2) v->n2->accept(*this); // DdlStmt

    assert(nodeSelectStmt_);
	nodeSqlStmt_ = new RelExpr(nodeSelectStmt_);
}

void SQL_RA_Translator::visit(TableList *v) {
	printf("<TableList>\n");
	if (v->n1) v->n1->accept(*this); // TableList
	if (v->n2) v->n2->accept(*this); // Table
}

void SQL_RA_Translator::visit(Table *v) {
	printf("<Table>\n");
	tableIds_.push_back(v->table_id);
}

} // Translate_Namespace
