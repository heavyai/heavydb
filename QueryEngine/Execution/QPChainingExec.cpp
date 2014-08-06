/**
 * @file	QPChainingExec.cpp
 * @author	Steven Stewart <steve@map-d.com>
 *
 * Implementation of RA query plan walker/executor.
 */
#include <iostream>
#include <utility>
#include "QPChainingExec.h"

#include "../Parse/RA/ast/AggrExpr.h"
#include "../Parse/RA/ast/Attribute.h"
#include "../Parse/RA/ast/AttrList.h"
#include "../Parse/RA/ast/BinaryOp.h"
#include "../Parse/RA/ast/Comparison.h"
#include "../Parse/RA/ast/MathExpr.h"
#include "../Parse/RA/ast/Predicate.h"
#include "../Parse/RA/ast/Program.h"
#include "../Parse/RA/ast/ProjectOp.h"
#include "../Parse/RA/ast/Relation.h"
#include "../Parse/RA/ast/RelExpr.h"
#include "../Parse/RA/ast/RelExprList.h"
#include "../Parse/RA/ast/SelectOp.h"
#include "../Parse/RA/ast/UnaryOp.h"

namespace Execution_Namespace {

void QPChainingExec::visit(Attribute *v) {
	printf("<Attribute>\n");
}

void QPChainingExec::visit(AggrExpr *v) {
	printf("<AggrExpr>\n");
	if (v->n1) v->n1->accept(*this);
}

void QPChainingExec::visit(AttrList *v) {
	printf("<AttrList>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPChainingExec::visit(Comparison *v) {
	printf("<Comparison>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPChainingExec::visit(MathExpr *v) {
	printf("<MathExpr>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	if (v->n4) v->n4->accept(*this);
}

void QPChainingExec::visit(Predicate *v) {
	printf("<Predicate>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
}

void QPChainingExec::visit(Program *v) {
	printf("<Program>\n");
	if (v->n1) v->n1->accept(*this);
}

void QPChainingExec::visit(ProjectOp *v) {
	printf("<ProjectOp>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPChainingExec::visit(Relation *v) {
	printf("<Relation>\n");
}

void QPChainingExec::visit(RelExpr *v) {
	printf("<RelExpr>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
	if (v->n3) v->n3->accept(*this);
	if (v->n4) v->n4->accept(*this);
}

void QPChainingExec::visit(RelExprList *v) {
	printf("<RelExprList>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

void QPChainingExec::visit(SelectOp *v) {
	printf("<SelectOp>\n");
	if (v->n1) v->n1->accept(*this);
	if (v->n2) v->n2->accept(*this);
}

} // Execution_Namespace
