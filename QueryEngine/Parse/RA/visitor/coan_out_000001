
/**
 * @file    QPTranslator.cpp
 * @author  Steven Stewart <steve@map-d.com>
 */
#include "QPTranslator.h"

#include "../ast/RelAlgNode.h"
#include "../ast/UnaryOp.h"
#include "../ast/BinaryOp.h"

#include "../ast/AggrExpr.h"
#include "../ast/AggrList.h"
#include "../ast/AntijoinOp.h"
#include "../ast/Attribute.h"
#include "../ast/AttrList.h"
#include "../ast/Comparison.h"
#include "../ast/DiffOp.h"
#include "../ast/Expr.h"
#include "../ast/ExtendOp.h"
#include "../ast/GroupbyOp.h"
#include "../ast/JoinOp.h"
#include "../ast/MathExpr.h"
#include "../ast/OuterjoinOp.h"
#include "../ast/Predicate.h"
#include "../ast/ProductOp.h"
#include "../ast/Program.h"
#include "../ast/ProjectOp.h"
#include "../ast/Relation.h"
#include "../ast/RelExpr.h"
#include "../ast/RelExprList.h"
#include "../ast/RenameOp.h"
#include "../ast/SelectOp.h"
#include "../ast/SemijoinOp.h"
#include "../ast/SortOp.h"
#include "../ast/UnionOp.h"

#include <iostream>
using std::cout;
using std::endl;

namespace RA_Namespace {

void QPTranslator::visit(Attribute *v) {
    printf("<Attribute>\n");
}

void QPTranslator::visit(AggrExpr *v) {
    printf("<AggrExpr>\n");
    if (v->n1) v->n1->accept(*this);
}

void QPTranslator::visit(AttrList *v) {
    printf("<AttrList>\n");
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
}

void QPTranslator::visit(Comparison *v) {
    printf("<Comparison>\n");
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
}

void QPTranslator::visit(MathExpr *v) {
    printf("<MathExpr>\n");
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
    if (v->n3) v->n3->accept(*this);
    if (v->n4) v->n4->accept(*this);
}

void QPTranslator::visit(Predicate *v) {
    printf("<Predicate>\n");
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
    if (v->n3) v->n3->accept(*this);
}

void QPTranslator::visit(Program *v) {
    printf("<Program>\n");
    if (v->n1) v->n1->accept(*this);
}

void QPTranslator::visit(ProjectOp *v) {
    printf("<ProjectOp>\n");
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
}

void QPTranslator::visit(Relation *v) {
    printf("<Relation>\n");
}

void QPTranslator::visit(RelExpr *v) {
    printf("<RelExpr>\n");
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
    if (v->n3) v->n3->accept(*this);
    if (v->n4) v->n4->accept(*this);
}

void QPTranslator::visit(RelExprList *v) {
    printf("<RelExprList>\n");
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
}

void QPTranslator::visit(SelectOp *v) {
    printf("<SelectOp>\n");
    if (v->n1) v->n1->accept(*this);
    if (v->n2) v->n2->accept(*this);
}

} // RA_Namespace
