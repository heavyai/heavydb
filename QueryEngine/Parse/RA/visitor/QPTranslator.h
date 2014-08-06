/**
 * @file    QPTranslator.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef QUERYPLAN_PARSE_RA_VISITOR_QPTRANSLATOR_H
#define QUERYPLAN_PARSE_RA_VISITOR_QPTRANSLATOR_H

#include <iostream>
#include <deque>
#include "../visitor/Visitor.h"

using namespace RA_Namespace;

namespace RA_Namespace {

/**
 * @todo brief and detailed descriptions
 */
class QPTranslator : public RA_Namespace::Visitor {

public:
    
    QPTranslator() {}

    virtual void visit(class AggrExpr *v);
    virtual void visit(class AggrList *v);
    virtual void visit(class AntijoinOp *v);
    virtual void visit(class Attribute *v);
    virtual void visit(class AttrList *v);
    virtual void visit(class Comparison *v);
    virtual void visit(class DiffOp *v);
    virtual void visit(class Expr *v);
    virtual void visit(class ExtendOp *v);
    virtual void visit(class GroupbyOp *v);
    virtual void visit(class JoinOp *v);
    virtual void visit(class MathExpr *v);
    virtual void visit(class OuterjoinOp *v);
    virtual void visit(class Predicate *v);
    virtual void visit(class ProductOp *v);
    virtual void visit(class Program *v);
    virtual void visit(class ProjectOp *v);
    virtual void visit(class Relation *v);
    virtual void visit(class RelExpr *v);
    virtual void visit(class RelExprList *v);
    virtual void visit(class RenameOp *v);
    virtual void visit(class SelectOp *v);
    virtual void visit(class SemijoinOp *v);
    virtual void visit(class SortOp *v);
    virtual void visit(class UnionOp *v);

private:
    std::deque<int> qTracker;

};

} // RA_Namespace

#endif // QUERYPLAN_PARSE_RA_VISITOR_QPTRANSLATOR_H


