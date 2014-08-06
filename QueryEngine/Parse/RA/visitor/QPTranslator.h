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
 * @class   QPTranslator
 * @brief   This class walks/executes a query plan, which is an AST of relational algebra statements.
 */
class QPTranslator : public Visitor {

public:
    /// Constructor
    QPTranslator() {}
    
    /// @brief Visit a AggrExpr node
    virtual void visit(AggrExpr *v);

    /// @brief Visit a Attribute node
    virtual void visit(Attribute *v);

    /// @brief Visit a AttrList node
    virtual void visit(AttrList *v);

    /// @brief Visit a Comparison node
    virtual void visit(Comparison *v);

    /// @brief Visit a MathExpr node
    virtual void visit(MathExpr *v);

    /// @brief Visit a Predicate node
    virtual void visit(Predicate *v);

    /// @brief Visit a Program node
    virtual void visit(Program *v);

    /// @brief Visit a ProjectOp node
    virtual void visit(ProjectOp *v);

    /// @brief Visit a Relation node
    virtual void visit(Relation *v);

    /// @brief Visit a RelExpr node
    virtual void visit(RelExpr *v);
    
    /// @brief Visit a RelExprList node
    virtual void visit(RelExprList *v);

    /// @brief Visit a SelectOp node
    virtual void visit(SelectOp *v);

private:
    std::deque<int> qTracker_;
};

} // RA_Namespace

#endif // QUERYPLAN_PARSE_RA_VISITOR_QPTRANSLATOR_H


