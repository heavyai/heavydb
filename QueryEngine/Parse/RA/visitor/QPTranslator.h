/**
 * @file    QPTranslator.h
 * @author  Steven Stewart <steve@map-d.com>
 */
#ifndef QUERYPLAN_PARSE_RA_VISITOR_QPTRANSLATOR_H
#define QUERYPLAN_PARSE_RA_VISITOR_QPTRANSLATOR_H

#include <iostream>
#include <deque>
#include <stdio.h>
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
    
    /// @brief Visit an AggrExpr node
    virtual void visit(AggrExpr *v);

    /// @brief Visit an AggrList node
    virtual void visit(AggrList *v);

    /// @brief Visit an AntijoinOp node
    virtual void visit(AntijoinOp *v);

    /// @brief Visit an Attribute node
    virtual void visit(Attribute *v);

    /// @brief Visit an AttrList node
    virtual void visit(AttrList *v);

    /// @brief Visit a Comparison node
    virtual void visit(Comparison *v);

    /// @brief Visit a DiffOp node
    virtual void visit(DiffOp *v);

    /// @brief Visit a Expr node
    virtual void visit(Expr *v);

    /// @brief Visit a ExtendOp node
    virtual void visit(ExtendOp *v);

    /// @brief Visit a GroupbyOp node
    virtual void visit(GroupbyOp *v);

    /// @brief Visit a JoinOp node
    virtual void visit(JoinOp *v);    

    /// @brief Visit a MathExpr node
    virtual void visit(MathExpr *v);

    /// @brief Visit an Outerjoin node
    virtual void visit(OuterjoinOp *v);

    /// @brief Visit a Predicate node
    virtual void visit(Predicate *v);

    /// @brief Visit a Program node
    virtual void visit(Program *v);

    /// @brief Visit a Product node
    virtual void visit(ProductOp *v);

    /// @brief Visit a ProjectOp node
    virtual void visit(ProjectOp *v);

    /// @brief Visit a Relation node
    virtual void visit(Relation *v);

    /// @brief Visit a RelExpr node
    virtual void visit(RelExpr *v);
    
    /// @brief Visit a RelExprList node
    virtual void visit(RelExprList *v);

    /// @brief Visit a Rename node
    virtual void visit(RenameOp *v);

    /// @brief Visit a SelectOp node
    virtual void visit(SelectOp *v);

    /// @brief Visit a Semijoin node
    virtual void visit(SemijoinOp *v);

    /// @brief Visit a SortOp node
    virtual void visit(SortOp *v);

    /// @brief Visit a UnionOp node
    virtual void visit(UnionOp *v);

private:
    std::deque<int> qTracker_;
    int queryNum_ = 0;
    /*
    void stackTrace() {
        printf("------Stack Trace ------\n");
        for (int i = 0; i < qTracker_.size(); i++) {
            printf("%d | %d\n", i, qTracker_[i]);
        }
        printf("-----------------------\n");
    } */
};

} // RA_Namespace

#endif // QUERYPLAN_PARSE_RA_VISITOR_QPTRANSLATOR_H


