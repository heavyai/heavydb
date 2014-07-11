#ifndef AST_VISITOR_H
#define AST_VISITOR_H

class Visitor {

public:
    /*
    // apparently this isn't supported, but it would be nice if it were:
    template <typename T>
    // virtual void visit(T &);*/

    virtual void visit(class RelAlgNode *v) = 0;
    virtual void visit(class Program *v) = 0;
    virtual void visit(class RelExprList *v) = 0;
    virtual void visit(class RelExpr *v) = 0;
    virtual void visit(class BinaryOp *v) = 0;
    virtual void visit(class UnaryOp *v) = 0;
    virtual void visit(class SelectOp *v) = 0;
    virtual void visit(class ProjectOp *v) = 0;
    virtual void visit(class ExtendOp *v) = 0;
    virtual void visit(class RenameOp *v) = 0;
    virtual void visit(class SortOp *v) = 0;
    virtual void visit(class GroupByOp *v) = 0;

    virtual void visit(class Join *v) = 0;
    virtual void visit(class Product *v) = 0;
    virtual void visit(class Semijoin *v) = 0;
    virtual void visit(class Outerjoin *v) = 0;
    virtual void visit(class Antijoin *v) = 0;
    virtual void visit(class Union *v) = 0;

    virtual void visit(class MathExpr *v) = 0;
    virtual void visit(class Predicate *v) = 0;
    virtual void visit(class AggrList *v) = 0;
    virtual void visit(class AttrList *v) = 0;
    virtual void visit(class AggrExpr *v) = 0;
    virtual void visit(class Attribute *v) = 0;
    virtual void visit(class Data *v) = 0;
    virtual void visit(class Comparison *v) = 0;
    virtual void visit(class Compared *v) = 0;
    virtual void visit(class CompOp *v) = 0;
    virtual void visit(class Relation *v) = 0;
};

#endif // AST_VISITOR_H
