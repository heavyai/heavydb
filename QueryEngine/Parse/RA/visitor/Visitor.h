#ifndef RA_AST_VISITOR_H
#define RA_AST_VISITOR_H

namespace RA_Namespace {

	class Visitor {

	public:
 	    virtual void visit(class AggrExpr *v) = 0;
 	    virtual void visit(class AggrList *v) = 0;
 	    virtual void visit(class AntijoinOp *v) = 0;
 	    virtual void visit(class Attribute *v) = 0;
 	    virtual void visit(class AttrList *v) = 0;
 	    virtual void visit(class Comparison *v) = 0;
 	    virtual void visit(class DiffOp *v) = 0;
 	    virtual void visit(class Expr *v) = 0;
 	    virtual void visit(class ExtendOp *v) = 0;
 	    virtual void visit(class GroupbyOp *v) = 0;
 	    virtual void visit(class JoinOp *v) = 0;
 	    virtual void visit(class MathExpr *v) = 0;
 	    virtual void visit(class OuterjoinOp *v) = 0;
 	    virtual void visit(class Predicate *v) = 0;
 	    virtual void visit(class ProductOp *v) = 0;
 	    virtual void visit(class Program *v) = 0;
 	    virtual void visit(class ProjectOp *v) = 0;
 	    virtual void visit(class Relation *v) = 0;
 	    virtual void visit(class RelExpr *v) = 0;
 	    virtual void visit(class RelExprList *v) = 0;
 	    virtual void visit(class RenameOp *v) = 0;
 	    virtual void visit(class SelectOp *v) = 0;
 	    virtual void visit(class SemijoinOp *v) = 0;
 	    virtual void visit(class SortOp *v) = 0;
 	    virtual void visit(class UnionOp *v) = 0;
};

} // RA_Namespace

#endif // RA_AST_VISITOR_H
