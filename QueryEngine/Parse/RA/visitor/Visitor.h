#ifndef RA_AST_VISITOR_H
#define RA_AST_VISITOR_H

namespace RA_Namespace {

class UnaryOp;
class BinaryOp;

class AggrExpr;
class AggrList;
class AntijoinOp;
class Attribute;
class AttrList;
class Comparison;
class DiffOp;
class Expr;
class ExtendOp;
class GroupbyOp;
class JoinOp;
class MathExpr;
class OuterjoinOp;
class Predicate;
class ProductOp;
class Program;
class ProjectOp;
class Relation;
class RelExpr;
class RelExprList;
class RenameOp;
class SelectOp;
class SemijoinOp;
class SortOp;
class UnionOp;

class Visitor {

	public:
 	    virtual void visit(AggrExpr *v) = 0;
 	    virtual void visit(AggrList *v) = 0;
 	    virtual void visit(AntijoinOp *v) = 0;
 	    virtual void visit(Attribute *v) = 0;
 	    virtual void visit(AttrList *v) = 0;
 	    virtual void visit(Comparison *v) = 0;
 	    virtual void visit(DiffOp *v) = 0;
 	    virtual void visit(Expr *v) = 0;
 	    virtual void visit(ExtendOp *v) = 0;
 	    virtual void visit(GroupbyOp *v) = 0;
 	    virtual void visit(JoinOp *v) = 0;
 	    virtual void visit(MathExpr *v) = 0;
 	    virtual void visit(OuterjoinOp *v) = 0;
 	    virtual void visit(Predicate *v) = 0;
 	    virtual void visit(ProductOp *v) = 0;
 	    virtual void visit(Program *v) = 0;
 	    virtual void visit(ProjectOp *v) = 0;
 	    virtual void visit(Relation *v) = 0;
 	    virtual void visit(RelExpr *v) = 0;
 	    virtual void visit(RelExprList *v) = 0;
 	    virtual void visit(RenameOp *v) = 0;
 	    virtual void visit(SelectOp *v) = 0;
 	    virtual void visit(SemijoinOp *v) = 0;
 	    virtual void visit(SortOp *v) = 0;
 	    virtual void visit(UnionOp *v) = 0;
};

} // RA_Namespace

#endif // RA_AST_VISITOR_H
