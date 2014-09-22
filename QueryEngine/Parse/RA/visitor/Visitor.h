#ifndef RA_AST_VISITOR_H
#define RA_AST_VISITOR_H

namespace RA_Namespace {

// forward declarations
class AggrExpr;
class AggrList;
class AntijoinOp;
class Attribute;
class AttrList;
class BinaryOp;
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
class ScanOp;
class SelectOp;
class SemijoinOp;
class SortOp;
class UnaryOp;
class UnionOp;

class Visitor {

	public:
 	    virtual void visit(AggrExpr *v) {}
 	    virtual void visit(AggrList *v) {}
 	    virtual void visit(AntijoinOp *v) {}
 	    virtual void visit(Attribute *v) {}
 	    virtual void visit(AttrList *v) {}
 	    virtual void visit(Comparison *v) {}
 	    virtual void visit(DiffOp *v) {}
 	    virtual void visit(Expr *v) {}
 	    virtual void visit(ExtendOp *v) {}
 	    virtual void visit(GroupbyOp *v) {}
 	    virtual void visit(JoinOp *v) {}
 	    virtual void visit(MathExpr *v) {}
 	    virtual void visit(OuterjoinOp *v) {}
 	    virtual void visit(Predicate *v) {}
 	    virtual void visit(ProductOp *v) {}
 	    virtual void visit(Program *v) {}
 	    virtual void visit(ProjectOp *v) {}
 	    virtual void visit(Relation *v) {}
 	    virtual void visit(RelExpr *v) {}
 	    virtual void visit(RelExprList *v) {}
 	    virtual void visit(RenameOp *v) {}
        virtual void visit(ScanOp *v) {}
 	    virtual void visit(SelectOp *v) {}
 	    virtual void visit(SemijoinOp *v) {}
 	    virtual void visit(SortOp *v) {}
 	    virtual void visit(UnionOp *v) {}
};

} // RA_Namespace

#endif // RA_AST_VISITOR_H
