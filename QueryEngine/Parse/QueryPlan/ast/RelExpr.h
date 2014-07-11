#ifndef REL_EXPR_NODE_H
#define REL_EXPR_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class RelExpr : public RelAlgNode {
    
public:

	RelExpr* relex;
	UnaryOp* uno;
	BinaryOp* dos;
	Relation* rel;

	RelExpr(RelExpr *n) : relex(n), uno(NULL), dos(NULL), rel(NULL) {}
	RelExpr(UnaryOp *n) : relex(NULL), uno(n), dos(NULL), rel(NULL) {}
	RelExpr(BinaryOp *n) : relex(NULL), uno(NULL), dos(n), rel(NULL) {}
	RelExpr(Relation *n) : relex(NULL), uno(NULL), dos(NULL), rel(n) {}
	
/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // REL_EXPR_NODE_H