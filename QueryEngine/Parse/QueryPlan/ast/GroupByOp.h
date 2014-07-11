#ifndef GROUP_BY_OP_NODE_H
#define GROUP_BY_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class GroupByOp : public UnaryOp {
    
public:

	AttrList *atLi;
	AggrList *agLi;

	GroupByOp(RelExpr *n1, AttrList* n2, AggrList* n3) : atLi(n2), agLi(n3) { relex = n1; }
	GroupByOp(RelExpr *n1, AggrList* n2) : atLi(NULL), agLi(n2) { relex = n1; }

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // GROUP_BY_OP_NODE_H