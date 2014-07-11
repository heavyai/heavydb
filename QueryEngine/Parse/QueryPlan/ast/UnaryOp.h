#ifndef UNARY_OP_NODE_H
#define UNARY_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class UnaryOp : public RelAlgNode {
    
public:

	RelExpr* relex;

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
};

#endif // UNARY_OP_NODE_H