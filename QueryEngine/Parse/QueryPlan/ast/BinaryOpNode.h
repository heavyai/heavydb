#ifndef BINARY_OP_NODE_H
#define BINARY_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class BinaryOp : public RelAlgNode {
    
public:
	RelExpr* relex1;
	RelExpr* relex2;

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // BINARY_OP_NODE_H