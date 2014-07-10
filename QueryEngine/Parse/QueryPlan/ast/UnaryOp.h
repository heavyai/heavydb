#ifndef UNARY_OP_NODE_H
#define UNARY_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class UnaryOp : public RelAlgNode {
    
public:

	RelExpr* relex;

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // UNARY_OP_NODE_H