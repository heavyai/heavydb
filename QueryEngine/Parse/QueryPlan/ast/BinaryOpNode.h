#ifndef BINARY_OP_NODE_H
#define BINARY_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class BinaryOp : public RelAlgNode {
    
public:

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // BINARY_OP_NODE_H