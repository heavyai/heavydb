#ifndef SORT_OP_NODE_H
#define SORT_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class SortOp : public UnaryOp {
    
public:

	AttrList* atLi;

	explicit SortOp(AttrList* n) : atLi(n) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // SORT_OP_NODE_H