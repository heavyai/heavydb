#ifndef SORT_OP_NODE_H
#define SORT_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class SortOp : public UnaryOp {
    
public:

	AttrList* atLi;

	explicit SortOp(RelExpr *n1, AttrList* n2) : relex(n1), atLi(n2) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // SORT_OP_NODE_H