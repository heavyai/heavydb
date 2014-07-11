#ifndef SELECT_OP_NODE_H
#define SELECT_OP_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class SelectOp : public UnaryOp {
    
public:

	Predicate* pred;

	explicit SelectOp(RelExpr *n1, Predicate* n2) : relex(n1), pred(n2) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // SELECT_OP_NODE_H