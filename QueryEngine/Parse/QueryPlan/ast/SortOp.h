#ifndef SORT_OP_NODE_H
#define SORT_OP_NODE_H

#include "RelAlgNode.h"
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

class SortOp : public UnaryOp {
    
public:

	AttrList* atLi;

	explicit SortOp(RelExpr *n1, AttrList* n2) : atLi(n2) { relex = n1; }

/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // SORT_OP_NODE_H