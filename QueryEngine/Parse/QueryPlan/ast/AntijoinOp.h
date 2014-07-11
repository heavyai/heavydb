#ifndef ANTIJOIN_OP_NODE_H
#define ANTIJOIN_OP_NODE_H

#include "RelAlgNode.h"
#include "BinaryOp.h"
#include "../visitor/Visitor.h"

class AntijoinOp : public BinaryOp {
    
public:
	Predicate* pred;

	explicit AntijoinOp(RelExpr *n1, RelExpr *n2, Predicate* n3) : pred(n3) { relex1 = n1; relex2 = n2; }
	
/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // ANTIJOIN_OP_NODE_H