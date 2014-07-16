#ifndef OUTERJOIN_OP_NODE_H
#define OUTERJOIN_OP_NODE_H

#include "RelAlgNode.h"
#include "BinaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class OuterjoinOp : public BinaryOp {
    
public:
	RA_Predicate* pred;

	OuterjoinOp(RelExpr *n1, RelExpr *n2, RA_Predicate* n3) : pred(n3) { relex1 = n1; relex2 = n2; }
	
/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // OUTERJOIN_OP_NODE_H