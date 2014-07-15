#ifndef SELECT_OP_NODE_H
#define SELECT_OP_NODE_H

#include "RelAlgNode.h"
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class SelectOp : public UnaryOp {
    
public:

	Predicate* pred;

	explicit SelectOp(RelExpr *n1, Predicate* n2) : pred(n2) { relex = n1; }

	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
    
	};
}

#endif // SELECT_OP_NODE_H