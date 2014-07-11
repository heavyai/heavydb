#ifndef COMPARISON_NODE_H
#define COMPARISON_NODE_H


#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class Comparison : public RelAlgNode {
    
public:

	Compared *c1;
	CompOp *co;
	Compared *c2;

	explicit Comparison(Compared *n1, CompOp *n2, Compared* n3) : c1(n1), co(n2), c2(n3) {}

		/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
};

#endif // COMPARISON_NODE_H