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
};

#endif // COMPARISON_NODE_H