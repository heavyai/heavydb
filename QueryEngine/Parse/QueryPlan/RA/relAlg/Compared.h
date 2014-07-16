#ifndef COMPARED_NODE_H
#define COMPARED_NODE_H


#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class Compared : public RelAlgNode {
    
public:

//	Attribute *a;
	MathExpr *me;

//	explicit Compared(Attribute *n1) : a(n1), me(NULL) {}
	explicit Compared(MathExpr *n1) : me(n1) {}
	
	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // COMPARED_NODE_H