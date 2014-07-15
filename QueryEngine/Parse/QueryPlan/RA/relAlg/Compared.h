#ifndef COMPARED_NODE_H
#define COMPARED_NODE_H


#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class Compared : public RelAlgNode {
    
public:

	Attribute *a;
	Data *d;

	explicit Compared(Attribute *n1) : a(n1), d(NULL) {}
	explicit Compared(Data *n1) : a(NULL), d(n1) {}
	
	/**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // COMPARED_NODE_H