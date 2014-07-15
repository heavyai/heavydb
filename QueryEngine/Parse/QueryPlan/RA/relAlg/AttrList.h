#ifndef ATTR_LIST_NODE_H
#define ATTR_LIST_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {
class AttrList : public RelAlgNode {
    
public:

	Attribute* at;
	AttrList* atLi;

	AttrList(AttrList *n1, Attribute *n2) : atLi(n1), at(n2) {}
	AttrList(Attribute *n) : at(n), atLi(NULL) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    void accept(Visitor &v) {
        v.visit(this);
    }
	};
}

#endif // ATTR_LIST_NODE_H