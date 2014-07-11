#ifndef ATTR_LIST_NODE_H
#define ATTR_LIST_NODE_H

#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

class AttrList : public RelAlgNode {
    
public:

	Attribute* at;
	AttrList* atLi;

	explicit AttrList(AttrList *n1, Attribute *n2) : atLi(n1), at(n2) {}
	AttrList(Attribute *n) : at(n), atLi(NULL) {}

    /**< Accepts the given void visitor by calling v.visit(this) */
    virtual void accept(class Visitor &v) = 0;
};

#endif // ATTR_LIST_NODE_H