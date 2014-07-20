/**
 * @file	AttrList.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_ATTRLIST_NODE_H
#define RA_ATTRLIST_NODE_H

#include <cassert>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class AttrList : public RelAlgNode {
    
public:
	AttrList *n1 = NULL;
	Attribute *n2 = NULL;

	explicit AttrList(Attribute *n2) {
		assert(n2);
		this->n2 = n2;
	}

	AttrList(AttrList *n1, Attribute *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

} // RA_Namespace

#endif // RA_ATTRLIST_NODE_H
