/**
 * @file	AggrList.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_AGGRLIST_NODE_H
#define RA_AGGRLIST_NODE_H

#include <cassert>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class AggrList : public RelAlgNode {
    
public:
	AggrList *n1 = NULL;
	AggrExpr *n2 = NULL;

	explicit AggrList(AggrExpr *n2) {
		assert(n2);
		this->n2 = n2;
	}

	AggrList(AggrList *n1, AggrExpr *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}
};

} // RA_Namespace

#endif // RA_AGGRLIST_NODE_H
