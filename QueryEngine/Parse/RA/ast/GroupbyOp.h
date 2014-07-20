/**
 * @file	GroupbyOp.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_GROUPBYOP_NODE_H
#define RA_GROUPBYOP_NODE_H

#include <cassert>
#include "UnaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class GroupbyOp : public UnaryOp {
    
public:
	RelExpr *n1 = NULL;
	AttrList *n2 = NULL;
	AggrList *n3 = NULL;

	/// Constructor
	GroupbyOp(RelExpr *n1, AttrList *n2, AggrList *n3) {
		assert(n1 && n2 && n3);
		this->n1 = n1;
		this->n2 = n2;
		this->n3 = n3;
	}

	/// Constructor
	GroupbyOp(RelExpr *n1, AggrList *n3) {
		assert(n1 && n3);
		this->n1 = n1;
		this->n3 = n3;
	}

	/// Constructor
	GroupbyOp(RelExpr *n1, AttrList *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_GROUPBYOP_NODE_H
