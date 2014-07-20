/**
 * @file	OuterjoinOp.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_OUTERJOINOP_NODE_H
#define RA_OUTERJOINOP_NODE_H

#include <cassert>
#include "BinaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class OuterjoinOp : public BinaryOp {
    
public:
	RelExpr *n1 = NULL;
	RelExpr *n2 = NULL;
	Predicate *n3 = NULL;

	/// Constructor
	OuterjoinOp(RelExpr *n1, RelExpr *n2, Predicate *n3) {
		assert(n1 && n2 && n3);
		this->n1 = n1;
		this->n2 = n2;
		this->n3 = n3;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_OUTERJOINOP_NODE_H
