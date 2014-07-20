/**
 * @file	SelectOp.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_SELECTOP_NODE_H
#define RA_SELECTOP_NODE_H

#include <cassert>
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class SelectOp : public UnaryOp {
    
public:
	RelExpr *n1;
	Predicate *n2;

	/// Constructor
	SelectOp(RelExpr *n1, Predicate *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_SELECTOP_NODE_H
