/**
 * @file	DiffOp.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_DIFFOP_NODE_H
#define RA_DIFFOP_NODE_H

#include <cassert>
#include "BinaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class DiffOp : public BinaryOp {
    
public:
	RelExpr *n1 = NULL;
	RelExpr *n2 = NULL;
	DiffOp *n3 = NULL;

	/// Constructor
	DiffOp(RelExpr *n1, RelExpr *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	/// Constructor: this is specifically for the INTERSECTION operation,
	/// which is equivalent to DiffOp(RelExpr, DiffOp(RelExpr, RelExpr))
	DiffOp(RelExpr *n1, DiffOp *n3) {
		assert(n1 && n3);
		this->n1 = n1;
		this->n3 = n3;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_DIFFOP_NODE_H
