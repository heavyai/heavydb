/**
 * @file	ExtendOp.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_EXTENDOP_NODE_H
#define RA_EXTENDOP_NODE_H

#include <cassert>
#include "../ast/UnaryOp.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class ExtendOp : public UnaryOp {
    
public:
	RelExpr *n1 = NULL;
	Expr *n2 = NULL;
	std::string name = "";

	/// Constructor
	ExtendOp(RelExpr *n1, Expr *n2, std::string name) {
		assert(n1 && n2 && name != "");
		this->n1 = n1;
		this->n2 = n2;
		this->name = name;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_EXTENDOP_NODE_H
