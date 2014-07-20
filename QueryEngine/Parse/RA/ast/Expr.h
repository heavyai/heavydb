/**
 * @file	Expr.h
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Gil Walzer <gil@map-d.com>
 */
#ifndef RA_EXPR_NODE_H
#define RA_EXPR_NODE_H

#include <cassert>
#include "RelAlgNode.h"
#include "../visitor/Visitor.h"

namespace RA_Namespace {

class Expr : public RelAlgNode {
    
public:
	MathExpr *n1 = NULL;
	Predicate *n2 = NULL;
	std::string str = "";

	/// Constructor
	explicit Expr(MathExpr *n1) {
		assert(n1);
		this->n1 = n1;
	}

	explicit Expr(Predicate *n2) {
		assert(n2);
		this->n2 = n2;
	}

	explicit Expr(const std::string &str) {
		assert(str != "");
		this->str = str;
	}

	virtual void accept(class Visitor &v) {
		v.visit(this);
	}

};

} // RA_Namespace

#endif // RA_EXPR_NODE_H
