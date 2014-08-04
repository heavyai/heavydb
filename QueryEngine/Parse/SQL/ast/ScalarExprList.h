#ifndef SQL_SCALAREXPLIST_H
#define SQL_SCALAREXPLIST_H

#include <cassert>
#include "ASTNode.h"

namespace SQL_Namespace {

class ScalarExprList : public ASTNode {

public:

	ScalarExprList *n1 = NULL;
	ScalarExpr *n2 = NULL;

	explicit ScalarExprList(ScalarExprList *n1, ScalarExpr *n2) {
		assert(n1 && n2);
		this->n1 = n1;
		this->n2 = n2;
	}

	explicit ScalarExprList(ScalarExpr *n2) {
		assert(n2);
		this->n2 = n2;
	}
	
	virtual void accept(Visitor &v) {
		v.visit(this);
	}

};

} // SQL_Namespace

#endif // SQL_SCALAREXPLIST_H
